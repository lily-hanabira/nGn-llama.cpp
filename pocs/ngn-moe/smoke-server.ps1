# nGn MoE offload PoC: does gpt-oss-20b work end to end under nGn-style launch on an 8GB GPU?
#   powershell -ExecutionPolicy Bypass -File smoke-server.ps1 [-CacheMiB 2048] [-NCpuMoe 24]
# 1) short chat answer (Harmony via --jinja, low reasoning effort), 2) a ~23.5K-token prompt
# plus 8,192 generated tokens: the full 32K window with Bastion's 8K output floor.
param(
  [string]$Model = "",
  [string]$ModelDir = "D:\MLModels\lmstudio-community\gpt-oss-20b-GGUF",
  [string]$Bin = "C:\ngn\poc-build\bin",
  [int]$NCpuMoe = 24, [int]$CacheMiB = 2048, [int]$Port = 18089, [int]$Threads = 8,
  [switch]$BiasOnGpu,      # experts on the host by -ot weight pattern, their biases stay on the GPU
  [int]$PromptTokens = 23500, [int]$OutputTokens = 8192
)
$ErrorActionPreference = "Stop"
# Consoles started from Explorer keep the PATH from before CUDA was installed: put its runtime DLLs first.
$cudaBin = Join-Path $(if ($env:CUDA_PATH_V12_8) { $env:CUDA_PATH_V12_8 } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" }) "bin"
$env:PATH = "$cudaBin;$env:PATH"
if (-not $Model) { $Model = (Get-ChildItem $ModelDir -Filter *.gguf -Recurse | Sort-Object Length -Descending | Select-Object -First 1).FullName }
$out = Join-Path $PSScriptRoot ("logs\smoke-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-ncmoe$NCpuMoe-cache$CacheMiB")
New-Item -ItemType Directory -Force $out | Out-Null
$base = "http://127.0.0.1:$Port"

# Mirrors ngn/agent/src/serving/LlamaServer.cs, plus expert placement.
$srvArgs = @("--model", $Model, "--alias", "gpt-oss-20b", "--host", "127.0.0.1", "--port", "$Port",
  "--ctx-size", "32768", "--parallel", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--flash-attn", "on",
  "--jinja", "--no-context-shift", "--no-webui", "--metrics", "--n-gpu-layers", "999", "--threads", "$Threads")
if ($BiasOnGpu) { $srvArgs += @("-ot", "\.ffn_(up|down|gate)_exps\.weight=CPU") } else { $srvArgs += @("--n-cpu-moe", "$NCpuMoe") }
if ($CacheMiB -gt 0) { $srvArgs += @("--moe-cache-mib", "$CacheMiB") }
$srv = Start-Process -FilePath (Join-Path $Bin "llama-server.exe") -ArgumentList $srvArgs -NoNewWindow -PassThru `
  -RedirectStandardOutput (Join-Path $out "server.stdout.txt") -RedirectStandardError (Join-Path $out "server.stderr.txt")
$null = $srv.Handle
$result = [ordered]@{ ncmoe = $NCpuMoe; cacheMiB = $CacheMiB; biasOnGpu = [bool]$BiasOnGpu; threads = $Threads;
  cpuMiss = $env:LLAMA_MOE_CACHE_CPU_MISS; promote = $env:LLAMA_MOE_CACHE_PROMOTE; promoteEvery = $env:LLAMA_MOE_CACHE_PROMOTE_EVERY }
try {
  $t0 = Get-Date
  while ($true) {
    if ($srv.HasExited) { throw "server exited with $($srv.ExitCode), see server.stderr.txt" }
    try { if ((Invoke-RestMethod "$base/health" -TimeoutSec 2).status -eq "ok") { break } } catch {}
    Start-Sleep -Seconds 1
  }
  $result.loadSeconds = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
  $result.vramAfterLoadMiB = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
  Write-Host "loaded in $($result.loadSeconds)s, VRAM $($result.vramAfterLoadMiB) MiB"

  # 1) Short chat.
  $body = @{ model = "gpt-oss-20b"; max_tokens = 512; temperature = 0; reasoning_effort = "low";
    messages = @(@{ role = "user"; content = "In one sentence: why do mixture-of-experts models need less compute per token than dense models of the same size?" }) } | ConvertTo-Json -Depth 5
  $r = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 600
  $result.chat = [ordered]@{ content = $r.choices[0].message.content; reasoningChars = "$($r.choices[0].message.reasoning_content)".Length;
    finish = $r.choices[0].finish_reason; promptTps = $r.timings.prompt_per_second; decodeTps = $r.timings.predicted_per_second }
  Write-Host "chat: $($result.chat.content)"

  # 2) Long window: repeat the fork's docs until the prompt is PromptTokens long, then generate OutputTokens.
  $docs = (Get-ChildItem (Join-Path $PSScriptRoot "..\..\docs") -Filter *.md -Recurse | Get-Content -Raw) -join "`n"
  $tok = (Invoke-RestMethod "$base/tokenize" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes((@{ content = $docs } | ConvertTo-Json)))).tokens
  while ($tok.Count -lt $PromptTokens) { $tok = $tok + $tok }
  $tok = $tok[0..($PromptTokens - 1)]
  $body = @{ prompt = $tok; n_predict = $OutputTokens; ignore_eos = $true; temperature = 0.7; cache_prompt = $false } | ConvertTo-Json -Depth 3 -Compress
  $t1 = Get-Date
  $r = Invoke-RestMethod "$base/completion" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 7200
  $result.window = [ordered]@{ promptTokens = $r.timings.prompt_n; generated = $r.timings.predicted_n;
    promptTps = [math]::Round($r.timings.prompt_per_second, 1); decodeTps = [math]::Round($r.timings.predicted_per_second, 2);
    wallSeconds = [math]::Round(((Get-Date) - $t1).TotalSeconds, 1); tail = $r.content.Substring([math]::Max(0, $r.content.Length - 300)) }
  $srv.Refresh()
  $result.vramAfterWindowMiB = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
  $result.serverPeakWorkingSetMiB = [int]($srv.PeakWorkingSet64/1MB); $result.serverPrivateMiB = [int]($srv.PrivateMemorySize64/1MB)
  Write-Host ("window: {0} prompt @ {1} tok/s, {2} generated @ {3} tok/s" -f $result.window.promptTokens, $result.window.promptTps, $result.window.generated, $result.window.decodeTps)
  $result.ok = $true
} catch { $result.ok = $false; $result.error = "$_"; Write-Host "FAILED: $_" -ForegroundColor Red }
finally {
  if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
  Start-Sleep -Seconds 1
  $result.serverLog = @(Get-Content (Join-Path $out "server.stderr.txt") -ErrorAction SilentlyContinue | Select-String -Pattern "moe_cache|MoE cache|init_cpu|CUDA0 model buffer|CPU_Mapped|KV buffer|compute buffer" | Select-Object -First 16 | ForEach-Object { $_.Line })
  $result | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $out "result.json")
  Write-Host "SMOKE DONE: $out" -ForegroundColor Green
}
