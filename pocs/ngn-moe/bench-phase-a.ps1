# nGn MoE offload PoC, Phase A: what upstream already offers, on gpt-oss-20b at 32K-class depth.
#   powershell -ExecutionPolicy Bypass -File bench-phase-a.ps1 [-Model <gguf>] [-Quick]
# Each config is a separate llama-bench process so an out-of-memory config does not end the run.
# Writes logs\phase-a-<stamp>\: per-config stdout (JSONL), stderr (cache hit rates), peak VRAM and
# process memory, and summary.json. Reads nothing outside this folder, the build and the model.
param(
  [string]$Model = "",
  [string]$ModelDir = "D:\MLModels\lmstudio-community\gpt-oss-20b-GGUF",
  [string]$Bin = "C:\ngn\poc-build\bin",
  [int]$Depth = 24000,     # Bastion compaction keeps input within 24K of the 32K window
  [switch]$Quick,          # depth 4096, one repetition: checks the configs load and run
  [int]$Threads = 0,       # 0 = llama-bench default
  [string]$Only = "",      # comma-separated config names to run (default all)
  [int]$CoolSeconds = 20   # idle between configs so thermal throttling carries over less
)
$ErrorActionPreference = "Stop"
# Consoles started from Explorer keep the PATH from before CUDA was installed: put its runtime DLLs first.
$cudaBin = Join-Path $(if ($env:CUDA_PATH_V12_8) { $env:CUDA_PATH_V12_8 } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" }) "bin"
$env:PATH = "$cudaBin;$env:PATH"
if (-not $Model) { $Model = (Get-ChildItem $ModelDir -Filter *.gguf -Recurse | Sort-Object Length -Descending | Select-Object -First 1).FullName }
if (-not (Test-Path $Model)) { throw "model not found under $ModelDir" }
$bench = Join-Path $Bin "llama-bench.exe"
if (-not (Test-Path $bench)) { throw "build first: $bench missing" }
if ($Quick) { $Depth = 4096 }
$reps = if ($Quick) { 1 } else { 2 }
$out = Join-Path $PSScriptRoot ("logs\phase-a-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force $out | Out-Null

# Host facts for the record.
$cs = Get-CimInstance Win32_ComputerSystem; $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$gpu = (& nvidia-smi --query-gpu=name,memory.total,memory.used,pcie.link.gen.max,pcie.link.width.max,driver_version --format=csv,noheader) -join ";"
@{ model = $Model; modelBytes = (Get-Item $Model).Length; ramBytes = $cs.TotalPhysicalMemory; cpu = $cpu.Name;
   cores = $cpu.NumberOfCores; threads = $cpu.NumberOfLogicalProcessors; gpu = $gpu; depth = $Depth; reps = $reps } |
  ConvertTo-Json | Set-Content (Join-Path $out "host.json")

# name, --n-cpu-moe, cache MiB, ubatch. 24 = every gpt-oss-20b expert layer in host memory.
$configs = @(
  @("static-ncmoe24", 24, 0,    512),
  @("static-ncmoe18", 18, 0,    512),
  @("static-ncmoe14", 14, 0,    512),
  @("cache-1024",     24, 1024, 512),
  @("cache-2048",     24, 2048, 512),
  @("cache-3072",     24, 3072, 512),
  @("static-ncmoe24-ub2048", 24, 0, 2048),
  # Expert weights on the CPU but their small biases on the GPU: --cpu-moe's pattern also moves gpt-oss's
  # expert biases to the host, so every decode step bounces add_id between devices.
  @("cache-2048-biasgpu", -1, 2048, 512),
  @("static-all-biasgpu", -1, 0,    512)
)

$summary = @()
foreach ($c in $configs) {
  $name, $ncmoe, $mib, $ub = $c
  if ($Only -and -not (($Only -split ",") -contains $name)) { continue }
  Start-Sleep -Seconds $CoolSeconds
  Write-Host "== $name" -ForegroundColor Cyan
  if ($mib -gt 0) { $env:LLAMA_ARG_MOE_CACHE_MIB = "$mib" } else { Remove-Item Env:LLAMA_ARG_MOE_CACHE_MIB -ErrorAction SilentlyContinue }
  $place = if ($ncmoe -ge 0) { @("-ncmoe", "$ncmoe") } else { @("-ot", "\.ffn_(up|down|gate)_exps\.weight=CPU") }
  $cliArgs = @("-m", $Model, "-ngl", "99") + $place + @("-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0",
            "-b", "2048", "-ub", "$ub", "-p", "512", "-n", "128", "-d", "0,$Depth", "-r", "$reps", "-o", "jsonl")
  if ($Threads -gt 0) { $cliArgs += @("-t", "$Threads") }
  $stdout = Join-Path $out "$name.jsonl"; $stderr = Join-Path $out "$name.stderr.txt"
  $p = Start-Process -FilePath $bench -ArgumentList $cliArgs -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  $null = $p.Handle  # keeps the exit code readable after the process ends
  $peakVram = 0; $peakWs = 0; $peakPriv = 0
  while (-not $p.HasExited) {
    try { $p.Refresh(); $peakWs = [math]::Max($peakWs, $p.PeakWorkingSet64); $peakPriv = [math]::Max($peakPriv, $p.PrivateMemorySize64) } catch {}
    $used = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
    $peakVram = [math]::Max($peakVram, $used)
    Start-Sleep -Milliseconds 500
  }
  $p.WaitForExit()
  $rows = @(Get-Content $stdout -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\{' } | ForEach-Object { $_ | ConvertFrom-Json })
  $hits = @(Select-String -Path $stderr -Pattern "llama_moe_cache: " -ErrorAction SilentlyContinue | ForEach-Object { $_.Line })
  $err  = @(Select-String -Path $stderr -Pattern "out of memory|failed|error" -ErrorAction SilentlyContinue | Select-Object -First 3 | ForEach-Object { $_.Line })
  $entry = [ordered]@{ name = $name; ncmoe = $ncmoe; cacheMiB = $mib; ubatch = $ub; exitCode = $p.ExitCode;
    peakVramMiB = $peakVram; peakWorkingSetMiB = [int]($peakWs/1MB); peakPrivateMiB = [int]($peakPriv/1MB);
    results = @($rows | ForEach-Object { [ordered]@{ test = $(if ($_.n_prompt -gt 0) { "pp$($_.n_prompt)" } else { "tg$($_.n_gen)" }); depth = $_.n_depth; tps = [math]::Round($_.avg_ts, 2); sd = [math]::Round($_.stddev_ts, 2) } });
    cache = $hits; errors = $err }
  $summary += $entry
  $entry.results | ForEach-Object { Write-Host ("   {0,-6} d{1,-6} {2,8} tok/s" -f $_.test, $_.depth, $_.tps) }
  if ($p.ExitCode -eq -1073741515) { Write-Host "   a DLL is missing (CUDA runtime or cuBLAS not found)" -ForegroundColor Red }
  Write-Host ("   exit {0}, peak VRAM {1} MiB, peak WS {2} MiB" -f $p.ExitCode, $peakVram, $entry.peakWorkingSetMiB)
}
Remove-Item Env:LLAMA_ARG_MOE_CACHE_MIB -ErrorAction SilentlyContinue
$summary | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $out "summary.json")
Write-Host "PHASE A DONE: $out" -ForegroundColor Green
