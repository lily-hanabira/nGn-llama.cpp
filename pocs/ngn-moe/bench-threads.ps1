# nGn MoE offload PoC: CPU thread count for expert decode (i9 HX parts mix 8 P-cores with E-cores).
#   powershell -ExecutionPolicy Bypass -File bench-threads.ps1
param(
  [string]$ModelDir = "D:\MLModels\lmstudio-community\gpt-oss-20b-GGUF",
  [string]$Bin = "C:\ngn\poc-build\bin",
  [string]$Threads = "4,8,12,16,24"
)
$ErrorActionPreference = "Stop"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;$env:PATH"
$Model = (Get-ChildItem $ModelDir -Filter *.gguf -Recurse | Sort-Object Length -Descending | Select-Object -First 1).FullName
$out = Join-Path $PSScriptRoot ("logs\threads-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force $out | Out-Null
$power = (powercfg /getactivescheme) -join " "
$ac = (Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1).PowerOnline
@{ powerScheme = $power; onMains = $ac } | ConvertTo-Json | Set-Content (Join-Path $out "power.json")
Write-Host "power: $power, on mains: $ac"
$cliArgs = @("-m", $Model, "-ngl", "99", "-ncmoe", "24", "-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0",
             "-p", "0", "-n", "64", "-t", $Threads, "-r", "2", "-o", "jsonl")
$ErrorActionPreference = "Continue"  # PowerShell 5 turns native stderr lines into errors
& (Join-Path $Bin "llama-bench.exe") @cliArgs 2> (Join-Path $out "stderr.txt") | Tee-Object (Join-Path $out "bench.jsonl") |
  ForEach-Object { if ($_ -match '^\{') { $r = $_ | ConvertFrom-Json; Write-Host ("threads {0,3}: {1,6:N1} tok/s decode" -f $r.n_threads, $r.avg_ts) } }
Write-Host "THREADS DONE: $out" -ForegroundColor Green
