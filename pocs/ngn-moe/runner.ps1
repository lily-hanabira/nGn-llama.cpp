# nGn MoE PoC job runner (Blade): runs each queue\*.ps1 in name order, one at a time, and logs it.
# A job is renamed .running while it runs and .done (or .failed) after; its output goes to logs\jobs\<name>.log.
# Stop with Ctrl+C or by closing the window. Only runs scripts from this folder's queue directory.
$ErrorActionPreference = "Continue"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;$env:PATH"
$queue = Join-Path $PSScriptRoot "queue"
$logs  = Join-Path $PSScriptRoot "logs\jobs"
New-Item -ItemType Directory -Force $queue, $logs | Out-Null
Write-Host "nGn PoC runner watching $queue" -ForegroundColor Cyan
while ($true) {
  $job = Get-ChildItem $queue -Filter *.ps1 | Where-Object { $_.Name -notlike "*.running.ps1" } | Sort-Object Name | Select-Object -First 1
  if (-not $job) { Start-Sleep -Seconds 2; continue }
  $name = $job.BaseName
  $running = Join-Path $queue "$name.running.ps1"
  Move-Item -Force $job.FullName $running
  $log = Join-Path $logs "$name.log"
  Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $name) -ForegroundColor Yellow
  "start $(Get-Date -Format o)" | Set-Content $log -Encoding utf8
  & powershell -NoProfile -ExecutionPolicy Bypass -File $running *>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
  $ok = $LASTEXITCODE -eq 0
  "end $(Get-Date -Format o) exit $LASTEXITCODE" | Add-Content $log -Encoding utf8
  Move-Item -Force $running (Join-Path $queue ("$name." + $(if ($ok) { "done" } else { "failed" })))
}
