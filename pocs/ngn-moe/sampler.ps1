# Samples free memory and paging every 2 s into a CSV until -StopFile appears.
param([string]$Csv, [string]$StopFile)
"time,availableMB,pagesInputPerSec,pageFaultsPerSec,commitPct" | Set-Content $Csv
$c = @("\Memory\Available MBytes", "\Memory\Pages Input/sec", "\Memory\Page Faults/sec", "\Memory\% Committed Bytes In Use")
while (-not (Test-Path $StopFile)) {
  try { $s = (Get-Counter -Counter $c -ErrorAction Stop).CounterSamples
        "{0},{1:F0},{2:F0},{3:F0},{4:F1}" -f (Get-Date -Format "HH:mm:ss"), $s[0].CookedValue, $s[1].CookedValue, $s[2].CookedValue, $s[3].CookedValue | Add-Content $Csv } catch {}
  Start-Sleep -Seconds 2
}
