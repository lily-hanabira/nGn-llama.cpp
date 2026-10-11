# Simulates a smaller machine: locks physical RAM so only -LeaveGiB remains for Windows, apps and the model.
# Locked pages cannot be paged out (unlike a plain allocation). Exits when -StopFile appears, or after -MaxMinutes.
param([double]$LeaveGiB = 16, [string]$StopFile, [int]$MaxMinutes = 120, [string]$LogFile)
Add-Type @"
using System; using System.Runtime.InteropServices;
public static class Hog {
  [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr VirtualAlloc(IntPtr a, UIntPtr s, uint t, uint p);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool VirtualLock(IntPtr a, UIntPtr s);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetProcessWorkingSetSize(IntPtr h, UIntPtr mn, UIntPtr mx);
  [DllImport("kernel32.dll")] public static extern IntPtr GetCurrentProcess();
}
"@
function Log($m) { $l = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m; Write-Output $l; if ($LogFile) { Add-Content $LogFile $l } }
$total = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$lock  = [int64]($total - $LeaveGiB*1GB)
if ($lock -le 0) { Log "nothing to lock: total $([math]::Round($total/1GB,1)) GiB"; exit 0 }
$chunk = 256MB; $n = [int][math]::Floor($lock / $chunk); $lock = [int64]$n*$chunk
if (-not [Hog]::SetProcessWorkingSetSize([Hog]::GetCurrentProcess(), [UIntPtr]([uint64]($lock + 512MB)), [UIntPtr]([uint64]($lock + 1GB)))) {
  Log "SetProcessWorkingSetSize failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"; exit 1 }
$locked = 0
for ($i = 0; $i -lt $n; $i++) {
  $p = [Hog]::VirtualAlloc([IntPtr]::Zero, [UIntPtr]([uint64]$chunk), 0x3000, 0x04)
  if ($p -eq [IntPtr]::Zero -or -not [Hog]::VirtualLock($p, [UIntPtr]([uint64]$chunk))) { Log "lock stopped at $([math]::Round($locked/1GB,2)) GiB: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"; break }
  $locked += $chunk
}
Log ("locked {0:N2} GiB of {1:N2} GiB, leaving {2:N2} GiB" -f ($locked/1GB), ($total/1GB), (($total-$locked)/1GB))
$deadline = (Get-Date).AddMinutes($MaxMinutes)
while (-not (Test-Path $StopFile) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
Log "releasing"
