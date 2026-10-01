# Sequential GPU queue. One training process at a time: on this RTX 5060 three parallel sweeps ran at a combined
# 0.11 epochs/s against 0.18 for one alone, so serial is faster.
#
# Queue file: runs\gpu_queue.txt, one job per line, "<name> | <command line>". Lines starting with # are ignored.
# A job is done when its name is in runs\gpu_queue_done.txt. The runner always takes the FIRST line not yet done, so
# jobs can be inserted or reordered while it runs. It polls for new work when the queue is empty and exits when a
# line "STOP" is present.
#
# Only a job that exits 0 is marked done. Before 2026-09-22 every END marked its job done, which lost work twice:
# a path bug made five sweeps exit in seconds and be skipped for 16 hours, and a shutdown killed the training
# process before the runner, so a sweep at 5 of 40 seeds was recorded as finished. A job that fails is now skipped
# for the rest of THIS runner session only (so one broken line cannot spin forever) and retried on the next start;
# the log line says FAILED so it is visible. Sweeps resume per row, so a retry costs nothing already computed.
$root = "C:\Users\maurice\Desktop\hand_IK"
$queue = Join-Path $root "runs\gpu_queue.txt"
$done = Join-Path $root "runs\gpu_queue_done.txt"
$log = Join-Path $root "runs\gpu_queue_log.txt"
Set-Location $root
if (-not (Test-Path $done)) { New-Item -ItemType File $done | Out-Null }
$env:OMP_NUM_THREADS = '8'; $env:MKL_NUM_THREADS = '8'
$failedThisSession = @()
Add-Content $log ("[runner] started {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
while ($true) {
    $lines = @(Get-Content $queue | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') })
    if ($lines -contains 'STOP') { Add-Content $log ("[runner] STOP at {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")); break }
    $finished = @(Get-Content $done) + $failedThisSession
    $next = $lines | Where-Object { $_ -match '\|' -and ($finished -notcontains $_.Split('|')[0].Trim()) } | Select-Object -First 1
    if (-not $next) { Start-Sleep -Seconds 60; continue }
    $name = $next.Split('|')[0].Trim()
    $cmd = $next.Substring($next.IndexOf('|') + 1).Trim()
    Add-Content $log ("[runner] {0} START {1}" -f $name, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "$cmd > runs\${name}_log.txt 2>&1" -WindowStyle Hidden -PassThru -Wait
    if ($p.ExitCode -eq 0) {
        Add-Content $log ("[runner] {0} END exit 0 {1}" -f $name, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        Add-Content $done $name
    } else {
        Add-Content $log ("[runner] {0} FAILED exit {1} {2} - not marked done; retried on the next runner start" -f $name, $p.ExitCode, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        $failedThisSession += $name
    }
}
