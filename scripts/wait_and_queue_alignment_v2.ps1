# Unattended gatekeeper for the alignment-v2 test (runs\PREREG_oakink_alignment_v2.md).
# Waits up to 14 h for each long3 bundle to exist and stop changing, runs scripts\gate_alignment_v2.py on it,
# and appends its sweep to runs\gpu_queue.txt only if the gate passes. Queued lines go before the lower-priority
# transitions re-run so the GPU runner takes them next.
$root = "C:\Users\maurice\Desktop\hand_IK"
$log = Join-Path $root "runs\gpu_queue_log.txt"
$queue = Join-Path $root "runs\gpu_queue.txt"
Set-Location $root
$env:OMP_NUM_THREADS = '6'
$seeds = (0..39) -join ' '
$todo = @(
    @{ name = 'oakink_long3_aligned';    run = 'pc_oakink_long3_aligned' },
    @{ name = 'oakink_long3_misaligned'; run = 'pc_oakink_long3_misaligned' }
)
Add-Content $log ("[gate-v2] waiting for long3 bundles, {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
$deadline = (Get-Date).AddHours(14)
while ($todo.Count -gt 0 -and (Get-Date) -lt $deadline) {
    foreach ($t in @($todo)) {
        $f = Join-Path $root ("data\bundles\{0}.npz" -f $t.name)
        if (-not (Test-Path $f)) { continue }
        $age = ((Get-Date) - (Get-Item $f).LastWriteTime).TotalSeconds
        if ($age -lt 120) { continue }   # still being written
        $p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList @('scripts\gate_alignment_v2.py', ("data\bundles\{0}.npz" -f $t.name), $t.name) -WorkingDirectory $root -RedirectStandardOutput "$root\runs\gates\gate_alignment_v2_$($t.name)_stdout.txt" -RedirectStandardError "$root\runs\gates\gate_alignment_v2_$($t.name)_stderr.txt" -WindowStyle Hidden -PassThru -Wait
        if ($p.ExitCode -eq 0) {
            $line = "{0} | .venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\{1}.npz --granularity oakink_category --out runs\{0} --budgets 64 --kinds perframe --held-compositions 5 --min-chains 4 --min-per-composition 5 --epochs 120 --window 32 --stride 4 --seeds {2}" -f $t.run, $t.name, $seeds
            $text = Get-Content $queue -Raw
            $anchor = "# Stopped at 13:04 to free the GPU"
            $text = $text.Replace($anchor, "$line`r`n$anchor")
            Set-Content -Path $queue -Value $text -NoNewline -Encoding UTF8
            Add-Content $log ("[gate-v2] {0} PASS, queued {1} at {2}" -f $t.name, $t.run, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        } else {
            Add-Content $log ("[gate-v2] {0} FAIL (exit {1}), not queued, {2}" -f $t.name, $p.ExitCode, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        }
        $todo = @($todo | Where-Object { $_.name -ne $t.name })
    }
    Start-Sleep -Seconds 60
}
Add-Content $log ("[gate-v2] finished, {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
