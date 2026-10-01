# Wait for the GPU side run grab_planted_s32 to finish, then run grab_grasp_s32 on the GPU with the same settings.
# CPU shards for grasp-only were tried and stopped: three of them took the CPU to 99% and slowed both GPU jobs
# about fivefold, while the CPU shards themselves ran at ~32 s per epoch.
$root = "C:\Users\maurice\Desktop\hand_IK"
$log = Join-Path $root "runs\gpu_queue_log.txt"
Set-Location $root
Add-Content $log ("[side] waiting for grab_planted_s32 to finish, {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
while (@(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'grab_planted_s32' }).Count -gt 0) {
    Start-Sleep -Seconds 30
}
Add-Content $log ("[side] grab_grasp_s32 START {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
$env:OMP_NUM_THREADS = '6'; $env:MKL_NUM_THREADS = '6'
$seeds = (0..39 | ForEach-Object { "$_" })
$a = @('-u', 'scripts\experiment_paired_composition.py', '--bundle', 'data\bundles\grab_grasp.npz', '--granularity', 'shape',
       '--out', 'runs\grab_grasp_s32', '--budgets', '256', '--kinds', 'perframe', '--held-compositions', '6',
       '--min-chains', '5', '--min-per-composition', '5', '--epochs', '120', '--window', '32', '--stride', '32', '--seeds') + $seeds
$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList $a -WorkingDirectory $root `
     -RedirectStandardOutput "$root\runs\grab_grasp_s32_log.txt" -RedirectStandardError "$root\runs\grab_grasp_s32_err.txt" `
     -WindowStyle Hidden -PassThru -Wait
Add-Content $log ("[side] grab_grasp_s32 END exit {0} {1}" -f $p.ExitCode, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
