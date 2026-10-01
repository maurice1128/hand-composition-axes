# GPU side chain: wait for grab_planted_s32 to finish, then run, one after another,
#   1. oakink_category_subject (readings: runs\PREREG_oakink_category_subject.md)
#   2. grab_grasp_s32          (readings: runs\PREREG_grab_diagnostics.md)
# Replaces scripts\after_planted_run_grasp_s32.ps1, which was stopped before it launched anything.
$root = "C:\Users\maurice\Desktop\hand_IK"
$log = Join-Path $root "runs\gpu_queue_log.txt"
Set-Location $root
$env:OMP_NUM_THREADS = '6'; $env:MKL_NUM_THREADS = '6'
$py = "$root\.venv\Scripts\python.exe"
$seeds = (0..39 | ForEach-Object { "$_" })

function Run-Job($name, $argList) {
    Add-Content $log ("[side] {0} START {1}" -f $name, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    $p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $root `
         -RedirectStandardOutput "$root\runs\${name}_log.txt" -RedirectStandardError "$root\runs\${name}_err.txt" `
         -WindowStyle Hidden -PassThru -Wait
    Add-Content $log ("[side] {0} END exit {1} {2}" -f $name, $p.ExitCode, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
}

Add-Content $log ("[side] chain waiting for grab_planted_s32, {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
while (@(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'grab_planted_s32' }).Count -gt 0) {
    Start-Sleep -Seconds 30
}

Run-Job "oakink_category_subject" (@('-u', 'scripts\experiment_paired_composition.py',
    '--bundle', 'data\bundles\oakink_category_subject.npz', '--granularity', 'oakink2_scene_verb',
    '--out', 'runs\oakink_category_subject', '--budgets', '256', '--kinds', 'perframe',
    '--held-compositions', '4', '--min-chains', '4', '--min-per-composition', '5',
    '--epochs', '120', '--window', '32', '--stride', '4', '--seeds') + $seeds)

Run-Job "grab_grasp_s32" (@('-u', 'scripts\experiment_paired_composition.py',
    '--bundle', 'data\bundles\grab_grasp.npz', '--granularity', 'shape',
    '--out', 'runs\grab_grasp_s32', '--budgets', '256', '--kinds', 'perframe',
    '--held-compositions', '6', '--min-chains', '5', '--min-per-composition', '5',
    '--epochs', '120', '--window', '32', '--stride', '32', '--seeds') + $seeds)
