# Shard c of the OakInk2 subject x verb (held 3) sweep failed at start with "Memory allocation failure": the system
# commit charge was within 2.7 GB of its limit because of other jobs. Wait until at least 8 GB of commit is free
# (checked every 30 s, for up to 36 h), then launch shard c. Rows already in results.json are skipped on restart.
$log = "C:\Users\maurice\Desktop\hand_IK\runs\queue_subject_verb_h3_log.txt"
$deadline = (Get-Date).AddHours(36)
Add-Content $log ("[shard c] relauncher waiting for 8 GB free commit, started {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
while ((Get-Date) -lt $deadline) {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGB = $os.FreeVirtualMemory / 1MB
    if ($freeGB -ge 8) {
        Add-Content $log ("[shard c] relaunching at {0}, commit free {1:N1} GB" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $freeGB)
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "C:\Users\maurice\Desktop\hand_IK\scripts\queue_subject_verb_h3_c.cmd" -WindowStyle Hidden
        exit 0
    }
    Start-Sleep -Seconds 30
}
Add-Content $log ("[shard c] relauncher gave up at {0}: commit never reached 8 GB free" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
