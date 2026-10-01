@echo off
cd /d C:\Users\maurice\Desktop\hand_IK
echo [shard c] OakInk2 subject x verb held 3, seeds 27-39, starting at %date% %time% >> runs\queue_subject_verb_h3_log.txt
.venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\oakink2_subject_verb.npz --granularity oakink2_subject_verb --out runs\oakink2_subject_verb_h3_c --budgets 256 --kinds perframe --held-compositions 3 --min-chains 4 --min-per-composition 5 --epochs 120 --seeds 27 28 29 30 31 32 33 34 35 36 37 38 39 > runs\oakink2_subject_verb_h3_c_log.txt 2>&1
echo [shard c] done exit %errorlevel% at %date% %time% >> runs\queue_subject_verb_h3_log.txt
