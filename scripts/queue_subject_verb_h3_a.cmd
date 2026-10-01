@echo off
cd /d C:\Users\maurice\Desktop\hand_IK
echo [shard a] OakInk2 subject x verb held 3, seeds 0-13, starting at %date% %time% >> runs\queue_subject_verb_h3_log.txt
.venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\oakink2_subject_verb.npz --granularity oakink2_subject_verb --out runs\oakink2_subject_verb_h3_a --budgets 256 --kinds perframe --held-compositions 3 --min-chains 4 --min-per-composition 5 --epochs 120 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 > runs\oakink2_subject_verb_h3_a_log.txt 2>&1
echo [shard a] done exit %errorlevel% at %date% %time% >> runs\queue_subject_verb_h3_log.txt
