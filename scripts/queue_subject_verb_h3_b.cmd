@echo off
cd /d C:\Users\maurice\Desktop\hand_IK
echo [shard b] OakInk2 subject x verb held 3, seeds 14-26, starting at %date% %time% >> runs\queue_subject_verb_h3_log.txt
.venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\oakink2_subject_verb.npz --granularity oakink2_subject_verb --out runs\oakink2_subject_verb_h3_b --budgets 256 --kinds perframe --held-compositions 3 --min-chains 4 --min-per-composition 5 --epochs 120 --seeds 14 15 16 17 18 19 20 21 22 23 24 25 26 > runs\oakink2_subject_verb_h3_b_log.txt 2>&1
echo [shard b] done exit %errorlevel% at %date% %time% >> runs\queue_subject_verb_h3_log.txt
