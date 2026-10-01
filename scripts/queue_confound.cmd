@echo off
cd /d C:\Users\maurice\Desktop\hand_IK
echo [queue] OakInk2 scene x primitive (perframe only, confound test) starting at %date% %time% >> runs\queue_confound_log.txt
.venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\oakink2_scene_primitive.npz --granularity oakink2_scene_primitive --out runs\oakink2_scene_primitive --budgets 256 --kinds perframe --held-compositions 5 --min-chains 4 --min-per-composition 5 --epochs 120 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 > runs\oakink2_scene_primitive_log.txt 2>&1
echo [queue] scene x primitive done exit %errorlevel% at %date% %time% >> runs\queue_confound_log.txt
echo [queue] pc_easy rerun under the current split builder (perframe only) starting at %date% %time% >> runs\queue_confound_log.txt
.venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\synthetic_big.npz --granularity fine --out runs\pc_easy_rerun --budgets 256 64 --kinds perframe --held-compositions 8 --min-chains 4 --min-per-composition 5 --epochs 120 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 > runs\pc_easy_rerun_log.txt 2>&1
echo [queue] pc_easy rerun done exit %errorlevel% at %date% %time% >> runs\queue_confound_log.txt
echo [queue] pc_hard rerun under the current split builder (perframe only) starting at %date% %time% >> runs\queue_confound_log.txt
.venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\synth_hard.npz --granularity fine --out runs\pc_hard_rerun --budgets 256 --kinds perframe --held-compositions 8 --min-chains 4 --min-per-composition 5 --epochs 120 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 > runs\pc_hard_rerun_log.txt 2>&1
echo [queue] pc_hard rerun done exit %errorlevel% at %date% %time% >> runs\queue_confound_log.txt
