@echo off
cd /d C:\Users\maurice\Desktop\hand_IK
echo [queue] GRAB shape x intent class, resuming at %date% %time% >> runs\queue_necessity_log.txt
.venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\grab.npz --granularity grab_shape_intentclass --out runs\grab_shapeclass --budgets 256 --kinds perframe modular --held-compositions 5 --min-chains 4 --min-per-composition 5 --epochs 120 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 >> runs\grab_shapeclass_log.txt 2>&1
echo [queue] GRAB done exit %errorlevel% at %date% %time% >> runs\queue_necessity_log.txt
echo [queue] OakInk2 scene x verb starting at %date% %time% >> runs\queue_necessity_log.txt
.venv\Scripts\python.exe -u scripts\experiment_paired_composition.py --bundle data\bundles\oakink2_scene_verb.npz --granularity oakink2_scene_verb --out runs\oakink2_scene_verb --budgets 256 --kinds perframe modular --held-compositions 5 --min-chains 4 --min-per-composition 5 --epochs 120 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 > runs\oakink2_scene_verb_log.txt 2>&1
echo [queue] scene x verb done exit %errorlevel% at %date% %time% >> runs\queue_necessity_log.txt
