@echo off
setlocal
cd /d D:\project\FuN_TD3_project
if not exist training_launch_logs\reward_v9_formal_20260715_0110 mkdir training_launch_logs\reward_v9_formal_20260715_0110
"D:\anaconda\anaconda\envs\python_3_8\python.exe" -u "D:\project\FuN_TD3_project\hierarchical_td3_electric_gas_optimized.py" --training-stage all --run-mode formal --device auto --checkpoint-dir hierarchical_td3_optimized_runs --fast-pretrain-episodes 50 --slow-pretrain-episodes 50 --manager-train-episodes 50 --joint-finetune-episodes 100 1>>"D:\project\FuN_TD3_project\training_launch_logs\reward_v9_formal_20260715_0110\stdout.log" 2>>"D:\project\FuN_TD3_project\training_launch_logs\reward_v9_formal_20260715_0110\stderr.log"
echo %errorlevel%>"D:\project\FuN_TD3_project\training_launch_logs\reward_v9_formal_20260715_0110\exit_code.txt"
endlocal
