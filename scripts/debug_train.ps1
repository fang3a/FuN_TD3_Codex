$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\anaconda\envs\python_3_8\python.exe"

Set-Location $ProjectRoot
& $Python hierarchical_td3_electric_gas.py `
  --training-stage joint_finetune `
  --episodes 1 `
  --episode-steps 20 `
  --manager-interval 40 `
  --batch-size 8 `
  --learning-starts 5 `
  --use-transition-model `
  --device cpu `
  --checkpoint-dir runs\debug
