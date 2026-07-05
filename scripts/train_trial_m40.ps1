$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\anaconda\envs\python_3_8\python.exe"

Set-Location $ProjectRoot
& $Python hierarchical_td3_electric_gas.py `
  --training-stage all `
  --episodes 40 `
  --episode-steps 480 `
  --manager-interval 40 `
  --batch-size 256 `
  --learning-starts 1000 `
  --slow-update-interval-steps 5 `
  --manager-update-interval-steps 20 `
  --target-noise 0.08 `
  --target-noise-clip 0.20 `
  --target-q-clip-abs 200000 `
  --use-transition-model `
  --fast-exploration-noise 0.10 `
  --slow-exploration-noise 0.06 `
  --manager-exploration-noise 0.03 `
  --min-fast-exploration-noise 0.02 `
  --min-slow-exploration-noise 0.02 `
  --min-manager-exploration-noise 0.01 `
  --noise-decay-episodes 40 `
  --lambda-projection 10.0 `
  --worker-reward-clip-abs 5000 `
  --manager-reward-clip-abs 25000 `
  --worker-action-l2-weight 0.01 `
  --projection-imitation-weight 5.0 `
  --checkpoint-dir runs\m40_trial_stability_v1
