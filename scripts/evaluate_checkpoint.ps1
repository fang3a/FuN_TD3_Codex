param(
  [Parameter(Mandatory = $true)]
  [string]$Checkpoint,
  [int]$Episodes = 1,
  [int]$EpisodeSteps = 480,
  [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\anaconda\envs\python_3_8\python.exe"

Set-Location $ProjectRoot
& $Python evaluate_hierarchical_agent.py `
  --checkpoint $Checkpoint `
  --episodes $Episodes `
  --episode-steps $EpisodeSteps `
  --device $Device
