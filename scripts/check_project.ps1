$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\anaconda\envs\python_3_8\python.exe"

Set-Location $ProjectRoot
& $Python -m py_compile electric_gas_microgrid_single.py hierarchical_td3_electric_gas.py evaluate_hierarchical_agent.py
& $Python hierarchical_td3_electric_gas.py --run-tests
