$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$jupyterData = Join-Path $project '.venv_gpu\share\jupyter'
$notebook = Join-Path $PSScriptRoot '手工与CNN特征融合_SVM全过程.ipynb'
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) { throw 'Python not found; activate the cpchild environment described in README.md.' }
$jupyterPython = $pythonCommand.Source
if (Test-Path -LiteralPath (Join-Path $jupyterData 'kernels\cpchild-gpu\kernel.json')) {
    if ($env:JUPYTER_PATH) { $env:JUPYTER_PATH = "$jupyterData;$env:JUPYTER_PATH" }
    else { $env:JUPYTER_PATH = $jupyterData }
}
Push-Location -LiteralPath $project
try { & $jupyterPython -m notebook $notebook }
finally { Pop-Location }
