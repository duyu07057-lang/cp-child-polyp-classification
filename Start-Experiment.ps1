param(
    [ValidateSet('all','baseline','prepare','hand','traditional','cnn-features','hybrid','deep','evaluate','report')]
    [string]$Stage = 'all',
    [switch]$Background,
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
if (-not $Python) { $Python = Join-Path $projectDirectory '.venv_gpu\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python" }
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$scriptPath = Join-Path $projectDirectory 'run_experiment.py'
$configPath = Join-Path $projectDirectory 'configs\cp_child.json'
if ($Background) {
    $experimentConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $experimentDirectory = Join-Path $projectDirectory $experimentConfig.output_dir
    $logDirectory = Join-Path $experimentDirectory 'logs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $jobArguments = @('-u', ('"{0}"' -f $scriptPath), '--config', ('"{0}"' -f $configPath), '--stage', $Stage)
    $experimentProcess = Start-Process -FilePath $Python -ArgumentList $jobArguments -WorkingDirectory $projectDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory "$Stage.log") -RedirectStandardError (Join-Path $logDirectory "$Stage.err.log")
    Write-Output "Experiment launched. PID: $($experimentProcess.Id)"
    Write-Output "Log: $(Join-Path $logDirectory "$Stage.log")"
} else {
    Push-Location -LiteralPath $projectDirectory
    try { & $Python -u $scriptPath --config $configPath --stage $Stage }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
