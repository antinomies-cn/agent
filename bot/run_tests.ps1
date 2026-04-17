param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$explicitPython = "C:/ProgramData/anaconda3/envs/agentenv/python.exe"

if (Test-Path $explicitPython) {
    & $explicitPython -m pytest @PytestArgs
    exit $LASTEXITCODE
}

$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($null -ne $conda) {
    & conda run -n agentenv python -m pytest @PytestArgs
    exit $LASTEXITCODE
}

Write-Warning "agentenv Python not found. Falling back to current python."
python -m pytest @PytestArgs
exit $LASTEXITCODE
