$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$CliScript = Join-Path $ScriptDir "scripts\posementor.py"

if (Get-Command uv -ErrorAction SilentlyContinue) {
  & uv run --project $ScriptDir python $CliScript @args
  exit $LASTEXITCODE
}

if (Test-Path $LocalPython) {
  & $LocalPython $CliScript @args
  exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
  & py -3 $CliScript @args
  exit $LASTEXITCODE
}

& python $CliScript @args
exit $LASTEXITCODE
