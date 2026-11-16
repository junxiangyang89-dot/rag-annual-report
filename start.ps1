param(
    [Parameter(ValueFromRemainingArguments=$true)][string[]] $Arguments
)
$ErrorActionPreference = 'Stop'
$ragPython = if ($env:FINANCIAL_RAG_PYTHON) { $env:FINANCIAL_RAG_PYTHON } else { 'python' }
if (-not $Arguments) { $Arguments = @('web') }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $ragPython -B -X utf8 -m financial_rag @Arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
