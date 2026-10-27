param([ValidateSet('chemical_500','chemical_5000')][string]$Case='chemical_500', [switch]$Smoke)
$ErrorActionPreference='Stop'
$suffix=if($Smoke){'-smoke'}else{'-full'}
$runDir=Join-Path $PSScriptRoot ($Case+$suffix)
$manifest=Get-Content -LiteralPath (Join-Path $runDir 'manifest.json') -Raw | ConvertFrom-Json
$nativeClass=$manifest.class_name
$comsolBin='E:\COMSOL\COMSOLMP64\COMSOL64\Multiphysics\bin\win64'
Push-Location $runDir
try {
    if(Test-Path -LiteralPath 'Ti-chemistry-contact.mph'){throw 'Completed model exists; inspect its comparison instead of overwriting.'}
    $start=Get-Date
    & (Join-Path $comsolBin 'comsolcompile.exe') ($nativeClass+'.java') *> compile.log
    if(-not (Test-Path -LiteralPath ($nativeClass+'.class'))){throw 'Compilation did not produce a class.'}
    if((Get-Item -LiteralPath ($nativeClass+'.class')).LastWriteTime -lt $start){throw 'Compiled class is stale.'}
    & (Join-Path $comsolBin 'comsolbatch.exe') -np 4 -alloc native -autosave off -inputfile ($nativeClass+'.class') -batchlog batch.log *> console.log
    if(-not (Test-Path -LiteralPath 'Ti-chemistry-contact.mph') -or -not (Test-Path -LiteralPath 'states.csv')){throw 'Native solve/export incomplete; inspect logs.'}
    $env:OPENBLAS_NUM_THREADS='1'
    $flags=if($Smoke){@('--smoke')}else{@()}
    & 'D:\Python\Python312-amd32\python.exe' (Join-Path $PSScriptRoot 'compare.py') --case $Case @flags *> comparison-console.log
    if($LASTEXITCODE -ne 0){throw 'Native comparison failed; no acceptance claim.'}
} finally {Pop-Location}
