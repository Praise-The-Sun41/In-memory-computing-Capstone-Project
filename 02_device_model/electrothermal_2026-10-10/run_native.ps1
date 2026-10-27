param(
    [string]$Python = 'D:\Python\Python312-amd32\python.exe',
    [string]$ComsolBin = 'E:\COMSOL\COMSOLMP64\COMSOL64\Multiphysics\bin\win64',
    [switch]$Adaptive
)
$ErrorActionPreference='Stop'
$env:OPENBLAS_NUM_THREADS='1'
Push-Location $PSScriptRoot
try {
    $nativeClass=if ($Adaptive) { 'NativeThermalAdaptive' } else { 'NativeThermalTi' }
    $nativeCsv=if ($Adaptive) { 'Ti-thermal-adaptive-native.csv' } else { 'Ti-thermal-native.csv' }
    $nativeMph=if ($Adaptive) { 'Ti-thermal-adaptive-geometry.mph' } else { 'Ti-thermal-geometry.mph' }
    if (Test-Path -LiteralPath $nativeCsv) { throw 'Native data already exists. Inspect the comparison report; do not overwrite a completed run.' }
    if ($Adaptive) {
        if (-not (Test-Path -LiteralPath 'Ti-thermal-unsolved.mph')) { throw 'Adaptive runner requires the generated unsolved model supplied with this folder.' }
    } else {
        & $Python build_native.py
        if ($LASTEXITCODE -ne 0) { throw 'Generator failed.' }
    }
    $compileStart=Get-Date
    & (Join-Path $ComsolBin 'comsolcompile.exe') ($nativeClass+'.java') *> native-compile.log
    if (-not (Test-Path -LiteralPath ($nativeClass+'.class'))) { throw 'No compiled class.' }
    if ((Get-Item -LiteralPath ($nativeClass+'.class')).LastWriteTime -lt $compileStart -or
        (Get-Content -LiteralPath native-compile.log -Raw) -match 'Compilation failed|Compilation error') { throw 'Fresh compilation not confirmed.' }
    & (Join-Path $ComsolBin 'comsolbatch.exe') -np 4 -alloc native -autosave off -inputfile ($nativeClass+'.class') -batchlog native-batch.log *> native-console.log
    if (-not (Test-Path -LiteralPath $nativeCsv) -or -not (Test-Path -LiteralPath $nativeMph)) { throw 'Native solve/export incomplete. Read native-batch.log.' }
    if ($Adaptive) { & $Python compare_native.py --adaptive } else { & $Python compare_native.py }
    if ($LASTEXITCODE -ne 0) { throw 'Cross-solver comparison failed.' }
    $auditFlags=if ($Adaptive) { @('--adaptive') } else { @() }
    & $Python export_native_states.py @auditFlags
    if ($LASTEXITCODE -ne 0) { throw 'State-export generator failed.' }
    $auditClass=if ($Adaptive) { 'ExportThermalAdaptiveStates' } else { 'ExportThermalStates' }
    $compileStart=Get-Date
    & (Join-Path $ComsolBin 'comsolcompile.exe') ($auditClass+'.java') *> native-state-compile.log
    if (-not (Test-Path -LiteralPath ($auditClass+'.class'))) { throw 'No state-export class.' }
    if ((Get-Item -LiteralPath ($auditClass+'.class')).LastWriteTime -lt $compileStart -or
        (Get-Content -LiteralPath native-state-compile.log -Raw) -match 'Compilation failed|Compilation error') { throw 'State-export compilation failed.' }
    & (Join-Path $ComsolBin 'comsolbatch.exe') -np 4 -alloc native -autosave off -inputfile ($auditClass+'.class') -batchlog native-state-batch.log *> native-state-console.log
    & $Python check_native_states.py @auditFlags
    if ($LASTEXITCODE -ne 0) { throw 'Native ionic-state audit failed.' }
} finally { Pop-Location }
