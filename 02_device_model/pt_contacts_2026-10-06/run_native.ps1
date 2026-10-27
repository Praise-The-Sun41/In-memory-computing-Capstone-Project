$ErrorActionPreference='Stop'
$bin='E:\COMSOL\COMSOLMP64\COMSOL64\Multiphysics\bin\win64'
$python='D:\Python\Python312-amd32\python.exe'
$env:OPENBLAS_NUM_THREADS='1'
Push-Location $PSScriptRoot
try {
 & $python build_native.py
 if($LASTEXITCODE -ne 0){throw 'Generation failed'}
 & "$bin/comsolcompile.exe" PrepareContacts.java *> prepare-compile.log
 if($LASTEXITCODE -ne 0){throw 'Compile failed'}
 & "$bin/comsolbatch.exe" -np 1 -alloc native -prefsdir '../../tmp/comsol-release/prefs' -tmpdir '../../tmp/comsol-release/temp' -autosave off -inputfile PrepareContacts.class -batchlog prepare-batch.log *> prepare-console.log
 if((Get-Content prepare-console.log -Raw) -notmatch 'PREPARED_CONTACTS_882'){throw 'Prepare failed'}
 & "$bin/comsolbatch.exe" -np 2 -alloc native -prefsdir '../../tmp/comsol-release/prefs' -tmpdir '../../tmp/comsol-release/temp' -autosave off -inputfile Pt-contacts-unsolved.mph -outputfile Pt-contacts-solved.mph -study std -batchlog solve-batch.log *> solve-console.log
 if((Get-Content solve-batch.log -Raw) -notmatch 'Time-stepping completed\.'){throw 'Solve incomplete'}
 Write-Output 'Native contacts history solved'
}finally{Pop-Location}
