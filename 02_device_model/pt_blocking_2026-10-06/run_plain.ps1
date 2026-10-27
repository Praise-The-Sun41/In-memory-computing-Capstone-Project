$ErrorActionPreference='Stop'
$bin='E:\COMSOL\COMSOLMP64\COMSOL64\Multiphysics\bin\win64'
Push-Location $PSScriptRoot
try {
 & "$bin/comsolcompile.exe" PreparePtPlain.java *> plain-compile.log
 if($LASTEXITCODE -ne 0){throw 'Compile failed'}
 & "$bin/comsolbatch.exe" -np 1 -alloc native -autosave off -inputfile PreparePtPlain.class -batchlog plain-prepare.log *> plain-prepare-console.log
 if((Get-Content plain-prepare-console.log -Raw) -notmatch 'PT_PLAIN_READY'){throw 'Prepare failed'}
 & "$bin/comsolbatch.exe" -np 2 -alloc native -prefsdir '../../tmp/comsol-release/prefs' -tmpdir '../../tmp/comsol-release/temp' -autosave off -inputfile Pt-plain-unsolved.mph -outputfile Pt-plain-solved.mph -study std -batchlog plain-solve.log *> plain-solve-console.log
 if((Get-Content plain-solve.log -Raw) -notmatch 'Time-stepping completed\.'){throw 'Solve incomplete'}
 Write-Output 'Plain Pt native solve complete'
}finally{Pop-Location}
