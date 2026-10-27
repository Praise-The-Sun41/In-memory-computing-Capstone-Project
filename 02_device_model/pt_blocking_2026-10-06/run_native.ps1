$ErrorActionPreference='Stop'
$bin='E:\COMSOL\COMSOLMP64\COMSOL64\Multiphysics\bin\win64'
$python='D:\Python\Python312-amd32\python.exe'
$env:OPENBLAS_NUM_THREADS='1'
Push-Location $PSScriptRoot
try {
 & $python pt_model.py java
 if($LASTEXITCODE -ne 0){throw 'Generation failed'}
 & "$bin/comsolcompile.exe" PreparePt.java *> prepare-compile.log
 if($LASTEXITCODE -ne 0){throw 'Compile failed'}
 & "$bin/comsolbatch.exe" -np 1 -alloc native -autosave off -inputfile PreparePt.class -batchlog prepare-batch.log *> prepare-console.log
 if((Get-Content prepare-console.log -Raw) -notmatch 'PREPARED_PT_881'){throw 'Preparation failed'}
 # Solve only the physical equations; attach display geometry afterwards.
 & ./run_plain.ps1
 & $python finish_pt.py
 if($LASTEXITCODE -ne 0){throw 'Finish generation failed'}
 & "$bin/comsolcompile.exe" FinishPt.java *> finish-compile.log
 if($LASTEXITCODE -ne 0){throw 'Finish compile failed'}
 & "$bin/comsolbatch.exe" -np 1 -alloc native -prefsdir '../../tmp/comsol-release/prefs' -tmpdir '../../tmp/comsol-release/temp' -autosave off -inputfile FinishPt.class -batchlog finish-batch.log *> finish-console.log
 if((Get-Content finish-console.log -Raw) -notmatch 'PT_GEOMETRY_AND_NATIVE_EXPORT_PASS'){throw 'Geometry finish failed'}
 & $python pt_hold_native.py
 if($LASTEXITCODE -ne 0){throw 'Hold generation failed'}
 & "$bin/comsolcompile.exe" PtHolds.java *> hold-compile.log
 if($LASTEXITCODE -ne 0){throw 'Hold compile failed'}
 & "$bin/comsolbatch.exe" -np 2 -alloc native -prefsdir '../../tmp/comsol-release/prefs' -tmpdir '../../tmp/comsol-release/temp' -autosave off -inputfile PtHolds.class -batchlog hold-batch.log *> hold-console.log
 if((Get-Content hold-console.log -Raw) -notmatch 'PT_HOLD_neg_DONE'){throw 'Native holds failed'}
 & $python audit_saved.py
 if($LASTEXITCODE -ne 0){throw 'Saved-model audit generation failed'}
 & "$bin/comsolcompile.exe" AuditPtSaved.java *> audit-compile.log
 if($LASTEXITCODE -ne 0){throw 'Saved-model audit compile failed'}
 & "$bin/comsolbatch.exe" -np 1 -alloc native -prefsdir '../../tmp/comsol-release/prefs' -tmpdir '../../tmp/comsol-release/temp' -autosave off -inputfile AuditPtSaved.class -batchlog audit-batch.log *> audit-console.log
 if((Get-Content audit-console.log -Raw) -notmatch 'PT_SAVED_HISTORY_AND_GEOMETRY_PASS'){throw 'Saved model verification failed'}
 & $python report_pt.py
 if($LASTEXITCODE -ne 0){throw 'Cross-solver audit failed'}
 Write-Output 'Native Pt solve and geometry saved.'
} finally {Pop-Location}
