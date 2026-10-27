$ErrorActionPreference='Stop'
$bin='E:\COMSOL\COMSOLMP64\COMSOL64\Multiphysics\bin\win64'
$python='D:\Python\Python312-amd32\python.exe'
$env:OPENBLAS_NUM_THREADS='1'
Push-Location $PSScriptRoot
try {
 & $python build_native.py
 if($LASTEXITCODE -ne 0){throw 'Builder generation failed'}
 foreach($stage in @(@('finish_contacts.py','FinishContacts','finish','CONTACT_GEOMETRY_AND_NATIVE_EXPORT_PASS'),@('hold_contacts.py','ContactHolds','hold','CONTACT_HOLD_neg_DONE'),@('audit_saved.py','AuditContactsSaved','audit','CONTACT_SAVED_HISTORY_AND_GEOMETRY_PASS'))) {
  $generator,$class,$log,$marker=$stage
  & $python $generator
  if($LASTEXITCODE -ne 0){throw "Generation failed: $class"}
  & "$bin/comsolcompile.exe" "$class.java" *> "$log-compile.log"
  if($LASTEXITCODE -ne 0){throw "Compile failed: $class"}
  & "$bin/comsolbatch.exe" -np 1 -alloc native -prefsdir '../../tmp/comsol-release/prefs' -tmpdir '../../tmp/comsol-release/temp' -autosave off -inputfile "$class.class" -batchlog "$log-batch.log" *> "$log-console.log"
  if((Get-Content "$log-console.log" -Raw) -notmatch $marker){throw "Failed: $class"}
 }
 & $python report_contacts.py
 if($LASTEXITCODE -ne 0){throw 'Comparison verification failed'}
 Write-Output 'Contacts history, holds, geometry and verification complete'
}finally{Pop-Location}
