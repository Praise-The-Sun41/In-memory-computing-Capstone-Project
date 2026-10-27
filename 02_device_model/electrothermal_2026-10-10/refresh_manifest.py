"""Verify frozen dependencies before recording this isolated extension's sources."""
from pathlib import Path
import hashlib,json

P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'provenance.json').read_text())
for name,expected in manifest['frozen_dependencies'].items():
    assert sha(P.parent.parent/name)==expected, f'Frozen dependency changed: {name}'
inputs=('comsol/outputs/ti-capacity-T30-L256-O8-C10-R1e-08.json',
        'comsol/outputs/bounded-comsol-linear10-L256-sparse-raw.csv',
        'comsol/presentation_geometry_2026-10-04/Memristor_Geometry_Bound.mph')
if 'frozen_inputs' not in manifest:
    manifest['frozen_inputs']={name:sha(P.parent.parent/name) for name in inputs}
for name,expected in manifest['frozen_inputs'].items():
    assert sha(P.parent.parent/name)==expected, f'Frozen input changed: {name}'
manifest['frozen_dependencies_unchanged']=True
manifest['new_sources']={p.name:sha(p) for p in sorted(P.iterdir())
                         if p.suffix in ('.py','.java','.ps1','.md')}
(P/'provenance.json').write_text(json.dumps(manifest,indent=2))
print('Frozen dependencies unchanged; current source hashes saved.')
