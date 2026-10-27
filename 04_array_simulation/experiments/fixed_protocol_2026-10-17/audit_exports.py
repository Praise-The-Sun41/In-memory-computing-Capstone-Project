"""Final completeness and conservation audit of the declared native experiment set."""
from pathlib import Path
import hashlib
import json
import numpy as np
from analyse_native import load, read_levels

R = Path(__file__).resolve().parent

def audit():
    h = json.loads((R / 'history_summary.json').read_text(encoding='utf-8'))
    chosen = h['selected'] or h['candidates'][-1]['reset_count']
    tags = ['polarity_positive', 'polarity_negative', 'polarity_positive_refined',
            'history_high', 'history_low', 'history_aged']
    for candidate in h['candidates']:
        for j in range(4):
            for kind in ['init', 'probe']:
                tags.append(f"{kind}{candidate['reset_count']}_history{j}")
    window = json.loads((R / 'window_summary.json').read_text(encoding='utf-8'))
    tags.append(window['reference_tag'])
    for state in range(3):
        tags.extend(f'hold{t}_state{state}' for t in [1, 10, 100])
        tags.extend(f'{kind}{n}_state{state}' for kind in ['reads', 'readsham'] for n in [1, 16, 80])
        tags.extend(f'array2_{sign}_N{n}_state{state}_factor{factor}'
                    for sign in ['set', 'reset'] for n in [1, 16] for factor in [1.0, .5, 0.0])
    reference = load('polarity_positive')[0, 2]
    records = []
    for tag in tags:
        a = load(tag)
        drift = float(np.max(np.abs(a[:, 2] - reference)))
        assert drift <= 1e-6, (tag, 'Cross-case inventory discrepancy', drift)
        records.append(dict(case=tag, rows=len(a), final_tau=float(a[-1, 0]),
            read_levels=len(read_levels(a)), global_inventory_absolute_error=drift,
            min_occupancy=float(a[:, 4:700].min()), max_occupancy=float(a[:, 4:700].max()),
            sha256=hashlib.sha256((R / 'results' / (tag + '.csv')).read_bytes()).hexdigest()))
    for source in json.loads((R / 'source_manifest.json').read_text(encoding='utf-8')):
        assert hashlib.sha256((R / source['copy']).read_bytes()).hexdigest() == source['sha256']
    result = dict(passed=True, expected_cases=len(tags), validated_cases=len(records),
        maximum_global_inventory_absolute_error=max(x['global_inventory_absolute_error'] for x in records),
        scope='Completeness, imported model hash, finite values, physical bounds and cross-case inventory; scientific acceptance gates remain separate.', cases=records)
    result['single_run_continuous_completed'] = window['single_run_continuous_completed']
    if not window['single_run_continuous_completed']:
        partial = load('continuous_attempt_partial')
        prefix = partial[partial[:, 0] <= 64.42 + 1e-8]
        assert abs(prefix[-1, 0] - 64.42) < 1e-8
        assert np.max(np.abs(prefix[:, 2] - reference)) <= 1e-6
        result['recovered_reference_prefix'] = dict(completed_tau=64.42, failed_full_attempt=True,
            full_attempt_excluded_from_completed_count=True,
            sha256=hashlib.sha256((R / 'results/continuous_attempt_partial.csv').read_bytes()).hexdigest())
    (R / 'export_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))

if __name__ == '__main__':
    audit()
