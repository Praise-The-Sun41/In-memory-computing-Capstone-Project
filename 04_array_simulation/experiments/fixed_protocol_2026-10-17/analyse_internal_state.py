"""Diagnose retained spatial state from native exports; does not evolve the model."""
from pathlib import Path
import json
import numpy as np
from analyse_native import load, read_levels

R = Path(__file__).resolve().parent

def analyse():
    history = json.loads((R / 'history_summary.json').read_text(encoding='utf-8'))
    rows = []
    profiles = []
    for candidate in history['candidates']:
        n = candidate['reset_count']
        states = np.array([load(f'init{n}_history{j}')[-1, 4:700] for j in range(4)])
        curves = np.array([read_levels(load(f'probe{n}_history{j}')) for j in range(4)])
        spans = np.ptp(states, axis=0)
        profiles.append((n, states))
        rows.append(dict(reset_count=n,
            oxide_max_absolute_occupancy_spread=float(spans[:440].max()),
            ti_max_absolute_occupancy_spread=float(spans[440:].max()),
            physical_state_rms_range=float(np.sqrt(np.mean(spans**2))),
            programming_only_max_relative_G_spread=float(np.max(np.ptp(curves[:, 1:], axis=0) / curves[:, 1:].mean(axis=0))),
            downward_steps_by_history=[int(np.sum(np.diff(x) < -1e-3 * np.abs(x[:-1]))) for x in curves]))
    result = dict(scope='Spatial occupancy diagnostics at native RESET endpoints. Absolute occupancy spread is diagnostic, not a new pass threshold. Programming-only spread excludes level zero for diagnosis; the original all-level gate remains unchanged.', candidates=rows)
    (R / 'internal_state_summary.json').write_text(json.dumps(result, indent=2))
    if profiles:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, len(profiles), figsize=(6 * len(profiles), 6), squeeze=False)
        for col, (n, states) in enumerate(profiles):
            for j, label in enumerate(['Reference', 'Previously SET', 'Previously RESET', 'SET then held']):
                axes[0, col].plot(states[j, :440], label=label, lw=1)
                axes[1, col].plot(states[j, 440:], label=label, lw=1)
            for row, name in enumerate(['Oxide occupancy', 'Ti reservoir occupancy']):
                axes[row, col].set(title=f'{n} RESET pulses: {name}', xlabel='Native spatial state index', ylabel='Occupancy')
                axes[row, col].grid(alpha=.2)
                axes[row, col].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(R / 'internal_state_profiles.png', dpi=150)
        plt.close(fig)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    analyse()
