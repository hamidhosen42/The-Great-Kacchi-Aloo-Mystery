"""Build five independent two-row flips of the submitted Q01 baseline."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/lb_probes"
KERNEL = ROOT / "solution/kaggle_lb_probe_kernel"


def fingerprint(rows):
    return hashlib.md5("".join(f"{r['wedding_id']}{r['went_back_for_seconds']}" for r in rows).encode()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    KERNEL.mkdir(exist_ok=True)
    baseline = list(csv.DictReader((ROOT / "outputs/portfolio_final/N0567.csv").open()))
    assert fingerprint(baseline) == "02cce4adcfe1dc9fe4dc99e3fd5021a4"
    template = list(csv.DictReader((ROOT / "data/sample_submission.csv").open()))
    assert [r['wedding_id'] for r in baseline] == [r['wedding_id'] for r in template]
    assert all(r['went_back_for_seconds'] in ('0', '1') for r in baseline)
    probs = list(csv.DictReader((ROOT / "outputs/edge_models/test_probabilities.csv").open()))
    assert [r['wedding_id'] for r in probs] == [r['wedding_id'] for r in baseline]
    # Match the original E30 model ranking; pair consecutive uncertainty ranks.
    lo, hi = 0.97068, 2.00983
    order = sorted(range(len(probs)), key=lambda i: (
        abs(float(probs[i]['E30 isotonic edges']) - .5),
        min(abs(float(probs[i]['apg']) - lo), abs(float(probs[i]['apg']) - hi)), i))
    manifest, hashes = [], {}
    for k in range(5):
        indices = order[2*k:2*k+2]
        rows = [dict(r) for r in baseline]
        for i in indices:
            rows[i]['went_back_for_seconds'] = str(1-int(rows[i]['went_back_for_seconds']))
        assert sum(a != b for a, b in zip(rows, baseline)) == 2
        name = f"LBP{k+1:02d}.csv"
        with (OUT / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(baseline[0])); w.writeheader(); w.writerows(rows)
        hashes[name] = fingerprint(rows)
        manifest.append(dict(file=name, baseline='Q01', baseline_score=0.96685,
                             uncertainty_ranks=[2*k+1, 2*k+2],
                             row_indices=indices, wedding_ids=[baseline[i]['wedding_id'] for i in indices],
                             old_predictions=[int(baseline[i]['went_back_for_seconds']) for i in indices],
                             new_predictions=[int(rows[i]['went_back_for_seconds']) for i in indices]))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    nb = json.loads((ROOT / 'submission_notebooks/01_0.96685_Q01_flip_E30_both_r1-7.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []; cell['execution_count'] = None
    code = '''# Independent probes, each compared against Q01 rather than the previous probe.
assert fp == '02cce4adcfe1dc9fe4dc99e3fd5021a4'
EXPECTED = EXPECTED_HASHES
manifest = []
for k in range(5):
    indices = order[2*k:2*k+2]
    probe = sub.copy()
    probe.loc[indices, 'went_back_for_seconds'] = 1 - probe.loc[indices, 'went_back_for_seconds']
    assert (probe.went_back_for_seconds != sub.went_back_for_seconds).sum() == 2
    name = f'LBP{k+1:02d}.csv'
    check = hashlib.md5(''.join(f'{i}{p}' for i,p in zip(probe.wedding_id, probe.went_back_for_seconds)).encode()).hexdigest()
    assert check == EXPECTED[name], (name, check)
    probe.to_csv(name, index=False)
    manifest.append({'file': name, 'wedding_ids': probe.loc[indices, 'wedding_id'].tolist(), 'baseline_score': 0.96685})
    print(name, manifest[-1]['wedding_ids'], check)
import json
with open('probe_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
'''.replace('EXPECTED_HASHES', repr(hashes))
    nb['cells'].append(dict(id='two-row-probes', cell_type='code', metadata={}, execution_count=None, outputs=[], source=code.splitlines(True)))
    (KERNEL / 'kacchi-aloo-two-row-probes.ipynb').write_text(json.dumps(nb, indent=1)+'\n')
    meta = json.loads((ROOT / 'solution/kaggle_final_kernel/kernel-metadata.json').read_text())
    meta.update(id='hosen42/kacchi-aloo-two-row-probes', title='Kacchi Aloo Two Row Probes', code_file='kacchi-aloo-two-row-probes.ipynb')
    (KERNEL / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    (OUT / 'README.md').write_text('Five independent two-row flips of Q01 (public accuracy 0.96685).\n'
        'Rows are chosen by the existing training-fitted E30 uncertainty ranking.\n'
        'Compare every score against Q01. Never treat an unchanged score as proof of private membership: '
        'two public rows may cancel. Positive/negative deltas describe net public improvement/decline, '
        'not individual labels. Assuming 181 public rows, deltas correspond to -2 through +2 net correct predictions.\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
