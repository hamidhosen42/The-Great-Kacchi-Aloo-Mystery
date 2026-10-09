"""Push the private notebook, submit its five outputs, and record actual scores."""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
token_matches = re.findall(r'(?m)^KGAT_[A-Za-z0-9_-]+$', (ROOT / 'token.txt').read_text())
assert token_matches, 'No standalone Kaggle API token found in token.txt'
os.environ['KAGGLE_API_TOKEN'] = token_matches[0]
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = 'the-great-kacchi-aloo-mystery'
KERNEL = 'hosen42/kacchi-aloo-two-row-probes'
OUT = ROOT / 'outputs/lb_probes'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['push', 'status', 'submit', 'scores'])
    p.add_argument('--version', type=int)
    args = p.parse_args()
    api = KaggleApi(); api.authenticate()
    if args.action == 'push':
        result = api.kernels_push(str(ROOT / 'solution/kaggle_lb_probe_kernel'))
        print(result)
        (OUT / 'push_result.txt').write_text(str(result)+'\n')
    elif args.action == 'status':
        print(api.kernels_status(KERNEL))
    elif args.action == 'submit':
        assert args.version is not None, 'Specify the verified completed kernel version'
        receipts_path = OUT / 'receipts.json'
        receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
        for item in json.loads((OUT / 'manifest.json').read_text()):
            name = item['file']
            if name in receipts:
                continue
            result = api.competition_submit_code(name,
                f"{name[:-4]}: Q01 + two E30 uncertainty rows flipped; independent probe",
                competition=COMP, kernel=KERNEL, kernel_version=args.version, quiet=True)
            receipts[name] = str(result)
            receipts_path.write_text(json.dumps(receipts, indent=2)+'\n')
            print(name, result, flush=True)
    else:
        submissions = api.competition_submissions(COMP, page_size=100)
        results = []
        for s in submissions or []:
            name = s.file_name
            if name not in {f'LBP{k:02d}.csv' for k in range(1,6)}:
                continue
            score = s.public_score
            row = dict(file=name, ref=s.ref, status=str(s.status), public_score=score)
            if score not in (None, ''):
                delta = round(float(score)-.96685, 5)
                row.update(delta=delta, net_correct_change=round(delta*181),
                           interpretation='unresolved: private or public cancellation' if delta == 0 else
                           'net public improvement' if delta > 0 else 'net public decline')
            results.append(row)
        (OUT / 'results.json').write_text(json.dumps(results, indent=2, default=str)+'\n')
        manifest = {r['file']: r for r in json.loads((OUT / 'manifest.json').read_text())}
        lines = ['# Two-row probe results', '', 'Baseline: Q01, public accuracy 0.96685.', '',
                 '| Probe | Rows | Score | Net public correct change |',
                 '|---|---|---:|---:|']
        for r in sorted(results, key=lambda r: r['file']):
            lines.append(f"| {r['file']} | {', '.join(manifest[r['file']]['wedding_ids'])} | "
                         f"{r['public_score']} | {r.get('net_correct_change', 'pending')} |")
        lines.extend(['', 'Assuming the inferred 181 public rows, a -1 change means exactly one row '
                      'in that pair is public and its flip hurt; the other is private. The pair does not '
                      'identify which row is public. Zero change remains ambiguous: both private, '
                      'or two public flips cancelling. No individual labels or final selections were changed.'])
        (OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
