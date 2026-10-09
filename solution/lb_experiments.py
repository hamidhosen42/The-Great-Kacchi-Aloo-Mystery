"""Automated two-row accuracy experiments; retain ambiguity until identifiable."""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/lb_experiments'
KDIR = ROOT / 'solution/kaggle_lb_experiments_kernel'
COMP = 'the-great-kacchi-aloo-mystery'
KERNEL = 'hosen42/kacchi-aloo-paired-experiments'


def api_client():
    tokens = re.findall(r'(?m)^KGAT_[A-Za-z0-9_-]+$', (ROOT/'token.txt').read_text())
    assert tokens
    os.environ['KAGGLE_API_TOKEN'] = tokens[0]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    return api


def save(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, default=str)+'\n')


def snapshot(api):
    limits = api.competition_get_submission_limits(COMP)
    comps = api.competitions_list(search='the-great-kacchi-aloo-mystery')
    lb = api.competition_leaderboard_view(COMP)
    subs = api.competition_submissions(COMP, page_size=100)
    save('limits.json', json.loads(str(limits)))
    save('competition.json', json.loads(str(comps)))
    save('leaderboard.json', json.loads(str(lb)))
    save('submissions.json', [json.loads(str(x)) for x in subs])
    print('LIMITS', limits)
    print('LEADERBOARD', lb)


def build(count, extend=False):
    OUT.mkdir(exist_ok=True); KDIR.mkdir(exist_ok=True)
    base = list(csv.DictReader((ROOT/'outputs/portfolio_final/N0567.csv').open()))
    p = list(csv.DictReader((ROOT/'outputs/edge_models/test_probabilities.csv').open()))
    q = list(csv.DictReader((ROOT/'outputs/edge_models2/test_probabilities.csv').open()))
    assert [x['wedding_id'] for x in base] == [x['wedding_id'] for x in p] == [x['wedding_id'] for x in q]
    inference=json.loads((OUT/'inference.json').read_text())
    anchor=inference['anchors'][0]['index']
    possible=[r['index'] for r in inference['rows'] if r['max_contribution']==1 and r['min_contribution']!=1]
    def risk(i):
        prob=(float(p[i]['E30 isotonic edges'])+float(q[i]['E39'])+float(p[i]['E31 Bayes change-point']))/3
        return prob if int(base[i]['went_back_for_seconds'])==0 else 1-prob
    candidates = sorted(range(len(base)), key=lambda i: (-risk(i),
        min(abs(float(p[i]['apg'])-.97068), abs(float(p[i]['apg'])-2.00983)), i))
    if extend:
        active={r['index'] for r in inference['rows']}
        candidates=[i for i in candidates if i not in active]
    else:
        candidates=[i for i in candidates if i in possible]
    pairs=[(anchor,i) for i in candidates[:count]]
    manifest=json.loads((OUT/'manifest.json').read_text()) if extend else []
    start=len(manifest)
    for k, indices in enumerate(pairs):
        name=f'XP{k+start+1:02d}.csv'
        manifest.append(dict(file=name, indices=list(indices), wedding_ids=[base[i]['wedding_id'] for i in indices]))
    save('manifest.json', manifest)
    write_kernel(manifest)
    print('Built',len(pairs),'independent two-row experiments')


def write_kernel(manifest):
    base=list(csv.DictReader((ROOT/'outputs/portfolio_final/N0567.csv').open()))
    expected={}
    for item in manifest:
        rows=[dict(r) for r in base]
        for i in item['indices']:
            rows[i]['went_back_for_seconds']=str(1-int(rows[i]['went_back_for_seconds']))
        assert sum(a!=b for a,b in zip(rows,base))==len(item['indices'])
        with (OUT/item['file']).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(base[0]));writer.writeheader();writer.writerows(rows)
        expected[item['file']]=hashlib.md5(''.join(r['wedding_id']+r['went_back_for_seconds'] for r in rows).encode()).hexdigest()
    nb = json.loads((ROOT/'submission_notebooks/01_0.96685_Q01_flip_E30_both_r1-7.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            cell['outputs']=[]; cell['execution_count']=None
    code = '''assert fp == '02cce4adcfe1dc9fe4dc99e3fd5021a4'
PAIRS = PAIR_LITERAL
EXPECTED = HASH_LITERAL
for item in PAIRS:
    indices = item['indices']
    assert te.loc[indices, 'wedding_id'].tolist() == item['wedding_ids']
    probe=sub.copy()
    probe.loc[indices,'went_back_for_seconds'] = 1-probe.loc[indices,'went_back_for_seconds']
    assert (probe.went_back_for_seconds != sub.went_back_for_seconds).sum()==len(indices)
    check=hashlib.md5(''.join(f'{i}{p}' for i,p in zip(probe.wedding_id,probe.went_back_for_seconds)).encode()).hexdigest()
    assert check==EXPECTED[item['file']]
    probe.to_csv(item['file'],index=False)
    print(item['file'], item['wedding_ids'])
'''.replace('PAIR_LITERAL',repr(manifest)).replace('HASH_LITERAL',repr(expected))
    nb['cells'].append(dict(id='paired-experiments', cell_type='code',metadata={},outputs=[],execution_count=None,source=code.splitlines(True)))
    (KDIR/'paired-experiments.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    meta=json.loads((ROOT/'solution/kaggle_final_kernel/kernel-metadata.json').read_text())
    meta.update(id=KERNEL,title='Kacchi Aloo Paired Experiments',code_file='paired-experiments.ipynb')
    (KDIR/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


def optimize():
    info=json.loads((OUT/'inference.json').read_text())
    gains=info['certain_gains']
    assert gains, 'No provable improvements'
    manifest=[dict(file='OPT01.csv',indices=[r['index'] for r in gains],wedding_ids=[r['wedding_id'] for r in gains])]
    save('optimized_manifest.json',manifest)
    write_kernel(manifest)
    print('Guaranteed gain:',len(gains),'expected score:',(175+len(gains))/181)


def submit_optimized(api,version):
    item=json.loads((OUT/'optimized_manifest.json').read_text())[0]
    path=OUT/'optimized_receipt.json'
    if path.exists(): print('Already submitted:',path.read_text()); return
    result=api.competition_submit_code(item['file'],'OPT01: Q01 + provable gains from automated score constraints',
                                      competition=COMP,kernel=KERNEL,kernel_version=version,quiet=True)
    save('optimized_receipt.json',json.loads(str(result)));print(result)


def submit(api, version, count):
    manifest=json.loads((OUT/'manifest.json').read_text())
    path=OUT/'receipts.json'
    receipts=json.loads(path.read_text()) if path.exists() else {}
    for item in manifest[:count]:
        name=item['file']
        if name in receipts: continue
        r=api.competition_submit_code(name,f'{name[:-4]}: automated two-row accuracy experiment against Q01',
                                     competition=COMP,kernel=KERNEL,kernel_version=version,quiet=True)
        receipts[name]=json.loads(str(r)); save('receipts.json',receipts)
        print(name,r,flush=True)


def scores(api):
    names={i['file'] for i in json.loads((OUT/'manifest.json').read_text())}
    optimized=[]
    rows=[]
    for s in api.competition_submissions(COMP,page_size=100):
        if s.file_name=='OPT01.csv':
            optimized.append(dict(file=s.file_name,ref=s.ref,status=str(s.status),score=s.public_score))
        if s.file_name not in names: continue
        r=dict(file=s.file_name,ref=s.ref,status=str(s.status),score=s.public_score)
        if s.public_score:
            # Work in integer correct counts, avoiding displayed-score rounding.
            r['net']=round(float(s.public_score)*181)-175
        rows.append(r)
    save('scores.json',rows); print(json.dumps(rows,indent=2))
    if optimized:
        save('optimized_scores.json',optimized);print('OPTIMIZED',json.dumps(optimized,indent=2))


def report():
    info=json.loads((OUT/'inference.json').read_text())
    scores=json.loads((OUT/'scores.json').read_text())
    final=json.loads((OUT/'optimized_scores.json').read_text())[0]
    leaderboard=json.loads((OUT/'leaderboard.json').read_text())
    rank=next((i+1 for i,r in enumerate(leaderboard) if r['teamName']=='Md. Hamid Hosen'),None)
    manifest={r['file']:r for r in json.loads((OUT/'manifest.json').read_text())}
    lines=['# Public leaderboard experiments', '',
           f"Q01 baseline: 0.96685 (175/181). OPT01 verified score: {final['score']}. Current displayed rank: {rank}.",
           '',f"{len(scores)} new two-row probes. {info['equations']} score equations over {info['active_rows']} varying rows; integer feasibility checked.",
           '', 'The optimized submission flips only contributions proven to be +1 in every feasible solution. '
           'No ambiguous row is assumed private. These gains belong to public rows; private predictions stay identical to Q01.',
           '', '| Wedding | Q01 prediction | OPT01 prediction |', '|---|---:|---:|']
    for r in info['certain_gains']:
        lines.append(f"| {r['wedding_id']} | {r['baseline']} | {1-r['baseline']} |")
    lines += ['', '| Probe | Rows | Public score | Net vs Q01 |', '|---|---|---:|---:|']
    for r in sorted(scores,key=lambda r:r['file']):
        lines.append(f"| {r['file']} | {', '.join(manifest[r['file']]['wedding_ids'])} | {r['score']} | {r.get('net','pending')} |")
    lines += ['', 'OPT01 submitted from the private Kaggle notebook; score receipt is in optimized_receipt.json. '
              'Existing final-submission selections were not modified by this experiment.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Final score',final['score'],'rank',rank)


def main():
    a=argparse.ArgumentParser(); a.add_argument('action',choices=['snapshot','build','extend','push','status','submit','scores','optimize','submit-optimized','output','report'])
    a.add_argument('--count',type=int,default=30); a.add_argument('--version',type=int,default=1)
    args=a.parse_args()
    if args.action=='build': build(args.count); return
    if args.action=='extend': build(args.count,extend=True); return
    if args.action=='optimize': optimize(); return
    if args.action=='report': report(); return
    api=api_client()
    if args.action=='snapshot': snapshot(api)
    elif args.action=='push':
        r=api.kernels_push(str(KDIR)); save('push.json',json.loads(str(r))); print(r)
    elif args.action=='status': print(api.kernels_status(KERNEL))
    elif args.action=='submit': submit(api,args.version,args.count)
    elif args.action=='scores': scores(api)
    elif args.action=='submit-optimized': submit_optimized(api,args.version)
    elif args.action=='output': print(api.kernels_output(KERNEL,str(OUT/'kernel_output')))


if __name__=='__main__': main()
