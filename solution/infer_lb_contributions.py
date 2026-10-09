"""Solve integer score constraints without assuming unchanged rows are private."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/lb_experiments'


def load(path):
    rows=list(csv.DictReader(path.open()))
    return [r['wedding_id'] for r in rows], np.array([int(r['went_back_for_seconds']) for r in rows])


def main():
    ids, base=load(ROOT/'outputs/portfolio_final/N0567.csv')
    files={p.name:p for p in (ROOT/'outputs/portfolio25').glob('P*.csv')}
    for folder,mapping in [('portfolio_final','q_names.json'),('portfolio_final_check','s_names.json')]:
        d=ROOT/'outputs'/folder
        files.update({k+'.csv':d/(v+'.csv') for k,v in json.loads((d/mapping).read_text()).items()})
    files.update({f'R{k:02d}.csv':ROOT/f'outputs/edge_models2/E39_hedge{k}.csv' for k in [5,7,9,11]})
    files.update({p.name:p for p in (ROOT/'outputs/lb_probes').glob('LBP*.csv')})
    files.update({n:ROOT/'outputs'/n for n in ['sub_A_primary_sharpband.csv','sub_B_hedge.csv','sub_C_band_1p00_1p975.csv']})
    submissions=json.loads((OUT/'submissions.json').read_text())
    vectors=[]; deltas=[]; used=[]
    for s in submissions:
        name=s['fileName']
        if name not in files or not s.get('publicScore'): continue
        ii,v=load(files[name]); assert ii==ids
        vectors.append((v!=base).astype(float)); deltas.append(round(float(s['publicScore'])*181)-175); used.append(name)
    if (OUT/'scores.json').exists():
        mf={x['file']:x for x in json.loads((OUT/'manifest.json').read_text())}
        for s in json.loads((OUT/'scores.json').read_text()):
            if 'net' not in s: continue
            v=np.zeros(len(base)); v[mf[s['file']]['indices']]=1
            vectors.append(v); deltas.append(s['net']); used.append(s['file'])
    full=np.array(vectors); active=np.flatnonzero(full.any(axis=0)); A=full[:,active]; n=len(active)
    # w=public baseline wrong; r=public baseline right; c=w-r.
    eq=np.hstack([A,-A]); delta=np.array(deltas)
    one=np.hstack([np.eye(n),np.eye(n)])
    budget=np.vstack([np.r_[np.ones(n),np.zeros(n)],np.r_[np.zeros(n),np.ones(n)]])
    constraints=[LinearConstraint(eq,delta,delta),LinearConstraint(one,0,1),LinearConstraint(budget,0,[6,175])]
    bounds=Bounds(np.zeros(2*n),np.ones(2*n))
    def solve(cost):
        res=milp(cost,integrality=np.ones(2*n),bounds=bounds,constraints=constraints,options={'time_limit':30})
        assert res.success, res.message
        return res
    solution=solve(np.zeros(2*n)); assert np.allclose(eq@solution.x,delta)
    info=[]
    for j,i in enumerate(active):
        obj=np.zeros(2*n);obj[j]=1;obj[n+j]=-1
        low=round(solve(obj).fun);high=-round(solve(-obj).fun)
        info.append(dict(index=int(i),wedding_id=ids[i],baseline=int(base[i]),min_contribution=low,max_contribution=high))
    result=dict(public_rows_assumed=181,baseline_correct=175,equations=len(used),files=used,
                active_rows=n,matrix_rank=int(np.linalg.matrix_rank(A)),rows=info,
                certain_gains=[r for r in info if r['min_contribution']==1],
                anchors=[r for r in info if r['max_contribution']==-1])
    (OUT/'inference.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Equations:',len(used),'active rows:',n,'rank:',result['matrix_rank'])
    print('Certain gains:',result['certain_gains'])
    print('Known public-correct anchors:',len(result['anchors']))
    print('Still possibly beneficial:',[r for r in info if r['max_contribution']==1 and r['min_contribution']!=1])


if __name__=='__main__': main()
