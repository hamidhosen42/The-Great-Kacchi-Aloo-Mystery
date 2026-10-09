"""Manage the reviewable theory notebook without consuming submission quota."""
import argparse
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/private/tmp/kacchi-final-delivery-20261009')
OUT.mkdir(exist_ok=True)
KDIR=ROOT/'solution/kaggle_theory_kernel'
KERNEL='hosen42/the-aloo-theory-count-potatoes-per-guest'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['push','status','output','publish','inspect'])
    args=parser.parse_args();api=KaggleApi();api.authenticate()
    if args.action in ['push','publish']:
        if args.action=='publish':
            path=KDIR/'kernel-metadata.json';meta=json.loads(path.read_text());meta['is_private']=False
            path.write_text(json.dumps(meta,indent=2)+'\n')
        result=api.kernels_push(str(KDIR))
        (OUT/('theory_publish.json' if args.action=='publish' else 'theory_push.json')).write_text(str(result)+'\n')
        print(result)
    elif args.action=='status':print(api.kernels_status(KERNEL))
    elif args.action=='output':print(api.kernels_output(KERNEL,str(OUT/'kaggle_theory_output')))
    elif args.action=='inspect':print(api.kernels_pull(KERNEL,str(OUT/'published'),metadata=True,quiet=True))


if __name__=='__main__':main()
