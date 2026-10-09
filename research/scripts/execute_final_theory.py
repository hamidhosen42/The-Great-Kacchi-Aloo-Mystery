"""Execute every cell and retain a reviewable notebook plus an HTML report."""
import base64
import html
import json
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from markdown_it import MarkdownIt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('KACCHI_FINAL_DELIVERY',str(ROOT/'outputs/finalization')))
OUT.mkdir(exist_ok=True)
os.environ['KACCHI_DATA']=str(ROOT/'data')
os.environ['MPLCONFIGDIR']='/private/tmp/kacchi-final-matplotlib'
os.environ['IPYTHONDIR']='/private/tmp/kacchi-final-ipython'
os.environ['JUPYTER_RUNTIME_DIR']='/private/tmp/kacchi-final-runtime'
os.environ['JUPYTER_PATH']='/private/tmp/kacchi-final-jupyter/share/jupyter'


def main():
    source=ROOT/'solution/best_aloo_theory.ipynb'
    nb=nbformat.read(source,as_version=4)
    def started(cell,cell_index,**kwargs):
        print('Executing cell',cell_index,flush=True)
    if '--render-only' not in sys.argv:
        NotebookClient(nb,timeout=1800,kernel_name='kacchi-final',
                       resources={'metadata':{'path':str(OUT)}},on_cell_start=started).execute()
    nbformat.validate(nb)
    assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    for p in [source,ROOT/'solution/kaggle_theory_kernel/aloo-theory.ipynb',OUT/'best_aloo_theory_executed.ipynb']:
        nbformat.write(nb,p)
    md=MarkdownIt('commonmark',{'html':True})
    body=[]
    for c in nb.cells:
        if c.cell_type=='markdown':body.append(md.render(c.source))
        else:
            body.append('<details><summary>Reproducible Python code</summary><pre>'+html.escape(c.source)+'</pre></details>')
            for o in c.outputs:
                if o.output_type=='stream':body.append('<pre>'+html.escape(o.text)+'</pre>')
                elif hasattr(o,'data'):
                    if 'text/html' in o.data:body.append(o.data['text/html'])
                    elif 'image/png' in o.data:body.append('<img alt="Analysis chart" src="data:image/png;base64,'+o.data['image/png']+'">')
                    elif 'text/plain' in o.data:body.append('<pre>'+html.escape(o.data['text/plain'])+'</pre>')
    page='<!doctype html><meta charset="utf-8"><title>The Aloo Theory</title><style>body{max-width:980px;margin:40px auto;padding:0 20px;font:17px/1.6 system-ui;color:#19221d}img{max-width:100%}pre{white-space:pre-wrap;font-size:13px;background:#f4f5f2;padding:14px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}details{margin:12px 0}a{color:#17633b}</style>'+''.join(body)
    (OUT/'best_aloo_theory.html').write_text(page)
    print('All cells executed; notebook and HTML report saved',flush=True)


if __name__=='__main__':main()
