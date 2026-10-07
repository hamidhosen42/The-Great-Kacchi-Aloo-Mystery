"""Render an .ipynb (cells + text outputs) to plain text for reading."""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

nb = json.load(open(sys.argv[1], encoding="utf-8"))
for i, c in enumerate(nb.get("cells", [])):
    src = "".join(c.get("source", []))
    print(f"\n######## CELL {i} [{c.get('cell_type')}] ########")
    print(src)
    for o in c.get("outputs", []) or []:
        if "text" in o:
            print("---- OUT(stream) ----")
            print("".join(o["text"])[:6000])
        elif "data" in o:
            d = o["data"]
            if "text/plain" in d:
                print("---- OUT(data) ----")
                print("".join(d["text/plain"])[:6000])
            if "image/png" in d:
                print("---- OUT(image) ----")
        elif o.get("output_type") == "error":
            print("---- ERROR ----", o.get("ename"), o.get("evalue"))
