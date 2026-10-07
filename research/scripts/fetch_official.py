"""Fetch official competition pages, metadata, topics, messages and leaderboard via the Kaggle Python API."""
import json
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "the-great-kacchi-aloo-mystery"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()


def to_dict(obj):
    for attr in ("to_dict", "to_json"):
        if hasattr(obj, attr):
            try:
                r = getattr(obj, attr)()
                return json.loads(r) if isinstance(r, str) else r
            except Exception:
                pass
    return {k: str(v) for k, v in vars(obj).items()} if hasattr(obj, "__dict__") else str(obj)


# Pages
try:
    pages = api.competition_list_pages(COMP)
    pd = [to_dict(p) for p in pages]
    (out / "pages.json").write_text(json.dumps(pd, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for p in pd:
        name = p.get("name") or p.get("pageName") or "page"
        content = p.get("content") or ""
        (out / f"page_{name}.md").write_text(content, encoding="utf-8")
        print("PAGE", name, len(content))
except Exception as e:
    print("pages error", repr(e))

# Competition metadata
try:
    comps = api.competitions_list(search="kacchi")
    md = [to_dict(c) for c in comps]
    (out / "competition_meta.json").write_text(json.dumps(md, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("META ok", len(md))
except Exception as e:
    print("meta error", repr(e))

# Topics + messages
try:
    allt = []
    for sort in (None,):
        resp = api.competition_list_topics(COMP)
        topics = getattr(resp, "topics", []) or []
        for t in topics:
            td = to_dict(t)
            allt.append(td)
    (out / "topics.json").write_text(json.dumps(allt, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("TOPICS", len(allt), "total_count", getattr(resp, "total_count", None))
    for td in allt:
        tid = td.get("id") or td.get("topicId")
        try:
            m = api.competition_list_topic_messages(COMP, int(tid), page_size=-1)
            (out / f"topic_{tid}.json").write_text(json.dumps(to_dict(m), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            print("MSG", tid, "ok")
        except Exception as e:
            print("msg error", tid, repr(e))
except Exception as e:
    print("topics error", repr(e))

# Full leaderboard
try:
    api.competition_leaderboard_download(COMP, str(out))
    print("LB downloaded")
except Exception as e:
    print("lb error", repr(e))
