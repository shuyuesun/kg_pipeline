"""公共工具：路径、文本切分、实体对齐与评估指标。"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SRC = DATA / "source.txt"
DICT = DATA / "entity_dict.json"
LLM_TRIPLES = DATA / "llm_triples.json"

OUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_source():
    return SRC.read_text(encoding="utf-8")


def split_sentences(text: str):
    """按中文标点切句，返回 [(sentence, start, end)]，保留原始偏移。"""
    parts = re.split(r"(?<=[。！？；\n])", text)
    out, pos = [], 0
    for p in parts:
        if not p:
            continue
        s = p.strip()
        if s:
            start = text.find(s, pos)
            if start < 0:
                start = pos
            end = start + len(s)
            out.append((s, start, end))
            pos = end
    return out


# ---------- 实体与别名 ----------

def build_alias_map(ent_dict):
    amap = {}
    for e in ent_dict["entities"]:
        for name in [e["name"]] + e.get("aliases", []):
            amap[name] = (e["name"], e["type"], e["id"])
    return amap


def canonical(text: str, alias_map):
    if text in alias_map:
        return alias_map[text][0]
    # 最长别名包含匹配，例如“金波之妹”优先于“金波”
    cands = [a for a in alias_map if a in text]
    if cands:
        best = max(cands, key=len)
        return alias_map[best][0]
    return text


def gold_mentions(text: str, ent_dict):
    """由 LLM 定义的词典 + 别名，在原文中定位实体提及（金标准）。"""
    found = []
    for e in ent_dict["entities"]:
        for name in [e["name"]] + e.get("aliases", []):
            for m in re.finditer(re.escape(name), text):
                found.append({
                    "start": m.start(), "end": m.end(), "text": name,
                    "canonical": e["name"], "type": e["type"], "id": e["id"],
                })
    # 去重 + 去掉被更长实体包含的短实体（如“金波” vs “金波之妹”）
    found.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    kept = []
    for m in found:
        if any(m["start"] >= k["start"] and m["end"] <= k["end"] for k in kept):
            continue
        kept = [k for k in kept if not (k["start"] >= m["start"] and k["end"] <= m["end"])]
        kept.append(m)
    return sorted(kept, key=lambda x: x["start"])


def type_match(pred_type, gold_type):
    """严格类型匹配；NER 模型标签与自定义类型做宽松映射。"""
    aliases = {
        "PERSON": "PER", "PER": "PER", "人名": "PER",
        "LOCATION": "LOC", "LOC": "LOC", "GPE": "LOC", "地名": "LOC",
        "ORGANIZATION": "ORG", "ORG": "ORG", "机构": "ORG",
        "TIME": "TIME", "DATE": "TIME", "时间": "TIME",
    }
    return aliases.get(pred_type, pred_type) == aliases.get(gold_type, gold_type)


def ner_scores(pred, gold):
    """pred/gold: list of dict(start,end,canonical/type)。返回严格与宽松指标。"""
    def key(m, loose=False):
        if loose:
            return (m["start"], m["end"], canonical(m.get("text", m.get("canonical", "")), AliasMap.CURRENT))
        return (m["start"], m["end"], m.get("type", ""))

    strict_p = sum(1 for p in pred if key(p) in {key(g) for g in gold})
    loose_p = sum(1 for p in pred if key(p, True) in {key(g, True) for g in gold})
    tp_s, tp_l = strict_p, loose_p
    prec_s = tp_s / len(pred) if pred else 0.0
    rec_s = tp_s / len(gold) if gold else 0.0
    prec_l = tp_l / len(pred) if pred else 0.0
    rec_l = tp_l / len(gold) if gold else 0.0
    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0
    return {
        "strict": {"precision": round(prec_s, 4), "recall": round(rec_s, 4), "f1": round(f1(prec_s, rec_s), 4)},
        "loose": {"precision": round(prec_l, 4), "recall": round(rec_l, 4), "f1": round(f1(prec_l, rec_l), 4)},
        "n_pred": len(pred), "n_gold": len(gold),
    }


class AliasMap:
    CURRENT = {}


def triple_key(triple, alias_map):
    h = canonical(triple["h"], alias_map)
    t = canonical(triple["t"], alias_map)
    return (h, triple["r"], t)


def triple_scores(pred, gold, alias_map):
    gset = {triple_key(t, alias_map) for t in gold}
    pset = {triple_key(t, alias_map) for t in pred}
    hit = pset & gset
    p = len(hit) / len(pset) if pset else 0.0
    r = len(hit) / len(gset) if gset else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "n_pred": len(pset), "n_gold": len(gset), "n_hit": len(hit)}
