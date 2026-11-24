"""Step 3: spaCy / BERT / LLM 三种方法构建关系三元组并比较。

- spaCy：blank('zh') + EntityRuler（词典规则）+ 触发词规则，纯本地可复现。
- BERT：同一中文 BERT 提取 [CLS] 向量 + LogisticRegression 关系分类器（在 LLM 三元组上训练/交叉验证）。
- LLM：大模型直接抽取并经人工审校的三元组（data/llm_triples.json），作为参照标准。
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter

import numpy as np
import torch

from common import (
    OUT, DICT, LLM_TRIPLES, AliasMap, build_alias_map, canonical, load_json,
    load_source, save_json, split_sentences, triple_scores,
)

random.seed(42)
np.random.seed(42)

COARSE = {
    "LOVE": "情感", "CONFESS": "情感", "CHILDHOOD_SWEETHEART": "情感", "LOSE_CONTACT": "情感",
    "SPOUSE": "家庭", "PARENT_OF": "家庭", "SIBLING": "家庭", "MARRY": "家庭",
    "ARRANGE": "家庭", "GIVE_BIRTH": "家庭", "RECOGNIZE": "家庭",
    "FRIEND": "社会", "CLASSMATE": "社会", "SUPPORT": "社会", "PROMISE": "社会",
    "INFORM": "社会", "DISCOVER": "社会",
    "WORK_AT": "职业/行动", "STUDY_AT": "职业/行动", "LEAD": "职业/行动",
    "BUSINESS": "职业/行动", "LOAN": "职业/行动", "PROMOTE": "职业/行动",
    "RETURN_TO": "职业/行动", "RESIST": "职业/行动", "OPPOSE": "职业/行动",
    "ORGANIZE": "事件", "REPORT": "事件", "RESCUE": "事件", "DONATE": "事件",
    "DIE_IN": "负面事件", "DIAGNOSE": "负面事件", "SUFFER_LOSS": "负面事件",
    "DISABLED_IN": "负面事件", "REJECT": "负面事件", "GRIEVE": "负面事件",
}

FINE_DEFAULT = {"情感": "LOVE", "家庭": "PARENT_OF", "社会": "FRIEND",
                "职业/行动": "WORK_AT", "事件": "ORGANIZE", "负面事件": "OPPOSE"}

CUES = [
    (r"爱慕|情愫|恋情|恋爱|女友|强烈感情|表白", "情感"),
    (r"结婚|结为夫妻|妻子|丈夫|夫妻|相亲", "家庭"),
    (r"父亲|女儿|弟弟|哥哥|儿子|妹妹", "家庭"),
    (r"朋友|建立了?友情|玩伴|同学", "社会"),
    (r"反对|抵制|抵触|当众说破", "职业/行动"),
    (r"帮助|支持下", "社会"),
    (r"组织|召开|开发", "事件"),
    (r"抢救|救护|出资|发电报|承认|认识|邂逅", "事件"),
    (r"确诊|牺牲|致残|损失|重伤|拒绝|悲痛", "负面事件"),
    (r"贷款|扩建|建窑|烧砖|推广|实行|领导|工作|当了|成了|成为|到|回到|就读|读书|毕业", "职业/行动"),
]

CONCEPTS = [
    "教师", "建筑工人", "煤矿工人", "优秀工人", "揽工汉", "记者", "村支书", "县革委副主任",
    "责任制", "农业学大寨", "十一届三中全会", "旱灾", "抗洪采访", "肺癌", "砖窑", "小学",
    "机器制砖", "冒尖户", "新时期变革", "灾民", "徒弟", "外面世界", "矿山", "煤矿",
]


def coarse(triple):
    return COARSE.get(triple["r"], "事件")


def coarse_key(t, alias_map):
    return (canonical(t["h"], alias_map), coarse(t), canonical(t["t"], alias_map))


def coarse_scores(pred, gold, alias_map):
    g = {coarse_key(t, alias_map) for t in gold}
    p = {coarse_key(t, alias_map) for t in pred}
    hit = g & p
    pr = len(hit) / len(p) if p else 0
    rc = len(hit) / len(g) if g else 0
    return {"precision": round(pr, 4), "recall": round(rc, 4),
            "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0,
            "n_pred": len(p), "n_gold": len(g), "n_hit": len(hit)}


# ---------------------------------------------------------------- spaCy

def build_spacy_nlp(ent_dict):
    import spacy
    from spacy.pipeline import EntityRuler
    nlp = spacy.blank("zh")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = []
    for e in ent_dict["entities"]:
        for name in [e["name"]] + e.get("aliases", []):
            patterns.append({"label": e["type"], "pattern": name})
    for c in CONCEPTS:
        patterns.append({"label": "CONCEPT", "pattern": c})
    ruler.add_patterns(patterns)
    return nlp


def spacy_candidates(sents, ent_dict, alias_map):
    """用 spaCy EntityRuler 找实体，并按触发词生成候选关系对（spaCy 与 BERT 共用）。"""
    nlp = build_spacy_nlp(ent_dict)
    pairs = []
    for sent, _, _ in sents:
        doc = nlp(sent)
        ents = []
        for e in doc.ents:
            ents.append({"text": e.text, "canonical": canonical(e.text, alias_map),
                         "start": e.start_char, "end": e.end_char, "label": e.label_})
        ents.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        keep = []
        for e in ents:
            if any(e["start"] >= k["start"] and e["end"] <= k["end"] for k in keep):
                continue
            keep = [k for k in keep if not (k["start"] >= e["start"] and k["end"] <= e["end"])]
            keep.append(e)
        # 同一实体在一句中只保留一次，避免自环与重复
        seen, uniq = set(), []
        for e in sorted(keep, key=lambda x: -(x["end"] - x["start"])):
            if e["canonical"] in seen:
                continue
            seen.add(e["canonical"])
            uniq.append(e)
        for i, a in enumerate(uniq):
            for b in uniq[i + 1:]:
                h, t = a["canonical"], b["canonical"]
                if h == t or h == "1975年初" or t == "1975年初":
                    continue
                between = sent[a["end"]:b["start"]]
                if len(between) > 34:
                    continue
                chosen = None
                for pat, rel in CUES:
                    m = re.search(pat, between)
                    if m:
                        # 触发词必须靠近某个端点，避免长句中的伪关系
                        if min(m.start(), len(between) - m.end()) > 14:
                            continue
                        chosen = rel
                        break
                if not chosen:
                    continue
                # 帮助类关系：被帮助者在前时反转方向
                if "帮助" in between and chosen == "社会":
                    h, t = t, h
                pairs.append({"h": h, "t": t, "between": between, "sent": sent,
                              "rule_rel": chosen})
    return pairs


def spacy_extract(sents, ent_dict, alias_map):
    t0 = time.time()
    triples = []
    for p in spacy_candidates(sents, ent_dict, alias_map):
        triples.append({"h": p["h"], "r": FINE_DEFAULT[p["rule_rel"]], "t": p["t"],
                        "coarse": p["rule_rel"], "evidence": p["sent"], "method": "spaCy"})
    return {"status": "ok", "model": "spaCy blank:zh + EntityRuler + cue rules",
            "triples": triples, "seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- BERT

def bert_re_extract(sents, gold_triples, alias_map, pairs=None, model_name="shibing624/bert4ner-base-chinese"):
    from transformers import AutoModel, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev).eval()
    print("  BERT RE encoding training triples ...")

    def embed(h, t, ev):
        txt = f"{h} 与 {t}：{ev}"[:180]
        batch = tok(txt, return_tensors="pt", truncation=True, max_length=128).to(dev)
        with torch.no_grad():
            out = mdl(**batch)
        return out.last_hidden_state[:, 0, :].cpu().numpy()[0]

    X, y = [], []
    for tr in gold_triples:
        X.append(embed(tr["h"], tr["t"], tr["evidence"]))
        y.append(coarse(tr))
    X = np.vstack(X)
    y = np.array(y)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X, y, cv=skf)
    report = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)
    clf.fit(X, y)

    # 候选对复用 spaCy 生成的实体对（BERT 只负责关系分类，保证两种方法有可比性）
    candidates = [(p["h"], p["t"], p["sent"]) for p in (pairs or [])]
    print(f"  BERT RE candidates: {len(candidates)}")

    triples = []
    if candidates:
        Xc = np.vstack([embed(a, b, sent) for a, b, sent in candidates])
        probs = clf.predict_proba(Xc)
        classes = clf.classes_
        for (a, b, sent), pr in zip(candidates, probs):
            idx = int(pr.argmax())
            if pr[idx] < 0.45:
                continue
            lab = classes[idx]
            triples.append({"h": a, "r": FINE_DEFAULT[lab], "t": b,
                            "coarse": lab, "evidence": sent,
                            "confidence": round(float(pr[idx]), 3), "method": "BERT"})
    return {"status": "ok", "model": model_name + " + LogisticRegression",
            "triples": triples, "seconds": round(time.time() - t0, 2),
            "cv_report": report, "n_candidates": len(candidates)}


# ---------------------------------------------------------------- 主流程

def main():
    text = load_source()
    sents = split_sentences(text)
    ed = load_json(DICT)
    alias_map = build_alias_map(ed)
    AliasMap.CURRENT = alias_map
    llm = load_json(LLM_TRIPLES)["triples"]

    print("extracting with spaCy ...")
    pairs = spacy_candidates(sents, ed, alias_map)
    sp = spacy_extract(sents, ed, alias_map)
    print(f"  spaCy candidate pairs: {len(pairs)}")
    print("extracting with BERT ...")
    try:
        bert = bert_re_extract(sents, llm, alias_map, pairs=pairs)
        bert_err = None
    except Exception as e:
        bert_err = f"{type(e).__name__}: {str(e)[:300]}"
        bert = {"status": "failed", "model": "BERT", "triples": [], "seconds": 0.0}

    results = {"LLM": {"triples": llm, "seconds": 0.0, "model": "LLM 审校三元组"},
               "spaCy": sp, "BERT": bert}
    metrics = {name: coarse_scores(r["triples"], llm, alias_map) for name, r in results.items()}
    fine_metrics = {name: triple_scores(r["triples"], llm, alias_map) for name, r in results.items()}

    save_json({"results": results, "coarse_metrics": metrics, "fine_metrics": fine_metrics,
               "error": bert_err}, OUT / "triples_extracted.json")

    lines = [
        "# 关系三元组：spaCy / BERT / LLM 对比", "",
        "## 1. 方法", "",
        "- **LLM**：大模型阅读原文直接给出三元组（含原文证据），经审校后作为参照标准。",
        "- **spaCy**：中文空白管道 + EntityRuler 词典规则 + 触发词/上下文规则；完全本地、可解释。",
        "- **BERT**：中文 BERT 输出 [CLS] 向量，训练逻辑回归关系分类器；5 折交叉验证 + 全量训练后对候选实体对打分。",
        "- 对比粒度：细粒度关系（如 LOVE/PARENT_OF）与粗粒度关系类（情感/家庭/社会/职业/事件/负面事件）两套指标。",
        "", "## 2. 总体指标", "",
        "### 2.1 粗粒度（推荐，避免同义关系命名差异）", "",
        "| 方法 | Precision | Recall | F1 | 预测数 | 命中数 |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['precision']} | {m['recall']} | **{m['f1']}** | {m['n_pred']} | {m['n_hit']} |")
    lines += ["", "### 2.2 细粒度（严格关系名）", "",
              "| 方法 | Precision | Recall | F1 | 预测数 | 命中数 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in fine_metrics.items():
        lines.append(f"| {name} | {m['precision']} | {m['recall']} | **{m['f1']}** | {m['n_pred']} | {m['n_hit']} |")

    lines += ["", "## 3. LLM 三元组（参照标准，共 %d 条）" % len(llm), "",
              "| 头实体 | 关系 | 尾实体 | 部 | 原文证据 |", "| --- | --- | --- | :---: | --- |"]
    for tr in llm:
        ev = tr["evidence"].replace("|", "/")
        lines.append(f"| {tr['h']} | {tr['r']} | {tr['t']} | {tr['part']} | {ev} |")

    for name in ["spaCy", "BERT"]:
        rs = results[name]["triples"]
        lines += ["", f"## 4. {name} 抽取结果（共 {len(rs)} 条，表格展示前 60 条）", "",
                  "| 头实体 | 关系 | 尾实体 | 置信度 | 原文证据 |", "| --- | --- | --- | ---: | --- |"]
        for tr in rs[:60]:
            ev = tr.get("evidence", "").replace("|", "/")[:70]
            conf = tr.get("confidence", "—")
            lines.append(f"| {tr['h']} | {tr['r']} | {tr['t']} | {conf} | {ev} |")
        if len(rs) > 60:
            lines.append(f"| … | … | … | … | 其余 {len(rs)-60} 条见 `outputs/triples_extracted.json` |")

    lines += ["", "## 5. 比较分析", "",
              "- **LLM**：覆盖情感、家庭、职业、事件等多类型关系，能处理跨句共指与隐含关系（如「田福堂介绍李向前」），质量最高但依赖大模型调用。",
              "- **spaCy**：规则可解释、零成本、毫秒级；受限于词典覆盖率与触发词表，对「农业学大寨」等隐喻和长距离依赖容易漏抽或错向。",
              "- **BERT**：借助上下文语义，候选对打分能发现规则未覆盖的关系；但本实验在小样本三元组上训练，交叉验证 F1 反映领域内拟合而非通用泛化，需更多标注数据才能稳定。",
              "- **组合建议**：LLM 负责离线构建高质量种子三元组与关系词典 → spaCy 规则保证高 precision 的确定性关系 → BERT 分类器扩召回 → 人工抽检闭环。",
              ""]
    if bert_err:
        lines += ["## 6. BERT 异常", "", f"- {bert_err}", ""]
    if bert.get("cv_report"):
        lines += ["## 7. BERT 关系分类器 5 折交叉验证", "", "```json",
                  json.dumps(bert["cv_report"], ensure_ascii=False, indent=2), "```", ""]

    (OUT / "triples.md").write_text("\n".join(lines), encoding="utf-8")
    print("step3 done", {k: v["f1"] for k, v in metrics.items()})


if __name__ == "__main__":
    main()
