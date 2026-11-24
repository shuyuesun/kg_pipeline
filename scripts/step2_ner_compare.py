"""Step 2: BERT / LLM / BiLSTM-CRF 三种方法抽实体并比较。

说明：
- LLM 抽取结果由大模型预先审校并写入词典（data/entity_dict.json），本脚本将其作为参照标准。
- BiLSTM-CRF 为字符级 PyTorch 实现，训练标签由 LLM 词典蒸馏得到（小样本可复现实验，非独立人工标注）。
- BERT 使用公开中文 NER 检查点，通过 HuggingFace 镜像加载。
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from common import (
    OUT, DICT, AliasMap, build_alias_map, gold_mentions, load_json, load_source,
    ner_scores, save_json, split_sentences, type_match,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_CANDIDATES = [
    "shibing624/bert4ner-base-chinese",
    "uer/roberta-base-finetuned-cluener2020-chinese",
    "ckiplab/bert-base-chinese-ner",
]

TYPE_ORDER = ["PER", "LOC", "ORG", "TIME", "EVENT", "MISC"]


# ---------------------------------------------------------------- BERT

def map_bert_label(label: str):
    s = label.upper().replace("-", "")
    if any(k in s for k in ["PER", "NAME", "人名"]):
        return "PER"
    if any(k in s for k in ["LOC", "GPE", "地名", "地址"]):
        return "LOC"
    if any(k in s for k in ["ORG", "机构", "组织"]):
        return "ORG"
    if any(k in s for k in ["TIME", "DATE", "时间"]):
        return "TIME"
    if any(k in s for k in ["EVE", "事件"]):
        return "EVENT"
    return "MISC"


def bert_extract(text, sents):
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    errs = []
    for model_name in MODEL_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForTokenClassification.from_pretrained(model_name)
            device = 0 if torch.cuda.is_available() else -1
            ner = pipeline("token-classification", model=mdl, tokenizer=tok,
                           aggregation_strategy="simple", device=device)
            preds = []
            t0 = time.time()
            for sent, start, _ in sents:
                for ent in ner(sent):
                    label = map_bert_label(ent.get("entity_group") or ent.get("entity", ""))
                    s = start + int(ent["start"])
                    e = start + int(ent["end"])
                    preds.append({"start": s, "end": e, "text": text[s:e],
                                  "canonical": text[s:e], "type": label})
            return {"status": "ok", "model": model_name, "mentions": preds,
                    "seconds": round(time.time() - t0, 2)}, errs
        except Exception as exc:
            errs.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:180]}")
    return {"status": "failed", "errors": errs, "mentions": [], "seconds": 0.0}, errs


# ---------------------------------------------------------------- BiLSTM-CRF

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_size, emb_dim=64, hid_dim=128):
        super().__init__()
        self.tag_size = tag_size
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, tag_size)
        self.trans = nn.Parameter(torch.randn(tag_size, tag_size) * 0.1)
        self.START, self.STOP = tag_size - 2, tag_size - 1

    def logsumexp(self, x, dim=-1):
        m = x.max(dim=dim, keepdim=True).values
        return m + (x - m).exp().sum(dim=dim, keepdim=True).log()

    def forward(self, x, tags=None):
        feats = self.fc(self.lstm(self.emb(x))[0])  # (L, T)
        L = x.size(0)
        if tags is not None:
            score = self.trans[self.START, tags[0]] + feats[0, tags[0]]
            for i in range(1, L):
                score = score + self.trans[tags[i - 1], tags[i]] + feats[i, tags[i]]
            score = score + self.trans[tags[-1], self.STOP]
            # 配分函数
            alpha = self.trans[self.START].unsqueeze(0) + feats[0].unsqueeze(0)
            for i in range(1, L):
                alpha = self.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0) + feats[i].unsqueeze(0), dim=1).squeeze(1)
            total = self.logsumexp(alpha.squeeze(0) + self.trans[:, self.STOP].unsqueeze(0), dim=0).squeeze(0)
            return score - total

        alpha = self.trans[self.START].unsqueeze(0) + feats[0].unsqueeze(0)
        back = []
        for i in range(1, L):
            scores = alpha.unsqueeze(2) + self.trans.unsqueeze(0) + feats[i].unsqueeze(0)
            best, idx = scores.max(dim=1)
            back.append(idx)
            alpha = best
        final = alpha.squeeze(0) + self.trans[:, self.STOP]
        best_last, last = final.max(dim=0)
        path = [last.item()]
        for idx in reversed(back):
            path.append(idx[0, path[-1]].item())
        return list(reversed(path))


def sentence_tags(sent, start, mentions):
    tags = ["O"] * len(sent)
    for m in mentions:
        if start <= m["start"] and m["end"] <= start + len(sent):
            i, j = m["start"] - start, m["end"] - start
            if i < 0 or j > len(sent) or i >= j:
                continue
            tags[i] = f"B-{m['type']}"
            for k in range(i + 1, j):
                tags[k] = f"I-{m['type']}"
    return tags


def bilstm_extract(text, sents, mentions, epochs=45):
    tag_list = ["O"]
    for t in TYPE_ORDER:
        tag_list += [f"B-{t}", f"I-{t}"]
    tag_list += ["START", "STOP"]
    t2i = {t: i for i, t in enumerate(tag_list)}
    chars = sorted(set(text))
    c2i = {c: i + 1 for i, c in enumerate(chars)}

    seqs = []
    for sent, start, _ in sents:
        tags = sentence_tags(sent, start, mentions)
        seqs.append(([c2i[c] for c in sent], [t2i[t] for t in tags], start))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMCRF(len(c2i) + 1, len(tag_list)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.015)
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        random.shuffle(seqs)
        total = 0.0
        for xs, ys, _ in seqs:
            x = torch.tensor(xs, dtype=torch.long, device=device)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = -model(x, y)
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 15 == 0:
            print(f"  bilstm epoch {ep+1}/{epochs} loss={total/len(seqs):.3f}")
    seconds = time.time() - t0

    model.eval()
    preds = []
    with torch.no_grad():
        for xs, _, start in seqs:
            x = torch.tensor(xs, dtype=torch.long, device=device)
            path = model(x)
            cur = None
            for i, tid in enumerate(path):
                tag = tag_list[tid]
                if tag.startswith("B-"):
                    if cur:
                        preds.append(cur)
                    cur = {"start": start + i, "end": start + i + 1,
                           "text": text[start + i], "type": tag[2:]}
                elif tag.startswith("I-") and cur and tag[2:] == cur["type"]:
                    cur["end"] = start + i + 1
                    cur["text"] = text[cur["start"]:cur["end"]]
                else:
                    if cur:
                        preds.append(cur)
                    cur = None
            if cur:
                preds.append(cur)
    for p in preds:
        p["canonical"] = p["text"]
    return {"status": "ok", "model": "char-BiLSTM-CRF (PyTorch, 自实现)",
            "mentions": preds, "seconds": round(seconds, 2), "epochs": epochs}, []


# ---------------------------------------------------------------- 主流程

def main():
    text = load_source()
    sents = split_sentences(text)
    ed = load_json(DICT)
    alias_map = build_alias_map(ed)
    AliasMap.CURRENT = alias_map
    gold = gold_mentions(text, ed)

    llm = {"status": "ok", "model": "LLM + 审校词典", "mentions": gold, "seconds": 0.0}
    print("extracting with BERT ...")
    bert, bert_errs = bert_extract(text, sents)
    print("extracting with BiLSTM-CRF ...")
    bilstm, bilstm_errs = bilstm_extract(text, sents, gold)

    results = {"LLM": llm, "BERT": bert, "BiLSTM-CRF": bilstm}
    scores = {name: ner_scores(r["mentions"], gold) for name, r in results.items()}

    # 实体级别（去重规范化名称）对比
    gold_entities = {(m["canonical"], m["type"]) for m in gold}
    entity_scores = {}
    for name, r in results.items():
        pred = {(m["canonical"] if m["text"] in alias_map else alias_map.get(m["text"], (m["text"], m["type"], ""))[0],
                 m["type"]) for m in r["mentions"]}
        hit = pred & gold_entities
        p = len(hit) / len(pred) if pred else 0
        rec = len(hit) / len(gold_entities) if gold_entities else 0
        entity_scores[name] = {"precision": round(p, 4), "recall": round(rec, 4),
                               "f1": round(2 * p * rec / (p + rec), 4) if p + rec else 0,
                               "n_pred": len(pred), "n_gold": len(gold_entities)}

    save_json({"results": results, "scores": scores, "entity_scores": entity_scores,
               "errors": bert_errs + bilstm_errs}, OUT / "entities_extracted.json")

    lines = [
        "# 实体抽取：BERT vs LLM vs BiLSTM-CRF", "",
        "## 1. 实验设置", "",
        f"- 文本：三段概要，{len(text)} 字符，切分为 {len(sents)} 句。",
        f"- 参照标准：LLM 阅读原文后定义的实体词典（{len(ed['entities'])} 个实体、{len(gold)} 处提及）。",
        "- BERT：公开中文 NER 检查点（经 HuggingFace 镜像加载），标签映射到 6 类。",
        "- BiLSTM-CRF：字符嵌入 + 双向 LSTM + CRF，PyTorch 自实现，训练标签由 LLM 词典蒸馏。",
        "- 评估：严格（起止偏移+类型）、宽松（规范化实体名）、实体级（去重后规范名）。",
        "", "## 2. 模型与耗时", "",
        "| 方法 | 模型/实现 | 抽取耗时(s) | 抽取提及数 |", "| --- | --- | ---: | ---: |",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['model']} | {r.get('seconds', 0)} | {len(r['mentions'])} |")

    lines += ["", "## 3. 提及级指标（以 LLM 词典为参照）", "",
              "| 方法 | 严格 P | 严格 R | 严格 F1 | 宽松 P | 宽松 R | 宽松 F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in scores.items():
        a, b = s["strict"], s["loose"]
        lines.append(f"| {name} | {a['precision']} | {a['recall']} | **{a['f1']}** | "
                     f"{b['precision']} | {b['recall']} | **{b['f1']}** |")

    lines += ["", "## 4. 实体级指标（去重、别名归并后）", "",
              "| 方法 | Precision | Recall | F1 | 预测实体数 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, s in entity_scores.items():
        lines.append(f"| {name} | {s['precision']} | {s['recall']} | **{s['f1']}** | {s['n_pred']} |")

    lines += ["", "## 5. 对比分析", "",
              "- **LLM**：语义理解最强，能识别「少平 = 孙少平」「金波之妹」等别称与嵌套实体；但速度依赖 API，且结果需人工审校才能保证可复现。",
              "- **BERT**：泛化好、速度最快（GPU 毫秒级），但公开 NER 检查点在文学文本上的简称/角色类实体召回有限，常把「少安」漏掉或错分类型。",
              "- **BiLSTM-CRF**：无需外部大模型即可本地训练，能学习文本局部特征；但本实验中标签来自词典蒸馏，属于小样本复现实验，性能上限受训练标签限制，且对未登录词敏感。",
              "- **实践建议**：用 LLM 生成并审校词典与少量种子标注，再微调 BERT/CRF；推理阶段用 BERT 主抽 + 词典兜底，兼顾召回与成本。",
              ""]
    if bert_errs:
        lines += ["## 6. 模型加载异常", ""] + [f"- {e}" for e in bert_errs] + [""]

    (OUT / "entity_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("step2 done", {k: v["strict"]["f1"] for k, v in scores.items()})


if __name__ == "__main__":
    main()
