"""Step 1: 建立实体词典（LLM 定义 + 原文统计）。"""
from __future__ import annotations

import collections
from common import DICT, OUT, load_json, load_source, gold_mentions, save_json


def main():
    ed = load_json(DICT)
    text = load_source()
    mentions = gold_mentions(text, ed)
    counts = collections.Counter(m["id"] for m in mentions)

    lines = [
        "# 实体词典构建结果",
        "",
        "## 1. 数据来源与方法",
        "",
        "- 原文：`data/source.txt`，由用户提供的三张图片逐字转录，共 3 部概要。",
        "- 词典：由大模型阅读原文后定义实体及别名，再用程序在原文中精确定位提及。",
        f"- 原文字符数：**{len(text)}**；词典实体数：**{len(ed['entities'])}**；"
        f"定位到的实体提及：**{len(mentions)}**。",
        "",
        "## 2. 实体类型体系",
        "",
        "| 类型 | 含义 |",
        "| --- | --- |",
    ]
    for k, v in ed["entity_types"].items():
        lines.append(f"| `{k}` | {v} |")

    lines += ["", "## 3. 实体词典", "", "| ID | 实体 | 类型 | 别名 | 提及次数 | 说明 |", "| --- | --- | --- | --- | ---: | --- |"]
    for e in ed["entities"]:
        lines.append(
            f"| {e['id']} | **{e['name']}** | {e['type']} | {'、'.join(e.get('aliases', [])) or '—'} | "
            f"{counts.get(e['id'], 0)} | {e.get('note', '')} |"
        )

    lines += ["", "## 4. 观察", "",
              "- 人物实体高度集中：孙少平、孙少安、田晓霞、田福堂、田润叶、郝红梅构成 relations 的主干。",
              "- 「少平/少安/晓霞/润叶/润生」等简称需要别名归并，否则会显著低估实体频次。",
              "- 时间、政策运动（责任制、农业学大寨、十一届三中全会）作为事件类实体保留，便于后续构图。",
              "- `MISC` 中的「肺癌」「砖窑」「冒尖户」属于领域概念，是否入库取决于下游任务。",
              ""]

    (OUT / "entity_dictionary.md").write_text("\n".join(lines), encoding="utf-8")
    save_json({"entities": ed["entities"], "mentions": mentions, "counts": dict(counts)}, OUT / "entities_dict_built.json")
    print(f"step1 done: {len(ed['entities'])} entities, {len(mentions)} mentions")


if __name__ == "__main__":
    main()
