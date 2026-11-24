# 《平凡的世界》三部概要：实体-关系-知识图谱流水线

本项目以用户提供的三张图片（小说《平凡的世界》第一部、第二部概要、第三部文本）为原始语料，完成：

1. 实体词典构建
2. BERT / LLM / BiLSTM-CRF 实体抽取与比较
3. spaCy / BERT / LLM 关系三元组构建与比较
4. Neo4j 与 LLM 知识图谱构建及性能比较
5. 一键化流水线脚本

## 目录结构

```
kg_pipeline/
├── data/
│   ├── source.txt              # 三张图片转录的原文
│   ├── entity_dict.json        # 实体词典（LLM 定义）+ 类型/关系体系
│   └── llm_triples.json        # LLM 抽取的三元组（含原文证据）
├── scripts/
│   ├── common.py               # 公共工具：切句、实体对齐、指标
│   ├── step1_dictionary.py     # 实体词典统计
│   ├── step2_ner_compare.py    # BERT / LLM / BiLSTM-CRF 实体抽取比较
│   ├── step3_re_compare.py     # spaCy / BERT / LLM 三元组比较
│   ├── step4_kg.py             # Neo4j / LLM 图谱与性能比较
│   └── pipeline.py             # 一键运行全部步骤
├── outputs/                    # 运行后生成的所有报告与 JSON
├── requirements.txt
└── README.md
```

## 环境与运行

```bash
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像；海外可省略
python scripts/pipeline.py
```

如需真正写入 Neo4j：

```bash
# 本机需 Java 21 + Neo4j 5.x，配置 dbms.security.auth_enabled=false
neo4j console
python scripts/step4_kg.py
```

Neo4j 不可用时，脚本仍会生成 `outputs/neo4j_import.cypher` 与 `outputs/kg.html`。

## 方法说明与诚实声明

- **LLM 结果**由大模型阅读原文直接给出，并经过人工一致性审校，作为本项目的参照标准（非独立盲标注）。
- **BiLSTM-CRF** 为字符级 PyTorch 自实现；由于缺少外部标注数据，其训练标签由 LLM 词典蒸馏而来，属于小样本可复现实验，不能等同于在人工标注语料上训练的通用 NER 模型。
- **BERT** 使用公开中文检查点 `shibing624/bert4ner-base-chinese`；关系抽取阶段用同一 BERT 的 `[CLS]` 向量训练逻辑回归分类器，并在 LLM 三元组上做 5 折交叉验证。
- **spaCy** 使用 `spacy.blank("zh")` + EntityRuler + 触发词规则；不依赖需额外下载的中文统计模型，便于完全离线复现。
- **性能比较**中 LLM 图谱为内存构图，Neo4j 为数据库持久化；两者定位不同，比较结果应结合「语义生成质量」与「存储查询性能」分别看待。

## 关键结果摘要（本次运行）

- 实体词典：43 个实体、81 处提及；LLM 参照标准下，BERT 实体抽取严格 F1 = 0.7042；BiLSTM-CRF 全量重训为记忆化结果（F1 = 1.0），80/20 句子留出集 F1 = 0.4615。
- 三元组：LLM 参照共 66 个去重粗粒度关系；spaCy 规则 F1 = 0.1871，BERT 分类器 F1 = 0.1538。低分反映词典/标注规模小与严格别名归并，方法差异与改进方向见 `outputs/triples.md`。
- 知识图谱：56 节点、74 条关系。NetworkX 构图约 0.001s，Neo4j 批量写入约 2.3s；小图上 NetworkX 查询更快，但 Neo4j 提供持久化与事务能力。

## 主要输出

| 文件 | 内容 |
| --- | --- |
| `outputs/entity_dictionary.md` | 实体词典与统计 |
| `outputs/entity_comparison.md` | 三种实体抽取方法的指标与分析 |
| `outputs/triples.md` | 三种方法的三元组与对比分析 |
| `outputs/kg_comparison.md` | Neo4j / LLM 图谱构建与查询性能比较 |
| `outputs/kg.html` | 可交互知识图谱可视化 |
| `outputs/*.json` | 上述步骤的结构化中间结果 |
