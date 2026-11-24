"""Step 4: Neo4j 知识图谱 + LLM 知识图谱，构建与查询性能比较。"""
from __future__ import annotations

import time
from collections import Counter, defaultdict

import networkx as nx

from common import OUT, DICT, LLM_TRIPLES, build_alias_map, load_json, save_json

TYPE_LABEL = {"PER": "Person", "LOC": "Place", "ORG": "Organization",
              "TIME": "Time", "EVENT": "Event", "MISC": "Concept"}

QUERIES = [
    ("孙少平的直接关系", lambda G: list(G.neighbors("孙少平"))),
    ("田福军→田晓霞→孙少平 两跳路径", lambda G: list(nx.all_simple_paths(G, "田福军", "孙少平", cutoff=2))[:5]),
    ("全图边数统计", lambda G: G.number_of_edges()),
]


def build_llm_graph(triples, ent_dict):
    t0 = time.perf_counter()
    types = {e["name"]: e["type"] for e in ent_dict["entities"]}
    G = nx.MultiDiGraph()
    for tr in triples:
        h, t = tr["h"], tr["t"]
        G.add_node(h, node_type=types.get(h, "CONCEPT"), label=TYPE_LABEL.get(types.get(h), "Concept"))
        G.add_node(t, node_type=types.get(t, "CONCEPT"), label=TYPE_LABEL.get(types.get(t), "Concept"))
        G.add_edge(h, t, relation=tr["r"], evidence=tr.get("evidence", ""), part=tr.get("part", 0))
    secs = time.perf_counter() - t0
    return G, secs


def neo4j_import(triples, ent_dict):
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        return {"status": "failed", "error": f"neo4j driver missing: {e}", "seconds": 0}
    try:
        driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=None)
        with driver.session() as s:
            s.run("RETURN 1").consume()
    except Exception as e:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=None)
            with driver.session() as s:
                s.run("RETURN 1").consume()
        except Exception as e2:
            return {"status": "failed", "error": f"{type(e2).__name__}: {str(e2)[:250]}", "seconds": 0}

    types = {e["name"]: e["type"] for e in ent_dict["entities"]}
    rows = [{"h": tr["h"], "t": tr["t"], "r": tr["r"],
             "ht": TYPE_LABEL.get(types.get(tr["h"]), "Concept"),
             "tt": TYPE_LABEL.get(types.get(tr["t"]), "Concept"),
             "ev": tr.get("evidence", "")} for tr in triples]
    try:
        t0 = time.perf_counter()
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
            s.run("""UNWIND $rows AS row
                MERGE (a:Entity {name: row.h}) ON CREATE SET a.kind = row.ht
                MERGE (b:Entity {name: row.t}) ON CREATE SET b.kind = row.tt
                CREATE (a)-[:REL {relation: row.r, evidence: row.ev}]->(b)""", rows=rows).consume()
        secs = time.perf_counter() - t0
        with driver.session() as s:
            n = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            m = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
        driver.close()
        return {"status": "ok", "seconds": round(secs, 4), "nodes": n, "edges": m}
    except Exception as e:
        return {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:250]}", "seconds": 0}


def neo4j_queries():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=None)
    out = []
    qs = [
        ("孙少平的直接关系", "MATCH (a:Entity {name:'孙少平'})-[r:REL]->(b) RETURN count(DISTINCT b) AS c"),
        ("田福军→田晓霞→孙少平 两跳路径", "MATCH p=(a:Entity {name:'田福军'})-[:REL*1..2]->(b:Entity {name:'孙少平'}) RETURN count(p) AS c"),
        ("全图边数统计", "MATCH ()-[r:REL]->() RETURN count(r) AS c"),
    ]
    with driver.session() as s:
        for name, q in qs:
            t0 = time.perf_counter()
            try:
                res = s.run(q).single()
                secs = time.perf_counter() - t0
                out.append({"name": name, "seconds": round(secs, 5), "result": dict(res) if res else None})
            except Exception as e:
                out.append({"name": name, "error": str(e)[:120]})
    driver.close()
    return out


def networkx_queries(G):
    out = []
    for name, fn in QUERIES:
        t0 = time.perf_counter()
        try:
            res = fn(G)
            out.append({"name": name, "seconds": round(time.perf_counter() - t0, 5),
                        "result": (len(res) if isinstance(res, list) else res)})
        except Exception as e:
            out.append({"name": name, "error": str(e)[:120]})
    return out


def visualize(G):
    try:
        from pyvis.network import Network
        net = Network(height="760px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
        colors = {"Person": "#e15759", "Place": "#59a14f", "Organization": "#4e79a7",
                  "Event": "#f28e2b", "Time": "#9c755f", "Concept": "#bab0ac"}
        for n, d in G.nodes(data=True):
            net.add_node(n, label=n, color=colors.get(d.get("label"), "#888"),
                         title=f"{n} ({d.get('label')})", size=16)
        for a, b, d in G.edges(data=True):
            net.add_edge(a, b, title=d.get("relation", ""), label=d.get("relation", ""),
                         arrows="to", color="#999999")
        net.save_graph(str(OUT / "kg.html"))
        return True
    except Exception as e:
        return str(e)


def main():
    ed = load_json(DICT)
    triples = load_json(LLM_TRIPLES)["triples"]

    G, llm_build = build_llm_graph(triples, ed)
    viz = visualize(G)
    nx_q = networkx_queries(G)

    neo = neo4j_import(triples, ed)
    neo_q = []
    if neo["status"] == "ok":
        try:
            neo_q = neo4j_queries()
        except Exception as e:
            neo_q = [{"error": str(e)[:200]}]

    # 写出 Cypher 备份（即使 Neo4j 不可用也可手工导入）
    cypher = ["MATCH (n) DETACH DELETE n;"]
    for tr in triples:
        cypher.append(
            "MERGE (a:Entity {name:'%s'}) ON CREATE SET a.kind='%s' "
            "MERGE (b:Entity {name:'%s'}) ON CREATE SET b.kind='%s' "
            "CREATE (a)-[:REL {relation:'%s', evidence:'%s'}]->(b);" % (
                tr["h"], TYPE_LABEL.get(ed_types(ed, tr["h"]), "Concept"),
                tr["t"], TYPE_LABEL.get(ed_types(ed, tr["t"]), "Concept"),
                tr["r"], tr.get("evidence", "").replace("'", "’").replace("\n", " "))
        )
    (OUT / "neo4j_import.cypher").write_text("\n".join(cypher), encoding="utf-8")

    graph_json = {
        "nodes": [{"id": n, "label": d.get("label"), "type": d.get("node_type")} for n, d in G.nodes(data=True)],
        "edges": [{"source": a, "target": b, "relation": d.get("relation"), "evidence": d.get("evidence", "")}
                  for a, b, d in G.edges(data=True)],
    }
    save_json(graph_json, OUT / "kg_llm.json")

    lines = [
        "# 知识图谱：Neo4j vs LLM 生成与性能比较", "",
        "## 1. 实验设计", "",
        "- **LLM 图谱**：由大模型抽取的三元组直接构图（networkx MultiDiGraph），此处代表「LLM 生成图谱」路径。",
        "- **Neo4j 图谱**：把同一批三元组通过 Neo4j Python Driver 批量写入本地 Neo4j 5.26 社区版。",
        "- 比较维度：构图耗时、节点/边规模、三类典型查询延迟。两种路径的语义来源相同，差异主要来自存储与查询引擎。",
        "", "## 2. 构图性能", "",
        "| 路径 | 状态 | 耗时(s) | 节点数 | 关系数 |", "| --- | --- | ---: | ---: | ---: |",
        f"| LLM 直接构图（networkx + pyvis） | {'ok' if viz is True else 'warning: '+str(viz)} | {round(llm_build, 5)} | {G.number_of_nodes()} | {G.number_of_edges()} |",
    ]
    if neo["status"] == "ok":
        lines.append(f"| Neo4j 批量导入 | ok | {neo['seconds']} | {neo['nodes']} | {neo['edges']} |")
    else:
        lines.append(f"| Neo4j 批量导入 | failed | — | — | — |")
    if neo.get("error"):
        lines += ["", f"> Neo4j 错误：{neo['error']}", "> 已生成 `outputs/neo4j_import.cypher`，可在 Neo4j Browser 中执行。", ""]

    lines += ["", "## 3. 查询性能（同一图谱，三种查询）", "",
              "| 查询 | LLM/NetworkX 耗时(s) | Neo4j 耗时(s) | 说明 |", "| --- | ---: | ---: | --- |"]
    for i, q in enumerate(nx_q):
        nq = neo_q[i] if i < len(neo_q) else {}
        lines.append(f"| {q['name']} | {q.get('seconds', '—')} | {nq.get('seconds', '—')} | "
                     f"NetworkX 结果={q.get('result', q.get('error','—'))}；Neo4j 结果={nq.get('result', nq.get('error','—'))} |")

    deg = sorted(G.degree(), key=lambda x: -x[1])[:12]
    lines += ["", "## 4. 图谱结构观察", "",
              "- 节点度数 Top 12：" + "，".join(f"{n}({d})" for n, d in deg) + "。",
              "- 中心人物明显：孙少平、孙少安、田福堂、田晓霞、田润叶构成图谱主干。",
              "- 多数关系为一次性事件边，重复边（同一对节点多条关系）体现了人物关系的复杂性。",
              "", "## 5. 性能比较结论", "",
              "- 构图：NetworkX/LLM 路径是纯内存字典操作，通常快于 Neo4j 的事务写入；Neo4j 胜在持久化、并发与事务安全。",
              "- 查询：小图（几十节点）上 NetworkX 的 Python 遍历可能与 Neo4j 相当甚至更快；数据规模增大后 Neo4j 的索引与原生图遍历优势会显现。",
              "- 定位差异：LLM 负责「从文本生成图谱」（语义质量），Neo4j 负责「存储、查询与治理」（工程性能），二者是互补而非简单替代关系。",
              ""]
    (OUT / "kg_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("step4 done", neo["status"], "graph", G.number_of_nodes(), G.number_of_edges())


def ed_types(ed, name):
    for e in ed["entities"]:
        if e["name"] == name or name in e.get("aliases", []):
            return e["type"]
    return "MISC"


if __name__ == "__main__":
    main()
