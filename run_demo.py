"""一键演示：真实豆包API + SQLite知识库"""
import sys, types
sys.path.insert(0, r"C:\Users\T480\workspace\投研能力\material-ai-diagnosis")

# Mock psycopg2/pgvector（演示不需要PG）
for m in ["psycopg2", "psycopg2.extras", "psycopg2.extensions", "pgvector", "pgvector.psycopg2"]:
    mod = types.ModuleType(m)
    sys.modules[m] = mod
# 给 psycopg2 子模块补属性
sys.modules["psycopg2"].extras = sys.modules["psycopg2.extras"]
sys.modules["psycopg2.extras"].RealDictCursor = None
sys.modules["psycopg2.extras"].Json = lambda x, **kw: x
sys.modules["pgvector.psycopg2"].register_vector = lambda *a, **k: None

# 加载 .env
from dotenv import load_dotenv
load_dotenv()

import os
os.environ.setdefault("PG_PASS", "demo")
os.environ.setdefault("DOUBAO_TIMEOUT", "300")

import config
config.MAX_HYPOTHESIS = 5
config.MAX_ROUNDS = 3

import demo_engine

print("\n" + "=" * 65)
print("  PCM 全品类材料 — 偏差排查系统（豆包API驱动）")
print("  Model: doubao-seed-2-1-lite-260915")
print("=" * 65)

test_problem = (
    "竹基碳/PEG复合PCM：500ml小试真空浸渍包封率85%；"
    "放大至50L釜同配方同参数，包封率跌至58%，热循环50次底部渗出PEG液体"
)

print(f"\n📋 测试场景: 竹基碳/PCM真空浸渍放大")
print(f"   小试包封率 85% → 中试暴跌至 58%，热循环 50 次渗漏")

result = demo_engine.start_diagnosis(test_problem)

print("\n" + "=" * 65)
print("  演示结论")
print("=" * 65)

cid = result["case_id"]
hypo = result["hypothesis_with_prob"]
questions = result["questions"]

print(f"\n  案例编号: {cid}")
print(f"  生成假设: {len(hypo)} 条")
print(f"  生成追问: {len(questions)} 个")
print(f"  召回知识: {len(result['retrieved'])} 条")
if hypo:
    top_h = max(hypo, key=hypo.get)
    top_p = hypo[top_h]
    print(f"  🎯 最高置信度假设 ({top_p:.0%}):")
    print(f"     {top_h}")
