"""用独立文件（避免PowerShell引号转义）验证知识库召回质量"""

import sys, types
sys.path.insert(0, r"C:\Users\T480\workspace\投研能力\material-ai-diagnosis")
for m in ["dotenv", "psycopg2", "pgvector", "pgvector.psycopg2"]:
    sys.modules[m] = types.ModuleType(m)

import db_sqlite as db

db.init_db()
problem = "美缝剂环氧E-51+E-44：加胺固化T-31，25°C养护48小时表面仍发粘不干"

print("问题：", problem)
print("\nTop6召回：")
results = db.retrieve(problem, top_k=6)
for i, r in enumerate(results, 1):
    sim = r["similarity"]
    bar = "█" * int(sim * 30)
    content = r["content"][:60].replace("\n", " ")
    print("  [%d] %s/%s sim=%.3f %s" % (i, r["source_type"], r["source_id"], sim, bar))
    print("       %s..." % content)
