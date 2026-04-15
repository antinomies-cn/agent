# 项目说明（简版）

## 概览
FastAPI + LangChain 的 Agent 服务，包含：
- HTTP 与 WebSocket 对话
- 工具调用（占星、搜索、向量检索、自检）
- URL 入库到 Qdrant
- Redis 记忆（不可用时降级进程内）

## 依赖
- Python 3.10+
- Qdrant（远程或本地存储）
- Redis（可选但推荐）
- 环境变量（见下方）

## 安装
```bash
pip install -r bot/requirements.txt
```

## 启动
```bash
python bot/app/main.py
```

## 部署（简版）
1) 准备环境变量（建议使用 .env 或系统环境变量）
- ENV=prod
- OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL
- QDRANT_URL 或 QDRANT_DB_PATH
- REDIS_URL（可选但推荐）

2) 启动服务（建议绑定 0.0.0.0）
```bash
set API_HOST=0.0.0.0
set API_PORT=8000
python bot/app/main.py
```

3) 健康检查
```bash
curl http://127.0.0.1:8000/health
```

## 核心接口
- POST /chat (query, session_id)
- WS /ws?session_id=...
- POST /add_urls, POST /add_urls/dry_run
- GET /health
- GET /memory/status?session_id=...
- POST /qdrant/init, POST /qdrant/recreate
- GET /qdrant/health, /qdrant/collections, /qdrant/status

## 关键环境变量
LLM:
- OPENAI_API_KEY
- OPENAI_API_BASE（不要带 /v1）
- OPENAI_MODEL

Embeddings:
- EMBEDDINGS_API（openai 或 local）
- EMBEDDINGS_MODEL
- EMBEDDINGS_DIMENSION
- EMBEDDINGS_CACHE_DIR（本地模型缓存目录）
- EMBEDDINGS_LOCAL_FILES_ONLY（true/false）
- EMBEDDINGS_HF_ENDPOINT（可选镜像）

Qdrant:
- QDRANT_URL（远程）
- QDRANT_API_KEY（可选）
- QDRANT_DB_PATH（本地路径）
- QDRANT_COLLECTION
- QDRANT_DISTANCE（Cosine|Dot|Euclid）

Redis 记忆:
- REDIS_URL（或 REDIS_HOST/REDIS_PORT/REDIS_DB/REDIS_PASSWORD）
- MEMORY_TTL
- MEMORY_COMPACT_MESSAGE_COUNT

占星工具:
- XINGPAN_APP_ID
- XINGPAN_APP_KEY
- XINGPAN_API_URL
- XINGPAN_TIMEOUT
- ASTRO_UID
- ASTRO_BIRTH_DT / ASTRO_LONGITUDE / ASTRO_LATITUDE（可选探测配置）

其他:
- API_HOST（默认 127.0.0.1）
- API_PORT（默认 8000）
- LOG_LEVEL, LOG_DIR
- WEB_LOADER_VERIFY_SSL（true/false）

## 说明
- /add_urls 会抓取 URL、切块、向量化并写入 Qdrant。
- /add_urls/dry_run 仅预览切块，不写入。
- 如果 Qdrant 维度与 embeddings 不一致，接口返回 409。

## 测试
```bash
pytest -q
```
