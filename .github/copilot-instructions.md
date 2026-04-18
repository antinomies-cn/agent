# 工作区指令

这是一个 FastAPI + LangChain 的 Agent 服务工作区。主应用位于 `bot/` 下；修改时请保持变更小、分层清晰，并与现有的 router/service 拆分保持一致。

## 项目地图

- `bot/app/main.py` 是应用入口和运行时装配层。
- `bot/app/api/routers/` 只放薄路由，负责 HTTP / WebSocket 的参数接收与分发。
- `bot/app/services/` 负责核心业务逻辑，例如 Agent 编排、URL 入库和 Qdrant 集成。
- `bot/app/llm/custom_llm.py` 封装 OpenAI 兼容网关行为。
- `bot/app/startup_check.py` 负责启动校验。
- `bot/tests/` 覆盖配置、安全、日志、路由和启动行为的回归测试。

## 工作原则

- 优先改 service 层，不要把业务逻辑塞进路由层。
- 除非任务明确要求修改 API，否则保持路由契约稳定。
- 不要重复实现已经存在于共享模块中的辅助逻辑，例如 `bot/app/api/deps.py` 或 `bot/app/services/add_urls_service.py`。
- 将运行时服务对象视为不可序列化对象，不要对它们执行 pickle 或 deepcopy。
- 日志和测试中要保护隐私，避免暴露密钥、令牌、原始凭据或敏感请求内容。
- 处理 OpenAI 兼容响应时，要防御 `choices`、`message`、`content` 缺失的情况，并安全处理非字符串内容。
- tool/function 消息要显式映射，确保工具调用上下文端到端保留。
- OpenAI 兼容基础地址要谨慎规范化；某些 embedding 网关需要单独的基础地址，或者需要一个 `/v1` 规范化端点。

## 快速开始

- 安装依赖：`pip install -r bot/requirements.txt`
- 运行测试：使用 VS Code 任务 `Test: bot`，或者在 `bot/` 下执行 `bot/run_tests.ps1 -q`
- 本地启动：`python bot/app/main.py`

## 项目概览

项目提供以下能力：

- HTTP 与 WebSocket 对话
- 工具调用（占星、搜索、向量检索、自检）
- URL 入库到 Qdrant
- Redis 记忆，不可用时自动降级到进程内内存

## 依赖与环境

- Python 3.10+
- Qdrant（远程或本地存储）
- Redis（可选，但推荐）
- 环境变量按项目说明配置

### LLM

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`，不要带尾部 `/v1`
- `OPENAI_MODEL`

### Embeddings

- `EMBEDDINGS_API`（`openai` 或 `local`）
- `EMBEDDINGS_MODEL`
- `EMBEDDINGS_DIMENSION`
- `EMBEDDINGS_CACHE_DIR`（本地模型缓存目录）
- `EMBEDDINGS_LOCAL_FILES_ONLY`（`true` / `false`）
- `EMBEDDINGS_HF_ENDPOINT`（可选镜像）

### Qdrant

- `QDRANT_URL`（远程）
- `QDRANT_API_KEY`（可选）
- `QDRANT_DB_PATH`（本地路径）
- `QDRANT_COLLECTION`
- `QDRANT_DISTANCE`（`Cosine` / `Dot` / `Euclid`）

### Redis 记忆

- `REDIS_URL`，或 `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` / `REDIS_PASSWORD`
- `MEMORY_TTL`
- `MEMORY_COMPACT_MESSAGE_COUNT`

### 占星工具

- `XINGPAN_APP_ID`
- `XINGPAN_APP_KEY`
- `XINGPAN_API_URL`
- `XINGPAN_TIMEOUT`
- `ASTRO_UID`
- `ASTRO_BIRTH_DT` / `ASTRO_LONGITUDE` / `ASTRO_LATITUDE`（可选探测配置）

### 其他

- `API_HOST`（默认 `127.0.0.1`）
- `API_PORT`（默认 `8000`）
- `LOG_LEVEL`
- `LOG_DIR`
- `WEB_LOADER_VERIFY_SSL`（`true` / `false`）

## 接口清单

- `POST /chat`（`query`、`session_id`）
- `WS /ws?session_id=...`
- `POST /add_urls`，`POST /add_urls/dry_run`
- `GET /health`
- `GET /memory/status?session_id=...`
- `POST /qdrant/init`，`POST /qdrant/recreate`
- `GET /qdrant/health`，`/qdrant/collections`，`/qdrant/status`

## 部署要点

1. 准备环境变量，建议使用 `.env` 或系统环境变量。
2. 生产环境建议绑定 `0.0.0.0`。
3. 启动示例：

```bash
set API_HOST=0.0.0.0
set API_PORT=8000
python bot/app/main.py
```

4. 健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## 说明

- `/add_urls` 会抓取 URL、切块、向量化并写入 Qdrant。
- `/add_urls/dry_run` 仅预览切块，不写入。
- 如果 Qdrant 维度与 embeddings 不一致，接口返回 `409`。
- URL 入库行为已经有文档和测试，修改时要保留现有的归一化、安全检查、切块和重试流程。

## 常见坑

- 如果当前 Python 环境里没有 `redis`，pytest 收集可能会失败。
- `OPENAI_API_BASE` 末尾不要带 `/v1`，LLM 适配层会自动规范化请求路径。
- Qdrant 和 embedding 配置要保持一致，维度不匹配要显式处理。
- 路由层应保持薄，不要把业务逻辑从 service 层搬进去。

## 文档

优先链接现有文档，不要重复粘贴内容：

- [bot/readme.md](../bot/readme.md)
- [bot/docs/gateway-security-env.md](../bot/docs/gateway-security-env.md)

如果新增了行为区域，请同步更新相关文档。