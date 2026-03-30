# Chatbot Agent 开发文档

## 项目概览

这是一个基于 FastAPI + LangChain 的占卜/问答 Agent 服务，支持：

1. HTTP 与 WebSocket 双通道对话
2. 工具调用（占星、搜索、向量检索、自检）
3. 会话记忆（Redis 优先，失败自动降级内存）
4. 日志观测与健康检查

---

## 目录

1. [架构总览](#架构总览)
2. [接口清单](#接口清单)
3. [客户端流程](#客户端流程)
4. [学习能力流程](#学习能力流程)
5. [方法手册](#方法手册)
6. [需求改造执行模板](#需求改造执行模板)

---

## 架构总览

### 服务器端

1. 技术选型：Python + FastAPI
2. 核心接口：
     - `POST /chat`
     - `POST /add_urls`
     - `POST /add_pdfs`
     - `POST /add_texts`
3. 记忆存储：Redis（不可用时降级为进程内内存）

### 客户端

1. 用户输入 -> 情绪识别 -> 角色设定
2. Prompt 清晰度越高，Agent 响应质量越高
3. 工具链路：用户请求 -> Agent 决策 -> 工具调用 -> 结果回填
4. 会话隔离：通过 `session_id` 保障多用户并发安全

### 能力列表

1. API 服务
2. Agent 框架
3. Tools：搜索、查询信息、专业知识库
4. 记忆（短期与长期）
5. 学习能力

---

## 接口清单

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/chat` | POST | 主对话接口 |
| `/add_urls` | POST | 从 URL 学习知识 |
| `/add_pdfs` | POST | 从 PDF 学习知识 |
| `/add_texts` | POST | 从文本学习知识 |
| `/health` | GET | 服务健康检查 |
| `/memory/status` | GET | 会话 memory 状态观测 |
| `/ws` | WebSocket | 实时会话通道 |

### add_urls 输入示例（支持 Query 与 JSON）

1. `/add_urls` JSON 方式（推荐）

```json
{
    "urls": ["https://example.com/a", "https://example.com/b"],
    "chunk_strategy": "balanced",
    "chunk_size": 900,
    "chunk_overlap": 120
}
```

1. `/add_urls` Query 方式

```http
POST /add_urls?url=https://example.com/a&chunk_strategy=balanced&chunk_size=900&chunk_overlap=120
```

1. `/add_urls` Query 方式传多个 urls（重复键）

```http
POST /add_urls?urls=https://example.com/a&urls=https://example.com/b&chunk_strategy=article
```

1. `/add_urls/dry_run` JSON 方式

```json
{
    "url": "https://example.com/a",
    "chunk_strategy": "faq",
    "preview_limit": 2
}
```

1. `/add_urls/dry_run` Query 方式

```http
POST /add_urls/dry_run?url=https://example.com/a&chunk_strategy=faq&preview_limit=2
```

说明：

1. 同名字段同时出现在 JSON 和 Query 时，优先使用 JSON。
2. `url` 和 `urls` 可混用，系统会自动合并并过滤空白项。
3. `dry_run` 只做抓取与切块预览，不写入 Qdrant。

---

## 客户端流程

1. 用户输入问题
2. 系统判断情绪倾向（规则优先，LLM 兜底）
3. Agent 按意图选择工具
4. 工具返回结构化结果
5. LLM 总结并回复用户

---

## 学习能力流程

1. 输入 URL
2. 抽取 HTML 文本
3. 文本切分与向量化
4. 向量检索命中文本块
5. LLM 结合检索结果作答

---

## Embedding 启动前自检

1. 容器启动前会先执行 `python app/startup_check.py`，通过后才启动 FastAPI。
2. 自检仅在 `EMBEDDINGS_API=local` 时生效；`openai` 模式会自动跳过。
3. 默认开启：`EMBEDDINGS_STARTUP_CHECK=true`。
4. 自检会校验：模型目录存在、关键文件齐全、可离线加载并输出向量。
5. 若需临时关闭：设置 `EMBEDDINGS_STARTUP_CHECK=false`。

---

## 方法手册

本章节用于代码评审、PR 描述和需求改造时快速定位：

1. 方法解决什么问题
2. 方法如何实现
3. 方法依赖哪些函数或类

### server.py 核心方法

#### setup_logger

- 目的：统一日志规范，输出到控制台和滚动文件。
- 实现：读取环境变量，创建日志目录，清空 root handler 后重新挂载两类 handler。
- 依赖：os.getenv、os.makedirs、logging.getLogger、logging.StreamHandler、logging.handlers.RotatingFileHandler。

#### CustomProxyLLM._to_proxy_role

- 目的：把 LangChain 消息类型转换为 OpenAI 兼容 role。
- 实现：按映射表转换 human/ai/system/tool/function；未命中时尝试 additional_kwargs.role，再兜底 user。
- 依赖：getattr。

#### CustomProxyLLM._extract_content

- 目的：稳健解析模型响应文本。
- 实现：校验 choices/message 后读取 content；兼容 string 与 list[dict] 两种结构。
- 依赖：dict.get、isinstance、str.strip。

#### CustomProxyLLM._extract_tool_calls

- 目的：提取并过滤 tool_calls，避免脏数据影响 Agent。
- 实现：读取 choices[0].message.tool_calls，仅保留 function.name 合法项。
- 依赖：dict.get、isinstance。

#### CustomProxyLLM._resolve_timeout_seconds

- 目的：统一超时预算，兼容请求级 deadline 裁剪。
- 实现：读取线程上下文 request_timeout/request_deadline，计算剩余预算并设置最小超时。
- 依赖：getattr、time.perf_counter、min、max。

#### CustomProxyLLM._request_completion

- 目的：实际发起模型请求并返回 content + tool_calls。
- 实现：
    1. 校验 api_key/base_url/model。
    2. 构建 proxy_messages 并透传 tool_calls/tool_call_id。
    3. 构建 data 并透传 tools/tool_choice/parallel_tool_calls。
    4. requests.post 调用网关，按 429/5xx 重试。
    5. 解析返回并分层处理 HTTP/Timeout/Connection/KeyError。
- 依赖：requests.post、resp.raise_for_status、resp.json、time.sleep、_to_proxy_role、_extract_content、_extract_tool_calls、_resolve_timeout_seconds。

#### Master._normalize_session_id

- 目的：会话 ID 标准化，杜绝空 session_id 导致串话。
- 实现：strip 后校验，空值抛 ValueError。
- 依赖：str.strip、ValueError。

#### Master._get_session_lock

- 目的：同一 session 请求串行执行。
- 实现：用字典缓存会话锁，不存在则创建。
- 依赖：threading.Lock、dict.get。

#### Master._extract_user_facts_from_messages

- 目的：历史压缩时保留用户事实（姓名、年龄、生日、偏好等）。
- 实现：聚合用户消息文本，按正则抽取结构化字段。
- 依赖：re.search、getattr、isinstance。

#### Master._get_chat_history

- 目的：按 session 获取历史，优先 Redis，失败降级进程内历史。
- 实现：尝试多种 RedisChatMessageHistory 参数组合，任一成功即返回；失败回退 InMemory。
- 依赖：RedisChatMessageHistory、_get_or_create_local_history。

#### Master._compact_history_if_needed

- 目的：历史消息达到阈值时压缩，降低上下文成本。
- 实现：
    1. 读取全部消息并转文本。
    2. 提取用户事实并构建摘要提示词。
    3. 调用 normal_llm 生成摘要（超时则退化为截断文本）。
    4. clear 历史后写入单条摘要。
- 依赖：ChatPromptTemplate.from_messages、StrOutputParser、_invoke_with_timeout、chat_history.clear、_append_summary_message。

#### Master._build_memory

- 目的：构建 Agent 可用会话记忆对象。
- 实现：拉取历史并按阈值压缩，再封装为 ConversationBufferMemory。
- 依赖：_get_chat_history、_compact_history_if_needed、ConversationBufferMemory。

#### Master._route_intent + _select_tools_by_intent

- 目的：先识别意图，再裁剪工具，减少误调用与时延。
- 实现：关键词映射意图，再按映射返回最小工具集。
- 依赖：_match_keywords、dict.get。

#### Master._invoke_with_timeout

- 目的：统一函数执行超时控制。
- 实现：写入线程上下文后执行 func；finally 清理上下文；超预算抛 TimeoutError（可携带 partial_result）。
- 依赖：time.perf_counter、setattr、delattr、TimeoutError。

#### Master._build_astro_fallback_output

- 目的：占星总结失败时仍返回可读结果。
- 实现：逆序扫描 intermediate_steps，找到占星工具结果 JSON，提取星座与片段生成兜底摘要。
- 依赖：json.loads、json.dumps、getattr、isinstance。

#### Master.mood_chain

- 目的：识别用户情绪并选择对应角色设定。
- 实现：先规则命中，未命中再走 LLM 分类链。
- 依赖：_rule_based_mood、ChatPromptTemplate.from_template、StrOutputParser、_invoke_with_timeout。

#### Master.run

- 目的：主编排入口。
- 实现：
    1. 会话 ID 校验与会话锁。
    2. 意图路由、情绪识别、prompt 与 memory 构建。
    3. Agent 创建与执行。
    4. 记录工具轨迹。
    5. 占星场景做短答或超时兜底。
- 依赖：_normalize_session_id、_get_session_lock、_route_intent、mood_chain、_build_memory、_select_tools_by_intent、create_openai_tools_agent、AgentExecutor、_invoke_with_timeout、_build_astro_fallback_output。

#### chat / ws / memory_status

- 目的：提供 HTTP、WebSocket 与 memory 观测接口。
- 实现：chat 与 ws 都强制 session_id；memory_status 提供会话记忆状态诊断。
- 依赖：master.run、master.run_async、master.get_memory_status、HTTPException、WebSocket。

### Mytools.py 核心方法


#### _tool_result

- 目的：统一工具返回结构，方便上层稳定解析。
- 实现：返回 ok/code/data/error 的 JSON 字符串。
- 依赖：json.dumps。

#### _normalize_birth_dt

- 目的：归一化出生时间格式。
- 实现：优先 fromisoformat，失败再按固定格式解析，仍失败返回原文。
- 依赖：datetime.fromisoformat、datetime.strptime、strftime。

#### _request_astro_api

- 目的：统一封装星盘 API 调用。
- 实现：
    1. 读取 app_id/app_key/timeout。
    2. 按 GET/POST 发请求。
    3. 成功返回 _tool_result(ok=True)。
    4. 失败按超时、HTTP、未知异常分层返回结构化错误码。
- 依赖：_astro_base_url、requests.get、requests.post、resp.raise_for_status、resp.json、_tool_result。

#### _get_vector_retriever

- 目的：向量检索器惰性单例初始化。
- 实现：双检锁 + 全局缓存，首次构建 Qdrant + Embeddings + Retriever。
- 依赖：threading.Lock、QdrantClient、Qdrant、OpenAIEmbeddings、as_retriever。

#### test

- 目的：系统自检。
- 实现：按 scope 检查 astro/vector/search 三类依赖并输出报告。
- 依赖：_astro_base_url、_get_vector_retriever、os.getenv、json.dumps。

#### search

- 目的：联网实时搜索。
- 实现：SerpAPIWrapper.run 执行查询并返回文本结果。
- 依赖：SerpAPIWrapper。

#### vector_search

- 目的：从本地向量库检索知识。
- 实现：retriever.invoke 后拼接命中文本，空命中返回提示。
- 依赖：_get_vector_retriever、retriever.invoke。

#### xingpan / astro_my_sign / astro_natal_chart / astro_current_chart / astro_transit_chart

- 目的：提供占星相关工具能力。
- 实现：参数校验后统一调用 _request_astro_api（或兼容函数 _call_xingpan_api）。
- 依赖：_normalize_birth_dt、float、_request_astro_api、_call_xingpan_api。

---

## 需求改造执行模板

每次提需求建议按下面模板走，避免“只改一处”：

1. 需求三元组：入口层（API/页面）- 资源层（文案/配置）- 引用层（逻辑/测试/文档）。
2. 影响分析：会影响哪些函数和调用路径。
3. 代码修改：按路径逐层修改并保持返回结构一致。
4. 回归检查：
     - HTTP 与 WS 是否行为一致。
     - memory 与日志是否包含新增字段。
     - 失败分支文案和错误码是否保持稳定。

---

## Docker 端口暴露自检（Windows）

目标：仅暴露 8000，确保 6379 不对宿主机开放。

1. 检查 8000 监听（应有结果）

```powershell
Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, OwningProcess, State
```

2. 检查 6379 监听（应无结果）

```powershell
Get-NetTCPConnection -State Listen -LocalPort 6379 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, OwningProcess, State
```

3. 若 6379 仍有监听，定位并清理

```powershell
# 找到占用进程
Get-NetTCPConnection -State Listen -LocalPort 6379 -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty OwningProcess

# 查看进程名
Get-Process -Id <PID>
```

4. 常见清理方式

```powershell
# 如为旧 Redis 容器
docker ps --filter "name=myredis"
docker stop myredis
docker rm myredis

# 如为本地 Redis 服务（按实际服务名）
Get-Service | Where-Object { $_.Name -match "redis" -or $_.DisplayName -match "redis" }
Stop-Service <ServiceName>
Set-Service <ServiceName> -StartupType Disabled
```
