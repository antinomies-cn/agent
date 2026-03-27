import os
import time
import json
import logging
import logging.handlers
import requests
import threading
from Mytools import *
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory
from texts import SYSTEMPL, MOODS, MOOD_CLASSIFY_PROMPT, USER_MESSAGES

# 线程本地上下文：为LLM请求传递每次调用的超时
_thread_ctx = threading.local()

# ===================== 日志规范化配置 =====================
# 加载环境变量
load_dotenv()

# 日志配置
def setup_logger():
    """配置规范化的日志系统"""
    # 日志级别映射
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    # 从环境变量获取日志配置，默认info级别
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    log_dir = os.getenv("LOG_DIR", "./logs")
    log_max_size = int(os.getenv("LOG_MAX_SIZE", 10 * 1024 * 1024))  # 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志格式：时间 | 日志级别 | 模块 | 函数 | 行号 | 内容
    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s | %(funcName)s | %(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_map.get(log_level, logging.INFO))
    
    # 清除默认处理器
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # 文件轮转处理器
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "divination_master.log"),
        maxBytes=log_max_size,
        backupCount=log_backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    return root_logger

# 初始化日志
logger = setup_logger()

# 区分开发/生产环境
IS_PROD = os.getenv("ENV", "dev") == "prod"

app = FastAPI()


# ===================== 自定义LLM =====================
class CustomProxyLLM(BaseChatModel):
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 3000

    @property
    def _llm_type(self) -> str:
        return "baishan"

    def __getstate__(self):
        """避免运行时对象被误序列化，导致多进程/拷贝路径下出现锁相关错误。"""
        raise TypeError("CustomProxyLLM is a runtime object and cannot be pickled")

    @staticmethod
    def _to_proxy_role(message: BaseMessage) -> str:
        """将LangChain消息类型映射到OpenAI兼容角色。"""
        role_mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
            "function": "tool",
        }
        role = role_mapping.get(message.type)
        if role:
            return role
        # 兼容部分消息把role放在additional_kwargs里的情况
        custom_role = getattr(message, "additional_kwargs", {}).get("role")
        if custom_role in {"user", "assistant", "system", "tool"}:
            return custom_role
        return "user"

    @staticmethod
    def _extract_content(payload: dict) -> str:
        """稳健提取模型回复文本。"""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise KeyError("choices")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise KeyError("message")

        content = message.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()

        # 兼容content为分段结构（list[dict]）的响应
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts).strip()

        return str(content).strip()

    @staticmethod
    def _extract_tool_calls(payload: dict) -> list[dict]:
        """提取OpenAI兼容响应中的tool_calls。"""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return []

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return []

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []

        # 只保留结构正确的条目，避免脏数据影响Agent解析
        valid_calls = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            if not isinstance(fn, dict):
                continue
            if not isinstance(fn.get("name"), str):
                continue
            valid_calls.append(call)
        return valid_calls

    @staticmethod
    def _resolve_timeout_seconds(default_timeout: float = 60.0) -> float:
        """根据线程上下文计算本次请求超时，包含deadline裁剪与最小值保护。"""
        request_timeout = getattr(_thread_ctx, "request_timeout", None)
        request_deadline = getattr(_thread_ctx, "request_deadline", None)

        try:
            timeout_seconds = float(request_timeout) if request_timeout is not None else default_timeout
        except (TypeError, ValueError):
            timeout_seconds = default_timeout

        if request_deadline is not None:
            remaining_budget = request_deadline - time.perf_counter()
            timeout_seconds = min(timeout_seconds, remaining_budget)

        return max(timeout_seconds, 3.0)

    def _request_completion(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs
    ) -> tuple[str, list[dict]]:
        # 参数校验
        try:
            call_start = time.perf_counter()
            required_config = [
                ("api_key", self.api_key, "模型密钥未配置"),
                ("base_url", self.base_url, "模型接口地址未配置"),
                ("model", self.model, "模型名称未配置"),
            ]
            for config_key, config_value, error_msg in required_config:
                if not config_value:
                    logger.error(f"LLM配置错误: {error_msg}")
                    return USER_MESSAGES["config"][config_key], []

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "LangChain-CustomLLM/1.0"
            }

            proxy_messages = []
            for m in messages:
                role = self._to_proxy_role(m)
                msg = {"role": role, "content": m.content}

                # assistant 消息若包含工具调用，必须透传 tool_calls。
                if role == "assistant":
                    assistant_tool_calls = getattr(m, "additional_kwargs", {}).get("tool_calls")
                    if isinstance(assistant_tool_calls, list) and assistant_tool_calls:
                        msg["tool_calls"] = assistant_tool_calls

                # tool 消息必须携带 tool_call_id，对齐上一个 assistant.tool_calls。
                if role == "tool":
                    tool_call_id = getattr(m, "tool_call_id", None)
                    if not tool_call_id:
                        tool_call_id = getattr(m, "additional_kwargs", {}).get("tool_call_id")
                    if tool_call_id:
                        msg["tool_call_id"] = tool_call_id

                proxy_messages.append(msg)

            data = {
                "model": self.model,
                "messages": proxy_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            # 透传工具调用相关字段，否则Agent永远拿不到tool schema。
            for key in ["tools", "tool_choice", "parallel_tool_calls", "response_format"]:
                value = kwargs.get(key)
                if value is not None:
                    data[key] = value
            if stop:
                data["stop"] = stop

            url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
            
            # 开发环境打印调试信息
            if not IS_PROD:
                logger.debug(f"LLM请求URL: {url}")
                logger.debug(f"LLM请求体: {data}")

            timeout_seconds = self._resolve_timeout_seconds()

            # 对网关瞬时失败做短重试，降低偶发500对用户的影响。
            try:
                max_retries = int(os.getenv("LLM_RETRY_COUNT", "1"))
            except ValueError:
                max_retries = 1
            max_retries = max(0, max_retries)
            retry_status_codes = {429, 500, 502, 503, 504}

            last_http_error = None
            resp = None
            for attempt in range(max_retries + 1):
                try:
                    resp = requests.post(url, headers=headers, json=data, timeout=timeout_seconds)
                    resp.raise_for_status()
                    break
                except requests.exceptions.HTTPError as e:
                    last_http_error = e
                    status_code = e.response.status_code if e.response is not None else 0
                    should_retry = status_code in retry_status_codes and attempt < max_retries
                    if not should_retry:
                        raise

                    sleep_seconds = min(1.5, 0.3 * (attempt + 1))
                    logger.warning(
                        "LLM调用触发重试 | attempt: %s/%s | status: %s | backoff: %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        status_code,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)

            if last_http_error is not None and resp is None:
                raise last_http_error
            
            if not IS_PROD:
                logger.debug(f"LLM响应状态码: {resp.status_code}")
                logger.debug(f"LLM响应内容: {resp.text[:500]}")

            result = resp.json()
            content = self._extract_content(result)
            tool_calls = self._extract_tool_calls(result)
            elapsed = time.perf_counter() - call_start
            logger.info(
                "LLM调用成功，响应长度: %s | tool_calls: %s | 耗时: %.2f秒",
                len(content),
                len(tool_calls),
                elapsed,
            )
            return content, tool_calls

        except requests.exceptions.HTTPError as e:
            status_code = resp.status_code if 'resp' in locals() else 0
            response_text = resp.text[:200] if 'resp' in locals() else ""
            
            error_info = {
                "type": "HTTP_ERROR",
                "status_code": status_code,
                "response_text": response_text,
                "error": str(e)[:100]
            }
            
            if status_code == 401:
                error_log = "LLM调用失败：密钥无效/过期"
                user_msg = USER_MESSAGES["http"][401]
            elif status_code == 400:
                error_log = f"LLM调用失败：请求参数错误 {response_text}"
                user_msg = USER_MESSAGES["http"][400]
            elif status_code == 500:
                error_log = "LLM调用失败：服务端内部错误"
                user_msg = USER_MESSAGES["http"][500]
            else:
                error_log = f"LLM HTTP错误 {status_code}：{response_text}"
                user_msg = USER_MESSAGES["http"]["default"]
            
            logger.error(f"{error_log} | 详细信息: {error_info}")
            return user_msg, []
            
        except requests.exceptions.Timeout:
            timeout_seconds = self._resolve_timeout_seconds()
            error_info = {"type": "TIMEOUT", "timeout": timeout_seconds}
            logger.error(f"LLM调用超时（{timeout_seconds}秒） | 详细信息: {error_info}")
            return USER_MESSAGES["timeout"], []
            
        except requests.exceptions.ConnectionError:
            error_info = {"type": "CONNECTION_ERROR", "url": url if 'url' in locals() else ""}
            logger.error(f"LLM调用失败：网络连接异常 | 详细信息: {error_info}")
            return USER_MESSAGES["connection_error"], []
            
        except KeyError as e:
            response_text = resp.text[:200] if 'resp' in locals() else ""
            error_info = {"type": "KEY_ERROR", "missing_key": str(e), "response_text": response_text}
            logger.error(f"LLM响应解析失败：缺失字段 {e} | 详细信息: {error_info}")
            return USER_MESSAGES["key_error"], []
            
        except Exception as e:
            error_info = {"type": "UNKNOWN_ERROR", "error": str(e)[:100]}
            logger.error(f"LLM未知异常 | 详细信息: {error_info}", exc_info=True)
            return USER_MESSAGES["unknown_error"], []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs
    ) -> ChatResult:
        content, tool_calls = self._request_completion(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}
        generation = ChatGeneration(message=AIMessage(content=content, additional_kwargs=additional_kwargs))
        return ChatResult(generations=[generation])

# ===================== 主逻辑类 =====================
class Master:
    def __init__(self):
        # 创建两个LLM实例
        self.normal_llm = CustomProxyLLM(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", ""),
            model=os.getenv("OPENAI_MODEL", "")
        )
        
        self.mood_llm = CustomProxyLLM(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", ""),
            model=os.getenv("OPENAI_MODEL", ""),
            temperature=0.0,
            max_tokens=12
        )

        self.all_tools = [
            search,
            test,
            vector_search,
            xingpan,
            astro_my_sign,
            astro_natal_chart,
            astro_current_chart,
            astro_transit_chart,
        ]

        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.memory_ttl = int(os.getenv("MEMORY_TTL", "86400"))
        self.memory_compact_message_count = int(os.getenv("MEMORY_COMPACT_MESSAGE_COUNT", "10"))
        self.mood_timeout_seconds = float(os.getenv("MOOD_TIMEOUT_SECONDS", "5"))
        self._local_histories: dict[str, InMemoryChatMessageHistory] = {}
        self._local_histories_lock = threading.Lock()

        logger.info("Master实例初始化完成")

    @staticmethod
    def _normalize_session_id(session_id: str | None) -> str:
        """标准化会话ID，避免空值导致多用户共享同一上下文。"""
        sid = (session_id or "").strip()
        return sid if sid else "default"

    @staticmethod
    def _history_messages_to_text(messages: list[BaseMessage]) -> str:
        """将历史消息转为可摘要文本。"""
        lines = []
        for m in messages:
            role = getattr(m, "type", "unknown")
            content = getattr(m, "content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                    elif isinstance(item, str):
                        parts.append(item)
                content_str = "".join(parts)
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content)
            lines.append(f"{role}: {content_str}")
        return "\n".join(lines)

    @staticmethod
    def _append_summary_message(chat_history, summary_text: str) -> None:
        """向历史中写入单条摘要消息。"""
        summary = f"历史对话摘要：{summary_text}"
        if hasattr(chat_history, "add_ai_message"):
            chat_history.add_ai_message(summary)
            return
        if hasattr(chat_history, "add_message"):
            chat_history.add_message(AIMessage(content=summary))
            return
        if hasattr(chat_history, "add_user_message"):
            chat_history.add_user_message(summary)

    def _get_or_create_local_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Redis不可用时，按session复用进程内历史。"""
        with self._local_histories_lock:
            history = self._local_histories.get(session_id)
            if history is None:
                history = InMemoryChatMessageHistory()
                self._local_histories[session_id] = history
            return history

    def _get_chat_history(self, session_id: str):
        """按session获取消息历史对象，优先Redis，失败降级本地。"""
        sid = self._normalize_session_id(session_id)

        # 不同版本RedisChatMessageHistory参数名存在差异，逐个尝试。
        candidate_kwargs = [
            {"session_id": sid, "url": self.redis_url, "ttl": self.memory_ttl},
            {"session_id": sid, "url": self.redis_url},
            {"session_id": sid, "redis_url": self.redis_url, "ttl": self.memory_ttl},
            {"session_id": sid, "redis_url": self.redis_url},
        ]

        try:
            for kwargs in candidate_kwargs:
                try:
                    chat_history = RedisChatMessageHistory(**kwargs)
                    _ = chat_history.messages
                    return chat_history, "redis"
                except TypeError:
                    continue

            # 最后尝试最小参数构造（部分版本只接收session_id）
            chat_history = RedisChatMessageHistory(session_id=sid)
            _ = chat_history.messages
            return chat_history, "redis"
        except Exception as e:
            logger.warning("Redis memory不可用，降级到进程内memory | session_id: %s | err: %s", sid, str(e)[:120])
            return self._get_or_create_local_history(sid), "in_memory"

    def _compact_history_if_needed(self, chat_history, session_id: str) -> None:
        """当消息数达到阈值时，将历史压缩为单条摘要。"""
        try:
            messages = list(chat_history.messages)
        except Exception as e:
            logger.warning("读取历史消息失败，跳过压缩 | session_id: %s | err: %s", session_id, str(e)[:120])
            return

        threshold = max(1, self.memory_compact_message_count)
        if len(messages) < threshold:
            return

        history_text = self._history_messages_to_text(messages)
        if not history_text.strip():
            return

        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是对话记忆压缩助手。请将对话压缩为一条简洁摘要，保留用户事实信息、偏好、约束、已完成事项和未完成事项；"
                "避免冗余和寒暄；输出纯文本。",
            ),
            ("user", "请压缩以下历史对话：\n{history_text}"),
        ])

        summary_chain = summary_prompt | self.normal_llm | StrOutputParser()
        summary_text = ""
        try:
            summary_text = self._invoke_with_timeout(
                lambda: summary_chain.invoke({"history_text": history_text}),
                timeout=15,
            ).strip()
        except Exception as e:
            logger.warning("历史压缩失败，回退截断摘要 | session_id: %s | err: %s", session_id, str(e)[:120])

        if not summary_text:
            summary_text = history_text[-1000:]

        try:
            chat_history.clear()
            self._append_summary_message(chat_history, summary_text)
            logger.info(
                "历史压缩完成 | session_id: %s | before_messages: %s | after_messages: 1",
                session_id,
                len(messages),
            )
        except Exception as e:
            logger.error("写回压缩摘要失败 | session_id: %s | err: %s", session_id, str(e)[:120], exc_info=True)

    def _build_memory(self, session_id: str) -> ConversationBufferMemory:
        """按会话ID构建Memory，历史落Redis，使用消息缓冲避免tokenizer外网依赖。"""
        sid = self._normalize_session_id(session_id)
        chat_history, _ = self._get_chat_history(sid)

        self._compact_history_if_needed(chat_history, sid)

        return ConversationBufferMemory(
            human_prefix="用户",
            ai_prefix="占卜大师",
            memory_key="history",
            output_key="output",
            return_messages=True,
            chat_memory=chat_history,
        )

    @staticmethod
    def _match_keywords(text: str, mapping: list[tuple[str, list[str]]], default: str) -> str:
        """按顺序匹配关键词并返回首个命中的标签。"""
        for label, keywords in mapping:
            if any(k in text for k in keywords):
                return label
        return default

    def _route_intent(self, query: str) -> str:
        """基于关键词的轻量意图路由，减少误调用工具。"""
        text = (query or "").strip().lower()
        if not text:
            return "default"

        intent_mapping = [
            ("astro_my_sign", ["星座信息", "我的星座", "我是什么星座", "什么星座"]),
            ("astro_natal_chart", ["本命盘", "出生盘", "星盘本命"]),
            ("astro_current_chart", ["天象盘", "当前天象", "今日天象", "现在天象"]),
            ("astro_transit_chart", ["行运盘", "流年盘", "推运", "transit"]),
            ("xingpan", ["星盘", "占星", "看盘"]),
            ("vector_search", ["马年", "知识库", "文档", "资料", "本地"]),
            ("search", ["最新", "新闻", "今天", "实时", "搜索", "查一下", "百科"]),
        ]
        return self._match_keywords(text, intent_mapping, "default")

    def _select_tools_by_intent(self, query: str, intent: str | None = None):
        """按意图动态裁剪工具集合。"""
        intent = intent or self._route_intent(query)
        map_tools = {
            "astro_my_sign": [astro_my_sign, test, search],
            "astro_natal_chart": [astro_natal_chart, xingpan, test, search],
            "astro_current_chart": [astro_current_chart, test, search],
            "astro_transit_chart": [astro_transit_chart, xingpan, test, search],
            "xingpan": [xingpan, astro_natal_chart, astro_transit_chart, test, search],
            "vector_search": [vector_search, search, test],
            "search": [search, vector_search, test],
        }
        selected = map_tools.get(intent, self.all_tools)
        logger.info(
            "工具路由完成 | intent: %s | tools: %s",
            intent,
            [getattr(t, "name", str(t)) for t in selected],
        )
        return selected

    def _invoke_with_timeout(self, func, timeout):
        """带超时的函数调用（同步执行，避免超时后线程继续运行）。"""
        start = time.perf_counter()
        res = None
        try:
            _thread_ctx.request_timeout = timeout
            _thread_ctx.request_deadline = start + timeout
            res = func()
        except Exception as e:
            logger.error(f"超时调用函数执行异常: {str(e)[:100]}", exc_info=True)
            raise
        finally:
            if hasattr(_thread_ctx, "request_timeout"):
                delattr(_thread_ctx, "request_timeout")
            if hasattr(_thread_ctx, "request_deadline"):
                delattr(_thread_ctx, "request_deadline")

        elapsed = time.perf_counter() - start
        if elapsed > timeout:
            grace = 2.0
            overrun = elapsed - timeout
            if overrun <= grace:
                logger.warning(
                    "函数调用轻微超预算但已返回结果 | timeout: %.2fs | elapsed: %.2fs | overrun: %.2fs",
                    timeout,
                    elapsed,
                    overrun,
                )
                return res
            logger.warning(f"函数调用超时（{timeout}秒）")
            err = TimeoutError(f"执行超时（{timeout}秒）")
            setattr(err, "partial_result", res)
            raise err
        return res

    def _build_astro_fallback_output(self, steps, query: str) -> str | None:
        """当总结阶段失败或超时时，从占星工具结果生成兜底摘要。"""
        if not steps:
            return None

        astro_tools = {"astro_natal_chart", "astro_current_chart", "astro_transit_chart", "astro_my_sign", "xingpan"}
        for step in reversed(steps):
            try:
                action, observation = step
                tool_name = getattr(action, "tool", "")
                if tool_name not in astro_tools:
                    continue

                raw_text = observation if isinstance(observation, str) else str(observation)
                payload = json.loads(raw_text)
                if not isinstance(payload, dict) or not payload.get("ok"):
                    continue

                data = payload.get("data")
                core = data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), dict) else data

                sign_names = []
                if isinstance(core, dict) and isinstance(core.get("sign"), list):
                    for item in core.get("sign", [])[:3]:
                        if isinstance(item, dict):
                            cn = item.get("sign_cn") or item.get("sign_en")
                            if cn:
                                sign_names.append(str(cn))

                sign_text = "、".join(sign_names) if sign_names else "暂无可提取星座名"
                snippet = json.dumps(core, ensure_ascii=False)[:500] if core is not None else json.dumps(data, ensure_ascii=False)[:500]

                return (
                    "已成功调用占星工具并拿到结果，但总结模型本轮响应异常，先返回核心信息：\n"
                    f"- 相关星座：{sign_text}\n"
                    f"- 查询问题：{query[:60]}\n"
                    f"- 原始结果片段：{snippet}"
                )
            except Exception:
                continue

        return None

    @staticmethod
    def _is_bad_astro_output(output: str) -> bool:
        """识别占星场景下的降级短回复。"""
        text = (output or "").strip()
        if not text:
            return True
        if len(text) <= 80:
            return True
        bad_markers = ["服务异常", "请稍后再试", "超时", "服务不可用", "开小差"]
        return any(m in text for m in bad_markers)

    def mood_chain(self, query: str, timeout=5):
        """情绪识别链"""
        start_time = time.perf_counter()
        rule_hit = self._rule_based_mood(query)
        if rule_hit:
            elapsed = time.perf_counter() - start_time
            logger.info(f"情绪识别完成 | 用户输入: {query[:50]} | 识别结果: {rule_hit} | 耗时: {elapsed:.2f}秒 | 来源: rule")
            return rule_hit

        try:
            chain = ChatPromptTemplate.from_template(MOOD_CLASSIFY_PROMPT) | self.mood_llm | StrOutputParser()
            r = self._invoke_with_timeout(lambda: chain.invoke({"query": query}), timeout).strip().lower()
            mood = r if r in MOODS else "default"
            elapsed = time.perf_counter() - start_time
            logger.info(f"情绪识别完成 | 用户输入: {query[:50]} | 识别结果: {mood} | 耗时: {elapsed:.2f}秒")
            return mood
        except TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.warning(f"情绪识别超时 | 用户输入: {query[:50]} | 超时: {timeout}秒 | 已耗时: {elapsed:.2f}秒")
            return "default"
        except Exception as e:
            logger.error(f"情绪识别异常 | 用户输入: {query[:50]} | 错误: {str(e)[:100]}", exc_info=True)
            return "default"

    def _rule_based_mood(self, query: str) -> str | None:
        """规则优先：命中强情绪关键词则直接返回"""
        text = (query or "").strip().lower()
        if not text:
            return "default"

        mood_mapping = [
            ("angry", ["气死", "生气", "愤怒", "垃圾", "骗子", "滚", "破", "怒", "不爽"]),
            ("depressed", ["绝望", "活着好累", "想死", "抑郁", "没动力", "无助", "崩溃"]),
            ("upset", ["难过", "委屈", "伤心", "失恋", "心情差", "沮丧"]),
            ("upbeat", ["太开心", "中奖了", "超激动", "好兴奋", "太棒了", "好爽"]),
            ("cheerful", ["心情不错", "挺开心", "很愉快", "好满足", "天气真好"]),
        ]
        mood = self._match_keywords(text, mood_mapping, "")
        if mood:
            return mood

        if text in ["你好", "您好", "在吗", "谢谢", "麻烦了"]:
            return "friendly"

        return None

    def run(self, query, timeout=60, session_id: str = "default"):
        """主运行方法"""
        st = time.time()
        try:
            sid = self._normalize_session_id(session_id)
            intent = self._route_intent(query)
            astro_intents = {
                "astro_my_sign",
                "astro_natal_chart",
                "astro_current_chart",
                "astro_transit_chart",
                "xingpan",
            }
            effective_timeout = max(timeout, 90) if intent in astro_intents else timeout
            logger.info(f"开始处理用户请求 | session_id: {sid} | 查询内容: {query[:100]} | 超时时间: {effective_timeout}秒")
            
            # 情绪识别
            motion = self.mood_chain(query, timeout=self.mood_timeout_seconds)

            # 每次请求使用独立的prompt、memory与agent，避免并发串话
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(SYSTEMPL.format(who_you_are=MOODS[motion]["roleSet"])),
                MessagesPlaceholder("history"),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            
            memory = self._build_memory(sid)
            tools = self._select_tools_by_intent(query, intent=intent)
            max_iterations = 2 if intent in astro_intents else 3
            agent = create_openai_tools_agent(self.normal_llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=not IS_PROD,
                handle_parsing_errors=lambda e: USER_MESSAGES["parse_error"],
                max_iterations=max_iterations,
                early_stopping_method="force",
                return_intermediate_steps=True,
                return_only_outputs=True,
            )

            # 计算剩余超时时间
            remain = effective_timeout - (time.time() - st)
            if remain <= 0:
                logger.warning(f"请求处理超时 | 查询内容: {query[:50]} | 已耗时: {time.time()-st:.2f}秒")
                return {"output": USER_MESSAGES["timeout_response"]}

            # 执行Agent
            agent_start = time.perf_counter()
            try:
                result = self._invoke_with_timeout(lambda: agent_executor.invoke({"input": query}), timeout=remain)
                steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
                if steps:
                    for i, step in enumerate(steps, start=1):
                        try:
                            action, observation = step
                            tool_name = getattr(action, "tool", "unknown")
                            tool_input = str(getattr(action, "tool_input", ""))[:200]
                            obs_preview = str(observation)[:200]
                            logger.info(
                                "工具调用轨迹 | step: %s | tool: %s | input: %s | output: %s",
                                i,
                                tool_name,
                                tool_input,
                                obs_preview,
                            )
                        except Exception as trace_err:
                            logger.warning("工具调用轨迹解析失败 | step: %s | error: %s", i, str(trace_err)[:100])
                else:
                    logger.info("工具调用轨迹 | 本轮未发生工具调用")

                if intent in astro_intents:
                    final_output = result.get("output", "") if isinstance(result, dict) else ""
                    if self._is_bad_astro_output(final_output):
                        fallback_output = self._build_astro_fallback_output(steps, query)
                        if fallback_output:
                            logger.warning("占星总结降级触发兜底 | 使用工具结果摘要替代短错误回复")
                            result["output"] = fallback_output

                agent_elapsed = time.perf_counter() - agent_start
                logger.info(f"Agent调用完成 | 查询内容: {query[:50]} | 耗时: {agent_elapsed:.2f}秒")
            except TimeoutError as e:
                agent_elapsed = time.perf_counter() - agent_start
                logger.warning(f"Agent调用超时 | 查询内容: {query[:50]} | 超时: {remain:.2f}秒 | 已耗时: {agent_elapsed:.2f}秒")
                partial = getattr(e, "partial_result", None)
                if intent in astro_intents and isinstance(partial, dict):
                    steps = partial.get("intermediate_steps", [])
                    fallback_output = self._build_astro_fallback_output(steps, query)
                    if fallback_output:
                        logger.warning("占星超时触发兜底 | 返回工具结果摘要")
                        partial["output"] = fallback_output
                        return partial
                raise
            
            total_time = time.time() - st
            logger.info(f"请求处理完成 | session_id: {sid} | 查询内容: {query[:50]} | 耗时: {total_time:.2f}秒")
            return result
            
        except Exception as e:
            error_msg = str(e)[:100]
            logger.error(f"run方法执行异常 | session_id: {session_id} | 查询内容: {query[:50]} | 错误: {error_msg}", exc_info=True)
            return {"output": f"服务异常：{error_msg}"}

    async def run_async(self, query, timeout=60, session_id: str = "default"):
        """异步封装"""
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            res = await loop.run_in_executor(None, self.run, query, timeout, session_id)
            return res
        except Exception as e:
            logger.error(f"异步调用异常 | session_id: {session_id} | 查询内容: {query[:50]} | 错误: {str(e)[:100]}", exc_info=True)
            return {"output": "服务异常，请稍后再试"}

    @staticmethod
    def _extract_preview_text(response) -> str:
        """提取健康检查响应预览文本。"""
        if isinstance(response, str):
            return response

        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)

        return str(content)

    def check_llm_health(self) -> bool:
        """检查LLM健康状态"""
        try:
            test_prompt = "健康检查"
            response = self.normal_llm.invoke(test_prompt)
            preview = self._extract_preview_text(response)

            logger.info(f"LLM健康检查通过 | 响应: {preview[:50]}")
            return True
        except Exception as e:
            logger.error(f"LLM健康检查失败 | 错误: {str(e)[:100]}", exc_info=True)
            return False

    def get_memory_status(self, session_id: str) -> dict:
        """返回指定session的memory状态，用于联调观察。"""
        sid = self._normalize_session_id(session_id)
        chat_history, backend = self._get_chat_history(sid)
        messages = list(chat_history.messages)
        msg_count = len(messages)
        threshold = max(1, self.memory_compact_message_count)

        latest_preview = ""
        if messages:
            latest_preview = self._extract_preview_text(messages[-1])[:120]

        return {
            "session_id": sid,
            "memory_backend": backend,
            "message_count": msg_count,
            "compact_threshold": threshold,
            "will_compact_on_next_request": msg_count >= threshold,
            "latest_message_preview": latest_preview,
        }

# 全局单例
master = Master()

# ===================== API接口 =====================
@app.get("/")
def read_root():
    logger.info("访问根路径")
    return {"Hello": "World"}

@app.post("/chat")
def chat(query: str, session_id: str = "default"):
    logger.info(f"接收Chat API请求 | session_id: {session_id} | 查询: {query[:100]}")
    try:
        res = master.run(query, session_id=session_id)
        response_text = res.get("output", "")
        logger.info(f"Chat API响应成功 | session_id: {session_id} | 查询: {query[:50]} | 响应长度: {len(response_text)}")
        return {"code": 200, "session_id": session_id, "query": query, "response": response_text}
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"Chat API执行异常 | 查询: {query[:50]} | 错误: {error_msg}", exc_info=True)
        return {"code": 500, "session_id": session_id, "query": query, "response": f"错误：{error_msg}"}

@app.post("/add_urls")
def add_urls():
    logger.info("调用add_urls接口")
    return {"response": "URLs added!"}

@app.post("/add_pdfs")
def add_pdfs():
    logger.info("调用add_pdfs接口")
    return {"response": "PDFs added!"}

@app.post("/add_texts")
def add_texts():
    logger.info("调用add_texts接口")
    return {"response": "Texts added!"}

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        import asyncio

        llm_ok = await asyncio.to_thread(master.check_llm_health)
        llm_status = "ok" if llm_ok else "error"
        health_info = {
            "status": "healthy",
            "timestamp": time.time(),
            "llm_status": llm_status,
            "env": "production" if IS_PROD else "development"
        }
        logger.info(f"健康检查 | 状态: {health_info}")
        return health_info
    except Exception as e:
        logger.error(f"健康检查异常 | 错误: {str(e)[:100]}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)[:100]
        }

@app.get("/memory/status")
def memory_status(session_id: str = "default"):
    """查看指定session的memory状态。"""
    try:
        status = master.get_memory_status(session_id)
        logger.info("memory状态查询成功 | session_id: %s | count: %s", status["session_id"], status["message_count"])
        return {"code": 200, "data": status}
    except Exception as e:
        err = str(e)[:120]
        logger.error("memory状态查询失败 | session_id: %s | err: %s", session_id, err, exc_info=True)
        return {"code": 500, "session_id": session_id, "error": err}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """WebSocket接口"""
    await websocket.accept()
    client_ip = websocket.client.host
    session_id = websocket.query_params.get("session_id") or f"ws_{client_ip}_{int(time.time())}"
    logger.info(f"WebSocket连接建立 | 客户端IP: {client_ip} | session_id: {session_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"WebSocket接收消息 | 客户端IP: {client_ip} | session_id: {session_id} | 消息: {data[:100]}")
            
            res = await master.run_async(data, session_id=session_id)
            clean_response = res.get("output", USER_MESSAGES["ws_default"])
            
            await websocket.send_text(clean_response)
            logger.info(f"WebSocket发送消息 | 客户端IP: {client_ip} | 响应长度: {len(clean_response)}")
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开 | 客户端IP: {client_ip}")
    except Exception as e:
        logger.error(f"WebSocket异常 | 客户端IP: {client_ip} | 错误: {str(e)[:100]}", exc_info=True)
        await websocket.close(code=1011, reason="服务器内部错误")

if __name__ == "__main__":
    import uvicorn
    logger.info("启动FastAPI服务 | 地址: 127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)