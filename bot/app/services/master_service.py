import json
import os
import re
import threading
import time

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import StrOutputParser
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

from app.tools.mytools import (
    astro_current_chart,
    astro_my_sign,
    astro_natal_chart,
    astro_transit_chart,
    search,
    test,
    vector_search,
    xingpan,
)
from app.core.config import IS_PROD
from app.llm.custom_llm import CustomProxyLLM, _thread_ctx
from app.core.logger_setup import logger
from app.core.texts import MOOD_CLASSIFY_PROMPT, MOODS, SYSTEMPL, USER_MESSAGES


class Master:
    @staticmethod
    def _build_redis_url_from_env() -> str:
        """构建Redis连接地址，优先兼容REDIS_*分项配置。"""
        host = os.getenv("REDIS_HOST", "127.0.0.1").strip() or "127.0.0.1"
        port = os.getenv("REDIS_PORT", "6379").strip() or "6379"
        db = os.getenv("REDIS_DB", "0").strip() or "0"
        password = os.getenv("REDIS_PASSWORD", "")

        # URL中若无密码，格式为 redis://host:port/db。
        if not password:
            return f"redis://{host}:{port}/{db}"

        # Redis URL规范：带密码时使用 redis://:password@host:port/db。
        return f"redis://:{password}@{host}:{port}/{db}"

    def __init__(self):
        self.normal_llm = CustomProxyLLM(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", ""),
            model=os.getenv("OPENAI_MODEL", ""),
        )

        self.mood_llm = CustomProxyLLM(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", ""),
            model=os.getenv("OPENAI_MODEL", ""),
            temperature=0.0,
            max_tokens=12,
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

        self.redis_url = os.getenv("REDIS_URL", "").strip() or self._build_redis_url_from_env()
        self.memory_ttl = int(os.getenv("MEMORY_TTL", "86400"))
        self.memory_compact_message_count = int(os.getenv("MEMORY_COMPACT_MESSAGE_COUNT", "10"))
        self.mood_timeout_seconds = float(os.getenv("MOOD_TIMEOUT_SECONDS", "5"))
        self._local_histories: dict[str, InMemoryChatMessageHistory] = {}
        self._local_histories_lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()

        logger.info("Master实例初始化完成")

    @staticmethod
    def _normalize_session_id(session_id: str | None) -> str:
        sid = (session_id or "").strip()
        if not sid:
            raise ValueError("session_id不能为空")
        return sid

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    @staticmethod
    def _history_messages_to_text(messages: list[BaseMessage]) -> str:
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
    def _extract_user_facts_from_messages(messages: list[BaseMessage]) -> dict[str, str]:
        facts: dict[str, str] = {}
        text_parts: list[str] = []

        for msg in messages:
            role = getattr(msg, "type", "")
            if role not in {"human", "user"}:
                continue

            content = getattr(msg, "content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                    elif isinstance(item, str):
                        text_parts.append(item)

        if not text_parts:
            return facts

        merged = "\n".join(text_parts)

        patterns = {
            "姓名": [r"我叫\s*([\u4e00-\u9fa5A-Za-z0-9_]{2,20})", r"名字是\s*([\u4e00-\u9fa5A-Za-z0-9_]{2,20})"],
            "年龄": [r"我(?:今年|现在)?\s*(\d{1,3})\s*岁"],
            "性别": [r"我是\s*(男生|女生|男性|女性|男|女)", r"性别\s*[是:：]?\s*(男生|女生|男性|女性|男|女)"],
            "生日": [r"(?:生日|出生日期)\s*[是:：]?\s*([0-9]{4}[年\-/\.][0-9]{1,2}[月\-/\.][0-9]{1,2}日?)"],
            "出生地": [r"(?:我来自|我是)\s*([\u4e00-\u9fa5]{2,15})人", r"(?:出生在|老家在|来自)\s*([\u4e00-\u9fa5]{2,20})"],
            "常驻地": [r"(?:我在|我住在|目前在)\s*([\u4e00-\u9fa5A-Za-z0-9\-\s]{2,30})"],
        }

        for key, regex_list in patterns.items():
            for pattern in regex_list:
                match = re.search(pattern, merged)
                if match:
                    value = match.group(1).strip()
                    if value:
                        facts[key] = value
                        break

        pref = re.search(r"我(?:比较)?喜欢\s*([^。！？\n]{1,30})", merged)
        if pref:
            facts["偏好"] = pref.group(1).strip()

        return facts

    @staticmethod
    def _format_user_facts(facts: dict[str, str]) -> str:
        if not facts:
            return "暂无明确用户信息"
        ordered_keys = ["姓名", "年龄", "性别", "生日", "出生地", "常驻地", "偏好"]
        parts = []
        for key in ordered_keys:
            value = facts.get(key)
            if value:
                parts.append(f"{key}:{value}")
        return "；".join(parts) if parts else "暂无明确用户信息"

    @staticmethod
    def _append_summary_message(chat_history, summary_text: str) -> None:
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
        with self._local_histories_lock:
            history = self._local_histories.get(session_id)
            if history is None:
                history = InMemoryChatMessageHistory()
                self._local_histories[session_id] = history
            return history

    def _get_chat_history(self, session_id: str):
        sid = self._normalize_session_id(session_id)

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

            chat_history = RedisChatMessageHistory(session_id=sid)
            _ = chat_history.messages
            return chat_history, "redis"
        except Exception as e:
            logger.warning("Redis memory不可用，降级到进程内memory | session_id: %s | err: %s", sid, str(e)[:120])
            return self._get_or_create_local_history(sid), "in_memory"

    def _compact_history_if_needed(self, chat_history, session_id: str) -> None:
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

        user_facts = self._extract_user_facts_from_messages(messages)
        facts_text = self._format_user_facts(user_facts)

        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是对话记忆压缩助手。请将对话压缩为一条简洁摘要，保留用户事实信息、偏好、约束、已完成事项和未完成事项；"
                "避免冗余和寒暄。请严格输出两段：\n"
                "用户关键信息：<仅保留可确认事实，未知写无>\n"
                "摘要：<对话摘要>",
            ),
            (
                "user",
                "请压缩以下历史对话，并确保保留用户信息。\n"
                "规则抽取到的用户信息：{facts_text}\n"
                "历史对话：\n{history_text}",
            ),
        ])

        summary_chain = summary_prompt | self.normal_llm | StrOutputParser()
        summary_text = ""
        try:
            summary_text = self._invoke_with_timeout(
                lambda: summary_chain.invoke({"history_text": history_text, "facts_text": facts_text}),
                timeout=15,
            ).strip()
        except Exception as e:
            logger.warning("历史压缩失败，回退截断摘要 | session_id: %s | err: %s", session_id, str(e)[:120])

        if not summary_text:
            summary_text = history_text[-1000:]

        summary_text = f"用户关键信息：{facts_text}\n摘要：{summary_text}"

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
        for label, keywords in mapping:
            if any(k in text for k in keywords):
                return label
        return default

    def _route_intent(self, query: str) -> str:
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
        start = time.perf_counter()
        res = None
        try:
            _thread_ctx.request_timeout = timeout
            _thread_ctx.request_deadline = start + timeout
            res = func()
        except Exception as e:
            logger.error("超时调用函数执行异常: %s", str(e)[:100], exc_info=True)
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
            logger.warning("函数调用超时（%s秒）", timeout)
            err = TimeoutError(f"执行超时（{timeout}秒）")
            setattr(err, "partial_result", res)
            raise err
        return res

    def _build_astro_fallback_output(self, steps, query: str) -> str | None:
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
        text = (output or "").strip()
        if not text:
            return True
        if len(text) <= 80:
            return True
        bad_markers = ["服务异常", "请稍后再试", "超时", "服务不可用", "开小差"]
        return any(m in text for m in bad_markers)

    def mood_chain(self, query: str, timeout=5):
        start_time = time.perf_counter()
        rule_hit = self._rule_based_mood(query)
        if rule_hit:
            elapsed = time.perf_counter() - start_time
            logger.info("情绪识别完成 | 用户输入: %s | 识别结果: %s | 耗时: %.2f秒 | 来源: rule", query[:50], rule_hit, elapsed)
            return rule_hit

        try:
            chain = ChatPromptTemplate.from_template(MOOD_CLASSIFY_PROMPT) | self.mood_llm | StrOutputParser()
            r = self._invoke_with_timeout(lambda: chain.invoke({"query": query}), timeout).strip().lower()
            mood = r if r in MOODS else "default"
            elapsed = time.perf_counter() - start_time
            logger.info("情绪识别完成 | 用户输入: %s | 识别结果: %s | 耗时: %.2f秒", query[:50], mood, elapsed)
            return mood
        except TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.warning("情绪识别超时 | 用户输入: %s | 超时: %s秒 | 已耗时: %.2f秒", query[:50], timeout, elapsed)
            return "default"
        except Exception as e:
            logger.error("情绪识别异常 | 用户输入: %s | 错误: %s", query[:50], str(e)[:100], exc_info=True)
            return "default"

    def _rule_based_mood(self, query: str) -> str | None:
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

    def run(self, query, timeout=60, session_id: str | None = None):
        st = time.time()
        try:
            sid = self._normalize_session_id(session_id)
            with self._get_session_lock(sid):
                intent = self._route_intent(query)
                astro_intents = {
                    "astro_my_sign",
                    "astro_natal_chart",
                    "astro_current_chart",
                    "astro_transit_chart",
                    "xingpan",
                }
                effective_timeout = max(timeout, 90) if intent in astro_intents else timeout
                logger.info("开始处理用户请求 | session_id: %s | 查询内容: %s | 超时时间: %s秒", sid, query[:100], effective_timeout)

                motion = self.mood_chain(query, timeout=self.mood_timeout_seconds)

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

                remain = effective_timeout - (time.time() - st)
                if remain <= 0:
                    logger.warning("请求处理超时 | 查询内容: %s | 已耗时: %.2f秒", query[:50], time.time() - st)
                    return {"output": USER_MESSAGES["timeout_response"]}

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
                    logger.info("Agent调用完成 | 查询内容: %s | 耗时: %.2f秒", query[:50], agent_elapsed)
                except TimeoutError as e:
                    agent_elapsed = time.perf_counter() - agent_start
                    logger.warning("Agent调用超时 | 查询内容: %s | 超时: %.2f秒 | 已耗时: %.2f秒", query[:50], remain, agent_elapsed)
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
                logger.info("请求处理完成 | session_id: %s | 查询内容: %s | 耗时: %.2f秒", sid, query[:50], total_time)
                return result

        except Exception as e:
            error_msg = str(e)[:100]
            logger.error("run方法执行异常 | session_id: %s | 查询内容: %s | 错误: %s", session_id, query[:50], error_msg, exc_info=True)
            return {"output": f"服务异常：{error_msg}"}

    async def run_async(self, query, timeout=60, session_id: str | None = None):
        import asyncio

        loop = asyncio.get_running_loop()
        try:
            res = await loop.run_in_executor(None, self.run, query, timeout, session_id)
            return res
        except Exception as e:
            logger.error("异步调用异常 | session_id: %s | 查询内容: %s | 错误: %s", session_id, query[:50], str(e)[:100], exc_info=True)
            return {"output": "服务异常，请稍后再试"}

    @staticmethod
    def _extract_preview_text(response) -> str:
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
        try:
            test_prompt = "健康检查"
            response = self.normal_llm.invoke(test_prompt)
            preview = self._extract_preview_text(response)

            logger.info("LLM健康检查通过 | 响应: %s", preview[:50])
            return True
        except Exception as e:
            logger.error("LLM健康检查失败 | 错误: %s", str(e)[:100], exc_info=True)
            return False

    def get_memory_status(self, session_id: str) -> dict:
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
