import os
import time
import threading
import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.config import IS_PROD
from app.core.logger_setup import logger
from app.core.texts import USER_MESSAGES

# 线程本地上下文：为LLM请求传递每次调用的超时。
_thread_ctx = threading.local()


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

    def _request_completion(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> tuple[str, list[dict]]:
        try:
            call_start = time.perf_counter()
            required_config = [
                ("api_key", self.api_key, "模型密钥未配置"),
                ("base_url", self.base_url, "模型接口地址未配置"),
                ("model", self.model, "模型名称未配置"),
            ]
            for config_key, config_value, error_msg in required_config:
                if not config_value:
                    logger.error("LLM配置错误: %s", error_msg)
                    return USER_MESSAGES["config"][config_key], []

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "LangChain-CustomLLM/1.0",
            }

            proxy_messages = []
            for m in messages:
                role = self._to_proxy_role(m)
                msg = {"role": role, "content": m.content}

                if role == "assistant":
                    assistant_tool_calls = getattr(m, "additional_kwargs", {}).get("tool_calls")
                    if isinstance(assistant_tool_calls, list) and assistant_tool_calls:
                        msg["tool_calls"] = assistant_tool_calls

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
                "stream": False,
            }

            for key in ["tools", "tool_choice", "parallel_tool_calls", "response_format"]:
                value = kwargs.get(key)
                if value is not None:
                    data[key] = value
            if stop:
                data["stop"] = stop

            url = f"{self.base_url.rstrip('/')}/v1/chat/completions"

            if not IS_PROD:
                logger.debug("LLM请求URL: %s", url)
                logger.debug("LLM请求体: %s", data)

            timeout_seconds = self._resolve_timeout_seconds()

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
                logger.debug("LLM响应状态码: %s", resp.status_code)
                logger.debug("LLM响应内容: %s", resp.text[:500])

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
            status_code = resp.status_code if "resp" in locals() else 0
            response_text = resp.text[:200] if "resp" in locals() else ""

            error_info = {
                "type": "HTTP_ERROR",
                "status_code": status_code,
                "response_text": response_text,
                "error": str(e)[:100],
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

            logger.error("%s | 详细信息: %s", error_log, error_info)
            return user_msg, []

        except requests.exceptions.Timeout:
            timeout_seconds = self._resolve_timeout_seconds()
            error_info = {"type": "TIMEOUT", "timeout": timeout_seconds}
            logger.error("LLM调用超时（%s秒） | 详细信息: %s", timeout_seconds, error_info)
            return USER_MESSAGES["timeout"], []

        except requests.exceptions.ConnectionError:
            error_info = {"type": "CONNECTION_ERROR", "url": url if "url" in locals() else ""}
            logger.error("LLM调用失败：网络连接异常 | 详细信息: %s", error_info)
            return USER_MESSAGES["connection_error"], []

        except KeyError as e:
            response_text = resp.text[:200] if "resp" in locals() else ""
            error_info = {"type": "KEY_ERROR", "missing_key": str(e), "response_text": response_text}
            logger.error("LLM响应解析失败：缺失字段 %s | 详细信息: %s", e, error_info)
            return USER_MESSAGES["key_error"], []

        except Exception as e:
            error_info = {"type": "UNKNOWN_ERROR", "error": str(e)[:100]}
            logger.error("LLM未知异常 | 详细信息: %s", error_info, exc_info=True)
            return USER_MESSAGES["unknown_error"], []

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        content, tool_calls = self._request_completion(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}
        generation = ChatGeneration(message=AIMessage(content=content, additional_kwargs=additional_kwargs))
        return ChatResult(generations=[generation])
