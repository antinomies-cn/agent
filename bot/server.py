import os
import time
import logging
import logging.handlers
import requests
import threading
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.schema import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
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

# ===================== 工具定义 =====================
@tool
def test():
    """Test tool"""
    logger.info("执行测试工具")
    return "test"

@tool
def search(query: str) -> str:
    """搜索工具:只有当你需要获取最新信息或者查询事实时才使用这个工具。"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    logger.info(f"执行搜索工具 | 查询: {query[:50]} | 结果: {result[:100]}")
    return result

@tool
def vector_search(query: str) -> str:
    """向量搜索工具:只有当你需要从已知文档中查询信息时才使用这个工具。"""
    client = Qdrant(
        QdrantClient(path = "./qdrant_data/qdrant.db"),
        "divination_master_collection",
        OpenAIEmbeddings(),
    )
    retriever = client.as_retriever(search_type="mmr")
    docs = retriever.get_relevant_documents(query)
    result = "\n\n".join([doc.page_content for doc in docs])
    logger.info(f"执行向量搜索工具 | 查询: {query[:50]} | 结果长度: {len(result)}")
    return result

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

    def _request_completion(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs
    ) -> str:
        # 参数校验
        try:
            call_start = time.perf_counter()
            if not self.api_key:
                error_msg = "模型密钥未配置"
                logger.error(f"LLM配置错误: {error_msg}")
                return USER_MESSAGES["config"]["api_key"]
            if not self.base_url:
                error_msg = "模型接口地址未配置"
                logger.error(f"LLM配置错误: {error_msg}")
                return USER_MESSAGES["config"]["base_url"]
            if not self.model:
                error_msg = "模型名称未配置"
                logger.error(f"LLM配置错误: {error_msg}")
                return USER_MESSAGES["config"]["model"]

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "LangChain-CustomLLM/1.0"
            }

            proxy_messages = []
            for m in messages:
                role = self._to_proxy_role(m)
                proxy_messages.append({"role": role, "content": m.content})

            data = {
                "model": self.model,
                "messages": proxy_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
            
            # 开发环境打印调试信息
            if not IS_PROD:
                logger.debug(f"LLM请求URL: {url}")
                logger.debug(f"LLM请求体: {data}")

            request_timeout = getattr(_thread_ctx, "request_timeout", None)
            try:
                timeout_seconds = float(request_timeout) if request_timeout is not None else 60.0
            except (TypeError, ValueError):
                timeout_seconds = 60.0
            # 最小超时保护，避免传入0或负数导致立即失败
            if timeout_seconds <= 0:
                timeout_seconds = 1.0

            resp = requests.post(url, headers=headers, json=data, timeout=timeout_seconds)
            resp.raise_for_status()
            
            if not IS_PROD:
                logger.debug(f"LLM响应状态码: {resp.status_code}")
                logger.debug(f"LLM响应内容: {resp.text[:500]}")

            result = resp.json()
            content = self._extract_content(result)
            elapsed = time.perf_counter() - call_start
            logger.info(f"LLM调用成功，响应长度: {len(content)} | 耗时: {elapsed:.2f}秒")
            return content

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
            return user_msg
            
        except requests.exceptions.Timeout:
            request_timeout = getattr(_thread_ctx, "request_timeout", None)
            try:
                timeout_seconds = float(request_timeout) if request_timeout is not None else 60.0
            except (TypeError, ValueError):
                timeout_seconds = 60.0
            error_info = {"type": "TIMEOUT", "timeout": timeout_seconds}
            logger.error(f"LLM调用超时（{timeout_seconds}秒） | 详细信息: {error_info}")
            return USER_MESSAGES["timeout"]
            
        except requests.exceptions.ConnectionError:
            error_info = {"type": "CONNECTION_ERROR", "url": url if 'url' in locals() else ""}
            logger.error(f"LLM调用失败：网络连接异常 | 详细信息: {error_info}")
            return USER_MESSAGES["connection_error"]
            
        except KeyError as e:
            response_text = resp.text[:200] if 'resp' in locals() else ""
            error_info = {"type": "KEY_ERROR", "missing_key": str(e), "response_text": response_text}
            logger.error(f"LLM响应解析失败：缺失字段 {e} | 详细信息: {error_info}")
            return USER_MESSAGES["key_error"]
            
        except Exception as e:
            error_info = {"type": "UNKNOWN_ERROR", "error": str(e)[:100]}
            logger.error(f"LLM未知异常 | 详细信息: {error_info}", exc_info=True)
            return USER_MESSAGES["unknown_error"]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs
    ) -> ChatResult:
        content = self._request_completion(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        generation = ChatGeneration(message=AIMessage(content=content))
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

        self.MOTION = "default"

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEMPL.format(who_you_are=MOODS["default"]["roleSet"])),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        self.memory = ""  # 后续可替换为真正的内存模块
        tools = [search,test]
        agent = create_openai_tools_agent(
            self.normal_llm, 
            tools, 
            self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=not IS_PROD,  # 生产环境关闭verbose
            handle_parsing_errors=lambda e: USER_MESSAGES["parse_error"],
            max_iterations=3,
            early_stopping_method="force",
            return_intermediate_steps=False,
            return_only_outputs=True,
        )
        logger.info("Master实例初始化完成")

    def _invoke_with_timeout(self, func, timeout):
        """带超时的函数调用"""
        res = None
        exc = None

        def run():
            nonlocal res, exc
            try:
                _thread_ctx.request_timeout = timeout
                res = func()
            except Exception as e:
                exc = e
                logger.error(f"超时调用函数执行异常: {str(e)[:100]}", exc_info=True)
            finally:
                if hasattr(_thread_ctx, "request_timeout"):
                    delattr(_thread_ctx, "request_timeout")

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()
        t.join(timeout)
        
        if t.is_alive():
            logger.warning(f"函数调用超时（{timeout}秒）")
            raise TimeoutError(f"执行超时（{timeout}秒）")
        if exc:
            raise exc
        return res

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

        angry_words = ["气死", "生气", "愤怒", "垃圾", "骗子", "滚", "破", "怒", "不爽"]
        depressed_words = ["绝望", "活着好累", "想死", "抑郁", "没动力", "无助", "崩溃"]
        upset_words = ["难过", "委屈", "伤心", "失恋", "心情差", "沮丧"]
        upbeat_words = ["太开心", "中奖了", "超激动", "好兴奋", "太棒了", "好爽"]
        cheerful_words = ["心情不错", "挺开心", "很愉快", "好满足", "天气真好"]

        for w in angry_words:
            if w in text:
                return "angry"
        for w in depressed_words:
            if w in text:
                return "depressed"
        for w in upset_words:
            if w in text:
                return "upset"
        for w in upbeat_words:
            if w in text:
                return "upbeat"
        for w in cheerful_words:
            if w in text:
                return "cheerful"

        if text in ["你好", "您好", "在吗", "谢谢", "麻烦了"]:
            return "friendly"

        return None

    def run(self, query, timeout=60):
        """主运行方法"""
        st = time.time()
        try:
            logger.info(f"开始处理用户请求 | 查询内容: {query[:100]} | 超时时间: {timeout}秒")
            
            # 情绪识别
            motion = self.mood_chain(query, timeout=3)

            # 每次请求使用独立的prompt与agent，避免并发串话
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(SYSTEMPL.format(who_you_are=MOODS[motion]["roleSet"])),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            
            tools = [search,test]
            agent = create_openai_tools_agent(self.normal_llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=not IS_PROD,
                handle_parsing_errors=lambda e: USER_MESSAGES["parse_error"],
                max_iterations=3,
                early_stopping_method="force",
                return_intermediate_steps=False,
                return_only_outputs=True,
            )

            # 计算剩余超时时间
            remain = timeout - (time.time() - st)
            if remain <= 0:
                logger.warning(f"请求处理超时 | 查询内容: {query[:50]} | 已耗时: {time.time()-st:.2f}秒")
                return {"output": USER_MESSAGES["timeout_response"]}

            # 执行Agent
            agent_start = time.perf_counter()
            try:
                result = self._invoke_with_timeout(
                    lambda: agent_executor.invoke({"input": query}),
                    timeout=remain
                )
                agent_elapsed = time.perf_counter() - agent_start
                logger.info(f"Agent调用完成 | 查询内容: {query[:50]} | 耗时: {agent_elapsed:.2f}秒")
            except TimeoutError:
                agent_elapsed = time.perf_counter() - agent_start
                logger.warning(f"Agent调用超时 | 查询内容: {query[:50]} | 超时: {remain:.2f}秒 | 已耗时: {agent_elapsed:.2f}秒")
                raise
            
            total_time = time.time() - st
            logger.info(f"请求处理完成 | 查询内容: {query[:50]} | 耗时: {total_time:.2f}秒")
            return result
            
        except Exception as e:
            error_msg = str(e)[:100]
            logger.error(f"run方法执行异常 | 查询内容: {query[:50]} | 错误: {error_msg}", exc_info=True)
            return {"output": f"服务异常：{error_msg}"}

    async def run_async(self, query, timeout=60):
        """异步封装"""
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            res = await loop.run_in_executor(None, self.run, query, timeout)
            return res
        except Exception as e:
            logger.error(f"异步调用异常 | 查询内容: {query[:50]} | 错误: {str(e)[:100]}", exc_info=True)
            return {"output": "服务异常，请稍后再试"}

    def check_llm_health(self) -> bool:
        """检查LLM健康状态"""
        try:
            test_prompt = "健康检查"
            response = self.normal_llm.invoke(test_prompt)

            if isinstance(response, str):
                preview = response
            else:
                content = getattr(response, "content", response)
                if isinstance(content, str):
                    preview = content
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                        elif isinstance(item, str):
                            text_parts.append(item)
                    preview = "".join(text_parts)
                else:
                    preview = str(content)

            logger.info(f"LLM健康检查通过 | 响应: {preview[:50]}")
            return True
        except Exception as e:
            logger.error(f"LLM健康检查失败 | 错误: {str(e)[:100]}", exc_info=True)
            return False

# 全局单例
master = Master()

# ===================== API接口 =====================
@app.get("/")
def read_root():
    logger.info("访问根路径")
    return {"Hello": "World"}

@app.post("/chat")
def chat(query: str):
    logger.info(f"接收Chat API请求 | 查询: {query[:100]}")
    try:
        res = master.run(query)
        response_text = res.get("output", "")
        logger.info(f"Chat API响应成功 | 查询: {query[:50]} | 响应长度: {len(response_text)}")
        return {"code": 200, "query": query, "response": response_text}
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"Chat API执行异常 | 查询: {query[:50]} | 错误: {error_msg}", exc_info=True)
        return {"code": 500, "query": query, "response": f"错误：{error_msg}"}

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

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """WebSocket接口"""
    await websocket.accept()
    client_ip = websocket.client.host
    logger.info(f"WebSocket连接建立 | 客户端IP: {client_ip}")
    
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"WebSocket接收消息 | 客户端IP: {client_ip} | 消息: {data[:100]}")
            
            res = await master.run_async(data)
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