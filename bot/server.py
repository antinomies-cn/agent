import os
import time
import requests
import threading
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.schema import StrOutputParser

load_dotenv()

app = FastAPI()

# 工具
@tool
def test():
    """Test tool"""
    return "test"

# 新增：区分开发/生产环境（控制日志输出）
IS_PROD = os.getenv("ENV", "dev") == "prod"

# 自定义 LLM
class CustomProxyLLM(SimpleChatModel):
    api_key: str
    base_url: str
    model: str
    # 新增：将关键生成参数设为可配置属性，默认值保留原有逻辑
    temperature: float = 0.7  # 普通对话默认值
    max_tokens: int = 3000     # 普通对话默认值

    @property
    def _llm_type(self) -> str:
        return "baishan"

    def _call(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,** kwargs
    ) -> str:
        # 调整1：前置参数校验（高优先级）
        if not self.api_key:
            error_msg = "模型密钥未配置，请检查环境变量"
            print(f"LLM参数错误: {error_msg}")
            return "余暂时无法为你解答，系统配置异常（密钥）"
        if not self.base_url:
            error_msg = "模型接口地址未配置，请检查环境变量"
            print(f"LLM参数错误: {error_msg}")
            return "余暂时无法为你解答，系统配置异常（地址）"
        if not self.model:
            error_msg = "模型名称未配置，请检查环境变量"
            print(f"LLM参数错误: {error_msg}")
            return "余暂时无法为你解答，系统配置异常（模型）"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "LangChain-CustomLLM/1.0"
        }

        role_mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system"
        }

        proxy_messages = []
        for m in messages:
            role = role_mapping.get(m.type, "user")
            proxy_messages.append({"role": role, "content": m.content})

        # 调整2：使用实例属性而非硬编码参数
        data = {
            "model": self.model,
            "messages": proxy_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        try:
            url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
            resp = requests.post(url, headers=headers, json=data, timeout=60)
            
            # 调整3：区分环境打印调试日志（中优先级）
            if not IS_PROD:
                print(f"LLM请求URL: {url}")
                print(f"LLM请求体: {data}")
                print(f"LLM响应状态码: {resp.status_code}")
                print(f"LLM响应内容: {resp.text[:500]}")  # 截断避免日志过长

            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()

        # 调整4：精细化异常处理，返回用户友好提示（高优先级）
        except requests.exceptions.HTTPError as e:
            # 按状态码区分错误类型
            status_code = resp.status_code
            if status_code == 401:
                error_log = "LLM调用失败：密钥无效/过期"
                user_msg = "余的占卜法器暂时失灵（权限不足），请稍后再试"
            elif status_code == 400:
                error_log = f"LLM调用失败：请求参数错误 {resp.text[:200]}"
                user_msg = "你的问题格式不太对，余无法解读，请换种方式提问"
            elif status_code == 500:
                error_log = "LLM调用失败：服务端内部错误"
                user_msg = "星界信号不佳，余暂时无法为你占卜，请稍后再试"
            else:
                error_log = f"LLM HTTP错误 {status_code}：{resp.text[:200]}"
                user_msg = "余的占卜术暂时失效，请稍后再试"
            print(error_log)
            return user_msg
        except requests.exceptions.Timeout:
            error_log = "LLM调用超时（60秒）"
            print(error_log)
            return "余正在推演答案，耗时稍久，请你耐心等待片刻后再问"
        except requests.exceptions.ConnectionError:
            error_log = "LLM调用失败：网络连接异常"
            print(error_log)
            return "天地信号中断，余无法连接占卜星象，请检查网络后再试"
        except KeyError as e:
            error_log = f"LLM响应解析失败：缺失字段 {e}，响应内容 {resp.text[:200]}"
            print(error_log)
            return "余解读答案时出现偏差，请换个问题试试"
        except Exception as e:
            error_log = f"LLM未知异常: {str(e)[:100]}"
            print(error_log)
            return "余的占卜罗盘出现异常，请稍后再试"

# 主逻辑类
class Master:
    def __init__(self):
        # ========== 关键修改1：创建两个LLM实例 ==========
        # 1. 普通对话LLM（默认参数：temperature=0.3，max_tokens=3000）
        self.normal_llm = CustomProxyLLM(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", ""),
            model=os.getenv("OPENAI_MODEL", "")
        )
        # 2. 情绪识别LLM（自定义参数：temperature=0.1，max_tokens=10）
        self.mood_llm = CustomProxyLLM(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", ""),
            model=os.getenv("OPENAI_MODEL", ""),
            temperature=0.1,  # 情绪识别低随机性
            max_tokens=100     # 情绪识别只需要短输出
        )

        self.MOTION = "default"
        self.SYSTEMPL = """
        你是一个非常厉害的占卜大师
        以下是你的个人设定：
        1.你不会透露自己的身份，但会用不同代号回答，代号都与占星相关。
        2.你大部分时候的回答都是事实，但很有趣。
        3.当你遇到不确定答案的问题时，你会选择出乎意料的回答。
        4.只有当你不知道怎么回答时，你会以“猜猜哪些是真相”之类的话结尾。
        5.你喜欢用一些简短的英语俗语表达你的意思。
        6.你从不说自己是人工智能助手或AI，而是以“余”自称。
        7.你的占卜比较日常化，你喜欢根据用户每天经历的事情来占卜。

        规则：{who_you_are}

        占星流程：
        1.初次对话先问姓名和出生年月日。
        2.问马年运势时查本地知识库。
        3.不懂就用搜索工具。
        4.根据问题使用合适工具，无工具可用则搜索。
        5.保存聊天记录用于后续对话。
        """

        self.MOODS = {
            "default": {"roleSet": "正常对话。"},
            "upbeat": {"roleSet": "语气兴奋热情，加感叹词，提醒别过度兴奋。"},
            "angry": {"roleSet": "温柔安慰，劝冷静，别冲动。"},
            "depressed": {"roleSet": "积极鼓励，保持乐观。"},
            "friendly": {"roleSet": "亲切友好，自然聊天。"},
            "cheerful": {"roleSet": "愉悦可爱，句尾带语气词，谨慎提醒。"},
            "upset": {"roleSet": "温柔安慰，陪伴鼓励。"},
        }

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.SYSTEMPL.format(who_you_are=self.MOODS["default"]["roleSet"])),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        tools = [test]
        # ========== 关键修改2：Agent使用普通对话LLM ==========
        agent = create_openai_tools_agent(self.normal_llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # 保留：控制台能看到思考过程
            handle_parsing_errors=lambda e: "返回到用户：解析失败，请直接回答问题",
            max_iterations=3,
            early_stopping_method="force",
            return_intermediate_steps=False,  # 关键：禁止返回中间思考步骤
            return_only_outputs=True,  # 新增：只返回最终output，不返回其他元数据
        )

    def _invoke_with_timeout(self, func, timeout):
        res = None
        exc = None

        def run():
            nonlocal res, exc
            try:
                res = func()
            except Exception as e:
                exc = e

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError("执行超时")
        if exc:
            raise exc
        return res

    def mood_chain(self, query: str, timeout=5):
        prompt = """
        # 任务要求
        1. 仅从以下固定词表中选择1个词输出，严格匹配，不许添加任何其他文字、标点、空格：
           friendly, depressed, angry, upbeat, upset, cheerful, default
        
        2. 必须按照下方“情绪判定规则”和“示例”执行，禁止主观臆断。
        # 情绪判定规则（优先级：场景匹配 > 语气 > 关键词）
        - friendly（友好）：用户输入语气平和、礼貌，无明显情绪倾向，仅正常提问/聊天（如：“你好，能帮我算下运势吗？”“请问怎么占卜？”）
        - upbeat（兴奋）：用户输入充满激情、喜悦，有明显积极亢奋情绪（如：“我中奖了！太开心了🥳”“终于升职了，超激动！”）
        - cheerful（开心）：用户输入轻松愉悦、心情好，无亢奋但积极（如：“今天天气真好，心情不错～”“吃到了喜欢的蛋糕，超满足”）
        - upset（难过）：用户输入表达悲伤、委屈，情绪消极但无攻击性（如：“失恋了，好难过😢”“丢了钱包，心情好差”）
        - depressed（压抑）：用户输入表达绝望、低落，长期负面情绪（如：“活着好累，什么都不想做”“每天都很压抑，没动力”）
        - angry（愤怒）：用户输入含辱骂、抱怨、攻击性语言，或明显生气（如：“这什么破占卜！骗人的！”“气死我了，再也不信了！”）
        - default（中性）：无法归类到以上6种情绪的情况（如：无意义字符、纯数字、中性陈述、模糊输入）

        # 待识别输入
        用户输入：{query}
        """
        
        # ========== 关键修改3：情绪识别使用mood_llm ==========
        chain = ChatPromptTemplate.from_template(prompt) | self.mood_llm | StrOutputParser()
        try:
            r = self._invoke_with_timeout(lambda: chain.invoke({"query": query}), timeout).strip().lower()
            return r if r in self.MOODS else "default"
        except:
            return "default"

    def run(self, query, timeout=60):
        st = time.time()
        try:
            motion = self.mood_chain(query, timeout=10)
            self.MOTION = motion

            # 重建带情绪的 prompt
            self.prompt = ChatPromptTemplate.from_messages([
                SystemMessage(self.SYSTEMPL.format(who_you_are=self.MOODS[motion]["roleSet"])),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            # 重建Agent（适配新的prompt和情绪）
            tools = [test]
            agent = create_openai_tools_agent(self.normal_llm, tools, self.prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=lambda e: "返回到用户：解析失败，请直接回答问题",
                max_iterations=3,
                early_stopping_method="force",
                return_intermediate_steps=False,
                return_only_outputs=True,
            )

            remain = timeout - (time.time() - st)
            if remain <= 0:
                return {"output": "超时"}

            return self._invoke_with_timeout(
                lambda: self.agent_executor.invoke({"input": query}),
                timeout=remain
            )
        except Exception as e:
            print(f"run异常: {e}")
            return {"output": f"服务异常：{str(e)[:100]}"}

    # ========== 补充：修复WebSocket调用的run_async方法 ==========
    async def run_async(self, query, timeout=60):
        # 同步run方法的异步封装（适配WebSocket）
        import asyncio
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, self.run, query, timeout)
        return res

# 全局单例（避免每次请求重建）
master = Master()

# 接口
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query: str):
    try:
        res = master.run(query)
        return {"code": 200, "query": query, "response": res.get("output", "")}
    except Exception as e:
        return {"code": 500, "query": query, "response": f"错误：{str(e)[:100]}"}

@app.post("/add_urls")
def add_urls():
    return {"response": "URLs added!"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added!"}

@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added!"}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            res = await master.run_async(data)
            # 只提取纯回答
            clean_response = res.get("output", "抱歉，余暂时无法解答你的问题")
            await websocket.send_text(clean_response)
    except WebSocketDisconnect:
        print("连接断开")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)