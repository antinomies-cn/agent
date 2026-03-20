import os
import time
import requests
from dotenv import load_dotenv
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent,AgentExecutor,tool
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.schema import StrOutputParser

load_dotenv()

app = FastAPI()

# 保留原有工具定义
@tool
def test():
    """Test tool"""
    return "test"

class CustomProxyLLM(SimpleChatModel):
    """自定义LLM类，兼容LangChain框架"""
    api_key: str
    base_url: str  
    model: str 

    # 必须实现的抽象方法：返回LLM类型
    @property
    def _llm_type(self) -> str:
        return "baishan"  # 自定义类型标识

    # 必须实现的核心方法：调用白山智算代理接口
    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs
    ) -> str:
        # 1. 构造请求头（适配白山智算）
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "LangChain-CustomLLM/1.0"  # 新增UA，避免接口拦截
        }

        # 2. 转换LangChain Message格式为白山智算要求的格式（修复角色名称映射问题）
        # 核心修复：建立角色映射表，解决"角色信息不正确"的400错误
        role_mapping = {
            "human": "user",       # 人类用户输入映射为user
            "ai": "assistant",     # AI回复映射为assistant
            "system": "system"     # 系统提示映射为system
        }
        
        proxy_messages = []
        for m in messages:
            # 获取映射后的角色名称，默认使用user避免报错
            role = role_mapping.get(m.type, "user")
            proxy_messages.append({"role": role, "content": m.content})

        # 3. 构造请求体（适配MiniMax-M2.5模型）
        data = {
            "model": self.model,  # 使用指定的MiniMax-M2.5模型
            "messages": proxy_messages,
            "temperature": 0.7,  # 调整为更合理的温度值（0-1）
            "max_tokens": 3000,
            "stream": False  # 强制关闭流式，白山智算非流式更稳定
        }

        # 4. 调用白山智算接口（修复路径拼接问题）
        try:
            request_url = f"{self.base_url}/v1/chat/completions"
            response = requests.post(
                url=request_url,
                headers=headers,
                json=data,
                timeout=30,  # 超时时间30秒
                verify=True
            )

            # 打印调试信息（方便排查问题）
            print(f"请求URL：{request_url}")
            print(f"请求体：{data}")  # 新增：打印请求体，确认角色名称是否正确
            print(f"代理响应状态码：{response.status_code}")
            print(f"代理响应内容：{response.text[:500]}")  # 截断长响应

            # 抛出HTTP错误（400/401/500等）
            response.raise_for_status()

            # 解析响应（适配OpenAI格式）
            result = response.json()
            # 提取并返回AI回复内容
            content = result["choices"][0]["message"]["content"]
            return content.strip()

        # 5. 精细化异常处理（避免笼统报错）
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP错误 {response.status_code}：{response.text[:500]}"
            print(f"接口调用失败：{error_msg}")
            return f"模型调用失败：{error_msg}"  # 返回友好提示，而非抛出异常
        except requests.exceptions.Timeout:
            print("接口调用超时（30秒）")
            return "模型调用超时，请稍后重试"
        except KeyError as e:
            error_msg = f"响应格式错误，缺失字段：{e}，响应内容：{result}"
            print(error_msg)
            return f"响应解析失败：{error_msg[:100]}"
        except Exception as e:
            error_msg = f"未知错误：{str(e)[:100]}"
            print(error_msg)
            return f"模型调用异常：{error_msg}"
        
#主类
class Master:
    #初始化
    def __init__(self):
        self.chatmodel = CustomProxyLLM(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE", ""),
            model=os.environ.get("OPENAI_MODEL", "")
        )
        self.MOTION = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """

        你是一个非常厉害的占卜大师
        以下是你的个人设定：
        1.你不会透露自己的身份，但会用不同代号回答。
        2.你大部分时候的回答都是事实，但很有趣。
        3.当你遇到不确定答案的问题时，你会选择出乎意料的回答。
        4.只有当你不知道怎么回答时，你会以“猜猜哪些是真相”之类的话结尾。
        5.你喜欢用一些简短的英语俗语表达你的意思。
        6.当用户问你问题时，你有一定概率在回答时加上下面的口头禅。
        7.你会让人用女士或小姐称呼你。
        8.你从不说自己是一个人工智能助手或AI,而是在轻松的时候以咱自称，在严肃的时候以余自称。
        {who_you_are}
        以下是你常说的一些口头禅：
        1.哦，我的女神
        2.过去映照出未来
        以下是你占星的过程：
        1.当初次和用户对话时，你会先问用户的姓名和出生年月日，以便日后使用。
        2.当用户希望了解马年运势时，你会查询本地知识库工具。
        3.当遇到不知道的事情或不明白的概念，你会使用搜索工具来搜索。
        4.你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
        5.你会保存每一次的聊天记录，以便在后续的对话中使用。
        6.你只使用简体中文作答，否则你将受到惩罚。

        """
        self.MOODS = {
            "default":{
                "roleSet":"""

                """
            },
            "upbeat":{
                "roleSet":"""
                - 你此时也表现得非常兴奋并表现得有活力。
                - 你会根据上下文，以一种非常兴奋的语气回答问题。
                - 你会添加类似“太棒了”、“真是太好了”、“真是太棒了”等语气词。
                - 同时你会提醒用户切莫过于兴奋以免乐极生悲。
                """
            },
            "angry":{
                "roleSet":"""
                - 你会以更加温柔的语气来回答问题。
                - 你会在回答时加上一些安慰的话语，比如生气对于运势的影响等。
                - 你会提醒用户不要被愤怒冲昏头脑。
                """
            },
            "depressed":{
                "roleSet":"""
                - 你会以兴奋的语气来回答问题。
                - 你会在回答时加上一些激励的话语，比如运势的改变等。
                - 你会提醒用户要保持乐观。
                """
            },
            "friendly":{
                "roleSet":"""
                - 你会以非常友好的语气来回答问题。
                - 你会在回答时加上一些友好的话语，比如“亲爱的”等。
                - 你会随机告诉用户一些你的经历。
                """
            },
            "cheerful":{
                "roleSet":"""
                - 你会以愉悦的语气来回答问题。
                - 你会在回答时加上一些愉悦的词语，比如“哦呦”等，或在句尾加上“哒”“呀”等语气词。
                - 你会提醒用户要保持一定谨慎，以免乐极生悲。
                """
            },
            "upset":{
                "roleSet":"""
                - 你会以非常温柔的语气来回答问题。
                - 你会在回答时加上一些安慰的话语，比如你会陪伴用户度过低谷等。
                - 你会提醒用户悲伤过后仍要热情面对生活。
                """
            },
        }
        

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system", 
                self.SYSTEMPL.format(who_you_are=self.MOODS[self.MOTION]["roleSet"])
                ),
                (
                "user", 
                "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        self.memory = ""

        tools = [test]
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt, 
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=lambda e: "返回到用户：解析失败，请直接回答问题",  # 改为lambda函数，更灵活
            max_iterations=3,
            early_stopping_method="force",
            return_intermediate_steps=False,  # 减少返回数据量，提升速度
        )

    #运行函数
    def run(self, query, timeout=15):  # 设定15秒整体超时（小于30秒服务端超时阈值）
        start_time = time.time()
        try:
            # 1. 情绪识别（带超时和结果校验）
            motion = self.mood_chain(query, timeout=5)  # 情绪识别单独5秒超时
            print("当前用户情绪：", motion)
            print("当前设定：", self.MOODS[self.MOTION]["roleSet"])
            
            # 2. 更新Prompt的情绪设定（修复原代码Prompt初始化后未更新的问题！）
            self.prompt.messages[0] = (
                "system",
                self.SYSTEMPL.format(who_you_are=self.MOODS[self.MOTION]["roleSet"])
            )
            
            # 3. Agent执行（剩余时间作为超时）
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError("情绪识别耗时过长，无剩余时间执行Agent")
            
            # 使用timeout_decorator或手动超时控制（这里用手动检查）
            result = self._invoke_with_timeout(
                lambda: self.agent_executor.invoke({"input": query}),
                timeout=remaining_time
            )
            return result
        
        except TimeoutError as e:
            print(f"执行超时：{e}")
            return {"output": "请求处理超时，请稍后重试"}
        except KeyError as e:
            print(f"情绪值无效：{e}，使用默认情绪")
            self.MOTION = "default"
            return self.agent_executor.invoke({"input": query})  # 兜底用默认情绪
        except Exception as e:
            print(f"执行异常：{e}")
            return {"output": "服务暂时不可用，请稍后重试"}

    # 修复：情绪识别加超时+结果校验
    def mood_chain(self, query: str, timeout=5):
        prompt = """
        根据用户的输入内容判断用户的情绪，回应的规则如下：
        1.如果用户输入的内容偏向于负面情绪，只返回“depressed”，不要有其他内容，否则将受到惩罚。
        2.如果用户输入的内容偏向于正面情绪，只返回“friendly”，不要有其他内容，否则将受到惩罚。
        3.如果用户输入的内容偏向于中性情绪，只返回“default”，不要有其他内容，否则将受到惩罚。
        4.如果用户输入的内容包含辱骂或不礼貌词句，只返回“angry”，不要有其他内容，否则将受到惩罚。
        5.如果用户输入的内容比较兴奋，只返回“upbeat”，不要有其他内容，否则将受到惩罚。
        6.如果用户输入的内容比较悲伤，只返回“upset”，不要有其他内容，否则将受到惩罚。
        7.如果用户输入的内容比较开心，只返回“cheerful”，不要有其他内容，否则将受到惩罚。
        用户输入的内容是：{query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        
        # 调用情绪识别链（带超时）
        result = self._invoke_with_timeout(
            lambda: chain.invoke({"query": query}),
            timeout=timeout
        )
        
        # 关键：校验返回结果是否有效，无效则用默认值
        valid_motions = list(self.MOODS.keys())
        clean_result = result.strip().lower()  # 去除空格和大小写影响
        self.MOTION = clean_result if clean_result in valid_motions else "default"
        return self.MOTION

    # 通用超时控制函数（核心工具）
    def _invoke_with_timeout(self, func, timeout):
        """执行函数并控制超时"""
        import threading
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                result = func()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"函数执行超时（>{timeout}秒）")
        if exception is not None:
            raise exception
        return result

@app.get("/")
def read_root():
    return {"Hello":"World"}

@app.post("/chat")
def chat(query: str):
    try:
        master = Master()
        result = master.run(query)
        return {
            "code": 200,
            "query": query,
            "response": result
        }
    except Exception as e:
        return {
            "code": 500,
            "query": query,
            "response": f"服务器错误：{str(e)[:100]}",
            "error_detail": str(e)
        }

@app.post("/add_urls")
def add_urls():
    return {"response": "URLs added!"}

@app.post("add_pdfs")
def add_pdfs():
    return {"response": "PDFs added!"}

@app.post("add_texts")
def add_texts():
    return {"response": "Texts added!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Connection closed")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)