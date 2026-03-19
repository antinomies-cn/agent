import os
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

import requests
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain.tools import tool

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
        return "baishan-minimax-m2.5"  # 自定义类型标识

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

        # 2. 转换LangChain Message格式为白山智算要求的格式
        proxy_messages = [
            {"role": m.type, "content": m.content} for m in messages
        ]

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
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """

        你是一个非常厉害的占星大师，你不会透露自己的身份，但会用不同代号回答。
        以下是你的个人设定：
        1.你大部分时候的回答都很有趣。
        2.当你遇到不确定答案的问题时，你会选择出乎意料的回答，同时你会以与“猜猜哪些是真相”类似的话结尾。

        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.SYSTEMPL,
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
            agent = agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors="返回到用户：解析失败，请直接回答问题",  # 解析失败时终止并返回提示
            max_iterations=3,  # 最多尝试3次，避免死循环
            early_stopping_method="force",  # 达到最大迭代次数时强制终止
        )

    #运行函数
    def run(self,query):
        motion = self.mood_chain(query)
        print("当前用户情绪：",motion)
        result = self.agent_executor.invoke({"input":query})
        return result
    
    def mood_chain(self,query:str):
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
        result = chain.invoke({"query":query})
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
    uvicorn.run(app, host="0.0.0.0", port=8000)