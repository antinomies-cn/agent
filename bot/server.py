import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage


load_dotenv()

app = FastAPI()

class CustomProxyLLM(SimpleChatModel):
    """自定义LLM类，适配非官方OpenAI代理接口，兼容LangChain框架"""
    api_key: str
    base_url: str
    model: str

    # 必须实现的抽象方法：返回LLM类型
    @property
    def _llm_type(self) -> str:
        return "custom-proxy-llm"  # 自定义类型标识，无特殊要求

    # 必须实现的核心方法：调用代理接口
    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs
    ) -> str:
        # 构造代理要求的请求格式（和你之前能成功的requests代码一致）
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # 转换LangChain的Message格式为代理要求的格式
        proxy_messages = [
            {"role": m.type, "content": m.content} for m in messages
        ]
        data = {
            "model": self.model,
            "messages": proxy_messages,
            "temperature": 0,
            "max_tokens": 2000
        }
        # 调用代理接口（手动拼接正确路径）
        try:
            response = requests.post(
                url=f"{self.base_url}/v1/chat/completions",  # 确保路径和代理一致
                headers=headers,
                json=data,
                timeout=30
            )

            print(f"代理响应状态码：{response.status_code}")
            print(f"代理响应内容：{response.text}")

            response.raise_for_status()  # 抛出HTTP错误
            # 解析响应（适配OpenAI格式）
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"代理接口调用失败：{str(e)}") from e

#主类
class Master:
    #初始化
    def __init__(self):
        self.chatmodel = CustomProxyLLM(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE", ""),
            model="DeepSeek-R1-0528-Qwen3-8B"
        )

        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = ""
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个助手",
                ),
                (
                    "user",
                    "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        self.memory = ""
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=[],
            prompt=self.prompt, 
        )
        self.agent_executor = AgentExecutor(
            agent = agent,
            tools=[],
            verbose=True,
        )

    #运行函数
    def run(self,query):
        result = self.agent_executor.invoke({"input":query})
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