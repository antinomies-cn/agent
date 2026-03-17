from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

app = FastAPI()

class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="gpt4-1106-preview",
            temperature=0,
            streaming=True,
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
                )
            ],
        )
        self.memory = ""
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=[],
            self.prompt, 
        )
        self.agent_executor = AgentExecutor(agent=agent)

    def run(self,query):
        result = self.agent_executor.invoke(query)

@app.get("/")
def read_root():
    return {"Hello":"World"}

@app.post("/chat")
def chat():
    return {"response": "I am a chatbot"}

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