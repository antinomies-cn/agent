import time
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import WebBaseLoader
from app.core.config import IS_PROD
from app.core.logger_setup import logger
from app.services.master_service import Master
from app.core.texts import USER_MESSAGES

app = FastAPI()
master = Master()

@app.get("/")
def read_root():
    logger.info("访问根路径")
    return {"Hello": "World"}

@app.post("/chat")
def chat(query: str, session_id: str):
    logger.info(f"接收Chat API请求 | session_id: {session_id} | 查询: {query[:100]}")
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id不能为空")
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
def add_urls(URL:str):
    web_loader = WebBaseLoader(URL)
    _documents = web_loader.load()
    _splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
    )
    logger.info("调用add_urls接口" )
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
def memory_status(session_id: str):
    """查看指定session的memory状态。"""
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id不能为空")
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
    session_id = (websocket.query_params.get("session_id") or "").strip()
    if not session_id:
        logger.warning("WebSocket连接拒绝 | 客户端IP: %s | 原因: session_id不能为空", client_ip)
        await websocket.close(code=1008, reason="session_id不能为空")
        return
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