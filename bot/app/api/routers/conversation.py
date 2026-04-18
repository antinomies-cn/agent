import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.api.deps import resolve_runtime_dependency
from app.core.logger_setup import logger, log_event, mask_session_id, summarize_error_for_log, summarize_text_for_log
from app.core.texts import USER_MESSAGES

router = APIRouter()


@router.get("/", summary="根路径", description="基础连通性检查，返回简单响应。")
def read_root():
    logger.info("访问根路径")
    return {"Hello": "World"}


@router.post("/chat", summary="对话接口", description="主对话入口。query 为用户输入，session_id 用于会话隔离。")
def chat(
    query: str = Query(...),
    session_id: str = Query(...),
):
    common_errors = resolve_runtime_dependency("_COMMON_ERROR_EXPLANATIONS", {})
    master = resolve_runtime_dependency("master", None)

    logger.info(
        "接收Chat API请求 | session_id: %s | query: %s",
        mask_session_id(session_id),
        summarize_text_for_log(query, preview_chars=24),
    )

    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "session_id不能为空",
                "error_code": "INVALID_SESSION_ID",
                "explanation": common_errors.get("INVALID_SESSION_ID", "session_id 参数非法"),
            },
        )

    if master is None:
        raise HTTPException(status_code=500, detail="master service unavailable")

    try:
        res = master.run(query, session_id=session_id)
        response_text = res.get("output", "")
        logger.info(
            "Chat API响应成功 | session_id: %s | response: %s",
            mask_session_id(session_id),
            summarize_text_for_log(response_text, preview_chars=24),
        )
        return {"code": 200, "session_id": session_id, "query": query, "response": response_text}
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(
            "Chat API执行异常 | session_id: %s | query: %s | 错误: %s",
            mask_session_id(session_id),
            summarize_text_for_log(query, preview_chars=24),
            summarize_error_for_log(error_msg),
            exc_info=True,
        )
        return {
            "code": 500,
            "error_code": "CHAT_RUNTIME_ERROR",
            "explanation": common_errors.get("CHAT_RUNTIME_ERROR", "对话处理失败"),
            "error": error_msg,
            "session_id": session_id,
            "query": query,
            "response": f"错误：{error_msg}",
        }


@router.websocket("/ws")
async def ws(websocket: WebSocket):
    is_prod_runtime = resolve_runtime_dependency("_is_prod_runtime", lambda: False)
    ensure_websocket_auth = resolve_runtime_dependency("ensure_websocket_auth", None)
    ensure_ws_rate_limit = resolve_runtime_dependency("ensure_ws_rate_limit", None)
    gateway_security_settings = resolve_runtime_dependency("gateway_security_settings", None)
    gateway_rate_limiter = resolve_runtime_dependency("gateway_rate_limiter", None)
    master = resolve_runtime_dependency("master", None)

    if is_prod_runtime():
        await websocket.close(code=1008, reason="Not Found")
        return

    if ensure_websocket_auth is None or ensure_ws_rate_limit is None or master is None:
        await websocket.close(code=1011, reason="service unavailable")
        return

    try:
        ensure_websocket_auth(gateway_security_settings, websocket)
    except HTTPException:
        await websocket.close(code=1008, reason="gateway auth failed")
        return

    await websocket.accept()
    client_ip = websocket.client.host
    session_id = (websocket.query_params.get("session_id") or "").strip()
    if not session_id:
        logger.warning("WebSocket连接拒绝 | 客户端IP: %s | 原因: session_id不能为空", client_ip)
        await websocket.close(code=1008, reason="session_id不能为空")
        return

    try:
        ensure_ws_rate_limit(gateway_security_settings, websocket, gateway_rate_limiter, session_id=session_id)
    except HTTPException:
        await websocket.close(code=1008, reason="gateway rate limited")
        return

    ws_start = time.perf_counter()
    logger.info("WebSocket连接建立 | 客户端IP: %s | session_id: %s", client_ip, mask_session_id(session_id))
    log_event(
        logging.INFO,
        "ws.open",
        client_ip=client_ip,
        session_id=session_id,
    )

    try:
        while True:
            data = await websocket.receive_text()
            msg_start = time.perf_counter()
            logger.info(
                "WebSocket接收消息 | 客户端IP: %s | session_id: %s | input: %s",
                client_ip,
                mask_session_id(session_id),
                summarize_text_for_log(data, preview_chars=24),
            )

            res = await master.run_async(data, session_id=session_id)
            clean_response = res.get("output", USER_MESSAGES["ws_default"])

            await websocket.send_text(clean_response)
            logger.info(
                "WebSocket发送消息 | 客户端IP: %s | response: %s",
                client_ip,
                summarize_text_for_log(clean_response, preview_chars=24),
            )
            log_event(
                logging.INFO,
                "ws.message",
                client_ip=client_ip,
                session_id=session_id,
                input_len=len(data),
                output_len=len(clean_response),
                elapsed_ms=int((time.perf_counter() - msg_start) * 1000),
            )

    except WebSocketDisconnect:
        logger.info("WebSocket连接断开 | 客户端IP: %s | session_id: %s", client_ip, mask_session_id(session_id))
        log_event(
            logging.INFO,
            "ws.close",
            client_ip=client_ip,
            session_id=session_id,
            elapsed_ms=int((time.perf_counter() - ws_start) * 1000),
        )
    except Exception as e:
        logger.error(f"WebSocket异常 | 客户端IP: {client_ip} | 错误: {str(e)[:100]}", exc_info=True)
        log_event(
            logging.ERROR,
            "ws.error",
            client_ip=client_ip,
            session_id=session_id,
            error=str(e)[:160],
            elapsed_ms=int((time.perf_counter() - ws_start) * 1000),
        )
        await websocket.close(code=1011, reason="服务器内部错误")
