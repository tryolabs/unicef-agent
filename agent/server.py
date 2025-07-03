import logging
import uuid

import uvicorn
from config import load_config
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from handlers import handle_response
from logging_config import get_logger
from schemas import Chat

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

logger = get_logger(__name__)

config = load_config()

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    logger.info("Root endpoint called")
    return {"message": "Hello World"}


@app.post("/ask")
async def ask(chat: Chat) -> StreamingResponse:
    """Process user question and return streaming response.

    Args:
        chat: Chat object containing chat messages.
        current_user: Current authenticated user from dependency injection.

    Returns:
        StreamingResponse: Streaming response containing the response from the agent.
    """
    logger.info("Received /ask endpoint call with session_id=%s", chat.session_id)
    if chat.chat_messages == [] or chat.chat_messages[-1].content == "":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Chat messages cannot be empty",
        )

    trace_id = uuid.uuid4().hex

    session_id = chat.session_id
    logger.info(
        "Processing question: %s with trace ID: %s and session ID: %s",
        chat.chat_messages[-1].content,
        trace_id,
        session_id,
    )

    return StreamingResponse(
        handle_response(
            chat.chat_messages,
            trace_id,
            session_id,
            config,
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    logger.info("🚀 Starting server... ")

    logger.info(
        'Check "http://%s:%s" for the server status',
        config.server.host,
        config.server.port,
    )

    uvicorn.run(app, host=config.server.host, port=config.server.port)
