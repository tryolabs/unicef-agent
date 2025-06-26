import logging
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from handlers import handle_response
from logging_config import get_logger
from schemas import Chat

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

logger = get_logger(__name__)

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
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
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    logger.info("🚀 Starting server... ")

    logger.info('Check "http://localhost:8000" for the server status')

    uvicorn.run(app, host="127.0.0.1", port=8000)
