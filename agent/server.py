import logging
import uuid
from typing import Annotated

import uvicorn
from auth import authenticate_user, create_access_token, get_current_user
from config import config
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from logging_config import get_logger
from pydantic import BaseModel
from schemas import Chat

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

logger = get_logger(__name__)

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    logger.info("Root endpoint called")
    return {"message": "Hello World"}


class Token(BaseModel):
    access_token: str
    username: str
    token_type: str = "bearer"  # noqa: S105


# Temp endpoint for auth
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """Authenticate user and return access token.

    Args:
        form_data: OAuth2 password request form containing username and password.

    Returns:
        Token: Access token with bearer type and username.
    """
    logger.info(
        "Received /token endpoint call with username=%s",
        form_data.username,
    )
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.username},
    )

    return Token(
        access_token=access_token,
        username=user.username,
    )


class User(BaseModel):
    username: str


@app.get("/users/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current authenticated user information.

    Args:
        current_user: Current authenticated user from dependency injection.

    Returns:
        User: Current authenticated user.
    """
    return current_user


@app.post("/ask")
async def ask(
    chat: Chat, _current_user: Annotated[User, Depends(get_current_user)]
) -> StreamingResponse:
    """Process user question and return streaming response.

    Args:
        chat: Chat object containing chat messages.
        current_user: Current authenticated user from dependency injection.

    Returns:
        StreamingResponse: Streaming response containing the response from the agent.
    """
    from handlers import handle_response

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
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    from initialize import initialize_app

    initialize_app()

    logger.info("🚀 Starting server... ")

    logger.info(
        'Check "http://%s:%s" for the server status',
        config.server.host,
        config.server.port,
    )

    uvicorn.run(app, host=config.server.host, port=config.server.port)
