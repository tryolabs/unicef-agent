from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    content: str
    role: Literal["user", "assistant"]
    trace_id: str


class Chat(BaseModel):
    chat_messages: list[Message]
    session_id: str


class ReturnChunk(BaseModel):
    trace_id: str
    response: str = ""
    tool_call: str = ""
    is_finished: bool = False
