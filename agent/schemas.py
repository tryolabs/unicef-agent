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


class Prompts(BaseModel):
    header_prompt: str
    system_prompt: str


class ServerConfig(BaseModel):
    """Server configuration settings."""

    port: int
    host: str


class LLMConfig(BaseModel):
    """LLM configuration settings."""

    model: str
    temperature: float


class MCPConfig(BaseModel):
    """MCP configuration settings."""

    datawarehouse_url: str
    rag_url: str
    geospatial_url: str


class Config(BaseModel):
    """Configuration settings."""

    server: ServerConfig
    mcp: MCPConfig
    llm: LLMConfig
