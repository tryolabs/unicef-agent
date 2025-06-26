import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast

import yaml
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.llms.litellm import LiteLLM
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from workflows.events import Event

with Path("agent/prompts.yaml").open("r") as f:
    prompts = yaml.safe_load(f)

header_prompt = prompts["header_prompt"]
system_prompt = prompts["system_prompt"]


async def get_tools() -> list[FunctionTool]:
    mcp_client_datawarehouse = BasicMCPClient("http://127.0.0.1:6000/sse")
    datawarehouse_tools = McpToolSpec(
        client=mcp_client_datawarehouse,
    )

    mcp_client_rag = BasicMCPClient("http://127.0.0.1:6001/sse")
    rag_tools = McpToolSpec(
        client=mcp_client_rag,
    )

    datawarehouse_tools_list = await datawarehouse_tools.to_tool_list_async()
    rag_tools_list = await rag_tools.to_tool_list_async()

    all_tools = [*datawarehouse_tools_list, *rag_tools_list]

    return all_tools


def get_llm(temperature: float, session_id: str) -> LiteLLM:
    """Get the LLM model.

    Args:
        temperature: The temperature to use for the model
        session_id: The session ID to associate with this model

    Returns:
        A configured ChatLiteLLM instance
    """
    return LiteLLM(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs={
            "metadata": {
                "session_id": session_id,
            }
        },
    )


def create_agent(
    session_id: str,
    temperature: float = 0.0,
    tools: list[FunctionTool] | None = None,
    system_prompt: str = system_prompt,
) -> ReActAgent:
    """Create a LangGraph ReAct agent with the given LLM, tools and system prompt.

    Args:
        session_id: The session ID to use for the agent
        temperature: The temperature to use for the agent
        tools: List of tools available to the agent
        system_prompt: System prompt to provide context to the agent

    Returns:
        A compiled LangGraph agent ready to be invoked
    """
    agent = ReActAgent(
        tools=tools,
        llm=get_llm(temperature, session_id),
        system_prompt=system_prompt,
    )

    agent.update_prompts(
        {
            "react_header": header_prompt,
        }
    )

    return agent


async def run_agent(
    agent: ReActAgent,
    inputs: dict[str, list[dict[str, str]]],
) -> AsyncGenerator[Event, None]:
    """Run a LangGraph agent with the given inputs and stream the results.

    Args:
        agent: The compiled LangGraph agent to run
        inputs: Dictionary of inputs to provide to the agent
        session_id: The session ID to associate with this model
        tags: List of tags to associate with the trace

    Yields:
        Chunks of the agent's response stream
    """
    handler = agent.run(str(inputs))

    async for chunk in handler.stream_events():
        yield chunk

    response = cast("Event", await handler)

    yield response
