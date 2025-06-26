import os
from collections.abc import AsyncGenerator
from typing import cast

from initialize import get_prompts, get_tools
from langfuse import get_client
from langfuse.types import TraceContext
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.litellm import LiteLLM
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from workflows.events import Event

langfuse = get_client()
LlamaIndexInstrumentor().instrument()


def get_llm(temperature: float) -> LiteLLM:
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
    )


async def create_agent(
    temperature: float = 0.0,
) -> ReActAgent:
    """Create a LangGraph ReAct agent with the given LLM, tools and system prompt.

    Args:
        temperature: The temperature to use for the agent
    Returns:
        A compiled LangGraph agent ready to be invoked
    """
    prompts = get_prompts()
    tools = await get_tools()
    llm = get_llm(temperature)

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=prompts.system_prompt,
    )

    agent.update_prompts(
        {
            "react_header": PromptTemplate(prompts.header_prompt),
        }
    )

    return agent


async def run_agent(
    agent: ReActAgent,
    inputs: dict[str, list[dict[str, str]]],
    trace_id: str,
    session_id: str,
) -> AsyncGenerator[Event, None]:
    """Run a LangGraph agent with the given inputs and stream the results.

    Args:
        agent: The compiled LangGraph agent to run
        inputs: Dictionary of inputs to provide to the agent
        trace_id: The trace ID to associate with this model
        session_id: The session ID to associate with this model

    Yields:
        Chunks of the agent's response stream
    """
    with langfuse.start_as_current_span(
        trace_context=TraceContext(trace_id=trace_id),
        input=inputs,
        name="",
    ) as root_span:
        root_span.update_trace(session_id=session_id)

        handler = agent.run(str(inputs))  # type: ignore[arg-type]

        async for chunk in handler.stream_events():
            yield chunk

        response = cast("Event", await handler)

        yield response
