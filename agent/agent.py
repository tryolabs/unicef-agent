from collections.abc import AsyncGenerator
from typing import cast

from config import config
from initialize import get_prompts, get_tools
from langfuse import get_client
from langfuse.types import TraceContext
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.litellm import LiteLLM
from logging_config import get_logger
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from workflows.events import Event

langfuse = get_client()
LlamaIndexInstrumentor().instrument()

logger = get_logger(__name__)


def get_llm() -> LiteLLM:
    """Get the LLM model.

    Returns:
        A configured ChatLiteLLM instance
    """
    logger.info("Getting LLM with model: %s", config.llm.model)
    return LiteLLM(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )


async def create_agent() -> ReActAgent:
    """Create a LangGraph ReAct agent with the given LLM, tools and system prompt.

    Returns:
        A compiled LangGraph agent ready to be invoked
    """
    logger.info("Creating agent")
    prompts = get_prompts()
    tools = await get_tools()
    llm = get_llm()

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=prompts.system_prompt,
        additional_kwargs={"stop": ["Observation:"]},
    )

    agent.update_prompts(
        {
            "react_header": PromptTemplate(prompts.header_prompt),
        }
    )
    logger.info("Agent created")

    return agent


async def run_agent(
    agent: ReActAgent,
    inputs: dict[str, list[dict[str, str]]],
    trace_id: str,
    session_id: str,
    tags: list[str] | None = None,
) -> AsyncGenerator[Event, None]:
    """Run a LangGraph agent with the given inputs and stream the results.

    Args:
        agent: The compiled LangGraph agent to run
        inputs: Dictionary of inputs to provide to the agent
        trace_id: The trace ID to associate with this model
        session_id: The session ID to associate with this model
        tags: List of tags to associate with the trace

    Yields:
        Chunks of the agent's response stream
    """
    logger.info("Running agent")
    with langfuse.start_as_current_span(
        trace_context=TraceContext(trace_id=trace_id),
        input=inputs,
        name="",
    ) as root_span:
        root_span.update_trace(session_id=session_id, tags=tags)
        try:
            handler = agent.run(str(inputs))  # type: ignore[arg-type]

            async for chunk in handler.stream_events():
                if hasattr(chunk, "delta") and chunk.delta == "":
                    continue
                yield chunk

            response = cast("Event", await handler)
            yield response
        except Exception as e:
            msg = f"Error running agent: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e
