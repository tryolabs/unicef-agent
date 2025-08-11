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
from schemas import Config, LLMConfig
from workflows.events import Event

langfuse = get_client()
LlamaIndexInstrumentor().instrument()

logger = get_logger(__name__)


def get_llm(specific_config: LLMConfig | None = None) -> LiteLLM:
    """Get the LLM model.

    Returns:
        A configured ChatLiteLLM instance
    """
    if specific_config is None:
        specific_config = config.llm

    logger.info("Getting LLM with model: %s", specific_config.model)
    return LiteLLM(
        model=specific_config.model,
        temperature=specific_config.temperature,
        additional_kwargs={
            "stop": ["Observation:"],
            "aws_region_name": specific_config.region_name,
        },
    )


async def create_agent(specific_config: Config | None = None) -> ReActAgent:
    """Create a ReAct agent with the given LLM, tools and system prompt.

    Returns:
        A compiled ReAct agent ready to be invoked
    """
    if specific_config is None:
        specific_config = config

    logger.info("Creating agent")
    prompts = get_prompts()
    tools = await get_tools(specific_config.mcp)
    llm = get_llm(specific_config.llm)

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
    logger.info("Agent created")

    return agent


async def run_agent(
    agent: ReActAgent,
    inputs: dict[str, list[dict[str, str]]],
    trace_id: str,
    session_id: str,
    tags: list[str] | None = None,
) -> AsyncGenerator[Event, None]:
    """Run a ReAct agent with the given inputs and stream the results.

    Args:
        agent: The compiled ReAct agent to run
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
