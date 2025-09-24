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


def extract_latest_user_prompt(inputs: dict[str, list[dict[str, str]]]) -> str:
    """Extract the most recent user message content from the inputs.

    Falls back to the last message content if no explicit user message exists,
    and finally to a stringified version of the full inputs if no content can
    be determined.

    Args:
        inputs: A dictionary expected to contain a "messages" list.

    Returns:
        The prompt text to provide to the agent.
    """
    messages_obj = inputs.get("messages")
    if not isinstance(messages_obj, list):
        return str(inputs)

    # Prefer the latest explicit user message
    for message in reversed(messages_obj):
        role = message.get("role")
        content = message.get("content")
        if role == "user" and content:
            return str(content)

    # Fallback: use the last message content if available
    if messages_obj:
        content = messages_obj[-1].get("content", "")
        if content:
            return str(content)

    # Final fallback: stringify the full inputs
    return str(inputs)


def build_conversation_prompt(
    inputs: dict[str, list[dict[str, str]]],
    *,
    max_messages: int | None = 20,
) -> str:
    """Build a single prompt string from the full conversation history.

    The prompt enumerates prior messages in order and includes both user and
    assistant turns. Optionally limits to the last N messages to control token
    usage.

    Args:
        inputs: Dict expected to contain a "messages" list of role/content dicts.
        max_messages: If provided, only the last N messages are included.

    Returns:
        A single string prompt representing the conversation.
    """
    messages_obj = inputs.get("messages")
    if not isinstance(messages_obj, list) or len(messages_obj) == 0:
        return extract_latest_user_prompt(inputs)

    if max_messages is not None and len(messages_obj) > max_messages:
        messages_obj = messages_obj[-max_messages:]

    lines: list[str] = []
    for message in messages_obj:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        else:  # default to user or any other role labels
            lines.append(f"User: {content}")

    if not lines:
        return extract_latest_user_prompt(inputs)

    return "\n".join(lines)


async def run_agent(
    agent: ReActAgent,
    prompt_text: str,
    trace_id: str,
    session_id: str,
    tags: list[str] | None = None,
) -> AsyncGenerator[Event, None]:
    """Run a ReAct agent with the given inputs and stream the results.

    Args:
        agent: The compiled ReAct agent to run
        prompt_text: The conversation prompt string to provide to the agent
        trace_id: The trace ID to associate with this model
        session_id: The session ID to associate with this model
        tags: List of tags to associate with the trace

    Yields:
        Chunks of the agent's response stream
    """
    logger.info("Running agent with prompt: %s", prompt_text)
    with langfuse.start_as_current_span(
        trace_context=TraceContext(trace_id=trace_id),
        input={"prompt": prompt_text},
        name="",
    ) as root_span:
        root_span.update_trace(session_id=session_id, tags=tags)
        try:
            handler = agent.run(prompt_text)  # type: ignore[arg-type]

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
