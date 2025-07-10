import json
import re
from collections.abc import AsyncGenerator

from llama_index.core.agent.workflow import AgentOutput, AgentStream, ToolCallResult
from llama_index.core.workflow import StopEvent
from logging_config import get_logger
from schemas import Config, Message, ReturnChunk, TextOutput, ToolOutput

from agent import create_agent, run_agent

logger = get_logger(__name__)


async def handle_response(
    messages: list[Message],
    trace_id: str,
    session_id: str,
    config: Config,
) -> AsyncGenerator[str, None]:
    """Handle the response by running respond, and cleaning up.

    Args:
        messages: List of messages to process
        trace_id: Unique identifier for tracing the request
        session_id: Unique identifier for the session
        config: The configuration to use for the agent

    Yields:
        JSON serialized chunks of the response
    """
    formatted_messages = _format_messages(messages)

    logger.info("Running agent with inputs %s", formatted_messages)

    async for chunk in respond(formatted_messages, trace_id, session_id, config):
        yield chunk


async def respond(
    messages: dict[str, list[dict[str, str]]],
    trace_id: str,
    session_id: str,
    config: Config,
) -> AsyncGenerator[str, None]:
    """Process messages and generate a response using the agent.

    Args:
        messages: List of messages to process
        trace_id: Unique identifier for tracing the request
        session_id: Unique identifier for the session
        config: The configuration to use for the agent

    Yields:
        JSON serialized chunks of the response, including tool calls, agent streams,
        and the final answer
    """
    agent = await create_agent(config)

    is_final_answer = False
    is_thought_chunk = True
    async for chunk in run_agent(  # type: ignore[arg-type]
        agent,
        messages,
        trace_id,
        session_id,
    ):
        match chunk:
            case ToolCallResult():
                return_chunk = _process_tool_call_chunk(chunk, trace_id)

            case AgentStream():
                if chunk.delta.startswith("Action"):
                    is_thought_chunk = False
                elif chunk.delta.startswith("Thought"):
                    is_thought_chunk = True

                # Skip non-thought chunks
                if not is_thought_chunk:
                    continue

                return_chunk = _process_agent_stream_chunk(chunk, f"th_{trace_id}")

            case StopEvent():
                # Signal that the thought is complete and the next chunk will be the response
                is_final_answer = True
                return_chunk = _process_stop_event(trace_id)

            case _ if is_final_answer:
                if not isinstance(chunk, AgentOutput):
                    logger.error("Unexpected chunk type: %s", type(chunk))
                    continue

                return_chunk = _process_final_answer(chunk, trace_id)

            case _:
                continue

        yield json.dumps(return_chunk.model_dump())
        yield "\n"

    # Signal that the response is complete
    return_chunk = ReturnChunk(trace_id=trace_id, is_finished=True)
    yield json.dumps(return_chunk.model_dump())
    yield "\n"


def _format_messages(chat_messages: list[Message]) -> dict[str, list[dict[str, str]]]:
    """Format chat messages into the expected format for the agent.

    Args:
        chat_messages: List of Message objects containing role, content and trace_id

    Returns:
        Dictionary with 'messages' key containing list of formatted message dictionaries
    """
    messages = [
        {
            "role": message.role,
            "content": message.content,
            "trace_id": message.trace_id,
        }
        for message in chat_messages
    ]

    return {"messages": messages}


def _process_tool_call_chunk(
    chunk: ToolCallResult,
    thinking_trace_id: str,
) -> ReturnChunk:
    """Process a tool call chunk and return the appropriate ReturnChunk.

    Args:
        chunk: ToolCallResult object containing tool name and output
        thinking_trace_id: Trace ID for the thinking phase

    Returns:
        ReturnChunk object with tool call details and any HTML content

    Raises:
        Exception if there is an error processing the tool call
    """
    try:
        try:
            content = _parse_string_to_tool_output(chunk.tool_output.content)
            input_arguments: dict[str, str] = content.content.text.get("input_arguments", {})

        except (ValueError, SyntaxError):
            input_arguments = {}
            content = None

        tool_name = chunk.tool_name
        logger.info("Handling tool call: %s", tool_name)
        tool_call_message = f"Calling {tool_name}"

        if input_arguments:
            tool_call_message += " with arguments:\n" + "".join(
                [f"   {key}: {value}\n" for key, value in input_arguments.items()]
            )

        if tool_name == "build_map" and content:
            html_content = content.content.text.get("html_content", "")
            return ReturnChunk(
                tool_call=tool_call_message,
                trace_id=thinking_trace_id,
                html_content=html_content,
            )

        return ReturnChunk(
            tool_call=tool_call_message,
            trace_id=thinking_trace_id,
        )
    except Exception as e:
        message = f"Exception handling tool call: {e}"
        logger.exception(message)
        logger.exception("Tool call: %s", chunk)
        raise


def _process_agent_stream_chunk(chunk: AgentStream, thinking_trace_id: str) -> ReturnChunk:
    """Process an agent stream chunk and return the appropriate ReturnChunk.

    Args:
        chunk: AgentStream object containing the delta response
        thinking_trace_id: Trace ID for the thinking phase

    Returns:
        ReturnChunk object with the processed response, adding a newline
        if the response ends with a closing brace
    """
    response = str(chunk.delta)

    # Send the actual response chunk
    return_chunk = ReturnChunk(response=response, trace_id=thinking_trace_id)

    if response.endswith("}"):
        # Insert a line break after action input
        return_chunk = ReturnChunk(
            response=f"{response}\n",
            trace_id=thinking_trace_id,
        )

    return return_chunk


def _process_stop_event(thinking_trace_id: str) -> ReturnChunk:
    """Process a stop event and return the appropriate ReturnChunk.

    Args:
        thinking_trace_id: Trace ID for the thinking phase

    Returns:
        ReturnChunk object indicating the thinking phase is finished
    """
    return ReturnChunk(trace_id=thinking_trace_id, is_finished=True)


def _process_final_answer(chunk: AgentOutput, response_trace_id: str) -> ReturnChunk:
    """Process the final answer chunk and return the appropriate ReturnChunk.

    Args:
        chunk: Expected to be an AgentOutput object containing the final response
        response_trace_id: Trace ID for the response phase

    Returns:
        ReturnChunk object containing the final response content

    Logs an error if the chunk is not of type AgentOutput
    """
    return ReturnChunk(response=str(chunk.response.content), trace_id=response_trace_id)


def _parse_string_to_tool_output(input_string: str) -> ToolOutput:
    """Parses a complex string into a structured ToolOutput.

    Args:
        input_string: The string to be parsed.

    Returns:
        A ToolOutput instance with the extracted data.
    """
    try:
        # Extract the 'meta' value
        meta_match = re.search(r"meta=([^ ]+)", input_string)
        meta_value = meta_match.group(1) if meta_match else None
        if meta_value == "None":
            meta_value = None

        # Extract the 'isError' value
        is_error_match = re.search(r"isError=(True|False)", input_string)
        is_error_value = is_error_match.group(1) == "True" if is_error_match else None

        # Extract the 'content' value
        text_content_match = re.search(
            r"content=\[TextContent\(type='text', text='(.*?)', annotations=None, meta=None\)\]",
            input_string,
            re.DOTALL,
        )

        if text_content_match:
            text_content = text_content_match.group(1)
            try:
                # First, decode the escaped string properly
                decoded_content = text_content.encode().decode("unicode_escape")
                content_value = TextOutput(
                    type="text",
                    text=json.loads(decoded_content),
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Fallback: try parsing the raw content directly
                content_value = TextOutput(
                    type="text",
                    text=json.loads(text_content),
                )
        else:
            content_value = None

        if content_value is None:
            msg = "Content value is None"
            logger.error(msg)
            raise ValueError(msg)

        if is_error_value is None:
            msg = "Is error value is None"
            logger.error(msg)
            raise ValueError(msg)

        return ToolOutput(
            meta=meta_value,
            content=content_value,
            is_error=is_error_value,
        )

    except (AttributeError, json.JSONDecodeError):
        msg = "An error occurred during parsing"
        logger.exception(msg)
        return ToolOutput(
            meta=None,
            content=TextOutput(type="text", text={"error": msg}),
            is_error=True,
        )
