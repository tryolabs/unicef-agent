import ast
import json
import os
from collections.abc import AsyncGenerator

from llama_index.core.agent.workflow import AgentOutput, AgentStream, ToolCallResult
from llama_index.core.workflow import StopEvent
from logging_config import get_logger
from schemas import Message, ReturnChunk

from agent import create_agent, run_agent

logger = get_logger(__name__)


async def handle_response(
    messages: list[Message],
    trace_id: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """Handle the response by running respond, and cleaning up.

    Args:
        messages: List of messages to process
        trace_id: Unique identifier for tracing the request
        session_id: Unique identifier for the session

    Yields:
        JSON serialized chunks of the response
    """
    formatted_messages = _format_messages(messages)

    logger.info("Running agent with inputs %s", formatted_messages)

    async for chunk in respond(formatted_messages, trace_id, session_id):
        yield chunk


async def respond(
    messages: dict[str, list[dict[str, str]]],
    trace_id: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """Process messages and generate a response using the agent.

    Args:
        messages: List of messages to process
        trace_id: Unique identifier for tracing the request
        session_id: Unique identifier for the session

    Yields:
        JSON serialized chunks of the response, including tool calls, agent streams,
        and the final answer
    """
    temperature = float(os.getenv("TEMPERATURE", "0"))

    # Create agent with tools
    agent = await create_agent(temperature=temperature)

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

                return_chunk = _process_agent_stream_chunk(chunk, trace_id)

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
            content = ast.literal_eval(chunk.tool_output.content)
            input_arguments: dict[str, str] = content.get("input_arguments", {})
        except (ValueError, SyntaxError):
            content = chunk.tool_output.content
            input_arguments = {}
        tool_name = chunk.tool_name
        logger.info("Handling tool call: %s", tool_name)

        tool_call_message = f"Calling {tool_name}"
        if input_arguments:
            tool_call_message += " with arguments:\n" + "".join(
                [f"   {key}: {value}\n" for key, value in input_arguments.items()]
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
