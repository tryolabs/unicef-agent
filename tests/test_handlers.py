import uuid

from handlers import (
    _format_messages,  # type: ignore[attr-defined]
    _process_agent_stream_chunk,  # type: ignore[attr-defined]
    _process_final_answer,  # type: ignore[attr-defined]
    _process_stop_event,  # type: ignore[attr-defined]
    _process_tool_call_chunk,  # type: ignore[attr-defined]
)
from llama_index.core.agent.workflow import AgentOutput, AgentStream, ToolCallResult
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import ToolOutput
from schemas import Message, ReturnChunk


class TestHandlers:
    """Test cases for handlers module."""

    def test_format_messages(self) -> None:
        """Test format_messages function."""
        trace_id_1 = uuid.uuid4().hex
        trace_id_2 = uuid.uuid4().hex
        messages = [
            Message(role="user", content="Hello, how are you?", trace_id=trace_id_1),
            Message(role="assistant", content="I'm doing well, thank you!", trace_id=trace_id_2),
        ]
        formatted_messages = _format_messages(messages)
        assert formatted_messages == {
            "messages": [
                {"role": "user", "content": "Hello, how are you?", "trace_id": trace_id_1},
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you!",
                    "trace_id": trace_id_2,
                },
            ]
        }

    def test_format_empty_messages(self) -> None:
        """Test format_messages function with empty messages."""
        messages: list[Message] = []
        formatted_messages = _format_messages(messages)
        assert formatted_messages == {"messages": []}

    def test_process_agent_stream_chunk(self) -> None:
        """Test _process_agent_stream_chunk function."""
        chunk = AgentStream(
            delta="This is a test response",
            response="",
            current_agent_name="",
            tool_calls=[],
            raw="",
        )
        trace_id = uuid.uuid4().hex
        result = _process_agent_stream_chunk(chunk.delta, trace_id)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ReturnChunk)
        assert result[0].response == "This is a test response"
        assert result[0].trace_id == trace_id
        assert result[0].is_finished is False

    def test_process_agent_stream_chunk_with_closing_brace(self) -> None:
        """Test _process_agent_stream_chunk function with closing brace."""
        chunk = AgentStream(
            delta="action input}", response="", current_agent_name="", tool_calls=[], raw=""
        )
        trace_id = uuid.uuid4().hex
        result = _process_agent_stream_chunk(chunk.delta, trace_id)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_process_final_answer(self) -> None:
        """Test _process_final_answer function."""
        # Create a response object with content attribute
        response_obj = ChatMessage(content="This is the final answer")

        chunk = AgentOutput(
            response=response_obj,
            current_agent_name="",
            tool_calls=[],
            raw="",
        )
        trace_id = uuid.uuid4().hex
        result = _process_final_answer(chunk, trace_id)

        assert isinstance(result, ReturnChunk)
        assert result.response == "This is the final answer"
        assert result.trace_id == trace_id
        assert result.tool_call == ""
        assert result.is_finished is False

    def test_process_stop_event(self) -> None:
        """Test _process_stop_event function."""
        trace_id = uuid.uuid4().hex
        result = _process_stop_event(trace_id)

        assert isinstance(result, ReturnChunk)
        assert result.response == ""
        assert result.trace_id == trace_id
        assert result.tool_call == ""
        assert result.is_finished is True

    def test_process_tool_call_chunk(self) -> None:
        """Test _process_tool_call_chunk function."""
        tool_output = ToolOutput(
            content="{'input_arguments': {'query': 'test query', 'limit': '10'}}",
            tool_name="search_tool",
            raw_input={},
            raw_output={},
        )
        chunk = ToolCallResult(
            tool_name="search_tool",
            tool_kwargs={},
            tool_id="test-tool-id",
            tool_output=tool_output,
            return_direct=False,
        )

        trace_id = uuid.uuid4().hex
        result = _process_tool_call_chunk(chunk, trace_id)

        assert isinstance(result, ReturnChunk)
        assert result.trace_id == trace_id
        assert result.tool_call == "Calling search_tool"
        assert result.response == ""
        assert result.is_finished is False

    def test_process_tool_call_chunk_no_arguments(self) -> None:
        """Test _process_tool_call_chunk function with no arguments."""
        tool_output = ToolOutput(
            content="simple string content",
            tool_name="simple_tool",
            raw_input={},
            raw_output={},
        )

        chunk = ToolCallResult(
            tool_name="simple_tool",
            tool_kwargs={},
            tool_id="test-tool-id",
            tool_output=tool_output,
            return_direct=False,
        )

        trace_id = uuid.uuid4().hex
        result = _process_tool_call_chunk(chunk, trace_id)

        assert isinstance(result, ReturnChunk)
        assert result.trace_id == trace_id
        assert result.tool_call == "Calling simple_tool"
        assert result.response == ""
        assert result.is_finished is False
