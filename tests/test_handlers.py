import uuid

import pytest
from handlers import (
    _format_messages,  # type: ignore[attr-defined]
    _process_agent_stream_chunk,  # type: ignore[attr-defined]
    _process_final_answer,  # type: ignore[attr-defined]
    _process_stop_event,  # type: ignore[attr-defined]
    _process_tool_call_chunk,  # type: ignore[attr-defined]
    handle_response,
    respond,
)
from llama_index.core.agent.workflow import AgentOutput, AgentStream, ToolCallResult
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import ToolOutput
from schemas import Config, LLMConfig, MCPConfig, Message, ReturnChunk, ServerConfig


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
        result = _process_agent_stream_chunk(chunk, trace_id)

        assert isinstance(result, ReturnChunk)
        assert result.response == "This is a test response"
        assert result.trace_id == trace_id
        assert result.tool_call == ""
        assert result.is_finished is False

    def test_process_agent_stream_chunk_with_closing_brace(self) -> None:
        """Test _process_agent_stream_chunk function with closing brace."""
        chunk = AgentStream(
            delta="action input}", response="", current_agent_name="", tool_calls=[], raw=""
        )
        trace_id = uuid.uuid4().hex
        result = _process_agent_stream_chunk(chunk, trace_id)

        assert isinstance(result, ReturnChunk)
        assert result.response == "action input}\n"
        assert result.trace_id == trace_id
        assert result.tool_call == ""
        assert result.is_finished is False

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
        assert result.tool_call.startswith("Calling search_tool with arguments:")
        assert "query: test query" in result.tool_call
        assert "limit: 10" in result.tool_call
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

    def _create_test_config(self) -> Config:
        """Create a test configuration."""
        return Config(
            server=ServerConfig(host="localhost", port=8000),
            mcp=MCPConfig(
                datawarehouse_url="http://127.0.0.1:6000/sse", rag_url="http://127.0.0.1:6001/sse"
            ),
            llm=LLMConfig(model="gpt-4o-mini", temperature=0.0),
        )

    @pytest.mark.asyncio
    async def test_handle_response_with_empty_messages(self) -> None:
        """Test handle_response function with empty messages."""
        messages: list[Message] = []
        trace_id = uuid.uuid4().hex
        session_id = "test-session-id"
        config = self._create_test_config()

        chunks = [event async for event in handle_response(messages, trace_id, session_id, config)]

        # Should return some response even with empty messages
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_handle_response_with_single_message(self) -> None:
        """Test handle_response function with a single message."""
        messages = [
            Message(role="user", content="Hello", trace_id=uuid.uuid4().hex),
        ]
        trace_id = uuid.uuid4().hex
        session_id = "test-session-id"
        config = self._create_test_config()

        chunks = [event async for event in handle_response(messages, trace_id, session_id, config)]
        # Should return some response
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_handle_response_with_multiple_messages(self) -> None:
        """Test handle_response function with multiple messages."""
        messages = [
            Message(role="user", content="Hello", trace_id=uuid.uuid4().hex),
            Message(role="assistant", content="Hi there!", trace_id=uuid.uuid4().hex),
            Message(role="user", content="How are you?", trace_id=uuid.uuid4().hex),
        ]
        trace_id = uuid.uuid4().hex
        session_id = "test-session-id"
        config = self._create_test_config()

        chunks = [event async for event in handle_response(messages, trace_id, session_id, config)]

        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_respond_with_formatted_messages(self) -> None:
        """Test respond function with properly formatted messages."""
        formatted_messages = {
            "messages": [{"role": "user", "content": "Hello", "trace_id": uuid.uuid4().hex}]
        }
        trace_id = uuid.uuid4().hex
        session_id = "test-session-id"
        config = self._create_test_config()

        chunks = [
            event async for event in respond(formatted_messages, trace_id, session_id, config)
        ]
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_respond_with_empty_formatted_messages(self) -> None:
        """Test respond function with empty formatted messages."""
        formatted_messages: dict[str, list[dict[str, str]]] = {"messages": []}
        trace_id = uuid.uuid4().hex
        session_id = "test-session-id"
        config = self._create_test_config()

        chunks = [
            event async for event in respond(formatted_messages, trace_id, session_id, config)
        ]

        assert isinstance(chunks, list)
        assert len(chunks) > 0
