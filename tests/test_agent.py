import asyncio
import os
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from schemas import Config, LLMConfig, MCPConfig, ServerConfig
from workflows.events import Event

from agent import create_agent, get_llm


class TestGetLLM:
    """Test cases for the get_llm function."""

    @patch.dict(
        os.environ,
        {
            "MODEL_NAME": "gpt-4o-mini",
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
            "LANGFUSE_PROJECT_ID": "test-project",
        },
    )
    @patch("agent.LiteLLM")
    def test_get_llm_with_default_model(self, mock_litellm: MagicMock) -> None:
        """Test get_llm function with default model configuration."""
        mock_instance = MagicMock()
        mock_litellm.return_value = mock_instance

        llm_config = LLMConfig(model="gpt-4o-mini", temperature=0.5, provider="openai")

        result = get_llm(llm_config)

        mock_litellm.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.5,
            additional_kwargs={"stop": ["Observation:"]},
        )
        assert result == mock_instance

    @patch.dict(
        os.environ,
        {
            "MODEL_NAME": "gpt-4",
            "LANGFUSE_PROJECT_ID": "custom-project",
        },
    )
    @patch("agent.LiteLLM")
    def test_get_llm_with_custom_model(self, mock_litellm: MagicMock) -> None:
        """Test get_llm function with custom model configuration."""
        mock_instance = MagicMock()
        mock_litellm.return_value = mock_instance

        llm_config = LLMConfig(model="gpt-4", temperature=0.7, provider="openai")

        result = get_llm(llm_config)

        mock_litellm.assert_called_once_with(
            model="gpt-4",
            temperature=0.7,
            additional_kwargs={"stop": ["Observation:"]},
        )
        assert result == mock_instance


class TestCreateAgent:
    """Test cases for the create_agent function."""

    @pytest.fixture
    def sample_config(self) -> Config:
        """Sample configuration for testing."""
        return Config(
            server=ServerConfig(host="localhost", port=8000),
            mcp=MCPConfig(
                datawarehouse_url="http://localhost:3001",
                rag_url="http://localhost:3002",
                geospatial_url="http://localhost:3003",
            ),
            llm=LLMConfig(model="gpt-4o-mini", temperature=0.0, provider="openai"),
        )

    @patch("agent.get_llm")
    @patch("agent.get_tools")
    @patch("agent.get_prompts")
    @patch("agent.ReActAgent")
    @pytest.mark.asyncio
    def test_create_agent_with_default_parameters(
        self,
        mock_react_agent: MagicMock,
        mock_get_prompts: MagicMock,
        mock_get_tools: MagicMock,
        mock_get_llm: MagicMock,
        sample_config: Config,
    ) -> None:
        """Test create_agent function with default parameters."""
        mock_prompts = {
            "system_prompt": "You are a helpful assistant.",
            "header_prompt": "Think step by step.",
        }

        mock_prompts_obj = MagicMock()
        mock_prompts_obj.system_prompt = mock_prompts["system_prompt"]
        mock_prompts_obj.header_prompt = mock_prompts["header_prompt"]
        mock_get_prompts.return_value = mock_prompts_obj

        mock_tools = [MagicMock(), MagicMock()]
        mock_get_tools.return_value = mock_tools

        mock_llm_instance = MagicMock()
        mock_get_llm.return_value = mock_llm_instance

        mock_agent_instance = MagicMock()
        mock_react_agent.return_value = mock_agent_instance

        result = asyncio.run(create_agent(sample_config))

        mock_get_llm.assert_called_once_with(sample_config.llm)
        mock_get_tools.assert_called_once_with(sample_config.mcp)
        mock_react_agent.assert_called_once_with(
            tools=mock_tools,
            llm=mock_llm_instance,
            system_prompt=mock_prompts["system_prompt"],
        )
        mock_agent_instance.update_prompts.assert_called_once()
        assert result == mock_agent_instance

    @patch("agent.get_llm")
    @patch("agent.get_tools")
    @patch("agent.get_prompts")
    @patch("agent.ReActAgent")
    @pytest.mark.asyncio
    async def test_create_agent_with_custom_temperature(
        self,
        mock_react_agent: MagicMock,
        mock_get_prompts: MagicMock,
        mock_get_tools: MagicMock,
        mock_get_llm: MagicMock,
    ) -> None:
        """Test create_agent function with custom temperature."""
        mock_prompts = {
            "system_prompt": "You are a helpful assistant.",
            "header_prompt": "Think step by step.",
        }

        config = Config(
            server=ServerConfig(host="localhost", port=8000),
            mcp=MCPConfig(
                datawarehouse_url="http://localhost:3001",
                rag_url="http://localhost:3002",
                geospatial_url="http://localhost:3003",
            ),
            llm=LLMConfig(model="gpt-4", temperature=0.8, provider="openai"),
        )

        mock_prompts_obj = MagicMock()
        mock_prompts_obj.system_prompt = mock_prompts["system_prompt"]
        mock_prompts_obj.header_prompt = mock_prompts["header_prompt"]
        mock_get_prompts.return_value = mock_prompts_obj

        mock_tools = [MagicMock(), MagicMock()]
        mock_get_tools.return_value = mock_tools

        mock_llm_instance = MagicMock()
        mock_get_llm.return_value = mock_llm_instance

        mock_agent_instance = MagicMock()
        mock_react_agent.return_value = mock_agent_instance

        result = await create_agent(config)

        mock_get_llm.assert_called_once_with(config.llm)
        mock_get_tools.assert_called_once_with(config.mcp)
        mock_react_agent.assert_called_once_with(
            tools=mock_tools,
            llm=mock_llm_instance,
            system_prompt=mock_prompts["system_prompt"],
        )
        mock_agent_instance.update_prompts.assert_called_once()
        assert result == mock_agent_instance

    @patch("agent.langfuse")
    @patch("agent.run_agent")
    @pytest.mark.asyncio
    async def test_run_agent(self, mock_run_agent: AsyncMock, mock_langfuse: MagicMock) -> None:
        """Test run_agent function."""
        mock_events = [
            {"event": "chunk1", "data": "test1"},
            {"event": "chunk2", "data": "test2"},
            {"event": "final", "data": "complete"},
        ]

        async def mock_async_generator() -> AsyncGenerator[Event, None]:
            for event in mock_events:
                yield cast("Event", event)

        mock_run_agent.return_value = mock_async_generator()

        mock_agent = MagicMock()

        inputs = {"messages": [{"role": "user", "content": "Hello"}]}
        trace_id = "test-trace-id"
        session_id = "test-session-id"

        results = [
            event async for event in mock_run_agent(mock_agent, inputs, trace_id, session_id)
        ]

        mock_run_agent.assert_called_once_with(mock_agent, inputs, trace_id, session_id)

        assert len(results) == len(mock_events)
        assert results == mock_events
