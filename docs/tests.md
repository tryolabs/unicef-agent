# Tests Documentation

## Overview

The UNICEF Agent project contains comprehensive unit and integration tests to ensure the reliability of the FastAPI server, LLM integration, and message handling functionality.

## Test Structure

### Test Files

- **`test_agent.py`** - Tests LLM initialization and agent creation
- **`test_config.py`** - Tests configuration loading and validation
- **`test_handlers.py`** - Tests message handling, formatting, and stream processing
- **`test_logging.py`** - Tests logging configuration and setup
- **`test_server.py`** - Tests FastAPI server endpoints and responses

### Test Categories

#### Unit Tests

- **Agent Tests**: LLM configuration, model initialization, agent workflow creation
- **Configuration Tests**: Config loading, validation, environment variable handling
- **Handler Tests**: Message formatting, stream processing, response handling
- **Server Tests**: FastAPI endpoint testing, request/response validation

#### Integration Tests

- **Server Integration**: End-to-end API endpoint testing with FastAPI TestClient
- **Agent Workflow**: Testing complete agent interaction flows
- **MCP Integration**: Testing MCP tool integration and communication

## Key Test Coverage

- ✅ LLM initialization with different models (OpenAI, Google)
- ✅ FastAPI server endpoints (`/`, `/ask`)
- ✅ Message formatting and stream processing
- ✅ Configuration validation and error handling
- ✅ Logging setup and configuration
- ✅ Async response handling and streaming
- ✅ Authentication and environment variables

## Running Tests

### Prerequisites

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Run All Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=agent --cov-report=html

# Run specific test file
uv run pytest tests/test_agent.py

# Run specific test class
uv run pytest tests/test_server.py::TestServer

# Run with verbose output
uv run pytest -v
```

### Environment Variables

Ensure these environment variables are set for testing:

- `MODEL_NAME` - LLM model to use
- `OPENAI_API_KEY` - OpenAI API key (for OpenAI models)
- `LANGFUSE_PROJECT_ID` - Langfuse project ID for tracing

## Test Configuration

Tests are configured via `pyproject.toml`:

- Test path: `tests/`
- Python path includes: `[".", "agent"]`
- Coverage excludes test files and `__init__.py`
- Async testing supported via `pytest-asyncio`
