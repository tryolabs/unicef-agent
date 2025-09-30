# UNICEF Technical Documentation Agent

The UNICEF Technical Documentation Agent is a FastAPI-based intelligent middleware that orchestrates communication between the frontend interface and multiple Model Context Protocol (MCP) servers. It serves as the central hub for processing natural language queries and coordinating responses from various data sources, including the Global Child Hazard Database - Technical Documentation.

### Core Components

- **LLM Integration**: Configurable language model via LiteLLM
- **MCP Orchestration**: Dynamic tool discovery and execution across data sources
- **Authentication**: JWT-based security with user management
- **Streaming Responses**: Real-time response delivery with tool call visibility
- **Observability**: Langfuse integration for monitoring and feedback

## Technology Stack

- **FastAPI**: Modern Python web framework for API development
- **LlamaIndex**: LLM application framework for workflow orchestration
- **LiteLLM**: Unified interface for multiple LLM providers
- **Langfuse**: LLM observability and analytics platform

## Project Structure

```
agent/
├── agent.py              # Core agent workflow and LLM setup
├── server.py             # FastAPI application and endpoints
├── handlers.py           # Request processing and response streaming
├── auth.py               # Authentication and user management
├── initialize.py         # MCP client setup and tool initialization
├── config.py             # Configuration loading and validation
├── schemas.py            # Pydantic models and type definitions
├── prompts.yaml          # System prompts and instructions
├── config.yaml           # Application configuration
└── logging_config.py     # Logging setup and configuration
```

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- Access to OpenAI API or compatible LLM service
- Running MCP servers (datawarehouse, RAG, GEE)

### Installation

```bash
# Install dependencies using uv
uv sync
```

### Configuration

The application supports environment-specific configuration files for better separation between development, production, and testing environments.

#### Environment-Based Configuration

The system automatically selects the appropriate configuration file based on the `ENVIRONMENT` environment variable:

- **Production** (`ENVIRONMENT=prod`): Uses `config-prod.yaml`
- **Default** (`ENVIRONMENT=dev`): Uses `config.yaml`

#### Configuration Files

**`agent/config.yaml`** (Development):

```yaml
llm:
  model: "gpt-4o-mini" # LLM model to use
  temperature: 0.5 # Response creativity (0-1)

mcp:
  # MCP servers listen on 6000/6001/6002
  datawarehouse_url: "http://datawarehouse_mcp:6000/sse"
  rag_url: "http://rag_mcp:6001/sse"
  geospatial_url: "http://geospatial_mcp:6002/sse"

server:
  host: "0.0.0.0" # Server bind address
  port: 8000 # Agent API port
```

### Development

1. **Start the server**:

   ```bash
   uv run agent/server.py
   ```

2. **Access the API**:
   - Server: `http://localhost:8000`
   - Health check: `http://localhost:8000/`

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_agent.py -v
```

### Environment variables and secrets

Secrets are read from Docker secrets when available, falling back to environment variables (see `agent/initialize.py`). For local development, you can use a `.env` file.

Required configuration:

| Variable                         | Description                                    | When required          |
| -------------------------------- | ---------------------------------------------- | ---------------------- |
| `LANGFUSE_HOST`                  | Langfuse server URL                            | Always                 |
| `LANGFUSE_PUBLIC_KEY`            | Langfuse public key                            | Always                 |
| `LANGFUSE_SECRET_KEY`            | Langfuse secret key                            | Always                 |
| `JWT_SECRET_KEY`                 | Secret for JWT token signing                   | Always                 |
| `USERS_PATH`                     | Path to users JSON file OR literal JSON string | Always                 |
| `OPENAI_API_KEY`                 | OpenAI API key                                 | If provider=`openai`   |
| `AWS_BEARER_TOKEN_BEDROCK`       | AWS Bedrock bearer token                       | If provider=`bedrock`  |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Vertex AI/Google JSON credentials      | If provider=`vertexai` |

Docker secret file names (for Compose): `langfuse_host`, `langfuse_public_key`, `langfuse_secret_key`, `openai_api_key`, `jwt_secret_key`, `users`, `vertex_auth.json`, `aws_bearer_token_bedrock`.

### User management

Set `USERS_PATH` to either:

- An absolute path to a JSON file mounted in the container, or
- A literal JSON string containing the users array

Example content:

```json
[
  {
    "username": "admin",
    "hashed_password": "<sha256_of_password>"
  }
]
```

## API Documentation

### Authentication

**POST `/token`**

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your_password"
```

Response:

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "username": "admin"
}
```

### Main query endpoint

When calling the agent directly:

**POST `/ask`**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "chat_messages": [
      {"content": "How many children are at risk of floods in Colombia?", "role": "user"}
    ],
    "session_id": "unique-session-id"
  }'
```

**Streaming Response Format**:

The agent will stream a response with the following format. Each response chunk follows the `ReturnChunk` schema with these fields:

- **`trace_id`** (string): Unique identifier for tracking the request throughout the processing pipeline
- **`response`** (string): The actual text content being streamed to the user (empty for non-text chunks)
- **`is_thinking`** (boolean): Indicates if this is part of the agent's reasoning process (true) or final response (false)
- **`tool_call`** (string): Details of backend tool operations being performed (empty when not calling tools)
- **`is_finished`** (boolean): Indicates if this is the final chunk in the stream
- **`html_content`** (string): HTML content for map visualizations or rich media (empty for text-only responses)

```json
{"response": "I'll help you find information about flood risks...", "is_thinking": true, "trace_id": "abc123", "is_finished": false}
{"tool_call": {"name": "get_dataset_image", "args": {...}}, "trace_id": "abc123"}
{"response": "Based on the analysis...", "is_thinking": false, "trace_id": "abc123", "is_finished": true}
{"html_content": "<html>...</html>", "trace_id": "abc123"}
```

## MCP Integration

### Tool Discovery

The agent automatically discovers and loads tools from all configured MCP servers:

```python
# Tools are loaded asynchronously from each MCP server
datawarehouse_tools = await datawarehouse_mcp.to_tool_list_async()
rag_tools = await rag_mcp.to_tool_list_async()
geospatial_tools = await geospatial_mcp.to_tool_list_async()
```

### Available Tool Categories

1. **Data Warehouse Tools**:

   - `get_available_dataflows()`: List available statistical datasets
   - `get_all_indicators_for_dataflow(dataflow_id)`: Get indicators for a dataset
   - `get_data_for_dataflow(...)`: Query specific data with filters

2. **RAG Tools**:

   - `get_ccri_relevant_information(question)`: Search Global Child Hazard Database - Technical Documentation

3. **Geospatial Tools** (12 tools):
   - Dataset and metadata operations
   - Image processing and analysis
   - Feature collection operations
   - Map generation and visualization

## Security

### Authentication Flow

1. User submits credentials to `/token` endpoint
2. Server validates against configured users
3. JWT token issued with configurable expiration
4. Token required for all `/api/*` endpoints
5. Token validated on each request

### Security Best Practices

- **Password Storage**: SHA256 hashing
- **Token Security**: JWT with secret-based signing
- **Network Security**: Internal-only MCP communication
- **Input Validation**: Pydantic schemas for all inputs
- **Error Handling**: No sensitive information in error responses

## Monitoring & Observability

### Langfuse Integration

- **Trace Collection**: All LLM interactions tracked
- **Performance Monitoring**: Response times and token usage
- **User Feedback**: Integration with frontend feedback system
- **Error Tracking**: Detailed error logs and metrics

## Contributing

1. **Code Style**: Follow PEP 8 and use type hints
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update README and docstrings

### Development Workflow

1. Create feature branch
2. Add tests for new functionality
3. Ensure all tests pass: `uv run pytest`
4. Update documentation
5. Submit pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Submit issues on GitHub
- **Security**: Report security issues privately to maintainers
