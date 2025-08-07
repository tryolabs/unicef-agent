import os
from pathlib import Path

import yaml
from config import config
from dotenv import load_dotenv
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from logging_config import get_logger
from schemas import MCPConfig, Prompts

logger = get_logger(__name__)


def initialize_app() -> None:
    """Initialize the app."""
    logger.info("Initializing app")
    load_dotenv(override=True)
    set_env_vars()


def _read_secret_or_env(secret_name: str, env_var_name: str) -> str:
    """Read from Docker secret file first, then fallback to environment variable.

    Args:
        secret_name: Name of the Docker secret file (without path)
        env_var_name: Name of the environment variable

    Returns:
        str: The secret value

    Raises:
        ValueError: If neither secret file nor environment variable is available
    """
    secret_path = Path(f"/run/secrets/{secret_name}")

    # First try Docker secrets
    if secret_path.exists():
        logger.debug("Reading %s from Docker secret", secret_name)
        return secret_path.read_text().strip()

    # Fallback to environment variable
    env_value = os.environ.get(env_var_name)
    if env_value:
        logger.debug("Reading %s from environment variable", env_var_name)
        return env_value.strip()

    msg = "Neither Docker secret '%s' nor environment variable '%s' is available"
    logger.error(msg, secret_name, env_var_name)
    raise ValueError(msg % (secret_name, env_var_name))


def set_llm_env_vars() -> None:
    """Set the environment variables for the LLM."""
    match config.llm.provider:
        case "openai":
            os.environ["OPENAI_API_KEY"] = _read_secret_or_env("openai_api_key", "OPENAI_API_KEY")
        case "bedrock":
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = _read_secret_or_env(
                "aws_bearer_token_bedrock", "AWS_BEARER_TOKEN_BEDROCK"
            )
        case "vertexai":
            vertex_auth_secret = Path("/run/secrets/vertex_auth.json")
            if vertex_auth_secret.exists():
                logger.debug("Using Docker secret for Google Cloud credentials")
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(vertex_auth_secret)
            elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                logger.debug("Using environment variable for Google Cloud credentials")
                # Environment variable already set, no action needed
            else:
                logger.warning(
                    "No Google Cloud credentials found.\
                    Set GOOGLE_APPLICATION_CREDENTIALS environment variable for local development."
                )


def set_env_vars() -> None:
    """Set the environment variables.

    This function works in both Docker (using secrets) and local environments (using env vars).

    Raises:
        ValueError: If the environment variables are not set.
    """
    try:
        # Read secrets with fallback to environment variables
        os.environ["LANGFUSE_PUBLIC_KEY"] = _read_secret_or_env(
            "langfuse_public_key", "LANGFUSE_PUBLIC_KEY"
        )
        os.environ["LANGFUSE_SECRET_KEY"] = _read_secret_or_env(
            "langfuse_secret_key", "LANGFUSE_SECRET_KEY"
        )
        os.environ["LANGFUSE_HOST"] = _read_secret_or_env("langfuse_host", "LANGFUSE_HOST")
        os.environ["JWT_SECRET_KEY"] = _read_secret_or_env("jwt_secret_key", "JWT_SECRET_KEY")
        os.environ["USERS_PATH"] = _read_secret_or_env("users", "USERS_PATH")

        set_llm_env_vars()

    except ValueError as e:
        msg = "Failed to set environment variables: %s"
        logger.exception(msg)
        raise ValueError(msg) from e


async def get_tools(mcp_config: MCPConfig | None = None) -> list[FunctionTool]:
    """Get the tools for the agent.

    Returns:
        list[FunctionTool]: The tools for the agent
    """
    if mcp_config is None:
        mcp_config = config.mcp

    mcp_client_datawarehouse = BasicMCPClient(config.mcp.datawarehouse_url)
    datawarehouse_tools = McpToolSpec(
        client=mcp_client_datawarehouse,
    )
    logger.info("Connecting to datawarehouse")
    mcp_client_rag = BasicMCPClient(config.mcp.rag_url)
    rag_tools = McpToolSpec(
        client=mcp_client_rag,
    )
    logger.info("Connecting to rag")
    mcp_client_geospatial = BasicMCPClient(config.mcp.geospatial_url)
    geospatial_tools = McpToolSpec(
        client=mcp_client_geospatial,
    )
    logger.info("Getting tools")
    datawarehouse_tools_list = await datawarehouse_tools.to_tool_list_async()
    logger.info("Got datawarehouse tools")
    rag_tools_list = await rag_tools.to_tool_list_async()
    logger.info("Got rag tools")
    geospatial_tools_list = await geospatial_tools.to_tool_list_async()
    logger.info("Got geospatial tools")

    all_tools = [
        *datawarehouse_tools_list,
        *rag_tools_list,
        *geospatial_tools_list,
    ]

    return all_tools


def get_prompts() -> Prompts:
    with Path("agent/prompts.yaml").open("r") as f:
        prompts = yaml.safe_load(f)

    header_prompt = prompts["header_prompt"]
    system_prompt = prompts["system_prompt"]

    return Prompts(header_prompt=header_prompt, system_prompt=system_prompt)
