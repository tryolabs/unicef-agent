import os
from pathlib import Path

import yaml
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from logging_config import get_logger
from schemas import MCPConfig, Prompts

logger = get_logger(__name__)


def initialize_app() -> None:
    """Initialize the app."""
    logger.info("Initializing app")
    set_env_vars()


def set_env_vars() -> None:
    """Set the environment variables.

    Raises:
        ValueError: If the environment variables are not set.
    """
    with Path("/run/secrets/langfuse_public_key").open("r") as f:
        langfuse_public_key = f.read()
    with Path("/run/secrets/langfuse_secret_key").open("r") as f:
        langfuse_secret_key = f.read()
    with Path("/run/secrets/langfuse_host").open("r") as f:
        langfuse_host = f.read()
    with Path("/run/secrets/openai_api_key").open("r") as f:
        openai_api_key = f.read()
    with Path("/run/secrets/jwt_secret_key").open("r") as f:
        jwt_secret_key = f.read()
    with Path("/run/secrets/users").open("r") as f:
        users = f.read()
    os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = langfuse_host
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["JWT_SECRET_KEY"] = jwt_secret_key
    os.environ["USERS"] = users


async def get_tools(mcp_config: MCPConfig) -> list[FunctionTool]:
    """Get the tools for the agent.

    Args:
        mcp_config: The configuration to use for the MCP clients

    Returns:
        list[FunctionTool]: The tools for the agent
    """
    mcp_client_datawarehouse = BasicMCPClient(mcp_config.datawarehouse_url)
    datawarehouse_tools = McpToolSpec(
        client=mcp_client_datawarehouse,
    )
    logger.info("Connecting to datawarehouse")
    mcp_client_rag = BasicMCPClient(mcp_config.rag_url)
    rag_tools = McpToolSpec(
        client=mcp_client_rag,
    )
    logger.info("Connecting to rag")
    mcp_client_geospatial = BasicMCPClient(mcp_config.geospatial_url)
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
