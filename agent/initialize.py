from pathlib import Path

import yaml
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from schemas import Prompts


async def get_tools() -> list[FunctionTool]:
    mcp_client_datawarehouse = BasicMCPClient("http://127.0.0.1:6000/sse")
    datawarehouse_tools = McpToolSpec(
        client=mcp_client_datawarehouse,
    )

    mcp_client_rag = BasicMCPClient("http://127.0.0.1:6001/sse")
    rag_tools = McpToolSpec(
        client=mcp_client_rag,
    )

    datawarehouse_tools_list = await datawarehouse_tools.to_tool_list_async()
    rag_tools_list = await rag_tools.to_tool_list_async()

    all_tools = [*datawarehouse_tools_list, *rag_tools_list]

    return all_tools


def get_prompts() -> Prompts:
    with Path("agent/prompts.yaml").open("r") as f:
        prompts = yaml.safe_load(f)

    header_prompt = prompts["header_prompt"]
    system_prompt = prompts["system_prompt"]

    return Prompts(header_prompt=header_prompt, system_prompt=system_prompt)
