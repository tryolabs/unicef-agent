from __future__ import annotations

from llama_index.core.tools.function_tool import FunctionTool
from logging_config import get_logger

logger = get_logger(__name__)


def add(a: float, b: float) -> dict[str, float | dict[str, float]]:
    """Return the sum of a and b."""
    logger.info("Adding %s and %s", a, b)
    return {"result": float(a) + float(b), "input_arguments": {"a": a, "b": b}}


def subtract(a: float, b: float) -> dict[str, float | dict[str, float]]:
    """Return the difference a - b."""
    logger.info("Subtracting %s and %s", a, b)
    return {"result": float(a) - float(b), "input_arguments": {"a": a, "b": b}}


def multiply(a: float, b: float) -> dict[str, float | dict[str, float]]:
    """Return the product a * b."""
    logger.info("Multiplying %s and %s", a, b)
    return {"result": float(a) * float(b), "input_arguments": {"a": a, "b": b}}


def divide(a: float, b: float) -> dict[str, float | dict[str, float]]:
    """Return the quotient a / b.

    Raises:
        ValueError: If b == 0
    """
    logger.info("Dividing %s by %s", a, b)
    if float(b) == 0.0:
        msg = "Division by zero is not allowed"
        logger.error(msg)
        raise ValueError(msg)
    return {"result": float(a) / float(b), "input_arguments": {"a": a, "b": b}}


def percentage(part: float, whole: float) -> dict[str, float | dict[str, float]]:
    """Return the percentage that 'part' is of 'whole'.

    Example: percentage(2, 8) => 25.0

    Raises:
        ValueError: If whole == 0
    """
    logger.info("Calculating percentage of %s by %s", part, whole)
    if float(whole) == 0.0:
        msg = "Percentage of zero is undefined"
        logger.error(msg)
        raise ValueError(msg)
    return {
        "result": (float(part) / float(whole)) * 100.0,
        "input_arguments": {"part": part, "whole": whole},
    }


def get_calculator_tools() -> list[FunctionTool]:
    """Return a list of FunctionTool instances for calculator operations."""
    tools: list[FunctionTool] = [
        FunctionTool.from_defaults(fn=add, name="add", description="Add two numbers: a + b"),
        FunctionTool.from_defaults(
            fn=subtract, name="subtract", description="Subtract two numbers: a - b"
        ),
        FunctionTool.from_defaults(
            fn=multiply, name="multiply", description="Multiply two numbers: a * b"
        ),
        FunctionTool.from_defaults(
            fn=divide, name="divide", description="Divide two numbers: a / b (no zero divisor)"
        ),
        FunctionTool.from_defaults(
            fn=percentage,
            name="percentage",
            description="Compute percentage: (part / whole) * 100",
        ),
    ]
    return tools
