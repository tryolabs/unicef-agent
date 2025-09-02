from datetime import datetime
from typing import Literal

from pydantic import BaseModel

RESPONSE_TYPE = Literal["numerical", "textual"]


class BechmarkQuestion(BaseModel):
    question: str
    response_type: RESPONSE_TYPE
    answer: int | str | float


class Benchmark(BaseModel):
    questions: list[BechmarkQuestion]


class NumericalAnswer(BaseModel):
    value: int | float | None


class TextualAnswer(BaseModel):
    result: int
    justification: str


class TextualEvaluation(BaseModel):
    faithfulness: TextualAnswer
    completeness: TextualAnswer
    conciseness: TextualAnswer


class Score:
    """A score object representing evaluation metrics from Langfuse.

    Attributes:
        name: The name of the score metric
        value: The numeric value of the score
        updated_at: The datetime when the score was last updated
        string_value: Optional string representation of the score value
    """

    def __init__(self, name: str, value: float, updated_at: datetime, string_value: str) -> None:
        self.name = name
        self.value = value
        self.updated_at = updated_at
        self.string_value = string_value

    def __str__(self) -> str:
        """Return a string representation of the score."""
        if self.string_value:
            return f"{self.name}: {self.value} ({self.string_value})"
        else:
            return f"{self.name}: {self.value}"

    def __repr__(self) -> str:
        """Return a string representation of the score."""
        return self.__str__()
