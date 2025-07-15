from typing import Literal

from pydantic import BaseModel

RESPONSE_TYPE = Literal["numerical", "textual"]


class BechmarkQuestion(BaseModel):
    question: str
    variations: list[str] | None
    response_type: RESPONSE_TYPE
    answer: int | str


class Benchmark(BaseModel):
    questions: list[BechmarkQuestion]


class NumericalAnswer(BaseModel):
    value: int | None


class TextualAnswer(BaseModel):
    result: int
    justification: str


class TextualEvaluation(BaseModel):
    faithfulness: TextualAnswer
    completeness: TextualAnswer
    conciseness: TextualAnswer
