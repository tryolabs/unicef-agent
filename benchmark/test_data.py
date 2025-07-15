from typing import cast

from config import config
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.litellm import LiteLLM

from benchmark.questions import gee_questions, technical_doc_questions, warehouse_questions
from benchmark.schemas import (
    RESPONSE_TYPE,
    BechmarkQuestion,
    Benchmark,
    NumericalAnswer,
    TextualEvaluation,
)


def extract_number_from_response(question: str, answer: str, prompt: str) -> int | None:
    """Extract numerical answer from response using an LLM.

    Args:
        question: The question
        answer: The answer text
        prompt: The prompt to use for the LLM
    Returns:
        The extracted number or None if no number is found
    """
    model = LiteLLM(model=config.llm.model)

    program = LLMTextCompletionProgram.from_defaults(  # type: ignore[call-arg]
        llm=model,
        prompt_template_str=prompt,
        output_cls=NumericalAnswer,
    )

    result = cast("NumericalAnswer", program(question=question, answer=answer))

    return result.value


def score_textual_answer(
    question: str, ground_truth: str, candidate_answer: str, prompt: str
) -> TextualEvaluation:
    """Evaluate the quality of a textual answer against a ground truth.

    Args:
        question: The original question
        ground_truth: The ground truth answer (considered ideal and accurate)
        candidate_answer: The answer to evaluate
        prompt: The prompt to use for the LLM

    Returns:
        A dictionary containing evaluation scores and justifications for faithfulness,
        completeness, and conciseness on a scale of 1-5
    """
    model = LiteLLM(model=config.llm.model)

    program = LLMTextCompletionProgram.from_defaults(  # type: ignore[call-arg]
        llm=model,
        prompt_template_str=prompt,
        output_cls=TextualEvaluation,
    )

    result = cast(
        "TextualEvaluation",
        program(
            question=question,
            ground_truth=ground_truth,
            candidate_answer=candidate_answer,
        ),
    )

    return result


def benchmark_question_to_tuple(
    bq: BechmarkQuestion,
) -> list[tuple[str, str | int, RESPONSE_TYPE, str]]:
    result: list[tuple[str, str | int, RESPONSE_TYPE, str]] = []
    questions = [bq.question]
    if bq.variations:
        questions += bq.variations
    for q in questions:
        instance = (q, bq.answer, bq.response_type, bq.question)
        result.append(instance)

    return result


def benchmark_to_list(benchmark: Benchmark) -> list[tuple[str, str | int, RESPONSE_TYPE, str]]:
    result: list[tuple[str, str | int, RESPONSE_TYPE, str]] = []
    for bq in benchmark.questions:
        instance = benchmark_question_to_tuple(bq)
        result.extend(instance)
    return result


benchmark_questions = [
    *technical_doc_questions,
    *gee_questions,
    *warehouse_questions,
]
benchmark = Benchmark(questions=benchmark_questions)
benchmark_list = benchmark_to_list(benchmark)
