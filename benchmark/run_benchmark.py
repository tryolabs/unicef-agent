import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langfuse import Langfuse

# Add the agent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

import pytest
import yaml
from dotenv import load_dotenv
from handlers import handle_response
from logging_config import get_logger
from schemas import Message

from benchmark.test_data import (
    benchmark_list,
    extract_number_from_response,
    score_textual_answer,
)

load_dotenv(override=True)

langfuse = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)

logger = get_logger(__name__)


# Define path for test results and session ID file
RESULTS_PATH = Path("benchmark/results")

if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir(parents=True)

# Use a file to share session ID across processes
SESSION_FILE = RESULTS_PATH / ".session_id"

# Read existing session ID or create a new one
try:
    if SESSION_FILE.exists():
        with SESSION_FILE.open("r") as f:
            session_id = f.read().strip()
            logger.info("Using existing session ID: %s", session_id)
    else:
        session_id = str(uuid.uuid4())
        with SESSION_FILE.open("w") as f:
            f.write(session_id)
            logger.info("Created new session ID: %s", session_id)
except Exception:
    logger.exception("Error handling session ID file")
    session_id = str(uuid.uuid4())
    logger.info("Using fallback session ID: %s", session_id)

# Create results file
timestamp = datetime.now(UTC).strftime("%Y%m%d_%H:%M")

NUMERICAL_RESULTS_FILE = Path(f"{RESULTS_PATH}/numerical/results_{timestamp}.tsv")
TEXTUAL_RESULTS_FILE = Path(f"{RESULTS_PATH}/textual/results_{timestamp}.tsv")
for file in [NUMERICAL_RESULTS_FILE, TEXTUAL_RESULTS_FILE]:
    if not file.parent.exists():
        file.parent.mkdir(parents=True)
    if file.exists():
        file.unlink()

with NUMERICAL_RESULTS_FILE.open("w") as fh:
    logger.info("Writing numerical results to %s", NUMERICAL_RESULTS_FILE)
    fh.write("correct\tquestion\texpected\tvalue\tanswer\n")
with TEXTUAL_RESULTS_FILE.open("w") as fh:
    logger.info("Writing textual results to %s", TEXTUAL_RESULTS_FILE)
    fh.write(
        "question\texpected\tanswer\tfaithfulness_score\t"
        "faithfulness_justification\tcompleteness_score\tcompleteness_justification\t"
        "conciseness_score\tconciseness_justification\n"
    )

with Path("agent/prompts.yaml").open("r") as f:
    prompts = yaml.safe_load(f)

extract_number_prompt = prompts["extract_number_prompt"]
score_textual_answer_prompt = prompts["score_textual_answer_prompt"]


@pytest.mark.parametrize(("question", "expected", "response_type"), benchmark_list)
@pytest.mark.asyncio
async def test_agent_question(question: str, expected: str | int, response_type: str) -> None:
    """Test agent with a specific question."""
    trace_id = uuid.uuid4().hex
    message = Message(role="user", content=question, trace_id=trace_id)

    final_answer = ""
    async for chunk in handle_response([message], trace_id, session_id, tags=["benchmark"]):
        try:
            json_chunk = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if json_chunk.get("is_final_answer", False):
            final_answer += json_chunk.get("response", "")
            break
    if response_type == "numerical":
        evaluate_numerical_answer(trace_id, question, float(expected), final_answer)
    else:
        evaluate_textual_answer(trace_id, question, str(expected), final_answer)


def evaluate_numerical_answer(trace_id: str, question: str, expected: float, answer: str) -> None:
    numerical_value = extract_number_from_response(question, answer, extract_number_prompt)
    if numerical_value is None:
        is_correct = False
    else:
        tolerance = expected * 0.01
        is_correct = abs(numerical_value - expected) <= tolerance
    with NUMERICAL_RESULTS_FILE.open("a+") as fh:
        answer = answer.replace("\n", "||")
        fh.write(f"{is_correct}\t{question}\t{expected}\t{numerical_value}\t{answer}\n")

    langfuse.create_score(
        trace_id=trace_id,
        name="answer_correct",
        value="correct" if is_correct else "incorrect",
        comment=f"Expected: {expected}, Got: {answer}",
    )

    if not is_correct:
        pytest.fail(
            f"Answer doesn't match expected value for question: {question}\n"
            f"Expected: {expected}\nGot: {numerical_value}\n"
            f"Full answer: {answer}"
        )


def evaluate_textual_answer(trace_id: str, question: str, expected: str, answer: str) -> None:
    result = score_textual_answer(question, expected, answer, score_textual_answer_prompt)

    with TEXTUAL_RESULTS_FILE.open("a+") as fh:
        fh.write(
            f"{question}\t{expected}\t{answer}\t"
            f"{result.faithfulness.result}\t{result.faithfulness.justification}\t"
            f"{result.completeness.result}\t{result.completeness.justification}\t"
            f"{result.conciseness.result}\t{result.conciseness.justification}\n"
        )

    langfuse.create_score(
        trace_id=trace_id,
        name="faithfulness",
        value=result.faithfulness.result,
        comment=result.faithfulness.justification,
    )

    langfuse.create_score(
        trace_id=trace_id,
        name="completeness",
        value=result.completeness.result,
        comment=result.completeness.justification,
    )

    langfuse.create_score(
        trace_id=trace_id,
        name="conciseness",
        value=result.conciseness.result,
        comment=result.conciseness.justification,
    )
