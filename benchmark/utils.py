import csv
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api.resources.score_v_2.types.get_scores_response_data import GetScoresResponseData

from benchmark.schemas import Score

load_dotenv(override=True)

langfuse = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)


def convert_scores(scores: list[GetScoresResponseData]) -> list[Score]:
    """Convert Langfuse score response data to Score objects.

    Args:
        scores: List of GetScoresResponseData objects from Langfuse API

    Returns:
        List of Score objects with converted data
    """
    converted_scores: list[Score] = []

    for score in scores:
        converted_score = Score(
            name=score.name,
            value=score.value if score.value is not None else 0,
            updated_at=score.updated_at,
            string_value=getattr(score, "string_value", ""),
        )
        converted_scores.append(converted_score)

    return converted_scores


def get_langfuse_scores(days: int = 30) -> list[Score]:
    """Retrieve scores from Langfuse for a specified time period.

    Args:
        days: Number of days to look back from today (default: 30)

    Returns:
        List of Score objects containing all scores from the specified time period
    """
    # Get scores from the last month
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days)

    scores: list[GetScoresResponseData] = []
    page = 1
    limit = 100

    while True:
        scores_batch = langfuse.api.score_v_2.get(
            limit=limit, page=page, from_timestamp=start_date, to_timestamp=end_date
        )

        if not scores_batch.data:
            break
        scores.extend(scores_batch.data)
        page += 1

        # Break if we got fewer results than the limit (last page)
        if len(scores_batch.data) < limit:
            break

    return convert_scores(scores)


def _process_numerical_results(results_path: Path) -> list[Score]:
    """Process numerical TSV files and return answer_correctness scores."""
    scores: list[Score] = []
    numerical_path = results_path / "numerical"

    if not numerical_path.exists():
        return scores

    for file_path in numerical_path.glob("*.tsv"):
        timestamp_str = file_path.stem.replace("results_", "")
        file_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H:%M").replace(tzinfo=UTC)

        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                is_correct = row.get("correct", "").lower() == "true"
                score = Score(
                    name="answer_correctness",
                    value=1.0 if is_correct else 0.0,
                    updated_at=file_datetime,
                    string_value="correct" if is_correct else "incorrect",
                )
                scores.append(score)

    return scores


def _process_textual_results(results_path: Path) -> list[Score]:
    """Process textual TSV files and return faithfulness, completeness, and conciseness scores."""
    scores: list[Score] = []
    textual_path = results_path / "textual"

    if not textual_path.exists():
        return scores

    score_types = [
        ("faithfulness", "faithfulness_score"),
        ("completeness", "completeness_score"),
        ("conciseness", "conciseness_score"),
    ]

    for file_path in textual_path.glob("*.tsv"):
        timestamp_str = file_path.stem.replace("results_", "")
        file_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H:%M").replace(tzinfo=UTC)

        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                for score_name, column_name in score_types:
                    score_value_str = row.get(column_name, "0")
                    try:
                        score_value = float(score_value_str)
                    except (ValueError, TypeError):
                        score_value = 0.0

                    score = Score(
                        name=score_name,
                        value=score_value,
                        updated_at=file_datetime,
                        string_value="",
                    )
                    scores.append(score)

    return scores


def get_local_scores() -> list[Score]:
    """Retrieve scores from local TSV files in benchmark/results directory.

    Returns:
        List of Score objects containing all scores from the local result files
    """
    results_path = Path("benchmark/results")

    # Process both numerical and textual results
    numerical_scores = _process_numerical_results(results_path)
    textual_scores = _process_textual_results(results_path)

    return numerical_scores + textual_scores
