import pytest
from calculator import add, divide, multiply, percentage, subtract


class TestAdd:
    EXPECTED_SUM = 5.0
    WRONG_SUM = 6.0

    def test_add_right(self) -> None:
        result = add(2, 3)
        assert result["result"] == self.EXPECTED_SUM
        assert result["input_arguments"] == {"a": 2, "b": 3}

    def test_add_wrong(self) -> None:
        result = add(2, 3)
        assert result["result"] != self.WRONG_SUM

    def test_add_validation_error(self) -> None:
        with pytest.raises(ValueError, match="(float|number)"):
            add("x", 2)  # type: ignore[arg-type]


class TestSubtract:
    EXPECTED_DIFF = 3.0
    WRONG_DIFF = 0.0

    def test_subtract_right(self) -> None:
        result = subtract(5, 2)
        assert result["result"] == self.EXPECTED_DIFF
        assert result["input_arguments"] == {"a": 5, "b": 2}

    def test_subtract_wrong(self) -> None:
        result = subtract(5, 2)
        assert result["result"] != self.WRONG_DIFF

    def test_subtract_validation_error(self) -> None:
        with pytest.raises(ValueError, match="(float|number)"):
            subtract("x", 1)  # type: ignore[arg-type]


class TestMultiply:
    EXPECTED_PROD = 8.0
    WRONG_PROD = 9.0

    def test_multiply_right(self) -> None:
        result = multiply(4, 2)
        assert result["result"] == self.EXPECTED_PROD
        assert result["input_arguments"] == {"a": 4, "b": 2}

    def test_multiply_wrong(self) -> None:
        result = multiply(4, 2)
        assert result["result"] != self.WRONG_PROD

    def test_multiply_validation_error(self) -> None:
        with pytest.raises(ValueError, match="(float|number)"):
            multiply("y", 2)  # type: ignore[arg-type]


class TestDivide:
    EXPECTED_QUOT = 5.0
    WRONG_QUOT = 6.0

    def test_divide_right(self) -> None:
        result = divide(10, 2)
        assert result["result"] == self.EXPECTED_QUOT
        assert result["input_arguments"] == {"a": 10, "b": 2}

    def test_divide_wrong(self) -> None:
        result = divide(10, 2)
        assert result["result"] != self.WRONG_QUOT

    def test_divide_validation_error_zero(self) -> None:
        with pytest.raises(ValueError, match="Division by zero"):
            divide(1, 0)


class TestPercentage:
    EXPECTED_PCT = 25.0
    WRONG_PCT = 20.0

    def test_percentage_right(self) -> None:
        result = percentage(2, 8)
        assert result["result"] == self.EXPECTED_PCT
        assert result["input_arguments"] == {"part": 2, "whole": 8}

    def test_percentage_wrong(self) -> None:
        result = percentage(2, 8)
        assert result["result"] != self.WRONG_PCT

    def test_percentage_validation_error_zero_whole(self) -> None:
        with pytest.raises(ValueError, match="Percentage of zero is undefined"):
            percentage(1, 0)
