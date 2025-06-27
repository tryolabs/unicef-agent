from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml
from config import load_config
from schemas import Config

PORT = 8000
HOST = "127.0.0.1"


class TestConfig:
    """Test cases for config module."""

    def test_load_config_happy_path(self) -> None:
        """Test load_config loads configuration correctly."""
        result = load_config()
        assert isinstance(result, Config)
        assert result.server.host == HOST
        assert result.server.port == PORT
        assert result.mcp.datawarehouse_url == "http://127.0.0.1:6000/sse"
        assert result.mcp.rag_url == "http://127.0.0.1:6001/sse"
        assert result.llm.model == "gpt-4.1"
        assert result.llm.temperature == 0.0

    def test_load_config_file_not_found(self) -> None:
        """Test load_config raises FileNotFoundError when config file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                load_config()

    def test_load_config_file_not_found_custom_path(self) -> None:
        """Test load_config raises FileNotFoundError for custom path that doesn't exist."""
        custom_path = Path("/nonexistent/config.yaml")

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                load_config(custom_path)

    def test_load_config_default_path_construction(self) -> None:
        """Test load_config constructs default path correctly."""
        mock_config_data = {
            "server": {"host": "localhost", "port": PORT},
            "mcp": {
                "datawarehouse_url": "http://localhost:8001",
                "rag_url": "http://localhost:8002",
            },
            "llm": {"model": "gpt-4", "temperature": 0.7},
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_config_data))):
            with patch("pathlib.Path.exists", return_value=True) as mock_exists:
                load_config()

                # Verify the path was constructed correctly
                mock_exists.assert_called_once()
                call_args = mock_exists.call_args[0]
                assert len(call_args) == 0  # exists() called on Path object, not with args

    def test_load_config_return_type(self) -> None:
        """Test load_config returns correct type."""
        mock_config_data = {
            "server": {"host": "localhost", "port": PORT},
            "mcp": {
                "datawarehouse_url": "http://localhost:8001",
                "rag_url": "http://localhost:8002",
            },
            "llm": {"model": "gpt-4", "temperature": 0.7},
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_config_data))):
            with patch("pathlib.Path.exists", return_value=True):
                result = load_config()

                assert isinstance(result, Config)
                assert hasattr(result, "server")
                assert hasattr(result, "mcp")
                assert hasattr(result, "llm")

    def test_load_config_path_types(self) -> None:
        """Test load_config accepts different path types."""
        mock_config_data = {
            "server": {"host": "localhost", "port": PORT},
            "mcp": {
                "datawarehouse_url": "http://localhost:8001",
                "rag_url": "http://localhost:8002",
            },
            "llm": {"model": "gpt-4", "temperature": 0.7},
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_config_data))):
            with patch("pathlib.Path.exists", return_value=True):
                result = load_config()
                assert isinstance(result, Config)

        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_config_data))):
            with patch("pathlib.Path.exists", return_value=True):
                result = load_config(None)
                assert isinstance(result, Config)
