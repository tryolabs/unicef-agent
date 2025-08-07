from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import status
from fastapi.testclient import TestClient
from server import User, app, get_current_user

# Override authentication dependency for tests


def override_get_current_user() -> User:  # type: ignore[valid-type]
    return User(username="test-user")


app.dependency_overrides[get_current_user] = override_get_current_user


class TestServer:
    """Test cases for server module."""

    def setup_method(self) -> None:
        """Set up test client for each test."""
        self.client = TestClient(app)

    def test_root_endpoint_returns_correct_response(self) -> None:
        """Test that root endpoint returns the expected message."""
        response = self.client.get("/")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"message": "Hello World"}

    def test_root_endpoint_response_type(self) -> None:
        """Test that root endpoint returns the correct response type."""
        response = self.client.get("/")

        assert isinstance(response.json(), dict)
        assert "message" in response.json()
        assert isinstance(response.json()["message"], str)

    @patch("server.handle_response")
    @patch("server.uuid.uuid4")
    def test_ask_endpoint_happy_path(
        self, mock_uuid: MagicMock, mock_handle_response: AsyncMock
    ) -> None:
        """Test ask endpoint with valid chat messages."""
        mock_uuid.return_value.hex = "test-trace-id"

        async def mock_async_generator() -> AsyncGenerator[bytes, None]:
            yield b"data: test response chunk 1\n"
            yield b"data: test response chunk 2\n"

        mock_handle_response.return_value = mock_async_generator()

        chat_data = {
            "chat_messages": [
                {"content": "Hello, how are you?", "role": "user", "trace_id": "test-trace"}
            ],
            "session_id": "test-session-123",
        }

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        mock_handle_response.assert_called_once()

    def test_ask_endpoint_empty_chat_messages(self) -> None:
        """Test ask endpoint with empty chat messages raises HTTPException."""
        chat_data: dict[str, Any] = {"chat_messages": [], "session_id": "test-session-123"}

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Chat messages cannot be empty" in response.json()["detail"]

    def test_ask_endpoint_empty_content(self) -> None:
        """Test ask endpoint with empty message content raises HTTPException."""
        chat_data = {
            "chat_messages": [{"content": "", "role": "user", "trace_id": "test-trace"}],
            "session_id": "test-session-123",
        }

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Chat messages cannot be empty" in response.json()["detail"]

    def test_ask_endpoint_invalid_role(self) -> None:
        """Test ask endpoint with invalid role in message."""
        chat_data = {
            "chat_messages": [
                {"content": "Hello", "role": "invalid_role", "trace_id": "test-trace"}
            ],
            "session_id": "test-session-123",
        }

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ask_endpoint_missing_required_fields(self) -> None:
        """Test ask endpoint with missing required fields."""
        chat_data = {
            "chat_messages": [
                {
                    "content": "Hello",
                }
            ],
            "session_id": "test-session-123",
        }

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ask_endpoint_missing_session_id(self) -> None:
        """Test ask endpoint with missing session_id."""
        chat_data = {
            "chat_messages": [{"content": "Hello", "role": "user", "trace_id": "test-trace"}]
        }

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("server.handle_response")
    @patch("server.uuid.uuid4")
    def test_ask_endpoint_multiple_messages(
        self, mock_uuid: MagicMock, mock_handle_response: AsyncMock
    ) -> None:
        """Test ask endpoint with multiple chat messages."""
        mock_uuid.return_value.hex = "test-trace-id"

        async def mock_async_generator() -> AsyncGenerator[bytes, None]:
            yield b"data: response\n"

        mock_handle_response.return_value = mock_async_generator()

        chat_data = {
            "chat_messages": [
                {"content": "Hello", "role": "user", "trace_id": "trace-1"},
                {"content": "Hi there!", "role": "assistant", "trace_id": "trace-2"},
                {"content": "How can I help you?", "role": "user", "trace_id": "trace-3"},
            ],
            "session_id": "test-session-123",
        }

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_200_OK
        mock_handle_response.assert_called_once()

        call_args = mock_handle_response.call_args[0]
        assert call_args[0][-1].content == "How can I help you?"

    @patch("server.handle_response")
    @patch("server.uuid.uuid4")
    def test_ask_endpoint_uuid_generation(
        self, mock_uuid: MagicMock, mock_handle_response: AsyncMock
    ) -> None:
        """Test that ask endpoint generates unique trace IDs."""
        mock_uuid.return_value.hex = "unique-trace-id-12345"

        async def mock_async_generator() -> AsyncGenerator[bytes, None]:
            return
            yield

        mock_handle_response.return_value = mock_async_generator()

        chat_data = {
            "chat_messages": [
                {"content": "Test message", "role": "user", "trace_id": "original-trace"}
            ],
            "session_id": "test-session",
        }

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_200_OK
        mock_uuid.assert_called_once()

        call_args = mock_handle_response.call_args[0]
        assert call_args[1] == "unique-trace-id-12345"

    def test_ask_endpoint_malformed_json(self) -> None:
        """Test ask endpoint with malformed JSON."""
        response = self.client.post(
            "/ask", content="invalid json", headers={"content-type": "application/json"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("server.handle_response")
    @patch("server.uuid.uuid4")
    def test_ask_endpoint_preserves_session_id(
        self, mock_uuid: MagicMock, mock_handle_response: AsyncMock
    ) -> None:
        """Test that ask endpoint preserves the session_id from request."""
        mock_uuid.return_value.hex = "test-trace-id"

        async def mock_async_generator() -> AsyncGenerator[bytes, None]:
            return
            yield

        mock_handle_response.return_value = mock_async_generator()

        test_session_id = "special-session-id-456"
        chat_data = {
            "chat_messages": [
                {"content": "Test message", "role": "user", "trace_id": "test-trace"}
            ],
            "session_id": test_session_id,
        }

        response = self.client.post("/ask", json=chat_data)

        assert response.status_code == status.HTTP_200_OK

        call_args = mock_handle_response.call_args[0]
        assert call_args[2] == test_session_id
