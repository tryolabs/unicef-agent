import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# Read the environment variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "secret")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str
    username: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str


class UserInDB(User):
    hashed_password: str


def get_users() -> list[dict[str, str]]:
    """Load users from the users.json file."""
    try:
        with Path("agent/users.json").open("r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Return empty list if file not found
        return []


def get_user(username: str) -> UserInDB | None:
    """Get a user by username."""
    users = get_users()
    for user in users:
        if user["username"] == username:
            return UserInDB(**user)
    return None


def verify_password(saved_hashed_password: str, password: str) -> bool:
    """Verify a password against its hash."""
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    return saved_hashed_password == hashed_password


def authenticate_user(username: str, password: str) -> User | None:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(user.hashed_password, password):
        return None
    return User(username=user.username)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current user from the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str | None = payload.get("sub")

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception from None

    if token_data.username is None:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception

    return User(username=user.username)
