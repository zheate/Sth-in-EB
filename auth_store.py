"""
Credential store helper that keeps bcrypt password hashes in a local JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

STORE_PATH = Path(__file__).parent / "config" / "users.json"
_LOCK = RLock()


@dataclass
class UserRecord:
    username: str
    password_hash: str
    role: str = "user"

    def to_dict(self) -> Dict[str, str]:
        return {
            "username": self.username,
            "password_hash": self.password_hash,
            "role": self.role,
        }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_store(path: Path = STORE_PATH) -> Dict[str, List[Dict[str, str]]]:
    """Load the credentials JSON, creating an empty structure if missing."""
    with _LOCK:
        if not path.exists():
            _ensure_parent(path)
            data = {"users": []}
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError(f"Credential store {path} is not valid JSON.")


def save_store(data: Dict[str, List[Dict[str, str]]], path: Path = STORE_PATH) -> None:
    """Persist credentials to disk."""
    with _LOCK:
        _ensure_parent(path)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_users(path: Path = STORE_PATH) -> List[UserRecord]:
    """Return all users from the store."""
    data = load_store(path)
    return [UserRecord(**entry) for entry in data.get("users", [])]


def get_user(username: str, path: Path = STORE_PATH) -> Optional[UserRecord]:
    """Fetch a single user by username."""
    for record in list_users(path):
        if record.username == username:
            return record
    return None


def upsert_user(record: UserRecord, path: Path = STORE_PATH) -> None:
    """Insert or update a user record (by username)."""
    with _LOCK:
        data = load_store(path)
        users = data.setdefault("users", [])
        for idx, existing in enumerate(users):
            if existing.get("username") == record.username:
                users[idx] = record.to_dict()
                break
        else:
            users.append(record.to_dict())
        save_store(data, path)


def remove_user(username: str, path: Path = STORE_PATH) -> bool:
    """Remove a user; returns True if one was deleted."""
    with _LOCK:
        data = load_store(path)
        original_len = len(data.get("users", []))
        data["users"] = [
            entry for entry in data.get("users", []) if entry.get("username") != username
        ]
        save_store(data, path)
    return len(data["users"]) < original_len


def ensure_store(path: Path = STORE_PATH) -> Path:
    """Guarantee that the credential store exists and return its path."""
    load_store(path)
    return path


