"""Authentication: report current platform state without inventing endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_jwt_dependency_not_implemented_yet():
    """Backend JWT guard is still a TODO; there is no protected route returning 401."""
    security_path = Path(__file__).resolve().parents[2] / "app" / "core" / "security.py"
    text = security_path.read_text(encoding="utf-8")
    assert "TODO" in text
    assert "def get_current_user" not in text
