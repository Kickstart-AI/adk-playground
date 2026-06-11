"""Tests for the package scaffold."""

import adk_playground


def test_package_imports() -> None:
    """Confirm the package can be imported from the src layout."""
    assert adk_playground.__doc__
