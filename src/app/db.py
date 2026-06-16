"""PostgreSQL connection helpers (psycopg v3)."""
from __future__ import annotations

import psycopg

from .config import DATABASE_URL


def get_connection() -> psycopg.Connection:
    """Open a new connection. Use as a context manager:

    with get_connection() as conn:
        ...
    """
    return psycopg.connect(DATABASE_URL)
