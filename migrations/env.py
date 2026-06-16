"""Alembic migration environment.

The connection URL is taken from the DATABASE_URL environment variable so no
credentials are hardcoded. It is normalized to the psycopg v3 driver
(`postgresql+psycopg://`), which is what the project uses.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# We don't use declarative models; migrations are explicit.
target_metadata = None


def get_url() -> str:
    url = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/bottles")
    # Force the psycopg v3 driver.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(get_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
