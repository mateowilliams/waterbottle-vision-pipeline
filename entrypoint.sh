#!/bin/sh
# Apply Alembic migrations (with retries in case the DB is slow to come up)
# and then start the API with uvicorn.
set -e

echo "==> Applying migrations (alembic upgrade head)..."
i=1
until alembic upgrade head; do
  if [ "$i" -ge 10 ]; then
    echo "ERROR: the database was not reachable after 10 attempts." >&2
    exit 1
  fi
  echo "DB not ready yet, retry $i/10..."
  i=$((i + 1))
  sleep 2
done

echo "==> Starting API on :8000"
exec uvicorn src.app.main:app --host 0.0.0.0 --port 8000
