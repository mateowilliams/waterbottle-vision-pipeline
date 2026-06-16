-- NOTE: reference snapshot of the schema. The source of truth is the Alembic
-- migrations (migrations/). To create the schema, run:
--   alembic upgrade head
-- This file is handy for quick inspection or to bootstrap without Alembic.

CREATE TABLE IF NOT EXISTS images (
  id BIGSERIAL PRIMARY KEY,
  filepath TEXT NOT NULL,
  source TEXT DEFAULT 'web_upload',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  image_id BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  model_version TEXT NOT NULL,
  pred_label TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  probs_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS labels (
  id BIGSERIAL PRIMARY KEY,
  image_id BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  true_label TEXT NOT NULL,
  labeled_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at);
