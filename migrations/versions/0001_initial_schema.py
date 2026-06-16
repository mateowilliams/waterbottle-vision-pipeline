"""Initial schema: images, predictions, labels.

Equivalent to sql/schema.sql. From here on, Alembic is the source of truth
for the schema (schema.sql remains a reference snapshot).

Revision ID: 0001
Revises:
Create Date: 2026-01-15
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("filepath", sa.Text, nullable=False),
        sa.Column("source", sa.Text, server_default="web_upload"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "image_id",
            sa.BigInteger,
            sa.ForeignKey("images.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_version", sa.Text, nullable=False),
        sa.Column("pred_label", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("probs_json", postgresql.JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "labels",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "image_id",
            sa.BigInteger,
            sa.ForeignKey("images.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("true_label", sa.Text, nullable=False),
        sa.Column(
            "labeled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index("idx_predictions_confidence", "predictions", ["confidence"])
    op.create_index("idx_predictions_created_at", "predictions", ["created_at"])
    op.create_index("idx_images_created_at", "images", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_images_created_at", table_name="images")
    op.drop_index("idx_predictions_created_at", table_name="predictions")
    op.drop_index("idx_predictions_confidence", table_name="predictions")
    op.drop_table("labels")
    op.drop_table("predictions")
    op.drop_table("images")
