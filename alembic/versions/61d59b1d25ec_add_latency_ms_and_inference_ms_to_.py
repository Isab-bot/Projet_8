"""add latency_ms and inference_ms to predictions

Revision ID: 61d59b1d25ec
Revises: 206600e28d40
Create Date: 2026-05-19 20:37:59.209518

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '61d59b1d25ec'
down_revision: Union[str, Sequence[str], None] = '206600e28d40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "predictions",
        sa.Column("latency_ms", sa.Float(), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("inference_ms", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("predictions", "inference_ms")
    op.drop_column("predictions", "latency_ms")
