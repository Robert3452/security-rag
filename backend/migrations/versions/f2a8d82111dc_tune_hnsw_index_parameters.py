"""tune hnsw index parameters

Revision ID: f2a8d82111dc
Revises: da8b74d20dcc
Create Date: 2026-10-25 17:23:29.943727

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = 'f2a8d82111dc'
down_revision: Union[str, Sequence[str], None] = 'da8b74d20dcc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_chunks_embeddings_hnsw", table_name="chunks")
    op.create_index(
        "ix_chunks_embeddings_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 24, "ef_construction": 128},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embeddings_hnsw", table_name="chunks")
    op.create_index(
        "ix_chunks_embeddings_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )