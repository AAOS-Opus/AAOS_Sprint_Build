"""add_task_timeout_retry_fields

Revision ID: 93bcc1a50a0e
Revises: 8245fc50ff27
Create Date: 2025-12-11 18:20:38.718528

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '93bcc1a50a0e'
down_revision: Union[str, None] = '8245fc50ff27'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add task timeout and retry fields for Fix #3"""
    with op.batch_alter_table('tasks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('assigned_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('retry_count', sa.Integer(), nullable=True, server_default='0'))
        batch_op.add_column(sa.Column('max_retries', sa.Integer(), nullable=True, server_default='3'))
        batch_op.add_column(sa.Column('retry_after', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Remove task timeout and retry fields"""
    with op.batch_alter_table('tasks', schema=None) as batch_op:
        batch_op.drop_column('retry_after')
        batch_op.drop_column('max_retries')
        batch_op.drop_column('retry_count')
        batch_op.drop_column('assigned_at')
