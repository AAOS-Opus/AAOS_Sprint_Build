# tests/test_task_timeout_retry.py
"""
Test suite for Fix #3: Task Timeout + Retry with Idempotency

Tests:
1. Stuck task detection and retry
2. Max retries results in failed status
3. Exponential backoff delays retries appropriately
4. Idempotency check via Redis processing_tasks set
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import redis

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import SessionLocal, Task
from src.orchestrator.core import (
    run_timeout_detection,
    calculate_backoff_seconds,
    TASK_TIMEOUT_SECONDS,
    TASK_MAX_RETRIES,
    REDIS_HOST,
    REDIS_PORT
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def db_session():
    """Create a fresh database session for testing"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def redis_client():
    """Create a Redis client for testing"""
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    yield client
    # Cleanup: remove test keys
    client.delete("task_queue")
    client.delete("processing_tasks")


@pytest.fixture
def stuck_task(db_session) -> str:
    """Create a task that appears stuck (assigned but timed out)"""
    task_id = str(uuid.uuid4())

    # Create task that was assigned more than TASK_TIMEOUT_SECONDS ago
    task = Task(
        task_id=task_id,
        task_type="code",
        description="Test stuck task",
        priority=5,
        status="assigned",
        metadata_json="{}",
        created_at=datetime.utcnow() - timedelta(minutes=10),
        updated_at=datetime.utcnow() - timedelta(minutes=10),
        assigned_at=datetime.utcnow() - timedelta(seconds=TASK_TIMEOUT_SECONDS + 60),  # Timed out
        retry_count=0,
        max_retries=TASK_MAX_RETRIES
    )

    db_session.add(task)
    db_session.commit()

    yield task_id

    # Cleanup
    db_session.query(Task).filter(Task.task_id == task_id).delete()
    db_session.commit()


@pytest.fixture
def max_retry_task(db_session) -> str:
    """Create a task that has exceeded max retries"""
    task_id = str(uuid.uuid4())

    task = Task(
        task_id=task_id,
        task_type="research",
        description="Test max retry task",
        priority=5,
        status="assigned",
        metadata_json="{}",
        created_at=datetime.utcnow() - timedelta(hours=1),
        updated_at=datetime.utcnow() - timedelta(minutes=10),
        assigned_at=datetime.utcnow() - timedelta(seconds=TASK_TIMEOUT_SECONDS + 60),
        retry_count=TASK_MAX_RETRIES,  # Already at max
        max_retries=TASK_MAX_RETRIES
    )

    db_session.add(task)
    db_session.commit()

    yield task_id

    # Cleanup
    db_session.query(Task).filter(Task.task_id == task_id).delete()
    db_session.commit()


@pytest.fixture
def idempotent_task(db_session, redis_client) -> str:
    """Create a task that is in the processing_tasks set (should be skipped)"""
    task_id = str(uuid.uuid4())

    task = Task(
        task_id=task_id,
        task_type="qa",
        description="Test idempotent task",
        priority=5,
        status="assigned",
        metadata_json="{}",
        created_at=datetime.utcnow() - timedelta(minutes=10),
        updated_at=datetime.utcnow() - timedelta(minutes=10),
        assigned_at=datetime.utcnow() - timedelta(seconds=TASK_TIMEOUT_SECONDS + 60),
        retry_count=0,
        max_retries=TASK_MAX_RETRIES
    )

    db_session.add(task)
    db_session.commit()

    # Add to processing_tasks set (simulates active processing)
    redis_client.sadd("processing_tasks", task_id)

    yield task_id

    # Cleanup
    redis_client.srem("processing_tasks", task_id)
    db_session.query(Task).filter(Task.task_id == task_id).delete()
    db_session.commit()


# =============================================================================
# Test: Exponential Backoff Calculation
# =============================================================================

class TestExponentialBackoff:
    """Tests for exponential backoff calculation"""

    def test_backoff_retry_1(self):
        """Retry 1 should have 2 second backoff"""
        assert calculate_backoff_seconds(1) == 2

    def test_backoff_retry_2(self):
        """Retry 2 should have 4 second backoff"""
        assert calculate_backoff_seconds(2) == 4

    def test_backoff_retry_3(self):
        """Retry 3 should have 8 second backoff"""
        assert calculate_backoff_seconds(3) == 8

    def test_backoff_retry_4(self):
        """Retry 4 should have 16 second backoff"""
        assert calculate_backoff_seconds(4) == 16

    def test_backoff_retry_5(self):
        """Retry 5 should have 32 second backoff"""
        assert calculate_backoff_seconds(5) == 32

    def test_backoff_retry_6(self):
        """Retry 6 should be capped at 60 seconds"""
        assert calculate_backoff_seconds(6) == 60

    def test_backoff_retry_10(self):
        """High retry count should be capped at 60 seconds"""
        assert calculate_backoff_seconds(10) == 60

    def test_backoff_formula(self):
        """Verify backoff follows min(60, 2^retry_count) formula"""
        for retry in range(1, 10):
            expected = min(60, 2 ** retry)
            assert calculate_backoff_seconds(retry) == expected


# =============================================================================
# Test: Stuck Task Detection and Retry
# =============================================================================

class TestStuckTaskRetry:
    """Tests for detecting and retrying stuck tasks"""

    @pytest.mark.asyncio
    async def test_stuck_task_detected_and_retried(self, db_session, redis_client, stuck_task):
        """Test that a stuck task gets detected and requeued"""
        # Run timeout detection
        result = await run_timeout_detection()

        # Verify task was retried
        assert result.get("retried", 0) >= 1

        # Verify task status changed to pending
        task = db_session.query(Task).filter(Task.task_id == stuck_task).first()
        db_session.refresh(task)

        assert task.status == "pending"
        assert task.retry_count == 1
        assert task.assigned_at is None  # Reset on retry
        assert task.retry_after is not None  # Backoff set

        # Verify task was requeued to Redis
        queue_contents = redis_client.lrange("task_queue", 0, -1)
        assert stuck_task in queue_contents

    @pytest.mark.asyncio
    async def test_retry_metadata_recorded(self, db_session, redis_client, stuck_task):
        """Test that retry history is recorded in metadata"""
        # Run timeout detection
        await run_timeout_detection()

        # Check metadata
        task = db_session.query(Task).filter(Task.task_id == stuck_task).first()
        db_session.refresh(task)

        metadata = json.loads(task.metadata_json)

        assert "retry_history" in metadata
        assert len(metadata["retry_history"]) == 1
        assert metadata["retry_history"][0]["retry_number"] == 1
        assert metadata["retry_history"][0]["reason"] == "timeout"
        assert "backoff_seconds" in metadata["retry_history"][0]
        assert "timestamp" in metadata["retry_history"][0]


# =============================================================================
# Test: Max Retries Results in Failed Status
# =============================================================================

class TestMaxRetriesFailed:
    """Tests for marking tasks as failed after max retries"""

    @pytest.mark.asyncio
    async def test_max_retries_marks_task_failed(self, db_session, max_retry_task):
        """Test that exceeding max retries marks task as failed"""
        # Run timeout detection
        result = await run_timeout_detection()

        # Verify task was marked as failed
        assert result.get("failed", 0) >= 1

        # Verify task status is failed
        task = db_session.query(Task).filter(Task.task_id == max_retry_task).first()
        db_session.refresh(task)

        assert task.status == "failed"

    @pytest.mark.asyncio
    async def test_failure_reason_recorded(self, db_session, max_retry_task):
        """Test that failure reason is recorded in metadata"""
        # Run timeout detection
        await run_timeout_detection()

        # Check metadata
        task = db_session.query(Task).filter(Task.task_id == max_retry_task).first()
        db_session.refresh(task)

        metadata = json.loads(task.metadata_json)

        assert metadata["failure_reason"] == "timeout_max_retries"
        assert "failed_at" in metadata
        assert "final_retry_count" in metadata


# =============================================================================
# Test: Idempotency Check
# =============================================================================

class TestIdempotencyCheck:
    """Tests for idempotency via Redis processing_tasks set"""

    @pytest.mark.asyncio
    async def test_task_in_processing_set_skipped(self, db_session, redis_client, idempotent_task):
        """Test that tasks in processing_tasks set are skipped"""
        # Verify task is in processing_tasks set
        assert redis_client.sismember("processing_tasks", idempotent_task)

        # Run timeout detection
        result = await run_timeout_detection()

        # Verify task was skipped (idempotent)
        assert result.get("skipped_idempotent", 0) >= 1

        # Verify task status is still assigned (not changed)
        task = db_session.query(Task).filter(Task.task_id == idempotent_task).first()
        db_session.refresh(task)

        assert task.status == "assigned"
        assert task.retry_count == 0  # Not incremented


# =============================================================================
# Test: Backoff Period Respected
# =============================================================================

class TestBackoffPeriod:
    """Tests for exponential backoff period being respected"""

    @pytest.mark.asyncio
    async def test_task_within_backoff_period_skipped(self, db_session):
        """Test that tasks within their backoff period are not retried"""
        task_id = str(uuid.uuid4())

        # Create task with retry_after in the future
        task = Task(
            task_id=task_id,
            task_type="analysis",
            description="Test backoff task",
            priority=5,
            status="assigned",
            metadata_json="{}",
            created_at=datetime.utcnow() - timedelta(minutes=10),
            updated_at=datetime.utcnow() - timedelta(minutes=5),
            assigned_at=datetime.utcnow() - timedelta(seconds=TASK_TIMEOUT_SECONDS + 60),
            retry_count=1,
            max_retries=TASK_MAX_RETRIES,
            retry_after=datetime.utcnow() + timedelta(minutes=5)  # Still in backoff
        )

        db_session.add(task)
        db_session.commit()

        try:
            # Run timeout detection
            result = await run_timeout_detection()

            # Task should not have been retried (still in backoff)
            task = db_session.query(Task).filter(Task.task_id == task_id).first()
            db_session.refresh(task)

            assert task.status == "assigned"  # Still assigned
            assert task.retry_count == 1  # Not incremented

        finally:
            # Cleanup
            db_session.query(Task).filter(Task.task_id == task_id).delete()
            db_session.commit()


# =============================================================================
# Test: Multiple Tasks Processing
# =============================================================================

class TestMultipleTasksProcessing:
    """Tests for processing multiple stuck tasks"""

    @pytest.mark.asyncio
    async def test_multiple_stuck_tasks_processed(self, db_session, redis_client):
        """Test that multiple stuck tasks are all processed correctly"""
        task_ids = []

        # Create 3 stuck tasks
        for i in range(3):
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                task_type="code",
                description=f"Test stuck task {i}",
                priority=5,
                status="assigned",
                metadata_json="{}",
                created_at=datetime.utcnow() - timedelta(minutes=10),
                updated_at=datetime.utcnow() - timedelta(minutes=10),
                assigned_at=datetime.utcnow() - timedelta(seconds=TASK_TIMEOUT_SECONDS + 60),
                retry_count=0,
                max_retries=TASK_MAX_RETRIES
            )
            db_session.add(task)
            task_ids.append(task_id)

        db_session.commit()

        try:
            # Run timeout detection
            result = await run_timeout_detection()

            # Verify all 3 tasks were retried
            assert result.get("retried", 0) >= 3

            # Verify all tasks are now pending
            for task_id in task_ids:
                task = db_session.query(Task).filter(Task.task_id == task_id).first()
                db_session.refresh(task)
                assert task.status == "pending"
                assert task.retry_count == 1

        finally:
            # Cleanup
            for task_id in task_ids:
                db_session.query(Task).filter(Task.task_id == task_id).delete()
            db_session.commit()


# =============================================================================
# Test: Task Model Fields
# =============================================================================

class TestTaskModelFields:
    """Tests for the new task model fields"""

    def test_task_has_timeout_fields(self, db_session):
        """Test that Task model has the new timeout-related fields"""
        task_id = str(uuid.uuid4())

        task = Task(
            task_id=task_id,
            task_type="documentation",
            description="Test field existence",
            priority=5,
            status="pending",
            metadata_json="{}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        db_session.add(task)
        db_session.commit()

        try:
            # Verify fields exist and have correct defaults
            task = db_session.query(Task).filter(Task.task_id == task_id).first()

            assert hasattr(task, 'assigned_at')
            assert hasattr(task, 'retry_count')
            assert hasattr(task, 'max_retries')
            assert hasattr(task, 'retry_after')

            # Check defaults
            assert task.assigned_at is None
            assert task.retry_count == 0 or task.retry_count is None
            assert task.max_retries == 3 or task.max_retries is None
            assert task.retry_after is None

        finally:
            # Cleanup
            db_session.query(Task).filter(Task.task_id == task_id).delete()
            db_session.commit()


# =============================================================================
# Integration Test: Full Retry Cycle
# =============================================================================

class TestFullRetryCycle:
    """Integration tests for the complete retry cycle"""

    @pytest.mark.asyncio
    async def test_task_retries_then_fails(self, db_session, redis_client):
        """Test complete cycle: task retries multiple times then fails"""
        task_id = str(uuid.uuid4())
        max_retries = 2  # Use smaller number for faster test

        task = Task(
            task_id=task_id,
            task_type="synthesis",
            description="Test full retry cycle",
            priority=5,
            status="assigned",
            metadata_json="{}",
            created_at=datetime.utcnow() - timedelta(hours=1),
            updated_at=datetime.utcnow() - timedelta(minutes=10),
            assigned_at=datetime.utcnow() - timedelta(seconds=TASK_TIMEOUT_SECONDS + 60),
            retry_count=0,
            max_retries=max_retries
        )

        db_session.add(task)
        db_session.commit()

        try:
            # Retry cycle
            for expected_retry in range(1, max_retries + 1):
                # Run timeout detection
                result = await run_timeout_detection()

                task = db_session.query(Task).filter(Task.task_id == task_id).first()
                db_session.refresh(task)

                assert task.status == "pending"
                assert task.retry_count == expected_retry

                # Simulate re-assignment (so it can time out again)
                task.status = "assigned"
                task.assigned_at = datetime.utcnow() - timedelta(seconds=TASK_TIMEOUT_SECONDS + 60)
                task.retry_after = None  # Clear backoff for test
                db_session.commit()

            # One more run should mark as failed
            result = await run_timeout_detection()

            task = db_session.query(Task).filter(Task.task_id == task_id).first()
            db_session.refresh(task)

            assert task.status == "failed"

            metadata = json.loads(task.metadata_json)
            assert metadata["failure_reason"] == "timeout_max_retries"

        finally:
            # Cleanup
            redis_client.lrem("task_queue", 0, task_id)
            db_session.query(Task).filter(Task.task_id == task_id).delete()
            db_session.commit()


# =============================================================================
# Main entry point for running tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
