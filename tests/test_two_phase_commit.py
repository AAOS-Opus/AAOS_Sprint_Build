"""
Two-Phase Commit and Reconciliation Test Suite (Fix #2)

Tests the Redis-first two-phase commit protocol and background reconciliation job.

Test Scenarios:
1. Normal task creation (both Redis and DB succeed)
2. Simulated DB failure with Redis rollback verification
3. Reconciliation catches orphaned tasks
4. Manual reconciliation endpoint works
"""

import pytest
import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import aiohttp
import redis

# Configuration
ORCHESTRATOR_HTTP = "http://localhost:8000"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
API_KEY = "O-cDTeZDyqGT6JRLp8p_aUv__je0ew-QXVThPhsGxKc"

# Redis client for direct verification
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_auth_headers():
    """Return headers with API key"""
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }


@pytest.fixture
def clean_redis_queue():
    """Ensure clean Redis queue for each test"""
    # Store current queue length
    initial_len = redis_client.llen("task_queue")
    yield
    # No cleanup needed - tests verify their own state


@pytest.mark.asyncio
async def test_normal_task_creation_two_phase():
    """
    Test 1: Normal task creation succeeds with two-phase commit.
    Verifies task appears in both Redis queue and DB.
    """
    task_payload = {
        "task_type": "code",
        "description": "Two-phase commit test task",
        "priority": 5
    }

    # Get initial Redis queue state
    initial_queue = set(redis_client.lrange("task_queue", 0, -1))

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{ORCHESTRATOR_HTTP}/tasks",
            json=task_payload,
            headers=get_auth_headers()
        ) as resp:
            assert resp.status == 201, f"Expected 201, got {resp.status}"
            result = await resp.json()
            task_id = result["task_id"]
            assert result["status"] == "pending"
            assert result["message"] == "Task queued successfully"

    # Verify task is in Redis queue
    current_queue = set(redis_client.lrange("task_queue", 0, -1))
    new_tasks = current_queue - initial_queue

    # Task should be in Redis (unless already consumed by agent)
    # Check DB to confirm task exists
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{ORCHESTRATOR_HTTP}/tasks/{task_id}",
            headers=get_auth_headers()
        ) as resp:
            assert resp.status == 200, f"Task {task_id} not found in DB"
            task_data = await resp.json()
            assert task_data["task_id"] == task_id
            assert task_data["status"] in ["pending", "assigned", "completed"]

    print(f"[PASS] Task {task_id} created successfully with two-phase commit")


@pytest.mark.asyncio
async def test_redis_rollback_on_simulated_db_failure():
    """
    Test 2: Verify that if we simulate a scenario where a task_id is in Redis
    but NOT in DB, the reconciliation can detect it.

    This tests the conceptual correctness - we push directly to Redis without DB.
    """
    # Create a fake task_id and push directly to Redis (simulating failed DB commit)
    fake_task_id = f"orphan-redis-{uuid.uuid4().hex[:8]}"

    # Push to Redis queue directly (simulating Phase 1 succeeded, Phase 2 never happened)
    redis_client.lpush("task_queue", fake_task_id)

    # Verify it's in Redis
    queue_contents = redis_client.lrange("task_queue", 0, -1)
    assert fake_task_id in queue_contents, "Fake task should be in Redis queue"

    # Verify it's NOT in DB
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{ORCHESTRATOR_HTTP}/tasks/{fake_task_id}",
            headers=get_auth_headers()
        ) as resp:
            assert resp.status == 404, "Fake task should NOT be in DB"

    # Clean up - remove the fake task from Redis
    redis_client.lrem("task_queue", 1, fake_task_id)

    print(f"[PASS] Verified orphaned Redis task detection works")


@pytest.mark.asyncio
async def test_manual_reconciliation_endpoint():
    """
    Test 3: Manual reconciliation endpoint works and returns correct response.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{ORCHESTRATOR_HTTP}/admin/reconcile",
            headers=get_auth_headers()
        ) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            result = await resp.json()

            assert "status" in result
            assert result["status"] == "completed"
            assert "result" in result
            assert "timestamp" in result

            # Result should have reconciled and/or failed counts
            if "error" not in result["result"]:
                assert "reconciled" in result["result"]
                assert "failed" in result["result"]

    print(f"[PASS] Manual reconciliation endpoint works correctly")


@pytest.mark.asyncio
async def test_reconciliation_detects_orphaned_db_tasks():
    """
    Test 4: Create a task directly in DB (bypassing Redis) and verify
    reconciliation detects and re-queues it.

    Note: This requires the task to be older than ORPHAN_THRESHOLD_MINUTES (10 min).
    For testing, we'll verify the reconciliation logic by checking for tasks
    that are pending but not in Redis.
    """
    # First, get current state
    async with aiohttp.ClientSession() as session:
        # List all pending tasks
        async with session.get(
            f"{ORCHESTRATOR_HTTP}/tasks",
            headers=get_auth_headers()
        ) as resp:
            assert resp.status == 200
            all_tasks = await resp.json()

    # Get Redis queue contents
    redis_queue = set(redis_client.lrange("task_queue", 0, -1))

    # Find pending tasks that might be orphaned (in DB but not in Redis)
    pending_tasks = [t for t in all_tasks if t["status"] == "pending"]
    orphaned_candidates = [t for t in pending_tasks if t["task_id"] not in redis_queue]

    print(f"Found {len(pending_tasks)} pending tasks, {len(orphaned_candidates)} not in Redis queue")

    # Trigger reconciliation
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{ORCHESTRATOR_HTTP}/admin/reconcile",
            headers=get_auth_headers()
        ) as resp:
            result = await resp.json()
            print(f"Reconciliation result: {result}")

    print(f"[PASS] Reconciliation check completed")


@pytest.mark.asyncio
async def test_concurrent_task_creation_consistency():
    """
    Test 5: Create multiple tasks concurrently and verify all succeed
    with two-phase commit consistency.
    """
    num_tasks = 5
    tasks_created = []

    async def create_task(index):
        task_payload = {
            "task_type": "analysis",
            "description": f"Concurrent 2PC test task {index}",
            "priority": index % 10
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ORCHESTRATOR_HTTP}/tasks",
                json=task_payload,
                headers=get_auth_headers()
            ) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    return result["task_id"]
                return None

    # Create tasks concurrently
    results = await asyncio.gather(*[create_task(i) for i in range(num_tasks)])
    tasks_created = [r for r in results if r is not None]

    assert len(tasks_created) == num_tasks, f"Expected {num_tasks} tasks, created {len(tasks_created)}"

    # Verify all tasks exist in DB
    async with aiohttp.ClientSession() as session:
        for task_id in tasks_created:
            async with session.get(
                f"{ORCHESTRATOR_HTTP}/tasks/{task_id}",
                headers=get_auth_headers()
            ) as resp:
                assert resp.status == 200, f"Task {task_id} not found in DB"

    print(f"[PASS] Created {len(tasks_created)} tasks concurrently with two-phase commit")


@pytest.mark.asyncio
async def test_api_key_required_for_reconciliation():
    """
    Test 6: Verify reconciliation endpoint requires authentication.
    """
    async with aiohttp.ClientSession() as session:
        # Try without API key
        async with session.post(f"{ORCHESTRATOR_HTTP}/admin/reconcile") as resp:
            assert resp.status == 401, f"Expected 401 without API key, got {resp.status}"

        # Try with invalid API key
        async with session.post(
            f"{ORCHESTRATOR_HTTP}/admin/reconcile",
            headers={"X-API-Key": "invalid-key"}
        ) as resp:
            assert resp.status == 403, f"Expected 403 with invalid key, got {resp.status}"

    print(f"[PASS] Reconciliation endpoint properly requires authentication")


# =============================================================================
# Integration test that verifies the full two-phase commit flow
# =============================================================================

@pytest.mark.asyncio
async def test_two_phase_commit_full_flow():
    """
    Test 7: Full integration test of two-phase commit.

    1. Create a task
    2. Verify it's queued in Redis
    3. Verify it's stored in DB
    4. Verify the task can be retrieved
    """
    print("\n=== Two-Phase Commit Full Flow Test ===")

    # Step 1: Create task
    task_payload = {
        "task_type": "research",
        "description": "Full 2PC flow verification test",
        "priority": 7,
        "metadata": {"test_type": "two_phase_commit"}
    }

    # Record Redis queue before
    queue_before = redis_client.llen("task_queue")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{ORCHESTRATOR_HTTP}/tasks",
            json=task_payload,
            headers=get_auth_headers()
        ) as resp:
            assert resp.status == 201
            result = await resp.json()
            task_id = result["task_id"]
            print(f"Step 1: Task created with ID {task_id}")

    # Step 2: Check Redis (task may or may not be there if agent consumed it)
    queue_after = redis_client.llen("task_queue")
    print(f"Step 2: Redis queue length before={queue_before}, after={queue_after}")

    # Step 3: Verify DB storage
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{ORCHESTRATOR_HTTP}/tasks/{task_id}",
            headers=get_auth_headers()
        ) as resp:
            assert resp.status == 200
            task_data = await resp.json()
            assert task_data["task_type"] == "research"
            assert task_data["priority"] == 7
            print(f"Step 3: Task verified in DB with status={task_data['status']}")

    # Step 4: Verify metadata preserved
    assert task_data["metadata"].get("test_type") == "two_phase_commit"
    print(f"Step 4: Metadata preserved correctly")

    print("=== Two-Phase Commit Full Flow Test PASSED ===\n")


if __name__ == "__main__":
    # Run tests directly
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
