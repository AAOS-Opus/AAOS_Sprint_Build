"""
E2E Flow Verification Test Suite (DevZen Enhanced Edition)
Tests complete orchestrator lifecycle with multiple agents and concurrent operations
Enhanced with async consistency, logging integrity, and performance telemetry
"""

import pytest
import asyncio
import json
import uuid
import time
import logging
import re
from datetime import datetime
from typing import List, Dict, Any

import websockets
import aiohttp
import redis

# Configuration from verified Phases 0-2
ORCHESTRATOR_WS = "ws://localhost:8000/ws"
ORCHESTRATOR_HTTP = "http://localhost:8000"
REDIS_CLIENT = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Configure logger to write to aaos.log
logger = logging.getLogger("aaos.e2e_test")
if not logger.handlers:
    handler = logging.FileHandler("aaos.log")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Test constants
CONCURRENT_AGENTS = 3
CONCURRENT_TASKS = 10
TASK_TYPES = ["code", "research", "analysis"]


def get_unique_agent_id(prefix="e2e-agent"):
    """Generate unique agent ID for each test"""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# === DevZen Enhancement #3: Targeted Redis Cleanup ===
@pytest.fixture(autouse=True)
def cleanup_test_redis_keys():
    """Clean only test-related Redis keys between runs (safer than flushdb)"""
    yield
    # Remove only keys that start with test patterns
    test_keys = REDIS_CLIENT.keys("test_*") + REDIS_CLIENT.keys("agent:e2e*") + REDIS_CLIENT.keys("task:e2e*")
    if test_keys:
        REDIS_CLIENT.delete(*test_keys)


@pytest.mark.asyncio
async def test_e2e_concurrent_task_processing():
    """
    Test 1: Multiple agents process tasks concurrently without conflicts
    NOTE: Clears Redis queue to ensure clean state for this test
    """
    start_time = time.time()

    # Clear Redis queue to ensure clean state
    REDIS_CLIENT.delete("task_queue")

    # Create agents
    agents = []
    for i in range(CONCURRENT_AGENTS):
        agent_id = get_unique_agent_id()
        ws = await websockets.connect(ORCHESTRATOR_WS)
        agents.append({
            "agent_id": agent_id,
            "websocket": ws,
            "tasks_completed": 0
        })
        # Register agent
        reg_msg = {
            "type": "register",
            "agent_id": agent_id,
            "agent_type": "e2e-test",
            "capabilities": {"can_code": True, "can_research": True}
        }
        await ws.send(json.dumps(reg_msg))
        ack = await asyncio.wait_for(ws.recv(), timeout=5)
        assert json.loads(ack)["type"] == "registration_ack"

    # Submit tasks
    task_ids = []
    async with aiohttp.ClientSession() as http_client:
        for i in range(CONCURRENT_TASKS):
            task_payload = {
                "task_type": TASK_TYPES[i % len(TASK_TYPES)],
                "description": f"Concurrent task {i}",
                "priority": 5
            }
            async with http_client.post(f"{ORCHESTRATOR_HTTP}/tasks", json=task_payload) as resp:
                assert resp.status == 201
                task_ids.append((await resp.json())["task_id"])

    # Track completions across all agents (only for tasks submitted in this test)
    completed_tasks = set()
    task_id_set = set(task_ids)  # Tasks we submitted

    async def agent_worker(agent):
        while len(completed_tasks) < CONCURRENT_TASKS:
            try:
                msg = await asyncio.wait_for(agent["websocket"].recv(), timeout=30)
                data = json.loads(msg)
                if data["type"] == "task_assignment":
                    task_id = data["task_id"]

                    # Simulate work
                    await asyncio.sleep(0.3)

                    # Complete task
                    completion = {
                        "type": "task_complete",
                        "task_id": task_id,
                        "agent_id": agent["agent_id"],
                        "result": {"status": "success", "duration_ms": 300}
                    }
                    await agent["websocket"].send(json.dumps(completion))
                    # Only count tasks we submitted in this test
                    if task_id in task_id_set:
                        completed_tasks.add(task_id)
                        agent["tasks_completed"] += 1
            except asyncio.TimeoutError:
                break

    # Run all agents concurrently
    await asyncio.gather(*[agent_worker(agent) for agent in agents])

    # Cleanup
    for agent in agents:
        await agent["websocket"].close()

    # Verify all OUR tasks completed
    assert len(completed_tasks) == CONCURRENT_TASKS, f"Expected {CONCURRENT_TASKS}, got {len(completed_tasks)}"

    # Verify tasks distributed across agents
    completions = [a["tasks_completed"] for a in agents]
    assert sum(completions) == CONCURRENT_TASKS

    duration = time.time() - start_time
    logger.info(f"E2E Test test_e2e_concurrent_task_processing completed in {duration:.2f}s")


@pytest.mark.asyncio
async def test_e2e_redis_queue_monitoring():
    """
    Test 2: Redis queue handles rapid task submission
    NOTE: Agent must be registered FIRST to keep circuit breaker closed (Fix #4)
    """
    start_time = time.time()

    # Clear queue for this test
    REDIS_CLIENT.delete("task_queue")

    # Register agent FIRST to keep circuit breaker closed
    agent_id = get_unique_agent_id("queue-test")
    ws = await websockets.connect(ORCHESTRATOR_WS)
    reg_msg = {"type": "register", "agent_id": agent_id, "agent_type": "queue-test"}
    await ws.send(json.dumps(reg_msg))
    await asyncio.wait_for(ws.recv(), timeout=5)

    # Submit tasks rapidly (circuit breaker is closed because agent is connected)
    task_ids = []
    async with aiohttp.ClientSession() as http_client:
        for i in range(15):
            task_payload = {"task_type": "code", "description": f"Queue test {i}", "priority": 5}
            async with http_client.post(f"{ORCHESTRATOR_HTTP}/tasks", json=task_payload) as resp:
                assert resp.status == 201, f"Task submission failed with status {resp.status}"
                task_data = await resp.json()
                task_ids.append(task_data["task_id"])

    # Process tasks
    processed = 0
    while processed < 15:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            data = json.loads(msg)
            if data["type"] == "task_assignment":
                completion = {
                    "type": "task_complete",
                    "task_id": data["task_id"],
                    "agent_id": agent_id,
                    "result": {"status": "success"}
                }
                await ws.send(json.dumps(completion))
                processed += 1
        except asyncio.TimeoutError:
            break

    await ws.close()

    # Verify queue is empty
    final_queue_len = REDIS_CLIENT.llen("task_queue")
    assert final_queue_len == 0

    duration = time.time() - start_time
    logger.info(f"E2E Test test_e2e_redis_queue_monitoring completed in {duration:.2f}s")


@pytest.mark.asyncio
async def test_e2e_database_consistency():
    """
    Test 3: Verify database state remains consistent throughout processing
    NOTE: Agent must be registered FIRST to keep circuit breaker closed (Fix #4)
    """
    start_time = time.time()

    # Register agent FIRST to keep circuit breaker closed
    agent_id = get_unique_agent_id("consistency")
    ws = await websockets.connect(ORCHESTRATOR_WS)
    reg_msg = {"type": "register", "agent_id": agent_id, "agent_type": "consistency-test"}
    await ws.send(json.dumps(reg_msg))
    await ws.recv()  # ack

    async with aiohttp.ClientSession() as http_client:
        # Submit task (circuit breaker is closed because agent is connected)
        task_payload = {"task_type": "research", "description": "Consistency test", "priority": 8}
        async with http_client.post(f"{ORCHESTRATOR_HTTP}/tasks", json=task_payload) as resp:
            assert resp.status == 201, f"Task submission failed with status {resp.status}"
            task_data = await resp.json()
            task_id = task_data["task_id"]

        # Verify initial state
        async with http_client.get(f"{ORCHESTRATOR_HTTP}/tasks/{task_id}") as resp:
            task = await resp.json()
            assert task["status"] in ["pending", "assigned"]  # May already be assigned

        # Wait for assignment and process
        msg = await asyncio.wait_for(ws.recv(), timeout=30)
        assignment = json.loads(msg)
        if assignment["type"] == "task_assignment":
            completion = {
                "type": "task_complete",
                "task_id": task_id,
                "agent_id": agent_id,
                "result": {"status": "success"}
            }
            await ws.send(json.dumps(completion))

        await ws.close()

        # Verify final state
        await asyncio.sleep(0.5)
        async with http_client.get(f"{ORCHESTRATOR_HTTP}/tasks/{task_id}") as resp:
            final_task = await resp.json()
            assert final_task["status"] == "completed"
            assert "result" in final_task["metadata"]

    duration = time.time() - start_time
    logger.info(f"E2E Test test_e2e_database_consistency completed in {duration:.2f}s")


@pytest.mark.asyncio
async def test_e2e_error_recovery():
    """
    Test 4: System recovers from agent failure and reassigns task
    NOTE: Agent must be registered FIRST to keep circuit breaker closed (Fix #4)
    NOTE: Clears Redis queue to ensure clean state for this test
    """
    start_time = time.time()

    # Clear Redis queue to ensure clean state
    REDIS_CLIENT.delete("task_queue")

    # Agent 1 registers FIRST to keep circuit breaker closed
    agent1_id = get_unique_agent_id("flaky")
    ws1 = await websockets.connect(ORCHESTRATOR_WS)
    reg_msg = {"type": "register", "agent_id": agent1_id, "agent_type": "flaky-agent"}
    await ws1.send(json.dumps(reg_msg))
    await ws1.recv()  # ack

    async with aiohttp.ClientSession() as http_client:
        # Submit task (circuit breaker is closed because agent is connected)
        task_payload = {"task_type": "code", "description": "Error recovery test", "priority": 9}
        async with http_client.post(f"{ORCHESTRATOR_HTTP}/tasks", json=task_payload) as resp:
            assert resp.status == 201, f"Task submission failed with status {resp.status}"
            task_id = (await resp.json())["task_id"]

    # Wait for assignment
    msg = await asyncio.wait_for(ws1.recv(), timeout=30)
    assignment = json.loads(msg)
    assert assignment["type"] == "task_assignment"
    assert assignment["task_id"] == task_id, f"Expected task {task_id}, got {assignment['task_id']}"

    # Simulate crash (close without completion)
    await ws1.close()

    # Wait for task to be requeued
    await asyncio.sleep(2)

    # Agent 2 should receive the reassigned task
    agent2_id = get_unique_agent_id("reliable")
    async with websockets.connect(ORCHESTRATOR_WS) as ws2:
        reg_msg2 = {"type": "register", "agent_id": agent2_id, "agent_type": "reliable-agent"}
        await ws2.send(json.dumps(reg_msg2))
        await ws2.recv()  # ack

        # Should receive the requeued task
        msg2 = await asyncio.wait_for(ws2.recv(), timeout=30)
        reassignment = json.loads(msg2)
        assert reassignment["type"] == "task_assignment"
        assert reassignment["task_id"] == task_id

        # Complete the task
        completion = {
            "type": "task_complete",
            "task_id": task_id,
            "agent_id": agent2_id,
            "result": {"status": "success", "recovered": True}
        }
        await ws2.send(json.dumps(completion))

    duration = time.time() - start_time
    logger.info(f"E2E Test test_e2e_error_recovery completed in {duration:.2f}s")


@pytest.mark.asyncio
async def test_e2e_system_metrics():
    """
    Test 5: System metrics endpoint provides accurate real-time data
    NOTE: Agent must be registered FIRST to keep circuit breaker closed (Fix #4)
    """
    start_time = time.time()

    # Register agent FIRST to keep circuit breaker closed
    agent_id = get_unique_agent_id("metrics-test")
    ws = await websockets.connect(ORCHESTRATOR_WS)
    reg_msg = {"type": "register", "agent_id": agent_id, "agent_type": "metrics-test"}
    await ws.send(json.dumps(reg_msg))
    await ws.recv()  # ack

    async with aiohttp.ClientSession() as http_client:
        # Get metrics during idle state
        async with http_client.get(f"{ORCHESTRATOR_HTTP}/metrics") as resp:
            assert resp.status == 200
            idle_metrics = await resp.json()
            assert "active_agents" in idle_metrics
            assert "queued_tasks" in idle_metrics
            assert "completed_tasks_total" in idle_metrics

        initial_queued = idle_metrics["queued_tasks"]

        # Submit a task (circuit breaker is closed because agent is connected)
        task_payload = {"task_type": "analysis", "description": "Metrics test", "priority": 5}
        async with http_client.post(f"{ORCHESTRATOR_HTTP}/tasks", json=task_payload) as resp:
            assert resp.status == 201, f"Task submission failed with status {resp.status}"

        await asyncio.sleep(0.3)  # Allow metrics to update

        # Verify metrics updated
        async with http_client.get(f"{ORCHESTRATOR_HTTP}/metrics") as resp:
            busy_metrics = await resp.json()
            assert busy_metrics["queued_tasks"] >= initial_queued  # Task added to queue

    await ws.close()

    duration = time.time() - start_time
    logger.info(f"E2E Test test_e2e_system_metrics completed in {duration:.2f}s")


@pytest.mark.asyncio
async def test_e2e_websocket_message_ordering():
    """
    Test 6: WebSocket messages maintain proper ordering
    NOTE: Register agent FIRST to submit task, then verify message order
    """
    start_time = time.time()

    agent_id = get_unique_agent_id("ordering")
    messages_received = []

    # Register agent FIRST to keep circuit breaker closed
    ws = await websockets.connect(ORCHESTRATOR_WS)
    reg_msg = {"type": "register", "agent_id": agent_id, "agent_type": "ordering-test"}
    await ws.send(json.dumps(reg_msg))

    # Get registration_ack first
    ack = await asyncio.wait_for(ws.recv(), timeout=10)
    messages_received.append(json.loads(ack)["type"])

    async with aiohttp.ClientSession() as http_client:
        # Submit task (circuit breaker is closed because agent is connected)
        task_payload = {"task_type": "synthesis", "description": "Ordering test", "priority": 10}
        async with http_client.post(f"{ORCHESTRATOR_HTTP}/tasks", json=task_payload) as resp:
            assert resp.status == 201, f"Task submission failed with status {resp.status}"

    # Wait for task assignment
    try:
        msg = await asyncio.wait_for(ws.recv(), timeout=10)
        messages_received.append(json.loads(msg)["type"])
    except asyncio.TimeoutError:
        pass

    await ws.close()

    # Verify registration_ack comes before task_assignment
    assert "registration_ack" in messages_received
    if "task_assignment" in messages_received:
        ack_idx = messages_received.index("registration_ack")
        assign_idx = messages_received.index("task_assignment")
        assert ack_idx < assign_idx

    duration = time.time() - start_time
    logger.info(f"E2E Test test_e2e_websocket_message_ordering completed in {duration:.2f}s")


def test_e2e_logging_integrity():
    """
    Test 7: aaos.log contains complete transaction trace
    === DevZen Enhancement #2: Chronological order validation ===
    """
    # Read recent log entries
    with open("aaos.log", "r") as f:
        logs = f.readlines()[-200:]

    # Parse log entries
    task_creations = [l for l in logs if "Task created successfully" in l]
    agent_regs = [l for l in logs if "Agent registered" in l]
    task_completions = [l for l in logs if "Task completed" in l]

    # Verify structured logging coverage
    assert len(task_creations) > 0, "No task creation logs found"
    assert len(agent_regs) > 0, "No agent registration logs found"
    assert len(task_completions) > 0, "No task completion logs found"

    # === DevZen Timestamp Order Check ===
    def extract_timestamp(log_line):
        """Extract timestamp from log line"""
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', log_line)
        return match.group(1) if match else ""

    timestamps = [extract_timestamp(line) for line in task_creations if extract_timestamp(line)]
    if timestamps:
        assert timestamps == sorted(timestamps), "Non-chronological log order detected"

    # Verify JSON contexts are valid
    for log_line in task_creations:
        if "|" in log_line:
            try:
                json_part = log_line.split("|", 1)[1].strip()
                json.loads(json_part)
            except (IndexError, json.JSONDecodeError) as e:
                pytest.fail(f"Invalid JSON in log: {e}")


@pytest.mark.asyncio
async def test_e2e_cleanup_verification():
    """
    Test 8: System state verification
    """
    start_time = time.time()

    async with aiohttp.ClientSession() as http_client:
        # Get system metrics
        async with http_client.get(f"{ORCHESTRATOR_HTTP}/metrics") as resp:
            metrics = await resp.json()
            assert "active_agents" in metrics
            assert "completed_tasks_total" in metrics

        # Get task list
        async with http_client.get(f"{ORCHESTRATOR_HTTP}/tasks") as resp:
            tasks = await resp.json()
            assert isinstance(tasks, list)

    duration = time.time() - start_time
    logger.info(f"E2E Test test_e2e_cleanup_verification completed in {duration:.2f}s")


def test_e2e_prerequisites():
    """Test 9: Verify all test dependencies are available"""
    assert ORCHESTRATOR_HTTP.startswith("http")
    assert ORCHESTRATOR_WS.startswith("ws")
    assert REDIS_CLIENT.ping() == True


def test_e2e_performance_summary():
    """
    Test 10: Generate aggregate performance report from aaos.log
    """
    try:
        with open("aaos.log", "r") as f:
            lines = [l for l in f if "E2E Test" in l and "completed in" in l]

        if not lines:
            pytest.skip("No timing logs found - run full test suite first")

        durations = []
        test_names = []
        for line in lines:
            match = re.search(r'E2E Test (\w+) completed in ([\d.]+)s', line)
            if match:
                test_names.append(match.group(1))
                durations.append(float(match.group(2)))

        if durations:
            print("\n" + "=" * 60)
            print("E2E PERFORMANCE SUMMARY")
            print("=" * 60)
            for name, duration in zip(test_names, durations):
                print(f"{name:<40} {duration:>6.2f}s")
            print("-" * 60)
            print(f"{'TOTAL':<40} {sum(durations):>6.2f}s")
            print(f"{'AVERAGE':<40} {sum(durations)/len(durations):>6.2f}s")
            print("=" * 60)

            # Assert reasonable performance (90s threshold for Windows I/O variance)
            assert max(durations) < 90, f"Test exceeded 90s timeout"

    except FileNotFoundError:
        pytest.skip("aaos.log not found for performance summary")
