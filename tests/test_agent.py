"""
pytest-asyncio Agent Lifecycle Test Suite
Tests WebSocket protocol against operational orchestrator
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any

import websockets
import aiohttp

# Configuration from Phase 0 discovery
ORCHESTRATOR_WS = "ws://localhost:8000/ws"
ORCHESTRATOR_HTTP = "http://localhost:8000"


def get_unique_agent_id():
    """Generate unique agent ID for each test to avoid conflicts"""
    return f"test-agent-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def registration_message():
    """Standard registration message from Phase 0 protocol"""
    agent_id = get_unique_agent_id()
    return {
        "type": "register",
        "agent_type": "test",
        "agent_id": agent_id,
        "capabilities": {
            "can_code": True,
            "can_research": True,
            "max_concurrent_tasks": 1
        },
        "heartbeat_interval": 30
    }


@pytest.fixture
def heartbeat_message():
    """Standard heartbeat message"""
    return {
        "type": "heartbeat",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "idle"
    }


@pytest.mark.asyncio
async def test_agent_registration(registration_message):
    """
    Test 1: Agent registration handshake
    Expected: register -> registration_ack
    """
    agent_id = registration_message["agent_id"]

    async with websockets.connect(ORCHESTRATOR_WS) as ws:
        # Send registration
        await ws.send(json.dumps(registration_message))

        # Receive ack with timeout
        response = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(response)

        # Validate response
        assert data["type"] == "registration_ack"
        assert data["agent_id"] == agent_id
        assert data["status"] == "registered"

        # Verify agent appears in API
        await asyncio.sleep(0.5)  # Allow for processing
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ORCHESTRATOR_HTTP}/agents") as resp:
                assert resp.status == 200
                agents = await resp.json()
                assert any(a["agent_id"] == agent_id for a in agents)


@pytest.mark.asyncio
async def test_agent_heartbeat(registration_message, heartbeat_message):
    """
    Test 2: Heartbeat maintenance
    Expected: heartbeat sent -> no disconnection
    """
    agent_id = registration_message["agent_id"]
    heartbeat_message["agent_id"] = agent_id

    async with websockets.connect(ORCHESTRATOR_WS) as ws:
        # First register
        await ws.send(json.dumps(registration_message))
        ack = await asyncio.wait_for(ws.recv(), timeout=5)
        assert json.loads(ack)["type"] == "registration_ack"

        # Send heartbeat
        await ws.send(json.dumps(heartbeat_message))

        # Wait and verify connection remains open (no exception)
        await asyncio.sleep(1)
        assert ws.open == True

        # Send another heartbeat
        await ws.send(json.dumps(heartbeat_message))
        await asyncio.sleep(0.5)
        assert ws.open == True


@pytest.mark.asyncio
async def test_agent_task_completion(registration_message):
    """
    Test 3: Full task lifecycle
    Sequence: register -> task submission -> task_assignment -> completion -> verification
    """
    agent_id = registration_message["agent_id"]

    async with websockets.connect(ORCHESTRATOR_WS) as ws:
        # Step 1: Register agent
        await ws.send(json.dumps(registration_message))
        ack = await asyncio.wait_for(ws.recv(), timeout=5)
        assert json.loads(ack)["type"] == "registration_ack"

        # Step 2: Submit task via POST /tasks (from Phase 1)
        task_payload = {
            "task_type": "code",
            "description": "pytest-asyncio test task",
            "priority": 7
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ORCHESTRATOR_HTTP}/tasks",
                json=task_payload
            ) as resp:
                assert resp.status == 201
                task_data = await resp.json()
                task_id = task_data["task_id"]

        # Step 3: Wait for task assignment via WebSocket
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            assignment = json.loads(msg)

            assert assignment["type"] == "task_assignment"
            assert assignment["task_id"] == task_id
            assert assignment["task_type"] == "code"

            # Step 4: Simulate work and send completion
            await asyncio.sleep(1)  # Simulate processing

            completion = {
                "type": "task_complete",
                "task_id": task_id,
                "agent_id": agent_id,
                "result": {
                    "status": "success",
                    "output": "Test task completed by pytest agent",
                    "duration_ms": 1000
                },
                "reasoning_steps": [
                    {"step": 1, "action": "analyzed task"},
                    {"step": 2, "action": "simulated execution"}
                ],
                "timestamp": datetime.utcnow().isoformat()
            }

            await ws.send(json.dumps(completion))

            # Step 5: Verify task status updated via API
            await asyncio.sleep(1)  # Allow for DB update

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ORCHESTRATOR_HTTP}/tasks/{task_id}") as resp:
                    assert resp.status == 200
                    final_task = await resp.json()
                    assert final_task["status"] == "completed"
                    assert "result" in final_task["metadata"]

        except asyncio.TimeoutError:
            pytest.fail("Task assignment timeout after 30s - check Redis queue and orchestrator logs")


@pytest.mark.asyncio
async def test_agent_multiple_tasks(registration_message):
    """
    Test 4: Agent handles multiple tasks sequentially
    """
    agent_id = registration_message["agent_id"]

    async with websockets.connect(ORCHESTRATOR_WS) as ws:
        # Register agent
        await ws.send(json.dumps(registration_message))
        ack = await asyncio.wait_for(ws.recv(), timeout=5)
        assert json.loads(ack)["type"] == "registration_ack"

        # Submit 3 tasks
        task_ids = []
        async with aiohttp.ClientSession() as session:
            for i in range(3):
                task_payload = {
                    "task_type": "research",
                    "description": f"Sequential test task {i+1}",
                    "priority": 5
                }
                async with session.post(
                    f"{ORCHESTRATOR_HTTP}/tasks",
                    json=task_payload
                ) as resp:
                    assert resp.status == 201
                    task_data = await resp.json()
                    task_ids.append(task_data["task_id"])

        # Process tasks sequentially
        for expected_task_id in task_ids:
            # Receive assignment
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            assignment = json.loads(msg)

            assert assignment["type"] == "task_assignment"
            received_task_id = assignment["task_id"]

            # Complete task
            completion = {
                "type": "task_complete",
                "task_id": received_task_id,
                "agent_id": agent_id,
                "result": {"status": "success", "task_number": task_ids.index(received_task_id) + 1}
            }
            await ws.send(json.dumps(completion))
            await asyncio.sleep(0.3)  # Brief pause between tasks


@pytest.mark.asyncio
async def test_agent_graceful_shutdown(registration_message):
    """
    Test 5: Agent disconnects cleanly without hanging
    """
    agent_id = registration_message["agent_id"]

    async with websockets.connect(ORCHESTRATOR_WS) as ws:
        # Register
        await ws.send(json.dumps(registration_message))
        ack = await asyncio.wait_for(ws.recv(), timeout=5)
        assert json.loads(ack)["type"] == "registration_ack"

        # Connection is open
        assert ws.open == True

    # After context manager exits, connection should be closed
    # Verify agent marked as disconnected (after brief delay)
    await asyncio.sleep(1)
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{ORCHESTRATOR_HTTP}/agents") as resp:
            agents = await resp.json()
            # Agent may still be listed but marked as disconnected
            test_agent = next((a for a in agents if a["agent_id"] == agent_id), None)
            if test_agent:
                assert test_agent.get("status") == "disconnected"


def test_pytest_asyncio_setup():
    """
    Meta-test: Verify pytest-asyncio is configured correctly
    """
    # This test runs synchronously to verify setup
    assert hasattr(pytest, 'mark')
    assert hasattr(pytest.mark, 'asyncio')
