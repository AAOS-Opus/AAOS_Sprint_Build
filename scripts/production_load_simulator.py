#!/usr/bin/env python3
"""
AAOS Production Load Simulator
DevZen Enhancement #5: Time-anchored load generation with correlation ID tracking

Simulates realistic production workload with configurable task rates and agent counts.
"""

import argparse
import asyncio
import json
import random
import signal
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp
import websockets


class LoadSimulator:
    """Simulates production load with multiple agents and configurable task rates"""

    TASK_TYPES = ["code", "research", "qa", "documentation", "analysis", "synthesis"]
    TASK_TEMPLATES = [
        "Analyze the {component} module for performance bottlenecks",
        "Implement {feature} feature with error handling",
        "Review and document the {component} API endpoints",
        "Research best practices for {topic} implementation",
        "Run QA tests on {component} integration",
        "Synthesize findings from {topic} analysis"
    ]

    def __init__(
        self,
        orchestrator_url: str = "http://localhost:8000",
        ws_url: str = "ws://localhost:8000/ws",
        tasks_per_hour: int = 100,
        agent_count: int = 5,
        log_level: str = "INFO"
    ):
        self.orchestrator_url = orchestrator_url
        self.ws_url = ws_url
        self.tasks_per_hour = tasks_per_hour
        self.agent_count = agent_count
        self.log_level = log_level

        self.running = True
        self.tasks_created = 0
        self.tasks_completed = 0
        self.start_time = None
        self.agents: Dict[str, Any] = {}

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        self.log("Received shutdown signal, stopping load simulation...")
        self.running = False

    def _timestamp(self) -> str:
        """ISO format timestamp"""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")

    def _corr_id(self) -> str:
        """Generate correlation ID"""
        return f"corr-{uuid.uuid4().hex[:12]}"

    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp and correlation ID format"""
        if level == "DEBUG" and self.log_level != "DEBUG":
            return
        corr_id = self._corr_id()
        print(f"[{self._timestamp()}] [corr-id:{corr_id}] {message}")

    def _generate_task(self) -> Dict[str, Any]:
        """Generate a random task"""
        task_type = random.choice(self.TASK_TYPES)
        template = random.choice(self.TASK_TEMPLATES)
        components = ["auth", "api", "database", "cache", "messaging", "scheduler"]
        features = ["caching", "retry", "validation", "logging", "metrics", "tracing"]
        topics = ["microservices", "event-driven", "REST", "GraphQL", "security", "scaling"]

        description = template.format(
            component=random.choice(components),
            feature=random.choice(features),
            topic=random.choice(topics)
        )

        return {
            "task_type": task_type,
            "description": description,
            "priority": random.randint(1, 10),
            "metadata": {
                "source": "load_simulator",
                "batch_id": f"batch-{uuid.uuid4().hex[:8]}",
                "generated_at": self._timestamp()
            }
        }

    async def create_task(self, session: aiohttp.ClientSession) -> Optional[str]:
        """Create a single task via REST API"""
        task_data = self._generate_task()
        corr_id = self._corr_id()

        try:
            async with session.post(
                f"{self.orchestrator_url}/tasks",
                json=task_data,
                headers={"X-Correlation-ID": corr_id}
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    task_id = result.get("task_id")
                    self.tasks_created += 1
                    self.log(f"Task created: task_id={task_id} type={task_data['task_type']}", "DEBUG")
                    return task_id
                else:
                    self.log(f"Task creation failed: HTTP {response.status}", "ERROR")
                    return None
        except Exception as e:
            self.log(f"Task creation error: {e}", "ERROR")
            return None

    async def run_agent(self, agent_id: str):
        """Run a simulated agent that processes tasks"""
        self.agents[agent_id] = {"status": "starting", "tasks_completed": 0}

        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    # Register agent
                    register_msg = {
                        "type": "register",
                        "agent_id": agent_id,
                        "agent_type": "load_test_agent",
                        "capabilities": {"task_types": self.TASK_TYPES}
                    }
                    await ws.send(json.dumps(register_msg))

                    # Wait for registration ack
                    ack = await ws.recv()
                    ack_data = json.loads(ack)
                    if ack_data.get("type") == "registration_ack":
                        self.agents[agent_id]["status"] = "idle"
                        self.log(f"Agent {agent_id} registered successfully")

                    # Process tasks
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            data = json.loads(msg)

                            if data.get("type") == "task_assignment":
                                task_id = data.get("task_id")
                                self.agents[agent_id]["status"] = "busy"
                                self.log(f"Agent {agent_id} received task {task_id}")

                                # Simulate task processing (0.5-3 seconds)
                                processing_time = random.uniform(0.5, 3.0)
                                await asyncio.sleep(processing_time)

                                # Complete task
                                complete_msg = {
                                    "type": "task_complete",
                                    "task_id": task_id,
                                    "agent_id": agent_id,
                                    "result": {
                                        "status": "success",
                                        "processing_time_ms": int(processing_time * 1000)
                                    },
                                    "reasoning_steps": [
                                        "Analyzed task requirements",
                                        "Executed task logic",
                                        "Validated results"
                                    ]
                                }
                                await ws.send(json.dumps(complete_msg))

                                self.tasks_completed += 1
                                self.agents[agent_id]["tasks_completed"] += 1
                                self.agents[agent_id]["status"] = "idle"
                                self.log(f"Agent {agent_id} completed task {task_id} in {processing_time:.2f}s")

                        except asyncio.TimeoutError:
                            # Send heartbeat
                            heartbeat = {"type": "heartbeat", "agent_id": agent_id}
                            await ws.send(json.dumps(heartbeat))

            except websockets.exceptions.ConnectionClosed:
                self.log(f"Agent {agent_id} connection closed, reconnecting...")
                self.agents[agent_id]["status"] = "reconnecting"
                await asyncio.sleep(2)
            except Exception as e:
                self.log(f"Agent {agent_id} error: {e}", "ERROR")
                self.agents[agent_id]["status"] = "error"
                await asyncio.sleep(5)

        self.agents[agent_id]["status"] = "stopped"

    async def task_generator(self, duration_seconds: int):
        """Generate tasks at configured rate"""
        interval = 3600 / self.tasks_per_hour  # seconds between tasks

        async with aiohttp.ClientSession() as session:
            end_time = datetime.utcnow() + timedelta(seconds=duration_seconds)

            while self.running and datetime.utcnow() < end_time:
                await self.create_task(session)

                # Add some randomness to the interval (80-120% of target)
                jitter = random.uniform(0.8, 1.2)
                await asyncio.sleep(interval * jitter)

    async def progress_reporter(self, duration_seconds: int):
        """Report progress every 5 minutes"""
        while self.running:
            await asyncio.sleep(300)  # 5 minutes

            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            rate = self.tasks_created / (elapsed / 3600) if elapsed > 0 else 0
            completion_rate = (self.tasks_completed / self.tasks_created * 100) if self.tasks_created > 0 else 0

            active_agents = sum(1 for a in self.agents.values() if a["status"] in ["idle", "busy"])

            self.log(
                f"Progress: {self.tasks_created} created, {self.tasks_completed} completed "
                f"({completion_rate:.1f}%), rate={rate:.1f}/hr, agents={active_agents}/{self.agent_count}"
            )

    async def run(self, duration_seconds: int):
        """Run the load simulation"""
        self.start_time = datetime.utcnow()

        # DevZen Enhancement #5: Timestamp anchor
        self.log(f"[START] Load simulation @ {self._timestamp()}")
        self.log(f"  Duration: {duration_seconds}s ({duration_seconds // 3600}h)")
        self.log(f"  Tasks per hour: {self.tasks_per_hour}")
        self.log(f"  Agent count: {self.agent_count}")
        self.log("-" * 60)

        # Start agents
        agent_tasks = []
        for i in range(self.agent_count):
            agent_id = f"load-agent-{i}"
            agent_tasks.append(asyncio.create_task(self.run_agent(agent_id)))

        # Wait for agents to connect
        await asyncio.sleep(2)

        # Start task generator and progress reporter
        generator_task = asyncio.create_task(self.task_generator(duration_seconds))
        reporter_task = asyncio.create_task(self.progress_reporter(duration_seconds))

        # Wait for generator to complete
        await generator_task

        # Stop simulation
        self.running = False

        # Cancel remaining tasks
        reporter_task.cancel()
        for task in agent_tasks:
            task.cancel()

        # Wait for cancellation
        await asyncio.sleep(1)

        # Final report
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        rate = self.tasks_created / (elapsed / 3600) if elapsed > 0 else 0
        completion_rate = (self.tasks_completed / self.tasks_created * 100) if self.tasks_created > 0 else 0

        self.log("-" * 60)
        self.log(f"[END] Load simulation complete")
        self.log(f"  Duration: {elapsed / 3600:.2f}h")
        self.log(f"  Tasks created: {self.tasks_created}")
        self.log(f"  Tasks completed: {self.tasks_completed} ({completion_rate:.1f}%)")
        self.log(f"  Actual rate: {rate:.1f} tasks/hour")

        for agent_id, info in self.agents.items():
            self.log(f"  {agent_id}: {info['tasks_completed']} tasks completed")


def parse_duration(duration_str: str) -> int:
    """Parse duration string to seconds"""
    duration_str = duration_str.lower().strip()

    if duration_str.endswith('h'):
        return int(duration_str[:-1]) * 3600
    elif duration_str.endswith('d'):
        return int(duration_str[:-1]) * 86400
    elif duration_str.endswith('m'):
        return int(duration_str[:-1]) * 60
    elif duration_str.endswith('s'):
        return int(duration_str[:-1])
    else:
        return int(duration_str)


def main():
    parser = argparse.ArgumentParser(
        description="AAOS Production Load Simulator (DevZen Enhanced)"
    )
    parser.add_argument(
        "--tasks-per-hour",
        type=int,
        default=100,
        help="Number of tasks to generate per hour. Default: 100"
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=5,
        help="Number of simulated agents. Default: 5"
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="24h",
        help="Simulation duration (e.g., '24h', '1d'). Default: 24h"
    )
    parser.add_argument(
        "--orchestrator-url",
        type=str,
        default="http://localhost:8000",
        help="Orchestrator REST URL. Default: http://localhost:8000"
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default="ws://localhost:8000/ws",
        help="Orchestrator WebSocket URL. Default: ws://localhost:8000/ws"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level. Default: INFO"
    )

    args = parser.parse_args()

    duration = parse_duration(args.duration)

    simulator = LoadSimulator(
        orchestrator_url=args.orchestrator_url,
        ws_url=args.ws_url,
        tasks_per_hour=args.tasks_per_hour,
        agent_count=args.agent_count,
        log_level=args.log_level
    )

    try:
        asyncio.run(simulator.run(duration))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
