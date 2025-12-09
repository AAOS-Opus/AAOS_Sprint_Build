#!/usr/bin/env python3
"""
AAOS Checkpoint Gate Script
DevZen Enhancement #9: Alert-condition gates for T+6h, T+12h, T+18h, T+24h checkpoints

Validates system health at predefined intervals and creates emergency snapshots if issues detected.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import redis
import requests


class CheckpointGate:
    """Executes checkpoint validation and alert gates"""

    def __init__(
        self,
        orchestrator_url: str = "http://localhost:8000",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        aaos_log: str = "aaos_prod.log",
        telemetry_log: str = "telemetry_baseline_24h.log",
        expected_agents: int = 5
    ):
        self.orchestrator_url = orchestrator_url
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.aaos_log = aaos_log
        self.telemetry_log = telemetry_log
        self.expected_agents = expected_agents

    def _timestamp(self) -> str:
        """ISO format timestamp"""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")

    def _log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        print(f"[{self._timestamp()}] [{level}] {message}")

    def check_errors(self) -> Tuple[bool, int]:
        """DevZen Enhancement #9: Error detection in logs"""
        error_count = 0

        if os.path.exists(self.aaos_log):
            with open(self.aaos_log, 'r') as f:
                for line in f:
                    if "[ERROR]" in line or "ERROR" in line.upper():
                        error_count += 1

        return error_count == 0, error_count

    def check_queue_health(self, max_queue_depth: int = 10) -> Tuple[bool, int]:
        """Check Redis queue depth"""
        try:
            queue_len = self.redis_client.llen("task_queue")
            return queue_len < max_queue_depth, queue_len
        except Exception as e:
            self._log(f"Redis error: {e}", "ERROR")
            return False, -1

    def check_agent_count(self) -> Tuple[bool, int]:
        """Check active agent count"""
        try:
            response = requests.get(f"{self.orchestrator_url}/agents", timeout=5)
            if response.status_code == 200:
                agents = response.json()
                active_count = len([a for a in agents if a.get("status") in ["idle", "busy"]])
                return active_count == self.expected_agents, active_count
            return False, 0
        except Exception as e:
            self._log(f"Orchestrator error: {e}", "ERROR")
            return False, 0

    def check_db_consistency(self, database_url: str = None) -> Tuple[bool, str]:
        """Check database consistency via orchestrator"""
        try:
            response = requests.get(f"{self.orchestrator_url}/tasks", timeout=10)
            if response.status_code == 200:
                tasks = response.json()
                # Count tasks by status
                status_counts = {}
                for task in tasks:
                    status = task.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                return True, json.dumps(status_counts)
            return False, "API error"
        except Exception as e:
            return False, str(e)

    def check_metrics(self) -> Tuple[bool, Dict[str, Any]]:
        """Check system metrics"""
        try:
            response = requests.get(f"{self.orchestrator_url}/metrics", timeout=5)
            if response.status_code == 200:
                metrics = response.json()
                return True, metrics
            return False, {}
        except Exception as e:
            self._log(f"Metrics error: {e}", "ERROR")
            return False, {}

    def check_redis_memory(self, max_mb: int = 100) -> Tuple[bool, float]:
        """Check Redis memory usage"""
        try:
            info = self.redis_client.info()
            memory_bytes = info.get("used_memory", 0)
            memory_mb = memory_bytes / (1024 * 1024)
            return memory_mb < max_mb, memory_mb
        except Exception as e:
            self._log(f"Redis memory check error: {e}", "ERROR")
            return False, 0

    def create_emergency_snapshot(self, checkpoint: str) -> str:
        """Create emergency snapshot archive"""
        snapshot_name = f"emergency_snapshot_{checkpoint}.tar.gz"

        files_to_archive = []
        for filepath in [self.aaos_log, self.telemetry_log, "baseline_summary.json"]:
            if os.path.exists(filepath):
                files_to_archive.append(filepath)

        if files_to_archive:
            with tarfile.open(snapshot_name, "w:gz") as tar:
                for filepath in files_to_archive:
                    tar.add(filepath)

        self._log(f"Emergency snapshot created: {snapshot_name}", "WARN")
        return snapshot_name

    def run_gate(self, checkpoint: str, auto_snapshot: bool = False, checkpoints_log: str = "checkpoints.log") -> bool:
        """Run checkpoint gate validation with DevZen Tweak #3: Detailed failure logging"""
        self._log(f"=" * 60)
        self._log(f"CHECKPOINT GATE: {checkpoint}")
        self._log(f"=" * 60)

        all_pass = True
        results = {}
        failure_reasons = []  # DevZen Tweak #3: Track detailed failure reasons

        # Error check (all checkpoints)
        error_pass, error_count = self.check_errors()
        results["errors"] = {"pass": error_pass, "count": error_count}
        self._log(f"Error check: {'PASS' if error_pass else 'FAIL'} ({error_count} errors)")
        if not error_pass:
            all_pass = False
            failure_reasons.append(f"Critical errors: {error_count}")
            self._log("Errors detected - pause telemetry capture and flag for review", "ERROR")

        # Queue health (all checkpoints) - congestion threshold is 50
        queue_pass, queue_len = self.check_queue_health(max_queue_depth=50)
        results["queue"] = {"pass": queue_pass, "length": queue_len}
        self._log(f"Queue health: {'PASS' if queue_pass else 'WARN'} ({queue_len} tasks)")
        if not queue_pass:
            failure_reasons.append(f"Queue congestion: {queue_len}")
            self._log("Queue building up", "WARN")

        # Agent count (all checkpoints)
        agent_pass, agent_count = self.check_agent_count()
        results["agents"] = {"pass": agent_pass, "count": agent_count}
        self._log(f"Agent count: {'PASS' if agent_pass else 'WARN'} ({agent_count}/{self.expected_agents})")
        if not agent_pass:
            failure_reasons.append(f"Agent loss: {agent_count}/{self.expected_agents}")
            self._log("Agent count mismatch", "WARN")

        # DB consistency (all checkpoints)
        db_pass, db_status = self.check_db_consistency()
        results["database"] = {"pass": db_pass, "status": db_status}
        self._log(f"DB consistency: {'PASS' if db_pass else 'WARN'} ({db_status})")

        # Metrics check (T+12h and later)
        if checkpoint in ["T12h", "T18h", "T24h"]:
            metrics_pass, metrics = self.check_metrics()
            results["metrics"] = {"pass": metrics_pass, "data": metrics}
            self._log(f"Metrics: {'PASS' if metrics_pass else 'WARN'}")
            if metrics:
                self._log(f"  Active agents: {metrics.get('active_agents', 'N/A')}")
                self._log(f"  Queued tasks: {metrics.get('queued_tasks', 'N/A')}")
                self._log(f"  Completed: {metrics.get('completed_tasks_total', 'N/A')}")

        # Memory trend (T+18h and later) - threshold 10MB/hour growth
        if checkpoint in ["T18h", "T24h"]:
            memory_pass, memory_mb = self.check_redis_memory()
            results["memory"] = {"pass": memory_pass, "mb": memory_mb}
            self._log(f"Redis memory: {'PASS' if memory_pass else 'WARN'} ({memory_mb:.1f} MB)")
            if not memory_pass:
                failure_reasons.append(f"Memory leak: {memory_mb:.1f} MB")
                self._log("Redis memory exceeding threshold", "WARN")

        # Final verdict
        self._log("-" * 60)

        # DevZen Tweak #3: Detailed failure logging for autonomous analysis
        status = "PASS" if not failure_reasons else "FAIL"
        log_line = f"[{datetime.utcnow().isoformat()}] CHECKPOINT: {checkpoint} STATUS={status}"
        if failure_reasons:
            log_line += f" REASONS={'; '.join(failure_reasons)}"

        # Write to checkpoints.log for autonomous monitoring
        with open(checkpoints_log, 'a') as f:
            f.write(log_line + "\n")
        self._log(f"Checkpoint logged to: {checkpoints_log}")

        if failure_reasons:
            all_pass = False
            results["verdict"] = "FAIL"
            results["failure_reasons"] = failure_reasons

            if auto_snapshot:
                snapshot = self.create_emergency_snapshot(checkpoint)
                self._log(f"CHECKPOINT FAILED - Emergency snapshot: {snapshot}", "ERROR")
                results["snapshot"] = snapshot

                # Generate emergency summary
                try:
                    import subprocess
                    subprocess.run(["python", "scripts/generate_baseline_summary.py", "--emergency"],
                                   capture_output=True, timeout=30)
                except Exception as e:
                    self._log(f"Emergency summary generation failed: {e}", "WARN")
            else:
                self._log(f"CHECKPOINT FAILED", "ERROR")
        else:
            self._log(f"CHECKPOINT PASSED", "INFO")
            results["verdict"] = "PASS"

        # Write results
        results_file = f"checkpoint_{checkpoint}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self._log(f"Results written to: {results_file}")

        # Print status for script capture
        print(status)

        return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="AAOS Checkpoint Gate Script (DevZen Enhancement #9)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        choices=["T6h", "T12h", "T18h", "T24h"],
        help="Checkpoint to execute (T6h, T12h, T18h, T24h)"
    )
    parser.add_argument(
        "--orchestrator-url",
        type=str,
        default="http://localhost:8000",
        help="Orchestrator URL. Default: http://localhost:8000"
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host. Default: localhost"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port. Default: 6379"
    )
    parser.add_argument(
        "--aaos-log",
        type=str,
        default="aaos_prod.log",
        help="AAOS production log file. Default: aaos_prod.log"
    )
    parser.add_argument(
        "--expected-agents",
        type=int,
        default=5,
        help="Expected number of agents. Default: 5"
    )
    parser.add_argument(
        "--auto-snapshot",
        action="store_true",
        help="Automatically create emergency snapshot on failure"
    )
    parser.add_argument(
        "--checkpoints-log",
        type=str,
        default="checkpoints.log",
        help="Checkpoint log file. Default: checkpoints.log"
    )

    args = parser.parse_args()

    gate = CheckpointGate(
        orchestrator_url=args.orchestrator_url,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        aaos_log=args.aaos_log,
        expected_agents=args.expected_agents
    )

    success = gate.run_gate(
        args.checkpoint,
        auto_snapshot=args.auto_snapshot,
        checkpoints_log=args.checkpoints_log
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
