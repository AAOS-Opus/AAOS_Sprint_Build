#!/usr/bin/env python3
"""
AAOS Baseline Telemetry Capture Script
DevZen Enhancement #4: Prevents runaway log growth with safety parameters

Captures system metrics at regular intervals for 24-hour baseline analysis.
"""

import argparse
import json
import os
import sys
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
import requests


class TelemetryCapture:
    """Captures and logs telemetry metrics with safety parameters"""

    def __init__(
        self,
        output_file: str,
        sample_interval: int = 60,
        flush_interval: int = 300,
        max_log_size_mb: int = 100,
        orchestrator_url: str = "http://localhost:8000",
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        self.output_file = output_file
        self.sample_interval = sample_interval
        self.flush_interval = flush_interval
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.orchestrator_url = orchestrator_url
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        self.running = True
        self.buffer = []
        self.last_flush = time.time()
        self.start_time = None
        self.samples_collected = 0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        print(f"\n[{self._timestamp()}] Received shutdown signal, flushing buffer...")
        self.running = False
        self._flush_buffer()

    def _timestamp(self) -> str:
        """ISO format timestamp"""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")

    def _check_log_size(self) -> bool:
        """DevZen Enhancement #4: Check if log file exceeds max size"""
        if os.path.exists(self.output_file):
            size = os.path.getsize(self.output_file)
            if size >= self.max_log_size_bytes:
                print(f"[{self._timestamp()}] WARNING: Log file exceeded {self.max_log_size_bytes // (1024*1024)}MB limit")
                return False
        return True

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from orchestrator and Redis"""
        metrics = {
            "timestamp": self._timestamp(),
            "sample_number": self.samples_collected + 1
        }

        # Orchestrator metrics
        try:
            response = requests.get(f"{self.orchestrator_url}/metrics", timeout=5)
            if response.status_code == 200:
                orchestrator_metrics = response.json()
                metrics.update({
                    "active_agents": orchestrator_metrics.get("active_agents", 0),
                    "queued_tasks": orchestrator_metrics.get("queued_tasks", 0),
                    "completed_tasks_total": orchestrator_metrics.get("completed_tasks_total", 0),
                    "agents_busy": orchestrator_metrics.get("agents_busy", 0)
                })
            else:
                metrics["orchestrator_error"] = f"HTTP {response.status_code}"
        except Exception as e:
            metrics["orchestrator_error"] = str(e)

        # Redis metrics
        try:
            info = self.redis_client.info()
            metrics.update({
                "redis_used_memory_human": info.get("used_memory_human", "N/A"),
                "redis_used_memory_bytes": info.get("used_memory", 0),
                "redis_connected_clients": info.get("connected_clients", 0),
                "redis_total_commands_processed": info.get("total_commands_processed", 0)
            })

            # Queue lengths
            metrics["task_queue_length"] = self.redis_client.llen("task_queue")
            metrics["completed_queue_length"] = self.redis_client.llen("completed_queue")
        except Exception as e:
            metrics["redis_error"] = str(e)

        # Health check
        try:
            response = requests.get(f"{self.orchestrator_url}/health", timeout=5)
            metrics["health_status"] = response.json().get("status", "unknown") if response.status_code == 200 else "unhealthy"
        except Exception as e:
            metrics["health_status"] = "unreachable"
            metrics["health_error"] = str(e)

        return metrics

    def _flush_buffer(self):
        """Flush buffer to output file"""
        if not self.buffer:
            return

        with open(self.output_file, "a") as f:
            for entry in self.buffer:
                f.write(json.dumps(entry) + "\n")

        print(f"[{self._timestamp()}] Flushed {len(self.buffer)} samples to {self.output_file}")
        self.buffer = []
        self.last_flush = time.time()

    def run(self, duration_seconds: int):
        """Run telemetry capture for specified duration"""
        self.start_time = datetime.utcnow()
        end_time = self.start_time + timedelta(seconds=duration_seconds)

        print(f"[{self._timestamp()}] Starting telemetry capture")
        print(f"  Duration: {duration_seconds}s ({duration_seconds // 3600}h)")
        print(f"  Sample interval: {self.sample_interval}s")
        print(f"  Flush interval: {self.flush_interval}s")
        print(f"  Max log size: {self.max_log_size_bytes // (1024*1024)}MB")
        print(f"  Output file: {self.output_file}")
        print("-" * 60)

        while self.running and datetime.utcnow() < end_time:
            # Check log size limit
            if not self._check_log_size():
                print(f"[{self._timestamp()}] Stopping: Log size limit reached")
                break

            # Collect metrics
            metrics = self._collect_metrics()
            self.buffer.append(metrics)
            self.samples_collected += 1

            # Log progress every 10 samples
            if self.samples_collected % 10 == 0:
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()
                remaining = duration_seconds - elapsed
                print(f"[{self._timestamp()}] Collected {self.samples_collected} samples, "
                      f"{remaining/3600:.1f}h remaining")

            # Flush if interval reached
            if time.time() - self.last_flush >= self.flush_interval:
                self._flush_buffer()

            # Wait for next sample
            time.sleep(self.sample_interval)

        # Final flush
        self._flush_buffer()

        print("-" * 60)
        print(f"[{self._timestamp()}] Telemetry capture complete")
        print(f"  Total samples: {self.samples_collected}")
        print(f"  Duration: {(datetime.utcnow() - self.start_time).total_seconds() / 3600:.2f}h")

        return self.samples_collected


def parse_duration(duration_str: str) -> int:
    """Parse duration string (e.g., '24h', '86400', '1d') to seconds"""
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


def parse_size(size_str: str) -> int:
    """Parse size string (e.g., '100MB', '1GB') to MB"""
    size_str = size_str.upper().strip()

    if size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2])
    else:
        return int(size_str)


def main():
    parser = argparse.ArgumentParser(
        description="AAOS Baseline Telemetry Capture (DevZen Enhanced)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=str,
        default="86400",
        help="Capture duration (e.g., '24h', '86400', '1d'). Default: 86400 (24h)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="telemetry_baseline_24h.log",
        help="Output file path. Default: telemetry_baseline_24h.log"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=60,
        help="Sample interval in seconds. Default: 60"
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=300,
        help="Flush interval in seconds. Default: 300"
    )
    parser.add_argument(
        "--max-log-size",
        type=str,
        default="100MB",
        help="Maximum log file size (e.g., '100MB', '1GB'). Default: 100MB"
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

    args = parser.parse_args()

    duration = parse_duration(args.duration)
    max_size = parse_size(args.max_log_size)

    capture = TelemetryCapture(
        output_file=args.output,
        sample_interval=args.sample_interval,
        flush_interval=args.flush_interval,
        max_log_size_mb=max_size,
        orchestrator_url=args.orchestrator_url,
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )

    try:
        capture.run(duration)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
