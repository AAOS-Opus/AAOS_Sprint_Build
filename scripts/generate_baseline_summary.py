#!/usr/bin/env python3
"""
AAOS Baseline Summary Generator
DevZen Enhancement: Generates JSON summary from telemetry capture data

Analyzes telemetry logs and produces baseline_summary.json with key metrics.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


def parse_telemetry_file(input_file: str) -> List[Dict[str, Any]]:
    """Parse telemetry log file into list of metric samples"""
    samples = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                continue

    return samples


def calculate_throughput_metrics(samples: List[Dict[str, Any]], window_seconds: int = 3600) -> Dict[str, Any]:
    """Calculate task throughput metrics"""
    if not samples:
        return {"avg": 0, "stddev": 0, "unit": "tasks/hour"}

    # Calculate completed tasks per hour
    hourly_counts = []
    completed_values = [s.get("completed_tasks_total", 0) for s in samples]

    if len(completed_values) < 2:
        return {"avg": 0, "stddev": 0, "unit": "tasks/hour"}

    # Calculate hourly rate based on samples (assuming 60s intervals)
    sample_interval = 60  # seconds
    samples_per_hour = 3600 // sample_interval

    for i in range(samples_per_hour, len(completed_values)):
        hourly_diff = completed_values[i] - completed_values[i - samples_per_hour]
        hourly_counts.append(hourly_diff)

    if not hourly_counts:
        # Not enough data for hourly calculation
        total_tasks = completed_values[-1] - completed_values[0]
        hours = len(completed_values) * sample_interval / 3600
        avg_rate = total_tasks / hours if hours > 0 else 0
        return {"avg": round(avg_rate, 2), "stddev": 0, "unit": "tasks/hour"}

    return {
        "avg": round(statistics.mean(hourly_counts), 2),
        "stddev": round(statistics.stdev(hourly_counts) if len(hourly_counts) > 1 else 0, 2),
        "unit": "tasks/hour"
    }


def calculate_queue_latency(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate queue depth and inferred latency metrics"""
    queue_depths = [s.get("task_queue_length", 0) for s in samples if "task_queue_length" in s]

    if not queue_depths:
        return {"avg": 0, "stddev": 0, "unit": "tasks_in_queue"}

    return {
        "avg": round(statistics.mean(queue_depths), 2),
        "stddev": round(statistics.stdev(queue_depths) if len(queue_depths) > 1 else 0, 2),
        "unit": "tasks_in_queue"
    }


def calculate_reassignment_rate(samples: List[Dict[str, Any]], aaos_log_file: Optional[str] = None) -> float:
    """Calculate task reassignment rate"""
    if not aaos_log_file:
        return 0.0

    try:
        with open(aaos_log_file, 'r') as f:
            content = f.read()
            reassign_count = content.lower().count("reassign")
            task_count = content.lower().count("task created")

            if task_count > 0:
                return round((reassign_count / task_count) * 100, 2)
    except FileNotFoundError:
        pass

    return 0.0


def count_errors(samples: List[Dict[str, Any]], aaos_log_file: Optional[str] = None) -> int:
    """Count errors from telemetry and log file"""
    error_count = 0

    # Count errors in telemetry samples
    for sample in samples:
        if sample.get("orchestrator_error"):
            error_count += 1
        if sample.get("redis_error"):
            error_count += 1
        if sample.get("health_status") == "unhealthy":
            error_count += 1

    # Count errors in aaos log
    if aaos_log_file:
        try:
            with open(aaos_log_file, 'r') as f:
                for line in f:
                    if "[ERROR]" in line or "ERROR" in line.upper():
                        error_count += 1
        except FileNotFoundError:
            pass

    return error_count


def calculate_redis_memory(samples: List[Dict[str, Any]]) -> float:
    """Calculate average Redis memory usage in MB"""
    memory_values = []

    for sample in samples:
        memory_bytes = sample.get("redis_used_memory_bytes", 0)
        if memory_bytes:
            memory_values.append(memory_bytes / (1024 * 1024))

    if not memory_values:
        return 0.0

    return round(statistics.mean(memory_values), 2)


def calculate_agent_uptime(samples: List[Dict[str, Any]], expected_agents: int = 5) -> float:
    """Calculate agent availability percentage"""
    if not samples or expected_agents == 0:
        return 0.0

    uptime_scores = []

    for sample in samples:
        active = sample.get("active_agents", 0)
        score = (active / expected_agents) * 100
        uptime_scores.append(min(score, 100))  # Cap at 100%

    return round(statistics.mean(uptime_scores), 2)


def check_stability_window(samples: List[Dict[str, Any]], window_hours: int = 6) -> str:
    """Check if system stayed within thresholds during rolling window"""
    if len(samples) < 10:
        return "INSUFFICIENT_DATA"

    # Simplified stability check: no health issues and reasonable queue depth
    health_ok = all(s.get("health_status") == "healthy" for s in samples[-100:])
    queue_stable = all(s.get("task_queue_length", 0) < 50 for s in samples[-100:])

    if health_ok and queue_stable:
        return "PASS"
    elif health_ok:
        return "WARN_QUEUE_DEPTH"
    else:
        return "FAIL"


def generate_summary(
    input_file: str,
    output_file: str,
    window_seconds: int = 3600,
    aaos_log_file: Optional[str] = None
) -> Dict[str, Any]:
    """Generate baseline summary JSON"""
    print(f"Parsing telemetry from: {input_file}")
    samples = parse_telemetry_file(input_file)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("ERROR: No samples found in telemetry file")
        return {}

    # Calculate metrics
    summary = {
        "throughput": calculate_throughput_metrics(samples, window_seconds),
        "latency": calculate_queue_latency(samples),
        "reassignment_rate": calculate_reassignment_rate(samples, aaos_log_file),
        "error_count": count_errors(samples, aaos_log_file),
        "redis_memory_mb": calculate_redis_memory(samples),
        "agent_uptime": calculate_agent_uptime(samples),
        "stability_window": check_stability_window(samples),
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "input_file": input_file,
            "sample_count": len(samples),
            "window_seconds": window_seconds,
            "first_sample": samples[0].get("timestamp") if samples else None,
            "last_sample": samples[-1].get("timestamp") if samples else None
        }
    }

    # Write summary to file
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to: {output_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="AAOS Baseline Summary Generator"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input telemetry log file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="baseline_summary.json",
        help="Output JSON file. Default: baseline_summary.json"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3600,
        help="Analysis window in seconds. Default: 3600 (1 hour)"
    )
    parser.add_argument(
        "--aaos-log",
        type=str,
        default="aaos_prod.log",
        help="AAOS production log file for error counting. Default: aaos_prod.log"
    )

    args = parser.parse_args()

    summary = generate_summary(
        input_file=args.input,
        output_file=args.output,
        window_seconds=args.window,
        aaos_log_file=args.aaos_log
    )

    if summary:
        print("\n" + "=" * 60)
        print("BASELINE SUMMARY")
        print("=" * 60)
        print(f"Throughput:        {summary['throughput']['avg']} +/- {summary['throughput']['stddev']} tasks/hour")
        print(f"Queue Depth:       {summary['latency']['avg']} +/- {summary['latency']['stddev']} tasks")
        print(f"Reassignment Rate: {summary['reassignment_rate']}%")
        print(f"Error Count:       {summary['error_count']}")
        print(f"Redis Memory:      {summary['redis_memory_mb']} MB")
        print(f"Agent Uptime:      {summary['agent_uptime']}%")
        print(f"Stability Window:  {summary['stability_window']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
