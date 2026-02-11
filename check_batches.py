from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def format_timestamp(ts: int | None) -> str:
    """Format Unix timestamp to readable string."""
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def check_batches(client: OpenAI, limit: int = 50) -> None:
    """Check for in-progress batches and display their details."""
    batches = client.batches.list(limit=limit)
    
    active_statuses = {"validating", "in_progress", "finalizing"}
    active_batches = [
        b for b in batches.data if b.status in active_statuses
    ]
    
    if not active_batches:
        print("No batches currently in progress.")
        return
    
    print(f"Found {len(active_batches)} batch(es) in progress:\n")
    print("=" * 80)
    
    for idx, batch in enumerate(active_batches, 1):
        print(f"\n[{idx}] Batch ID: {batch.id}")
        print(f"    Status: {batch.status}")
        print(f"    Model: {batch.model}")
        print(f"    Endpoint: {batch.endpoint}")
        print(f"    Created: {format_timestamp(batch.created_at)}")
        
        if batch.in_progress_at:
            elapsed = None
            if batch.in_progress_at:
                now = datetime.now().timestamp()
                elapsed = now - batch.in_progress_at
            print(f"    Started: {format_timestamp(batch.in_progress_at)}")
            if elapsed:
                print(f"    Elapsed: {format_duration(elapsed)}")
        
        if batch.finalizing_at:
            print(f"    Finalizing since: {format_timestamp(batch.finalizing_at)}")
        
        if batch.request_counts:
            counts = batch.request_counts
            total = getattr(counts, "total", 0)
            completed = getattr(counts, "completed", 0)
            failed = getattr(counts, "failed", 0)
            print(f"    Requests: {completed}/{total} completed, {failed} failed")
            if total > 0:
                pct = (completed / total) * 100
                print(f"    Progress: {pct:.1f}%")
        
        if batch.completion_window:
            print(f"    Completion window: {batch.completion_window}")
        
        if batch.expires_at:
            expires_str = format_timestamp(batch.expires_at)
            print(f"    Expires: {expires_str}")
        
        if batch.errors and batch.errors.data:
            print(f"    Errors: {len(batch.errors.data)} error(s)")
            for err in batch.errors.data[:3]:  # Show first 3 errors
                print(f"      - {err.get('message', 'Unknown error')}")
            if len(batch.errors.data) > 3:
                print(f"      ... and {len(batch.errors.data) - 3} more")
        
        print("-" * 80)
    
    # Summary
    print(f"\nSummary: {len(active_batches)} active batch(es)")
    status_counts = {}
    for batch in active_batches:
        status_counts[batch.status] = status_counts.get(batch.status, 0) + 1
    
    for status, count in sorted(status_counts.items()):
        print(f"  - {status}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check for in-progress OpenAI batch jobs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of batches to check (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    
    client = OpenAI(timeout=30.0, max_retries=3)
    check_batches(client, args.limit)


if __name__ == "__main__":
    main()

