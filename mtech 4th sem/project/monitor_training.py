#!/usr/bin/env python3
"""
Real-time Training Monitor
Watch training progress and display key metrics
"""

import time
import sys
from pathlib import Path
import json
import subprocess

def get_latest_log():
    """Get latest training log content"""
    log_file = Path("training_live.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            return f.read()
    return ""

def extract_progress(log_content):
    """Extract training progress from logs"""
    lines = log_content.split('\n')

    # Find latest epoch progress
    latest_epoch = None
    latest_loss = None
    latest_acc = None

    for line in reversed(lines):
        if 'Epoch' in line and 'it/s' in line:
            # Parse tqdm line
            if 'loss=' in line and 'acc=' in line:
                try:
                    loss_part = line.split('loss=')[1].split(',')[0]
                    acc_part = line.split('acc=')[1].split(']')[0]
                    latest_loss = float(loss_part)
                    latest_acc = float(acc_part)

                    # Extract epoch
                    epoch_part = line.split('Epoch ')[1].split(':')[0]
                    latest_epoch = epoch_part
                    break
                except:
                    pass

    return latest_epoch, latest_loss, latest_acc

def check_data_loading_reports():
    """Check latest data loading reports"""
    log_dir = Path("logs/data_loading")
    if not log_dir.exists():
        return None

    reports = sorted(log_dir.glob("epoch_*.json"))
    if not reports:
        return None

    latest_report = reports[-1]
    with open(latest_report, 'r') as f:
        return json.load(f)

def main():
    print("=" * 70)
    print("🚀 TRAINING MONITOR - Real-time Progress Tracker")
    print("=" * 70)
    print("\nPress Ctrl+C to stop monitoring\n")

    last_content_length = 0

    try:
        while True:
            # Get latest log
            log_content = get_latest_log()

            # Only update if new content
            if len(log_content) > last_content_length:
                last_content_length = len(log_content)

                # Clear screen
                print("\033[2J\033[H", end='')

                print("=" * 70)
                print("🚀 TRAINING MONITOR")
                print("=" * 70)

                # Extract progress
                epoch, loss, acc = extract_progress(log_content)

                if epoch:
                    print(f"\n📊 Current Progress:")
                    print(f"  Epoch: {epoch}")
                    print(f"  Loss:  {loss:.4f}")
                    print(f"  Acc:   {acc:.4f}")

                # Check data loading reports
                report = check_data_loading_reports()
                if report:
                    stats = report.get('statistics', {})
                    print(f"\n📁 Data Loading (Epoch {report['epoch']}):")
                    print(f"  Total samples: {stats.get('total_samples_attempted', 'N/A')}")
                    print(f"  Failed: {stats.get('failed_samples', 0)} ({stats.get('error_rate', 0)*100:.2f}%)")
                    print(f"  Skipped: {stats.get('skip_count', 0)}")
                    print(f"  Avg load time: {stats.get('avg_load_time_ms', 0):.1f}ms")

                # Check if training is still running
                result = subprocess.run(
                    ["ps", "aux"],
                    capture_output=True,
                    text=True
                )

                if "train.py" in result.stdout:
                    print(f"\n✅ Training process: RUNNING")
                else:
                    print(f"\n⚠️  Training process: STOPPED")

                # Show last few log lines
                print(f"\n📝 Recent logs:")
                print("-" * 70)
                recent_lines = log_content.split('\n')[-5:]
                for line in recent_lines:
                    if line.strip():
                        print(f"  {line[:68]}")

                print("-" * 70)
                print(f"\n⏱️  Last updated: {time.strftime('%H:%M:%S')}")
                print(f"🔄 Refreshing every 5 seconds...")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped")
        print("=" * 70)

        # Show final status
        print("\nFinal Status:")
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )

        if "train.py" in result.stdout:
            print("✅ Training is still running in background")
            print("\nTo view logs: tail -f training_live.log")
            print("To stop training: pkill -f train.py")
        else:
            print("✓ Training has completed")
            print("\nCheck logs: cat training_live.log")
            print("View results: ls -la checkpoints/ logs/")

if __name__ == "__main__":
    main()
