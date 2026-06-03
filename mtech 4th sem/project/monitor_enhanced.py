#!/usr/bin/env python3
"""
Enhanced Training Monitor - Real-time training progress with metrics
"""

import time
import re
from pathlib import Path
import argparse
from datetime import datetime


def parse_progress_line(line):
    """Extract progress info from tqdm progress lines"""
    # Match pattern like: "Epoch 1/5:  10%|...|17/172 [11:02<1:37:51, 37.88s/it, loss=1.3335, acc=0.4246]"
    pattern = r'Epoch\s+(\d+)/(\d+):\s+(\d+)%.*?\|\s*(\d+)/(\d+)\s+\[([^\]]+)<([^\]]+),\s*([^,]+),\s*loss=([^,]+),\s*acc=([^\]]+)\]'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'total_epochs': int(match.group(2)),
            'progress_pct': int(match.group(3)),
            'current_batch': int(match.group(4)),
            'total_batches': int(match.group(5)),
            'elapsed': match.group(6),
            'remaining': match.group(7),
            'batch_time': match.group(8),
            'loss': float(match.group(9)),
            'acc': float(match.group(10))
        }
    return None


def format_metrics_bar(progress_pct, width=50):
    """Create a visual progress bar"""
    filled = int(width * progress_pct / 100)
    bar = '█' * filled + '░' * (width - filled)
    return bar


def get_latest_progress(log_file):
    """Extract the latest training progress from log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Search backwards for the latest progress line
        for line in reversed(lines):
            progress = parse_progress_line(line)
            if progress:
                return progress, lines
        return None, lines
    except Exception as e:
        return None, []


def monitor_training(log_dir="logs", checkpoint_dir="checkpoints", refresh_interval=10):
    """
    Enhanced monitoring with real-time metrics
    """
    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)

    print("\033[2J\033[H")  # Clear screen
    print("=" * 100)
    print("🚀 ENHANCED TRAINING MONITOR")
    print("=" * 100)
    print(f"📁 Log directory: {log_dir}")
    print(f"💾 Checkpoint directory: {checkpoint_dir}")
    print(f"🔄 Refresh interval: {refresh_interval}s")
    print("=" * 100)
    print("\n⌨️  Press Ctrl+C to stop monitoring\n")

    start_time = time.time()
    iteration = 0
    last_progress = None

    try:
        while True:
            iteration += 1
            elapsed = time.time() - start_time

            # Clear screen and reposition cursor
            print("\033[2J\033[H")

            print("=" * 100)
            print(f"📊 TRAINING MONITOR - Update #{iteration} | {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 100)
            print()

            # Find latest log file
            log_files = list(log_dir.glob("*.log")) if log_dir.exists() else []

            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                progress, log_lines = get_latest_progress(latest_log)

                if progress:
                    last_progress = progress

                    # Display current epoch progress
                    print(f"📈 CURRENT PROGRESS")
                    print("-" * 100)
                    print(f"   Epoch: {progress['epoch']}/{progress['total_epochs']}")
                    print(f"   Batch: {progress['current_batch']}/{progress['total_batches']} ({progress['progress_pct']}%)")
                    print()
                    print(f"   {format_metrics_bar(progress['progress_pct'], 60)}")
                    print()
                    print(f"   📉 Loss: {progress['loss']:.4f}")
                    print(f"   ✅ Accuracy: {progress['acc']*100:.2f}%")
                    print(f"   ⏱️  Batch Time: {progress['batch_time']}")
                    print(f"   ⏳ Elapsed: {progress['elapsed']} | Remaining: {progress['remaining']}")
                    print()

                elif last_progress:
                    # Show last known progress
                    print(f"📈 LAST KNOWN PROGRESS (waiting for updates...)")
                    print("-" * 100)
                    print(f"   Epoch: {last_progress['epoch']}/{last_progress['total_epochs']}")
                    print(f"   Batch: {last_progress['current_batch']}/{last_progress['total_batches']} ({last_progress['progress_pct']}%)")
                    print(f"   Loss: {last_progress['loss']:.4f} | Accuracy: {last_progress['acc']*100:.2f}%")
                    print()
                else:
                    print("⏳ Training initializing... (no progress data yet)")
                    print()

                # Show recent log entries
                print("📝 RECENT LOG ENTRIES")
                print("-" * 100)
                relevant_logs = [l for l in log_lines[-15:] if 'INFO' in l and not re.search(r'Epoch\s+\d+/\d+:', l)]
                for line in relevant_logs[-8:]:
                    # Clean up the line
                    clean_line = line.strip()
                    if clean_line:
                        # Truncate if too long
                        if len(clean_line) > 95:
                            clean_line = clean_line[:92] + "..."
                        print(f"   {clean_line}")
                print()

            else:
                print("📝 No log files found - waiting for training to start...")
                print()

            # Check for checkpoints
            print("💾 CHECKPOINTS")
            print("-" * 100)
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pth"))
                if checkpoints:
                    for ckpt in sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
                        size_mb = ckpt.stat().st_size / (1024**2)
                        mtime = time.strftime('%H:%M:%S', time.localtime(ckpt.stat().st_mtime))
                        print(f"   ✓ {ckpt.name:30s} {size_mb:8.1f} MB  (saved at {mtime})")
                else:
                    print("   No checkpoints yet (saved after each epoch)")
            else:
                print("   Checkpoint directory not found")
            print()

            # Process info
            print("🖥️  SYSTEM INFO")
            print("-" * 100)
            import subprocess
            try:
                result = subprocess.run(
                    ["ps", "aux"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.split('\n'):
                    if 'python train.py' in line and 'grep' not in line:
                        parts = line.split()
                        if len(parts) >= 11:
                            print(f"   CPU: {parts[2]}% | Memory: {parts[3]}% | Runtime: {parts[9]}")
                            break
                else:
                    print("   Training process not found")
            except:
                print("   Could not retrieve process info")

            print()
            print("=" * 100)
            print(f"⏰ Next update in {refresh_interval}s... | Monitoring for: {int(elapsed//60)}m {int(elapsed%60)}s")
            print("=" * 100)

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
        print(f"Total monitoring time: {int(elapsed//60)}m {int(elapsed%60)}s")


def main():
    parser = argparse.ArgumentParser(description="Enhanced training monitor with real-time metrics")
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory containing training logs"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=10,
        help="Refresh interval in seconds"
    )

    args = parser.parse_args()

    monitor_training(
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        refresh_interval=args.refresh
    )


if __name__ == "__main__":
    main()
