"""
Enhanced Training Monitor with Data Loading Health Tracking
Tracks errors, performance metrics, and dataset quality during training
"""

import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoadingMonitor:
    """Monitor data loading health during training"""

    def __init__(self, log_dir="logs/data_loading"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Tracking
        self.epoch = 0
        self.total_samples_attempted = 0
        self.failed_samples = []
        self.error_counts = defaultdict(int)
        self.skip_counts = 0

        # Performance
        self.load_times = []
        self.start_time = None

    def start_epoch(self, epoch: int):
        """Start monitoring a new epoch"""
        self.epoch = epoch
        self.total_samples_attempted = 0
        self.failed_samples = []
        self.error_counts = defaultdict(int)
        self.skip_counts = 0
        self.load_times = []
        self.start_time = time.time()

        logger.info(f"📊 Started monitoring epoch {epoch}")

    def record_error(self, idx: int, video_path: str, error: str):
        """Record a failed sample"""
        self.failed_samples.append({
            'idx': idx,
            'video_path': video_path,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.error_counts[error] += 1

    def record_skip(self):
        """Record when a sample is skipped"""
        self.skip_counts += 1

    def record_load_time(self, load_time: float):
        """Record sample loading time"""
        self.load_times.append(load_time)

    def record_sample_attempt(self):
        """Increment total sample attempts"""
        self.total_samples_attempted += 1

    def end_epoch(self, total_batches: int):
        """End epoch and generate report"""
        duration = time.time() - self.start_time

        # Calculate statistics
        error_rate = len(self.failed_samples) / self.total_samples_attempted if self.total_samples_attempted > 0 else 0
        skip_rate = self.skip_counts / self.total_samples_attempted if self.total_samples_attempted > 0 else 0

        avg_load_time = sum(self.load_times) / len(self.load_times) if self.load_times else 0

        # Generate report
        report = {
            'epoch': self.epoch,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'statistics': {
                'total_samples_attempted': self.total_samples_attempted,
                'total_batches': total_batches,
                'failed_samples': len(self.failed_samples),
                'error_rate': error_rate,
                'skip_count': self.skip_counts,
                'skip_rate': skip_rate,
                'avg_load_time_ms': avg_load_time * 1000
            },
            'error_breakdown': dict(self.error_counts),
            'failed_samples': self.failed_samples
        }

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA LOADING REPORT - Epoch {self.epoch}")
        logger.info(f"{'='*60}")
        logger.info(f"  Total samples: {self.total_samples_attempted}")
        logger.info(f"  Failed: {len(self.failed_samples)} ({error_rate:.2%})")
        logger.info(f"  Skipped: {self.skip_counts} ({skip_rate:.2%})")
        logger.info(f"  Avg load time: {avg_load_time*1000:.1f}ms")

        if self.error_counts:
            logger.info(f"\n  Error breakdown:")
            for error, count in sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    - {error}: {count}")

        # Alert on high error rate
        if error_rate > 0.05:  # 5% threshold
            logger.warning(f"\n  ⚠️  HIGH ERROR RATE: {error_rate:.1%} of samples failed!")
            logger.warning(f"  Consider investigating dataset quality")
        elif error_rate > 0:
            logger.info(f"\n  ✓ Error rate is acceptable: {error_rate:.2%}")
        else:
            logger.info(f"\n  ✓ Perfect! No errors detected")

        logger.info(f"{'='*60}\n")

        # Save report
        report_file = self.log_dir / f"epoch_{self.epoch:03d}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Report saved: {report_file}")

        return report

    def get_summary(self):
        """Get current summary statistics"""
        return {
            'epoch': self.epoch,
            'total_attempted': self.total_samples_attempted,
            'failed': len(self.failed_samples),
            'skipped': self.skip_counts
        }


class TrainingProgressTracker:
    """Track overall training progress and health"""

    def __init__(self, log_dir="logs/training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.start_time = time.time()
        self.epoch_metrics = []

    def log_epoch(self, epoch: int, metrics: dict, data_loading_report: dict = None):
        """Log epoch metrics"""

        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            'metrics': metrics
        }

        if data_loading_report:
            epoch_data['data_loading'] = data_loading_report['statistics']

        self.epoch_metrics.append(epoch_data)

        # Save cumulative log
        log_file = self.log_dir / "training_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)

        # Print summary
        logger.info(f"\n📈 EPOCH {epoch} SUMMARY")
        logger.info(f"  Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"  Train Acc:  {metrics.get('train_accuracy', 'N/A'):.4f}")
        logger.info(f"  Val Loss:   {metrics.get('val_loss', 'N/A'):.4f}")
        logger.info(f"  Val Acc:    {metrics.get('val_accuracy', 'N/A'):.4f}")

        if 'val_auc_roc' in metrics:
            logger.info(f"  Val AUC:    {metrics['val_auc_roc']:.4f}")

    def get_best_epoch(self, metric='val_auc_roc'):
        """Get epoch with best metric"""
        if not self.epoch_metrics:
            return None

        best = max(self.epoch_metrics, key=lambda x: x['metrics'].get(metric, 0))
        return best['epoch'], best['metrics'][metric]


def main():
    """Test monitoring system"""
    import random

    print("Testing Data Loading Monitor...\n")

    monitor = DataLoadingMonitor(log_dir="logs/test_monitor")
    tracker = TrainingProgressTracker(log_dir="logs/test_monitor")

    # Simulate 2 epochs
    for epoch in range(1, 3):
        monitor.start_epoch(epoch)

        # Simulate loading samples
        for i in range(100):
            monitor.record_sample_attempt()

            # Simulate occasional errors
            if random.random() < 0.02:  # 2% error rate
                monitor.record_error(
                    idx=i,
                    video_path=f"test/video_{i}.mp4",
                    error="Cannot open with OpenCV"
                )
                monitor.record_skip()

            # Simulate load times
            load_time = random.uniform(0.01, 0.05)  # 10-50ms
            monitor.record_load_time(load_time)

        # End epoch
        data_report = monitor.end_epoch(total_batches=10)

        # Log training metrics
        metrics = {
            'train_loss': random.uniform(0.5, 1.5),
            'train_accuracy': random.uniform(0.6, 0.9),
            'val_loss': random.uniform(0.5, 1.5),
            'val_accuracy': random.uniform(0.6, 0.9),
            'val_auc_roc': random.uniform(0.7, 0.95)
        }

        tracker.log_epoch(epoch, metrics, data_report)

    # Get best epoch
    best_epoch, best_score = tracker.get_best_epoch('val_auc_roc')
    print(f"\n✓ Best epoch: {best_epoch} (AUC: {best_score:.4f})")
    print(f"✓ Logs saved to: logs/test_monitor/")


if __name__ == "__main__":
    main()
