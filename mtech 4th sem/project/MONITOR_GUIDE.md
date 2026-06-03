# Training Monitor Guide

## ✅ What's Been Fixed

1. **File Logging**: Training now writes logs to `logs/training_YYYYMMDD_HHMMSS.log`
2. **MPS Support**: Training will now use Apple Silicon GPU (Metal Performance Shaders) automatically
3. **Enhanced Monitor**: New real-time dashboard showing progress, metrics, and checkpoints

## 🚀 How to Use

### Start Training (with GPU acceleration)
```bash
cd "/Users/I768770/Documents/Mtech/mtech 4th sem/project"
source .venv/bin/activate
python train.py --config configs/config.yaml
```

### Monitor Training (Enhanced Dashboard)
```bash
# In a separate terminal
cd "/Users/I768770/Documents/Mtech/mtech 4th sem/project"
python monitor_enhanced.py

# Or with custom settings
python monitor_enhanced.py --refresh 5  # Update every 5 seconds
```

### Quick Status Check
```bash
./check_progress.sh
```

## 📊 What You'll See

The enhanced monitor shows:
- **Real-time progress**: Current epoch, batch, loss, accuracy
- **Visual progress bar**: Easy-to-read training progress
- **Time estimates**: Elapsed and remaining time
- **Recent logs**: Last few log entries
- **Checkpoints**: All saved models with timestamps
- **System stats**: CPU/memory usage

## ⚡ Performance Improvement

**Before (CPU only):**
- ~40 seconds per batch
- ~2 hours per epoch
- ~10 hours total (5 epochs)

**After (MPS GPU):**
- Should be **3-10x faster**
- Estimated: ~10-15 minutes per epoch
- Estimated: ~1 hour total

## 🔄 To Apply GPU Acceleration

**Your current training is running on CPU.** To use the GPU:

1. Stop current training (Ctrl+C in training terminal)
2. Restart: `python train.py --config configs/config.yaml`
3. It will automatically detect and use MPS (Apple GPU)

## 📁 File Locations

- **Logs**: `logs/training_*.log`
- **Checkpoints**: `checkpoints/*.pth`
- **Config**: `configs/config.yaml`
- **Enhanced Monitor**: `monitor_enhanced.py`
- **Quick Check**: `check_progress.sh`

## 💡 Tips

- The enhanced monitor updates every 10 seconds (configurable)
- Checkpoints are saved after each epoch
- Training metrics are logged to both console and file
- Monitor can run while training is in progress
