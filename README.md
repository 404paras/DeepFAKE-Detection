## Deepfake Detection – M.Tech Project

End‑to‑end deepfake detection project developed as part of an M.Tech 2nd semester course.  
It contains:
- **Training pipeline** in `deepfake-model.py` (Colab‑style script for dataset preparation, model training and evaluation)
- **Flask web application** in `WebApp/` for interactive video‑based deepfake detection

---

## 1. Project Overview

- **Goal**: Detect whether a given face video is **REAL** or **FAKE (deepfake)**.
- **Input**: Short videos (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) containing faces.
- **Output**: Binary decision (**REAL / FAKE**) with evaluation metrics (accuracy, confusion matrix, precision, recall, F1, ROC‑AUC) from the training pipeline and a simple prediction from the web app.
- **Core idea**:
  - Sample frames from each video.
  - Use a **CNN (ResNet/ResNeXt)** as frame‑level feature extractor.
  - Feed frame features into a **temporal RNN (GRU/LSTM)**.
  - Apply **attention** and a final classifier to predict REAL vs FAKE.

---

## 2. Repository Structure

High‑level layout around this project:

```text
Mtech 2nd Sem/
├── README.md                     # This file (project‑level docs for GitHub)
├── Code.pdf                      # Export of the training notebook (for reference)
├── deepfake-model.py             # Training / evaluation script (Colab‑style)
├── WebApp/                       # Flask web application
│   ├── app.py                    # Main Flask app (HTTP endpoints, model loading)
│   ├── model.py                  # CNN + GRU + attention model for web app
│   ├── utils.py                  # Preprocessing & inference utilities
│   ├── model.pt                  # Deployed model weights used by `app.py`
│   ├── new_combined_checkpoint.pt# (Large) research checkpoint from training
│   ├── requirements.txt          # Web app Python dependencies
│   ├── README.md                 # Web‑app‑specific documentation
│   ├── templates/                # HTML templates (index, about, test‑dataset)
│   ├── static/                   # CSS, sample dataset videos, thumbnails, etc.
│   ├── uploads/                  # Temporary uploads at runtime
│   └── venv/                     # Local virtual environment (do NOT commit)
└── (other thesis / report files)
```

For GitHub, you will usually keep:
- Source code (`*.py`), configuration, HTML/CSS/JS templates
- Lightweight sample media files (optional, if not too large)
- Documentation (`README.md`, reports if size is acceptable)

and you should **exclude**:
- `WebApp/venv/`
- `WebApp/__pycache__/`
- `WebApp/uploads/`
- Large model checkpoints (`*.pt`) and large videos, unless pushed via Git LFS

You can enforce this with a `.gitignore` (example shown later).

---

## 3. Training Pipeline (`deepfake-model.py`)

`deepfake-model.py` is a notebook‑exported script originally designed to run in **Google Colab with GPU**.

### 3.1. Datasets & Layout (as used in the script)

The script expects pre‑processed datasets organised on Google Drive, e.g.:

- `/content/drive/My Drive/Mtech_Datasets/Celeb_fake_face/*.mp4`
- `/content/drive/My Drive/Mtech_Datasets/Celeb_real_face/*.mp4`
- `/content/drive/My Drive/Mtech_Datasets/DFDC_FAKE_Face/*.mp4`
- `/content/drive/My Drive/Mtech_Datasets/DFDC_REAL_Face/*.mp4`
- `/content/drive/My Drive/Mtech_Datasets/FF_Face/*.mp4`

with labels loaded from a CSV such as:
- `/content/drive/My Drive/Model Creation/labels/Gobal_metadata.csv`  
  (columns: `file`, `label` where `label ∈ {FAKE, REAL}`).

These paths should be adapted for your own environment if you are not using Colab.

### 3.2. Data Pipeline

- **Video validation**:
  - `validate_video(path, train_transforms)` reads frames with OpenCV and applies transforms to check that the video is not corrupted.
- **Frame extraction**:
  - `frame_extract(path)` is a generator yielding frames from `cv2.VideoCapture`.
- **Transforms**:
  - Uses `torchvision.transforms` to:
    - Resize frames to `112×112`
    - Apply augmentations (random flip, rotation, colour jitter) for training
    - Normalize with ImageNet mean/std.
- **Custom dataset**:
  - `video_dataset(Dataset)`:
    - Receives a list of video paths and a pandas `labels` DataFrame.
    - For each video, samples a fixed **sequence length** (e.g. 10 or 60 frames).
    - Looks up the label (`FAKE` → 0, `REAL` → 1).
    - Returns a tensor of shape `(sequence_length, 3, H, W)` and the label.
  - Train/validation split is created by shuffling and slicing the video list (80/20).

### 3.3. Model Architecture (Training Script)

The training script defines a hybrid model `DeepfakeDetector` (slightly tuned from the web‑app version):

- **Backbone CNN**:
  - `ResNet50` pre‑trained on ImageNet.
  - Final classification layers removed; only convolutional feature maps are used.
  - Global average pooling to get a 2048‑D feature vector **per frame**.
- **Temporal Module**:
  - **Bidirectional GRU**:
    - `input_size = 2048`
    - `hidden_size = 1024`
    - `num_layers = 2`
    - `bidirectional = True` (output dim 2048)
    - With dropout between layers.
- **Attention**:
  - `nn.MultiheadAttention` with:
    - `embed_dim = 2048`
    - `num_heads = 8`
  - Residual connection: GRU output + attention output.
- **Classifier**:
  - `LayerNorm(2048)`
  - Dropouts
  - `Linear(2048 → 512) + GELU`
  - `Linear(512 → 2)` for binary classification.

The web‑app `model.py` defines a closely related architecture (ResNet50 + GRU + attention) adapted for deployment.

### 3.4. Training Loop

- **Loss**: `nn.CrossEntropyLoss` (optionally with label smoothing in some variants).
- **Optimiser**: `AdamW` (or `Adam` in earlier part of the script) with weight decay.
- **Scheduler**: `ReduceLROnPlateau` on validation accuracy.
- **Metrics & logging**:
  - `AverageMeter` tracks running loss and accuracy.
  - `calculate_accuracy` computes top‑1 accuracy.
  - Confusion matrix, precision, recall, F1 and ROC‑AUC are computed with `sklearn`.
- **Checkpoints**:
  - Model weights are periodically saved to Google Drive (e.g. `new_combined_checkpoint.pt`).

### 3.5. How to Run Training (Recommended: Google Colab)

1. Upload `model_and_train_csv.py` or the original `.ipynb` to Google Colab.
2. Set runtime to **GPU**:
   - Runtime → Change runtime type → Hardware accelerator: **GPU**.
3. Mount Google Drive:
   - The script already calls `drive.mount('/content/drive')`.
4. Adjust dataset and label CSV paths to match your Drive structure.
5. Run cells in order:
   - Data validation and statistics
   - Dataset and dataloader creation
   - Model definition
   - Training loop
   - Evaluation and metric plots.

---

## 4. Web Application (`WebApp/`)

The `WebApp/` directory contains a Flask application that loads a trained model and exposes it through a browser UI.

### 4.1. Main Components

- `app.py`
  - Creates a Flask app.
  - Configures upload folder, allowed extensions, and max upload size.
  - Loads the `DeepfakeDetector` model and weights from `model.pt`.
  - Defines endpoints:
    - `GET /` – main upload UI.
    - `POST /upload` – upload a user video, run prediction, return JSON (`REAL` / `FAKE`).
    - `GET /test-dataset` – page to test pre‑defined sample videos.
    - `POST /test-sample/<id>` – run prediction on a sample dataset video.
    - `GET /get-sample-videos` – JSON listing of sample videos.
    - `GET /model-info` – returns basic information about the loaded model.
    - `GET /about` – project information page.
- `model.py`
  - Deployment architecture: ResNet50 backbone + GRU + Multi‑head Attention + classifier.
- `utils.py`
  - `preprocess_video(path, sequence_length, im_size)`:
    - Reads frames with OpenCV, samples a fixed number of frames.
    - Resizes to `im_size × im_size`, converts to tensor and normalizes.
    - Returns a video tensor of shape `(1, sequence_length, 3, H, W)`.
  - Contains an alternative `DeepfakeDetector` definition and helper functions for CLI use.

### 4.2. Web App Dependencies & Setup

From inside `WebApp/`:

```bash
cd "WebApp"
pip install -r requirements.txt
```

(Optionally create and activate your own virtual environment instead of using the committed `venv/`.)

### 4.3. Running the Web App

From inside `WebApp/`:

```bash
python app.py
```

Then open a browser at:
- `http://localhost:5001/` – main upload page
- `http://localhost:5001/test-dataset` – test sample dataset videos

The app will:
- Load `model.pt` into memory (CPU or GPU depending on availability).
- Accept video uploads.
- Run the model and respond with a JSON prediction and the filename.

---

## 5. How to Use This Repository on GitHub

### 5.1. Suggested `.gitignore`

Create a `.gitignore` at the root of the repo (if you don’t already have one):

```gitignore
# Python caches
__pycache__/
*.py[cod]

# Virtual environments
venv/
WebApp/venv/

# Large artifacts (adjust as needed)
*.pt
WebApp/uploads/

# OS / editor files
.DS_Store
.vscode/
.idea/
```

If you want to keep a **single small model** (e.g. `WebApp/model.pt`) in the repo, you can remove `*.pt` from `.gitignore` and just commit that one file (or use Git LFS for all large checkpoints).

### 5.2. Minimal Files to Commit for a Clean Public Repo

For a typical GitHub portfolio / thesis repo, you can keep:
- `README.md` (this file)
- `model_and_train_csv.py`
- `WebApp/app.py`, `WebApp/model.py`, `WebApp/utils.py`
- `WebApp/templates/`, `WebApp/static/css/`
- `WebApp/requirements.txt`
- A **few small sample videos/images** if needed for demonstration.

And either:
- Exclude all `.pt` weights, or
- Keep only one deployment model (e.g., `WebApp/model.pt`) and ignore others / use Git LFS.

---

## 6. Reproducing and Extending the Work

- **Change backbone**: Replace ResNet50/ResNeXt with other CNNs (EfficientNet, ViT‑backed encoder, etc.).
- **Change temporal modelling**: Swap GRU/LSTM for temporal convolution (TCN), Transformer encoders, or 3D CNNs.
- **Improve sampling**: Instead of uniform sampling, detect and crop faces, then sample frames around high‑motion regions.
- **Add calibration**: Output calibrated probabilities or uncertainty estimates for more reliable deployment.

---

## 7. Credits

- Developed as an **M.Tech 2nd Semester project** on **Deepfake Detection using Hybrid CNN–RNN Architectures with Attention**.
- Training, evaluation and analysis scripts are in `deepfake-model.py` and `Code.pdf`.
- Web deployment is implemented in the `WebApp/` folder.


