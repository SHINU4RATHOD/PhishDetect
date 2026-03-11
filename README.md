<p align="center">
  <img src="https://img.shields.io/badge/Model-MiniLM--L12--H384-blue?style=for-the-badge&logo=microsoft" alt="Model">
  <img src="https://img.shields.io/badge/Task-Phishing%20URL%20Detection-red?style=for-the-badge&logo=shield" alt="Task">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/PEFT-LoRA-green?style=for-the-badge" alt="LoRA">
  <img src="https://img.shields.io/badge/Export-ONNX-005CED?style=for-the-badge&logo=onnx" alt="ONNX">
</p>

# 🛡️ PhishGuard-MiniLM: Production-Grade Phishing URL Detection

> **A high-performance, parameter-efficient phishing URL detection system built on Microsoft's MiniLM-L12-H384 transformer, fine-tuned with LoRA on 26.5 million URLs for real-time cybersecurity applications.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [KPI Targets](#-kpi-targets--strict-production-standards)
- [Hyperparameter Configuration](#%EF%B8%8F-hyperparameter-configuration)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Workflow](#-pipeline-workflow)
- [Project Structure](#-project-structure)
- [Output Artifacts](#-output-artifacts)
- [Technical Deep Dive](#-technical-deep-dive)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

Phishing attacks remain one of the most prevalent cybersecurity threats globally, responsible for **over 80% of security incidents**. Traditional approaches rely on static blacklists and hand-crafted feature engineering — both of which fail against rapidly evolving, adversarially-crafted phishing URLs.

**PhishGuard-MiniLM** takes a fundamentally different approach: it treats the URL as a **raw character sequence** and leverages the contextual understanding of a pre-trained transformer to distinguish legitimate URLs from phishing attempts — **without any manual feature extraction**.

### Why MiniLM?

| Property | Value | Benefit |
|----------|-------|---------|
| **Parameters** | ~33.5M (base) | Lightweight — ideal for edge deployment |
| **Hidden Size** | 384 | 2× smaller than BERT-base (768) |
| **Layers** | 12 | Full depth for rich representations |
| **Inference Speed** | ~3ms/URL (GPU) | Real-time compatible |
| **Model Size** | < 40 MB (quantized) | Deployable on resource-constrained devices |

MiniLM achieves **knowledge distillation** from larger models while retaining their representational power — the perfect balance of **accuracy and efficiency** for production cybersecurity systems.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   PhishGuard-MiniLM Pipeline                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Raw URL ──► Tokenizer ──► MiniLM-L12-H384 ──► Classifier ──► │
│             (AutoTokenizer)   (+ LoRA Adapters)    [384→192→64→2]│
│                                                                  │
│   "http://evil-bank.com/login" ──────────────────► [0.02, 0.98] │
│                                                     Benign Phish │
├──────────────────────────────────────────────────────────────────┤
│  Loss: Focal Loss (γ=2.5, α=[0.28, 0.72])                       │
│  Optimizer: AdamW (lr=2e-5, cosine warmup)                       │
│  Regularization: LoRA dropout (0.15) + Weight Decay (0.02)       │
│  Export: PyTorch → ONNX → INT8 Quantized                        │
└──────────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | Details |
|-----------|---------|
| **Base Encoder** | `microsoft/MiniLM-L12-H384-uncased` — 12 layers, 384 hidden dim |
| **Adaptation** | LoRA (rank=32, α=64) on `query`, `key`, `value`, `dense`, `output.dense` |
| **Classifier Head** | `384 → 192 → 64 → 2` with LayerNorm + GELU + Dropout(0.3) |
| **Loss Function** | Focal Loss with class-aware weighting for imbalanced data |
| **Tokenization** | Byte-Pair Encoding via AutoTokenizer (max_len=192) |

### Parameter Efficiency (LoRA)

Instead of fine-tuning all 33.5M parameters, LoRA injects small low-rank matrices into the attention layers, training only **~2.9M parameters (7.98%)** while keeping the base model frozen:

```
Total Parameters:      36,509,828
Trainable (LoRA):       2,914,306  (7.98%)
Frozen (Base Model):   33,595,522  (92.02%)
```

This approach provides:
- ✅ **Faster training** — fewer gradients to compute
- ✅ **Lower memory** — frozen weights don't store optimizer states
- ✅ **No catastrophic forgetting** — base model knowledge preserved
- ✅ **Efficient deployment** — LoRA weights merge into base model post-training

---

## 📊 Dataset

The model is trained on a **large-scale, real-world URL dataset** sourced from URLScan.io:

| Split | Legitimate | Phishing | Total | Ratio |
|-------|-----------|----------|-------|-------|
| **Train** | 18,961,552 | 7,528,090 | **26,489,642** | 2.52:1 |
| **Validation** | 3,160,259 | 1,254,681 | **4,414,940** | 2.52:1 |
| **Test** | 3,160,259 | 1,254,682 | **4,414,941** | 2.52:1 |
| **Total** | **25,282,070** | **10,037,453** | **35,319,523** | 2.52:1 |

### Class Imbalance Handling

The 2.52:1 imbalance is addressed through a **multi-layered strategy**:

1. **Weighted Random Sampling** — Oversamples minority class during training (weight: 3.79× for phishing)
2. **Focal Loss (γ=2.5)** — Down-weights easy examples, focuses on hard misclassifications
3. **Class-Aware Alpha (α=[0.28, 0.72])** — Gives 2.57× more weight to phishing samples in loss computation
4. **Threshold Optimization** — Post-training threshold search to satisfy both FPR and FNR constraints simultaneously

### Expected Data Format

CSV files with two columns:

```csv
input,label
https://www.google.com/search?q=example,0
http://evil-bank-login.phish.xyz/verify,1
```

| Column | Type | Description |
|--------|------|-------------|
| `input` | `str` | Raw URL string |
| `label` | `int` | `0` = Legitimate (Benign), `1` = Phishing (Malicious) |

---

## 🎯 KPI Targets — Strict Production Standards

These KPIs are designed for **real-world deployment** where both false positives (blocking legitimate sites) and false negatives (missing phishing sites) carry severe consequences:

| KPI | Target | Priority | Rationale |
|-----|--------|----------|-----------|
| **Accuracy** | ≥ 98% | High | Overall correctness |
| **Precision** | ≥ 95% | High | Minimize false alarms (user trust) |
| **Recall** | ≥ 95% | High | Catch phishing attacks (security) |
| **FPR** | ≤ 1% | 🔴 Critical | Max 1 in 100 legitimate URLs wrongly flagged |
| **FNR** | ≤ 10% | High | Max 1 in 10 phishing URLs missed |
| **Model Size** | < 40 MB | Medium | Edge/mobile deployment feasibility |

> **⚠️ Note:** Achieving FPR ≤ 1% **and** FNR ≤ 10% simultaneously is extremely challenging — it requires the model to be both highly conservative (low FP) and highly sensitive (low FN). The strict threshold optimization algorithm systematically searches for operating points that satisfy both constraints.

---

## ⚙️ Hyperparameter Configuration

Every hyperparameter has been **research-optimized** based on transformer fine-tuning literature, LoRA best practices, and the specific characteristics of this 26.5M-sample dataset:

### Training

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 128 | Stable gradients for large-scale training |
| Gradient Accumulation | 4 | Effective batch = 512 |
| Epochs | 8 | Sufficient for large dataset convergence |
| Early Stopping Patience | 5 | Prevents premature termination |
| Gradient Clip Norm | 0.5 | Tight clipping for training stability |
| Mixed Precision (AMP) | Enabled | 2× speedup, lower memory |

### Optimizer & Schedule

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for transformer fine-tuning |
| Learning Rate | 2e-5 | Optimal for pre-trained transformer adaptation |
| Weight Decay | 0.02 | L2 regularization to prevent overfitting |
| Warmup | 6% of steps | Prevents early instability |
| Schedule | Cosine annealing | Smooth LR decay to 0.1% of peak |

### LoRA

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 32 | Sufficient expressiveness for binary classification |
| Alpha (α) | 64 | α = 2r (standard scaling) |
| Dropout | 0.15 | Regularization on adapter weights |
| Target Modules | query, key, value, dense, output.dense | Full attention + projections |

### Focal Loss

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gamma (γ) | 2.5 | Strong focus on hard/misclassified examples |
| Alpha (α) | [0.28, 0.72] | Inverse class frequency weighting |
| Label Smoothing | 0.05 | Better probability calibration |

---

## 📦 Installation

### Prerequisites

- **Python** ≥ 3.8
- **CUDA** ≥ 11.7 (for GPU training)
- **RAM** ≥ 32 GB (for loading 26.5M samples)
- **GPU VRAM** ≥ 8 GB (for batch_size=128)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd 05_MiniLM

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft
pip install scikit-learn pandas numpy tqdm matplotlib seaborn
pip install onnx onnxruntime torchinfo

# Optional: ONNX quantization support
pip install onnxruntime-gpu
```

### Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Deep learning framework |
| `transformers` | ≥ 4.35 | MiniLM model & tokenizer |
| `peft` | ≥ 0.7 | LoRA adapter implementation |
| `scikit-learn` | ≥ 1.3 | Evaluation metrics |
| `pandas` | ≥ 2.0 | Data loading & processing |
| `onnx` | ≥ 1.14 | Model export |
| `onnxruntime` | ≥ 1.16 | ONNX inference & quantization |
| `matplotlib` / `seaborn` | Latest | Visualization |

---

## 🚀 Usage

### Quick Start

```bash
# Train from scratch
python MiniLM_1.py --mode train

# Resume training from latest checkpoint
python MiniLM_1.py --mode train --interactive

# Inference only (load checkpoint, evaluate test set)
python MiniLM_1.py --mode inference

# Auto-detect mode (train if no checkpoint, infer if checkpoint exists)
python MiniLM_1.py
```

### CLI Options

```bash
python MiniLM_1.py --help
```

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--mode` | `train`, `inference`, `auto` | `auto` | Execution mode |
| `--interactive` | flag | `False` | Enable checkpoint resume prompt |
| `--epochs` | int | `4` | Override number of epochs |
| `--batch_size` | int | `128` | Override batch size |
| `--lr` | float | `2e-5` | Override learning rate |
| `--verbose` | flag | `False` | Enable verbose logging |

### CLI Override Examples

```bash
# Quick experiment with fewer epochs
python MiniLM_1.py --mode train --epochs 2 --batch_size 64

# Fine-tune learning rate
python MiniLM_1.py --mode train --lr 1e-5

# Small-scale test run
python MiniLM_1.py --mode train --epochs 1 --batch_size 16
```

---

## 🔄 Pipeline Workflow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│  Load Data  │───►│  Build Model │───►│  Train Loop  │───►│  Evaluate  │
│  (CSV→DF)   │    │ (MiniLM+LoRA)│    │ (+ Checkpts) │    │  Test Set  │
└─────────────┘    └──────────────┘    └──────────────┘    └────────────┘
                                              │                    │
                                              ▼                    ▼
                                       ┌──────────────┐    ┌────────────┐
                                       │  Save Best   │    │   Export   │
                                       │    Model     │    │   Reports  │
                                       └──────┬───────┘    └────────────┘
                                              │
                                              ▼
                                ┌─────────────────────────┐
                                │  Merge LoRA → Export     │
                                │  PyTorch → ONNX → INT8  │
                                └─────────────────────────┘
```

### Stage-by-Stage Breakdown

| Stage | Description | Key Operations |
|-------|-------------|----------------|
| **1. Data Loading** | Load train/val/test CSVs, encode labels, create weighted sampler | Label mapping, class balancing |
| **2. Model Building** | Initialize MiniLM encoder, attach LoRA adapters, build classifier | AutoModel + LoRA injection |
| **3. Checkpoint Resume** | Scan for existing checkpoints, restore full training state | Model, optimizer, scheduler, scaler, history |
| **4. Training Loop** | Epoch-level training with gradient accumulation and AMP | Forward/backward pass, gradient clipping |
| **5. Validation** | Per-epoch evaluation with strict threshold optimization | FPR/FNR-aware threshold search |
| **6. Best Model Saving** | Save LoRA adapter + full model + merge LoRA into base | Production model creation |
| **7. ONNX Export** | Convert merged model to ONNX, apply INT8 dynamic quantization | Size reduction for deployment |
| **8. Test Evaluation** | Final evaluation on held-out test set with production model | Confusion matrix, ROC, PR curves |
| **9. Artifact Generation** | Save metrics, plots, predictions, and deployment metadata | Full reproducibility |

---

## 📁 Project Structure

```
05_MiniLM/
├── MiniLM_1.py                          # Main training & inference script (1,874 lines)
├── README.md                            # This documentation
│
└── saved_models/
    └── MiniLM_data10/                   # Model outputs root
        ├── checkpoints/                 # Training checkpoints
        │   ├── checkpoint_epoch_001.pt
        │   ├── checkpoint_epoch_002.pt
        │   └── ...
        │
        ├── best_model_epoch_XXX/        # Best model artifacts
        │   ├── lora_adapter/            # LoRA adapter weights (for HuggingFace)
        │   │   ├── adapter_config.json
        │   │   └── adapter_model.bin
        │   ├── model_full.pt            # Full model (with LoRA)
        │   ├── model_merged_full.pt     # Merged model (LoRA → base)
        │   ├── model_merged_state_dict.pt
        │   ├── model.onnx              # ONNX export
        │   ├── model_quantized.onnx    # INT8 quantized ONNX
        │   ├── model_summery.txt       # Architecture summary
        │   ├── deployment_metadata.json # Hyperparams & metrics
        │   ├── tokenizer_config.json   # Tokenizer configuration
        │   ├── vocab.txt               # Vocabulary file
        │   └── training_history.csv    # Loss & accuracy curves
        │
        ├── final_test_evaluation/       # Test set results
        │   ├── test_predictions.csv     # Per-URL predictions
        │   ├── test_metrics.csv         # Summary metrics
        │   ├── confusion_matrix.png     # Confusion matrix plot
        │   ├── roc_curve.png           # ROC curve
        │   └── pr_curve.png            # Precision-Recall curve
        │
        └── final_results.json           # Overall experiment results
```

---

## 📤 Output Artifacts

### Model Exports

| File | Format | Size (Approx.) | Use Case |
|------|--------|----------------|----------|
| `model_full.pt` | PyTorch | ~140 MB | Full model with LoRA (for continued training) |
| `model_merged_full.pt` | PyTorch | ~130 MB | Merged model (LoRA integrated — for production) |
| `model.onnx` | ONNX | ~130 MB | Cross-platform inference |
| `model_quantized.onnx` | ONNX INT8 | **< 40 MB** | Edge/mobile deployment |

### Evaluation Outputs

| File | Description |
|------|-------------|
| `test_predictions.csv` | Per-URL predictions with probabilities and correctness |
| `test_metrics.csv` | Accuracy, Precision, Recall, F1, AUC, FPR, FNR |
| `confusion_matrix.png` | Visual confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC score |
| `pr_curve.png` | Precision-Recall curve |
| `training_history.csv` | Epoch-wise loss and accuracy tracking |
| `deployment_metadata.json` | Full experiment configuration for reproducibility |

---

## 🔬 Technical Deep Dive

### Focal Loss for Class Imbalance

Standard cross-entropy loss gives equal weight to all samples, causing the model to be dominated by the majority class (legitimate URLs). **Focal Loss** solves this by down-weighting easy examples:

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

Where:
- **γ = 2.5** — Modulates the focusing effect. Higher γ means more focus on hard examples
- **α = [0.28, 0.72]** — Class weights (benign, phishing). Phishing gets 2.57× more weight
- **p_t** — Model's estimated probability for the correct class

### Strict Threshold Optimization

Instead of the default 0.5 threshold, the system performs a **comprehensive search** across 10,000 candidate thresholds to find the operating point that:

1. **First** satisfies FPR ≤ 1% **AND** FNR ≤ 10% (hard constraints)
2. **Then** maximizes F1-score among all valid thresholds
3. **Falls back** to minimum KPI violation if no perfect threshold exists

This is critical because different thresholds create fundamentally different tradeoffs:

```
Lower threshold (e.g., 0.3)  → More phishing detected → Higher recall → Higher FPR
Higher threshold (e.g., 0.7) → Fewer false alarms   → Higher precision → Higher FNR
                                    Optimal threshold lies in between ↑
```

### Checkpoint Resumability

Training on 26.5M samples requires significant time. The system provides **full checkpoint resumability**:

- Model weights + LoRA adapters
- Optimizer state (Adam momentum & velocity)
- Learning rate scheduler state
- Gradient scaler state (for AMP)
- Complete training history (loss, accuracy, KPI scores)
- Best KPI score and patience counter

This allows training to be interrupted and resumed **at any point** with no loss of training progress.

### ONNX Export & Quantization

For production deployment, the trained model goes through a multi-stage export:

```
PyTorch Model (LoRA) → Merge LoRA → PyTorch Model (standalone)
                                          ↓
                                    ONNX Export (FP32)
                                          ↓
                                  Dynamic Quantization (INT8)
                                          ↓
                                  < 40 MB production model
```

INT8 quantization reduces model size by ~4× with minimal accuracy loss, making it suitable for:
- 🌐 Web browser extensions
- 📱 Mobile applications
- 🖥️ Edge servers with limited resources
- ☁️ Cost-efficient cloud inference

---

## ❓ Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch size too large for GPU | Reduce `BATCH_SIZE` or increase `GRAD_ACCUM_STEPS` |
| `Merged model not found` | Best model saved at different epoch | System auto-scans for `best_model_epoch_*` dirs |
| `Weights only load failed` | PyTorch 2.6+ default change | Already handled with `weights_only=False` |
| `Training already completed` | Checkpoint epoch > NUM_EPOCHS | Increase `NUM_EPOCHS` or delete checkpoints |
| `FNR=1.0, TP=0` | Model not learning minority class | Verify weighted sampling & focal loss alpha |
| `NaN/Inf in loss` | Numerical instability | Reduce `LR`, increase `GRAD_CLIP_NORM` |

### Resetting Training

```bash
# Delete all checkpoints and start fresh
rm -rf saved_models/MiniLM_data10/

# Then retrain
python MiniLM_1.py --mode train
```

### Checking GPU Status

```bash
nvidia-smi  # Monitor GPU memory and utilization
watch -n 1 nvidia-smi  # Continuous monitoring
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{phishguard_minilm_2026,
  title     = {PhishGuard-MiniLM: Production-Grade Phishing URL Detection},
  author    = {Shinu Rathod},
  institution = {IIT Ropar},
  year      = {2026},
  note      = {MiniLM-L12-H384 + LoRA fine-tuning on 26.5M URLs}
}
```

### References

1. Wang, W., et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS.
2. Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
3. Lin, T.-Y., et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.
4. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization (AdamW)*. ICLR.

---

## 📄 License

This project is developed as part of academic research at **IIT Ropar**. Please contact the author for licensing inquiries.

---

<p align="center">
  <b>Built with ❤️ for a safer internet</b>
  <br>
  <sub>IIT Ropar — Cybersecurity Research</sub>
</p>
