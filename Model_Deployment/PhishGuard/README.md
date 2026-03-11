# 🛡️ PhishGuard — On-Device Phishing URL Detection for Android

> **Real-time, privacy-first phishing URL classification powered by a fine-tuned MiniLM transformer running entirely on-device via ONNX Runtime.**

[![Android](https://img.shields.io/badge/Android-26%2B-brightgreen?logo=android&logoColor=white)](https://developer.android.com/about/versions)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.2-blueviolet?logo=kotlin&logoColor=white)](https://kotlinlang.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.24.2-blue?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Compose](https://img.shields.io/badge/Jetpack%20Compose-Material%203-4285F4?logo=jetpackcompose&logoColor=white)](https://developer.android.com/jetpack/compose)
[![License](https://img.shields.io/badge/License-Research-orange)]()

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Model Details](#-model-details)
- [Inference Pipeline](#-inference-pipeline)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Setup & Build](#%EF%B8%8F-setup--build)
- [Generating Golden Vectors](#-generating-golden-vectors)
- [CSV Batch Evaluation](#-csv-batch-evaluation)
- [Benchmarking](#-benchmarking)
- [Security Considerations](#-security-considerations)
- [Tech Stack](#-tech-stack)
- [Research Context](#-research-context)

---

## 🌐 Overview

**PhishGuard** is an Android application that classifies URLs as _Safe_ or _Phishing_ using a transformer-based language model running **entirely on-device** — no internet connection required, no data leaves the phone.

The app deploys a **MiniLM-L12-H384** transformer, fine-tuned with **LoRA** (Low-Rank Adaptation) on **26.5 million phishing/benign URL samples**, quantized to **8-bit integers** (QUInt8) for mobile inference. The full pipeline — URL normalization → BERT WordPiece tokenization → ONNX inference → softmax → threshold — executes in **under 100ms** on modern Android devices.

### Why On-Device?

| Approach | Latency | Privacy | Offline | Cost |
|:---|:---:|:---:|:---:|:---:|
| Cloud API | 200–500ms | ❌ URLs sent to server | ❌ | Per-query |
| **PhishGuard (On-Device)** | **<100ms** | **✅ Zero data exfiltration** | **✅ Full offline** | **Free** |

---

## ✨ Key Features

### 🔍 Real-Time URL Scanning
Paste or type any URL → get an instant Safe/Phishing verdict with probability score, latency breakdown, and security warnings (punycode detection, homograph attack flagging).

### 📊 CSV Batch Evaluation
Upload a labeled CSV file → run batch inference on thousands of URLs → view full binary classification metrics:

| Metric | Description |
|:---|:---|
| **Accuracy** | Overall correct classification rate |
| **Precision** | Phishing predictions that are actually phishing |
| **Recall** | Actual phishing URLs correctly detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve (threshold-independent) |
| **FNR** | False Negative Rate — missed phishing (critical for security) |
| **FPR** | False Positive Rate — false alarms |

Plus a **visual confusion matrix** showing TN (L→L), FP (L→M), FN (M→L), TP (M→M).

### ⚡ Performance Benchmark
Built-in latency profiler: 5 warmup + 20 timed runs → reports p50, p90, mean, min, max latency with per-stage breakdown (tokenization vs. inference).

### 🧪 Golden Vector Parity Test
Two-stage validation ensuring Android inference matches the Python training pipeline:
- **Stage 1**: Exact token-ID match (zero tolerance)
- **Stage 2**: Probability match (ε = 0.001)

### 📜 Scan History
Persistent history of the last 50 scans with verdict, probability, and latency.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PhishGuard App                          │
├─────────────┬─────────────┬──────────────┬─────────────────────┤
│  ScanScreen │HistoryScreen│BenchmarkScreen│  EvaluateScreen    │
│  (Tab 1)    │  (Tab 2)    │   (Tab 3)     │    (Tab 4)        │
├─────────────┴─────────────┴──────────────┴─────────────────────┤
│                     MainViewModel                              │
│         (State management, coroutine orchestration)            │
├────────────────────────────────────────────────────────────────┤
│                    Domain Layer                                 │
│     ScanUrlUseCase │ BenchmarkRunner │ CsvEvaluator            │
├────────────────────────────────────────────────────────────────┤
│                     Data Layer                                  │
│  PhishingUrlDetector → BertWordPieceTokenizer → OnnxModelLoader│
├────────────────────────────────────────────────────────────────┤
│                     Core Layer                                  │
│       UrlNormalizer │ PhishGuardConfig │ SecureLogger           │
├────────────────────────────────────────────────────────────────┤
│                   ONNX Runtime (CPU / NNAPI)                   │
│             model_quant_8bit.onnx (32.5 MB)                    │
└────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Package | Purpose |
|:---|:---|:---|
| **UI** | `com.phishguard.app.ui` | Jetpack Compose screens, MainViewModel, theme |
| **Domain** | `com.phishguard.app.domain` | Use cases, result data classes |
| **Data** | `com.phishguard.app.data` | Tokenizer, ONNX loader, detector, CSV evaluator |
| **Core** | `com.phishguard.app.core` | Config, URL normalizer, secure logging |
| **Benchmark** | `com.phishguard.app.benchmark` | Latency profiler |
| **Test** | `com.phishguard.app.test` | Golden vector parity tests |

---

## 🧠 Model Details

### Base Architecture

| Property | Value |
|:---|:---|
| **Base Model** | `microsoft/MiniLM-L12-H384-uncased` |
| **Architecture** | 12-layer Transformer, hidden size 384, 12 attention heads |
| **Fine-Tuning** | LoRA (Low-Rank Adaptation), rank r |
| **Classifier** | Linear(384) → Linear(192) → Linear(64) → Linear(2) |
| **Quantization** | QUInt8 (INT8 weights, FP32 compute) |
| **Model Size** | 32.5 MB (ONNX) |

### Training Configuration

| Parameter | Value |
|:---|:---|
| **Training Samples** | ~26.5 million URLs |
| **Epochs** | 20 (best @ epoch 18) |
| **Batch Size** | 128 (effective 512 with grad. accum.) |
| **Learning Rate** | 2e-5 with cosine warmup (6%) |
| **MAX_LEN** | 192 tokens |
| **Optimizer** | AdamW (weight decay 0.02) |
| **Dropout** | 0.3 |
| **Gradient Clipping** | 0.5 |

### Inference Contract

| Parameter | Value |
|:---|:---|
| **Inputs** | `input_ids` [1, 192] int64, `attention_mask` [1, 192] int64 |
| **Output** | `logits` [1, 2] float32 |
| **Threshold** | 0.59 (Youden's J optimized) |
| **Label Map** | 0 = Benign (SAFE), 1 = Phishing (PHISHING) |

---

## 🔄 Inference Pipeline

Every URL goes through a 6-stage pipeline, faithfully reproducing the Python training-time preprocessing:

```
              ┌──────────────┐
              │   Raw URL    │
              └──────┬───────┘
                     ▼
        ┌────────────────────────┐
Step 1  │    URL Normalization   │  Unicode NFKC, control char stripping,
        │    (UrlNormalizer)     │  punycode decoding, confusable detection
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
Step 2  │  BERT BasicTokenizer   │  Clean text → Chinese char spacing →
        │  (9-step pipeline)     │  lowercase → strip accents → split punct
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
Step 3  │  WordPiece Tokenizer   │  Greedy longest-match with ## sub-words
        │  (vocab: 30,522)       │  Result: [CLS] tokens [SEP] [PAD]...
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
Step 4  │   ONNX Inference       │  ORT session → CPU or NNAPI delegate
        │   (QUInt8, 32.5 MB)    │  Output: logits [1, 2]
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
Step 5  │   Stable Softmax       │  Subtract max → exp → normalize
        │                        │  → probabilities [p_benign, p_phishing]
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
Step 6  │   Threshold Decision   │  p_phishing ≥ 0.59 → PHISHING
        │   (Youden's J = 0.59) │  p_phishing <  0.59 → SAFE
        └────────────────────────┘
```

### Tokenization Faithfulness

The BERT WordPiece tokenizer is implemented from scratch in Kotlin, replicating the exact 9-step HuggingFace `BertTokenizer` pipeline:

1. Clean text (remove control chars, collapse whitespace)
2. Chinese character spacing
3. Whitespace tokenization
4. Lowercase
5. Strip Unicode accents (NFD decomposition)
6. Split on punctuation
7. WordPiece greedy longest-match (`##` sub-word prefix)
8. Wrap with `[CLS]` ... `[SEP]`, truncate to MAX_LEN, pad with `[PAD]`
9. Generate attention mask (1 for real tokens, 0 for padding)

This is validated with **Golden Vector Parity Tests** (zero-tolerance token-ID matching).

---

## 📁 Project Structure

```
PhishGuard/
├── app/
│   └── src/main/
│       ├── assets/phishing/
│       │   ├── model_quant_8bit.onnx    # Quantized ONNX model (32.5 MB)
│       │   ├── vocab.txt                 # BERT WordPiece vocabulary (30,522 tokens)
│       │   ├── phishguard_config.json    # Model metadata & inference contract
│       │   └── golden_vectors.json       # Python-generated reference vectors
│       │
│       ├── java/com/phishguard/app/
│       │   ├── core/
│       │   │   ├── PhishGuardConfig.kt   # Central configuration constants
│       │   │   ├── UrlNormalizer.kt       # Cybersecurity-grade URL normalizer
│       │   │   └── SecureLogger.kt        # Redacted logging (no URLs in prod)
│       │   │
│       │   ├── data/
│       │   │   ├── BertWordPieceTokenizer.kt  # Full BERT 9-step tokenization
│       │   │   ├── OnnxModelLoader.kt         # Streaming ONNX + NNAPI/CPU
│       │   │   ├── PhishingUrlDetector.kt     # End-to-end detection orchestrator
│       │   │   └── CsvEvaluator.kt            # Batch CSV evaluation engine
│       │   │
│       │   ├── domain/
│       │   │   ├── DetectionResult.kt     # Scan result + BenchmarkResult
│       │   │   ├── CsvEvaluationResult.kt # Metrics + confusion matrix
│       │   │   └── ScanUrlUseCase.kt      # Clean-architecture use case
│       │   │
│       │   ├── benchmark/
│       │   │   └── BenchmarkRunner.kt     # Warmup + timed profiling
│       │   │
│       │   ├── test/
│       │   │   └── GoldenVectorTest.kt    # 2-stage parity validation
│       │   │
│       │   └── ui/
│       │       ├── MainActivity.kt        # Entry point + 4-tab navigation
│       │       ├── MainViewModel.kt       # State management for all screens
│       │       ├── ScanScreen.kt          # Real-time URL scanning
│       │       ├── HistoryScreen.kt       # Scan history (last 50)
│       │       ├── BenchmarkScreen.kt     # Latency profiling + golden tests
│       │       ├── EvaluateScreen.kt      # CSV evaluation + metrics dashboard
│       │       └── theme/                 # Material 3 dark/light theme
│       │
│       └── res/                           # Resources, icons, strings
│
├── tools/
│   └── generate_golden_vectors.py         # Python script for parity vectors
│
├── build.gradle.kts                       # Root build config
├── settings.gradle.kts                    # Plugin management
├── gradle.properties                      # JVM heap, AndroidX flags
└── RUN.md                                 # Detailed setup guide
```

---

## 📋 Prerequisites

| Tool | Version | Purpose |
|:---|:---|:---|
| **Android Studio** | Ladybug+ (2024.2+) | IDE with AGP 9 support |
| **JDK** | 17+ | Required by Gradle 9 |
| **Gradle** | 9.1.0 (bundled) | Build system |
| **AGP** | 9.0.1 | Android Gradle Plugin |
| **Kotlin** | 2.2.10 (bundled by AGP 9) | Language |
| **Android Device/Emulator** | API 26+ (Android 8.0+) | Runtime |
| **Python** | 3.10 (for golden vectors only) | Parity test generation |

---

## ⚙️ Setup & Build

### 1. Clone & Open

```bash
# Open the PhishGuard directory in Android Studio
# File → Open → navigate to PhishGuard/
```

### 2. Verify Model Assets

Ensure these files exist in `app/src/main/assets/phishing/`:

```
✅ model_quant_8bit.onnx    (~32.5 MB)
✅ vocab.txt                 (~226 KB)
✅ phishguard_config.json    (~1 KB)
```

If `model_quant_8bit.onnx` is missing, copy it from the training output:
```bash
cp <training_output>/best_model_epoch_018/model_quant_8bit.onnx \
   app/src/main/assets/phishing/
```

### 3. Sync & Build

```bash
# Sync Gradle (automatic in Android Studio)
./gradlew assembleDebug
```

### 4. Run on Device

```bash
./gradlew installDebug
```

Or press ▶️ **Run** in Android Studio.

---

## 🧪 Generating Golden Vectors

Golden vectors validate that the Android tokenizer + ONNX inference produce identical outputs to the Python training pipeline.

### Install Dependencies (Python 3.10)

```bash
py -3.10 -m pip install --user onnxruntime transformers numpy
```

### Generate Vectors

```bash
cd tools/
py -3.10 generate_golden_vectors.py \
    --model_dir "<path-to>/best_model_epoch_018" \
    --output "../app/src/main/assets/phishing/golden_vectors.json"
```

### Run Parity Test (In-App)

1. Open PhishGuard on your device
2. Navigate to **Benchmark** tab
3. Scroll to **Golden Vector Parity Test**
4. Tap **Run Parity Test**
5. Check Logcat: `PhishGuard/GoldenTest`

| Stage | Test | Tolerance |
|:---|:---|:---|
| Stage 1 | Token-ID exact match | Zero (bit-exact) |
| Stage 2 | Probability match | ε = 0.001 |

---

## 📊 CSV Batch Evaluation

Evaluate the model against a labeled dataset directly on-device.

### CSV Format

The CSV must have two columns:

| Column | Type | Description |
|:---|:---|:---|
| `input` | String | The raw URL |
| `label` | Integer | `0` = Benign, `1` = Phishing |

Example:
```csv
input,label
https://www.google.com,0
http://suspicious-paypal-login.xyz/verify,1
https://github.com/login,0
```

### How to Use

1. Transfer your CSV file to the Android device
2. Open PhishGuard → **Evaluate** tab
3. Tap **Select CSV** → browse to your file
4. Tap **Evaluate** → watch the progress bar
5. View results:
   - **Classification Metrics**: Accuracy, Precision, Recall, F1
   - **Extended Metrics**: ROC-AUC, FNR, FPR
   - **Confusion Matrix**: Visual 2×2 grid (TN/FP/FN/TP)
   - **Performance**: Throughput (URLs/sec), elapsed time

### Metrics Computed

| Metric | Formula | Significance |
|:---|:---|:---|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | How trustworthy are phishing alerts? |
| **Recall** | TP / (TP + FN) | What fraction of phishing URLs are caught? |
| **F1 Score** | 2 × (P × R) / (P + R) | Balanced precision-recall measure |
| **ROC-AUC** | Area under ROC curve | Threshold-independent discrimination ability |
| **FNR** | FN / (FN + TP) | Rate of missed phishing (security-critical) |
| **FPR** | FP / (FP + TN) | Rate of false alarms |

### Confusion Matrix Labels

```
                    Predicted
                 Benign    Phishing
Actual  Benign  │ TN (L→L) │ FP (L→M) │
      Phishing  │ FN (M→L) │ TP (M→M) │

L = Legitimate (Benign)    M = Malicious (Phishing)
```

- **TN (L→L)**: Correctly identified as benign ✅
- **FP (L→M)**: Benign URL flagged as phishing (false alarm) ❌
- **FN (M→L)**: Phishing URL missed (most dangerous!) ❌
- **TP (M→M)**: Correctly identified as phishing ✅

---

## ⚡ Benchmarking

The built-in benchmarking tool measures end-to-end inference latency:

1. Open PhishGuard → **Benchmark** tab
2. Tap **Run Benchmark**
3. Runs: 5 warmup + 20 timed iterations
4. Reports:
   - **p50 / p90 / Mean latency** (end-to-end)
   - **Tokenization** vs **Inference** breakdown
   - **Execution provider** (CPU or NNAPI)
   - **Device info**

---

## 🔒 Security Considerations

| Feature | Implementation |
|:---|:---|
| **Zero Data Exfiltration** | All inference runs on-device; no network calls |
| **URL Sanitization** | Control character stripping, Unicode NFKC normalization |
| **Punycode Detection** | Flags IDN homograph attack vectors (xn-- domains) |
| **Unicode Confusable Detection** | Detects Cyrillic, Greek, Armenian lookalike characters |
| **Secure Logging** | URLs are redacted from production logs |
| **No URL Storage** | Scan history is in-memory only (cleared on app close) |
| **Model Integrity** | ONNX model loaded from signed APK assets |

---

## 🛠 Tech Stack

| Component | Technology |
|:---|:---|
| **Language** | Kotlin 2.2.10 |
| **UI Framework** | Jetpack Compose + Material Design 3 |
| **ML Runtime** | ONNX Runtime 1.24.2 (CPU + NNAPI) |
| **Model** | MiniLM-L12-H384 + LoRA, QUInt8 quantized |
| **Build System** | Gradle 9.1.0 + AGP 9.0.1 |
| **Min SDK** | Android 8.0 (API 26) — covers 95%+ devices |
| **Target SDK** | Android 15 (API 36) |
| **JSON Parsing** | Gson 2.13.2 |
| **Architecture** | MVVM (ViewModel + StateFlow + Compose) |

---

## 📚 Research Context

This application is part of a phishing URL detection research project at **IIT Ropar**, exploring the deployment of large language models for cybersecurity on resource-constrained mobile devices.

### Research Pipeline

```
Raw URL Dataset (26.5M samples)
         │
         ▼
  URL Preprocessing & Feature Extraction
         │
         ▼
  MiniLM-L12-H384 Fine-Tuning (LoRA)
   • 20 epochs, cosine warmup
   • Weighted random sampling
   • Mixed-precision training (FP16)
         │
         ▼
  ONNX Export & INT8 Quantization
   • Dynamic quantization (QUInt8)
   • 32.5 MB model footprint
         │
         ▼
  ┌──────────────────────────────┐
  │  PhishGuard Android App      │  ← You are here
  │  On-device inference          │
  │  <100ms per URL               │
  └──────────────────────────────┘
```

### Key Research Questions

1. **Deployment Feasibility**: Can a 33M-parameter transformer run efficiently on mobile?
2. **Parity Guarantee**: Does the mobile inference pipeline produce identical results to the training pipeline?
3. **Threshold Optimization**: How does the Youden's J statistic (0.59) balance FNR vs FPR in deployment?
4. **Quantization Impact**: What is the accuracy loss from FP32 → INT8 quantization?

---

<p align="center">
  <strong>PhishGuard</strong> — Bringing transformer-grade phishing detection to every Android device.<br/>
  <em>Built with 🔬 at IIT Ropar</em>
</p>
