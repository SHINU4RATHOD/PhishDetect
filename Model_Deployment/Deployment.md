3) Android edge deployment (Samsung) — the clean, high-performance path

You have two solid runtime options:

Option A (simplest): ONNX Runtime Android loads ONNX directly

Use onnxruntime-android and load model_quant_8bit.onnx.

Option B (smaller runtime): ONNX Runtime Mobile (requires ORT format)

onnxruntime-mobile expects the model converted into ORT format, not raw ONNX.
(If APK size matters a lot, do this later; get Option A working first.)

ONNX Runtime Java supports adding NNAPI as an execution provider, which is what you want on Samsung/Android for acceleration.






################################ prompt2 
## Prompt (Claude Opus 4.6 Thinking)
You are a world-class AI scientist + mobile deployment engineer + cybersecurity ML researcher with decades of experience deploying Transformer/LLM models on constrained edge devices (Samsung Android). Your mission is to implement a **production-grade Android edge deployment** for a **MiniLM phishing URL detector** using a **quantized ONNX model (~32MB)**.

The solution must be **world class**: clean, modular, scalable, auditable, object-oriented design, production-ready engineering standards. It must be easy to maintain, easy to test, and easy to extend.

### 0) Absolute rule: Deep dive first. No rushing.

Before writing any deployment code, take as much time as needed to **deeply read and understand the entire `05_MiniLM` folder** end-to-end:

Folder contains (at minimum):

* `saved_models/` (best model artifacts: ONNX, quantized ONNX, tokenizer, metadata)
* `MiniLM_V1.py`
* `MiniLM_V2_inference.py` (or similar inferencing script)
* `README.md`

You must:

1. Open and inspect **every file**, including subfolders and all generated artifacts (ONNX files, tokenizer assets, JSON metadata).
2. Build a complete mental model of the end goal: phishing URL detection on-device.
3. Produce a precise **Inference Contract** summary **from the code**, not assumptions.

Do not be in a hurry. Accuracy and correctness matter more than speed.

---

## 1) Deliverable #1: System Understanding Report (must be exact)

After reading everything in `05_MiniLM`, output a structured report:

### A) Repository Map

* List all important files/folders and what each does
* Identify the single source of truth for:

  * preprocessing/tokenization
  * model export (ONNX)
  * quantization
  * inference pipeline
  * threshold selection

### B) Model + Tokenizer Artifacts

From `saved_models/...best_model.../` identify and document:

* quantized model file name (e.g., `model_quant_8bit.onnx`)
* non-quantized ONNX file
* tokenizer files present (`vocab.txt`, `tokenizer.json`, etc.)
* metadata (`deployment_metadata.json`, `final_results.json`, etc.)
* constants: `MAX_LEN`, label mapping, threshold, class index

### C) Inference Contract (most important)

State precisely:

* input tensor names (e.g., `input_ids`, `attention_mask`)
* input types (int64 vs int32)
* input shapes ([1, max_len])
* output name(s) and shape ([1,2] logits)
* probability computation (softmax)
* class index corresponding to phishing
* threshold rule (>= threshold → phishing)

---

## 2) Deliverable #2: Android Edge Deployment (Option A)

Implement **Option A**: Deploy raw ONNX on Samsung Android using **ONNX Runtime Android** (NNAPI optional, CPU fallback required). The result must be production-grade.

### A) Android Integration Plan

Provide:

* Gradle dependencies
* asset directory layout (where to put model + tokenizer + configs)
* model loading approach (asset → internal file → ORT session)
* session tuning:

  * optimization level
  * intra/inter threads
  * optional `addNnapi()` with safe fallback

### B) Tokenization on Android (accuracy-critical)

Tokenizer must match Python/HuggingFace MiniLM uncased WordPiece exactly.

You must pick and fully implement one:

**Option 1 (preferred if feasible):**
Use `tokenizer.json` with an offline tokenizer runtime suitable for Android.

**Option 2:**
Implement Kotlin BERT WordPiece tokenizer using `vocab.txt`:

* lowercasing
* punctuation splitting (BERT basic tokenizer style)
* WordPiece greedy with `##`
* add `[CLS]`, `[SEP]`
* pad/truncate to max_len
* attention mask creation

You must ensure token IDs produced on Android match Python exactly for the same input URL.

### C) Kotlin Inference Wrapper

Write Kotlin ready:

* `PhishingUrlDetector` class with `predict(url: String)` returning:

  * phishing probability
  * label
  * optional debug info (token count, latency)
* Correct tensor creation with int64
* Stable softmax
* Avoid unnecessary allocations (reuse buffers where possible)

### D) Benchmark Harness (mobile KPI)

Implement:

* warmup runs
* timing for tokenize-only, inference-only, end-to-end
* p50/p90/mean latency reporting
* print device info + execution provider used (NNAPI vs CPU)

### E) Parity / Golden Test Vectors (must-have)

Define a rigorous validation pipeline:

* On Python side: generate “golden vectors” for N URLs:

  * `input_ids`, `attention_mask`, expected `p_phishing`
* On Android: verify:

  * tokenization IDs/masks match exactly
  * probabilities match within tolerance

Include a short Python snippet/pseudocode for generating golden vectors and how Android should load and check them.

---

## 3) Deliverable #3: Cybersecurity-grade robustness

Include edge hardening:

* URL normalization (Unicode NFKC, strip control chars)
* handle extremely long URLs safely (cap length)
* handle empty/malformed inputs
* logging best practice (hash URL before logging; avoid leaking user info)
* optional: detect Unicode confusables / punycode and flag risk

---

## 4) NEW Deliverable #4: Android Studio Project Skeleton (must generate full project)

You must output a **complete Android Studio project skeleton**, including:

### A) Project setup (Gradle + modules)

* Android Studio project structure, module name (e.g., `app`)
* Minimum SDK + target SDK choices with justification for Samsung devices
* `build.gradle` (project + app) with dependencies and packagingOptions if needed
* ProGuard/R8 configuration notes (even if not enabled initially)

### B) Package structure (clean architecture)

Provide a clean, scalable structure, e.g.:

* `com.company.phishguard`

  * `ui/` (screens, composables, theme)
  * `domain/` (use-cases, models)
  * `data/` (tokenizer, model loader, detector implementation)
  * `core/` (utils, logging, normalization, config)
  * `di/` (optional simple dependency wiring)
  * `benchmark/` (perf measurement utilities)
  * `test/` (golden vector tests)

Use best-practice separation so the model logic is not tangled in UI.

### C) MainActivity + Beautiful world-class UI (must be included)

Create a **world top-class, beautiful UI** (prefer Jetpack Compose unless there is a strong reason not to), with:

* A clean minimal “security product” aesthetic:

  * professional typography
  * smooth spacing
  * modern card layout
  * subtle animations/transitions (lightweight)
  * dark mode support

UI features:

1. URL input box with validation
2. “Scan” button
3. Result card showing:

   * verdict (SAFE / PHISHING)
   * probability score
   * threshold used
   * inference latency (ms)
   * execution provider (NNAPI/CPU)
4. A small “Details” expandable section:

   * token count
   * max length
   * model file name
5. A “History” list (last N scans) stored in-memory (optionally persistent later)
6. A “Benchmark” page/section that runs 20 scans and prints p50/p90/mean

### D) Wiring

* UI must call the detector asynchronously (coroutines)
* Proper lifecycle handling (initialize session/tokenizer once)
* Thread safety and cancellation
* No UI freezes

### E) Assets packaging

Provide exact instructions and code for copying assets to internal storage and loading or do it by youself by your agent 

* `assets/phishing/model_quant_8bit.onnx`
* tokenizer files
* metadata/config JSON

---

## 5) Output Format Requirements

Final response must be structured as:

1. **Repository Map**
2. **Inference Contract**
3. **Artifacts to Ship**
4. **Android Deployment Steps**
5. **Android Studio Project Skeleton**
6. **Kotlin Code (Tokenizer + ORT Wrapper + UI + Benchmark)**
7. **Parity Testing Plan**
8. **Performance Tuning Checklist**
9. **Debugging Playbook (common failures + fixes)**

Again: do not rush. Read all files and artifacts first, then implement.

If you discover missing files or mismatches (tokenizer.json missing, wrong vocab, input name mismatch), propose the smallest fix and update the deployment plan accordingly.

---
output a complete Android Studio project with a premium UI.
1. All code must be provided as complete files with exact paths (e.g., app/src/main/java/.../MainActivity.kt). No pseudo-code for the main implementation.
2. Choose specific dependency versions (Compose BOM, Kotlin, AGP, ONNX Runtime) and keep them internally consistent. Provide gradle.properties and settings.gradle as needed
3. Default decision rule: if tokenizer.json exists, use it; otherwise implement WordPiece tokenizer from vocab.txt.
4. Do not assume input/output node names; confirm them from ONNX (session metadata) and cross-check with MiniLM_V2_inference.py
5. Include a simple in-app smoke test: 5 sample URLs as chips + one-tap scan.
6. Warmup runs required before benchmarking; report p50/p90/mean.
7. Every Kotlin file must be in clean architecture packages as specified.
8. Include a ‘RUN.md’ with step-by-step build/run instructions.




























---------------
PhishGuard Android Edge Deployment — Implementation Plan
Problem & Goal
Deploy a MiniLM phishing URL detector as a production-grade Android app using the quantized ONNX model (
model_quant_8bit.onnx
, ~32MB) from the 05_MiniLM research pipeline. The app targets Samsung Android devices with ONNX Runtime (NNAPI optional, CPU fallback).

Deliverable #1: System Understanding Report
A) Repository Map
File/Folder	Purpose
MiniLM_V2_inference.py
Training + inference pipeline (2517 lines) — single source of truth
MiniLM_V1.py
Earlier version (superseded)
README.md
Architecture docs, KPIs, setup
Model_Deployment/Deployment.md
Deployment strategy notes
saved_models/MiniLM_data10/	All model artifacts root
saved_models/.../best_model_epoch_018/	Best model (latest, higher KPI score 0.9323)
saved_models/.../final_results.json	Overall experiment results
saved_models/.../final_test_evaluation/	Test metrics, confusion matrix, ROC/PR curves
Sources of Truth:

Concern	Location
Preprocessing/Tokenization	
MiniLM_V2_inference.py
 line 329: AutoTokenizer from microsoft/MiniLM-L12-H384-uncased
ONNX Export	
MiniLM_V2_inference.py
 lines 880–923: ModelExporter.export_onnx()
Quantization	
MiniLM_V2_inference.py
 lines 925–953: 
_quantize_onnx()
 using QuantType.QUInt8
Inference	
MiniLM_V2_inference.py
 lines 1575–1589: softmax → probs[:, 1]
Threshold	
final_results.json
: optimal_threshold = 0.59
B) Model & Tokenizer Artifacts (from best_model_epoch_018)
Artifact	File
Quantized ONNX	
model_quant_8bit.onnx
 (~32.5 MB)
Full ONNX	
model.onnx
 (~130 MB)
Tokenizer JSON	
tokenizer.json
 (HuggingFace fast tokenizer, 30672 lines)
Tokenizer Config	
tokenizer_config.json
 (BertTokenizer, do_lower_case=true)
Vocab	
vocab.txt
 (30523 tokens)
Special Tokens	
special_tokens_map.json
Metadata	
deployment_metadata.json
Constants:

Constant	Value
MAX_LEN	192
Label 0	Benign (Legitimate)
Label 1	Phishing (Malicious)
Threshold	0.59 (≥ threshold → phishing)
Vocab Size	30522
[PAD] ID	0
[UNK] ID	100
[CLS] ID	101
[SEP] ID	102
C) Inference Contract
Property	Value
Input 1 name	input_ids
Input 1 type	int64 (torch.long in export)
Input 1 shape	[1, 192]
Input 2 name	attention_mask
Input 2 type	int64
Input 2 shape	[1, 192]
Output name	logits
Output shape	[1, 2] (benign logit, phishing logit)
Probability	softmax(logits, dim=1) → p_phishing = probs[0][1]
Decision rule	p_phishing >= 0.59 → PHISHING
IMPORTANT

ONNX Runtime on Android may expect int64 tensors. The export code uses torch.long (int64). We must create OnnxTensor with long[] arrays on Android.

Deliverable #2–4: Android Studio Project
A) Project Setup
Min SDK: 26 (Android 8.0 — covers 95%+ Samsung devices)
Target SDK: 34
Kotlin: 1.9.22
AGP: 8.2.2
Compose BOM: 2024.02.00
ONNX Runtime: 1.17.0
B) Package Structure
com.phishguard.app/
├── core/
│   ├── PhishGuardConfig.kt       — Constants (MAX_LEN, threshold, etc.)
│   ├── UrlNormalizer.kt          — NFKC, strip control chars, punycode
│   └── SecureLogger.kt           — SHA-256 hashing for URLs in logs
├── data/
│   ├── BertWordPieceTokenizer.kt — WordPiece tokenizer from vocab.txt
│   ├── OnnxModelLoader.kt        — Asset copy + ORT session creation
│   └── PhishingUrlDetector.kt    — Main predict(url) wrapper
├── domain/
│   ├── DetectionResult.kt        — Data classes
│   └── ScanUrlUseCase.kt         — Use-case orchestration
├── benchmark/
│   └── BenchmarkRunner.kt        — Warmup + p50/p90/mean + device info
├── ui/
│   ├── theme/
│   │   ├── Color.kt
│   │   ├── Theme.kt
│   │   └── Type.kt
│   ├── MainViewModel.kt
│   ├── MainActivity.kt
│   ├── ScanScreen.kt
│   ├── HistoryScreen.kt
│   └── BenchmarkScreen.kt
└── test/
    └── GoldenVectorTest.kt
C) Files to Create
NOTE

The project root is d:\IIT ROPAR\phishing URL Detection\01_Research Tracker\2_Model_Building\PhishURLDetect-with-LLMS\1_Model_On_Raw_data\05_MiniLM\Model_Deployment\PhishGuard\

Build System (5 files)
[NEW] 
build.gradle.kts
Project-level build file with AGP + Kotlin plugin declarations.

[NEW] 
app/build.gradle.kts
App module with Compose, ONNX Runtime, coroutines deps. packagingOptions with META-INF exclusions for ORT.

[NEW] 
settings.gradle.kts
[NEW] 
gradle.properties
[NEW] 
app/proguard-rules.pro
Android Manifest & Resources (1 file)
[NEW] 
AndroidManifest.xml
Core Layer (3 files)
[NEW] 
PhishGuardConfig.kt
All inference constants: MAX_LEN=192, threshold=0.59, model filename, vocab filename, special token IDs.

[NEW] 
UrlNormalizer.kt
Unicode NFKC normalization, strip control chars, cap length, detect punycode/IDN.

[NEW] 
SecureLogger.kt
SHA-256 hash before logging URLs.

Data Layer (3 files)
[NEW] 
BertWordPieceTokenizer.kt
Complete BERT WordPiece tokenizer from 
vocab.txt
. Implements: lowercase, BERT basic tokenizer (whitespace + punctuation split), WordPiece greedy match with ##, [CLS]/[SEP] wrapping, pad/truncate to MAX_LEN, attention mask.

[NEW] 
OnnxModelLoader.kt
Copies model from assets/ to internal storage, creates OrtSession with optimization level ALL, 2 intra-op threads, optional NNAPI with CPU fallback.

[NEW] 
PhishingUrlDetector.kt
Main predict(url: String): DetectionResult. Creates int64 OnnxTensor, runs session, stable softmax, returns probability + label + latency + debug info.

Domain Layer (2 files)
[NEW] 
DetectionResult.kt
[NEW] 
ScanUrlUseCase.kt
Benchmark Layer (1 file)
[NEW] 
BenchmarkRunner.kt
5 warmup runs + 20 timed runs, reports p50/p90/mean for tokenize-only, inference-only, end-to-end. Prints device info + execution provider.

UI Layer (7 files)
[NEW] 
Color.kt
[NEW] 
Theme.kt
[NEW] 
Type.kt
[NEW] 
MainViewModel.kt
[NEW] 
MainActivity.kt
[NEW] 
ScanScreen.kt
Premium security UI with URL input, sample URL chips, scan button, animated result card (SAFE green / PHISHING red), expandable details, latency display.

[NEW] 
BenchmarkScreen.kt
Test & Validation (2 files)
[NEW] 
generate_golden_vectors.py
Python script that produces golden_vectors.json with input_ids, attention_mask, p_phishing for N sample URLs.

[NEW] 
GoldenVectorTest.kt
Documentation (1 file)
[NEW] 
RUN.md
Assets to Package
The following files from best_model_epoch_018 need to be placed in app/src/main/assets/phishing/:

model_quant_8bit.onnx
 — quantized model (~32.5MB)
vocab.txt
 — tokenizer vocabulary (30523 tokens)
phishguard_config.json — generated config with threshold, MAX_LEN, special tokens
WARNING

The actual ONNX and vocab files are very large. Instead of copying them as code, the RUN.md will provide exact copy commands. The phishguard_config.json will be generated as a new small file.

Verification Plan
1. Build Verification
Command: Open project in Android Studio → Build → Make Project. Should compile without errors.

2. Manual Smoke Test (on-device)
Install APK on Samsung device or emulator (API 26+)
Tap each of the 5 sample URL chips → verify scan results appear
Verify safe URLs show green "SAFE", phishing URLs show red "PHISHING"
Navigate to Benchmark tab → run benchmark → verify p50/p90/mean latency displays
Check History tab shows previous scans
3. Golden Vector Parity Test (Python + Android)
Run python tools/generate_golden_vectors.py on PC to generate golden_vectors.json
Copy JSON to assets
Run in-app golden vector test (button in Benchmark screen)
Verify tokenization IDs match exactly, probabilities within ε=0.001
IMPORTANT

Manual testing on a real Samsung device (or emulator with API 26+) is required since ONNX Runtime cannot be unit-tested without the Android runtime. The golden vector test is designed to run in-app.


 