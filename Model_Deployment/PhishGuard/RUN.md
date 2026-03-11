# PhishGuard — Build & Run Guide

## Prerequisites

- **Android Studio** Hedgehog (2023.1.1) or newer
- **JDK 17** (bundled with Android Studio)
- **Android SDK** API 34
- **Python 3.8+** with `transformers`, `onnxruntime`, `numpy` (for golden vector generation)

---

## 1. Copy Model Assets

Copy the following files from the trained model directory into `app/src/main/assets/phishing/`:

```powershell
# From the PhishGuard project root:
$MODEL_DIR = "..\saved_models\MiniLM_data10\best_model_epoch_018"

# Create assets directory (if not already there)
New-Item -ItemType Directory -Force -Path "app\src\main\assets\phishing"

# Copy quantized ONNX model (~32.5 MB)
Copy-Item "$MODEL_DIR\model_quant_8bit.onnx" "app\src\main\assets\phishing\"

# Copy vocabulary file
Copy-Item "$MODEL_DIR\vocab.txt" "app\src\main\assets\phishing\"
```

After copying, the `app/src/main/assets/phishing/` directory should contain:
- `model_quant_8bit.onnx` (~32.5 MB)
- `vocab.txt` (30523 lines)
- `phishguard_config.json` (already created)

---

## 2. Open in Android Studio

1. Open Android Studio
2. **File → Open** → navigate to the `PhishGuard` folder
3. Wait for Gradle sync to complete
4. If prompted, accept any SDK or plugin updates

---

## 3. Build

```
Build → Make Project   (or Ctrl+F9)
```

Expected: Build succeeds with 0 errors.

---

## 4. Run on Device / Emulator

### Option A: Physical Samsung device (recommended)
1. Enable **Developer Options** + **USB Debugging** on your Samsung phone
2. Connect via USB
3. Select device in target dropdown
4. Click **Run** (Shift+F10)

### Option B: Emulator
1. **Tools → Device Manager → Create Device**
2. Select a Pixel device, API 34, x86_64 image
3. Click **Run**

> **Note**: ONNX Runtime on emulator will use CPU only. NNAPI is only available on real devices.

---

## 5. Smoke Test (in-app)

1. On the **Scan** tab, tap any of the 5 sample URL chips
2. Verify scan completes in <500ms
3. Check that safe URLs (google.com, amazon.com) show green "SAFE"
4. Check that suspicious URLs show red "PHISHING"
5. Switch to **History** tab — verify scans are listed
6. Switch to **Benchmark** tab — tap "Run Benchmark"
7. Verify p50, p90, and mean latency are displayed

---

## 6. Generate Golden Vectors (Parity Testing)

```bash
cd PhishGuard/tools

python generate_golden_vectors.py \
    --model_dir "D:\IIT ROPAR\phishing URL Detection\01_Research Tracker\2_Model_Building\PhishURLDetect-with-LLMS\1_Model_On_Raw_data\05_MiniLM\saved_models\MiniLM_data10\best_model_epoch_018" \
    --output "D:\IIT ROPAR\phishing URL Detection\01_Research Tracker\2_Model_Building\PhishURLDetect-with-LLMS\1_Model_On_Raw_data\05_MiniLM\Model_Deployment\PhishGuard\app\src\main\assets\phishing\golden_vectors.json"
```

This generates `golden_vectors.json` with reference token IDs and probabilities.

---

## 7. Run Golden Vector Tests

The golden vector test is integrated into the app. To trigger it programmatically, you can add a test button or run it from the benchmark screen. The test performs:

- **Stage 1**: Exact token-ID match (zero tolerance)
- **Stage 2**: Probability match (ε=0.001)

Check **Logcat** with tag filter `PhishGuard/GoldenTest` for results.

---

## Project Structure

```
PhishGuard/
├── app/
│   ├── build.gradle.kts           — Dependencies & build config
│   ├── proguard-rules.pro         — R8/ProGuard rules
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/phishing/       — Model + vocab + config
│       ├── res/values/            — Strings, themes
│       └── java/com/phishguard/app/
│           ├── core/              — Config, UrlNormalizer, SecureLogger
│           ├── data/              — Tokenizer, ModelLoader, Detector
│           ├── domain/            — DetectionResult, ScanUrlUseCase
│           ├── benchmark/         — BenchmarkRunner
│           ├── ui/                — Compose screens + ViewModel
│           │   └── theme/         — Colors, Typography, Theme
│           └── test/              — GoldenVectorTest
├── tools/
│   └── generate_golden_vectors.py — Python golden vector generator
├── build.gradle.kts               — Project-level Gradle
├── settings.gradle.kts
├── gradle.properties
└── RUN.md                         — This file
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails with "ONNX Runtime" error | Check `app/build.gradle.kts` has `com.microsoft.onnxruntime:onnxruntime-android:1.17.0` |
| App crashes on model load | Ensure `model_quant_8bit.onnx` is in `assets/phishing/` and is ~32MB |
| Token IDs mismatch in golden test | Check that `vocab.txt` matches the one from `best_model_epoch_018` |
| NNAPI not activating | NNAPI only works on real devices with API 27+. Check logcat for EP used. |
| OOM on model copy | Model loader uses streaming 8KB buffer; if still OOM, increase heap in manifest |
| Slow first scan | First scan includes model-to-internal-storage copy. Subsequent scans are cached. |

---

## Key Constants

| Constant | Value |
|----------|-------|
| MAX_LENGTH | 192 |
| Threshold | 0.59 |
| Model size | ~32.5 MB (quantized INT8) |
| Vocab size | 30,522 tokens |
| Min SDK | 26 (Android 8.0) |
| Target SDK | 34 |
