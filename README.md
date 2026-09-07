---
title: OIL SIF Precursor Detection Engine
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
---

# 🛡️ Oil India Limited (OIL) — SIF Precursor Detection Engine
### Smart India Hackathon (SIH 2026) | Problem Statement: SIH26165
**Team**: Purple Team  
**Champion Production Model**: DistilBERT V2 (`./sif_model_binary_v2`)  
**Package Type**: Standalone, Fully Self-Contained Portable Deployment  

---

## 📌 Overview

This directory contains the **frozen, presentation-ready copy** of the AI/NLP SIF Precursor Detection Engine. It has been prepared for execution on another presentation laptop with **zero external API dependencies, zero cloud connectivity requirements at runtime, and pre-bundled model weights**.

The dashboard triages safety observations, near-misses, and unsafe acts into **Critical-SIF** vs. **Non-Critical** using our fine-tuned DistilBERT V2 architecture with class-weighted loss, safety threshold calibration, domain barrier extraction, and SHAP token-level attribution.

---

## 💻 System Prerequisites

* **Operating System**: macOS, Linux, or Windows (64-bit)
* **Python Version**: **Python 3.9, 3.10, or 3.11** *(Python 3.9.6+ recommended)*
* **RAM**: 4 GB minimum (8 GB recommended)
* **Disk Space**: ~1.5 GB free space for environment & dependencies
* **Hardware**: CPU only (no GPU required; inference runs efficiently on CPU)

---

## 🚀 4-Step Quickstart Guide

### 1. Open Terminal and Navigate to this Folder
Open your terminal (macOS/Linux) or PowerShell / Command Prompt (Windows) and `cd` into this directory:
```bash
cd PurpleTeam_SIH26165_PORTABLE
```

### 2. Create and Activate a Fresh Virtual Environment
* **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

### 3. Install Required Dependencies
Upgrade `pip` and install the pinned dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
*(Note: If `python -m spacy download` is skipped, `app.py` has an automated download fallback on first startup).*

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

### 5. Access the Dashboard
Once started, open your web browser to:
```
http://localhost:8501
```

---

## 📁 Portable Directory Structure

```
PurpleTeam_SIH26165_PORTABLE/
├── app.py                                # Production Streamlit multi-tab dashboard
├── requirements.txt                      # Complete, pinned Python package dependencies
├── README.md                             # Setup instructions & operational documentation
├── PORTABILITY_CHECK.txt                 # Audit report & clean-room validation log
│
├── sif_model_binary_v2/                  # FROZEN CHAMPION MODEL V2 (Complete local bundle)
│   ├── config.json                       # DistilBERT model configuration
│   ├── model.safetensors                 # Pretrained & fine-tuned model weights (267 MB)
│   ├── tokenizer.json                    # Fast HuggingFace tokenizer
│   ├── tokenizer_config.json             # Tokenizer configuration
│   ├── vocab.txt                         # Vocabulary dictionary
│   ├── special_tokens_map.json           # Special token markers
│   └── training_args.bin                 # HuggingFace training metadata
│
├── sif_threshold_config.json             # Phase 6C safety threshold calibration config
├── model_v2_validation_report.json       # V2 multi-benchmark JSON metrics (loaded by UI)
├── model_v2_comparison.csv               # Model V1 vs V2 comparison table (loaded by UI)
├── example_shap_cache.json               # Precomputed SHAP values for instant preset clicks
│
├── demo_reports/                         # Presentation Incident Reports for Live Demo
│   ├── critical.txt                      # Clear Critical-SIF case (High-voltage electrical)
│   ├── noncritical.txt                   # Clear Non-Critical compliant case (LOTO verified)
│   ├── ambiguous.txt                     # Borderline / low-energy housekeeping case
│   └── backup/                           # Additional representative scenario reports
│       ├── backup_confined_space_critical.txt
│       ├── backup_confined_space_safe.txt
│       ├── backup_lifting_crane_critical.txt
│       ├── backup_fall_protection_safe.txt
│       └── backup_gas_hotwork_critical.txt
│
└── benchmarks_and_evidence/              # Preserved Evidence & Frozen Benchmark Data
    ├── validation_50.csv                 # Unseen 50-report strict evaluation benchmark
    ├── validation_50_results.csv         # Per-sample prediction & probability audit
    ├── validation_50_report.txt          # Complete official evaluation summary
    ├── original_test_preserved.csv       # Preserved 120-report original test set
    ├── hard_negative_test.csv            # Unseen 39-report diagnostic hard-negative set
    ├── model_v2_validation_report.txt    # Detailed V2 performance text report
    ├── split_integrity_report.json       # SHA-256 split leakage audit report
    ├── split_integrity_report.txt        # Split integrity summary
    └── validation_10_scenarios_explainability.json
```

---

## 🎯 Live Presentation Demo Workflow

During the presentation, demonstrate the model's contextual safety understanding using the reports in `demo_reports/` or the 4 instant buttons on **Tab 1: SIF Triage & Investigation**:

1. **Preset Button "⚡ Case A: Critical Electrical"** (or paste `demo_reports/critical.txt`):
   - *Report*: `"Electrician worked on energized 480V MCC panel without applying lockout tagout or verifying absence of voltage."`
   - *Result*: **🚨 POTENTIAL SIF PRECURSOR DETECTED** (P(Critical-SIF) = ~53.3% > Threshold 0.45).
   - *Supervisor Action*: Escalates immediately to High-Priority Supervisor Review; flags missing LOTO & zero energy verification.

2. **Preset Button "🛡️ Case B: Safe Electrical"** (or paste `demo_reports/noncritical.txt`):
   - *Report*: `"Electrician isolated 480V MCC cabinet, applied personal lockout tagout padlock, and verified zero energy using a calibrated multimeter before entering."`
   - *Result*: **🟢 NON-CRITICAL / CONTROLS VERIFIED** (P(Critical-SIF) drops to ~34.4%).
   - *Supervisor Action*: Classified as routine; highlights verified barriers and mitigates false alarm fatigue.

3. **Paste Ambiguous Case (`demo_reports/ambiguous.txt`)**:
   - *Report*: `"Technician observed minor mineral oil seepage around pump skid base during routine perimeter walk; area wiped down and absorbent pad placed."`
   - *Result*: Non-Critical observation; demonstrates that low-energy housekeeping without pressure/toxic release does not trigger false SIF alarms.

4. **Navigate to Tab 2: Model Benchmark Analytics**:
   - Shows live offline benchmark comparison between Model V1 and Model V2:
     - Critical-SIF Recall: **92.45%** (+13.2% gain)
     - Hard-Negative Accuracy: **66.67%** (up from 25.64%)
     - Compliant False Alarms: reduced to **37.9%** (down from 96.6%)

5. **Navigate to Tab 3: Counterfactual Safety Validation**:
   - Demonstrates positive mean counterfactual separation ($\Delta P = +0.0788$, 7/10 pairs correct) proving the neural network evaluates control presence rather than keyword matching alone.

---

## 📊 Official V2 Benchmark & Evidence Record

The frozen Model V2 evaluation records preserved in `benchmarks_and_evidence/` match the presentation deck:

| Evaluation Scope | Metric | Model V2 Result |
| :--- | :--- | :--- |
| **Strict 50-Report Benchmark (`validation_50.csv`)** | **Overall Accuracy** | **72.00%** (36 / 50 correct) |
| | **Critical-SIF Precision** | **73.91%** |
| | **Critical-SIF Recall** | **68.00%** (17 / 25 SIFs caught) |
| | **Critical-SIF F1-Score** | **0.7083** |
| | **Macro F1-Score** | **0.7196** |
| | **Confusion Matrix** | **TN: 19, FP: 6, FN: 8, TP: 17** |
| **Held-Out Test Set (120 Reports)** | **Critical-SIF Recall (0.40 Thresh)** | **92.45%** (49 / 53 SIFs caught) |
| | **Critical-SIF F1-Score** | **0.6323** |
| **Diagnostic Hard Negatives (39 Reports)**| **Overall Accuracy** | **66.67%** (26 / 39 correct) |
| | **Compliant Specificity** | **62.07%** (18 / 29 correct) |
| **Contextual Matched Pairs (10 Pairs)** | **Mean $\Delta P$ (Unsafe - Safe)** | **+0.0788** (7 / 10 pairs correct) |

---

## 🔧 Troubleshooting & FAQ

* **Q: Port 8501 is already in use.**
  * *Fix*: Launch Streamlit on an alternative port:
    ```bash
    streamlit run app.py --server.port 8502
    ```
* **Q: Error: `Can't find model 'en_core_web_sm'`**
  * *Fix*: The dashboard will automatically attempt downloading it, or you can manually run:
    ```bash
    python -m spacy download en_core_web_sm
    ```
* **Q: Error: `numpy.dtype size changed` or Numba compilation error**
  * *Fix*: Ensure NumPy is restricted to `<2.0.0` as specified in `requirements.txt`:
    ```bash
    pip install "numpy<2.0.0"
    ```
* **Q: Does the app require internet at runtime?**
  * *No.* All model weights, tokenizers, configurations, and preset attributions are stored locally. Internet is only used during initial `pip install`.

---
*Created by Purple Team for Oil India Limited (OIL) — Smart India Hackathon SIH26165.*
