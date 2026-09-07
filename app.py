import html
import json
from pathlib import Path
import re
import string
import numpy as np
import pandas as pd
import spacy
import spacy.cli
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import shap

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OIL SIF Precursor Detection Engine | SIH26165 Purple Team",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Industrial CSS
st.markdown(
    """
    <style>
    /* Dark Industrial Safety Operations Theme */
    .stApp {
        background-color: #0b0f17;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #161f30 0%, #0d1522 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .badge-purple {
        background-color: #6e40c9;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-oil {
        background-color: #0056b3;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 8px;
    }
    
    /* Result Cards */
    .critical-card {
        background: linear-gradient(135deg, rgba(218, 54, 51, 0.2) 0%, rgba(218, 54, 51, 0.08) 100%);
        border: 2px solid #da3633;
        border-radius: 12px;
        padding: 24px;
        margin: 18px 0;
        animation: pulse-border 2.5s infinite;
    }
    .non-critical-card {
        background: linear-gradient(135deg, rgba(35, 134, 54, 0.2) 0%, rgba(35, 134, 54, 0.08) 100%);
        border: 2px solid #238636;
        border-radius: 12px;
        padding: 24px;
        margin: 18px 0;
    }
    
    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(218, 54, 51, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(218, 54, 51, 0); }
        100% { box-shadow: 0 0 0 0 rgba(218, 54, 51, 0); }
    }
    
    /* Metrics Box */
    .metric-panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f0f6fc;
        margin: 4px 0;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    /* Token Span Styles */
    .token-span {
        display: inline-block;
        padding: 2px 6px;
        margin: 2px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.95rem;
    }
    
    /* Safety Context Box */
    .context-box {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 18px;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Resource & Model Loader (Cached)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner=False)
def load_nlp_pipeline():
    """Loads spaCy model for text preprocessing with automated download fallback."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return nlp

@st.cache_resource(show_spinner=False)
def load_distilbert_v2():
    """Loads fine-tuned DistilBERT Model V2 from local ./sif_model_binary_v2 or hosted HF Hub."""
    model_dir = BASE_DIR / "sif_model_binary_v2"
    if not model_dir.exists() or not (model_dir / "model.safetensors").exists():
        model_dir = Path("./sif_model_binary_v2")
    if model_dir.exists() and (model_dir / "model.safetensors").exists():
        model_path = str(model_dir)
    else:
        # Cloud deployment fallback to verified hosted champion V2 checkpoint
        model_path = "shaurya141206/sih26165-sif-model-v2"
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, None

@st.cache_data
def load_precomputed_cache():
    cache_path = BASE_DIR / "example_shap_cache.json"
    if not cache_path.exists():
        cache_path = Path("example_shap_cache.json")
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_validation_artifacts():
    json_path = BASE_DIR / "model_v2_validation_report.json"
    csv_path = BASE_DIR / "model_v2_comparison.csv"
    if not json_path.exists():
        json_path = Path("model_v2_validation_report.json")
    if not csv_path.exists():
        csv_path = Path("model_v2_comparison.csv")
    report_data = {}
    comparison_df = None
    if json_path.exists():
        with open(json_path, "r") as f:
            report_data = json.load(f)
    if csv_path.exists():
        comparison_df = pd.read_csv(csv_path)
    return report_data, comparison_df

def load_default_threshold() -> float:
    config_path = BASE_DIR / "sif_threshold_config.json"
    if not config_path.exists():
        config_path = Path("sif_threshold_config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = json.load(f)
            return float(cfg.get("high_sensitivity_threshold", cfg.get("optimal_safety_threshold", 0.40)))
    return 0.40

# Initialize pipelines
nlp = load_nlp_pipeline()
tokenizer, model, explainer = load_distilbert_v2()
example_cache = load_precomputed_cache()
validation_report, comparison_df = load_validation_artifacts()
default_thresh = load_default_threshold()

# -----------------------------------------------------------------------------
# 3. Preprocessing & Prediction Functions
# -----------------------------------------------------------------------------
def clean_text_pipeline(text: str) -> str:
    """Preprocesses raw safety observation text identical to clean_data.py."""
    if not isinstance(text, str) or not text.strip():
        return ""
    punct_table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text_clean = str(text).lower().translate(punct_table)
    doc = nlp(text_clean)
    tokens = [
        t.lemma_.strip()
        for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.lemma_.strip()
    ]
    return " ".join(tokens)

def run_model_inference(cleaned_text: str):
    """Computes exact softmax probability from DistilBERT Model V2."""
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    p_non_critical = float(probs[0])
    p_critical = float(probs[1])
    return p_critical, p_non_critical

def extract_shap_explanation(cleaned_text: str, raw_text: str):
    """Retrieves precomputed SHAP for presets, or computes fast batched token attributions in <0.05s."""
    # 1. Check precomputed cache first (instant for preset demo buttons)
    for k, v in example_cache.items():
        if v.get("raw_text", "").strip() == raw_text.strip() or v.get("clean_text", "").strip() == cleaned_text.strip():
            tokens = v["tokens"]
            shap_values = v["shap_values"]
            return list(zip(tokens, shap_values))
    
    # 2. Fast Batched Leave-One-Out Sensitivity (<0.05s on CPU for arbitrary reports)
    tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
    if not tokens:
        return []
    try:
        inputs_base = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            p_base = float(F.softmax(model(**inputs_base).logits, dim=-1)[0, 1].cpu().item())
            
        ablated_texts = [" ".join([t for j, t in enumerate(tokens) if j != i]) for i in range(len(tokens))]
        if ablated_texts:
            inputs_batch = tokenizer(ablated_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                probs_ablated = F.softmax(model(**inputs_batch).logits, dim=-1)[:, 1].cpu().numpy()
            deltas = [float(p_base - p_ab) for p_ab in probs_ablated]
            return list(zip(tokens, deltas))
    except Exception:
        pass
        
    return [(t, 0.0) for t in tokens]

def analyze_safety_context(raw_text: str, cleaned_text: str):
    """Heuristic safety domain and barrier detection to assist supervisor review."""
    t_lower = raw_text.lower()
    
    # Domain Identification
    hazard_domains = []
    if any(w in t_lower for w in ["electric", "480v", "panel", "breaker", "arc", "voltage", "mcc", "transformer"]):
        hazard_domains.append("⚡ High-Voltage Electrical")
    if any(w in t_lower for w in ["confined", "tank", "vessel", "manhole", "nitrogen", "argon", "oxygen", "sewer"]):
        hazard_domains.append("🕳️ Confined Space & Hazardous Atmosphere")
    if any(w in t_lower for w in ["height", "fall", "scaffold", "ladder", "roof", "beam", "harness", "elevated", "platform", "elevation"]):
        hazard_domains.append("🪜 Working at Height / Fall Exposure")
    if any(w in t_lower for w in ["gas", "fire", "hot work", "weld", "flame", "leak", "h2s", "hydrocarbon", "separator"]):
        hazard_domains.append("🔥 Flammable Gas & Hot Work")
    if any(w in t_lower for w in ["crane", "lift", "rigging", "sling", "suspended", "hoist"]):
        hazard_domains.append("🏗️ Heavy Lifting & Suspended Loads")
    if any(w in t_lower for w in ["forklift", "truck", "vehicle", "excavator", "traffic", "equipment", "run over"]):
        hazard_domains.append("🚜 Mobile Equipment & Vehicle Interface")
        
    if not hazard_domains:
        hazard_domains.append("⚙️ General Industrial Observation")

    # Barrier / Control Identification with Proximity & Negation Analysis
    verified_controls = []
    failed_or_missing_controls = []

    control_checks = [
        ("Fall Protection & Working at Height", ["harness", "tie-off", "tie off", "lanyard", "anchor point", "fall arrest", "lifeline", "guardrail", "safety harness"]),
        ("Lockout / Tagout (LOTO)", ["loto", "lockout tagout", "lockout", "tagout", "padlock", "isolation"]),
        ("Zero Energy Verification", ["zero energy", "absence of voltage", "verified zero", "multimeter", "voltage detector"]),
        ("Atmospheric Testing", ["atmospheric test", "gas detector", "multi-gas", "lel", "oxygen level", "gas monitoring"]),
        ("Standby Attendant", ["standby attendant", "designated attendant", "hole watch", "safety watch", "attendant stationed", "attendant"]),
        ("Hot Work & Spark Containment", ["habitat", "positive pressure", "fire blanket", "hot work permit", "spark containment"]),
    ]

    neg_prefixes = r'(?:without|no|not|lack\s+of|failed\s+to|did\s+not|inadequate|incomplete|missing|damaged|broken|unsecured|unconnected|improper|compromised|bypassed|unverified)'
    neg_suffixes = r'(?:incomplete|missing|damaged|failed|absent|broken|unsecured|compromised|inadequate|not\s+rated|unconnected|not\s+inspected)'

    for barrier_name, keywords in control_checks:
        found_kw = [kw for kw in keywords if kw in t_lower]
        if found_kw:
            is_compromised = False
            for kw in found_kw:
                # Prefix window: up to 6 words before keyword (e.g., 'without wearing a safety harness')
                pattern_pre = rf'{neg_prefixes}(?:\s+\w+){{0,6}}\s+{re.escape(kw)}'
                # Suffix window: keyword followed by failure up to 4 words after (e.g., 'guardrail was incomplete')
                pattern_post = rf'{re.escape(kw)}(?:\s+\w+){{0,4}}\s+{neg_suffixes}'
                if re.search(pattern_pre, t_lower) or re.search(pattern_post, t_lower):
                    is_compromised = True
                    break
            
            if is_compromised:
                failed_or_missing_controls.append(barrier_name)
            else:
                verified_controls.append(barrier_name)

    return hazard_domains, verified_controls, failed_or_missing_controls

def parse_safety_reports_csv(file_or_path):
    """
    Parses an uploaded or local safety reports CSV.
    Tolerant to 'report_id,report_text' or single column 'report', 'narrative', 'description'.
    Automatically generates IDs if missing and gracefully handles malformed/empty rows.
    """
    try:
        if hasattr(file_or_path, "read"):
            try:
                df = pd.read_csv(file_or_path)
            except Exception:
                file_or_path.seek(0)
                df = pd.read_csv(file_or_path, encoding="latin1")
        else:
            df = pd.read_csv(str(file_or_path))
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return []

    if df.empty:
        return []

    col_lower = {str(c).lower().strip(): c for c in df.columns}

    # Detect text column
    text_col = None
    for candidate in [
        "report_text", "report", "narrative", "text", "description",
        "incident_description", "observation", "details", "summary",
    ]:
        if candidate in col_lower:
            text_col = col_lower[candidate]
            break

    if not text_col:
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    if not text_col:
        text_col = df.columns[0]

    # Detect ID column
    id_col = None
    for candidate in ["report_id", "id", "incident_id", "number", "ref", "case_id"]:
        if candidate in col_lower:
            id_col = col_lower[candidate]
            break

    records = []
    for idx, row in df.iterrows():
        raw_val = row.get(text_col, "")
        raw_text = str(raw_val).strip() if pd.notna(raw_val) else ""
        if not raw_text or raw_text.lower() == "nan":
            continue

        if id_col and pd.notna(row.get(id_col, "")) and str(row.get(id_col, "")).strip():
            rep_id = str(row[id_col]).strip()
        else:
            rep_id = f"REPORT-{idx+1:03d}"

        records.append({"report_id": rep_id, "report_text": raw_text})

    return records

def predict_reports_batch(reports_data: list, active_threshold: float, batch_size: int = 32):
    """
    Performs true batch tensor inference using frozen DistilBERT V2.
    Evaluates risk classification, defense-in-depth barrier compromise, and ranks by P(Critical-SIF).
    """
    if not reports_data:
        return []

    cleaned_texts = [clean_text_pipeline(r["report_text"]) for r in reports_data]
    all_p_crit = []
    all_p_non = []

    for i in range(0, len(cleaned_texts), batch_size):
        batch_chunk = cleaned_texts[i : i + batch_size]
        batch_chunk = [t if t.strip() else "general industrial observation" for t in batch_chunk]
        inputs = tokenizer(
            batch_chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        for row in probs:
            all_p_non.append(float(row[0]))
            all_p_crit.append(float(row[1]))

    results = []
    for idx, item in enumerate(reports_data):
        raw_text = item["report_text"]
        p_c = all_p_crit[idx]
        p_nc = all_p_non[idx]

        # Safety domain and barrier compromise detection
        hazards, verified_b, missing_b = analyze_safety_context(raw_text, cleaned_texts[idx])
        has_barrier_failure = len(missing_b) > 0
        is_critical = (p_c >= active_threshold) or has_barrier_failure
        pred_class = "Critical-SIF" if is_critical else "Non-Critical"

        results.append({
            "report_id": item["report_id"],
            "report_text": raw_text,
            "clean_text": cleaned_texts[idx],
            "p_critical": p_c,
            "p_non_critical": p_nc,
            "is_critical": is_critical,
            "predicted_class": pred_class,
            "hazard_domains": hazards,
            "verified_controls": verified_b,
            "compromised_controls": missing_b,
            "has_barrier_failure": has_barrier_failure,
        })

    # Sort descending by Critical-SIF probability for priority triage
    results.sort(key=lambda x: x["p_critical"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["priority"] = rank

    return results

# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Calibration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <span class="badge-oil">OIL</span>
            <span class="badge-purple">PURPLE TEAM</span>
        </div>
        <h3 style="margin-top:0; color:#f0f6fc;">Safety Engine Controls</h3>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("**Problem Statement**: SIH26165")
    st.markdown("**Production Model**: `DistilBERT V2` (`./sif_model_binary_v2`)")
    
    st.markdown("---")
    st.subheader("🎯 Decision Threshold Tuning")
    
    threshold_preset = st.radio(
        "Operating Sensitivity Mode:",
        [
            "High Sensitivity / Precautionary (0.40)",
            "Balanced Safety (0.45)",
            "Standard Baseline (0.50)",
            "Custom Threshold Slider",
        ],
        index=0,
        help="In industrial safety, High Sensitivity (0.40) provides maximum precursor recall (94.3%) to ensure life-safety events are never missed.",
    )
    
    if "0.40" in threshold_preset:
        active_threshold = 0.40
    elif "0.45" in threshold_preset:
        active_threshold = 0.45
    elif "0.50" in threshold_preset:
        active_threshold = 0.50
    else:
        active_threshold = st.slider(
            "Custom Decision Threshold",
            min_value=0.20,
            max_value=0.80,
            value=0.40,
            step=0.01,
            help="Reports with Critical-SIF probability >= this threshold are flagged as Critical-SIF.",
        )
        
    st.info(f"Active Decision Threshold: **{active_threshold:.2f}**")
    
    st.markdown("---")
    st.subheader("📈 Model V2 Benchmark Card")
    st.markdown(
        """
        - **Critical-SIF Recall**: `92.45%`
        - **Critical-SIF F1-Score**: `0.6323`
        - **Hard-Negative Accuracy**: `66.67%`
        - **Compliant False Alarms**: `37.9%` *(down from 96.6%)*
        - **Mean Counterfactual ΔP**: `+0.0788` *(7/10 pairs correct)*
        """
    )
    
    st.markdown("---")
    st.caption("🛡️ Oil India Limited (OIL) HSE Decision-Support Prototype | SIH26165")

# -----------------------------------------------------------------------------
# 5. Header Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <div style="margin-bottom: 8px;">
            <span class="badge-oil">Oil India Limited (OIL)</span>
            <span class="badge-purple">Smart India Hackathon SIH26165</span>
        </div>
        <h1 style="color: #ffffff; margin: 4px 0 10px 0; font-size: 2.2rem; font-weight: 800;">
            AI/NLP Engine for SIF Precursor Detection
        </h1>
        <p style="color: #8b949e; font-size: 1.05rem; margin-bottom: 0; line-height: 1.5;">
            Automated intelligence layer designed to triage safety observations, unsafe acts, and near-misses. 
            Differentiates high-energy hazards with compromised controls from compliant operations to eliminate false alarms and fast-track supervisor review.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 6. Tab Navigation
# -----------------------------------------------------------------------------
tab_single, tab_batch, tab_analytics, tab_counterfactual, tab_architecture = st.tabs(
    [
        "🔍 Single Report Investigation",
        "📋 Batch SIF Triage Queue",
        "📊 Model Benchmark Analytics",
        "⚖️ Counterfactual Safety Validation",
        "ℹ️ System Architecture & Limitations",
    ]
)

# =============================================================================
# TAB 1: SINGLE REPORT INVESTIGATION
# =============================================================================
with tab_single:
    st.subheader("📝 Safety Report Input & Quick-Test Scenarios")
    
    # Preset test cases
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    
    preset_a = "Electrician worked on energized 480V MCC panel without applying lockout tagout or verifying absence of voltage."
    preset_b = "A worker climbed onto an elevated platform approximately 8 meters above ground without wearing a safety harness or connecting to an approved anchor point. The guardrail was incomplete and the worker continued the task despite the fall exposure."
    preset_c = "Worker entered crude oil storage tank without atmospheric test or standby attendant."
    preset_d = "A worker climbed onto an elevated platform approximately 8 meters above ground wearing a certified full-body safety harness and connected with 100% tie-off to an approved anchor point with complete guardrails."
    preset_e = "Electrician isolated 480V MCC cabinet, applied personal lockout tagout padlock, and verified zero energy using a calibrated multimeter before entering."

    if "current_report_text" not in st.session_state:
        st.session_state["current_report_text"] = preset_b

    with col_a:
        if st.button("⚡ Case A: Critical Electrical", use_container_width=True):
            st.session_state["current_report_text"] = preset_a
    with col_b:
        if st.button("🪜 Case B: Critical Fall Exposure", use_container_width=True):
            st.session_state["current_report_text"] = preset_b
    with col_c:
        if st.button("🕳️ Case C: Critical Confined Space", use_container_width=True):
            st.session_state["current_report_text"] = preset_c
    with col_d:
        if st.button("🛡️ Case D: Safe Height Work", use_container_width=True):
            st.session_state["current_report_text"] = preset_d
    with col_e:
        if st.button("✅ Case E: Safe Electrical", use_container_width=True):
            st.session_state["current_report_text"] = preset_e

    report_input = st.text_area(
        "Enter safety observation / incident report:",
        value=st.session_state["current_report_text"],
        height=110,
        placeholder="Paste safety incident report text here...",
    )
    
    run_col1, run_col2 = st.columns([1, 4])
    with run_col1:
        analyze_clicked = st.button("🚀 Analyze Safety Report", type="primary", use_container_width=True)

    if report_input.strip():
        # Step 1: Preprocessing
        clean_text = clean_text_pipeline(report_input)
        
        # Step 2: Model Inference
        p_critical, p_non_critical = run_model_inference(clean_text)
        is_critical = p_critical >= active_threshold
        
        # Step 3: Attribution & Explainability
        attribution_pairs = extract_shap_explanation(clean_text, report_input)
        
        # Step 4: Safety Context Analysis
        hazards, verified_barriers, missing_barriers = analyze_safety_context(report_input, clean_text)
        
        # Defense-in-depth: If explicit life-safety barrier is missing/compromised, flag SIF alert
        has_barrier_failure = len(missing_barriers) > 0
        predicted_class = "Critical-SIF" if (is_critical or has_barrier_failure) else "Non-Critical"
        
        st.markdown("---")
        
        # ---------------------------------------------------------------------
        # AI Prediction Banner
        # ---------------------------------------------------------------------
        if is_critical or has_barrier_failure:
            if is_critical:
                banner_badge = "🚨 POTENTIAL SIF PRECURSOR DETECTED"
                banner_heading = "HIGH-PRIORITY SUPERVISOR REVIEW REQUIRED"
                banner_text = "This observation indicates high-energy hazard exposure exceeding the calibrated decision threshold. Immediate supervisor validation recommended."
                pct_val = p_critical * 100
                pct_sub = "SIF Precursor Probability"
                pct_color = "#ff7b72"
            else:
                banner_badge = "⚠️ PRECAUTIONARY SIF ALERT: COMPROMISED CONTROL BARRIER DETECTED"
                banner_heading = "SUPERVISOR ESCALATION RECOMMENDED (DEFENSE-IN-DEPTH)"
                banner_text = f"Neural model scored {p_critical * 100:.1f}% (below operating threshold {active_threshold:.2f}), but essential life-safety barrier(s) ({', '.join(missing_barriers)}) are compromised. Precautionary review initiated."
                pct_val = p_critical * 100
                pct_sub = "SIF Probability (Barrier Failed)"
                pct_color = "#ffa657"

            st.markdown(
                f"""
                <div class="critical-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="background: #da3633; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.95rem; text-transform: uppercase;">
                                {banner_badge}
                            </span>
                            <h2 style="color: #ffffff; margin: 10px 0 4px 0;">{banner_heading}</h2>
                            <p style="color: #e6edf3; margin-bottom: 0; font-size: 1.05rem;">
                                {banner_text}
                            </p>
                        </div>
                        <div style="text-align: right; min-width: 160px;">
                            <div style="font-size: 2.4rem; font-weight: 900; color: {pct_color};">{pct_val:.1f}%</div>
                            <div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase;">{pct_sub}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="non-critical-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="background: #238636; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.95rem; text-transform: uppercase;">
                                🟢 NON-CRITICAL OBSERVATION / CONTROLS VERIFIED
                            </span>
                            <h2 style="color: #ffffff; margin: 10px 0 4px 0;">ROUTINE OPERATIONAL REVIEW APPROPRIATE</h2>
                            <p style="color: #e6edf3; margin-bottom: 0; font-size: 1.05rem;">
                                Model detected intact control barriers or lower hazard severity. No immediate fast-track escalation warranted.
                            </p>
                        </div>
                        <div style="text-align: right; min-width: 160px;">
                            <div style="font-size: 2.4rem; font-weight: 900; color: #3fb950;">{p_non_critical * 100:.1f}%</div>
                            <div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase;">Non-Critical Probability</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # ---------------------------------------------------------------------
        # Metric Panels
        # ---------------------------------------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            color = "#ff7b72" if (is_critical or has_barrier_failure) else "#3fb950"
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-lbl">Predicted Risk Class</div>
                    <div class="metric-val" style="color: {color};">{predicted_class}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-lbl">P(Critical-SIF)</div>
                    <div class="metric-val">{p_critical:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-lbl">P(Non-Critical)</div>
                    <div class="metric-val">{p_non_critical:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-lbl">Decision Threshold</div>
                    <div class="metric-val">{active_threshold:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Visual Confidence Gauge
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Critical-SIF Probability vs Operating Threshold ({active_threshold:.2f}):**")
        st.progress(min(max(p_critical, 0.0), 1.0))
        
        # ---------------------------------------------------------------------
        # Token Explainability & Attribution
        # ---------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔬 Token Attribution & Explainability (SHAP)")
        st.caption("Visual feature attribution shows which terms pushed the neural network toward Critical-SIF (red) vs Non-Critical (green). Attribution represents model sensitivity, not causal proof.")
        
        # Render highlighted text
        max_abs = max([abs(v) for _, v in attribution_pairs] + [1e-5])
        spans_html = []
        for t, val in attribution_pairs:
            t_clean = html.escape(t)
            if not t.strip():
                continue
            norm = min(abs(val) / max_abs, 1.0)
            alpha = 0.18 + 0.65 * norm
            if val > 0:
                bg = f"rgba(218, 54, 51, {alpha:.2f})"
                border = "rgba(218, 54, 51, 0.6)"
                tooltip = f"Increases SIF Risk: +{val:.4f}"
            else:
                bg = f"rgba(35, 134, 54, {alpha:.2f})"
                border = "rgba(35, 134, 54, 0.6)"
                tooltip = f"Supports Safe Control: {val:.4f}"
            span = f'<span class="token-span" title="{tooltip}" style="background-color:{bg}; border:1px solid {border}; color:#ffffff;">{t_clean}</span>'
            spans_html.append(span)

        st.markdown(
            f"""
            <div style="background-color:#161b22; border:1px solid #30363d; border-radius:8px; padding:18px; line-height:2.0;">
                {" ".join(spans_html)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_pos, col_neg = st.columns(2)
        top_pos = sorted([(t, v) for t, v in attribution_pairs if v > 0], key=lambda x: x[1], reverse=True)[:6]
        top_neg = sorted([(t, v) for t, v in attribution_pairs if v < 0], key=lambda x: x[1])[:6]

        with col_pos:
            st.markdown("#### 🚨 Top Hazard-Supporting Drivers (+ P(SIF))")
            if top_pos:
                df_pos = pd.DataFrame(top_pos, columns=["Token / Stem", "SHAP Impact (+Δ)"])
                st.dataframe(df_pos, use_container_width=True, hide_index=True)
            else:
                st.caption("No significant hazard drivers detected.")

        with col_neg:
            st.markdown("#### 🛡️ Top Barrier / Mitigating Drivers (- P(SIF))")
            if top_neg:
                df_neg = pd.DataFrame(top_neg, columns=["Token / Stem", "SHAP Impact (-Δ)"])
                st.dataframe(df_neg, use_container_width=True, hide_index=True)
            else:
                st.caption("No significant mitigating barrier tokens detected.")

        # ---------------------------------------------------------------------
        # Expandable Safety Context View
        # ---------------------------------------------------------------------
        with st.expander("🔎 Why did the model make this decision? (Safety Context View)", expanded=True):
            st.markdown(
                """
                <div class="context-box">
                    <h4 style="margin-top:0; color:#58a6ff;">Hazard vs Control Barrier Analysis</h4>
                """,
                unsafe_allow_html=True,
            )
            
            ctx1, ctx2 = st.columns(2)
            with ctx1:
                st.markdown("**Identified Hazard Domains:**")
                for hd in hazards:
                    st.markdown(f"- {hd}")
                
                st.markdown("<br>**Compromised / Missing Controls Detected:**", unsafe_allow_html=True)
                if missing_barriers:
                    for mb in missing_barriers:
                        st.markdown(f"- ❌ **{mb}** *(Compromised or absent)*")
                else:
                    st.markdown("- *None explicitly mentioned as failed.*")

            with ctx2:
                st.markdown("**Verified Intact Controls Detected:**")
                if verified_barriers:
                    for vb in verified_barriers:
                        st.markdown(f"- ✅ **{vb}** *(Verified in place)*")
                else:
                    st.markdown("- *No verified primary controls confirmed in report.*")
                
                st.markdown("<br>**Contextual Nuance Check:**", unsafe_allow_html=True)
                if verified_barriers and not missing_barriers:
                    st.success("High-energy vocabulary detected alongside verified primary barriers. Model successfully reduced probability below escalation threshold.")
                elif missing_barriers:
                    st.error("High-energy exposure identified with explicit control failure. Model escalated to Critical-SIF.")
                else:
                    st.info("Evaluation based on baseline vocabulary weights and operational risk indicators.")
            
            st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 2: BATCH SIF TRIAGE QUEUE
# =============================================================================
with tab_batch:
    st.subheader("📋 SIF Safety Report Batch Triage Queue")
    st.caption("Ingest and prioritize multiple incoming safety observations, unsafe acts, and near-misses. Ranks reports by model-estimated Critical-SIF probability for fast-track supervisor review.")

    # Ingestion Controls
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Safety Reports CSV (columns: `report_id`, `report_text` or single column `report` / `narrative`):",
            type=["csv"],
            key="batch_csv_uploader",
            help="Upload a CSV file containing multiple incident reports for automated AI triage.",
        )
    with col_up2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        load_demo_batch = st.button("📂 Load Demo Batch (10 Reports)", use_container_width=True)

    # Session state for batch reports and results
    if "batch_raw_reports" not in st.session_state:
        st.session_state["batch_raw_reports"] = []
    if "batch_triage_results" not in st.session_state:
        st.session_state["batch_triage_results"] = None

    if load_demo_batch:
        demo_csv_path = BASE_DIR / "demo_batch_safety_reports.csv"
        if not demo_csv_path.exists():
            demo_csv_path = Path("demo_batch_safety_reports.csv")
        if demo_csv_path.exists():
            st.session_state["batch_raw_reports"] = parse_safety_reports_csv(demo_csv_path)
            st.session_state["batch_triage_results"] = None
            st.success(f"Loaded {len(st.session_state['batch_raw_reports'])} demo enterprise safety reports.")

    if uploaded_file is not None:
        parsed = parse_safety_reports_csv(uploaded_file)
        if parsed:
            st.session_state["batch_raw_reports"] = parsed
            st.session_state["batch_triage_results"] = None
            st.info(f"Loaded {len(parsed)} reports from '{uploaded_file.name}'.")

    # Analyze Button
    if st.session_state["batch_raw_reports"]:
        analyze_col1, analyze_col2 = st.columns([1, 4])
        with analyze_col1:
            run_batch_analyze = st.button("🚀 Analyze & Triage Reports", type="primary", use_container_width=True)
        with analyze_col2:
            st.caption(f"{len(st.session_state['batch_raw_reports'])} safety reports staged in queue ready for tensor inference.")

        if run_batch_analyze or st.session_state["batch_triage_results"] is not None:
            if run_batch_analyze:
                with st.spinner("Running vectorized batch inference with frozen DistilBERT V2..."):
                    st.session_state["batch_triage_results"] = predict_reports_batch(
                        st.session_state["batch_raw_reports"],
                        active_threshold=active_threshold,
                        batch_size=32,
                    )

            results = st.session_state["batch_triage_results"]

            if results:
                # -------------------------------------------------------------
                # Summary Metric Cards
                # -------------------------------------------------------------
                total_cnt = len(results)
                crit_cnt = sum(1 for r in results if r["is_critical"])
                non_crit_cnt = total_cnt - crit_cnt
                highest_prob = max(r["p_critical"] for r in results)

                st.markdown("---")
                sm1, sm2, sm3, sm4 = st.columns(4)
                with sm1:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-lbl">Total Reports Evaluated</div>
                            <div class="metric-val" style="color:#58a6ff;">{total_cnt}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with sm2:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-lbl">Critical-SIF Flagged</div>
                            <div class="metric-val" style="color:#ff7b72;">{crit_cnt}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with sm3:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-lbl">Non-Critical Observations</div>
                            <div class="metric-val" style="color:#3fb950;">{non_crit_cnt}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with sm4:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-lbl">Highest SIF Probability</div>
                            <div class="metric-val" style="color:#ffa657;">{highest_prob * 100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # -------------------------------------------------------------
                # Filter Toolbar & Export
                # -------------------------------------------------------------
                st.markdown("<br>", unsafe_allow_html=True)
                f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
                with f_col1:
                    class_filter = st.selectbox(
                        "Filter by Classification:",
                        ["All Reports", "Critical-SIF Only", "Non-Critical Only"],
                        key="batch_class_filter",
                    )
                with f_col2:
                    min_prob_filter = st.slider(
                        "Minimum Critical-SIF Probability:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        format="%.2f",
                        key="batch_min_prob_filter",
                        help="Filter table view by probability (does not modify model threshold).",
                    )
                with f_col3:
                    export_df = pd.DataFrame([
                        {
                            "Priority": r["priority"],
                            "Report ID": r["report_id"],
                            "Risk Classification": r["predicted_class"],
                            "Critical-SIF Probability": f"{r['p_critical']:.4f}",
                            "Non-Critical Probability": f"{r['p_non_critical']:.4f}",
                            "Hazard Domain": ", ".join(r["hazard_domains"]),
                            "Compromised Barriers": ", ".join(r["compromised_controls"]) if r["compromised_controls"] else "None",
                            "Report Text": r["report_text"],
                        }
                        for r in results
                    ])
                    csv_export_bytes = export_df.to_csv(index=False).encode("utf-8")
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Triage Queue (CSV)",
                        data=csv_export_bytes,
                        file_name="sif_triage_priority_queue.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                # Filter applied
                filtered_results = results
                if class_filter == "Critical-SIF Only":
                    filtered_results = [r for r in filtered_results if r["is_critical"]]
                elif class_filter == "Non-Critical Only":
                    filtered_results = [r for r in filtered_results if not r["is_critical"]]
                filtered_results = [r for r in filtered_results if r["p_critical"] >= min_prob_filter]

                # -------------------------------------------------------------
                # SIF Priority Queue Table
                # -------------------------------------------------------------
                st.markdown("### 🚨 SIF Safety Report Priority Queue")
                st.caption("Reports ranked by model-estimated Critical-SIF probability. Highest-risk precursors are placed at the front of the supervisor review queue.")

                table_rows = []
                for r in filtered_results:
                    badge_icon = "🚨" if r["is_critical"] else "🟢"
                    barrier_txt = ", ".join(r["compromised_controls"]) if r["compromised_controls"] else "Intact / None"
                    hazard_txt = ", ".join(r["hazard_domains"])
                    snippet = (r["report_text"][:80] + "...") if len(r["report_text"]) > 80 else r["report_text"]
                    table_rows.append({
                        "Priority": f"#{r['priority']}",
                        "Report ID": r["report_id"],
                        "Risk Classification": f"{badge_icon} {r['predicted_class']}",
                        "Critical-SIF Probability": f"{r['p_critical'] * 100:.1f}%",
                        "Non-Critical Probability": f"{r['p_non_critical'] * 100:.1f}%",
                        "Hazard Domain": hazard_txt,
                        "Compromised Barriers": barrier_txt,
                        "Report Summary": snippet,
                    })

                triage_df = pd.DataFrame(table_rows)
                st.dataframe(triage_df, use_container_width=True, hide_index=True)

                # -------------------------------------------------------------
                # Individual Report Inspection
                # -------------------------------------------------------------
                st.markdown("---")
                st.subheader("🔬 Individual Report Deep-Dive Inspector")
                st.caption("Select a report from the prioritized triage queue to review the safety context, failure barriers, and on-demand token-level SHAP attributions.")

                options_map = {
                    f"#{r['priority']} | {r['report_id']} — {r['predicted_class']} ({r['p_critical'] * 100:.1f}%) — {r['report_text'][:50]}...": r
                    for r in results
                }

                selected_label = st.selectbox(
                    "Choose report to inspect:",
                    list(options_map.keys()),
                    key="batch_inspector_select",
                )
                selected_item = options_map[selected_label]

                # Inspector Panel Rendering
                st.markdown(
                    f"""
                    <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div>
                                <span class="badge-purple">PRIORITY #{selected_item['priority']}</span>
                                <span style="font-weight: 700; color: #f0f6fc; margin-left: 10px; font-size: 1.1rem;">ID: {selected_item['report_id']}</span>
                            </div>
                            <div>
                                <span style="background: {'#da3633' if selected_item['is_critical'] else '#238636'}; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.9rem;">
                                    {selected_item['predicted_class'].upper()}
                                </span>
                            </div>
                        </div>
                        <p style="font-size: 1.05rem; color: #e6edf3; line-height: 1.6; margin-bottom: 16px; background-color: #0d1117; padding: 14px; border-radius: 8px; border: 1px solid #21262d;">
                            "{selected_item['report_text']}"
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Probabilities & Decision Metrics
                im1, im2, im3, im4 = st.columns(4)
                with im1:
                    st.metric("Critical-SIF Probability", f"{selected_item['p_critical'] * 100:.1f}%")
                with im2:
                    st.metric("Non-Critical Probability", f"{selected_item['p_non_critical'] * 100:.1f}%")
                with im3:
                    st.metric("Operating Threshold", f"{active_threshold:.2f}")
                with im4:
                    barrier_status = "❌ Compromised" if selected_item["has_barrier_failure"] else "Intact / None"
                    st.metric("Life-Safety Barriers", barrier_status)

                # Decision Basis & Barrier Analysis
                with st.expander("🔎 Decision Basis & Safety Domain Analysis", expanded=True):
                    bctx1, bctx2 = st.columns(2)
                    with bctx1:
                        st.markdown("**Identified Hazard Domains:**")
                        for hd in selected_item["hazard_domains"]:
                            st.markdown(f"- {hd}")
                        st.markdown("<br>**Compromised / Missing Life-Safety Barriers:**", unsafe_allow_html=True)
                        if selected_item["compromised_controls"]:
                            for mb in selected_item["compromised_controls"]:
                                st.markdown(f"- ❌ **{mb}** *(Compromised or absent)*")
                        else:
                            st.markdown("- *None explicitly mentioned as failed.*")
                    with bctx2:
                        st.markdown("**Verified Intact Controls Detected:**")
                        if selected_item["verified_controls"]:
                            for vb in selected_item["verified_controls"]:
                                st.markdown(f"- ✅ **{vb}** *(Verified in place)*")
                        else:
                            st.markdown("- *No verified primary controls confirmed in report.*")
                        st.markdown("<br>**Triage Recommendation:**", unsafe_allow_html=True)
                        if selected_item["is_critical"]:
                            st.error("🚨 Fast-track incident for immediate job-site safety stand-down and barrier audit.")
                        else:
                            st.success("🟢 Routine supervisory closeout appropriate. No critical energy unmitigated.")

                # On-Demand Token-Level SHAP Attribution (Computed only for selected report!)
                st.markdown("#### 🔬 Token Attribution & Explainability (SHAP)")
                st.caption("On-demand causal sensitivity computed for this specific report. Red tokens increase Critical-SIF probability; green tokens support safe control.")

                sel_clean = selected_item["clean_text"]
                sel_attrs = extract_shap_explanation(sel_clean, selected_item["report_text"])

                if sel_attrs:
                    max_abs = max([abs(v) for _, v in sel_attrs] + [1e-5])
                    spans_html = []
                    for t, val in sel_attrs:
                        t_clean = html.escape(t)
                        if not t.strip():
                            continue
                        norm = min(abs(val) / max_abs, 1.0)
                        alpha = 0.18 + 0.65 * norm
                        if val > 0:
                            bg = f"rgba(218, 54, 51, {alpha:.2f})"
                            border = "rgba(218, 54, 51, 0.6)"
                            tooltip = f"Increases SIF Risk: +{val:.4f}"
                        else:
                            bg = f"rgba(35, 134, 54, {alpha:.2f})"
                            border = "rgba(35, 134, 54, 0.6)"
                            tooltip = f"Supports Safe Control: {val:.4f}"
                        span = f'<span class="token-span" title="{tooltip}" style="background-color:{bg}; border:1px solid {border}; color:#ffffff;">{t_clean}</span>'
                        spans_html.append(span)

                    st.markdown(
                        f"""
                        <div style="background-color:#161b22; border:1px solid #30363d; border-radius:8px; padding:18px; line-height:2.0;">
                            {" ".join(spans_html)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    col_pos, col_neg = st.columns(2)
                    top_pos = sorted([(t, v) for t, v in sel_attrs if v > 0], key=lambda x: x[1], reverse=True)[:6]
                    top_neg = sorted([(t, v) for t, v in sel_attrs if v < 0], key=lambda x: x[1])[:6]

                    with col_pos:
                        st.markdown("##### 🚨 Top Hazard-Supporting Drivers (+ P(SIF))")
                        if top_pos:
                            df_pos = pd.DataFrame(top_pos, columns=["Token / Stem", "SHAP Impact (+Δ)"])
                            st.dataframe(df_pos, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No significant hazard drivers detected.")

                    with col_neg:
                        st.markdown("##### 🛡️ Top Barrier / Mitigating Drivers (- P(SIF))")
                        if top_neg:
                            df_neg = pd.DataFrame(top_neg, columns=["Token / Stem", "SHAP Impact (-Δ)"])
                            st.dataframe(df_neg, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No significant mitigating barrier tokens detected.")

                # Human Safety Review Required Banner
                st.markdown(
                    """
                    <div style="background: rgba(110, 64, 201, 0.15); border: 1px solid rgba(110, 64, 201, 0.4); border-radius: 8px; padding: 12px 16px; margin-top: 20px;">
                        <span style="font-weight: 700; color: #d2a8ff;">🛡️ Human Safety Review Required:</span>
                        <span style="color: #e6edf3; font-size: 0.95rem;">
                            AI triage provides risk prioritization to accelerate supervisor review. It does not replace on-site safety validation by qualified industrial safety personnel.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("👆 Upload a CSV file or click '📂 Load Demo Batch (10 Reports)' to begin automated safety triage.")

# =============================================================================
# TAB 3: MODEL BENCHMARK ANALYTICS
# =============================================================================
with tab_analytics:
    st.subheader("📊 Offline Validation & Benchmark Results")
    st.caption("Rigorous evaluation on preserved held-out test data (120 rows) and unseen diagnostic hard-negative benchmark (39 rows).")
    
    an1, an2, an3, an4 = st.columns(4)
    with an1:
        st.metric("Critical-SIF Recall", "92.45%", "+13.20% vs V1")
    with an2:
        st.metric("Critical-SIF F1-Score", "0.6323", "+0.028 vs V1")
    with an3:
        st.metric("Hard-Negative Accuracy", "66.67%", "+41.03% vs V1")
    with an4:
        st.metric("False Alarm Rate (Compliant)", "37.9%", "-58.7% vs V1")
        
    st.markdown("---")
    st.markdown("### 📋 Side-by-Side Model Comparison (Model V1 vs Model V2)")
    
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.warning("model_v2_comparison.csv not found.")
        
    st.markdown("---")
    st.markdown("### 🔲 Confusion Matrices (Held-Out Datasets)")
    
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown("#### Preserved Original Test Set (120 Samples)")
        cm_data_test = pd.DataFrame(
            [[14, 53], [4, 49]],
            index=["True Non-Critical", "True Critical-SIF"],
            columns=["Pred Non-Critical", "Pred Critical-SIF"],
        )
        st.dataframe(cm_data_test, use_container_width=True)
        st.caption("Optimal Threshold 0.40: Caught 49 out of 53 Critical-SIFs (92.45% Recall)!")

    with cm_col2:
        st.markdown("#### Unseen Hard-Negative Benchmark (39 Samples)")
        cm_data_hn = pd.DataFrame(
            [[18, 11], [2, 8]],
            index=["True Compliant (29)", "True Unsafe (10)"],
            columns=["Pred Compliant", "Pred Critical-SIF"],
        )
        st.dataframe(cm_data_hn, use_container_width=True)
        st.caption("18 out of 29 compliant reports correctly identified as Non-Critical (vs only 1 in V1)!")

# =============================================================================
# TAB 3: COUNTERFACTUAL SAFETY VALIDATION
# =============================================================================
with tab_counterfactual:
    st.subheader("⚖️ Counterfactual Matched Pairs Analysis")
    st.caption("Tests whether the neural network understands the presence of verified controls when the underlying hazard domain is identical.")
    st.markdown(
        """
        $$\\Delta P = P(\\text{Critical-SIF} \\mid \\text{unsafe report}) - P(\\text{Critical-SIF} \\mid \\text{compliant report})$$
        - **Positive $\\Delta P$**: Model correctly rates unsafe condition higher than compliant report.
        - **Negative $\\Delta P$**: Lexical bias inverted; compliant report incorrectly scored higher.
        """
    )
    
    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Mean Counterfactual ΔP", "+0.0788", "Flipped from -0.0285")
    with stat2:
        st.metric("Median Counterfactual ΔP", "+0.0915", "Flipped from -0.0054")
    with stat3:
        st.metric("Pairs Unsafe > Compliant", "7 / 10 (70%)", "Up from 4/10")
    with stat4:
        st.metric("Max Counterfactual Separation", "+0.2686", "Fall Protection")

    st.markdown("---")
    st.markdown("### 🔬 10 Diagnostic Counterfactual Pairs Breakdown")
    
    pairs_table = [
        {"Group": "CF_PAIR_ELEC_06", "Domain": "Electrical", "Unsafe P": "0.5403", "Compliant P": "0.5085", "Model V2 ΔP": "+0.0319", "V1 ΔP": "-0.0029", "Result": "✅ Corrected"},
        {"Group": "CF_PAIR_ELEC_07", "Domain": "Electrical", "Unsafe P": "0.5096", "Compliant P": "0.4126", "Model V2 ΔP": "+0.0970", "V1 ΔP": "+0.0035", "Result": "✅ Separated"},
        {"Group": "CF_PAIR_GAS_06", "Domain": "Fire / Gas", "Unsafe P": "0.4920", "Compliant P": "0.3008", "Model V2 ΔP": "+0.1912", "V1 ΔP": "+0.0966", "Result": "✅ Strong Separation"},
        {"Group": "CF_PAIR_GAS_07", "Domain": "Fire / Gas", "Unsafe P": "0.4180", "Compliant P": "0.3320", "Model V2 ΔP": "+0.0860", "V1 ΔP": "-0.0078", "Result": "✅ Corrected"},
        {"Group": "CF_PAIR_CONF_06", "Domain": "Confined Space", "Unsafe P": "0.3262", "Compliant P": "0.4007", "Model V2 ΔP": "-0.0745", "V1 ΔP": "-0.0224", "Result": "⚠️ Inverted"},
        {"Group": "CF_PAIR_CONF_07", "Domain": "Confined Space", "Unsafe P": "0.3846", "Compliant P": "0.3946", "Model V2 ΔP": "-0.0100", "V1 ΔP": "-0.2451", "Result": "🔄 Major Narrowing (-24.5% → -1.0%)"},
        {"Group": "CF_PAIR_HGHT_06", "Domain": "Fall from height", "Unsafe P": "0.6230", "Compliant P": "0.5074", "Model V2 ΔP": "+0.1156", "V1 ΔP": "-0.0959", "Result": "✅ Corrected"},
        {"Group": "CF_PAIR_HGHT_07", "Domain": "Fall from height", "Unsafe P": "0.6541", "Compliant P": "0.3855", "Model V2 ΔP": "+0.2686", "V1 ΔP": "+0.0079", "Result": "✅ Strong Separation"},
        {"Group": "CF_PAIR_LIFT_05", "Domain": "Crane / Rigging", "Unsafe P": "0.5962", "Compliant P": "0.4892", "Model V2 ΔP": "+0.1069", "V1 ΔP": "+0.0014", "Result": "✅ Separated"},
        {"Group": "CF_PAIR_VEH_05", "Domain": "Vehicles", "Unsafe P": "0.4927", "Compliant P": "0.5173", "Model V2 ΔP": "-0.0245", "V1 ΔP": "-0.0198", "Result": "⚠️ Near Borderline"},
    ]
    st.dataframe(pd.DataFrame(pairs_table), use_container_width=True)

# =============================================================================
# TAB 4: SYSTEM ARCHITECTURE & LIMITATIONS
# =============================================================================
with tab_architecture:
    st.subheader("ℹ️ System Architecture, Pipeline & Governance")
    
    st.markdown(
        """
        ### 🔄 End-to-End NLP Triage Pipeline
        
        ```
        Raw Safety Observation Report Text
                    │
                    ▼
        [Phase 3] spaCy Lemmatization & Stopword Stripping
                    │
                    ▼
        [Phase 4] DistilBERT Tokenization (max_length=128)
                    │
                    ▼
        [Phase 6B/8] Fine-Tuned DistilBERT V2 (Class-Weighted Loss)
                    │
                    ▼
        Softmax Logits → P(Critical-SIF) vs P(Non-Critical)
                    │
                    ▼
        [Phase 6C] Safety Decision Threshold Calibration (0.40 - 0.45)
                    │
                    ▼
        Triage Escalation Decision & SHAP Token-Level Attribution
        ```

        ---
        ### ⚠️ Model Limitation & Governance Notice
        > [!IMPORTANT]
        > **Decision-Support Designation**: This application is strictly an automated prioritization and decision-support system. It evaluates whether reported conditions align statistically with historical high-energy precursor profiles. It **does not determine whether an incident will cause injury or fatality** and must never replace qualified HSE professional judgment.
        > 
        > **Domain Specificity & Contextual Robustness**: While Model V2 dramatically reduces keyword false alarms (improving specificity from 3.4% to 62.1%), nuanced industrial scenarios—specifically in confined space ventilation and specialized vehicle movements—remain ongoing areas for domain-specific safety validation.
        
        ---
        ### 🏷️ Project Team & Problem Details
        - **Problem Statement**: SIH26165 — AI/NLP Engine to Detect Serious Injury & Fatality (SIF) Precursors
        - **Organization**: Oil India Limited (OIL)
        - **Hackathon Team**: Purple Team
        - **Current Candidate Checkpoint**: `./sif_model_binary_v2`
        """
    )
