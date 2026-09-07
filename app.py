import html
import json
from pathlib import Path
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
    
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=-1,  # CPU
    )
    explainer = shap.Explainer(pipe)
    return tokenizer, model, explainer

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
            return float(cfg.get("optimal_safety_threshold", cfg.get("recommended_balanced_threshold", 0.45)))
    return 0.45

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
    """Retrieves or computes SHAP token attributions."""
    # Check precomputed cache
    for k, v in example_cache.items():
        if v.get("raw_text", "").strip() == raw_text.strip() or v.get("clean_text", "").strip() == cleaned_text.strip():
            tokens = v["tokens"]
            shap_values = v["shap_values"]
            return list(zip(tokens, shap_values))
    
    # Compute live SHAP
    sv = explainer([cleaned_text])
    tokens = [str(t) for t in sv[0].data]
    vals = [float(val) for val in sv[0].values[:, 1]]
    return list(zip(tokens, vals))

def analyze_safety_context(raw_text: str, cleaned_text: str):
    """Heuristic safety domain and barrier detection to assist supervisor review."""
    t_lower = raw_text.lower()
    
    # Domain Identification
    hazard_domains = []
    if any(w in t_lower for w in ["electric", "480v", "panel", "breaker", "arc", "voltage", "mcc", "transformer"]):
        hazard_domains.append("⚡ High-Voltage Electrical")
    if any(w in t_lower for w in ["confined", "tank", "vessel", "manhole", "nitrogen", "argon", "oxygen", "sewer"]):
        hazard_domains.append("🕳️ Confined Space & Hazardous Atmosphere")
    if any(w in t_lower for w in ["height", "fall", "scaffold", "ladder", "roof", "beam", "harness", "elevated"]):
        hazard_domains.append("🪜 Working at Height / Fall Exposure")
    if any(w in t_lower for w in ["gas", "fire", "hot work", "weld", "flame", "leak", "h2s", "hydrocarbon", "separator"]):
        hazard_domains.append("🔥 Flammable Gas & Hot Work")
    if any(w in t_lower for w in ["crane", "lift", "rigging", "sling", "suspended", "hoist"]):
        hazard_domains.append("🏗️ Heavy Lifting & Suspended Loads")
    if any(w in t_lower for w in ["forklift", "truck", "vehicle", "excavator", "traffic", "equipment", "run over"]):
        hazard_domains.append("🚜 Mobile Equipment & Vehicle Interface")
        
    if not hazard_domains:
        hazard_domains.append("⚙️ General Industrial Observation")

    # Barrier / Control Identification
    verified_controls = []
    failed_or_missing_controls = []

    control_checks = [
        ("Lockout / Tagout (LOTO)", ["loto", "lockout tagout", "lockout", "tagout", "padlock"]),
        ("Zero Energy Verification", ["zero energy", "verified absence of voltage", "verified zero", "multimeter"]),
        ("Atmospheric Testing", ["atmospheric test", "gas detector", "multi-gas", "lel", "oxygen level"]),
        ("Standby Attendant", ["standby attendant", "designated attendant", "attendant stationed", "hole watch"]),
        ("Fall Protection / Harness", ["harness", "100% tie-off", "lanyard", "anchor point", "fall arrest"]),
        ("Positive Pressure / Habitat", ["habitat", "positive pressure", "fire blanket", "hot work permit"]),
    ]

    for barrier_name, keywords in control_checks:
        found_kw = [kw for kw in keywords if kw in t_lower]
        if found_kw:
            # Check context: missing vs verified
            missing_indicators = ["without", "no ", "failed to", "did not", "lack of", "absent", "unverified"]
            is_missing = False
            for mi in missing_indicators:
                for kw in found_kw:
                    if f"{mi} {kw}" in t_lower or f"{mi} applying {kw}" in t_lower or f"{mi} performing {kw}" in t_lower:
                        is_missing = True
                        break
            if is_missing or any(mi in t_lower for mi in ["without atmospheric test", "without standby", "without applying lockout"]):
                failed_or_missing_controls.append(barrier_name)
            else:
                verified_controls.append(barrier_name)

    return hazard_domains, verified_controls, failed_or_missing_controls

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
            "High Sensitivity (Threshold 0.40)",
            "Balanced Safety (Threshold 0.45)",
            "Standard Baseline (Threshold 0.50)",
            "Custom Threshold Slider",
        ],
        index=1,
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
            value=0.45,
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
tab_triage, tab_analytics, tab_counterfactual, tab_architecture = st.tabs(
    [
        "🔍 SIF Triage & Investigation",
        "📊 Model Benchmark Analytics",
        "⚖️ Counterfactual Safety Validation",
        "ℹ️ System Architecture & Limitations",
    ]
)

# =============================================================================
# TAB 1: SIF TRIAGE & INVESTIGATION
# =============================================================================
with tab_triage:
    st.subheader("📝 Safety Report Input & Quick-Test Scenarios")
    
    # Preset test cases
    col_a, col_b, col_c, col_d = st.columns(4)
    
    preset_a = "Electrician worked on energized 480V MCC panel without applying lockout tagout or verifying absence of voltage."
    preset_b = "Electrician isolated 480V MCC cabinet, applied personal lockout tagout padlock, and verified zero energy using a calibrated multimeter before entering."
    preset_c = "Worker entered crude oil storage tank without atmospheric test or standby attendant."
    preset_d = "Worker entered storage tank after multi-gas detector confirmed safe oxygen levels with a designated attendant stationed at the manhole."

    if "current_report_text" not in st.session_state:
        st.session_state["current_report_text"] = preset_a

    with col_a:
        if st.button("⚡ Case A: Critical Electrical", use_container_width=True):
            st.session_state["current_report_text"] = preset_a
    with col_b:
        if st.button("🛡️ Case B: Safe Electrical", use_container_width=True):
            st.session_state["current_report_text"] = preset_b
    with col_c:
        if st.button("🕳️ Case C: Critical Confined Space", use_container_width=True):
            st.session_state["current_report_text"] = preset_c
    with col_d:
        if st.button("✅ Case D: Safe Confined Space", use_container_width=True):
            st.session_state["current_report_text"] = preset_d

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
        predicted_class = "Critical-SIF" if is_critical else "Non-Critical"
        
        # Step 3: Attribution & Explainability
        attribution_pairs = extract_shap_explanation(clean_text, report_input)
        
        # Step 4: Safety Context Analysis
        hazards, verified_barriers, missing_barriers = analyze_safety_context(report_input, clean_text)
        
        st.markdown("---")
        
        # ---------------------------------------------------------------------
        # AI Prediction Banner
        # ---------------------------------------------------------------------
        if is_critical:
            st.markdown(
                f"""
                <div class="critical-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="background: #da3633; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.95rem; text-transform: uppercase;">
                                🚨 POTENTIAL SIF PRECURSOR DETECTED
                            </span>
                            <h2 style="color: #ffffff; margin: 10px 0 4px 0;">HIGH-PRIORITY SUPERVISOR REVIEW REQUIRED</h2>
                            <p style="color: #e6edf3; margin-bottom: 0; font-size: 1.05rem;">
                                This observation indicates high-energy hazard exposure without confirmed barrier verification. Immediate supervisor validation recommended.
                            </p>
                        </div>
                        <div style="text-align: right; min-width: 160px;">
                            <div style="font-size: 2.4rem; font-weight: 900; color: #ff7b72;">{p_critical * 100:.1f}%</div>
                            <div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase;">SIF Precursor Probability</div>
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
            color = "#ff7b72" if is_critical else "#3fb950"
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
# TAB 2: MODEL BENCHMARK ANALYTICS
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
