# ============================================================
# MONITORING DASHBOARD — Real-Time Object Detection ML System
# ============================================================

import os
import sys
import json

# ── Make project root importable — MUST happen before any `from src...`
# import. Streamlit adds the script's own directory (monitoring/) to
# sys.path, not the project root, so `from src.metrics import ...`
# fails with ModuleNotFoundError unless this is done explicitly —
# regardless of whether the dashboard is launched via
# `streamlit run monitoring/monitoring_dashboard.py` directly or via
# scripts/run_dashboard.py.
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Object Detection Dashboard", layout="wide")
st.title("📷 Real-Time Object Detection — Monitoring Dashboard")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "OBJECT_DETECTION_API_URL",
    "http://127.0.0.1:8000"
) + "/predict"

PSI_MODERATE     = 0.10
PSI_HIGH         = 0.20
CONF_MEAN_ALERT  = 0.30

# ── Path resolution — Streamlit Cloud safe ────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MODEL_DIR        = os.path.join(BASE_DIR, "detection_models")
LOG_PATH         = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
CHALLENGER_PATH  = os.path.join(MODEL_DIR, "challenger_log.json")
EXPERIMENTS_PATH = os.path.join(MODEL_DIR, "model_experiment_results.csv")
REF_CONF_PATH    = os.path.join(MODEL_DIR, "reference_confidence_scores.csv")
REF_COUNT_PATH   = os.path.join(MODEL_DIR, "reference_object_counts.csv")


# ============================================================
# SIDEBAR — LIVE PREDICTION
# ============================================================

st.sidebar.header("🔍 Detect Objects in an Image")

domain = st.sidebar.selectbox(
    "Business Domain",
    ["retail_shelf", "warehouse_safety", "security_intrusion"],
)
uploaded_file = st.sidebar.file_uploader("Upload a frame", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and st.sidebar.button("Run Detection"):
    with st.sidebar:
        with st.spinner("Calling API... First request may take 30-60s (Render cold start)"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                response = requests.post(API_URL, files=files, params={"domain": domain}, timeout=90)
                if response.status_code == 200:
                    result = response.json()
                    color  = {"NORMAL": "green", "ALERT": "orange", "CRITICAL": "red"}.get(
                        result["decision"], "gray"
                    )
                    st.success("Prediction received!")
                    st.markdown(f"**Objects detected:** `{result['detection_count']}`")
                    st.markdown(f"**Persons:** `{result['person_count']}`")
                    st.markdown(f"<h3 style='color:{color}'>Decision: {result['decision']}</h3>",
                                unsafe_allow_html=True)
                    if result.get("rule_triggered"):
                        st.warning(f"Rule: {result['rule_triggered']}")
                    st.json(result["detections"])
                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except requests.exceptions.Timeout:
                st.warning("Request timed out (90s). Render is waking up. Wait 30s and try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    if "decision" in df_log.columns and len(df_log):
        critical_rate = (df_log["decision"] == "CRITICAL").mean()
        alert_rate    = (df_log["decision"] == "ALERT").mean()
        if critical_rate > 0.15:
            st.error(f"🔴 HIGH CRITICAL RATE: {critical_rate:.1%} of recent frames (expected < 15%).")
            alerts_found = True
        if alert_rate > 0.35:
            st.warning(f"🟡 HIGH ALERT RATE: {alert_rate:.1%} of recent frames (expected < 35%).")
            alerts_found = True

if os.path.exists(REF_CONF_PATH) and os.path.exists(LOG_PATH):
    ref_conf = pd.read_csv(REF_CONF_PATH)
    if "confidence" in ref_conf.columns and ref_conf["confidence"].mean() < CONF_MEAN_ALERT:
        st.warning(f"🟡 LOW AVG CONFIDENCE in reference set: {ref_conf['confidence'].mean():.3f}")
        alerts_found = True

if not alerts_found:
    st.success("All systems normal — no alerts triggered")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER HISTORY
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        challenger_log = json.load(f)
    if challenger_log:
        latest = challenger_log[-1]
        decision_color = "green" if latest["decision"] == "PROMOTED" else "red"
        icon = "✅" if latest["decision"] == "PROMOTED" else "❌"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Challenger", latest.get("challenger_variant", "—"))
        with col2:
            st.metric("Challenger mAP50-95", latest.get("challenger_map50_95", "—"))
        with col3:
            st.metric("Champion mAP50-95", latest.get("champion_map50_95", "—") or "First Run")
        with col4:
            st.metric("Latency (ms)", latest.get("challenger_latency_ms", "—"))

        st.markdown(
            f"<h4 style='color:{decision_color}'>{icon} Decision: {latest['decision']} — {latest.get('reason', '')}</h4>",
            unsafe_allow_html=True,
        )

        def _gate_label(passed):
            if passed is None:
                return "➖ N/A"
            return "✅ Pass" if passed else "❌ Fail"

        if latest.get("gates"):
            g = latest["gates"]
            gcol1, gcol2, gcol3 = st.columns(3)
            gcol1.metric("mAP Improvement Gate", _gate_label(g.get("map_improvement_passed")))
            gcol2.metric("Precision Gate", _gate_label(g.get("precision_floor_passed")))
            gcol3.metric("Latency Gate", _gate_label(g.get("latency_ceiling_passed")))

        if len(challenger_log) > 1:
            with st.expander("View full challenger history"):
                history_df = pd.DataFrame(challenger_log)
                cols = [c for c in [
                    "evaluated_at", "challenger_variant", "challenger_map50_95",
                    "champion_variant", "champion_map50_95", "decision", "reason",
                ] if c in history_df.columns]
                st.dataframe(history_df[cols])
else:
    st.info("No challenger log found. Run `python scripts/train_model.py` first.")


# ============================================================
# SECTION 3 — KPI METRICS + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Detection Performance KPIs")

# ── Production champion — read from the registry, not just "best mAP" ──
# The registered champion may NOT be the highest-mAP row below: it's
# filtered by the real-time SLA (precision floor + latency safety
# margin) in training_pipeline.py. Showing both avoids the confusing
# mismatch of two different "best" models on the same dashboard.
registry_path = os.path.join(MODEL_DIR, "latest_model.json")
if os.path.exists(registry_path):
    with open(registry_path) as f:
        registry = json.load(f)
    st.info(
        f"🏆 **Production champion:** {registry.get('variant', '—')}  "
        f"(conf={registry.get('conf_threshold', '—')}, iou={registry.get('iou_threshold', '—')}) "
        "— selected under the real-time SLA (precision floor + latency safety margin), "
        "not necessarily the highest-mAP row in the grid below."
    )

if os.path.exists(EXPERIMENTS_PATH):
    exp_df = pd.read_csv(EXPERIMENTS_PATH)
    col1, col2, col3, col4 = st.columns(4)
    best = exp_df.sort_values("coco128_map50_95", ascending=False).iloc[0]
    with col1:
        st.metric("Highest-mAP Variant (all, unfiltered)", best["variant"])
    with col2:
        st.metric("mAP50-95 (coco128)", f"{best['coco128_map50_95']:.3f}")
    with col3:
        st.metric("Published mAP50-95", f"{best['published_map50_95']:.1f}")
    with col4:
        st.metric("Latency", f"{best['mean_latency_ms']:.1f} ms")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("mAP50-95 by Variant (real, on coco128)")
        st.bar_chart(exp_df.groupby("variant")["coco128_map50_95"].max())
    with col_b:
        st.subheader("Latency by Variant (ms)")
        st.bar_chart(exp_df.groupby("variant")["mean_latency_ms"].min())

    st.subheader("Full Variant × Threshold Grid")
    st.dataframe(exp_df, use_container_width=True)
else:
    st.warning("No experiment results found. Run `python scripts/train_model.py` first.")

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    if len(log_df):
        st.subheader("Live Decision Distribution")
        st.bar_chart(log_df["decision"].value_counts())


# ============================================================
# SECTION 4 — PSI DRIFT REPORT
# ============================================================

st.markdown("---")
st.subheader("📉 Confidence & Detection-Count Drift (PSI)")

if os.path.exists(REF_CONF_PATH) and os.path.exists(LOG_PATH):
    ref_conf = pd.read_csv(REF_CONF_PATH)
    log_df   = pd.read_csv(LOG_PATH)

    st.write(
        "PSI compares the reference (benchmark-time) confidence distribution "
        "against live production traffic. This chart shows the reference "
        "distribution; live PSI is computed once enough production volume "
        "has accumulated."
    )
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(ref_conf["confidence"], bins=20, alpha=0.7, color="steelblue")
    ax.set_title("Reference Confidence Distribution (benchmark set)")
    ax.set_xlabel("Confidence")
    st.pyplot(fig)
    plt.close(fig)

    if "detection_count" in log_df.columns and len(log_df) >= 10:
        from src.metrics import psi as psi_fn
        ref_counts_df = pd.read_csv(REF_COUNT_PATH) if os.path.exists(REF_COUNT_PATH) else None
        if ref_counts_df is not None:
            drift_score = psi_fn(ref_counts_df["objects_per_frame"], log_df["detection_count"])
            status = "🔴 CRITICAL" if drift_score >= PSI_HIGH else "🟡 MODERATE" if drift_score >= PSI_MODERATE else "🟢 OK"
            st.metric("Objects-per-frame PSI (live vs reference)", f"{drift_score:.4f}", status)
    else:
        st.info("Accumulate >= 10 live predictions to compute live PSI drift.")
else:
    st.warning("Reference confidence stats not found. Run `python scripts/train_model.py` first.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("📋 Recent Predictions")

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    st.dataframe(log_df.tail(20), use_container_width=True)
else:
    st.info(
        "Prediction logs are written by the deployed API — not available on "
        "Streamlit Cloud by default. Run locally, or point OBJECT_DETECTION_API_URL "
        "to your deployed API, to see logs."
    )