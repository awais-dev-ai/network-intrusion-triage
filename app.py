"""Lightweight Streamlit SOC Triage UI for Network Intrusion Detection."""

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import load_config, resolve_path
from src.preprocessing import prepare_features

st.set_page_config(
    page_title="Network Intrusion Triage (UNSW-NB15)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0369a1 100%);
            padding: 1.5rem 2rem;
            border-radius: 14px;
            color: #f8fafc;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(56, 189, 248, 0.25);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
        }
        .hero-banner h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff;
        }
        .hero-banner p {
            margin: 0.45rem 0 0 0;
            font-size: 0.96rem;
            color: #cbd5e1;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.18);
            color: #7dd3fc;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.55rem;
            border: 1px solid rgba(125, 211, 252, 0.3);
        }
        [data-testid="stMetric"] {
            background-color: rgba(248, 250, 252, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 0.85rem 1rem;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _load_pipeline(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def _load_metrics(metrics_path: str) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def _load_numeric_medians(train_path: str, num_cols: tuple[str, ...]) -> dict[str, float]:
    path = resolve_path(train_path)
    if not path.exists():
        return {col: 0.0 for col in num_cols}
    df = pd.read_parquet(path, columns=list(num_cols))
    return {col: float(df[col].median()) for col in num_cols}


@st.cache_data
def _load_demo_sample(test_path: str, n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(resolve_path(test_path))
    return df.sample(n=min(n_rows, len(df)), random_state=seed).reset_index(drop=True)


def _triage_priority(prob: float, high_cutoff: float, med_cutoff: float) -> str:
    if prob >= high_cutoff:
        return "HIGH"
    if prob >= med_cutoff:
        return "MEDIUM"
    return "LOW"


def _render_sidebar(config: dict) -> tuple[float, float]:
    st.sidebar.markdown("### 🛡️ Model & Triage Controls")
    metrics_path = resolve_path(config["paths"]["metrics_file"])
    if metrics_path.exists():
        payload = _load_metrics(str(metrics_path))
        best_name = payload.get("best_model", "random_forest")
        best_metrics = payload.get("models", {}).get(best_name, {})
        st.sidebar.caption(f"Active Model: **{best_name.replace('_', ' ').title()}**")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Recall", f"{best_metrics.get('recall', 0):.2%}")
        c2.metric("F1-Score", f"{best_metrics.get('f1_score', 0):.2%}")
        c3, c4 = st.sidebar.columns(2)
        c3.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.2%}")
        c4.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.4f}")

    st.sidebar.divider()
    alert_threshold = st.sidebar.slider(
        "Suspicious Alert Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )
    high_cutoff = st.sidebar.slider(
        "High-Priority Cutoff",
        min_value=max(0.55, alert_threshold),
        max_value=0.99,
        value=0.80,
        step=0.05,
    )
    st.sidebar.caption(
        "Flows below the alert threshold are marked **LOW**, between thresholds **MEDIUM**, "
        "and above the high-priority cutoff **HIGH**."
    )
    return alert_threshold, high_cutoff


def main() -> None:
    """Render the SOC Network Intrusion Triage interface."""
    _inject_styles()
    config = load_config()
    model_path = resolve_path(config["paths"]["best_model"])

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">UNSW-NB15 • SOC DECISION SUPPORT</div>
            <h1>🛡️ Network Intrusion Triage Dashboard</h1>
            <p>Prioritizes suspicious network flows for analyst review. Human verification is required before any traffic-blocking action.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_path.exists():
        st.error(
            f"Trained model not found at `{model_path}`. "
            "Run `python -m src.pipeline` first."
        )
        return

    model = _load_pipeline(str(model_path))
    alert_threshold, high_cutoff = _render_sidebar(config)

    tab_batch, tab_single = st.tabs(["📊 Batch Flow Triage", "🔍 Single-Flow Check"])

    with tab_batch:
        with st.container(border=True):
            col_up, col_demo = st.columns([3, 1], vertical_alignment="bottom")
            uploaded = col_up.file_uploader(
                "Upload a UNSW-NB15 Parquet file for batch triage",
                type=["parquet"],
            )
            use_demo = col_demo.button(
                "Load 200 Sample Test Flows",
                type="primary",
                use_container_width=True,
            )

        if use_demo:
            st.session_state["batch_df"] = _load_demo_sample(
                config["paths"]["raw_test"],
                n_rows=200,
                seed=int(config["project"]["seed"]),
            )
        elif uploaded is not None:
            st.session_state["batch_df"] = pd.read_parquet(uploaded)

        df = st.session_state.get("batch_df")
        if df is not None:
            try:
                X, _ = prepare_features(df, config, require_target=False)
            except ValueError as exc:
                st.error(f"Schema validation error: {exc}")
                return

            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= alert_threshold).astype(int)

            out = df.copy()
            out["predicted_label"] = preds
            out["suspicious_prob"] = probs.round(4)
            out["triage_priority"] = [
                _triage_priority(float(p), high_cutoff, alert_threshold)
                for p in probs
            ]

            high_n = int((out["triage_priority"] == "HIGH").sum())
            med_n = int((out["triage_priority"] == "MEDIUM").sum())
            low_n = int((out["triage_priority"] == "LOW").sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Flows", f"{len(out):,}")
            k2.metric("🔴 High Priority", f"{high_n:,}")
            k3.metric("🟠 Medium Priority", f"{med_n:,}")
            k4.metric("🟢 Low Priority", f"{low_n:,}")

            with st.container(border=True):
                st.subheader("Priority Distribution")
                priority_counts = (
                    out["triage_priority"]
                    .value_counts()
                    .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
                )
                st.bar_chart(priority_counts, color="#0284c7")

            with st.container(border=True):
                priority_filter = st.multiselect(
                    "Filter by Triage Priority",
                    options=["HIGH", "MEDIUM", "LOW"],
                    default=["HIGH", "MEDIUM", "LOW"],
                )
                filtered = out[out["triage_priority"].isin(priority_filter)].sort_values(
                    "suspicious_prob", ascending=False
                )

                st.dataframe(filtered.head(100), use_container_width=True)
                st.download_button(
                    label="⬇️ Download Triaged Flows (CSV)",
                    data=filtered.to_csv(index=False).encode("utf-8"),
                    file_name="triaged_flows.csv",
                    mime="text/csv",
                )

    with tab_single:
        with st.container(border=True):
            st.write("Select a quick preset or adjust flow parameters (unlisted features use training medians):")
            p1, p2, _ = st.columns([1, 1, 2])
            if p1.button("🟢 Normal Web Flow Preset", use_container_width=True):
                st.session_state["preset"] = {
                    "proto": "tcp", "service": "http", "state": "FIN",
                    "dur": 0.12, "sbytes": 1200, "dbytes": 4500, "rate": 35.0,
                }
            if p2.button("🔴 Suspicious Burst Preset", use_container_width=True):
                st.session_state["preset"] = {
                    "proto": "udp", "service": "dns", "state": "INT",
                    "dur": 0.00001, "sbytes": 114, "dbytes": 0, "rate": 100000.0,
                }

            preset = st.session_state.get(
                "preset",
                {"proto": "tcp", "service": "-", "state": "FIN", "dur": 0.05, "sbytes": 512, "dbytes": 256, "rate": 45.0},
            )

            col1, col2, col3 = st.columns(3)
            proto_opts = ["tcp", "udp", "arp", "ospf"]
            svc_opts = ["-", "http", "dns", "ftp", "smtp", "ssh"]
            state_opts = ["FIN", "INT", "CON", "REQ", "RST"]

            proto = col1.selectbox("Protocol (proto)", proto_opts, index=proto_opts.index(preset["proto"]))
            service = col2.selectbox("Service", svc_opts, index=svc_opts.index(preset["service"]))
            state = col3.selectbox("Connection State", state_opts, index=state_opts.index(preset["state"]))

            c4, c5, c6, c7 = st.columns(4)
            dur = c4.number_input("Duration (dur)", min_value=0.0, value=float(preset["dur"]), format="%.5f")
            sbytes = c5.number_input("Source Bytes (sbytes)", min_value=0, value=int(preset["sbytes"]))
            dbytes = c6.number_input("Dest Bytes (dbytes)", min_value=0, value=int(preset["dbytes"]))
            rate = c7.number_input("Packet Rate (rate)", min_value=0.0, value=float(preset["rate"]))

            if st.button("Score Flow", type="primary"):
                num_cols = tuple(config["schema"]["numeric_cols"])
                row = _load_numeric_medians(config["paths"]["raw_train"], num_cols).copy()
                row.update(
                    {
                        "proto": proto,
                        "service": service,
                        "state": state,
                        "dur": dur,
                        "sbytes": sbytes,
                        "dbytes": dbytes,
                        "rate": rate,
                    }
                )
                single_df = pd.DataFrame([row])
                X_single, _ = prepare_features(single_df, config, require_target=False)
                prob = float(model.predict_proba(X_single)[0, 1])
                pred = int(prob >= alert_threshold)
                priority = _triage_priority(prob, high_cutoff, alert_threshold)

                st.progress(min(max(prob, 0.0), 1.0), text=f"Suspicious Risk Score: {prob:.2%}")
                if pred == 1:
                    st.error(f"Prediction: **Suspicious (1)** | Priority: **{priority}** | Score: **{prob:.2%}**")
                else:
                    st.success(f"Prediction: **Normal (0)** | Priority: **{priority}** | Score: **{prob:.2%}**")


if __name__ == "__main__":
    main()
