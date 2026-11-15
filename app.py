"""Lightweight Streamlit SOC Triage UI for Network Intrusion Detection."""

import joblib
import pandas as pd
import streamlit as st

from src.config import load_config, resolve_path
from src.preprocessing import prepare_features

st.set_page_config(
    page_title="Network Intrusion Triage (UNSW-NB15)",
    layout="wide",
)


@st.cache_resource
def _load_pipeline(model_path: str):
    return joblib.load(model_path)


def _triage_priority(prob: float) -> str:
    if prob >= 0.80:
        return "HIGH"
    if prob >= 0.50:
        return "MEDIUM"
    return "LOW"


def main() -> None:
    """Render the SOC Network Intrusion Triage interface."""
    config = load_config()
    model_path = resolve_path(config["paths"]["best_model"])

    st.title("Network Intrusion Triage — SOC Decision Support")
    st.warning(
        "Academic Prototype Disclaimer: This model is trained on the UNSW-NB15 "
        "benchmark dataset for educational triage support only. Do not use it "
        "for autonomous production blocking without human analyst verification."
    )

    if not model_path.exists():
        st.error(
            f"Trained model not found at `{model_path}`. "
            "Run `python -m src.pipeline` first."
        )
        return

    model = _load_pipeline(str(model_path))

    tab_batch, tab_single = st.tabs(["Batch Parquet Triage", "Single-Flow Check"])

    with tab_batch:
        uploaded = st.file_uploader(
            "Upload a UNSW-NB15 Parquet file for batch triage",
            type=["parquet"],
        )
        if uploaded is not None:
            df = pd.read_parquet(uploaded)
            try:
                X, _ = prepare_features(df, config, require_target=False)
            except ValueError as exc:
                st.error(f"Schema validation error: {exc}")
                return

            probs = model.predict_proba(X)[:, 1]
            preds = model.predict(X)

            out = df.copy()
            out["predicted_label"] = preds
            out["suspicious_prob"] = probs.round(4)
            out["triage_priority"] = [_triage_priority(float(p)) for p in probs]

            suspicious_count = int((preds == 1).sum())
            st.metric("Flows Flagged Suspicious", f"{suspicious_count} / {len(out)}")
            st.dataframe(
                out.sort_values("suspicious_prob", ascending=False).head(100),
                use_container_width=True,
            )

    with tab_single:
        st.write("Adjust key flow parameters (remaining features default to medians):")
        col1, col2, col3 = st.columns(3)
        proto = col1.selectbox("Protocol (proto)", ["tcp", "udp", "arp", "ospf"])
        service = col2.selectbox(
            "Service", ["-", "http", "dns", "ftp", "smtp", "ssh"]
        )
        state = col3.selectbox("Connection State", ["FIN", "INT", "CON", "REQ", "RST"])

        c4, c5, c6, c7 = st.columns(4)
        dur = c4.number_input("Duration (dur)", min_value=0.0, value=0.05, step=0.01)
        sbytes = c5.number_input("Source Bytes (sbytes)", min_value=0, value=512)
        dbytes = c6.number_input("Dest Bytes (dbytes)", min_value=0, value=256)
        rate = c7.number_input("Packet Rate (rate)", min_value=0.0, value=45.0)

        if st.button("Score Flow"):
            row = {col: 0.0 for col in config["schema"]["numeric_cols"]}
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
            pred = int(model.predict(X_single)[0])

            status = "Suspicious (1)" if pred == 1 else "Normal (0)"
            st.success(
                f"Prediction: **{status}** | "
                f"Suspicious Probability: **{prob:.2%}** | "
                f"Triage Priority: **{_triage_priority(prob)}**"
            )


if __name__ == "__main__":
    main()
