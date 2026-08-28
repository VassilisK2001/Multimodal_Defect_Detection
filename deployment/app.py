from pathlib import Path

import streamlit as st

from utils import call_inspect_api, is_prediction_correct, load_manifest


APP_DIR = Path(__file__).resolve().parent
DEMO_SAMPLES_DIR = APP_DIR / "demo_samples"
MANIFEST_PATH = DEMO_SAMPLES_DIR / "manifest.json"
API_URL = "http://127.0.0.1:8000/inspect"

st.set_page_config(page_title="Multimodal Defect Detection", layout="centered")
st.title("Multimodal Defect Detection Live Demo")
st.caption(
    "Select a scenario and a sample to run inference on. The sample consists of an image and a vibration window. The model will predict whether the sample is defective or normal, and if defective, it will also predict the bearing fault type that caused the defect."
)

scenarios = load_manifest(MANIFEST_PATH)
scenario_names = list(scenarios.keys())

scenario = st.selectbox("Scenario", scenario_names)
assert scenario is not None  # manifest.json always has non-empty scenarios

samples = scenarios[scenario]
sample_labels = [entry["sample"] for entry in samples]
sample_label = st.selectbox("Sample", sample_labels)
assert sample_label is not None  # each scenario always has at least one sample

selected_entry = next(entry for entry in samples if entry["sample"] == sample_label)

sample_dir = DEMO_SAMPLES_DIR / scenario / sample_label
image_path = sample_dir / "part.png"
window_path = sample_dir / "vibration_window.npy"

st.image(str(image_path), caption=f"{scenario} / {sample_label}", width=300)

true_label = selected_entry["true_fault_class"] if selected_entry["true_is_defect"] else "normal"
st.write(f"**Ground truth:** {true_label}")

if st.button("Run Inspection"):
    with st.spinner("Running inference..."):
        try:
            result, latency_ms = call_inspect_api(image_path, window_path, API_URL)
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            st.subheader("Result")
            st.write(f"**Status:** {result['status']}")
            st.write(f"**Defect probability:** {result['defect_probability']:.4f}")
            if result["status"] == "defective":
                st.write(f"**Fault type:** {result['fault_type']}")
                st.write(f"**Fault confidence:** {result['fault_confidence']:.4f}")

            correct = is_prediction_correct(
                selected_entry["true_is_defect"], selected_entry["true_fault_class"],
                result["status"], result.get("fault_type"),
            )
            if correct:
                st.success("Prediction matches ground truth.")
            else:
                st.warning("Prediction does NOT match ground truth.")

            st.write(f"**Latency:** {latency_ms:.1f} ms")