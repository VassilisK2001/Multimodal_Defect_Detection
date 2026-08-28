---
title: Multimodal Defect Detection
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Multimodal Defect Detection — Live Demo

A fusion model combining **image** and **vibration** data to detect and classify
industrial component defects, deployed as an edge-optimized ONNX Runtime.

## How it works

- **Defect gate**: given a component image and a vibration signal window
  together, the model outputs a defect probability, thresholded at a
  recall-constrained value tuned to prioritize catching real defects.
- **Fault type**: if flagged as defective, a second head classifies the
  specific bearing fault type (`outer_race`, `inner_race`, `ball`).
- Both predictions come from **one shared fusion model** in a single forward
  pass.

## Try it

No file upload needed. Select one of four scenarios (`normal` and each
fault type) and one of several real, held-out test-set samples per scenario
from the dropdowns. Each sample's ground-truth label is shown alongside the
model's prediction.

## Note on cold starts

This Space runs on free CPU hardware, which sleeps after a period of
inactivity. The first request after a period of inactivity may take
30-60 seconds while the container restarts. Subsequent requests are fast
(typically well under 100ms end to end).

## Architecture

FastAPI backend (ONNX Runtime, CPU) and Streamlit frontend run as two
processes in a single Docker container. Streamlit is the public-facing
process (port 7860, this platform's requirement). FastAPI runs internally
on port 8000, never exposed externally.