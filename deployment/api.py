import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from inference.inspection import build_inspection_result
from inference.onnx_model import OnnxFusionModel
from inference.preprocessing import preprocess_image, preprocess_vibration


APP_DIR = Path(__file__).resolve().parent
ONNX_PATH = APP_DIR / "artifacts" / "fusion_model.onnx"
NORMALIZATION_STATS_PATH = APP_DIR / "artifacts" / "normalization_stats.json"
CONFIG_PATH = APP_DIR / "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = json.load(f)

DEFECT_THRESHOLD = _config["defect_threshold"]
SAMPLING_RATE_HZ = _config["sampling_rate_hz"]
FAULT_CLASS_NAMES = _config["fault_class_names"]

_state: dict = {"model": None, "vib_mean": None, "vib_std": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(NORMALIZATION_STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    _state["model"] = OnnxFusionModel(ONNX_PATH)
    _state["vib_mean"] = np.array(stats["vib_mean"], dtype=np.float32)
    _state["vib_std"] = np.array(stats["vib_std"], dtype=np.float32)
    yield
    _state.clear()


app = FastAPI(title="Multimodal Defect Detection API", lifespan=lifespan)


class InspectResponse(BaseModel):
    status: str
    defect_probability: float
    fault_type: Optional[str] = None
    fault_confidence: Optional[float] = None


@app.get("/health")
def health() -> dict:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok"}


@app.post("/inspect", response_model=InspectResponse)
async def inspect(image: UploadFile = File(...), vibration_window: UploadFile = File(...)) -> InspectResponse:
    model = _state["model"]
    vib_mean = _state["vib_mean"]
    vib_std = _state["vib_std"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    image_array = preprocess_image(pil_image)

    window_bytes = await vibration_window.read()
    window = np.load(io.BytesIO(window_bytes))
    vib_array = preprocess_vibration(window, SAMPLING_RATE_HZ, vib_mean, vib_std)

    defect_proba, fault_proba = model.predict(image_array, vib_array)
    result = build_inspection_result(defect_proba, fault_proba, DEFECT_THRESHOLD, FAULT_CLASS_NAMES)

    return InspectResponse(**result)