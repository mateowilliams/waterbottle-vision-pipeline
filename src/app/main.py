from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import resnet18

from . import monitoring
from .config import (
    LOW_CONF_THRESHOLD,
    MODEL_ARTIFACT,
    REFERENCE_STATS,
    ROOT,
    UPLOAD_DIR,
    WEB_DIR,
)
from .db import get_connection

INDEX_HTML = WEB_DIR / "index.html"
DASHBOARD_HTML = WEB_DIR / "dashboard.html"
STYLES_CSS = WEB_DIR / "styles.css"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Model state (loaded once) ----------
state: Dict[str, Any] = {
    "model": None,
    "tfm": None,
    "id_to_label": {},
    "model_version": MODEL_ARTIFACT.stem,
    "reference": None,
}


def load_artifact(
    artifact_path: Path,
) -> Tuple[torch.nn.Module, transforms.Compose, Dict[int, str], str]:
    ckpt = torch.load(str(artifact_path), map_location="cpu")

    num_classes = int(ckpt["num_classes"])
    label_to_id = ckpt["label_to_id"]
    id_to_label = {int(v): str(k) for k, v in label_to_id.items()}

    img_size = int(ckpt["img_size"])
    mean = tuple(ckpt["imagenet_mean"])
    std = tuple(ckpt["imagenet_std"])

    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # The version comes from the artifact itself; fall back to the file name.
    model_version = str(ckpt.get("model_version") or artifact_path.stem)
    return m, tfm, id_to_label, model_version


def load_reference() -> Optional[dict]:
    if REFERENCE_STATS.exists():
        try:
            return json.loads(REFERENCE_STATS.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_ARTIFACT.exists():
        raise RuntimeError(f"Artifact not found: {MODEL_ARTIFACT}")
    model, tfm, id_to_label, model_version = load_artifact(MODEL_ARTIFACT)
    state.update(
        model=model,
        tfm=tfm,
        id_to_label=id_to_label,
        model_version=model_version,
        reference=load_reference(),
    )
    yield
    state.clear()


app = FastAPI(title="Water Bottle Classifier", lifespan=lifespan)


# ---------- Persistence ----------
def db_insert_prediction(filepath: str, pred_label: str, confidence: float, probs: dict) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO images (filepath, source) VALUES (%s, %s) RETURNING id;",
                (filepath, "web_upload"),
            )
            image_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO predictions (image_id, model_version, pred_label, confidence, probs_json)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (image_id, state["model_version"], pred_label, confidence, json.dumps(probs)),
            )
        conn.commit()
    return image_id


# ---------- Static pages ----------
@app.get("/")
def home():
    if not INDEX_HTML.exists():
        raise HTTPException(500, f"Missing {INDEX_HTML}")
    return FileResponse(INDEX_HTML)


@app.get("/dashboard")
def dashboard():
    if not DASHBOARD_HTML.exists():
        raise HTTPException(500, f"Missing {DASHBOARD_HTML}")
    return FileResponse(DASHBOARD_HTML)


@app.get("/styles.css")
def styles():
    if not STYLES_CSS.exists():
        raise HTTPException(500, f"Missing {STYLES_CSS}")
    return FileResponse(STYLES_CSS)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model_version": state["model_version"],
        "reference_loaded": state["reference"] is not None,
    }


# ---------- Inference ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    model, tfm, id_to_label = state["model"], state["tfm"], state["id_to_label"]
    if model is None or tfm is None:
        raise HTTPException(500, "Model not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image/*, got {file.content_type}")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    if len(data) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(413, "File too large (max 10MB)")

    try:
        img = Image.open(BytesIO(data)).convert("RGBA").convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {repr(e)}")

    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename
    save_path.write_bytes(data)
    rel_path = str(save_path.relative_to(ROOT))

    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_id = probs.max(dim=0)

    pred_id_int = int(pred_id.item())
    conf_f = float(conf.item())
    probs_dict = {id_to_label[i]: float(probs[i].item()) for i in range(len(id_to_label))}

    image_id = db_insert_prediction(rel_path, id_to_label[pred_id_int], conf_f, probs_dict)

    return {
        "pred_label": id_to_label[pred_id_int],
        "confidence": conf_f,
        "probs": probs_dict,
        "image_id": image_id,
        "saved_path": rel_path,
        "model_version": state["model_version"],
    }


# ---------- Ground truth (labels) ----------
class LabelIn(BaseModel):
    image_id: int
    true_label: str


@app.post("/label")
def add_label(payload: LabelIn) -> Dict[str, Any]:
    """Store the ground-truth label for an already-predicted image.

    Enables real-accuracy computation and retraining with fresh data.
    """
    valid_labels = set(state["id_to_label"].values())
    if valid_labels and payload.true_label not in valid_labels:
        raise HTTPException(400, f"Invalid true_label. Expected one of: {sorted(valid_labels)}")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM images WHERE id = %s;", (payload.image_id,))
            if cur.fetchone() is None:
                raise HTTPException(404, f"image_id {payload.image_id} does not exist")
            cur.execute(
                "INSERT INTO labels (image_id, true_label) VALUES (%s, %s) RETURNING id;",
                (payload.image_id, payload.true_label),
            )
            label_id = cur.fetchone()[0]
        conn.commit()

    return {"status": "ok", "label_id": label_id, "image_id": payload.image_id}


# ---------- Monitoring / drift ----------
@app.get("/monitoring")
def monitoring_report(model_version: Optional[str] = None) -> JSONResponse:
    """Monitoring + drift report (JSON). Consumed by the dashboard."""
    try:
        with get_connection() as conn:
            report = monitoring.build_report(
                conn,
                reference=state["reference"],
                low_conf_threshold=LOW_CONF_THRESHOLD,
                model_version=model_version,
            )
    except Exception as e:
        raise HTTPException(503, f"Could not query the database: {repr(e)}")
    report["active_model_version"] = state["model_version"]
    return JSONResponse(report)
