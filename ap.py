# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F
import io
import os
import traceback
import numpy as np


# === CONFIG ===
MODEL_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === APP ===
app = FastAPI(title="Pneumonia Detector (ViT)")

# Serve index.html and other files from project root
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# === MODEL LOAD ===
print(f"Loading model from: {MODEL_DIR} ...")
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print("ERROR loading model:", e)
    traceback.print_exc()
    raise RuntimeError(f"Failed to load model from {MODEL_DIR}. Check path and files.") from e

# id2label / mapping
id2label = getattr(model.config, "id2label", None)
if id2label is None:
    # fallback default mapping
    id2label = {0: "NORMAL", 1: "PNEUMONIA"}
print("id2label:", id2label)
print("Using device:", DEVICE)

# === PREDICTION ===
def predict_from_pil(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # processor returns tensors; typical key is 'pixel_values'
    inputs = processor(images=image, return_tensors="pt")
    # move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # if model returns single logit per sample (rare), handle sigmoid case
        if logits.shape[-1] == 1:
            probs_pos = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            probs = []
            for p in probs_pos:
                probs.append([1.0 - float(p), float(p)])  # [NORMAL, PNEUMONIA]
            probs = np.array(probs)
        else:
            probs = F.softmax(logits, dim=-1).cpu().numpy()

    # build mapping from id->label string (ensure integer keys)
    label_map = {int(k): v for k, v in id2label.items()}

    # take first (single image)
    probs_map = {label_map[i]: float(probs[0][i]) for i in range(probs.shape[1])}
    top_idx = int(probs[0].argmax())
    return {"probs": probs_map, "top_label": label_map[top_idx], "top_prob": float(probs[0][top_idx])}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (png/jpeg).")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported image format.")

    try:
        result = predict_from_pil(image)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    message = "Pneumonia detected" if result["top_label"].lower().startswith("pneu") else "No pneumonia detected"
    return JSONResponse({"message": message, **result})

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_dir": MODEL_DIR}
