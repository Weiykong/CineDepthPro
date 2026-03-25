from __future__ import annotations

import gc
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageOps


SERVER_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SERVER_DIR / "vendor" / "ml-depth-pro"
CHECKPOINT_PATH = Path(
    os.environ.get(
        "DEPTH_PRO_CHECKPOINT",
        VENDOR_DIR / "checkpoints" / "depth_pro.pt",
    )
)
MAX_INPUT_DIM = int(os.environ.get("DEPTH_PRO_MAX_DIM", "1152"))

if str(VENDOR_DIR / "src") not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR / "src"))

import depth_pro  # noqa: E402
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig  # noqa: E402


def choose_device() -> torch.device:
    forced = os.environ.get("DEPTH_PRO_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_inverse_depth(depth_meters: np.ndarray) -> np.ndarray:
    safe_depth = np.clip(depth_meters, 1e-4, None)
    inverse_depth = 1.0 / safe_depth
    lo, hi = np.percentile(inverse_depth, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-6:
        return np.full_like(inverse_depth, 0.5, dtype=np.float32)
    normalized = np.clip((inverse_depth - lo) / (hi - lo), 0.0, 1.0)
    return normalized.astype(np.float32)


def encode_grayscale_png(values: np.ndarray) -> bytes:
    image = Image.fromarray(np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


@dataclass
class LoadedModel:
    model: torch.nn.Module
    transform: object
    device: torch.device


class ModelHolder:
    def __init__(self) -> None:
        self._lock = Lock()
        self._loaded: LoadedModel | None = None

    def health(self) -> dict[str, object]:
        loaded = self._loaded is not None
        return {
            "ok": True,
            "checkpoint_exists": CHECKPOINT_PATH.exists(),
            "model_loaded": loaded,
            "device": str(self._loaded.device if loaded else choose_device()),
            "checkpoint_path": str(CHECKPOINT_PATH),
        }

    def get(self) -> LoadedModel:
        with self._lock:
            if self._loaded is not None:
                return self._loaded

            if not CHECKPOINT_PATH.exists():
                raise FileNotFoundError(
                    f"Depth Pro checkpoint not found at {CHECKPOINT_PATH}. "
                    "Run ./setup.sh first."
                )

            device = choose_device()
            config = DepthProConfig(
                patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
                image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
                decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
                checkpoint_uri=str(CHECKPOINT_PATH),
                fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
                use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
            )
            model, transform = depth_pro.create_model_and_transforms(
                config=config,
                device=device,
                precision=torch.float32,
            )
            model.eval()
            self._loaded = LoadedModel(model=model, transform=transform, device=device)
            return self._loaded


model_holder = ModelHolder()
inference_lock = Lock()
app = FastAPI(title="CineDepth Ultra Depth Server", version="0.1.0")


@app.on_event("startup")
def preload_model() -> None:
    if os.environ.get("DEPTH_PRO_PRELOAD", "1") != "0":
        model_holder.get()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(model_holder.health())


@app.post("/v1/depth")
async def infer_depth(
    image: UploadFile = File(...),
) -> StreamingResponse:
    loaded = model_holder.get()

    try:
        payload = await image.read()
        pil_image = Image.open(io.BytesIO(payload))
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        if max(pil_image.size) > MAX_INPUT_DIM:
            pil_image.thumbnail((MAX_INPUT_DIM, MAX_INPUT_DIM), Image.Resampling.LANCZOS)
    except Exception as exc:  # pragma: no cover - defensive request parsing
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

    start = time.perf_counter()
    with inference_lock:
        transformed = loaded.transform(pil_image)
        prediction = loaded.model.infer(transformed, f_px=None)
        depth = prediction["depth"].detach().to("cpu").numpy().astype(np.float32)
        normalized = normalize_inverse_depth(depth)
        png_bytes = encode_grayscale_png(normalized)
        del transformed
        del prediction
        del depth
        del normalized
        gc.collect()
        if loaded.device.type == "mps":
            torch.mps.empty_cache()
        elif loaded.device.type == "cuda":
            torch.cuda.empty_cache()
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    headers = {
        "X-Depth-Server": "depth-pro-local",
        "X-Depth-Device": str(loaded.device),
        "X-Depth-Inference-Ms": str(elapsed_ms),
        "X-Depth-Max-Input-Dim": str(MAX_INPUT_DIM),
    }
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png", headers=headers)
