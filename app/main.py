from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.ste_service import (
    FEATURE_NAMES,
    build_feature_vector,
    compute_ste_outputs,
    get_model,
    training_range_report,
)
from app.efficiency_service import compute_efficiency_outputs
from app.optimization_service import optimize_design

ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = ROOT_DIR / "index.html"
LEGACY_HTML_FILE = ROOT_DIR / "GEARCALC-STUDENTS-DEMO-V3-01-03-2026.html"


class STEPredictRequest(BaseModel):
    z1: float = Field(..., gt=3)
    z2: float = Field(..., gt=3)
    cc1: float
    cc2: float
    ck1: float = Field(..., gt=0)
    ck2: float = Field(..., gt=0)
    cf1: float = Field(..., gt=0)
    cf2: float = Field(..., gt=0)
    a0_rad: float = Field(..., gt=0.0, lt=1.5)
    cs1: float
    cs2: float
    e_to_sy: float = Field(..., gt=1e-9)
    nu: float = Field(..., gt=0.0, lt=0.4999)
    log_tmid_to_max: float
    b_to_m: float = Field(..., gt=0)
    da12_to_m: float
    m: float = Field(..., gt=0)
    x_points: int = Field(1000, ge=64, le=4000)
    sy_ref: float = Field(1000.0, gt=0)


class EfficiencyPredictRequest(BaseModel):
    z1: float = Field(..., gt=3)
    z2: float = Field(..., gt=3)
    m: float = Field(..., gt=0)
    alpha_deg: float = Field(..., gt=0.0, lt=45.0)
    x1: float = 0.0
    x2: float = 0.0
    ck: float = Field(1.0, gt=0)
    b_mm: float = Field(..., gt=0)
    n1_rpm: float = Field(..., gt=0)
    torque_nm: float = Field(..., gt=0)
    eps_alpha: float = Field(1.5, gt=0.2)
    eta_mode: str = Field("auto")
    lube_family: str = Field("mineral")
    iso_vg: float = Field(68.0, ge=10, le=680)
    oil_temp_c: float = Field(60.0, ge=-10, le=180)
    additive: str = Field("none")
    eta0_pa_s: float = Field(0.02, gt=0.0001)
    roughness_um: float = Field(0.30, gt=0.01)
    n_points: int = Field(700, ge=128, le=4000)


class OptimizeRequest(BaseModel):
    z1: float = Field(..., gt=3)
    z2: float = Field(..., gt=3)
    m: float = Field(..., gt=0)
    b_mm: float = Field(..., gt=0)
    alpha_deg: float = Field(..., gt=0, lt=45)
    x1: float = 0.0
    x2: float = 0.0
    torque_nm: float = Field(..., gt=0)
    n1_rpm: float = Field(..., gt=0)
    kv: float = Field(1.1, gt=0)
    ka: float = Field(1.0, gt=0)
    khb: float = Field(1.3, gt=0)
    kfb: float = Field(1.3, gt=0)
    agma_sf_min: float | None = None
    agma_sh_min: float | None = None
    iso_sf_min: float | None = None
    iso_sh_min: float | None = None
    tip1_mm: float | None = None
    tip2_mm: float | None = None
    undercut_ok: bool = True
    interference_ok: bool = True
    efficiency_percent: float = 97.0
    target_sf: float = Field(1.35, gt=0.5, le=5.0)
    target_sh: float = Field(1.20, gt=0.5, le=5.0)
    target_tip_mm: float = Field(0.60, gt=0.01, le=10.0)
    x_limit_abs: float = Field(1.0, gt=0.1, le=2.0)
    min_z: int = Field(12, ge=6, le=200)
    max_z: int = Field(200, ge=12, le=500)
    top_k: int = Field(8, ge=3, le=20)


app = FastAPI(
    title="GEARCALC STE API",
    description="In-memory NN inference for STE integrated with GEARCALC.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    if HTML_FILE.exists():
        return FileResponse(HTML_FILE)
    if LEGACY_HTML_FILE.exists():
        return FileResponse(LEGACY_HTML_FILE)
    raise HTTPException(status_code=404, detail="Missing frontend file: index.html")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ste/meta")
def ste_meta() -> dict[str, object]:
    return {
        "feature_names": FEATURE_NAMES,
        "default_sy_ref": 1000.0,
        "x_domain": [0.0, 1.0],
    }


@app.post("/predict_ste")
def predict_ste(payload: STEPredictRequest) -> dict[str, object]:
    try:
        model_loaded = get_model() is not None
        features_dict = payload.model_dump(exclude={"x_points", "sy_ref"})
        features = build_feature_vector(features_dict)
        x_vals = np.linspace(0.0, 1.0, int(payload.x_points), dtype=float)
        result = compute_ste_outputs(features, x_vals, sy_ref=float(payload.sy_ref))
        range_report = training_range_report(features)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"STE prediction failed: {exc}") from exc

    out_of_range = [item for item in range_report if not item["in_range"]]
    return {
        "ok": True,
        "model_loaded": model_loaded,
        "features": features_dict,
        "range_report": range_report,
        "out_of_range_count": len(out_of_range),
        "results": result,
    }


@app.post("/predict_efficiency")
def predict_efficiency(payload: EfficiencyPredictRequest) -> dict[str, object]:
    try:
        inputs = payload.model_dump()
        result = compute_efficiency_outputs(inputs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Efficiency prediction failed: {exc}") from exc

    return {
        "ok": True,
        "inputs": inputs,
        "results": result,
    }


@app.post("/optimize_design")
def optimize_design_endpoint(payload: OptimizeRequest) -> dict[str, object]:
    try:
        data = payload.model_dump()
        result = optimize_design(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Optimization failed: {exc}") from exc
    return result
