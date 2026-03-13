from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

ROOT_DIR = Path(__file__).resolve().parents[1]
NN_DIR = ROOT_DIR / "NN_model"

if str(NN_DIR) not in sys.path:
    sys.path.insert(0, str(NN_DIR))

from auxiliary_functions_NN import create_NN_features  # noqa: E402

FEATURE_NAMES = [
    "z1",
    "z2",
    "cc1",
    "cc2",
    "ck1",
    "ck2",
    "cf1",
    "cf2",
    "a0_rad",
    "cs1",
    "cs2",
    "e_to_sy",
    "nu",
    "log_tmid_to_max",
    "b_to_m",
    "da12_to_m",
    "m",
]

MEANS = np.array(
    [
        60.0,
        60.0,
        0.15,
        0.15,
        1.0,
        1.0,
        1.25,
        1.25,
        0.392699081698724,
        0.475,
        0.475,
        200.0,
        0.3,
        -1.0,
        25.5,
        0.05,
        4.0,
    ],
    dtype=float,
)
HALFRANGES = np.array(
    [
        40.0,
        40.0,
        0.15,
        0.15,
        0.05,
        0.05,
        0.1,
        0.1,
        0.0436332312998582,
        0.025,
        0.025,
        100.0,
        0.05,
        1.0,
        24.5,
        0.05,
        3.0,
    ],
    dtype=float,
)
LOWER_BOUNDS = MEANS - HALFRANGES
UPPER_BOUNDS = MEANS + HALFRANGES

_MODEL: keras.Model | None = None
_MODEL_LOCK = threading.Lock()


def _model_paths() -> tuple[Path, Path]:
    model_path = NN_DIR / "_nn_model.h5"
    weights_path = NN_DIR / "_nn_model.weights.h5"
    return model_path, weights_path


def get_model() -> keras.Model:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        model_path, weights_path = _model_paths()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = keras.models.load_model(model_path, compile=False)

        if weights_path.exists():
            try:
                model.load_weights(weights_path)
            except Exception:
                # Keep compatibility with legacy snapshots where weights are
                # already baked in _nn_model.h5.
                pass

        _MODEL = model
        return _MODEL


def build_feature_vector(payload: dict[str, float]) -> np.ndarray:
    vals = [float(payload[k]) for k in FEATURE_NAMES]
    vec = np.asarray(vals, dtype=float)
    if vec.shape != (17,):
        raise ValueError("Feature vector must have 17 values.")
    if not np.all(np.isfinite(vec)):
        raise ValueError("All STE features must be finite numbers.")
    return vec


def predict_ste_bar(features: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    x2d = np.atleast_2d(np.asarray(x_vals, dtype=float))
    if x2d.shape[0] != 1:
        raise ValueError("Only a single STE curve per request is supported.")
    if x2d.shape[1] < 8:
        raise ValueError("x_vals must include at least 8 points.")
    if np.any(~np.isfinite(x2d)):
        raise ValueError("x_vals contain invalid numbers.")

    features_scaled = (features - MEANS) / HALFRANGES
    features_scaled = np.atleast_2d(features_scaled)

    flags_flip = np.full((x2d.shape[0],), True, dtype=bool)
    nn_input = create_NN_features(
        x2d,
        features_scaled,
        periodic_features=True,
        flip_features=False,
        iflip1=[0, 2, 4, 6, 9],
        iflip2=[1, 3, 5, 7, 10],
        flip_flags=flags_flip,
    )

    model = get_model()
    y_pred = model.predict(nn_input, batch_size=1, verbose=0)
    return y_pred.reshape(x2d.shape)[0]


def compute_ste_outputs(
    features: np.ndarray,
    x_vals: np.ndarray,
    sy_ref: float = 1000.0,
) -> dict[str, Any]:
    ste_bar = predict_ste_bar(features, x_vals)

    (
        z1,
        z2,
        _cc1,
        _cc2,
        ck1,
        ck2,
        cf1,
        cf2,
        a0,
        _cs1,
        _cs2,
        e_to_sy,
        _nu,
        log_tmid_to_max,
        b_to_m,
        _da12_to_m,
        m,
    ) = features.tolist()

    if sy_ref <= 0:
        raise ValueError("sy_ref must be positive.")

    b = b_to_m * m
    if b <= 0:
        raise ValueError("b_to_m and m must produce positive face width.")

    if abs(np.cos(a0)) < 1e-12:
        raise ValueError("a0 leads to invalid base radii (cos(a0) ~ 0).")

    rg1 = m * z1 / 2.0 / np.cos(a0)
    rg2 = m * z2 / 2.0 / np.cos(a0)
    if rg1 <= 0 or rg2 <= 0:
        raise ValueError("Computed base radii must be positive.")

    e1 = sy_ref * e_to_sy
    e2 = sy_ref * e_to_sy
    e_eq = 2.0 * e1 * e2 / max(1e-12, e1 + e2)

    tmid_max1 = (
        sy_ref
        * m**3
        * b_to_m
        * (z1 / 2.0 + ck1)
        * ((z1 / 2.0 - cf1) * np.pi) ** 2
        / (24.0 * z1**2 * max(1e-12, (ck1 + cf1)))
    )
    tmid_max2 = (
        sy_ref
        * m**3
        * b_to_m
        * (z2 / 2.0 + ck2)
        * ((z2 / 2.0 - cf2) * np.pi) ** 2
        / (24.0 * z2**2 * max(1e-12, (ck2 + cf2)))
    )
    if tmid_max1 <= 0 or tmid_max2 <= 0:
        raise ValueError("Invalid Tmid_max computation; check STE feature values.")

    tmid = (10.0 ** log_tmid_to_max) * np.sqrt(tmid_max1 * tmid_max2)
    t2 = tmid * np.sqrt(z2 / max(1e-12, z1))

    conv_factor = tmid / max(1e-12, e_eq * b * rg1 * rg2)
    ste_phi_mid = ste_bar * conv_factor
    ste_phi_2 = ste_phi_mid * np.sqrt(z1 / max(1e-12, z2))

    with np.errstate(divide="ignore", invalid="ignore"):
        g_arr = np.where(np.abs(ste_phi_2) > 1e-16, t2 / ste_phi_2, np.nan)

    return {
        "x_vals": x_vals.tolist(),
        "ste_bar": ste_bar.tolist(),
        "ste_phi_mid": ste_phi_mid.tolist(),
        "ste_phi_2": ste_phi_2.tolist(),
        "g_nmm_per_rad": g_arr.tolist(),
        "scalars": {
            "conv_factor_bar_to_rad": float(conv_factor),
            "tmid_nmm": float(tmid),
            "t2_nmm": float(t2),
            "e_equivalent_mpa": float(e_eq),
        },
    }


def training_range_report(features: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for i, name in enumerate(FEATURE_NAMES):
        v = float(features[i])
        lo = float(LOWER_BOUNDS[i])
        hi = float(UPPER_BOUNDS[i])
        out.append(
            {
                "name": name,
                "value": v,
                "min": lo,
                "max": hi,
                "in_range": bool(lo <= v <= hi),
            }
        )
    return out

