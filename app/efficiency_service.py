from __future__ import annotations

from typing import Any

import numpy as np


def _safe(v: float, lo: float) -> float:
    return float(max(lo, v))


def _lube_family_props(family: str) -> dict[str, float]:
    fam = (family or "mineral").strip().lower()
    table = {
        "mineral": {"k_temp": 0.032, "rho40": 860.0, "mu_factor": 1.00},
        "pao": {"k_temp": 0.026, "rho40": 835.0, "mu_factor": 0.94},
        "ester": {"k_temp": 0.024, "rho40": 900.0, "mu_factor": 0.91},
        "polyglycol": {"k_temp": 0.025, "rho40": 1020.0, "mu_factor": 0.90},
        "bio": {"k_temp": 0.028, "rho40": 920.0, "mu_factor": 0.95},
    }
    return table.get(fam, table["mineral"])


def _additive_factor(additive: str) -> float:
    key = (additive or "none").strip().lower()
    table = {
        "none": 1.00,
        "aw_ep": 0.95,
        "friction_modifier": 0.90,
        "solid_lubricant": 0.88,
    }
    return table.get(key, 1.00)


def _estimate_eta0_pa_s(
    family: str,
    iso_vg: float,
    oil_temp_c: float,
    eta_manual_pa_s: float,
    eta_mode: str,
) -> tuple[float, dict[str, float]]:
    mode = (eta_mode or "auto").strip().lower()
    if mode == "manual":
        eta = _safe(float(eta_manual_pa_s), 0.0002)
        return eta, {"eta0_pa_s": eta, "nu_cst": eta * 1e6 / 860.0, "rho_kgpm3": 860.0}

    props = _lube_family_props(family)
    vg = _safe(float(iso_vg), 10.0)
    t = float(np.clip(oil_temp_c, -10.0, 180.0))
    k = props["k_temp"]
    rho40 = props["rho40"]

    # compact engineering approximation: nu(T)=nu40*exp(-k*(T-40))
    nu_cst = vg * float(np.exp(-k * (t - 40.0)))
    rho = max(700.0, rho40 - 0.65 * (t - 40.0))
    eta = _safe(nu_cst * 1e-6 * rho, 0.0002)
    return eta, {"eta0_pa_s": eta, "nu_cst": nu_cst, "rho_kgpm3": rho}


def _gear_geometry(
    z1: float,
    z2: float,
    m: float,
    alpha_deg: float,
    x1: float,
    x2: float,
    ck: float,
) -> dict[str, float]:
    alpha = np.deg2rad(alpha_deg)
    alpha = float(np.clip(alpha, np.deg2rad(5.0), np.deg2rad(35.0)))
    rw1 = 0.5 * m * z1
    rw2 = 0.5 * m * z2
    rb1 = rw1 * np.cos(alpha)
    rb2 = rw2 * np.cos(alpha)
    ra1 = 0.5 * m * (z1 + 2.0 * (ck + x1))
    ra2 = 0.5 * m * (z2 + 2.0 * (ck + x2))
    a = rw1 + rw2 + m * (x1 + x2)
    return {
        "alpha_rad": float(alpha),
        "rw1": float(rw1),
        "rw2": float(rw2),
        "rb1": float(rb1),
        "rb2": float(rb2),
        "ra1": float(ra1),
        "ra2": float(ra2),
        "a": float(a),
    }


def _line_of_action(geom: dict[str, float], n_points: int) -> dict[str, np.ndarray]:
    rb1 = _safe(geom["rb1"], 1e-9)
    rb2 = _safe(geom["rb2"], 1e-9)
    ra1 = _safe(geom["ra1"], rb1 + 1e-9)
    ra2 = _safe(geom["ra2"], rb2 + 1e-9)
    alpha = geom["alpha_rad"]

    path_pre = np.sqrt(max(0.0, ra2 * ra2 - rb2 * rb2)) - rb2 * np.tan(alpha)
    path_post = np.sqrt(max(0.0, ra1 * ra1 - rb1 * rb1)) - rb1 * np.tan(alpha)
    loa_len = _safe(path_pre + path_post, 1e-9)

    s = np.linspace(-path_pre, path_post, int(max(64, n_points)), dtype=float)
    xi = s / loa_len
    return {
        "s_mm": s,
        "xi": xi,
        "path_pre": np.array([path_pre]),
        "path_post": np.array([path_post]),
        "loa_len": np.array([loa_len]),
    }


def _load_distribution(xi: np.ndarray, s_mm: np.ndarray, eps_alpha: float) -> np.ndarray:
    eps = _safe(eps_alpha, 1.0)
    core = np.clip(1.0 - np.abs(xi) * (2.0 / max(1.0, eps)), 0.25, 1.0)
    edge_boost = 1.0 - 0.10 * np.exp(-((np.abs(xi) - 0.5) / 0.18) ** 2)
    q = core * edge_boost
    area = np.trapz(q, s_mm)
    return q / _safe(area, 1e-12)


def _sliding_speeds(
    s_mm: np.ndarray,
    alpha_rad: float,
    n1_rpm: float,
    rw1: float,
    rw2: float,
    z1: float,
    z2: float,
) -> tuple[np.ndarray, np.ndarray]:
    omega1 = 2.0 * np.pi * n1_rpm / 60.0
    omega2 = omega1 * z1 / _safe(z2, 1e-9)
    v_roll = omega1 * rw1 * 1e-3
    ds = s_mm * 1e-3
    # In involute mesh, sliding changes sign at pitch point and grows away from it.
    # A compact kinematic approximation is proportional to (omega1 + omega2)*distance.
    v_sl = (omega1 + omega2) * ds / _safe(np.cos(alpha_rad), 0.05)
    sr = v_sl / _safe(v_roll, 1e-6)
    return v_sl, sr


def _ehl_mu(
    sr: np.ndarray,
    v_roll: float,
    eta0_pa_s: float,
    load_n_per_mm: float,
    rough_um: float,
    lube_mu_factor: float,
    additive_mu_factor: float,
) -> np.ndarray:
    eta = _safe(eta0_pa_s, 0.002)
    w = _safe(load_n_per_mm, 1.0)
    rough = _safe(rough_um, 0.05)
    base = 0.024 + 0.060 * np.tanh(3.0 * np.abs(sr))
    speed_term = 0.008 * np.log10(1.0 + _safe(v_roll, 0.05))
    load_term = 0.003 * np.log10(1.0 + w)
    rough_term = 0.0025 * rough
    mu = base + speed_term + load_term + rough_term + 0.018 * np.log10(1.0 + 100.0 * eta)
    mu *= _safe(lube_mu_factor, 0.5)
    mu *= _safe(additive_mu_factor, 0.5)
    return np.clip(mu, 0.012, 0.16)


def _film_thickness_and_lambda(
    eta0_pa_s: float,
    v_roll: float,
    load_n_per_mm: float,
    rough_um: float,
    sr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # EHL-inspired screening model (not a strict full Hamrock-Dowson implementation)
    eta = _safe(eta0_pa_s, 1e-4)
    vr = _safe(v_roll, 1e-3)
    w = _safe(load_n_per_mm, 0.2)
    rq = _safe(rough_um, 0.03)
    h0 = 0.22 * (eta / 0.01) ** 0.67 * (vr / 1.0) ** 0.67 * (50.0 / w) ** 0.12
    h_um = h0 * (1.0 + 0.16 * np.tanh(2.5 * np.abs(sr)))
    lam = h_um / _safe(np.sqrt(2.0) * rq, 1e-4)
    return h_um, lam


def compute_efficiency_outputs(payload: dict[str, float]) -> dict[str, Any]:
    z1 = float(payload["z1"])
    z2 = float(payload["z2"])
    m = float(payload["m"])
    alpha_deg = float(payload["alpha_deg"])
    x1 = float(payload["x1"])
    x2 = float(payload["x2"])
    ck = float(payload["ck"])
    b = float(payload["b_mm"])
    n1 = float(payload["n1_rpm"])
    torque_nm = float(payload["torque_nm"])
    eps_alpha = float(payload.get("eps_alpha", 1.5))
    eta0 = float(payload.get("eta0_pa_s", 0.02))
    eta_mode = str(payload.get("eta_mode", "auto"))
    lube_family = str(payload.get("lube_family", "mineral"))
    iso_vg = float(payload.get("iso_vg", 68.0))
    oil_temp_c = float(payload.get("oil_temp_c", 60.0))
    additive = str(payload.get("additive", "none"))
    rough = float(payload.get("roughness_um", 0.30))
    n_pts = int(payload.get("n_points", 700))

    eta0_used, lube_state = _estimate_eta0_pa_s(
        family=lube_family,
        iso_vg=iso_vg,
        oil_temp_c=oil_temp_c,
        eta_manual_pa_s=eta0,
        eta_mode=eta_mode,
    )
    fam_props = _lube_family_props(lube_family)
    add_factor = _additive_factor(additive)

    geom = _gear_geometry(z1, z2, m, alpha_deg, x1, x2, ck)
    loa = _line_of_action(geom, n_pts)
    s_mm = loa["s_mm"]
    xi = loa["xi"]
    q = _load_distribution(xi, s_mm, eps_alpha)

    v_sl, sr = _sliding_speeds(
        s_mm=s_mm,
        alpha_rad=geom["alpha_rad"],
        n1_rpm=n1,
        rw1=geom["rw1"],
        rw2=geom["rw2"],
        z1=z1,
        z2=z2,
    )
    omega1 = 2.0 * np.pi * n1 / 60.0
    v_roll = 0.5 * (omega1 * geom["rw1"] * 1e-3 + (omega1 * z1 / _safe(z2, 1e-9)) * geom["rw2"] * 1e-3)

    Ft = 2.0 * torque_nm / _safe(geom["rw1"] * 1e-3, 1e-9)
    load_n_per_mm = Ft / _safe(b, 1e-9)
    mu = _ehl_mu(
        sr=sr,
        v_roll=v_roll,
        eta0_pa_s=eta0_used,
        load_n_per_mm=load_n_per_mm,
        rough_um=rough,
        lube_mu_factor=fam_props["mu_factor"],
        additive_mu_factor=add_factor,
    )
    h_um, lam = _film_thickness_and_lambda(
        eta0_pa_s=eta0_used,
        v_roll=v_roll,
        load_n_per_mm=load_n_per_mm,
        rough_um=rough,
        sr=sr,
    )

    dP = np.abs(mu * Ft * v_sl) * q
    p_loss = float(np.trapz(dP, s_mm))
    p_in = float(max(1e-9, torque_nm * omega1))
    eta_mesh = float(np.clip(1.0 - p_loss / p_in, 0.0, 1.0))

    return {
        "s_mm": s_mm.tolist(),
        "load_distribution": q.tolist(),
        "sliding_speed_mps": v_sl.tolist(),
        "sliding_ratio": sr.tolist(),
        "mu_ehl": mu.tolist(),
        "dP_loss_w_per_mm": dP.tolist(),
        "film_thickness_um": h_um.tolist(),
        "lambda_ratio": lam.tolist(),
        "scalars": {
            "Ft_n": float(Ft),
            "v_roll_mps": float(v_roll),
            "P_loss_w": p_loss,
            "P_in_w": p_in,
            "efficiency_percent": float(100.0 * eta_mesh),
            "LOA_mm": float(loa["loa_len"][0]),
            "path_pre_mm": float(loa["path_pre"][0]),
            "path_post_mm": float(loa["path_post"][0]),
            "eta0_used_pa_s": float(eta0_used),
            "nu_used_cst": float(lube_state["nu_cst"]),
            "rho_used_kgpm3": float(lube_state["rho_kgpm3"]),
            "mu_mean": float(np.mean(mu)),
            "lambda_mean": float(np.mean(lam)),
        },
        "lube": {
            "mode": eta_mode,
            "family": lube_family,
            "iso_vg": float(iso_vg),
            "oil_temp_c": float(oil_temp_c),
            "additive": additive,
        },
    }
