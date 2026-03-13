from __future__ import annotations

from typing import Any

import numpy as np


def _safe(v: float, lo: float) -> float:
    return float(max(lo, v))


def _trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    # Numpy API differences across versions:
    # - newer versions provide np.trapezoid
    # - legacy versions provide np.trapz
    # This keeps the service robust on different Render build images.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    if yy.size < 2 or xx.size < 2:
        return 0.0
    return float(np.sum((yy[1:] + yy[:-1]) * (xx[1:] - xx[:-1]) * 0.5))


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
        "s_start": np.array([-path_pre]),
        "s_end": np.array([path_post]),
        "path_pre": np.array([path_pre]),
        "path_post": np.array([path_post]),
        "loa_len": np.array([loa_len]),
    }


def _load_sharing_profile(s_mm: np.ndarray, loa_len_mm: float, eps_alpha: float) -> tuple[np.ndarray, dict[str, float]]:
    # Piecewise load-sharing ratio from the MATLAB reference:
    # A-B (double contact ramp), B-D (single contact), D-E (double contact ramp down)
    # with discontinuities at B and D.
    eps = float(np.clip(eps_alpha, 1.02, 1.98))
    sA = float(np.min(s_mm))
    sE = float(np.max(s_mm))
    L = _safe(loa_len_mm, 1e-9)
    delta_pb = L / eps
    sB = sA + (eps - 1.0) * delta_pb
    sD = sA + 1.0 * delta_pb

    r_m = np.zeros_like(s_mm, dtype=float)
    den = max(1e-9, eps - 1.0)
    for i, s in enumerate(s_mm):
        if s <= sB:
            # 0.36 -> 0.64
            r_m[i] = 0.36 + (0.28 / den) * ((s - sA) / max(delta_pb, 1e-9))
        elif s <= sD:
            # Single pair contact
            r_m[i] = 1.0
        else:
            # 0.64 -> 0.36
            r_m[i] = 0.36 - (0.28 / den) * (((s - sA) / max(delta_pb, 1e-9)) - eps)
    r_m = np.clip(r_m, 0.0, 1.25)
    return r_m, {"sA": sA, "sB": sB, "sP": 0.0, "sD": sD, "sE": sE}


def _sliding_kinematics(
    s_mm: np.ndarray,
    n1_rpm: float,
    rw1: float,
    rw2: float,
    z1: float,
    z2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    omega1 = 2.0 * np.pi * n1_rpm / 60.0
    omega2 = -omega1 * z1 / _safe(z2, 1e-9)
    ds = s_mm * 1e-3
    # Compact LOA-based representation: signed sliding speed around pitch point.
    v_sl_signed = (abs(omega1) + abs(omega2)) * ds
    v_rel = np.abs(v_sl_signed)
    v_roll = 0.5 * (abs(omega1) * rw1 * 1e-3 + abs(omega2) * rw2 * 1e-3)
    sr = v_sl_signed / _safe(v_roll, 1e-9)
    return v_rel, sr, float(v_roll)


def _ehl_traction_and_losses(
    s_mm: np.ndarray,
    theta_arr: np.ndarray,
    v_rel: np.ndarray,
    sr: np.ndarray,
    fn_shared: np.ndarray,
    r_eq_m: np.ndarray,
    b_mm: float,
    omega1: float,
    z1: float,
    rpm_pinion: float,
    p_in: float,
    eta0_pa_s: float,
    alpha_pv: float,
    mu_lim: float,
    e_red: float,
    rough_um: float,
    omega2: float,
    ro1_mm: float,
    ro2_mm: float,
    lube_mu_factor: float,
    additive_mu_factor: float,
) -> dict[str, Any]:
    mm = 1e-3
    b_m = _safe(b_mm * mm, 1e-9)
    eta0 = _safe(eta0_pa_s, 2e-4)
    a_pv = _safe(alpha_pv, 1e-10)
    mu_lim = float(np.clip(mu_lim, 0.03, 0.20))

    w_line = fn_shared / b_m
    a_hz = np.sqrt(np.maximum(4.0 * w_line * r_eq_m / (np.pi * _safe(e_red, 1e6)), 0.0))
    p0 = w_line / np.maximum(np.pi * a_hz, 1e-12)
    p_mean = (np.pi / 4.0) * p0

    # Mean entrainment speed at pitch circles (same model form as MATLAB script)
    u_e = 0.5 * (abs(omega1) * ro1_mm + abs(omega2) * ro2_mm) * mm
    U = eta0 * np.full_like(v_rel, u_e) / np.maximum(e_red * r_eq_m, 1e-12)
    G = a_pv * e_red
    W = w_line / np.maximum(e_red * r_eq_m, 1e-12)
    h_c = 2.69 * r_eq_m * (np.maximum(U, 1e-12) ** 0.67) * (max(G, 1e-12) ** 0.53) * (np.maximum(W, 1e-12) ** -0.067)
    h_c = np.maximum(h_c, 5e-9)

    a_eff = min(a_pv, 3e-8)
    eta_p = eta0 * np.exp(np.minimum(a_eff * p_mean, 25.0))
    tau0 = 5e6
    gamma_dot = v_rel / np.maximum(h_c, 5e-9)
    tau_e = eta_p * gamma_dot
    tau = tau0 * np.arcsinh(tau_e / tau0)
    mu_eyr = tau / np.maximum(p_mean, 1e-12)
    mu_eyr *= _safe(lube_mu_factor, 0.5)
    mu_eyr *= _safe(additive_mu_factor, 0.5)

    h_um = h_c * 1e6
    lam = h_um / _safe(np.sqrt(2.0) * max(rough_um, 0.03), 1e-6)

    # Sliding-dependent traction buildup around the pitch point.
    sr_abs = np.abs(sr)
    f_sr = 1.0 - np.exp(-sr_abs / 0.06)
    mu_fluid = 0.002 + mu_eyr * (0.25 + 0.75 * f_sr)

    # Boundary / mixed component: stronger with roughness and load, weaker with speed.
    p_gpa = p_mean / 1e9
    mu_b = 0.075 * _safe(lube_mu_factor, 0.5) * _safe(additive_mu_factor, 0.5)
    mu_b *= 1.0 + 0.25 * max(0.0, rough_um / 0.30 - 1.0)
    mu_b *= np.maximum(0.80, np.minimum(1.35, (np.maximum(p_gpa, 0.2) / 1.2) ** 0.08))
    mu_b *= np.maximum(0.75, np.minimum(1.20, (0.015 / np.maximum(eta0, 3e-4)) ** 0.08))
    mu_b *= np.maximum(0.70, np.minimum(1.15, (2.0 / np.maximum(u_e, 0.2)) ** 0.06))
    mu_b = np.clip(mu_b, 0.03, 0.16)

    # Lambda-based blending (full-film EHL to mixed/boundary).
    f_ehl = 1.0 - np.exp(-(np.maximum(lam, 0.0) / 1.5) ** 2.0)
    mu = f_ehl * mu_fluid + (1.0 - f_ehl) * mu_b
    mu = np.clip(mu, 0.002, mu_lim)

    p_fric = mu * fn_shared * v_rel
    e_per_tooth = _trapz_integral(p_fric, theta_arr) / _safe(abs(omega1), 1e-9)
    mesh_rate = z1 * rpm_pinion / 60.0
    p_loss = mesh_rate * e_per_tooth
    eta_mesh = float(np.clip(1.0 - p_loss / _safe(p_in, 1e-9), 0.0, 1.0))

    # Keep local friction-power evolution along the LOA for plotting.
    dP_per_mm = np.abs(p_fric)

    return {
        "mu": mu,
        "p_fric": p_fric,
        "dP_per_mm": dP_per_mm,
        "h_um": h_um,
        "lam": lam,
        "p_mean": p_mean,
        "p_loss": float(p_loss),
        "eta_mesh": eta_mesh,
    }


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
    alpha_pv = float(payload.get("alpha_pv", 20e-8))
    mu_lim = float(payload.get("mu_lim", 0.11))
    e_mpa = float(payload.get("e_mpa", 206000.0))
    nu = float(payload.get("nu", 0.30))
    rough = float(payload.get("roughness_um", 0.30))
    n_pts = int(payload.get("n_points", 1200))

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
    r_m, regions = _load_sharing_profile(s_mm, float(loa["loa_len"][0]), eps_alpha)

    v_rel, sr, v_roll = _sliding_kinematics(
        s_mm=s_mm,
        n1_rpm=n1,
        rw1=geom["rw1"],
        rw2=geom["rw2"],
        z1=z1,
        z2=z2,
    )
    omega1 = 2.0 * np.pi * n1 / 60.0
    omega2 = -omega1 * z1 / _safe(z2, 1e-9)
    p_in = float(max(1e-9, torque_nm * omega1))

    ft = 2.0 * torque_nm / _safe(geom["rw1"] * 1e-3, 1e-9)
    fn = ft / _safe(np.cos(geom["alpha_rad"]), 1e-6)
    fn_shared = fn * r_m

    e_pa = _safe(e_mpa * 1e6, 1e8)
    nu = float(np.clip(nu, 0.05, 0.49))
    e_red = 1.0 / (2.0 * (1.0 - nu**2) / e_pa)
    r_eq0 = (_safe(geom["rw1"], 1e-6) * _safe(geom["rw2"], 1e-6)) / _safe(geom["rw1"] + geom["rw2"], 1e-6)
    pos = np.abs(s_mm) / _safe(0.5 * float(loa["loa_len"][0]), 1e-9)
    r_eq_m = (r_eq0 * (1.0 + 0.25 * pos + 0.10 * pos**2)) * 1e-3

    theta_arr = (s_mm * 1e-3) / _safe(geom["rw1"] * 1e-3, 1e-9)
    ehl = _ehl_traction_and_losses(
        s_mm=s_mm,
        theta_arr=theta_arr,
        v_rel=v_rel,
        sr=sr,
        fn_shared=fn_shared,
        r_eq_m=r_eq_m,
        b_mm=b,
        omega1=omega1,
        z1=z1,
        rpm_pinion=n1,
        p_in=p_in,
        eta0_pa_s=eta0_used,
        alpha_pv=alpha_pv,
        mu_lim=mu_lim,
        e_red=e_red,
        rough_um=rough,
        omega2=omega2,
        ro1_mm=geom["rw1"],
        ro2_mm=geom["rw2"],
        lube_mu_factor=fam_props["mu_factor"],
        additive_mu_factor=add_factor,
    )
    mu = np.asarray(ehl["mu"], dtype=float)
    dP = np.asarray(ehl["dP_per_mm"], dtype=float)
    h_um = np.asarray(ehl["h_um"], dtype=float)
    lam = np.asarray(ehl["lam"], dtype=float)
    p_mean = np.asarray(ehl["p_mean"], dtype=float)
    p_loss = float(ehl["p_loss"])
    eta_mesh = float(ehl["eta_mesh"])

    return {
        "s_mm": s_mm.tolist(),
        "load_distribution": r_m.tolist(),
        "normal_load_shared_n": fn_shared.tolist(),
        "sliding_speed_mps": v_rel.tolist(),
        "sliding_ratio": sr.tolist(),
        "mu_ehl": mu.tolist(),
        "dP_loss_w_per_mm": dP.tolist(),
        "film_thickness_um": h_um.tolist(),
        "lambda_ratio": lam.tolist(),
        "mean_hertz_pressure_pa": p_mean.tolist(),
        "regions": regions,
        "scalars": {
            "Ft_n": float(ft),
            "Fn_n": float(fn),
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
            "h_um_mean": float(np.mean(h_um)),
            "hertz_mean_mpa": float(np.mean(p_mean) / 1e6),
        },
        "lube": {
            "mode": eta_mode,
            "family": lube_family,
            "iso_vg": float(iso_vg),
            "oil_temp_c": float(oil_temp_c),
            "additive": additive,
        },
    }
