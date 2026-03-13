from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


@dataclass
class DesignState:
    z1: float
    z2: float
    m: float
    b_mm: float
    alpha_deg: float
    x1: float
    x2: float
    torque_nm: float
    n1_rpm: float
    kv: float
    ka: float
    khb: float
    kfb: float
    sf_min: float
    sh_min: float
    tip_min_mm: float
    undercut_ok: bool
    interference_ok: bool
    eff_pct: float
    contact_ratio: float
    dynamic_index: float
    mass_kg: float
    mat_bend_gain: float = 1.0
    mat_cont_gain: float = 1.0
    quality_gain: float = 1.0


def _baseline(payload: dict[str, Any]) -> DesignState:
    sf_vals = [float(payload.get("agma_sf_min", np.nan)), float(payload.get("iso_sf_min", np.nan))]
    sh_vals = [float(payload.get("agma_sh_min", np.nan)), float(payload.get("iso_sh_min", np.nan))]
    sf_min = min(v for v in sf_vals if np.isfinite(v)) if any(np.isfinite(sf_vals)) else 0.8
    sh_min = min(v for v in sh_vals if np.isfinite(v)) if any(np.isfinite(sh_vals)) else 0.8

    tip1 = float(payload.get("tip1_mm", np.nan))
    tip2 = float(payload.get("tip2_mm", np.nan))
    tip_min = min(v for v in [tip1, tip2] if np.isfinite(v)) if (np.isfinite(tip1) or np.isfinite(tip2)) else 0.0

    return DesignState(
        z1=float(payload["z1"]),
        z2=float(payload["z2"]),
        m=float(payload["m"]),
        b_mm=float(payload["b_mm"]),
        alpha_deg=float(payload["alpha_deg"]),
        x1=float(payload["x1"]),
        x2=float(payload["x2"]),
        torque_nm=float(payload["torque_nm"]),
        n1_rpm=float(payload["n1_rpm"]),
        kv=float(payload.get("kv", 1.1)),
        ka=float(payload.get("ka", 1.0)),
        khb=float(payload.get("khb", 1.3)),
        kfb=float(payload.get("kfb", 1.3)),
        sf_min=float(sf_min),
        sh_min=float(sh_min),
        tip_min_mm=float(tip_min),
        undercut_ok=bool(payload.get("undercut_ok", True)),
        interference_ok=bool(payload.get("interference_ok", True)),
        eff_pct=float(payload.get("efficiency_percent", 97.0)),
        contact_ratio=float(payload.get("contact_ratio", 1.5)),
        dynamic_index=float(payload.get("dynamic_index", 1.0)),
        mass_kg=float(payload.get("mass_kg", 1.0)),
    )


def _clone(base: DesignState) -> DesignState:
    return DesignState(**base.__dict__)


def _predict_safeties(base: DesignState, cand: DesignState) -> tuple[float, float, float]:
    rb = cand.b_mm / max(1e-9, base.b_mm)
    rm = cand.m / max(1e-9, base.m)
    rz = cand.z1 / max(1e-9, base.z1)
    rt = cand.torque_nm / max(1e-9, base.torque_nm)
    rn = cand.n1_rpm / max(1e-9, base.n1_rpm)
    rx1 = cand.x1 - base.x1
    rx2 = cand.x2 - base.x2
    rkq = (cand.kv * cand.khb * cand.kfb * cand.ka) / max(1e-9, base.kv * base.khb * base.kfb * base.ka)

    sf = base.sf_min
    sh = base.sh_min
    sf *= rb**1.0
    sf *= rm**1.75
    sf *= rz**0.65
    sf *= (1.0 / max(1e-9, rt))
    sf *= (1.0 / max(1e-9, rn)) ** 0.08
    sf *= cand.mat_bend_gain
    sf *= cand.quality_gain
    sf *= (1.0 / max(1e-9, rkq)) ** 0.45
    sf *= 1.0 + 0.15 * max(0.0, rx1) + 0.06 * max(0.0, rx2)

    sh *= rb**0.52
    sh *= rm**1.10
    sh *= rz**0.45
    sh *= (1.0 / max(1e-9, rt)) ** 0.50
    sh *= (1.0 / max(1e-9, rn)) ** 0.05
    sh *= cand.mat_cont_gain
    sh *= cand.quality_gain
    sh *= (1.0 / max(1e-9, rkq)) ** 0.35
    sh *= 1.0 + 0.08 * max(0.0, rx1 + rx2)

    tip = base.tip_min_mm + 0.28 * (cand.m - base.m) + 0.35 * max(0.0, cand.x1 - base.x1) * cand.m + 0.20 * max(
        0.0, cand.x2 - base.x2
    ) * cand.m
    return float(sf), float(sh), float(tip)


def _predict_additional(base: DesignState, cand: DesignState) -> tuple[float, float, float]:
    # Contact ratio trend approximation around the current point.
    eps = base.contact_ratio
    eps += 0.10 * (cand.x1 - base.x1 + cand.x2 - base.x2)
    eps += 0.05 * (cand.m - base.m) / max(1e-9, base.m)
    eps += 0.03 * (cand.z1 - base.z1) / max(1e-9, base.z1)
    eps = _clamp(eps, 0.9, 3.0)

    # Dynamic index: lower is better.
    dyn = base.dynamic_index
    dyn *= (cand.n1_rpm / max(1e-9, base.n1_rpm)) ** 0.55
    dyn *= (cand.kv / max(1e-9, base.kv)) ** 0.35
    dyn *= (base.m / max(1e-9, cand.m)) ** 0.22
    dyn *= 1.0 / max(1e-9, cand.quality_gain)
    dyn = _clamp(dyn, 0.01, 1e4)

    # Mass approximation scaling.
    mass = base.mass_kg
    mass *= cand.b_mm / max(1e-9, base.b_mm)
    mass *= (cand.m / max(1e-9, base.m)) ** 2.0
    mass *= ((cand.z1 + cand.z2) / max(1e-9, base.z1 + base.z2)) ** 0.70
    mass = _clamp(mass, 0.001, 1e8)
    return float(eps), float(dyn), float(mass)


def _predict_efficiency(base: DesignState, cand: DesignState, eps: float, dyn: float, mass: float) -> float:
    e = base.eff_pct
    e += 0.10 * (eps - base.contact_ratio)
    e -= 0.07 * max(0.0, (dyn / max(1e-9, base.dynamic_index) - 1.0)) * 100.0
    e -= 0.025 * max(0.0, (mass / max(1e-9, base.mass_kg) - 1.0)) * 100.0
    e += 0.35 * (cand.quality_gain - 1.0) * 10.0
    e -= 0.04 * max(0.0, cand.torque_nm - base.torque_nm) / max(1e-9, base.torque_nm) * 100.0
    return _clamp(e, 60.0, 99.9)


def _scenario_points(
    *,
    sf: float,
    sh: float,
    tip: float,
    eps: float,
    dyn: float,
    eff: float,
    mass: float,
    base_mass: float,
    target_sf: float,
    target_sh: float,
    target_tip: float,
    target_eps: float,
    target_dyn: float,
    target_eff: float,
    cost: float,
    undercut_ok: bool,
    interference_ok: bool,
) -> dict[str, float]:
    p_sf = 30.0 * min(sf / max(1e-9, target_sf), 1.35)
    p_sh = 24.0 * min(sh / max(1e-9, target_sh), 1.35)
    p_tip = 10.0 * min(tip / max(1e-9, target_tip), 1.40)
    p_eps = 10.0 * min(eps / max(1e-9, target_eps), 1.40)
    p_dyn = 10.0 * min(target_dyn / max(1e-9, dyn), 1.40)
    p_eff = 8.0 * min(eff / max(1e-9, target_eff), 1.25)

    p_mass_pen = 0.70 * max(0.0, (mass - base_mass))
    p_geo_pen = 0.0
    if not undercut_ok:
        p_geo_pen += 8.0
    if not interference_ok:
        p_geo_pen += 10.0

    total = p_sf + p_sh + p_tip + p_eps + p_dyn + p_eff - p_mass_pen - p_geo_pen - cost
    return {
        "sf": float(p_sf),
        "sh": float(p_sh),
        "tip": float(p_tip),
        "eps": float(p_eps),
        "dyn": float(p_dyn),
        "eff": float(p_eff),
        "mass_penalty": float(p_mass_pen),
        "geometry_penalty": float(p_geo_pen),
        "cost_penalty": float(cost),
        "total": float(total),
    }


def _apply_action(base: DesignState, action_key: str) -> tuple[DesignState, dict[str, Any]]:
    c = _clone(base)
    meta = {"key": action_key, "title": "", "cost": 0.0, "note": ""}

    if action_key == "b_plus_10":
        c.b_mm *= 1.10
        meta.update(title="Increase face width by 10%", cost=4.0, note="Primary lever for root/contact stress reduction.")
    elif action_key == "b_plus_20":
        c.b_mm *= 1.20
        meta.update(title="Increase face width by 20%", cost=8.0, note="Strong safety gain with mass penalty.")
    elif action_key == "m_plus_0p5":
        c.m += 0.5
        c.b_mm += 1.8
        meta.update(title="Increase module by +0.5 mm", cost=13.0, note="High impact on SF/SH and tip thickness.")
    elif action_key == "m_plus_1p0":
        c.m += 1.0
        c.b_mm += 3.2
        meta.update(title="Increase module by +1.0 mm", cost=25.0, note="Aggressive strengthening action.")
    elif action_key == "z_plus_pair":
        ratio = max(1.01, base.z2 / max(1e-9, base.z1))
        c.z1 = np.round(base.z1 + 3.0)
        c.z2 = np.round(c.z1 * ratio)
        meta.update(title="Increase tooth counts (ratio lock)", cost=9.0, note="Better curvature and smoother kinematics.")
    elif action_key == "x_shift_pinion":
        c.x1 += 0.18
        c.x2 -= 0.05
        meta.update(title="Shift bias to pinion (x1 up)", cost=6.0, note="Improves undercut margin and pinion root safety.")
    elif action_key == "x_shift_both":
        c.x1 += 0.10
        c.x2 += 0.10
        meta.update(title="Positive shifts on both gears", cost=7.0, note="Raises contact ratio and tip thickness reserve.")
    elif action_key == "torque_minus_10":
        c.torque_nm *= 0.90
        meta.update(title="Reduce torque by 10%", cost=4.5, note="Fast relief when overload is dominant.")
    elif action_key == "speed_minus_15":
        c.n1_rpm *= 0.85
        meta.update(title="Reduce speed by 15%", cost=5.5, note="Improves dynamic response and losses.")
    elif action_key == "quality_plus":
        c.kv *= 0.92
        c.khb *= 0.93
        c.kfb *= 0.93
        c.quality_gain *= 1.06
        meta.update(title="Improve quality / micro-geometry", cost=8.5, note="Reduces dynamic and load-distribution factors.")
    elif action_key == "material_plus":
        c.mat_bend_gain *= 1.18
        c.mat_cont_gain *= 1.15
        meta.update(title="Upgrade material/heat treatment", cost=12.5, note="Boosts allowable stresses for SF/SH.")
    elif action_key == "dynamic_tune":
        c.kv *= 0.90
        c.khb *= 0.92
        c.kfb *= 0.92
        c.n1_rpm *= 0.94
        c.quality_gain *= 1.05
        meta.update(title="Dynamic response tuning package", cost=11.0, note="Joint reduction of dynamic excitation and amplification.")
    elif action_key == "contact_ratio_boost":
        c.x1 += 0.08
        c.x2 += 0.08
        c.z1 += 1.0
        c.z2 += 1.0
        meta.update(title="Contact ratio boost package", cost=10.5, note="Targets smoother mesh and reduced transmission ripple.")
    else:
        meta.update(title="No-op", cost=0.0, note="")

    return c, meta


def _finalize_constraints(c: DesignState, x_limit_abs: float, min_z: float, max_z: float) -> DesignState:
    c.x1 = _clamp(c.x1, -x_limit_abs, x_limit_abs)
    c.x2 = _clamp(c.x2, -x_limit_abs, x_limit_abs)
    c.z1 = np.round(_clamp(c.z1, min_z, max_z))
    c.z2 = np.round(_clamp(c.z2, min_z, max_z))
    c.m = _clamp(c.m, 1.0, 12.0)
    c.b_mm = _clamp(c.b_mm, 6.0, 180.0)
    c.torque_nm = _clamp(c.torque_nm, 0.1, 1e6)
    c.n1_rpm = _clamp(c.n1_rpm, 50.0, 2e4)
    return c


def _diagnosis(
    base: DesignState,
    target_sf: float,
    target_sh: float,
    target_tip: float,
    target_eps: float,
    target_dyn: float,
) -> dict[str, Any]:
    tags: list[str] = []
    if base.sf_min < target_sf:
        tags.append("Bending stress limited")
    if base.sh_min < target_sh:
        tags.append("Contact stress limited")
    if base.tip_min_mm < target_tip:
        tags.append("Tip thickness limited")
    if base.contact_ratio < target_eps:
        tags.append("Contact ratio limited")
    if base.dynamic_index > target_dyn:
        tags.append("Dynamic response limited")
    if not base.undercut_ok:
        tags.append("Undercut risk")
    if not base.interference_ok:
        tags.append("Interference risk")
    if base.eff_pct < 95.0:
        tags.append("Efficiency risk")
    if not tags:
        tags = ["Balanced design window"]

    severity = "low"
    if (
        base.sf_min < 0.9 * target_sf
        or base.sh_min < 0.9 * target_sh
        or base.tip_min_mm < 0.75 * target_tip
        or base.dynamic_index > 1.2 * target_dyn
    ):
        severity = "high"
    elif (
        base.sf_min < target_sf
        or base.sh_min < target_sh
        or base.tip_min_mm < target_tip
        or base.contact_ratio < target_eps
        or base.dynamic_index > target_dyn
    ):
        severity = "medium"
    return {"severity": severity, "dominant_modes": tags}


def optimize_design(payload: dict[str, Any]) -> dict[str, Any]:
    base = _baseline(payload)
    target_sf = float(payload.get("target_sf", 1.35))
    target_sh = float(payload.get("target_sh", 1.20))
    target_tip = float(payload.get("target_tip_mm", max(0.2 * base.m, 0.4)))
    target_eps = float(payload.get("target_contact_ratio", 1.40))
    target_dyn = float(payload.get("target_dynamic_index", max(0.1, base.dynamic_index * 0.90)))
    target_eff = float(payload.get("target_efficiency", 97.0))
    x_limit_abs = float(payload.get("x_limit_abs", 1.0))
    min_z = float(payload.get("min_z", 12))
    max_z = float(payload.get("max_z", 200))
    top_k = int(payload.get("top_k", 8))

    action_pool = [
        "b_plus_10",
        "b_plus_20",
        "m_plus_0p5",
        "m_plus_1p0",
        "z_plus_pair",
        "x_shift_pinion",
        "x_shift_both",
        "contact_ratio_boost",
        "torque_minus_10",
        "speed_minus_15",
        "quality_plus",
        "material_plus",
        "dynamic_tune",
    ]

    base_points = _scenario_points(
        sf=base.sf_min,
        sh=base.sh_min,
        tip=base.tip_min_mm,
        eps=base.contact_ratio,
        dyn=base.dynamic_index,
        eff=base.eff_pct,
        mass=base.mass_kg,
        base_mass=base.mass_kg,
        target_sf=target_sf,
        target_sh=target_sh,
        target_tip=target_tip,
        target_eps=target_eps,
        target_dyn=target_dyn,
        target_eff=target_eff,
        cost=0.0,
        undercut_ok=base.undercut_ok,
        interference_ok=base.interference_ok,
    )

    scenarios: list[dict[str, Any]] = []
    sid = 1

    def add_scenario(cand: DesignState, actions: list[str], title: str, note: str, cost: float) -> None:
        nonlocal sid
        c = _finalize_constraints(cand, x_limit_abs=x_limit_abs, min_z=min_z, max_z=max_z)
        sf, sh, tip = _predict_safeties(base, c)
        eps, dyn, mass = _predict_additional(base, c)
        eff = _predict_efficiency(base, c, eps, dyn, mass)

        undercut_ok = base.undercut_ok or (c.x1 > base.x1)
        interference_ok = base.interference_ok or (c.m > base.m and c.z1 >= base.z1)

        pts = _scenario_points(
            sf=sf,
            sh=sh,
            tip=tip,
            eps=eps,
            dyn=dyn,
            eff=eff,
            mass=mass,
            base_mass=base.mass_kg,
            target_sf=target_sf,
            target_sh=target_sh,
            target_tip=target_tip,
            target_eps=target_eps,
            target_dyn=target_dyn,
            target_eff=target_eff,
            cost=cost,
            undercut_ok=undercut_ok,
            interference_ok=interference_ok,
        )
        scenarios.append(
            {
                "id": f"S{sid:03d}",
                "title": title,
                "actions": actions,
                "note": note,
                "score": float(pts["total"]),
                "score_gain": float(pts["total"] - base_points["total"]),
                "points": pts,
                "predicted": {
                    "sf_min": float(sf),
                    "sh_min": float(sh),
                    "tip_min_mm": float(tip),
                    "efficiency_percent": float(eff),
                    "contact_ratio": float(eps),
                    "dynamic_index": float(dyn),
                    "mass_kg": float(mass),
                },
                "params": {
                    "z1": float(c.z1),
                    "z2": float(c.z2),
                    "m": float(c.m),
                    "b_mm": float(c.b_mm),
                    "x1": float(c.x1),
                    "x2": float(c.x2),
                    "torque_nm": float(c.torque_nm),
                    "n1_rpm": float(c.n1_rpm),
                },
            }
        )
        sid += 1

    # Single actions
    for act in action_pool:
        cand, meta = _apply_action(base, act)
        add_scenario(cand, [meta["key"]], meta["title"], meta["note"], float(meta["cost"]))

    # Two-action combos
    for a1, a2 in combinations(action_pool, 2):
        c1, m1 = _apply_action(base, a1)
        c2, m2 = _apply_action(c1, a2)
        add_scenario(
            c2,
            [m1["key"], m2["key"]],
            f"{m1['title']} + {m2['title']}",
            f"{m1['note']} | {m2['note']}",
            float(m1["cost"]) + float(m2["cost"]) + 1.5,
        )

    scenarios.sort(key=lambda s: s["score"], reverse=True)
    top = scenarios[: max(3, top_k)]
    best = top[0] if top else None

    # Sensitivity in "points gain" for core levers
    sensitivity: list[dict[str, Any]] = []
    for lv in ["b_plus_10", "m_plus_0p5", "z_plus_pair", "x_shift_pinion", "quality_plus", "dynamic_tune"]:
        c, meta = _apply_action(base, lv)
        c = _finalize_constraints(c, x_limit_abs=x_limit_abs, min_z=min_z, max_z=max_z)
        sf, sh, tip = _predict_safeties(base, c)
        eps, dyn, mass = _predict_additional(base, c)
        eff = _predict_efficiency(base, c, eps, dyn, mass)
        pts = _scenario_points(
            sf=sf,
            sh=sh,
            tip=tip,
            eps=eps,
            dyn=dyn,
            eff=eff,
            mass=mass,
            base_mass=base.mass_kg,
            target_sf=target_sf,
            target_sh=target_sh,
            target_tip=target_tip,
            target_eps=target_eps,
            target_dyn=target_dyn,
            target_eff=target_eff,
            cost=float(meta["cost"]),
            undercut_ok=base.undercut_ok or (c.x1 > base.x1),
            interference_ok=base.interference_ok or (c.m > base.m and c.z1 >= base.z1),
        )
        gain_pts = float(pts["total"] - base_points["total"])
        sensitivity.append({"lever": meta["title"], "gain_points": gain_pts, "gain_percent": gain_pts})
    sensitivity.sort(key=lambda x: x["gain_percent"], reverse=True)

    diagnosis = _diagnosis(base, target_sf, target_sh, target_tip, target_eps, target_dyn)

    advice: list[str] = []
    if best:
        p = best["predicted"]
        advice.append(
            f'Primary recommendation "{best["title"]}" -> SFmin={p["sf_min"]:.2f}, SHmin={p["sh_min"]:.2f}, '
            f'eps={p["contact_ratio"]:.3f}, dyn={p["dynamic_index"]:.3f}, eta={p["efficiency_percent"]:.2f}%.'
        )
    if base.contact_ratio < target_eps:
        advice.append("Contact ratio is low: prioritize profile-shift and tooth-count actions.")
    if base.dynamic_index > target_dyn:
        advice.append("Dynamic response is high: prioritize quality/dynamic tuning and moderate speed.")
    if base.mass_kg > 0 and best and best["predicted"]["mass_kg"] > 1.20 * base.mass_kg:
        advice.append("Best-score scenario increases mass significantly; inspect next-ranked lower-mass options.")
    if len(advice) == 0:
        advice.append("Current design is close to targets; use optimization mainly for trade-off refinement.")

    return {
        "ok": True,
        "diagnosis": diagnosis,
        "targets": {
            "sf_min": target_sf,
            "sh_min": target_sh,
            "tip_min_mm": target_tip,
            "contact_ratio": target_eps,
            "dynamic_index": target_dyn,
            "efficiency_percent": target_eff,
        },
        "baseline": {
            "sf_min": float(base.sf_min),
            "sh_min": float(base.sh_min),
            "tip_min_mm": float(base.tip_min_mm),
            "efficiency_percent": float(base.eff_pct),
            "contact_ratio": float(base.contact_ratio),
            "dynamic_index": float(base.dynamic_index),
            "mass_kg": float(base.mass_kg),
            "undercut_ok": bool(base.undercut_ok),
            "interference_ok": bool(base.interference_ok),
            "points": base_points,
        },
        "best": best,
        "scenarios": top,
        "sensitivity": sensitivity,
        "advice": advice,
        "points_legend": {
            "sf": "bending safety contribution",
            "sh": "contact safety contribution",
            "tip": "tip-thickness reserve contribution",
            "eps": "contact-ratio contribution",
            "dyn": "dynamic-response contribution (lower dynamic index -> higher points)",
            "eff": "efficiency contribution",
            "mass_penalty": "mass growth penalty",
            "geometry_penalty": "undercut/interference penalty",
            "cost_penalty": "implementation complexity penalty",
        },
    }
