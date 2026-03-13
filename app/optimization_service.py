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
    )


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

    sf *= rb ** 1.0
    sf *= rm ** 1.75
    sf *= rz ** 0.65
    sf *= (1.0 / max(1e-9, rt))
    sf *= (1.0 / max(1e-9, rn)) ** 0.08
    sf *= cand.mat_bend_gain
    sf *= cand.quality_gain
    sf *= (1.0 / max(1e-9, rkq)) ** 0.45
    sf *= 1.0 + 0.15 * max(0.0, rx1) + 0.06 * max(0.0, rx2)

    sh *= rb ** 0.52
    sh *= rm ** 1.10
    sh *= rz ** 0.45
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


def _predict_efficiency(base: DesignState, cand: DesignState) -> float:
    e = base.eff_pct
    e += 0.06 * (cand.m - base.m)
    e -= 0.03 * max(0.0, cand.b_mm - base.b_mm)
    e += 0.55 * (cand.quality_gain - 1.0) * 10.0
    e -= 0.08 * max(0.0, cand.torque_nm - base.torque_nm) / max(1e-9, base.torque_nm) * 100.0
    e -= 0.02 * max(0.0, cand.n1_rpm - base.n1_rpm) / max(1e-9, base.n1_rpm) * 100.0
    return _clamp(e, 60.0, 99.9)


def _clone(base: DesignState) -> DesignState:
    return DesignState(**base.__dict__)


def _apply_action(base: DesignState, action_key: str) -> tuple[DesignState, dict[str, Any]]:
    c = _clone(base)
    meta = {"key": action_key, "title": "", "cost": 0.0, "note": ""}

    if action_key == "b_plus_10":
        c.b_mm *= 1.10
        meta.update(title="Increase face width by 10%", cost=4.0, note="Best first move for bending + contact stress.")
    elif action_key == "b_plus_20":
        c.b_mm *= 1.20
        meta.update(title="Increase face width by 20%", cost=8.0, note="Strong safety improvement with moderate mass growth.")
    elif action_key == "m_plus_0p5":
        c.m += 0.5
        c.b_mm += 1.8
        meta.update(title="Increase module by +0.5 mm", cost=14.0, note="High impact on both root and contact capacity.")
    elif action_key == "m_plus_1p0":
        c.m += 1.0
        c.b_mm += 3.2
        meta.update(title="Increase module by +1.0 mm", cost=26.0, note="Aggressive geometry upgrade with large safety reserve.")
    elif action_key == "z_plus_pair":
        ratio = max(1.01, base.z2 / max(1e-9, base.z1))
        c.z1 = np.round(base.z1 + 3.0)
        c.z2 = np.round(c.z1 * ratio)
        meta.update(title="Increase tooth counts with ratio lock", cost=10.0, note="Improves curvature/strength at similar transmission ratio.")
    elif action_key == "x_shift_pinion":
        c.x1 += 0.18
        c.x2 -= 0.05
        meta.update(
            title="Redistribute profile shifts (x1 up)",
            cost=6.5,
            note="Improves pinion root robustness and undercut margin.",
        )
    elif action_key == "x_shift_both":
        c.x1 += 0.12
        c.x2 += 0.10
        meta.update(
            title="Apply positive profile shifts on both gears",
            cost=7.5,
            note="Raises tooth-root safety and usually improves tip thickness.",
        )
    elif action_key == "torque_minus_10":
        c.torque_nm *= 0.90
        meta.update(title="Reduce operating torque by 10%", cost=5.0, note="Useful when available power margin allows de-rating.")
    elif action_key == "torque_minus_20":
        c.torque_nm *= 0.80
        meta.update(title="Reduce operating torque by 20%", cost=9.0, note="Fastest route when overload is the dominant source.")
    elif action_key == "speed_minus_15":
        c.n1_rpm *= 0.85
        meta.update(title="Reduce input speed by 15%", cost=5.5, note="Lowers dynamic amplification and thermal loading.")
    elif action_key == "quality_plus":
        c.kv *= 0.92
        c.khb *= 0.93
        c.kfb *= 0.93
        c.quality_gain *= 1.06
        meta.update(title="Improve gear quality / micro-geometry", cost=8.5, note="Reduces dynamic and distribution factors.")
    elif action_key == "material_plus":
        c.mat_bend_gain *= 1.18
        c.mat_cont_gain *= 1.15
        meta.update(title="Upgrade material / heat treatment", cost=13.0, note="Strong allowable-stress increase for both modes.")
    else:
        meta.update(title="No-op", cost=0.0, note="")
    return c, meta


def _scenario_score(
    sf: float,
    sh: float,
    tip_mm: float,
    target_sf: float,
    target_sh: float,
    target_tip_mm: float,
    cost: float,
    eff_pct: float,
    undercut_ok: bool,
    interference_ok: bool,
) -> float:
    pass_sf = sf / max(1e-9, target_sf)
    pass_sh = sh / max(1e-9, target_sh)
    pass_tip = tip_mm / max(1e-9, target_tip_mm)
    hard = min(pass_sf, pass_sh, pass_tip)

    geom_pen = 0.0
    if not undercut_ok:
        geom_pen += 10.0
    if not interference_ok:
        geom_pen += 12.0

    eff_bonus = max(-2.0, min(2.0, (eff_pct - 97.0) * 0.18))
    return 100.0 * hard - cost - geom_pen + eff_bonus


def _diagnosis(base: DesignState, target_sf: float, target_sh: float, target_tip_mm: float) -> dict[str, Any]:
    tags: list[str] = []
    if base.sf_min < target_sf:
        tags.append("Bending stress limited")
    if base.sh_min < target_sh:
        tags.append("Contact stress limited")
    if base.tip_min_mm < target_tip_mm:
        tags.append("Tip thickness limited")
    if not base.undercut_ok:
        tags.append("Undercut risk")
    if not base.interference_ok:
        tags.append("Interference risk")
    if base.eff_pct < 95.0:
        tags.append("Efficiency risk")
    if not tags:
        tags = ["Balanced design window"]

    sev = "low"
    if (base.sf_min < 0.9 * target_sf) or (base.sh_min < 0.9 * target_sh) or (base.tip_min_mm < 0.75 * target_tip_mm):
        sev = "high"
    elif (base.sf_min < target_sf) or (base.sh_min < target_sh) or (base.tip_min_mm < target_tip_mm):
        sev = "medium"
    return {"severity": sev, "dominant_modes": tags}


def _sensitivity(base: DesignState, target_sf: float, target_sh: float, target_tip_mm: float) -> list[dict[str, Any]]:
    levers = ["b_plus_10", "m_plus_0p5", "z_plus_pair", "x_shift_pinion", "quality_plus", "material_plus"]
    out: list[dict[str, Any]] = []
    s0 = min(base.sf_min / max(1e-9, target_sf), base.sh_min / max(1e-9, target_sh), base.tip_min_mm / max(1e-9, target_tip_mm))
    for lv in levers:
        cand, meta = _apply_action(base, lv)
        sf, sh, tip = _predict_safeties(base, cand)
        s1 = min(sf / max(1e-9, target_sf), sh / max(1e-9, target_sh), tip / max(1e-9, target_tip_mm))
        out.append(
            {
                "lever": meta["title"],
                "gain_percent": float((s1 - s0) * 100.0),
            }
        )
    out.sort(key=lambda x: x["gain_percent"], reverse=True)
    return out


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


def optimize_design(payload: dict[str, Any]) -> dict[str, Any]:
    base = _baseline(payload)
    target_sf = float(payload.get("target_sf", 1.35))
    target_sh = float(payload.get("target_sh", 1.20))
    target_tip_mm = float(payload.get("target_tip_mm", max(0.2 * base.m, 0.4)))
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
        "torque_minus_10",
        "torque_minus_20",
        "speed_minus_15",
        "quality_plus",
        "material_plus",
    ]

    scenarios: list[dict[str, Any]] = []
    sid = 1

    # single actions
    for act in action_pool:
        c0, meta = _apply_action(base, act)
        c = _finalize_constraints(c0, x_limit_abs=x_limit_abs, min_z=min_z, max_z=max_z)
        sf, sh, tip = _predict_safeties(base, c)
        eff = _predict_efficiency(base, c)
        score = _scenario_score(
            sf=sf,
            sh=sh,
            tip_mm=tip,
            target_sf=target_sf,
            target_sh=target_sh,
            target_tip_mm=target_tip_mm,
            cost=float(meta["cost"]),
            eff_pct=eff,
            undercut_ok=base.undercut_ok or (c.x1 > base.x1),
            interference_ok=base.interference_ok or (c.m > base.m and c.z1 >= base.z1),
        )
        scenarios.append(
            {
                "id": f"S{sid:03d}",
                "title": meta["title"],
                "actions": [meta["key"]],
                "note": meta["note"],
                "score": float(score),
                "predicted": {
                    "sf_min": float(sf),
                    "sh_min": float(sh),
                    "tip_min_mm": float(tip),
                    "efficiency_percent": float(eff),
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

    # 2-action combos for flexible strategy suggestions
    for a1, a2 in combinations(action_pool, 2):
        c1, m1 = _apply_action(base, a1)
        c2, m2 = _apply_action(c1, a2)
        c = _finalize_constraints(c2, x_limit_abs=x_limit_abs, min_z=min_z, max_z=max_z)
        sf, sh, tip = _predict_safeties(base, c)
        eff = _predict_efficiency(base, c)
        score = _scenario_score(
            sf=sf,
            sh=sh,
            tip_mm=tip,
            target_sf=target_sf,
            target_sh=target_sh,
            target_tip_mm=target_tip_mm,
            cost=float(m1["cost"]) + float(m2["cost"]) + 1.5,
            eff_pct=eff,
            undercut_ok=base.undercut_ok or (c.x1 > base.x1),
            interference_ok=base.interference_ok or (c.m > base.m and c.z1 >= base.z1),
        )
        title = f"{m1['title']} + {m2['title']}"
        note = f"{m1['note']} | {m2['note']}"
        scenarios.append(
            {
                "id": f"S{sid:03d}",
                "title": title,
                "actions": [m1["key"], m2["key"]],
                "note": note,
                "score": float(score),
                "predicted": {
                    "sf_min": float(sf),
                    "sh_min": float(sh),
                    "tip_min_mm": float(tip),
                    "efficiency_percent": float(eff),
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

    scenarios.sort(key=lambda s: s["score"], reverse=True)
    best = scenarios[0] if scenarios else None
    top = scenarios[: max(3, top_k)]
    diagnosis = _diagnosis(base, target_sf, target_sh, target_tip_mm)
    sensitivity = _sensitivity(base, target_sf, target_sh, target_tip_mm)

    actions_txt: list[str] = []
    if best:
        p = best["predicted"]
        actions_txt.append(
            f'Primary recommendation: "{best["title"]}" -> predicted SFmin={p["sf_min"]:.2f}, SHmin={p["sh_min"]:.2f}, eta={p["efficiency_percent"]:.2f}%.'
        )
    if base.sf_min < target_sf:
        actions_txt.append("Bending is below target: prioritize face width/module/material upgrades.")
    if base.sh_min < target_sh:
        actions_txt.append("Contact is below target: prioritize module, quality factors and surface capacity.")
    if base.tip_min_mm < target_tip_mm:
        actions_txt.append("Tip thickness is below threshold: increase m and/or apply positive profile shifts (x1, x2).")
    if not base.undercut_ok:
        actions_txt.append("Undercut risk detected: raise z1 or increase positive profile shift at pinion.")
    if not base.interference_ok:
        actions_txt.append("Interference risk detected: adjust center distance / profile shifts / tip geometry.")
    if len(actions_txt) == 0:
        actions_txt.append("Current design is near target; use optimization for weight/efficiency trade-off refinement.")

    return {
        "ok": True,
        "diagnosis": diagnosis,
        "targets": {"sf_min": target_sf, "sh_min": target_sh, "tip_min_mm": target_tip_mm},
        "baseline": {
            "sf_min": float(base.sf_min),
            "sh_min": float(base.sh_min),
            "tip_min_mm": float(base.tip_min_mm),
            "efficiency_percent": float(base.eff_pct),
            "undercut_ok": bool(base.undercut_ok),
            "interference_ok": bool(base.interference_ok),
        },
        "best": best,
        "scenarios": top,
        "sensitivity": sensitivity,
        "advice": actions_txt,
    }

