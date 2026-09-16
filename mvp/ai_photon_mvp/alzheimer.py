"""Demonstrador 4: triagem morfométrica de atrofia tipo Alzheimer.

Não diagnostica doença de Alzheimer (isso é PET/líquor). O papel do PCCT
aqui, na proposta, é morfometria de atrofia e rastreamento oportunista em
TC de crânio. Sem dado NAEOTOM real, o teste PCCT é simulação em imagem
reconstruída: mesmo cérebro digital, TC convencional (mais ruído, menos
contraste) vs. PCCT (menos ruído eletrônico, melhor contraste GM/WM).
"""

import numpy as np
from scipy import ndimage as ndi

from ai_photon_mvp import neuro
from ai_photon_mvp import pipeline as pl

# HU aproximados em imagem reconstruída (não são espectros NIST).
HU_EID = {"air": -1000.0, "skull": 900.0, "csf": 12.0, "gm": 34.0, "wm": 30.0}
HU_PCCT = {"air": -1000.0, "skull": 1100.0, "csf": 8.0, "gm": 42.0, "wm": 26.0}
NOISE = {"eid": 12.0, "pcct": 6.0}

PHENOTYPES = {
    "controle": {
        "vent_rx": 5.0, "vent_ry": 8.0, "vent_rz": 7.0, "vent_dx": 6.0,
        "csf_rim": 1,
    },
    "atrofia_alzheimer": {
        "vent_rx": 12.0, "vent_ry": 18.0, "vent_rz": 14.0, "vent_dx": 12.0,
        "csf_rim": 3,
    },
}


def _evans(vent, icv, spacing):
    if not vent.any() or not icv.any():
        return None
    z0 = int(np.argmax(vent.sum(axis=(1, 2))))
    vy = np.argwhere(vent[z0])
    iy = np.argwhere(icv[z0])
    if not len(vy) or not len(iy):
        return None
    w_v = (vy[:, 1].max() - vy[:, 1].min()) * spacing[2]
    w_i = (iy[:, 1].max() - iy[:, 1].min()) * spacing[2]
    return float(w_v / w_i) if w_i > 0 else None


def _metrics_from_masks(icv, csf, vent, spacing):
    voxel_ml = np.prod(spacing) / 1000.0
    icv_ml = float(icv.sum() * voxel_ml)
    csf_ml = float(csf.sum() * voxel_ml)
    vent_ml = float(vent.sum() * voxel_ml)
    evans = _evans(vent, icv, spacing)
    return {
        "icv_mL": round(icv_ml, 0),
        "csf_mL": round(csf_ml, 0),
        "ventricles_mL": round(vent_ml, 1),
        "csf_fraction_pct": round(100 * csf_ml / icv_ml, 1) if icv_ml else None,
        "vent_icv_pct": round(100 * vent_ml / icv_ml, 2) if icv_ml else None,
        "evans_like_index": round(evans, 3) if evans is not None else None,
    }


def make_brain_phantom(phenotype="atrofia_alzheimer", shape=(88, 150, 136),
                       spacing_mm=1.5):
    """Cérebro digital com crânio, GM, WM, líquor e ventrículos.

    phenotype: 'controle' ou 'atrofia_alzheimer'.
    """
    p = PHENOTYPES[phenotype]
    z, y, x = np.indices(shape).astype(np.float32)
    cz, cy, cx = [s / 2.0 for s in shape]
    sp = float(spacing_mm)

    head = (((z - cz) / (shape[0] * 0.42)) ** 2
            + ((y - cy) / 62.0) ** 2
            + ((x - cx) / 54.0) ** 2) <= 1.0
    inner = ndi.binary_erosion(head, iterations=4)
    skull = head & ~inner
    icv = inner.copy()

    vent = np.zeros(shape, dtype=bool)
    rx, ry, rz, dx = p["vent_rx"], p["vent_ry"], p["vent_rz"], p["vent_dx"]
    for sign in (-1.0, 1.0):
        vent |= (((z - cz) / rz) ** 2
                 + ((y - cy + 4) / ry) ** 2
                 + ((x - (cx + sign * dx)) / rx) ** 2) <= 1.0
    vent &= icv

    rim = icv & ~ndi.binary_erosion(icv, iterations=p["csf_rim"])
    csf = (vent | rim) & icv
    par = icv & ~csf
    dist = ndi.distance_transform_edt(par)
    gm = par & (dist <= 4)
    wm = par & ~gm

    labels = np.zeros(shape, dtype=np.int8)
    labels[skull] = 1
    labels[csf] = 2
    labels[gm] = 3
    labels[wm] = 4

    spacing = (sp, sp, sp)
    truth = _metrics_from_masks(icv, csf, vent, spacing)
    return {
        "labels": labels,
        "masks": {"icv": icv, "csf": csf, "gm": gm, "wm": wm,
                  "ventricles": vent, "skull": skull},
        "spacing": spacing,
        "phenotype": phenotype,
        "truth": truth,
    }


def compose(labels, table):
    hu = np.full(labels.shape, table["air"], dtype=np.float32)
    hu[labels == 1] = table["skull"]
    hu[labels == 2] = table["csf"]
    hu[labels == 3] = table["gm"]
    hu[labels == 4] = table["wm"]
    return hu


def acquire(labels, scanner="pcct", rng=None):
    rng = rng or np.random.default_rng(0)
    table = HU_PCCT if scanner == "pcct" else HU_EID
    hu = compose(labels, table)
    return hu + rng.normal(0.0, NOISE[scanner], hu.shape).astype(np.float32)


def cnr_gm_wm(ct, gm, wm):
    if not gm.any() or not wm.any():
        return None
    diff = float(ct[gm].mean() - ct[wm].mean())
    noise = float(ct[wm].std())
    return round(diff / max(noise, 1e-6), 2)


def recover(ct, spacing):
    r = neuro.atrophy_indices(ct, spacing)
    lo, hi = pl.seg_uncertainty_ml(r["masks"]["ventricles"], spacing)
    r["metrics"]["ventricles_IC_mL"] = [round(lo, 1), round(hi, 1)]
    return r


def interpret(metrics):
    """Leitura de pesquisa, não diagnóstico de Alzheimer."""
    evans = metrics.get("evans_like_index")
    vent = metrics.get("vent_icv_pct")
    flags = []
    if evans is not None and evans >= 0.30:
        flags.append(f"Evans-like {evans:.2f} ≥ 0,30 (alargamento ventricular possível)")
    if vent is not None and vent >= 2.5:
        flags.append(f"fração ventricular {vent:.2f}% elevada neste protótipo")
    if flags:
        nivel = "indício de atrofia / hidrocefalia ex vacuo no critério do protótipo"
    else:
        nivel = "sem alargamento grosseiro no critério deste protótipo"
    return {
        "nivel": nivel,
        "sinais": flags,
        "aviso": ("Não é diagnóstico de Alzheimer. Amiloide/tau são PET ou líquor. "
                  "Isto é morfometria de TC, para pesquisa."),
    }


def calcium_ml(ct, icv, spacing):
    cal = icv & (ct > 130) & (ct < 1500)
    cal = ndi.binary_opening(cal, iterations=1)
    return round(float(cal.sum() * np.prod(spacing) / 1000.0), 2)


def analyze_exam(ct, spacing):
    r = recover(ct, spacing)
    r["metrics"]["calcium_intracraniano_mL"] = calcium_ml(
        ct, r["masks"]["icv"], spacing)
    r["triagem"] = interpret(r["metrics"])
    r["qa"] = pl.qa_check(ct, spacing)
    return r


def _err_pct(est, true):
    if true in (0, None) or est is None:
        return None
    return round(100.0 * (est - true) / true, 1)


def compare_scanners(phenotype="atrofia_alzheimer", n_real=6, seed=0):
    """Compara EID vs PCCT na recuperação dos biomarcadores de atrofia."""
    ph = make_brain_phantom(phenotype)
    truth = ph["truth"]
    spacing = ph["spacing"]
    labels = ph["labels"]
    rows = []
    for scanner in ("eid", "pcct"):
        evans, vents, cnrs = [], [], []
        last_ct, last_masks = None, None
        for i in range(n_real):
            rng = np.random.default_rng(seed + 17 * i + (0 if scanner == "eid" else 90))
            ct = acquire(labels, scanner, rng)
            rec = recover(ct, spacing)
            m = rec["metrics"]
            evans.append(m["evans_like_index"])
            vents.append(m["ventricles_mL"])
            cnrs.append(cnr_gm_wm(ct, ph["masks"]["gm"], ph["masks"]["wm"]))
            last_ct, last_masks = ct, rec["masks"]
        ev = [e for e in evans if e is not None]
        rows.append({
            "scanner": scanner,
            "noise_HU": NOISE[scanner],
            "cnr_gm_wm": round(float(np.mean(cnrs)), 2),
            "ventricles_mL": round(float(np.mean(vents)), 1),
            "erro_ventriculos_pct": _err_pct(float(np.mean(vents)), truth["ventricles_mL"]),
            "evans_like": round(float(np.mean(ev)), 3) if ev else None,
            "erro_evans_pct": _err_pct(float(np.mean(ev)), truth["evans_like_index"]) if ev else None,
            "ct": last_ct,
            "masks": last_masks,
        })
    return {"phenotype": phenotype, "truth": truth, "phantom": ph, "scanners": rows}
