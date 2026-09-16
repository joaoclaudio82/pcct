"""Demos rápidos do phantom espectral e do acompanhamento longitudinal."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from ai_photon_mvp import alzheimer as az
from ai_photon_mvp import pipeline as pl
from ai_photon_mvp import spectral as sp

RESPONSE = {
    "L1": (0.0, 0.0),
    "L2": (0.75, 0.6),
    "L3": (0.85, 0.7),
    "L4": (1.1, 1.15),
}


def _save_maps(path, panels, cmaps, clims, titles):
    fig, ax = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4))
    ax = np.atleast_1d(ax)
    im = None
    for a, img, cmap, clim, title in zip(ax, panels, cmaps, clims, titles):
        im = a.imshow(img, cmap=cmap, vmin=clim[0], vmax=clim[1])
        a.set_title(title, fontsize=10)
        a.axis("off")
    fig.colorbar(im, ax=ax[-1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_phantom(outdir, dose_factor=1.0, seed=42):
    os.makedirs(outdir, exist_ok=True)
    ph = sp.make_phantom()
    sp_mm = ph["spacing_mm"]
    rng = np.random.default_rng(seed)
    v50 = sp.simulate_vmi(ph["water"], ph["iodine"], 50, dose_factor, rng)
    v70 = sp.simulate_vmi(ph["water"], ph["iodine"], 70, dose_factor, rng)
    _, iod = sp.decompose(v50, v70)

    rows = []
    for k, (name, _c, r_mm, conc) in enumerate(ph["inserts"], start=1):
        m = ph["labels"] == k
        est = float(iod[m].mean())
        q_true = sp.iodine_load_mg(ph["iodine"], m, sp_mm)
        q_est = sp.iodine_load_mg(iod, m, sp_mm)
        rows.append({
            "lesao": name, "diametro_mm": 2 * r_mm, "conc_verdadeira": conc,
            "conc_estimada": round(est, 2),
            "erro_pct": round(100 * (est - conc) / conc, 1),
            "Q_verdadeiro_mg": round(q_true, 1),
            "Q_estimado_mg": round(q_est, 1),
        })

    zc = ph["labels"].shape[0] // 2
    figpath = os.path.join(outdir, "phantom_maps.png")
    _save_maps(
        figpath,
        [v50[zc], v70[zc], ph["iodine"][zc], iod[zc]],
        ["gray", "gray", "inferno", "inferno"],
        [(-200, 400), (-200, 400), (0, 8), (0, 8)],
        ["VMI 50 keV", "VMI 70 keV", "iodo verdadeiro (mg/mL)", "iodo estimado (mg/mL)"],
    )
    return {"acuracia": rows, "dose_factor": dose_factor, "figura": figpath}


def _build_phase(phase):
    ph = sp.make_phantom()
    if phase != "followup":
        return ph
    iod = ph["iodine"].copy()
    lab = np.zeros_like(ph["labels"])
    z, y, x = np.indices(iod.shape).astype(np.float32)
    sp_mm = ph["spacing_mm"]
    for k, (name, (lz, ly, lx), r_mm, conc) in enumerate(ph["inserts"], 1):
        iod[ph["labels"] == k] = 0.8
        fr, fc = RESPONSE[name]
        if fr > 0:
            r_vox = (r_mm * fr) / sp_mm
            m = ((z - lz) ** 2 + (y - ly) ** 2 + (x - lx) ** 2) <= r_vox ** 2
            iod[m] = conc * fc
            lab[m] = k
    ph["iodine"], ph["labels"] = iod, lab
    return ph


def run_longitudinal(outdir, seed=7):
    os.makedirs(outdir, exist_ok=True)
    sp_mm = 1.5
    base = _build_phase("baseline")
    fu = _build_phase("followup")
    liver = base["liver_mask"]
    rng = np.random.default_rng(seed)

    def measure(ph, shift=(0, 0, 0)):
        v50 = sp.simulate_vmi(ph["water"], ph["iodine"], 50, 1.0, rng)
        v70 = sp.simulate_vmi(ph["water"], ph["iodine"], 70, 1.0, rng)
        if any(shift):
            v50 = ndi.shift(v50, shift, order=1, mode="nearest")
            v70 = ndi.shift(v70, shift, order=1, mode="nearest")
        return v50, v70

    b50, b70 = measure(base)
    true_shift = (2.0, 4.0, -3.0)
    f50, f70 = measure(fu, shift=true_shift)
    est_shift, _, _ = phase_cross_correlation(b50, f50, upsample_factor=4)
    f50r = ndi.shift(f50, est_shift, order=1, mode="nearest")
    f70r = ndi.shift(f70, est_shift, order=1, mode="nearest")
    _, iod_b = sp.decompose(b50, b70)
    _, iod_f = sp.decompose(f50r, f70r)
    lab_b, nb = pl.segment_iodine_lesions(iod_b, liver, thr_mgml=1.05)
    lab_f, nf = pl.segment_iodine_lesions(iod_f, liver, thr_mgml=1.05)
    pairs, new, gone = pl.match_lesions(lab_b, lab_f)
    voxel_ml = (sp_mm / 10.0) ** 3

    def row(lb, iod, i):
        m = lb == i
        return {"vol_mL": round(float(m.sum() * voxel_ml), 2),
                "Q_mg": round(sp.iodine_load_mg(iod, m, sp_mm), 2)}

    rows = []
    for i, j, _d in pairs:
        a, b = row(lab_b, iod_b, i), row(lab_f, iod_f, j)
        rows.append({
            "lesao_basal": i, "lesao_reav": j,
            "vol_basal_mL": a["vol_mL"], "vol_reav_mL": b["vol_mL"],
            "Q_basal_mg": a["Q_mg"], "Q_reav_mg": b["Q_mg"],
            "delta_Q_pct": round(100 * (b["Q_mg"] - a["Q_mg"]) / a["Q_mg"], 1),
        })
    for i in gone:
        a = row(lab_b, iod_b, i)
        rows.append({
            "lesao_basal": i, "lesao_reav": None,
            "vol_basal_mL": a["vol_mL"], "vol_reav_mL": 0.0,
            "Q_basal_mg": a["Q_mg"], "Q_reav_mg": 0.0, "delta_Q_pct": -100.0,
        })

    zc = iod_b.shape[0] // 2
    figpath = os.path.join(outdir, "longitudinal.png")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax[0].imshow(iod_b[zc], cmap="inferno", vmin=0, vmax=6)
    ax[0].contour(lab_b[zc] > 0, colors="cyan", linewidths=0.8)
    ax[0].set_title("Basal")
    ax[1].imshow(iod_f[zc], cmap="inferno", vmin=0, vmax=6)
    ax[1].contour(lab_f[zc] > 0, colors="cyan", linewidths=0.8)
    ax[1].set_title("Reavaliação (registrada)")
    im = ax[2].imshow((iod_f - iod_b)[zc], cmap="coolwarm", vmin=-4, vmax=4)
    ax[2].set_title("Mudança de iodo")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(figpath, dpi=110)
    plt.close(fig)

    q_b = round(sp.iodine_load_mg(iod_b, lab_b > 0, sp_mm), 1)
    q_f = round(sp.iodine_load_mg(iod_f, lab_f > 0, sp_mm), 1)
    return {
        "registro": {
            "shift_verdadeiro": list(true_shift),
            "shift_estimado": [round(float(s), 2) for s in est_shift],
            "erro_vox": round(float(np.linalg.norm(
                np.array(true_shift) + np.array(est_shift))), 2),
        },
        "lesoes_basal": int(nb), "lesoes_reav": int(nf),
        "novas": len(new), "desaparecidas": len(gone),
        "tabela_lesoes": rows,
        "carga": {
            "Q_basal_mg": q_b, "Q_reav_mg": q_f,
            "delta_Q_pct": round(100 * (q_f - q_b) / q_b, 1),
        },
        "verdade": RESPONSE,
        "figura": figpath,
    }


def run_alzheimer(outdir, phenotype="atrofia_alzheimer", n_real=5, seed=0):
    """Teste PCCT vs TC convencional na morfometria de atrofia."""
    os.makedirs(outdir, exist_ok=True)
    cmp_ = az.compare_scanners(phenotype, n_real=n_real, seed=seed)
    ph = cmp_["phantom"]
    truth = cmp_["truth"]
    spacing = ph["spacing"]
    zc = int(np.argmax(ph["masks"]["ventricles"].sum(axis=(1, 2))))
    eid = next(s for s in cmp_["scanners"] if s["scanner"] == "eid")
    pcct = next(s for s in cmp_["scanners"] if s["scanner"] == "pcct")

    figpath = os.path.join(outdir, "alzheimer_pcct.png")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax[0].imshow(az.compose(ph["labels"], az.HU_PCCT)[zc], cmap="gray", vmin=0, vmax=80)
    ax[0].contour(ph["masks"]["ventricles"][zc], colors="deepskyblue", linewidths=1.0)
    ax[0].set_title(f"Verdade · Evans {truth['evans_like_index']}")
    ax[1].imshow(eid["ct"][zc], cmap="gray", vmin=0, vmax=80)
    ax[1].contour(eid["masks"]["ventricles"][zc], colors="tomato", linewidths=1.0)
    ax[1].set_title(f"TC convencional · Evans {eid['evans_like']}")
    ax[2].imshow(pcct["ct"][zc], cmap="gray", vmin=0, vmax=80)
    ax[2].contour(pcct["masks"]["ventricles"][zc], colors="lime", linewidths=1.0)
    ax[2].set_title(f"PCCT simulado · Evans {pcct['evans_like']}")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(figpath, dpi=110)
    plt.close(fig)

    def slim(row):
        return {k: v for k, v in row.items() if k not in ("ct", "masks")}

    return {
        "o_que_e": ("Não diagnostica Alzheimer. Compara ruído/contraste de TC "
                    "convencional vs PCCT na recuperação de Evans e volume ventricular."),
        "phenotype": phenotype,
        "truth": truth,
        "eid": slim(eid),
        "pcct": slim(pcct),
        "spacing_mm": spacing[0],
        "figura": figpath,
    }
