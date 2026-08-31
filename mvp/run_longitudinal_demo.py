"""Avaliação 3: acompanhamento longitudinal adaptativo no phantom espectral.

Simula exame basal e reavaliação após tratamento (lesões menores e menos
captantes), executa o pipeline completo (decomposição, registro, segmentação
no mapa de iodo, pareamento de lesões) e estima a resposta com intervalo de
confiança por realizações de ruído.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from ai_photon_mvp import spectral as sp
from ai_photon_mvp import pipeline as pl

OUT = "results"
sp_mm = 1.5

# fator de resposta por lesão: raio e concentração após tratamento
RESPONSE = {"L1": (0.0, 0.0),      # resposta completa: lesão some
            "L2": (0.75, 0.6),     # resposta parcial
            "L3": (0.85, 0.7),     # resposta parcial
            "L4": (1.1, 1.15)}     # progressão (cresce e capta mais)


def build(phase):
    ph = sp.make_phantom()
    if phase == "followup":
        iod = ph["iodine"].copy()
        lab = np.zeros_like(ph["labels"])
        z, y, x = np.indices(iod.shape).astype(np.float32)
        for k, (name, (lz, ly, lx), r_mm, conc) in enumerate(ph["inserts"], 1):
            iod[ph["labels"] == k] = 0.8          # volta a parênquima
            fr, fc = RESPONSE[name]
            if fr > 0:
                r_vox = (r_mm * fr) / sp_mm
                m = ((z - lz) ** 2 + (y - ly) ** 2 + (x - lx) ** 2) <= r_vox ** 2
                iod[m] = conc * fc
                lab[m] = k
        ph["iodine"], ph["labels"] = iod, lab
    return ph


def measure(ph, rng, shift=(0, 0, 0)):
    v50 = sp.simulate_vmi(ph["water"], ph["iodine"], 50, 1.0, rng)
    v70 = sp.simulate_vmi(ph["water"], ph["iodine"], 70, 1.0, rng)
    if any(shift):
        v50 = ndi.shift(v50, shift, order=1, mode="nearest")
        v70 = ndi.shift(v70, shift, order=1, mode="nearest")
    return v50, v70


base_ph = build("baseline")
fu_ph = build("followup")
liver = base_ph["liver_mask"]

rng = np.random.default_rng(7)
b50, b70 = measure(base_ph, rng)
true_shift = (2.0, 4.0, -3.0)
f50, f70 = measure(fu_ph, rng, shift=true_shift)

# registro rígido por correlação de fase
est_shift, _, _ = phase_cross_correlation(b50, f50, upsample_factor=4)
f50r = ndi.shift(f50, est_shift, order=1, mode="nearest")
f70r = ndi.shift(f70, est_shift, order=1, mode="nearest")

_, iod_b = sp.decompose(b50, b70)
_, iod_f = sp.decompose(f50r, f70r)

lab_b, nb = pl.segment_iodine_lesions(iod_b, liver, thr_mgml=1.05)
lab_f, nf = pl.segment_iodine_lesions(iod_f, liver, thr_mgml=1.05)
pairs, new, gone = pl.match_lesions(lab_b, lab_f)

voxel_ml = (sp_mm / 10.0) ** 3


def lesion_row(lb, iod, i):
    m = lb == i
    return {"vol_mL": round(float(m.sum() * voxel_ml), 2),
            "Q_mg": round(sp.iodine_load_mg(iod, m, sp_mm), 2)}


rows = []
for i, j, d in pairs:
    a, b = lesion_row(lab_b, iod_b, i), lesion_row(lab_f, iod_f, j)
    rows.append({"lesao_basal": i, "lesao_reav": j,
                 "vol_basal_mL": a["vol_mL"], "vol_reav_mL": b["vol_mL"],
                 "Q_basal_mg": a["Q_mg"], "Q_reav_mg": b["Q_mg"],
                 "delta_Q_pct": round(100 * (b["Q_mg"] - a["Q_mg"]) / a["Q_mg"], 1)})
for i in gone:
    a = lesion_row(lab_b, iod_b, i)
    rows.append({"lesao_basal": i, "lesao_reav": None,
                 "vol_basal_mL": a["vol_mL"], "vol_reav_mL": 0.0,
                 "Q_basal_mg": a["Q_mg"], "Q_reav_mg": 0.0,
                 "delta_Q_pct": -100.0})

tb_b = round(float((lab_b > 0).sum() * voxel_ml), 2)
tb_f = round(float((lab_f > 0).sum() * voxel_ml), 2)
q_b = round(sp.iodine_load_mg(iod_b, lab_b > 0, sp_mm), 1)
q_f = round(sp.iodine_load_mg(iod_f, lab_f > 0, sp_mm), 1)

# incerteza: repete a medição com novas realizações de ruído
deltas = []
for r in range(15):
    rr = np.random.default_rng(500 + r)
    bb50, bb70 = measure(base_ph, rr)
    ff50, ff70 = measure(fu_ph, rr, shift=true_shift)
    ff50 = ndi.shift(ff50, est_shift, order=1, mode="nearest")
    ff70 = ndi.shift(ff70, est_shift, order=1, mode="nearest")
    _, ib = sp.decompose(bb50, bb70)
    _, if_ = sp.decompose(ff50, ff70)
    qb = sp.iodine_load_mg(ib, lab_b > 0, sp_mm)
    qf = sp.iodine_load_mg(if_, lab_f > 0, sp_mm)
    deltas.append(100 * (qf - qb) / qb)
ci = [round(v, 1) for v in np.percentile(deltas, [2.5, 97.5])]
delta_q_pct = round(100 * (q_f - q_b) / q_b, 1)

zc = iod_b.shape[0] // 2
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
ax[0].imshow(iod_b[zc], cmap="inferno", vmin=0, vmax=6)
ax[0].contour(lab_b[zc] > 0, colors="cyan", linewidths=0.8)
ax[0].set_title("Basal: mapa de iodo + lesões")
ax[1].imshow(iod_f[zc], cmap="inferno", vmin=0, vmax=6)
ax[1].contour(lab_f[zc] > 0, colors="cyan", linewidths=0.8)
ax[1].set_title("Reavaliação (registrada)")
diff = iod_f - iod_b
im = ax[2].imshow(diff[zc], cmap="coolwarm", vmin=-4, vmax=4)
ax[2].set_title("Mapa de mudança de iodo")
for a in ax:
    a.axis("off")
fig.colorbar(im, ax=ax[2], fraction=0.046)
fig.tight_layout()
fig.savefig(f"{OUT}/longitudinal.png", dpi=110)
plt.close(fig)

res = {
    "registro": {"shift_verdadeiro": list(true_shift),
                 "shift_estimado": [round(float(s), 2) for s in est_shift],
                 "erro_vox": round(float(np.linalg.norm(np.array(true_shift) + np.array(est_shift))), 2)},
    "lesoes_basal": int(nb), "lesoes_reav": int(nf),
    "novas": len(new), "desaparecidas": len(gone),
    "tabela_lesoes": rows,
    "carga": {"TB_basal_mL": tb_b, "TB_reav_mL": tb_f,
              "Q_basal_mg": q_b, "Q_reav_mg": q_f,
              "delta_Q_pct": delta_q_pct, "IC95_delta_Q_pct": ci},
    "verdade": {"L1": "resposta completa", "L2": "resposta parcial",
                "L3": "resposta parcial", "L4": "progressão"},
}
with open(f"{OUT}/longitudinal_metrics.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
