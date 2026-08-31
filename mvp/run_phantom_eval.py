"""Avaliação 1 e 2: exatidão da quantificação de iodo e varredura de dose."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

from ai_photon_mvp import spectral as sp

OUT = "results"
np.random.seed(0)

ph = sp.make_phantom()
sp_mm = ph["spacing_mm"]
rng = np.random.default_rng(42)

# ---------------------------------------------- exatidão em dose plena
vmi50 = sp.simulate_vmi(ph["water"], ph["iodine"], 50, 1.0, rng)
vmi70 = sp.simulate_vmi(ph["water"], ph["iodine"], 70, 1.0, rng)
_, iod_est = sp.decompose(vmi50, vmi70)

acc_rows = []
for k, (name, _c, r_mm, conc) in enumerate(ph["inserts"], start=1):
    m = ph["labels"] == k
    core = m.copy()
    est_mean = float(iod_est[m].mean())
    q_true = sp.iodine_load_mg(ph["iodine"], m, sp_mm)
    q_est = sp.iodine_load_mg(iod_est, m, sp_mm)
    acc_rows.append({
        "lesao": name, "diametro_mm": 2 * r_mm, "conc_verdadeira": conc,
        "conc_estimada": round(est_mean, 2),
        "erro_pct": round(100 * (est_mean - conc) / conc, 1),
        "Q_verdadeiro_mg": round(q_true, 1), "Q_estimado_mg": round(q_est, 1),
    })

aorta = ph["iodine"] == 8.0
acc_rows.append({
    "lesao": "aorta", "diametro_mm": 24, "conc_verdadeira": 8.0,
    "conc_estimada": round(float(iod_est[aorta].mean()), 2),
    "erro_pct": round(100 * (float(iod_est[aorta].mean()) - 8.0) / 8.0, 1),
    "Q_verdadeiro_mg": round(sp.iodine_load_mg(ph["iodine"], aorta, sp_mm), 1),
    "Q_estimado_mg": round(sp.iodine_load_mg(iod_est, aorta, sp_mm), 1),
})

# ---------------------------------------------- figura: VMI e mapas
zc = ph["labels"].shape[0] // 2
zs = [zc, zc + 12, zc - 12, zc + 5]
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
ax[0].imshow(vmi50[zc], cmap="gray", vmin=-200, vmax=400)
ax[0].set_title("VMI 50 keV (simulada)")
ax[1].imshow(vmi70[zc], cmap="gray", vmin=-200, vmax=400)
ax[1].set_title("VMI 70 keV (simulada)")
ax[2].imshow(ph["iodine"][zc], cmap="inferno", vmin=0, vmax=8)
ax[2].set_title("Mapa de iodo verdadeiro (mg/mL)")
im = ax[3].imshow(iod_est[zc], cmap="inferno", vmin=0, vmax=8)
ax[3].set_title("Mapa de iodo estimado (mg/mL)")
for a in ax:
    a.axis("off")
fig.colorbar(im, ax=ax[3], fraction=0.046)
fig.tight_layout()
fig.savefig(f"{OUT}/phantom_maps.png", dpi=110)
plt.close(fig)

# ---------------------------------------------- varredura de dose
doses = [1.0, 0.5, 0.25, 0.125]
n_real = 12
sweep = {"dose": doses, "erro_Q_pct": [], "erro_Q_pct_denoised": [],
         "ci_Q": [], "ruido_hu50": []}
big = ph["labels"] == 4          # lesão de 24 mm
q_true_big = sp.iodine_load_mg(ph["iodine"], big, sp_mm)

for d in doses:
    errs, errs_dn, qs = [], [], []
    for r in range(n_real):
        rr = np.random.default_rng(100 * r + int(1000 * d))
        v50 = sp.simulate_vmi(ph["water"], ph["iodine"], 50, d, rr)
        v70 = sp.simulate_vmi(ph["water"], ph["iodine"], 70, d, rr)
        _, ie = sp.decompose(v50, v70)
        q = sp.iodine_load_mg(ie, big, sp_mm)
        qs.append(q)
        errs.append(100 * abs(q - q_true_big) / q_true_big)
        # denoising com preservação quantitativa (TV leve por corte)
        v50d = denoise_tv_chambolle(v50[zc], weight=8.0)
        v70d = denoise_tv_chambolle(v70[zc], weight=6.0)
        _, ied = sp.decompose(v50d, v70d)
        m2 = big[zc]
        if m2.any():
            c_true = float(ph["iodine"][zc][m2].mean())
            c_est = float(ied[m2].mean())
            errs_dn.append(100 * abs(c_est - c_true) / c_true)
    sweep["erro_Q_pct"].append(round(float(np.mean(errs)), 2))
    sweep["erro_Q_pct_denoised"].append(round(float(np.mean(errs_dn)), 2))
    lo, hi = np.percentile(qs, [2.5, 97.5])
    sweep["ci_Q"].append([round(lo, 1), round(hi, 1)])
    sweep["ruido_hu50"].append(round(sp.NOISE_HU[50] / np.sqrt(d), 1))

fig, ax = plt.subplots(figsize=(7, 4.4))
x = [100 * d for d in doses]
ax.plot(x, sweep["erro_Q_pct"], "o-", color="#1f4e79", label="sem denoising")
ax.plot(x, sweep["erro_Q_pct_denoised"], "s--", color="#c0504d",
        label="com denoising TV")
ax.set_xlabel("Dose relativa (%)")
ax.set_ylabel("Erro médio no biomarcador de iodo (%)")
ax.set_title("Preservação quantitativa vs. redução de dose (lesão de 24 mm)")
ax.invert_xaxis()
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/dose_sweep.png", dpi=110)
plt.close(fig)

with open(f"{OUT}/phantom_metrics.json", "w") as f:
    json.dump({"acuracia": acc_rows, "dose_sweep": sweep,
               "q_true_lesao24mm_mg": round(q_true_big, 1)}, f, indent=2)

print(json.dumps(acc_rows, indent=2))
print("dose sweep:", sweep["erro_Q_pct"], sweep["erro_Q_pct_denoised"])
