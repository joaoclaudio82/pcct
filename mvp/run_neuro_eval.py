"""Avaliação 6: módulo cerebral (Demonstrador 4) em TC de crânio pública.

Imagem: CT_Philips.nii.gz do repositório niivue-images (Rorden Lab,
licença BSD-2). Morfometria aproximada por regras clássicas: volume
intracraniano, parênquima, líquor, ventrículos, fração de líquor e índice
tipo Evans, com incerteza por perturbação de segmentação.
Protótipo, sem validação clínica.
"""

import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_photon_mvp import neuro
from ai_photon_mvp import pipeline as pl

OUT = "results"

img = nib.load("data/CT_head.nii.gz")
ct = np.asarray(img.get_fdata(), dtype=np.float32)
zooms = img.header.get_zooms()[:3]
# reordena para (z, y, x): eixo mais curto costuma ser axial? aqui usamos
# a orientação do arquivo (x, y, z) -> transpõe para (z, y, x)
ct = np.transpose(ct, (2, 1, 0))
spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

qa = pl.qa_check(ct, spacing)
res_n = neuro.atrophy_indices(ct, spacing)
m = res_n["metrics"]
masks = res_n["masks"]

# incerteza por perturbação de segmentação
lo_v, hi_v = pl.seg_uncertainty_ml(masks["ventricles"], spacing)
lo_i, hi_i = pl.seg_uncertainty_ml(masks["icv"], spacing)

out = {
    "fonte": "CT_Philips.nii.gz (niivue-images, Rorden Lab, BSD-2)",
    "qa": {k: v for k, v in qa.items() if k != "shape"} | {"shape": list(qa["shape"])},
    "morfometria": m,
    "incerteza": {
        "ventriculos_IC_mL": [round(lo_v, 1), round(hi_v, 1)],
        "icv_IC_mL": [round(lo_i, 0), round(hi_i, 0)],
    },
}

zc = int(np.argmax(masks["ventricles"].sum(axis=(1, 2))))
z2 = min(ct.shape[0] - 1, zc + 18)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))
ax[0].imshow(ct[zc], cmap="gray", vmin=0, vmax=80)
ax[0].set_title("TC de crânio pública (janela cerebral)")
ax[1].imshow(ct[zc], cmap="gray", vmin=0, vmax=80)
ax[1].contour(masks["icv"][zc], colors="lime", linewidths=0.8)
ax[1].contour(masks["ventricles"][zc], colors="deepskyblue", linewidths=1.1)
ax[1].set_title(f"ICV {m['icv_mL']:.0f} mL · ventrículos {m['ventricles_mL']} mL")
ax[2].imshow(ct[z2], cmap="gray", vmin=0, vmax=80)
ax[2].contour(masks["icv"][z2], colors="lime", linewidths=0.8)
ax[2].contour(masks["csf"][z2], colors="cyan", linewidths=0.5)
ax[2].set_title(f"Líquor {m['csf_fraction_pct']}% da ICV")
for a in ax:
    a.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/neuro_eval.png", dpi=110)
plt.close(fig)

with open(f"{OUT}/neuro_metrics.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
