"""Avaliações 4 e 5: pipeline anatômico em imagens públicas reais de TC.

Tórax: CTChest (3D Slicer SampleData, licença livre) e
Abdome: CTLiver (caso liver_100 do Medical Segmentation Decathlon, CC-BY-SA,
redistribuído pelo projeto 3D Slicer). Módulos de protótipo, sem validação
clínica: o objetivo é demonstrar QA, segmentação, deteção de candidatos e
biomarcadores volumétricos com incerteza de segmentação.
"""

import json
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from ai_photon_mvp import pipeline as pl

OUT = "results"
res = {}


def load(path, downsample=1):
    img = sitk.ReadImage(path)
    if downsample > 1:
        img = sitk.BinShrink(img, [downsample] * 3)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z, y, x
    spacing = img.GetSpacing()[::-1]                      # z, y, x
    return arr, spacing


# ------------------------------------------------------------- tórax
ct, spc = load("data/CTChest.nrrd")
qa = pl.qa_check(ct, spc)
lungs, lstats = pl.segment_lungs(ct, spc)
cands = pl.nodule_candidates(ct, lungs, spc, hu_min=-150)
lo, hi = pl.seg_uncertainty_ml(lungs, spc)
res["torax"] = {
    "fonte": "CTChest.nrrd (3D Slicer SampleData)",
    "qa": {k: v for k, v in qa.items() if k != "shape"} | {"shape": list(qa["shape"])},
    "pulmoes": {**{k: round(v, 2) for k, v in lstats.items()},
                "volume_IC_L": [round(lo / 1000, 2), round(hi / 1000, 2)]},
    "candidatos_nodulo": [{k: v for k, v in c.items()} for c in cands],
}

zc = int(np.argmax(lungs.sum(axis=(1, 2))))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
ax[0].imshow(ct[zc], cmap="gray", vmin=-1000, vmax=300)
ax[0].set_title("TC de tórax pública (axial)")
ax[1].imshow(ct[zc], cmap="gray", vmin=-1000, vmax=300)
ax[1].contour(lungs[zc], colors="cyan", linewidths=0.9)
ax[1].set_title(f"Pulmões segmentados: {lstats['volume_L']:.2f} L")
cor = int(ct.shape[1] * 0.52)
ax[2].imshow(ct[:, cor, :], cmap="gray", vmin=-1000, vmax=300, aspect=spc[0] / spc[2])
ax[2].contour(lungs[:, cor, :], colors="cyan", linewidths=0.9)
for c in cands[:5]:
    z, y, x = c["centroid"]
    if abs(y - cor) < 40:
        ax[2].plot(x, z, "o", mfc="none", mec="orange", ms=14, mew=1.6)
ax[2].set_title("Coronal + candidatos a nódulo")
ax[2].invert_yaxis()
for a in ax:
    a.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/chest_eval.png", dpi=110)
plt.close(fig)

# marca candidatos no corte axial de cada um
fig, axs = plt.subplots(1, min(4, max(1, len(cands))), figsize=(14, 3.8))
axs = np.atleast_1d(axs)
for a, c in zip(axs, cands):
    z, y, x = [int(round(v)) for v in c["centroid"]]
    a.imshow(ct[z], cmap="gray", vmin=-1000, vmax=300)
    a.plot(x, y, "o", mfc="none", mec="orange", ms=18, mew=1.8)
    a.set_title(f"d={c['diam_eq_mm']} mm, {c['mean_hu']} HU", fontsize=9)
    a.axis("off")
fig.suptitle("Candidatos a nódulo (protótipo, sem validação clínica)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/chest_nodules.png", dpi=110)
plt.close(fig)

# ------------------------------------------------------------- abdome
ct2, spc2 = load("data/CTLiver.nrrd", downsample=2)
qa2 = pl.qa_check(ct2, spc2)
liver = pl.segment_liver_rough(ct2, spc2)
voxel_ml = np.prod(spc2) / 1000.0
liv_ml = float(liver.sum() * voxel_ml)
lesions = pl.hypodense_lesions(ct2, liver, spc2)
les_out = []
for c in lesions:
    l_lo, l_hi = pl.seg_uncertainty_ml(c["mask"], spc2)
    les_out.append({k: v for k, v in c.items() if k != "mask"}
                   | {"volume_IC_mL": [round(l_lo, 2), round(l_hi, 2)]})
tb = round(sum(c["volume_mL"] for c in lesions), 2)
res["abdome"] = {
    "fonte": "CTLiver.nrrd (Medical Segmentation Decathlon, liver_100)",
    "qa": {k: v for k, v in qa2.items() if k != "shape"} | {"shape": list(qa2["shape"])},
    "figado_mL": round(liv_ml, 0),
    "lesoes_hipodensas": les_out,
    "carga_tumoral_mL": tb,
}

if lesions:
    z0 = int(round(lesions[0]["centroid"][0]))
else:
    z0 = int(np.argmax(liver.sum(axis=(1, 2))))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
ax[0].imshow(ct2[z0], cmap="gray", vmin=-150, vmax=250)
ax[0].set_title("TC de abdome pública (axial)")
ax[1].imshow(ct2[z0], cmap="gray", vmin=-150, vmax=250)
ax[1].contour(liver[z0], colors="lime", linewidths=0.9)
ax[1].set_title(f"Fígado aproximado: {liv_ml:.0f} mL")
ax[2].imshow(ct2[z0], cmap="gray", vmin=-150, vmax=250)
ax[2].contour(liver[z0], colors="lime", linewidths=0.7)
for c in lesions:
    m = c["mask"]
    if m[z0].any():
        ax[2].contour(m[z0], colors="red", linewidths=1.1)
ax[2].set_title(f"Lesões hipodensas: carga {tb} mL")
for a in ax:
    a.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/liver_eval.png", dpi=110)
plt.close(fig)

with open(f"{OUT}/real_ct_metrics.json", "w") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps(res, indent=2, default=str)[:3000])
