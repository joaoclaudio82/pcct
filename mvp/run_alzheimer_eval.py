"""Teste do Demonstrador 4: Alzheimer / atrofia em PCCT simulado e TC real.

1) Phantom cerebral controle vs atrofia, adquirido como TC convencional (EID)
   e como PCCT (menos ruído, melhor contraste GM/WM). Recupera Evans e
   volume ventricular contra a verdade do phantom.
2) Triagem oportunista nos crânios públicos (TC convencional, não NAEOTOM).

Protótipo de pesquisa, sem validação clínica e sem diagnóstico de Alzheimer.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_any
from ai_photon_mvp import alzheimer as az
from demos import run_alzheimer

OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------ phantom PCCT vs EID
cmp_ad = run_alzheimer(OUT, phenotype="atrofia_alzheimer", n_real=6, seed=1)
cmp_ct = az.compare_scanners("controle", n_real=6, seed=3)

# classificação grosseira: Evans ≥ 0,30 distingue atrofia vs controle?
def classif(phenotype, scanner, seed):
    ph = az.make_brain_phantom(phenotype)
    ct = az.acquire(ph["labels"], scanner, np.random.default_rng(seed))
    m = az.recover(ct, ph["spacing"])["metrics"]
    pred = "atrofia" if (m["evans_like_index"] or 0) >= 0.30 else "controle"
    return pred, m["evans_like_index"]


hits = {"eid": 0, "pcct": 0}
n = 8
for i in range(n):
    for pheno, label in (("controle", "controle"), ("atrofia_alzheimer", "atrofia")):
        for sc in ("eid", "pcct"):
            pred, _ = classif(pheno, sc, 20 + i + (0 if sc == "eid" else 50))
            if pred == label:
                hits[sc] += 1
acc = {k: round(100 * v / (2 * n), 1) for k, v in hits.items()}

# ------------------------------------------------ crânios públicos
heads = [
    ("data/CT_head.nii.gz", "CT Philips (niivue)"),
    ("data/CT_brain.nrrd", "CT-MR Brain (3D Slicer)"),
]
exames = []
for path, fonte in heads:
    if not os.path.isfile(path):
        continue
    ct, spacing = load_any(path)
    r = az.analyze_exam(ct, spacing)
    exames.append({
        "fonte": fonte,
        "arquivo": os.path.basename(path),
        "metricas": r["metrics"],
        "triagem": r["triagem"],
        "qa": {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in r["qa"].items() if k != "shape"}
        | {"shape": list(r["qa"]["shape"])},
    })
    zc = int(np.argmax(r["masks"]["ventricles"].sum(axis=(1, 2)))) \
        if r["masks"]["ventricles"].any() else ct.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.5))
    ax[0].imshow(ct[zc], cmap="gray", vmin=0, vmax=80)
    ax[0].set_title(fonte)
    ax[1].imshow(ct[zc], cmap="gray", vmin=0, vmax=80)
    ax[1].contour(r["masks"]["icv"][zc], colors="lime", linewidths=0.8)
    ax[1].contour(r["masks"]["ventricles"][zc], colors="deepskyblue", linewidths=1.1)
    m = r["metrics"]
    ax[1].set_title(f"Evans {m['evans_like_index']} · vent {m['ventricles_mL']} mL")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    slug = os.path.splitext(os.path.basename(path))[0].replace(".nii", "")
    fig.savefig(f"{OUT}/alzheimer_{slug}.png", dpi=110)
    plt.close(fig)

out = {
    "aviso": "protótipo de pesquisa; não diagnostica Alzheimer; PCCT é simulado",
    "phantom_atrofia": {
        "truth": cmp_ad["truth"],
        "eid": cmp_ad["eid"],
        "pcct": cmp_ad["pcct"],
    },
    "phantom_controle": {
        "truth": cmp_ct["truth"],
        "eid": {k: v for k, v in cmp_ct["scanners"][0].items() if k not in ("ct", "masks")},
        "pcct": {k: v for k, v in cmp_ct["scanners"][1].items() if k not in ("ct", "masks")},
    },
    "acuracia_classe_evans_030": acc,
    "exames_publicos": exames,
}
with open(f"{OUT}/alzheimer_metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
