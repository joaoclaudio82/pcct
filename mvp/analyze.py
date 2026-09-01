"""AI-Photon MVP: análise de uma imagem de TC fornecida pelo usuário.

Uso:
    python analyze.py CAMINHO [--modulo auto|neuro|torax|abdome] [--saida DIR]

Aceita .nii, .nii.gz, .nrrd, .mha/.mhd ou uma pasta DICOM. Executa o módulo
escolhido (ou detecta automaticamente), grava figuras e métricas em JSON no
diretório de saída e imprime o resumo. Protótipo de pesquisa, sem validação
clínica: não usar para decisão diagnóstica ou terapêutica.
"""

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_photon_mvp import neuro
from ai_photon_mvp import pipeline as pl


def load_any(path, target_mm=1.4):
    """Carrega NIfTI/NRRD/MHA ou pasta DICOM; retorna (array zyx, spacing zyx).

    O volume é reamostrado por binning para perto de 1,4 mm, a resolução em
    que os parâmetros morfológicos do protótipo foram ajustados.
    """
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(path)
        if not ids:
            raise ValueError(f"nenhuma série DICOM encontrada em {path}")
        files = reader.GetGDCMSeriesFileNames(path, ids[0])
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(path)
    factors = [max(1, int(round(target_mm / s))) for s in img.GetSpacing()]
    if any(f > 1 for f in factors):
        img = sitk.BinShrink(img, factors)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = tuple(float(s) for s in img.GetSpacing()[::-1])
    return arr, spacing


def detect_module(ct, spacing):
    """Heurística simples de detecção do módulo pelo conteúdo do volume."""
    from scipy import ndimage as _ndi
    body = np.stack([_ndi.binary_fill_holes(b) for b in (ct > -400)])
    core = _ndi.binary_erosion(body, iterations=3)
    lunglike = (ct > -950) & (ct < -600) & core
    per_slice = lunglike.sum(axis=(1, 2)) / np.maximum(core.sum(axis=(1, 2)), 1)
    frac_slices_lung = float(np.mean(per_slice > 0.08))
    frac_bone = float(np.mean(ct > 200))
    # tórax: fatias dominadas por pulmão no interior do corpo; o abdome tem
    # apenas as bases pulmonares e gás intestinal esparso
    if frac_slices_lung > 0.35:
        return "torax"
    # crânio: muito osso em anel e pouco ar interno
    if frac_bone > 0.04 and ct.shape[0] * spacing[0] < 350:
        return "neuro"
    return "abdome"


def _fig(path, panels):
    fig, ax = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.6))
    ax = np.atleast_1d(ax)
    for a, (img, title, overlays, clim) in zip(ax, panels):
        a.imshow(img, cmap="gray", vmin=clim[0], vmax=clim[1])
        for mask, color in overlays:
            if mask is not None and mask.any():
                a.contour(mask, colors=color, linewidths=0.9)
        a.set_title(title, fontsize=10)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def analyze_neuro(ct, spacing, outdir, name):
    r = neuro.atrophy_indices(ct, spacing)
    m = r["metrics"]
    ms = r["masks"]
    lo_v, hi_v = pl.seg_uncertainty_ml(ms["ventricles"], spacing)
    m["ventricles_IC_mL"] = [round(lo_v, 1), round(hi_v, 1)]
    zc = int(np.argmax(ms["ventricles"].sum(axis=(1, 2)))) if ms["ventricles"].any() \
        else ct.shape[0] // 2
    _fig(os.path.join(outdir, f"{name}_neuro.png"), [
        (ct[zc], "janela cerebral", [], (0, 80)),
        (ct[zc], f"ICV {m['icv_mL']:.0f} mL · ventrículos {m['ventricles_mL']} mL",
         [(ms["icv"][zc], "lime"), (ms["ventricles"][zc], "deepskyblue")], (0, 80)),
    ])
    return m


def analyze_chest(ct, spacing, outdir, name):
    lungs, stats = pl.segment_lungs(ct, spacing)
    cands = pl.nodule_candidates(ct, lungs, spacing, hu_min=-150)
    lo, hi = pl.seg_uncertainty_ml(lungs, spacing)
    m = {**{k: round(v, 2) for k, v in stats.items()},
         "volume_IC_L": [round(lo / 1000, 2), round(hi / 1000, 2)],
         "candidatos_nodulo": cands}
    zc = int(np.argmax(lungs.sum(axis=(1, 2)))) if lungs.any() else ct.shape[0] // 2
    _fig(os.path.join(outdir, f"{name}_torax.png"), [
        (ct[zc], "TC de tórax", [], (-1000, 300)),
        (ct[zc], f"pulmões {stats.get('volume_L', 0):.2f} L",
         [(lungs[zc], "cyan")], (-1000, 300)),
    ])
    return m


def analyze_abdomen(ct, spacing, outdir, name):
    liver = pl.segment_liver_rough(ct, spacing)
    voxel_ml = np.prod(spacing) / 1000.0
    lesions = pl.hypodense_lesions(ct, liver, spacing)
    les = []
    for c in lesions:
        l_lo, l_hi = pl.seg_uncertainty_ml(c["mask"], spacing)
        les.append({k: v for k, v in c.items() if k != "mask"}
                   | {"volume_IC_mL": [round(l_lo, 2), round(l_hi, 2)]})
    m = {"figado_mL": round(float(liver.sum() * voxel_ml), 0),
         "lesoes_hipodensas": les,
         "carga_tumoral_mL": round(sum(c["volume_mL"] for c in lesions), 2)}
    z0 = int(round(lesions[0]["centroid"][0])) if lesions else \
        (int(np.argmax(liver.sum(axis=(1, 2)))) if liver.any() else ct.shape[0] // 2)
    over = [(liver[z0], "lime")] + [(c["mask"][z0], "red") for c in lesions
                                    if c["mask"][z0].any()]
    _fig(os.path.join(outdir, f"{name}_abdome.png"), [
        (ct[z0], "TC de abdome", [], (-150, 250)),
        (ct[z0], f"fígado {m['figado_mL']:.0f} mL · carga {m['carga_tumoral_mL']} mL",
         over, (-150, 250)),
    ])
    return m


MODULES = {"neuro": analyze_neuro, "torax": analyze_chest, "abdome": analyze_abdomen}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("caminho", help="arquivo .nii/.nii.gz/.nrrd/.mha ou pasta DICOM")
    ap.add_argument("--modulo", default="auto", choices=["auto"] + list(MODULES))
    ap.add_argument("--saida", default="results")
    args = ap.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    ct, spacing = load_any(args.caminho)
    qa = pl.qa_check(ct, spacing)
    mod = args.modulo if args.modulo != "auto" else detect_module(ct, spacing)
    name = os.path.splitext(os.path.basename(args.caminho.rstrip("/")))[0].replace(".nii", "")

    metrics = MODULES[mod](ct, spacing, args.saida, name)
    out = {"arquivo": args.caminho, "modulo": mod,
           "qa": {k: (list(v) if isinstance(v, tuple) else v) for k, v in qa.items()},
           "metricas": metrics,
           "aviso": "protótipo de pesquisa, sem validação clínica"}
    jpath = os.path.join(args.saida, f"{name}_{mod}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print(f"\nresultados: {jpath} e figuras em {args.saida}/")


if __name__ == "__main__":
    main()
