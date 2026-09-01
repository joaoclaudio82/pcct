"""AI-Photon MVP: módulo cerebral (Demonstrador 4, neuroimagem de demências).

Morfometria aproximada em TC de crânio por regras clássicas de HU e
morfologia: volume intracraniano, parênquima, líquor, ventrículos e índices
de atrofia. Protótipo, sem validação clínica: demonstra o fluxo que, no
projeto, seria executado com segmentação por aprendizado sobre PCCT.
"""

import numpy as np
from scipy import ndimage as ndi

HU_SKULL = 160
HU_CSF = (-5, 18)
HU_PARENQ = (18, 60)


def segment_intracranial(ct, spacing):
    """Cavidade intracraniana: interior do crânio, maior componente."""
    head = ct > -400
    head = np.stack([ndi.binary_fill_holes(s) for s in head])
    skull = ct > HU_SKULL
    skull = ndi.binary_dilation(skull, iterations=1)
    inner = head & ~skull
    soft = inner & (ct > -20) & (ct < 100)
    soft = ndi.binary_opening(soft, iterations=2)
    lab, n = ndi.label(soft)
    if n == 0:
        return np.zeros_like(soft)
    sizes = ndi.sum(soft, lab, range(1, n + 1))
    icv = lab == (np.argmax(sizes) + 1)
    icv = ndi.binary_closing(icv, iterations=4)
    icv = np.stack([ndi.binary_fill_holes(s) for s in icv])
    icv = icv & ~skull
    # mantém apenas a faixa axial contígua da cavidade craniana, cortando
    # face e pescoço abaixo da base do crânio
    area = icv.sum(axis=(1, 2)).astype(float)
    zmax = int(np.argmax(area))
    nz = np.nonzero(area)[0]
    keep = np.zeros(len(area), dtype=bool)
    if len(nz):
        z0 = int(nz.min())
        # base do crânio: vale (mínimo local) do perfil de área entre o
        # início do volume e o plano de maior área; abaixo dele é face/pescoço
        lo = z0
        if zmax - z0 > 10:
            seg = area[z0:zmax]
            zmin = z0 + int(np.argmin(seg))
            if area[zmin] < 0.75 * area[z0:zmin + 1].max():
                lo = zmin
        keep[lo:int(nz.max()) + 1] = True
    icv[~keep] = False
    # propagação corte a corte a partir do plano de maior área: em cada
    # fatia, mantém apenas os componentes 2D contíguos à fatia vizinha,
    # eliminando face e órbitas nas fatias baixas
    ref = int(np.argmax(icv.sum(axis=(1, 2))))
    for z in range(ref - 1, -1, -1):
        lab2, n2 = ndi.label(icv[z])
        if n2 == 0:
            continue
        ok = np.zeros_like(icv[z])
        for c in range(1, n2 + 1):
            comp = lab2 == c
            if (comp & icv[z + 1]).sum() > 0.3 * comp.sum():
                ok |= comp
        icv[z] = ok
    for z in range(ref + 1, icv.shape[0]):
        lab2, n2 = ndi.label(icv[z])
        if n2 == 0:
            continue
        ok = np.zeros_like(icv[z])
        for c in range(1, n2 + 1):
            comp = lab2 == c
            if (comp & icv[z - 1]).sum() > 0.3 * comp.sum():
                ok |= comp
        icv[z] = ok
    return icv


def brain_compartments(ct, icv):
    """Separa líquor e parênquima dentro da cavidade intracraniana."""
    csf = icv & (ct >= HU_CSF[0]) & (ct < HU_CSF[1])
    par = icv & (ct >= HU_PARENQ[0]) & (ct < HU_PARENQ[1])
    return csf, par


def segment_ventricles(ct, icv, csf):
    """Ventrículos: componentes de líquor profundos (longe da borda da ICV)."""
    dist = ndi.distance_transform_edt(icv)
    deep = dist > 0.25 * dist.max()
    vent_seed = csf & deep
    vent_seed = ndi.binary_opening(vent_seed, iterations=1)
    lab, n = ndi.label(vent_seed)
    if n == 0:
        return vent_seed
    sizes = ndi.sum(vent_seed, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = [order[0] + 1]
    for k in order[1:3]:
        if sizes[k] > 0.2 * sizes[order[0]]:
            keep.append(k + 1)
    vent = np.isin(lab, keep)
    # recupera a extensão ventricular completa dentro do líquor
    vent = ndi.binary_propagation(vent, mask=ndi.binary_dilation(vent, iterations=6) & csf)
    return vent


def atrophy_indices(ct, spacing):
    """Calcula os índices morfométricos e retorna máscaras e métricas."""
    icv = segment_intracranial(ct, spacing)
    csf, par = brain_compartments(ct, icv)
    vent = segment_ventricles(ct, icv, csf)
    voxel_ml = np.prod(spacing) / 1000.0

    icv_ml = float(icv.sum() * voxel_ml)
    csf_ml = float(csf.sum() * voxel_ml)
    par_ml = float(par.sum() * voxel_ml)
    vent_ml = float(vent.sum() * voxel_ml)

    # índice tipo Evans aproximado: largura ventricular máxima /
    # largura intracraniana interna no corte de maior área ventricular
    evans = np.nan
    if vent.any():
        z0 = int(np.argmax(vent.sum(axis=(1, 2))))
        vy = np.argwhere(vent[z0])
        iy = np.argwhere(icv[z0])
        if len(vy) and len(iy):
            w_v = (vy[:, 1].max() - vy[:, 1].min()) * spacing[2]
            w_i = (iy[:, 1].max() - iy[:, 1].min()) * spacing[2]
            evans = float(w_v / w_i) if w_i > 0 else np.nan

    return {
        "masks": {"icv": icv, "csf": csf, "parenchyma": par, "ventricles": vent},
        "metrics": {
            "icv_mL": round(icv_ml, 0),
            "parenchyma_mL": round(par_ml, 0),
            "csf_mL": round(csf_ml, 0),
            "ventricles_mL": round(vent_ml, 1),
            "csf_fraction_pct": round(100 * csf_ml / icv_ml, 1) if icv_ml else np.nan,
            "vent_icv_pct": round(100 * vent_ml / icv_ml, 2) if icv_ml else np.nan,
            "evans_like_index": round(evans, 3) if np.isfinite(evans) else None,
            "parenchyma_mean_hu": round(float(ct[par].mean()), 1) if par.any() else None,
        },
    }
