"""AI-Photon MVP: QA, segmentação clássica, biomarcadores e incerteza.

Módulos de protótipo, sem validação clínica: destinados a demonstrar o
fluxo ponta a ponta descrito na proposta (MVP 1) sobre dados públicos e
sobre o phantom espectral.
"""

import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------- QA

def qa_check(volume_hu, spacing, expected_range=(-1100, 3200)):
    """Controle de qualidade simples: retorna alertas e um QA score 0-100."""
    alerts = []
    vmin, vmax = float(volume_hu.min()), float(volume_hu.max())
    if vmin < expected_range[0] or vmax > expected_range[1]:
        alerts.append(f"faixa de HU inesperada: [{vmin:.0f}, {vmax:.0f}]")
    if max(spacing) > 5.0:
        alerts.append(f"espacamento grosso: {spacing}")
    if min(volume_hu.shape) < 32:
        alerts.append(f"volume pequeno: {volume_hu.shape}")
    frac_air = float(np.mean(volume_hu < -500))
    if frac_air > 0.9:
        alerts.append("volume quase vazio")
    nan_frac = float(np.mean(~np.isfinite(volume_hu)))
    if nan_frac > 0:
        alerts.append(f"valores invalidos: {nan_frac:.2%}")
    score = max(0, 100 - 20 * len(alerts))
    return {"score": score, "alerts": alerts,
            "hu_range": (vmin, vmax), "shape": volume_hu.shape,
            "spacing": tuple(round(s, 2) for s in spacing)}


# ------------------------------------------------- segmentação pulmonar

def segment_lungs(volume_hu, spacing):
    """Segmentação clássica de pulmões: limiar + morfologia + componentes."""
    body = volume_hu > -500
    # preenchimento 2D por corte: os pulmões comunicam com o ar externo
    # pela traqueia, então o preenchimento 3D não os fecharia
    body = np.stack([ndi.binary_fill_holes(b) for b in body])
    air = (volume_hu < -320) & body
    air = ndi.binary_opening(air, iterations=1)
    lab, n = ndi.label(air)
    if n == 0:
        return np.zeros_like(air), {}
    sizes = ndi.sum(air, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = [order[0] + 1]
    if len(order) > 1 and sizes[order[1]] > 0.15 * sizes[order[0]]:
        keep.append(order[1] + 1)
    lungs = np.isin(lab, keep)
    lungs = ndi.binary_closing(lungs, iterations=3)
    voxel_ml = np.prod(spacing) / 1000.0
    stats = {
        "volume_L": float(lungs.sum() * voxel_ml / 1000.0),
        "mean_hu": float(volume_hu[lungs].mean()) if lungs.any() else np.nan,
        "emphysema_pct": float(np.mean(volume_hu[lungs] < -950) * 100) if lungs.any() else np.nan,
    }
    return lungs, stats


def nodule_candidates(volume_hu, lungs, spacing, hu_min=-300,
                      d_min_mm=5.0, d_max_mm=30.0, max_cands=8):
    """Candidatos a nódulo: componentes densos no interior do pulmão,
    filtrados por tamanho e esfericidade. Protótipo, sem validação clínica."""
    core = ndi.binary_erosion(lungs, iterations=2)
    dense = (volume_hu > hu_min) & core
    dense = ndi.binary_opening(dense, iterations=1)
    lab, n = ndi.label(dense)
    voxel_ml = np.prod(spacing) / 1000.0
    v_min = (4 / 3) * np.pi * (d_min_mm / 2) ** 3 / 1000.0
    v_max = (4 / 3) * np.pi * (d_max_mm / 2) ** 3 / 1000.0
    cands = []
    for i in range(1, n + 1):
        m = lab == i
        vol_ml = m.sum() * voxel_ml
        if not (v_min <= vol_ml <= v_max):
            continue
        idx = np.argwhere(m)
        ext_mm = (idx.max(0) - idx.min(0) + 1) * np.array(spacing)
        elong = ext_mm.max() / max(ext_mm.min(), 1e-3)
        if elong > 2.6:      # descarta vasos alongados
            continue
        d_eq = 2 * (3 * vol_ml * 1000 / (4 * np.pi)) ** (1 / 3)
        cands.append({
            "centroid": tuple(np.round(idx.mean(0), 1)),
            "volume_mL": round(vol_ml, 3),
            "diam_eq_mm": round(d_eq, 1),
            "mean_hu": round(float(volume_hu[m].mean()), 1),
            "elongation": round(float(elong), 2),
        })
    cands.sort(key=lambda c: -c["volume_mL"])
    return cands[:max_cands]


# ------------------------------------------------- segmentação hepática

def segment_liver_rough(volume_hu, spacing):
    """Fígado aproximado: faixa de HU de tecido mole + maior componente
    no quadrante superior direito. Protótipo, sem validação clínica."""
    soft = (volume_hu > 25) & (volume_hu < 180)
    # erosão forte para desconectar o fígado de órgãos vizinhos,
    # maior componente, e reconstrução morfológica dentro da faixa de HU
    core = ndi.binary_erosion(soft, iterations=5)
    lab, n = ndi.label(core)
    if n == 0:
        return np.zeros_like(soft)
    sizes = ndi.sum(core, lab, range(1, n + 1))
    seed = lab == (np.argmax(sizes) + 1)
    # reconstrução limitada: recupera a borda erodida sem vazar para vizinhos
    grow = ndi.binary_opening(soft, iterations=2)
    liver = seed
    for _ in range(7):
        liver = ndi.binary_dilation(liver) & grow
    liver = ndi.binary_closing(liver, iterations=3)
    liver = ndi.binary_fill_holes(liver)
    # preenchimento 2D adicional: inclui lesões hipodensas internas que
    # comunicam com a borda em alguma fatia
    liver = np.stack([ndi.binary_fill_holes(s) for s in liver])
    return liver


def hypodense_lesions(volume_hu, liver, spacing, delta_hu=25,
                      d_min_mm=8.0, max_cands=10):
    """Lesões hipodensas dentro do fígado (metástases típicas em fase portal)."""
    if not liver.any():
        return []
    core = ndi.binary_erosion(liver, iterations=3)
    ref = float(volume_hu[core].mean()) if core.any() else float(volume_hu[liver].mean())
    hypo = (volume_hu < ref - delta_hu) & (volume_hu > -20) & core
    hypo = ndi.binary_opening(hypo, iterations=1)
    lab, n = ndi.label(hypo)
    voxel_ml = np.prod(spacing) / 1000.0
    v_min = (4 / 3) * np.pi * (d_min_mm / 2) ** 3 / 1000.0
    out = []
    for i in range(1, n + 1):
        m = lab == i
        vol_ml = m.sum() * voxel_ml
        if vol_ml < v_min:
            continue
        d_eq = 2 * (3 * vol_ml * 1000 / (4 * np.pi)) ** (1 / 3)
        idx = np.argwhere(m)
        out.append({
            "centroid": tuple(np.round(idx.mean(0), 1)),
            "volume_mL": round(vol_ml, 2),
            "diam_eq_mm": round(d_eq, 1),
            "mean_hu": round(float(volume_hu[m].mean()), 1),
            "ref_liver_hu": round(ref, 1),
            "mask": m,
        })
    out.sort(key=lambda c: -c["volume_mL"])
    return out[:max_cands]


# --------------------------------------------------------- incerteza

def seg_uncertainty_ml(mask, spacing, iterations=1):
    """Intervalo de volume por perturbação da segmentação (erosão/dilatação)."""
    voxel_ml = np.prod(spacing) / 1000.0
    lo = ndi.binary_erosion(mask, iterations=iterations).sum() * voxel_ml
    hi = ndi.binary_dilation(mask, iterations=iterations).sum() * voxel_ml
    return float(lo), float(hi)


# ------------------------------------------------- lesões no mapa de iodo

def segment_iodine_lesions(iodine_map, liver_mask, thr_mgml=0.9, min_vox=20,
                           smooth_sigma=1.3):
    """Segmenta lesões captantes no mapa de iodo (acima do parênquima).

    A decomposição espectral amplifica o ruído, então o mapa é suavizado
    antes do limiar; o tamanho mínimo remove componentes espúrios.
    """
    smoothed = ndi.gaussian_filter(iodine_map, smooth_sigma)
    core = ndi.binary_erosion(liver_mask, iterations=2)
    seed = (smoothed > thr_mgml) & core
    seed = ndi.binary_opening(seed, iterations=1)
    lab, n = ndi.label(seed)
    out = np.zeros_like(lab)
    k = 0
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_vox:
            continue
        k += 1
        out[m] = k
    return out, k


def match_lesions(labels_a, labels_b, max_dist_vox=12):
    """Pareamento de lesões entre duas datas pelo centróide mais próximo."""
    def cents(lb):
        return {i: np.array(ndi.center_of_mass(lb == i))
                for i in range(1, lb.max() + 1)}
    ca, cb = cents(labels_a), cents(labels_b)
    pairs = []
    used = set()
    for i, pa in ca.items():
        best, bd = None, 1e9
        for j, pb in cb.items():
            if j in used:
                continue
            d = np.linalg.norm(pa - pb)
            if d < bd:
                best, bd = j, d
        if best is not None and bd <= max_dist_vox:
            pairs.append((i, best, bd))
            used.add(best)
    new = [j for j in cb if j not in used]
    gone = [i for i in ca if i not in {p[0] for p in pairs}]
    return pairs, new, gone
