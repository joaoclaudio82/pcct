"""AI-Photon MVP: simulação espectral e decomposição de materiais.

Modela o comportamento espectral de um PCCT em nível de imagem reconstruída:
duas imagens monoenergéticas virtuais (VMI) são simuladas a partir de um
phantom digital com concentrações de iodo conhecidas, e a decomposição de
materiais recupera o mapa de iodo. O ruído é modelado como gaussiano
pós-reconstrução com desvio padrão escalado por 1/sqrt(fator de dose).
"""

import numpy as np
from scipy import ndimage as _ndi

from .validation import paired_arrays, spacing_zyx


def ndimage_dilate(mask, iterations):
    return _ndi.binary_dilation(mask, iterations=iterations)


# Coeficientes de atenuação mássica (cm^2/g), valores aproximados NIST
MU_RHO = {
    50: {"water": 0.2269, "iodine": 12.32},
    70: {"water": 0.1929, "iodine": 5.16},
}
RHO_WATER = 1.0  # g/cm^3

# Ruído base (desvio padrão em HU) por energia, em dose plena
NOISE_HU = {50: 14.0, 70: 9.0}


def mu_voxel(energy, water_frac, iodine_mgml):
    """Coeficiente de atenuação linear (1/cm) de um voxel."""
    c = MU_RHO[energy]
    return c["water"] * RHO_WATER * water_frac + c["iodine"] * (iodine_mgml / 1000.0)


def to_hu(mu, energy):
    mu_w = MU_RHO[energy]["water"] * RHO_WATER
    return 1000.0 * (mu - mu_w) / mu_w


def from_hu(hu, energy):
    mu_w = MU_RHO[energy]["water"] * RHO_WATER
    return mu_w * (1.0 + hu / 1000.0)


def simulate_vmi(water_frac, iodine_mgml, energy, dose_factor=1.0, rng=None):
    """Simula uma VMI em HU com ruído dependente da dose."""
    if energy not in MU_RHO:
        raise ValueError("supported energies are 50 and 70 keV")
    if not np.isfinite(dose_factor) or dose_factor <= 0:
        raise ValueError("dose_factor must be finite and positive")
    water_frac, iodine_mgml = paired_arrays(water_frac, iodine_mgml)
    rng = rng or np.random.default_rng(0)
    mu = mu_voxel(energy, water_frac, iodine_mgml)
    hu = to_hu(mu, energy)
    sigma = NOISE_HU[energy] / np.sqrt(dose_factor)
    return hu + rng.normal(0.0, sigma, hu.shape)


def decompose(vmi50, vmi70):
    """Decomposição de dois materiais (água + iodo) a partir de duas VMI.

    Resolve, voxel a voxel, o sistema 2x2:
        mu(E) = mu_rho_w(E) * rho_w + mu_rho_I(E) * c_I
    Retorna (fração de água, mapa de iodo em mg/mL).
    """
    vmi50, vmi70 = paired_arrays(vmi50, vmi70)
    mu50 = from_hu(vmi50, 50)
    mu70 = from_hu(vmi70, 70)
    a = np.array([
        [MU_RHO[50]["water"] * RHO_WATER, MU_RHO[50]["iodine"] / 1000.0],
        [MU_RHO[70]["water"] * RHO_WATER, MU_RHO[70]["iodine"] / 1000.0],
    ])
    inv = np.linalg.inv(a)
    water = inv[0, 0] * mu50 + inv[0, 1] * mu70
    iodine = inv[1, 0] * mu50 + inv[1, 1] * mu70
    return water, iodine


def make_phantom(shape=(90, 220, 220), spacing_mm=1.5):
    """Phantom digital abdominal com aorta, parênquima e lesões de iodo.

    Retorna dicionário com mapas verdadeiros e a lista de insertos
    (nome, centro, raio_mm, concentração verdadeira em mg/mL).
    """
    z, y, x = np.indices(shape).astype(np.float32)
    cz, cy, cx = [s / 2.0 for s in shape]

    water = np.zeros(shape, dtype=np.float32)
    iodine = np.zeros(shape, dtype=np.float32)

    # corpo: elipse de água
    body = (((y - cy) / 95.0) ** 2 + ((x - cx) / 105.0) ** 2) <= 1.0
    water[body] = 1.0

    # "fígado": região com leve realce de parênquima
    liver = (((z - cz) / 38.0) ** 2 + ((y - cy + 12) / 55.0) ** 2
             + ((x - cx - 30) / 62.0) ** 2) <= 1.0
    water[liver] = 1.03
    iodine[liver] = 0.8

    # aorta: cilindro com alta concentração (excluída da máscara hepática,
    # como um módulo de exclusão de vasos faria em um pipeline real)
    aorta = (((y - cy + 40) ** 2 + (x - cx) ** 2) <= 8.0 ** 2) & (np.abs(z - cz) < 40)
    iodine[aorta] = 8.0
    liver = liver & ~ndimage_dilate(aorta, 3)

    inserts = [
        ("L1", (cz, cy - 10, cx + 55), 4.0, 1.8),
        ("L2", (cz + 12, cy + 5, cx + 25), 6.0, 2.5),
        ("L3", (cz - 12, cy - 25, cx + 30), 8.0, 3.5),
        ("L4", (cz + 5, cy + 20, cx + 55), 12.0, 5.0),
    ]
    labels = np.zeros(shape, dtype=np.int16)
    for k, (_name, (lz, ly, lx), r_mm, conc) in enumerate(inserts, start=1):
        r_vox = r_mm / spacing_mm
        sph = ((z - lz) ** 2 + (y - ly) ** 2 + (x - lx) ** 2) <= r_vox ** 2
        iodine[sph] = conc
        labels[sph] = k

    return {
        "water": water,
        "iodine": iodine,
        "labels": labels,
        "inserts": inserts,
        "spacing_mm": spacing_mm,
        "liver_mask": liver,
    }


def iodine_load_mg(iodine_mgml, mask, spacing_mm):
    """Carga de iodo Q em mg dentro de uma máscara."""
    iodine_mgml, mask = paired_arrays(iodine_mgml, mask)
    voxel_ml = np.prod(spacing_zyx(spacing_mm)) / 1000.0
    return float(np.sum(iodine_mgml[mask.astype(bool)], dtype=np.float64) * voxel_ml)
