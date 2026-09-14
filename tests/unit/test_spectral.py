import numpy as np

from ai_photon_mvp import spectral as sp


def test_hu_roundtrip():
    hu = np.array([-1000.0, -100.0, 0.0, 100.0, 500.0])
    for energy in (50, 70):
        recovered = sp.to_hu(sp.from_hu(hu, energy), energy)
        assert np.allclose(recovered, hu, atol=1e-10)


def test_material_decomposition_without_noise_recovers_iodine():
    water = np.ones((8, 8, 8), dtype=np.float32)
    iodine = np.full_like(water, 3.5)

    vmi50 = sp.to_hu(sp.mu_voxel(50, water, iodine), 50)
    vmi70 = sp.to_hu(sp.mu_voxel(70, water, iodine), 70)

    water_est, iodine_est = sp.decompose(vmi50, vmi70)

    assert np.allclose(water_est, water, atol=1e-5)
    assert np.allclose(iodine_est, iodine, atol=1e-4)


def test_iodine_load_uses_physical_voxel_volume():
    iodine = np.full((10, 10, 10), 2.0, dtype=np.float32)
    mask = np.ones_like(iodine, dtype=bool)

    # 1 mm isotropic -> 0.001 mL per voxel; 1000 voxels -> 1 mL.
    q = sp.iodine_load_mg(iodine, mask, spacing_mm=1.0)

    assert q == 2.0
