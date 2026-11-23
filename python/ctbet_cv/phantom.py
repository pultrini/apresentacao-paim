"""Fantoma sintético de TC de crânio com máscara de referência.

Inclui couro cabeludo, calota com espessura fixa, substância cinzenta/branca, ventrículos (hipodensos,
viram buracos na máscara candidata), uma hemorragia (70 HU), canais ópticos finos que ligam o cérebro
às órbitas (vazamento que a erosão precisa cortar), medula no canal vertebral e musculatura cervical.
"""

from __future__ import annotations

import numpy as np


def _ellipsoid(x, y, z, c, a) -> np.ndarray:
    return ((x - c[0]) / a[0]) ** 2 + ((y - c[1]) / a[1]) ** 2 + ((z - c[2]) / a[2]) ** 2 <= 1.0


def make_phantom(
    shape_zyx: tuple[int, int, int] = (60, 256, 256),
    spacing: tuple[float, float, float] = (0.9, 0.9, 3.0),
    noise_hu: float = 6.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    nz, ny, nx = shape_zyx
    sx, sy, sz = spacing
    z, y, x = np.meshgrid(np.arange(nz) * sz, np.arange(ny) * sy, np.arange(nx) * sx, indexing="ij")
    cx, cy = nx * sx / 2, ny * sy / 2
    cz = 0.58 * nz * sz
    c = (cx, cy, cz)

    brain_axes = (68.0, 84.0, 58.0)
    inner = _ellipsoid(x, y, z, c, brain_axes)
    skull = _ellipsoid(x, y, z, c, tuple(a + 7 for a in brain_axes)) & ~inner
    head = _ellipsoid(x, y, z, c, tuple(a + 12 for a in brain_axes))

    ct = np.full(shape_zyx, -1000.0, np.float32)
    ct[head] = 40.0  # couro cabeludo / partes moles
    ct[skull] = 1000.0
    ct[inner] = 38.0  # substância cinzenta
    ct[_ellipsoid(x, y, z, c, (52.0, 68.0, 44.0))] = 28.0  # substância branca
    for side in (-1, 1):
        ct[_ellipsoid(x, y, z, (cx + side * 9, cy, cz + 5), (5.0, 22.0, 14.0))] = 3.0  # ventrículos
    ct[_ellipsoid(x, y, z, (cx + 30, cy - 20, cz + 10), (12.0, 15.0, 10.0))] = 70.0  # hemorragia

    # pescoço: musculatura, vértebras e medula abaixo da base do crânio
    skull_base = cz - brain_axes[2] - 7
    neck = z < skull_base + 10
    r2 = (x - cx) ** 2 + (y - cy - 10) ** 2
    ct[neck & (r2 <= 55**2) & ~head] = 40.0
    canal = neck & (r2 <= 14**2) & ~inner
    ct[canal] = 1000.0
    ct[neck & (r2 <= 6**2) & ~inner] = 30.0  # medula

    # órbitas ligadas ao cérebro por canais finos (raio 2.5 mm) atravessando o osso frontal
    front = cy - brain_axes[1]
    for side in (-1, 1):
        ox = cx + side * 30
        oz = cz - 25
        orbit = _ellipsoid(x, y, z, (ox, front - 14, oz), (16.0, 16.0, 14.0))
        ct[orbit & ~inner] = 40.0
        canal_xz = (x - ox) ** 2 + (z - oz) ** 2 <= 2.5**2
        ct[canal_xz & (y >= front - 14) & (y <= front + 4) & ~inner] = 35.0

    rng = np.random.default_rng(seed)
    ct += rng.normal(0.0, noise_hu, size=shape_zyx).astype(np.float32)
    return np.round(ct).astype(np.int16), inner.astype(np.uint8)
