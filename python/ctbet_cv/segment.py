"""Extração do parênquima cerebral em TC sem FSL.

Pipeline (etapas 1–2 iguais a Muschelli et al., 2015; o BET é trocado por morfologia):

1. Limiar 0–100 HU (fora da faixa → 0).
2. Suavização gaussiana 3D (σ = 1 mm) e novo limiar → máscara candidata
   (o voxel original também precisa estar em 0–100 HU).
3. Erosão 2D (desfaz pontes finas entre cérebro e tecidos extracranianos, p. ex. órbitas).
4. Maior componente conexo 3D.
5. Dilatação com o mesmo elemento estruturante, restrita à máscara candidata.
6. Preenchimento de buracos fatia a fatia (ventrículos, hipodensidades).

A implementação em C++ (cpp/) segue exatamente os mesmos passos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Params:
    lower_hu: float = 0.0  # limiar inferior do artigo
    upper_hu: float = 100.0  # limiar superior do artigo
    sigma_mm: float = 1.0  # σ da gaussiana (0 desliga a suavização)
    brain_min_hu: float = 5.0  # mínimo na imagem suavizada para entrar na candidata
    erosion_mm: float = 4.0  # raio do elemento estruturante


def kernel_size(sigma_vox: float) -> int:
    return 2 * math.ceil(3.0 * sigma_vox) + 1


def radius_vox(mm: float, spacing: float) -> int:
    return max(0, int(math.floor(mm / spacing + 0.5)))


def reflect101(i: int, n: int) -> int:
    if n == 1:
        return 0
    while i < 0 or i >= n:
        i = -i if i < 0 else 2 * n - 2 - i
    return i


def threshold_hu(vol: np.ndarray, lower: float, upper: float) -> np.ndarray:
    out = vol.astype(np.float32, copy=True)
    out[(vol < lower) | (vol > upper)] = 0.0
    return out


def gaussian_smooth(vol: np.ndarray, spacing: tuple[float, float, float], sigma_mm: float) -> np.ndarray:
    if sigma_mm <= 0:
        return vol.copy()
    sx, sy, sz = (sigma_mm / s for s in spacing)
    kx = cv2.getGaussianKernel(kernel_size(sx), sx, cv2.CV_32F)
    ky = cv2.getGaussianKernel(kernel_size(sy), sy, cv2.CV_32F)
    planar = np.empty_like(vol, dtype=np.float32)
    for z in range(vol.shape[0]):
        planar[z] = cv2.sepFilter2D(vol[z], cv2.CV_32F, kx, ky, borderType=cv2.BORDER_REFLECT_101)

    kz = cv2.getGaussianKernel(kernel_size(sz), sz, cv2.CV_32F).ravel()
    r = len(kz) // 2
    nz = vol.shape[0]
    out = np.zeros_like(planar)
    for z in range(nz):
        acc = np.zeros(vol.shape[1:], np.float32)
        for k, w in enumerate(kz):
            acc += np.float32(w) * planar[reflect101(z + k - r, nz)]
        out[z] = acc
    return out


def structuring_element(spacing: tuple[float, float, float], radius_mm: float) -> np.ndarray | None:
    rx = radius_vox(radius_mm, spacing[0])
    ry = radius_vox(radius_mm, spacing[1])
    if rx == 0 and ry == 0:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))


class _UnionFind:
    """Union-find em que a raiz é sempre o menor rótulo (resultado determinístico)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


def largest_component_3d(mask: np.ndarray) -> np.ndarray:
    """Maior componente 3D (8-vizinhança no plano, 6-vizinhança entre fatias)."""
    nz = mask.shape[0]
    labels = np.zeros(mask.shape, np.int32)
    areas = [0]
    offset = 0
    for z in range(nz):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask[z], connectivity=8, ltype=cv2.CV_32S)
        lab[lab > 0] += offset
        labels[z] = lab
        areas.extend(int(a) for a in stats[1:, cv2.CC_STAT_AREA])
        offset += n - 1
    if offset == 0:
        return np.zeros_like(mask)

    uf = _UnionFind(offset + 1)
    for z in range(nz - 1):
        a, b = labels[z], labels[z + 1]
        both = (a > 0) & (b > 0)
        if not both.any():
            continue
        pairs = np.unique(np.stack([a[both], b[both]], axis=1), axis=0)
        for p, q in pairs:
            uf.union(int(p), int(q))

    roots = np.array([uf.find(i) for i in range(offset + 1)], np.int64)
    totals = np.bincount(roots[1:], weights=np.asarray(areas[1:], np.float64), minlength=offset + 1)
    best = int(np.argmax(totals))  # empate → menor rótulo
    keep = (roots == best).astype(np.uint8)
    keep[0] = 0
    return keep[labels]


def fill_holes_2d(mask_slice: np.ndarray) -> np.ndarray:
    h, w = mask_slice.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = mask_slice
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 1)  # fundo ligado à borda (4-vizinhança)
    holes = padded[1:-1, 1:-1] == 0
    return (mask_slice | holes).astype(np.uint8)


def segment_brain(vol: np.ndarray, spacing: tuple[float, float, float], params: Params = Params()) -> np.ndarray:
    """Recebe TC em HU (z, y, x) e devolve a máscara cerebral uint8 {0, 1}."""
    thr = threshold_hu(vol, params.lower_hu, params.upper_hu)
    smooth = gaussian_smooth(thr, spacing, params.sigma_mm)
    # a suavização "vaza" ~1 voxel para dentro do osso; exigir que o voxel original esteja na faixa evita isso
    in_range = (vol >= params.lower_hu) & (vol <= params.upper_hu)
    candidate = ((smooth >= params.brain_min_hu) & (smooth <= params.upper_hu) & in_range).astype(np.uint8)

    se = structuring_element(spacing, params.erosion_mm)
    eroded = candidate if se is None else np.stack([cv2.erode(s, se) for s in candidate])
    core = largest_component_3d(np.ascontiguousarray(eroded))
    grown = core if se is None else np.stack([cv2.dilate(s, se) for s in core])
    mask = grown & candidate
    return np.stack([fill_holes_2d(s) for s in mask]).astype(np.uint8)
