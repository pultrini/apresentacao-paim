"""Figuras de comparação (PNG e GIF) feitas só com OpenCV.

Cada linha da figura mostra uma fatia axial:
    TC (janela cerebral) | contorno de cada máscara (+ referência) | mapa de diferenças
No mapa de diferenças: verde = acerto, vermelho = falso positivo, azul = falso negativo.
Sem referência, o mapa compara as duas primeiras máscaras (vermelho = só na 1ª, azul = só na 2ª).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import cv2
import numpy as np

# cores em BGR
GREEN = (80, 200, 80)
RED = (60, 60, 230)
BLUE = (230, 140, 40)
YELLOW = (40, 220, 240)
WHITE = (255, 255, 255)
PALETTE = [(80, 200, 80), (240, 170, 60), (200, 90, 220), (60, 200, 230)]

TILE = 256  # lado de cada painel em pixels
HEADER = 28


def window(ct_slice: np.ndarray, level: float = 40.0, width: float = 80.0) -> np.ndarray:
    lo = level - width / 2
    gray = np.clip((ct_slice - lo) / width * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _fit(img: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Redimensiona respeitando o espaçamento físico, centralizado num quadrado TILE × TILE."""
    h_mm = img.shape[0] * spacing[1]
    w_mm = img.shape[1] * spacing[0]
    scale = TILE / max(h_mm, w_mm)
    size = (max(1, round(w_mm * scale)), max(1, round(h_mm * scale)))
    interp = cv2.INTER_NEAREST if img.dtype == np.uint8 and img.ndim == 2 else cv2.INTER_AREA
    resized = cv2.resize(img, size, interpolation=interp)
    canvas = np.zeros((TILE, TILE) + resized.shape[2:], resized.dtype)
    y0, x0 = (TILE - size[1]) // 2, (TILE - size[0]) // 2
    canvas[y0 : y0 + size[1], x0 : x0 + size[0]] = resized
    return canvas


def _contour(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(base, contours, -1, color, thickness, cv2.LINE_AA)


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    head = np.full((HEADER, tile.shape[1], 3), 30, np.uint8)
    cv2.putText(head, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack([head, tile])


def _diff_tile(base: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = (base * 0.45).astype(np.uint8)
    for sel, color in ((a & b, GREEN), (a & ~b, RED), (~a & b, BLUE)):
        out[sel] = (0.35 * out[sel] + 0.65 * np.array(color)).astype(np.uint8)
    return out


def render_slice(
    ct: np.ndarray,
    masks: list[tuple[str, np.ndarray]],
    ref: np.ndarray | None,
    z: int,
    spacing: tuple[float, float, float],
    flip_y: bool = False,
) -> np.ndarray:
    def prep(img: np.ndarray) -> np.ndarray:
        img = img[::-1] if flip_y else img
        return _fit(np.ascontiguousarray(img), spacing)

    base = prep(window(ct[z]))
    ref_s = prep(ref[z].astype(np.uint8)) > 0 if ref is not None else None
    tiles = [_label(base.copy(), f"TC  z={z}")]

    pred_slices = []
    for i, (name, mask) in enumerate(masks):
        m = prep(mask[z].astype(np.uint8)) > 0
        pred_slices.append(m)
        tile = base.copy()
        if ref_s is not None:
            _contour(tile, ref_s, YELLOW, 3)  # mais grosso, por baixo, para aparecer quando coincide
        _contour(tile, m, PALETTE[i % len(PALETTE)], 1)
        tiles.append(_label(tile, name + (" (amarelo = ref.)" if ref_s is not None else "")))

    if ref_s is not None and pred_slices:
        tiles.append(_label(_diff_tile(base, pred_slices[0], ref_s), f"{masks[0][0]} x ref."))
    elif len(pred_slices) >= 2:
        tiles.append(_label(_diff_tile(base, pred_slices[0], pred_slices[1]), f"{masks[0][0]} x {masks[1][0]}"))

    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, np.full((row.shape[0], 2, 3), 60, np.uint8), t])
    return row


def _legend(width: int, has_ref: bool, names: list[str]) -> np.ndarray:
    bar = np.full((26, width, 3), 20, np.uint8)
    if has_ref:
        items = [("acerto", GREEN), ("falso positivo", RED), ("falso negativo", BLUE), ("referencia", YELLOW)]
    else:
        items = [("ambas", GREEN), (f"so {names[0]}", RED), (f"so {names[1] if len(names) > 1 else '?'}", BLUE)]
    x = 8
    for text, color in items:
        cv2.rectangle(bar, (x, 8), (x + 12, 20), color, -1)
        cv2.putText(bar, text, (x + 18, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        x += 30 + 9 * len(text)
    return bar


def pick_slices(reference: np.ndarray, n: int) -> list[int]:
    """Fatias igualmente espaçadas dentro da extensão do cérebro."""
    area = reference.reshape(reference.shape[0], -1).sum(1)
    zs = np.flatnonzero(area > 0.05 * area.max()) if area.max() > 0 else np.arange(reference.shape[0])
    picks = np.linspace(zs[0], zs[-1], num=min(n, len(zs)))
    return sorted({int(round(p)) for p in picks})


def save_grid(path: Path, rows: list[np.ndarray], legend: np.ndarray) -> None:
    sep = np.full((2, rows[0].shape[1], 3), 60, np.uint8)
    stacked = rows[0]
    for r in rows[1:]:
        stacked = np.vstack([stacked, sep, r])
    cv2.imwrite(str(path), np.vstack([stacked, legend]))


def save_gif(path: Path, frames: list[np.ndarray], frame_ms: int = 120) -> None:
    if not hasattr(cv2, "imwriteanimation"):
        raise RuntimeError("esta versão do OpenCV não grava GIF (precisa de >= 4.11)")
    anim = cv2.Animation()
    # o GIF fica mais leve se reduzido pela metade quando a figura é larga
    scale = 0.5 if frames[0].shape[1] > 1200 else 1.0
    anim.frames = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in frames]
    anim.durations = [frame_ms] * len(frames)
    if not cv2.imwriteanimation(str(path), anim):
        raise RuntimeError(f"falha ao gravar {path}")


def timing_chart(path: Path, timings_jsonl: Path) -> None:
    """Barras horizontais com a mediana dos tempos (segmentação e processo) por implementação."""
    runs = [json.loads(line) for line in timings_jsonl.read_text().splitlines() if line.strip()]
    impls: dict[str, list[dict]] = {}
    for r in runs:
        impls.setdefault(r["impl"], []).append(r)
    fields = [("segment_s", "segmentacao"), ("total_s", "total interno"), ("wall_s", "processo")]
    data = [
        (impl, label, statistics.median(r[f] for r in rs) * 1000)
        for impl, rs in impls.items()
        for f, label in fields
        if f in rs[0]
    ]
    if not data:
        return
    w, row_h, left = 760, 30, 230
    h = 50 + row_h * len(data) + 20
    img = np.full((h, w, 3), 250, np.uint8)
    cv2.putText(img, "Tempo mediano (ms)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    vmax = max(v for *_, v in data)
    colors = {impl: PALETTE[i % len(PALETTE)] for i, impl in enumerate(impls)}
    for i, (impl, label, value) in enumerate(data):
        y = 50 + i * row_h
        cv2.putText(img, f"{impl} - {label}", (12, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        bar = int((w - left - 90) * value / vmax)
        cv2.rectangle(img, (left, y + 4), (left + max(bar, 1), y + row_h - 6), colors[impl], -1)
        cv2.putText(img, f"{value:.0f}", (left + bar + 6, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def make_figures(
    ct: np.ndarray,
    spacing: tuple[float, float, float],
    masks: list[tuple[str, np.ndarray]],
    ref: np.ndarray | None,
    out_dir: Path,
    n_slices: int = 6,
    flip_y: bool = False,
    timings: Path | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in masks]
    guide = ref if ref is not None else masks[0][1]
    written: list[Path] = []

    rows = [render_slice(ct, masks, ref, z, spacing, flip_y) for z in pick_slices(guide, n_slices)]
    legend = _legend(rows[0].shape[1], ref is not None, names)
    grid = out_dir / "comparacao_fatias.png"
    save_grid(grid, rows, legend)
    written.append(grid)

    zs = pick_slices(guide, ct.shape[0])
    frames = [np.vstack([render_slice(ct, masks, ref, z, spacing, flip_y), legend]) for z in zs]
    gif = out_dir / "comparacao_fatias.gif"
    save_gif(gif, frames)
    written.append(gif)

    if timings is not None and timings.exists():
        chart = out_dir / "tempos.png"
        timing_chart(chart, timings)
        if chart.exists():
            written.append(chart)
    return written
