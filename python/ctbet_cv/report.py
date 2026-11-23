"""Resume os JSONL gerados por scripts/compare.sh em tabelas (terminal e Markdown)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

_TIME_FIELDS = ("wall_s", "total_s", "segment_s", "read_s", "write_s")
_METRIC_FIELDS = ("dice", "jaccard", "sensitivity", "specificity", "volume_pred_ml", "volume_ref_ml", "volume_diff_pct")


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build_report(timings_path: Path, metrics_path: Path) -> str:
    timings = _load(timings_path)
    metrics = _load(metrics_path)
    parts: list[str] = ["# Comparação Python × C++", ""]

    by_impl: dict[str, list[dict]] = {}
    for t in timings:
        by_impl.setdefault(t["impl"], []).append(t)
    if by_impl:
        rows = []
        for impl, runs in by_impl.items():
            row = [impl, str(len(runs))]
            for field in _TIME_FIELDS:
                values = [r[field] for r in runs if field in r]
                if values:
                    sd = statistics.stdev(values) if len(values) > 1 else 0.0
                    row.append(f"{statistics.median(values) * 1000:.1f} ± {sd * 1000:.1f}")
                else:
                    row.append("—")
            rows.append(row)
        parts += ["## Tempo (ms, mediana ± desvio-padrão)", ""]
        parts.append(_table(["impl", "execuções", "processo", "total interno", "segmentação", "leitura", "escrita"], rows))
        if {"python", "cpp"} <= by_impl.keys():
            seg = {k: statistics.median(r["segment_s"] for r in v) for k, v in by_impl.items()}
            wall = {k: statistics.median(r["wall_s"] for r in v) for k, v in by_impl.items() if "wall_s" in v[0]}
            parts += ["", f"Aceleração do C++ na segmentação: **{seg['python'] / seg['cpp']:.1f}×**"]
            if len(wall) == 2:
                parts.append(f"Aceleração do C++ no processo inteiro (inclui inicialização do Python): **{wall['python'] / wall['cpp']:.1f}×**")
        parts.append("")

    if metrics:
        rows = [[m.get("label", "?")] + [f"{m[f]:.4f}" if abs(m[f]) < 10 else f"{m[f]:.1f}" for f in _METRIC_FIELDS] for m in metrics]
        parts += ["## Métricas de sobreposição", ""]
        parts.append(_table(["comparação", "Dice", "Jaccard", "sens.", "espec.", "vol. pred (mL)", "vol. ref (mL)", "Δvol (%)"], rows))
        parts.append("")
    return "\n".join(parts)
