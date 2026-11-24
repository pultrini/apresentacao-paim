"""Avaliação no dataset público do artigo (Muschelli et al., 2015): "Non-Contrast Head CT of Stroke
Patients with Brain Masks" (JHU Data Archive, doi:10.7281/T1/CZDPSX, CC BY-NC 4.0).

Estrutura esperada (cada tarball NN.tar.xz extraído):

    <dados>/NN/BRAIN_k_Anonymized.nii.gz        TC em HU
    <dados>/NN/BRAIN_k_Anonymized_Mask.nii.gz   máscara do método original (FSL/BET)
    <dados>/NN/Manual_Mask_k_Reader_r.nii.gz    máscara manual do leitor r (1 ou 2)

Para cada TC compara Python, C++ (opcional) e o BET original com os leitores, além da
concordância entre leitores (teto prático do Dice).
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .metrics import compute_overlap
from .nifti import DT_UINT8, read_nifti, write_nifti
from .segment import Params, segment_brain
from .visualize import PALETTE, make_figures

METHOD_LABELS = {"python": "OpenCV (Python)", "cpp": "OpenCV (C++)", "fsl_bet": "FSL/BET (artigo)", "reader_2": "Leitor 2"}


@dataclass
class Scan:
    subject: str
    index: str
    kind: str  # BRAIN ou BONE
    ct: Path
    fsl_mask: Path | None
    readers: dict[int, Path] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.subject}_{self.kind}{self.index}"


def discover(data_dir: Path, include_bone: bool = False) -> list[Scan]:
    scans: list[Scan] = []
    kinds = ("BRAIN", "BONE") if include_bone else ("BRAIN",)
    for subject_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        for kind in kinds:
            for ct in sorted(subject_dir.glob(f"{kind}_*_Anonymized.nii.gz")):
                index = ct.name.split("_")[1]
                fsl = subject_dir / f"{kind}_{index}_Anonymized_Mask.nii.gz"
                readers = {
                    r: subject_dir / f"Manual_Mask_{index}_Reader_{r}.nii.gz"
                    for r in (1, 2)
                    if (subject_dir / f"Manual_Mask_{index}_Reader_{r}.nii.gz").exists()
                }
                if readers:
                    scans.append(Scan(subject_dir.name, index, kind, ct, fsl if fsl.exists() else None, readers))
    return scans


def _run_cpp(binary: Path, ct: Path, out: Path, params: Params) -> float:
    cmd = [
        str(binary), "segment", str(ct), str(out),
        "--lower", str(params.lower_hu), "--upper", str(params.upper_hu), "--sigma", str(params.sigma_mm),
        "--brain-min", str(params.brain_min_hu), "--erosion", str(params.erosion_mm),
    ]  # fmt: skip
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(json.loads(result.stdout)["segment_s"])


def _fmt(values: list[float], digits: int = 4) -> str:
    if not values:
        return "—"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.{digits}f} ± {sd:.{digits}f}"


def boxplot(path: Path, groups: list[tuple[str, list[float]]], title: str) -> None:
    """Boxplot simples (mediana, quartis, extremos e pontos) desenhado com OpenCV."""
    groups = [(n, v) for n, v in groups if v]
    w, h, left, top, bottom = 220 + 170 * len(groups), 460, 80, 50, 70
    img = np.full((h, w, 3), 250, np.uint8)
    lo = min(min(v) for _, v in groups)
    lo = max(0.0, np.floor((lo - 0.005) * 100) / 100)
    hi = 1.0

    def y(v: float) -> int:
        return int(top + (hi - v) / (hi - lo) * (h - top - bottom))

    cv2.putText(img, title, (left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    for t in np.linspace(lo, hi, 6):
        cv2.line(img, (left, y(t)), (w - 20, y(t)), (225, 225, 225), 1)
        cv2.putText(img, f"{t:.3f}", (10, y(t) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    rng = np.random.default_rng(0)
    for i, (name, values) in enumerate(groups):
        cx = left + 90 + i * 170
        color = PALETTE[i % len(PALETTE)]
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        cv2.line(img, (cx, y(min(values))), (cx, y(max(values))), (120, 120, 120), 1)
        cv2.rectangle(img, (cx - 35, y(q3)), (cx + 35, y(q1)), color, -1)
        cv2.rectangle(img, (cx - 35, y(q3)), (cx + 35, y(q1)), (60, 60, 60), 1)
        cv2.line(img, (cx - 35, y(med)), (cx + 35, y(med)), (30, 30, 30), 2)
        for v in values:
            cv2.circle(img, (cx + int(rng.uniform(-50, 50)), y(v)), 3, (50, 50, 50), -1, cv2.LINE_AA)
        cv2.putText(img, name, (cx - 75, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, f"mediana {med:.4f}", (cx - 70, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def evaluate_dataset(
    data_dir: Path,
    out_dir: Path,
    cpp_binary: Path | None = None,
    params: Params = Params(),
    limit: int | None = None,
    n_figures: int = 3,
    include_bone: bool = False,
    flip_y: bool = True,
) -> Path:
    scans = discover(data_dir, include_bone)[:limit]
    if not scans:
        raise ValueError(f"nenhuma TC encontrada em {data_dir}")
    masks_dir = out_dir / "mascaras"
    masks_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, scan in enumerate(scans, 1):
        image = read_nifti(scan.ct)
        t0 = time.perf_counter()
        mask = segment_brain(image.data, image.spacing, params)
        py_s = time.perf_counter() - t0
        py_path = masks_dir / f"{scan.name}_python.nii.gz"
        write_nifti(py_path, mask, image, DT_UINT8)

        preds: dict[str, tuple[np.ndarray, float | None]] = {"python": (mask, py_s)}
        if cpp_binary is not None:
            cpp_path = masks_dir / f"{scan.name}_cpp.nii.gz"
            cpp_s = _run_cpp(cpp_binary, scan.ct, cpp_path, params)
            preds["cpp"] = (read_nifti(cpp_path).data > 0, cpp_s)
        if scan.fsl_mask is not None:
            preds["fsl_bet"] = (read_nifti(scan.fsl_mask).data > 0, None)
        refs = {r: read_nifti(p).data > 0 for r, p in scan.readers.items()}
        if 2 in refs and 1 in refs:
            preds["reader_2"] = (refs[2], None)

        for method, (pred, seconds) in preds.items():
            for r, ref in refs.items():
                if method == "reader_2" and r == 2:
                    continue
                m = compute_overlap(pred, ref, image.spacing).as_dict()
                rows.append({"scan": scan.name, "method": method, "reference": f"reader_{r}", "segment_s": seconds, **m})
        if "cpp" in preds:
            m = compute_overlap(preds["cpp"][0], mask, image.spacing).as_dict()
            rows.append({"scan": scan.name, "method": "cpp", "reference": "python", "segment_s": None, **m})
        dice = next(r["dice"] for r in rows if r["scan"] == scan.name and r["method"] == "python")
        print(f"[{i:2d}/{len(scans)}] {scan.name}: Dice python×leitor = {dice:.4f}  ({py_s:.2f} s)", flush=True)

    csv_path = out_dir / "metricas_por_tc.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows, scans, data_dir, params)
    (out_dir / "resumo.md").write_text(summary)

    fig_dir = out_dir / "figuras"
    fig_dir.mkdir(exist_ok=True)
    groups = [
        (METHOD_LABELS[m], [r["dice"] for r in rows if r["method"] == m and r["reference"] == "reader_1"])
        for m in ("python", "cpp", "fsl_bet", "reader_2")
    ]
    boxplot(fig_dir / "dice_boxplot.png", groups, f"Dice contra o leitor 1 ({len(groups[0][1])} TCs)")

    ranked = sorted(
        (r for r in rows if r["method"] == "python" and r["reference"] == "reader_1"), key=lambda r: r["dice"]
    )
    picks = {"pior": ranked[0], "mediana": ranked[len(ranked) // 2], "melhor": ranked[-1]}
    for label, row in list(picks.items())[:n_figures]:
        scan = next(s for s in scans if s.name == row["scan"])
        image = read_nifti(scan.ct)
        named = [("python", read_nifti(masks_dir / f"{scan.name}_python.nii.gz").data > 0)]
        if cpp_binary is not None:
            named.append(("cpp", read_nifti(masks_dir / f"{scan.name}_cpp.nii.gz").data > 0))
        if scan.fsl_mask is not None:
            named.append(("FSL/BET", read_nifti(scan.fsl_mask).data > 0))
        ref = read_nifti(scan.readers[min(scan.readers)]).data > 0
        case_dir = fig_dir / f"{label}_{scan.name}"
        make_figures(image.data, image.spacing, named, ref, case_dir, n_slices=5, flip_y=flip_y)
    print(summary)
    return csv_path


def _summarize(rows: list[dict], scans: list[Scan], data_dir: Path, params: Params) -> str:
    def sel(method: str, ref: str, key: str) -> list[float]:
        return [r[key] for r in rows if r["method"] == method and r["reference"] == ref and r[key] is not None]

    lines = [
        "# Avaliação em TC real: stroke_ct_2019 (Muschelli et al., 2015)",
        "",
        f"Dados: `{data_dir}` · {len(scans)} TCs de {len({s.subject for s in scans})} pacientes · "
        f"parâmetros: {params}",
        "",
    ]
    for ref in ("reader_1", "reader_2"):
        lines += [f"## Contra o {ref.replace('_', ' ').replace('reader', 'leitor')}", ""]
        lines += ["| método | n | Dice | Jaccard | sensibilidade | especificidade | Δvolume (%) | Dice mín. |", "|---|---|---|---|---|---|---|---|"]
        for method in ("python", "cpp", "fsl_bet", "reader_2"):
            dice = sel(method, ref, "dice")
            if not dice:
                continue
            lines.append(
                f"| {METHOD_LABELS[method]} | {len(dice)} | {_fmt(dice)} | {_fmt(sel(method, ref, 'jaccard'))} | "
                f"{_fmt(sel(method, ref, 'sensitivity'))} | {_fmt(sel(method, ref, 'specificity'))} | "
                f"{_fmt(sel(method, ref, 'volume_diff_pct'), 2)} | {min(dice):.4f} |"
            )
        lines.append("")

    lines += ["## Tempo de segmentação por TC (s)", "", "| método | mediana | média ± dp |", "|---|---|---|"]
    for method in ("python", "cpp"):
        t = sel(method, "reader_1", "segment_s")
        if t:
            lines.append(f"| {METHOD_LABELS[method]} | {statistics.median(t):.2f} | {_fmt(t, 2)} |")
    agree = sel("cpp", "python", "dice")
    if agree:
        lines += ["", f"Concordância C++ × Python: Dice mínimo = {min(agree):.6f} em {len(agree)} TCs."]
    lines.append("")
    return "\n".join(lines)
