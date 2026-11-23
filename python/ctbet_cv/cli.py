"""Linha de comando: ctbet-cv {segment, metrics, phantom, visualize, report}.

Saídas de `segment` e `metrics` são uma linha JSON (mesmo formato do binário C++).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .metrics import compute_overlap
from .nifti import DT_FLOAT32, DT_INT16, DT_UINT8, new_image, read_nifti, write_nifti
from .phantom import make_phantom
from .report import build_report
from .visualize import make_figures
from .segment import Params, segment_brain


def _cmd_segment(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    img = read_nifti(args.input)
    t1 = time.perf_counter()
    params = Params(
        lower_hu=args.lower,
        upper_hu=args.upper,
        sigma_mm=args.sigma,
        brain_min_hu=args.brain_min,
        erosion_mm=args.erosion,
    )
    mask = segment_brain(img.data, img.spacing, params)
    t2 = time.perf_counter()
    write_nifti(args.output, mask, img, DT_UINT8)
    if args.brain:
        write_nifti(args.brain, np.where(mask > 0, img.data, 0).astype(np.float32), img, DT_FLOAT32)
    t3 = time.perf_counter()
    print(
        json.dumps(
            {
                "impl": "python",
                "input": str(args.input),
                "read_s": t1 - t0,
                "segment_s": t2 - t1,
                "write_s": t3 - t2,
                "total_s": t3 - t0,
                "voxels": int(mask.size),
                "brain_voxels": int(mask.sum()),
            }
        )
    )
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    pred = read_nifti(args.pred)
    ref = read_nifti(args.ref)
    result = compute_overlap(pred.data, ref.data, ref.spacing).as_dict()
    result.update({"pred": str(args.pred), "ref": str(args.ref)})
    print(json.dumps(result))
    return 0


def _cmd_phantom(args: argparse.Namespace) -> int:
    ct, gt = make_phantom(seed=args.seed)
    spacing = (0.9, 0.9, 3.0)
    ref = new_image(ct, spacing)
    Path(args.ct).parent.mkdir(parents=True, exist_ok=True)
    write_nifti(args.ct, ct, ref, DT_INT16)
    write_nifti(args.gt, gt, ref, DT_UINT8)
    print(f"fantoma: {args.ct} ({ct.shape[2]}x{ct.shape[1]}x{ct.shape[0]}), referência: {args.gt}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    text = build_report(args.timings, args.metrics)
    print(text)
    if args.markdown:
        args.markdown.write_text(text + "\n")
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    ct = read_nifti(args.ct)
    ref = read_nifti(args.ref).data > 0 if args.ref else None
    masks = []
    for item in args.mask:
        name, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"--mask espera nome=caminho, recebeu {item!r}")
        masks.append((name, read_nifti(path).data > 0))
    if not masks:
        raise ValueError("informe ao menos uma --mask nome=caminho")
    written = make_figures(ct.data, ct.spacing, masks, ref, args.out, args.slices, args.flip_y, args.timings)
    for path in written:
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    defaults = Params()
    parser = argparse.ArgumentParser(prog="ctbet-cv", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("segment", help="extrai a máscara cerebral de uma TC (.nii/.nii.gz em HU)")
    seg.add_argument("input", type=Path)
    seg.add_argument("output", type=Path, help="máscara de saída (uint8)")
    seg.add_argument("--brain", type=Path, help="também grava a TC mascarada")
    seg.add_argument("--lower", type=float, default=defaults.lower_hu)
    seg.add_argument("--upper", type=float, default=defaults.upper_hu)
    seg.add_argument("--sigma", type=float, default=defaults.sigma_mm, help="σ em mm (0 = sem suavização)")
    seg.add_argument("--brain-min", type=float, default=defaults.brain_min_hu)
    seg.add_argument("--erosion", type=float, default=defaults.erosion_mm, help="raio em mm")
    seg.set_defaults(func=_cmd_segment)

    met = sub.add_parser("metrics", help="Dice/Jaccard/sens./espec. entre duas máscaras")
    met.add_argument("pred", type=Path)
    met.add_argument("ref", type=Path)
    met.set_defaults(func=_cmd_metrics)

    ph = sub.add_parser("phantom", help="gera um fantoma sintético com máscara de referência")
    ph.add_argument("ct", type=Path)
    ph.add_argument("gt", type=Path)
    ph.add_argument("--seed", type=int, default=0)
    ph.set_defaults(func=_cmd_phantom)

    vis = sub.add_parser("visualize", help="gera PNG e GIF comparando máscaras sobre a TC")
    vis.add_argument("ct", type=Path)
    vis.add_argument("--mask", action="append", default=[], metavar="NOME=CAMINHO", help="repita para cada máscara")
    vis.add_argument("--ref", type=Path, help="máscara de referência (opcional)")
    vis.add_argument("--out", type=Path, default=Path("results/figuras"))
    vis.add_argument("--slices", type=int, default=6, help="fatias na figura estática")
    vis.add_argument("--flip-y", action="store_true", help="inverte verticalmente (útil para dados em RAS)")
    vis.add_argument("--timings", type=Path, help="timings.jsonl do compare.sh, para o gráfico de tempos")
    vis.set_defaults(func=_cmd_visualize)

    rep = sub.add_parser("report", help="resume os resultados de scripts/compare.sh")
    rep.add_argument("timings", type=Path)
    rep.add_argument("metrics", type=Path)
    rep.add_argument("--markdown", type=Path, help="também grava o resumo em Markdown")
    rep.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
