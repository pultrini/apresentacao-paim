#!/usr/bin/env bash
# Avalia Python, C++ e o FSL/BET original contra as máscaras manuais do dataset do artigo.
#
# Dataset: "Non-Contrast Head CT of Stroke Patients with Brain Masks" (Muschelli et al., 2015)
#   https://archive.data.jhu.edu/dataset.xhtml?persistentId=doi:10.7281/T1/CZDPSX  (CC BY-NC 4.0)
#   Baixe 01.tar.xz … 35.tar.xz para data/stroke_ct_2019/tarballs/ (≈305 MB).
#
# uso: scripts/avaliar_stroke_ct.sh [pasta_dados] [pasta_resultados]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${1:-$ROOT/data/stroke_ct_2019}"
OUT="${2:-$ROOT/results/stroke_ct_2019}"

[[ -d "$DATA" ]] || { echo "erro: $DATA não existe (veja o cabeçalho deste script)" >&2; exit 1; }

if [[ -d "$DATA/tarballs" ]]; then
    for tarball in "$DATA"/tarballs/*.tar.xz; do
        subject="$(basename "$tarball" .tar.xz)"
        [[ -d "$DATA/$subject" ]] || tar xJf "$tarball" -C "$DATA"
    done
fi

uv sync --project "$ROOT" --quiet
make -C "$ROOT/cpp" --no-print-directory -s
"$ROOT/.venv/bin/ctbet-cv" evaluate-dataset "$DATA" --cpp "$ROOT/cpp/bin/ctbrain_cv" --out "$OUT"
