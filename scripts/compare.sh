#!/usr/bin/env bash
# Compara tempo de execução e métricas entre as versões Python (uv) e C++.
#
# uso: scripts/compare.sh [-i ct.nii.gz] [-g referencia.nii.gz] [-n repeticoes] [-o pasta]
#
#   -i  TC de entrada em HU (.nii/.nii.gz). Sem -i, gera e usa o fantoma sintético.
#   -g  máscara de referência (manual). Com o fantoma, a referência é gerada junto.
#   -n  número de repetições por implementação (padrão: 5)
#   -o  pasta de resultados (padrão: results/)
#
# Saídas: <pasta>/timings.jsonl, metrics.jsonl, summary.md, as máscaras e figuras/ (PNG e GIF).
# Compatível com o bash 3.2 do macOS.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS=5
INPUT=""
REF=""
OUT="$ROOT/results"

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts "i:g:n:o:h" opt; do
    case "$opt" in
        i) INPUT="$OPTARG" ;;
        g) REF="$OPTARG" ;;
        n) RUNS="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done

die() { echo "erro: $*" >&2; exit 1; }
log() { echo "==> $*" >&2; }

command -v uv >/dev/null || die "uv não encontrado (https://docs.astral.sh/uv/)"
command -v perl >/dev/null || die "perl não encontrado (usado para medir o tempo)"
[[ "$RUNS" =~ ^[0-9]+$ && "$RUNS" -ge 1 ]] || die "-n deve ser um inteiro >= 1"

# relógio de parede em segundos com microssegundos (date +%N não existe no macOS)
now() { perl -MTime::HiRes=time -e 'printf "%.6f\n", time'; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f", b - a }'; }
# acrescenta campos a um objeto JSON de uma linha: add_fields '{"a":1}' '"b": 2'
add_fields() { printf '%s, %s}\n' "${1%\}}" "$2"; }

log "preparando ambiente Python (uv sync)"
uv sync --project "$ROOT" --quiet
PY="$ROOT/.venv/bin/ctbet-cv"

log "compilando C++ (make)"
make -C "$ROOT/cpp" --no-print-directory -s
CPP="$ROOT/cpp/bin/ctbrain_cv"

mkdir -p "$OUT"
if [[ -z "$INPUT" ]]; then
    INPUT="$ROOT/data/phantom_ct.nii.gz"
    REF="$ROOT/data/phantom_gt.nii.gz"
    if [[ ! -f "$INPUT" || ! -f "$REF" ]]; then
        log "gerando fantoma sintético"
        "$PY" phantom "$INPUT" "$REF" >&2
    fi
fi
[[ -f "$INPUT" ]] || die "entrada não encontrada: $INPUT"
[[ -z "$REF" || -f "$REF" ]] || die "referência não encontrada: $REF"

TIMINGS="$OUT/timings.jsonl"
METRICS="$OUT/metrics.jsonl"
: > "$TIMINGS"
: > "$METRICS"

for impl in python cpp; do
    if [[ "$impl" == python ]]; then bin="$PY"; else bin="$CPP"; fi
    mask="$OUT/mask_${impl}.nii.gz"
    log "$impl: $RUNS execução(ões)"
    "$bin" segment "$INPUT" "$mask" >/dev/null  # aquecimento (cache de disco, carga das bibliotecas)
    for run in $(seq 1 "$RUNS"); do
        t0="$(now)"
        json="$("$bin" segment "$INPUT" "$mask")"
        t1="$(now)"
        add_fields "$json" "\"wall_s\": $(elapsed "$t0" "$t1"), \"run\": $run" >> "$TIMINGS"
    done
done

log "calculando métricas"
metric() {  # metric <rótulo> <predita> <referência>
    add_fields "$("$PY" metrics "$2" "$3")" "\"label\": \"$1\"" >> "$METRICS"
}
metric "python_vs_cpp" "$OUT/mask_python.nii.gz" "$OUT/mask_cpp.nii.gz"
if [[ -n "$REF" ]]; then
    metric "python_vs_ref" "$OUT/mask_python.nii.gz" "$REF"
    metric "cpp_vs_ref" "$OUT/mask_cpp.nii.gz" "$REF"
fi

log "gerando figuras (PNG e GIF)"
vis_args=(--mask "python=$OUT/mask_python.nii.gz" --mask "cpp=$OUT/mask_cpp.nii.gz" --out "$OUT/figuras" --timings "$TIMINGS")
if [[ -n "$REF" ]]; then vis_args+=(--ref "$REF"); fi
"$PY" visualize "$INPUT" "${vis_args[@]}" >&2

"$PY" report "$TIMINGS" "$METRICS" --markdown "$OUT/summary.md"
log "resultados em $OUT"
