# Avaliação em TC real: stroke_ct_2019 (Muschelli et al., 2015)

Dados: `data/stroke_ct_2019` · 37 TCs de 35 pacientes · parâmetros: Params(lower_hu=0.0, upper_hu=100.0, sigma_mm=1.0, brain_min_hu=5.0, erosion_mm=4.0)

## Contra o leitor 1

| método | n | Dice | Jaccard | sensibilidade | especificidade | Δvolume (%) | Dice mín. |
|---|---|---|---|---|---|---|---|
| OpenCV (Python) | 36 | 0.9753 ± 0.0203 | 0.9526 ± 0.0367 | 0.9655 ± 0.0228 | 0.9979 ± 0.0040 | -2.00 ± 3.42 | 0.8852 |
| OpenCV (C++) | 36 | 0.9753 ± 0.0203 | 0.9526 ± 0.0367 | 0.9655 ± 0.0228 | 0.9979 ± 0.0040 | -2.00 ± 3.42 | 0.8852 |
| FSL/BET (artigo) | 36 | 0.9811 ± 0.0113 | 0.9631 ± 0.0216 | 0.9728 ± 0.0224 | 0.9984 ± 0.0005 | -1.71 ± 2.36 | 0.9493 |
| Leitor 2 | 34 | 0.9881 ± 0.0057 | 0.9766 ± 0.0109 | 0.9810 ± 0.0112 | 0.9993 ± 0.0002 | -1.45 ± 1.15 | 0.9596 |

## Contra o leitor 2

| método | n | Dice | Jaccard | sensibilidade | especificidade | Δvolume (%) | Dice mín. |
|---|---|---|---|---|---|---|---|
| OpenCV (Python) | 35 | 0.9753 ± 0.0196 | 0.9525 ± 0.0353 | 0.9731 ± 0.0197 | 0.9967 ± 0.0041 | -0.43 ± 3.36 | 0.8822 |
| OpenCV (C++) | 35 | 0.9753 ± 0.0196 | 0.9525 ± 0.0353 | 0.9731 ± 0.0197 | 0.9967 ± 0.0041 | -0.43 ± 3.36 | 0.8822 |
| FSL/BET (artigo) | 35 | 0.9806 ± 0.0089 | 0.9621 ± 0.0169 | 0.9799 ± 0.0185 | 0.9971 ± 0.0008 | -0.16 ± 2.13 | 0.9575 |

## Tempo de segmentação por TC (s)

| método | mediana | média ± dp |
|---|---|---|
| OpenCV (Python) | 0.96 | 1.10 ± 0.46 |
| OpenCV (C++) | 0.34 | 0.40 ± 0.17 |

Concordância C++ × Python: Dice mínimo = 1.000000 em 37 TCs.

## C++ em detalhe (37 TCs, medianas por TC)

| etapa | Python | C++ | aceleração |
|---|---|---|---|
| leitura do NIfTI | 0,199 s | 0,096 s | 2,1× |
| segmentação | 0,995 s | 0,329 s | 3,0× |
| escrita da máscara | 0,083 s | 0,031 s | 2,7× |
| processo inteiro (inclui iniciar o Python) | 1,409 s | 0,470 s | 3,0× |
| soma das 37 TCs (processo) | 61,6 s | 21,1 s | 2,9× |

Contra o FSL/BET, o C++ fica em média 0,0027 de Dice abaixo por TC (mediana da diferença pareada) e vence em 6 de 36 TCs.
As máscaras do C++ são idênticas às do Python em todas as 37 TCs (Dice = 1,0).
