# Artigo de referência

Muschelli J, Ullman NL, Mould WA, Vespa P, Hanley DF, Crainiceanu CM.
**Validated automatic brain extraction of head CT images.**
*NeuroImage*, 114:379–385, 2015. doi:10.1016/j.neuroimage.2015.03.074
PMID 25862260 · PMCID PMC4446187 (acesso aberto)

- PubMed Central: https://pmc.ncbi.nlm.nih.gov/articles/PMC4446187/
- DOI: https://doi.org/10.1016/j.neuroimage.2015.03.074
- Código original (R + FSL): https://github.com/muschellij2/CT_BET
- Adaptação em Python (FSL): https://github.com/GravO8/CT-preprocess

> O PDF não pôde ser baixado automaticamente (o PMC bloqueia acesso
> automatizado). Baixe pelo link do PMC acima e salve nesta pasta como
> `Muschelli2015_NeuroImage.pdf`.

## Método original

1. Limiarizar a TC em 0–100 HU (valores fora da faixa → 0).
2. (Variante recomendada) Suavizar com gaussiana 3D, σ = 1 mm, e limiarizar de novo em 0–100 HU.
3. Aplicar o BET do FSL com fração de intensidade (FI) 0,01, 0,1 ou 0,35.
4. Preencher os buracos da máscara.

Resultado: suavizar antes do BET melhora todas as métricas, exceto a especificidade; FI 0,01 ou 0,1 é melhor do que 0,35.
As métricas usadas foram Dice, Jaccard, sensibilidade, especificidade e diferença de volume, comparadas com máscaras manuais
(teste de postos sinalizados de Wilcoxon).

## O que muda nesta reimplementação

O BET (superfície deformável) é substituído por morfologia matemática com OpenCV:
erosão → maior componente conexo 3D → dilatação reconstrutiva → preenchimento de buracos.
Não há registro em template (FLIRT) nem dependência do FSL.
