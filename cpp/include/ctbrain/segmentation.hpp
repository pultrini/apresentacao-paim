#pragma once

#include "ctbrain/volume.hpp"

namespace ctbrain {

/// Parâmetros da extração (mesmos padrões da versão Python).
struct SegmentationParams {
    double lower_hu{0.0};       ///< limiar inferior (artigo: 0 HU)
    double upper_hu{100.0};     ///< limiar superior (artigo: 100 HU)
    double sigma_mm{1.0};       ///< σ da gaussiana 3D; 0 desliga a suavização
    double brain_min_hu{5.0};   ///< valor mínimo na imagem suavizada para entrar na candidata
    double erosion_mm{4.0};     ///< raio do elemento estruturante elíptico
};

/// Zera os voxels fora de [lower, upper].
[[nodiscard]] Volume threshold_hu(const Volume& ct, double lower, double upper);

/// Gaussiana 3D separável com borda REFLECT_101 (σ em mm convertido por eixo).
[[nodiscard]] Volume gaussian_smooth(const Volume& ct, const Spacing& spacing, double sigma_mm);

/// Maior componente conexo 3D (8-vizinhança no plano, 6-vizinhança entre fatias).
[[nodiscard]] Volume largest_component_3d(const Volume& mask);

/// Preenche buracos de uma fatia binária (fundo não ligado à borda vira 1).
[[nodiscard]] cv::Mat fill_holes_2d(const cv::Mat& mask_slice);

/// Pipeline completo: TC em HU → máscara cerebral CV_8U {0, 1}.
[[nodiscard]] Volume segment_brain(const Volume& ct,
                                   const Spacing& spacing,
                                   const SegmentationParams& params = {});

}  // namespace ctbrain
