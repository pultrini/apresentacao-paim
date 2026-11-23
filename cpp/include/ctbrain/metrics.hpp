#pragma once

#include "ctbrain/volume.hpp"

namespace ctbrain {

/// Métricas de sobreposição usadas por Muschelli et al. (2015).
struct OverlapMetrics {
    double dice{0.0};
    double jaccard{0.0};
    double sensitivity{0.0};
    double specificity{0.0};
    double volume_pred_ml{0.0};
    double volume_ref_ml{0.0};
    double volume_diff_pct{0.0};
};

/// Compara duas máscaras (voxel > 0 = cérebro). Lança std::invalid_argument se as formas diferirem.
[[nodiscard]] OverlapMetrics compute_overlap(const Volume& pred, const Volume& ref, const Spacing& spacing);

}  // namespace ctbrain
