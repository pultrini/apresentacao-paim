#include "ctbrain/metrics.hpp"

#include <stdexcept>

namespace ctbrain {
namespace {

double ratio(double num, double den) { return den > 0.0 ? num / den : 1.0; }

}  // namespace

OverlapMetrics compute_overlap(const Volume& pred, const Volume& ref, const Spacing& spacing) {
    if (pred.size() != ref.size() || pred.empty() || pred.front().size() != ref.front().size()) {
        throw std::invalid_argument("máscaras com formas diferentes");
    }
    double tp = 0.0;
    double fp = 0.0;
    double fn = 0.0;
    double total = 0.0;
    for (std::size_t z = 0; z < pred.size(); ++z) {
        const cv::Mat p = pred[z] > 0;
        const cv::Mat r = ref[z] > 0;
        tp += cv::countNonZero(p & r);
        fp += cv::countNonZero(p & ~r);
        fn += cv::countNonZero(~p & r);
        total += static_cast<double>(p.total());
    }
    const double tn = total - tp - fp - fn;
    const double voxel_ml = spacing.x * spacing.y * spacing.z / 1000.0;

    OverlapMetrics m;
    m.dice = ratio(2.0 * tp, 2.0 * tp + fp + fn);
    m.jaccard = ratio(tp, tp + fp + fn);
    m.sensitivity = ratio(tp, tp + fn);
    m.specificity = ratio(tn, tn + fp);
    m.volume_pred_ml = (tp + fp) * voxel_ml;
    m.volume_ref_ml = (tp + fn) * voxel_ml;
    m.volume_diff_pct = m.volume_ref_ml > 0.0 ? 100.0 * (m.volume_pred_ml - m.volume_ref_ml) / m.volume_ref_ml : 0.0;
    return m;
}

}  // namespace ctbrain
