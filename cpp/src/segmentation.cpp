#include "ctbrain/segmentation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace ctbrain {
namespace {

int kernel_size(double sigma_vox) { return 2 * static_cast<int>(std::ceil(3.0 * sigma_vox)) + 1; }

int radius_vox(double mm, double spacing) {
    return std::max(0, static_cast<int>(std::floor(mm / spacing + 0.5)));
}

int reflect101(int i, int n) {
    if (n == 1) {
        return 0;
    }
    while (i < 0 || i >= n) {
        i = i < 0 ? -i : 2 * n - 2 - i;
    }
    return i;
}

std::optional<cv::Mat> structuring_element(const Spacing& spacing, double radius_mm) {
    const int rx = radius_vox(radius_mm, spacing.x);
    const int ry = radius_vox(radius_mm, spacing.y);
    if (rx == 0 && ry == 0) {
        return std::nullopt;
    }
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{2 * rx + 1, 2 * ry + 1});
}

/// Union-find cuja raiz é sempre o menor rótulo (resultado independente da ordem das uniões).
class UnionFind {
public:
    explicit UnionFind(std::size_t n) : parent_(n) { std::iota(parent_.begin(), parent_.end(), 0); }

    int find(int a) {
        int root = a;
        while (parent_[static_cast<std::size_t>(root)] != root) {
            root = parent_[static_cast<std::size_t>(root)];
        }
        while (parent_[static_cast<std::size_t>(a)] != root) {
            const int next = parent_[static_cast<std::size_t>(a)];
            parent_[static_cast<std::size_t>(a)] = root;
            a = next;
        }
        return root;
    }

    void unite(int a, int b) {
        const int ra = find(a);
        const int rb = find(b);
        if (ra != rb) {
            parent_[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);
        }
    }

private:
    std::vector<int> parent_;
};

cv::Mat binary_mask(const cv::Mat& condition) {
    cv::Mat out;
    condition.convertTo(out, CV_8U, 1.0 / 255.0);  // cv::compare devolve 0/255
    return out;
}

void require_non_empty(const Volume& volume) {
    if (volume.empty()) {
        throw std::invalid_argument("volume vazio");
    }
}

}  // namespace

Volume threshold_hu(const Volume& ct, double lower, double upper) {
    Volume out;
    out.reserve(ct.size());
    for (const auto& slice : ct) {
        cv::Mat in_range = (slice >= lower) & (slice <= upper);
        cv::Mat kept = cv::Mat::zeros(slice.size(), CV_32F);
        slice.copyTo(kept, in_range);
        out.push_back(kept);
    }
    return out;
}

Volume gaussian_smooth(const Volume& ct, const Spacing& spacing, double sigma_mm) {
    require_non_empty(ct);
    if (sigma_mm <= 0.0) {
        return clone_volume(ct);
    }
    const double sx = sigma_mm / spacing.x;
    const double sy = sigma_mm / spacing.y;
    const double sz = sigma_mm / spacing.z;
    const cv::Mat kx = cv::getGaussianKernel(kernel_size(sx), sx, CV_32F);
    const cv::Mat ky = cv::getGaussianKernel(kernel_size(sy), sy, CV_32F);
    const cv::Mat kz = cv::getGaussianKernel(kernel_size(sz), sz, CV_32F);

    Volume planar;
    planar.reserve(ct.size());
    for (const auto& slice : ct) {
        cv::Mat filtered;
        cv::sepFilter2D(slice, filtered, CV_32F, kx, ky, cv::Point{-1, -1}, 0.0, cv::BORDER_REFLECT_101);
        planar.push_back(filtered);
    }

    const int nz = static_cast<int>(ct.size());
    const int radius = kz.rows / 2;
    Volume out;
    out.reserve(ct.size());
    for (int z = 0; z < nz; ++z) {
        cv::Mat acc = cv::Mat::zeros(ct.front().size(), CV_32F);
        for (int k = 0; k < kz.rows; ++k) {
            const float weight = kz.at<float>(k);
            acc += weight * planar[static_cast<std::size_t>(reflect101(z + k - radius, nz))];
        }
        out.push_back(acc);
    }
    return out;
}

Volume largest_component_3d(const Volume& mask) {
    require_non_empty(mask);
    const std::size_t nz = mask.size();
    Volume labels(nz);
    std::vector<double> areas{0.0};
    int offset = 0;
    for (std::size_t z = 0; z < nz; ++z) {
        cv::Mat stats;
        cv::Mat centroids;
        const int n = cv::connectedComponentsWithStats(mask[z], labels[z], stats, centroids, 8, CV_32S);
        cv::Mat positive = labels[z] > 0;
        cv::add(labels[z], cv::Scalar(offset), labels[z], positive);
        for (int i = 1; i < n; ++i) {
            areas.push_back(stats.at<int>(i, cv::CC_STAT_AREA));
        }
        offset += n - 1;
    }

    Volume out = make_volume(static_cast<int>(nz), mask.front().rows, mask.front().cols, CV_8U);
    if (offset == 0) {
        return out;
    }

    UnionFind sets{static_cast<std::size_t>(offset) + 1};
    for (std::size_t z = 0; z + 1 < nz; ++z) {
        for (int y = 0; y < labels[z].rows; ++y) {
            const int* a = labels[z].ptr<int>(y);
            const int* b = labels[z + 1].ptr<int>(y);
            for (int x = 0; x < labels[z].cols; ++x) {
                if (a[x] > 0 && b[x] > 0) {
                    sets.unite(a[x], b[x]);
                }
            }
        }
    }

    std::vector<int> root(static_cast<std::size_t>(offset) + 1);
    std::vector<double> total(root.size(), 0.0);
    for (int i = 1; i <= offset; ++i) {
        root[static_cast<std::size_t>(i)] = sets.find(i);
        total[static_cast<std::size_t>(root[static_cast<std::size_t>(i)])] += areas[static_cast<std::size_t>(i)];
    }
    int best = 1;
    for (int i = 1; i <= offset; ++i) {  // empate → menor rótulo, como na versão Python
        if (total[static_cast<std::size_t>(i)] > total[static_cast<std::size_t>(best)]) {
            best = i;
        }
    }

    for (std::size_t z = 0; z < nz; ++z) {
        for (int y = 0; y < labels[z].rows; ++y) {
            const int* lab = labels[z].ptr<int>(y);
            auto* dst = out[z].ptr<std::uint8_t>(y);
            for (int x = 0; x < labels[z].cols; ++x) {
                dst[x] = (lab[x] > 0 && root[static_cast<std::size_t>(lab[x])] == best) ? 1 : 0;
            }
        }
    }
    return out;
}

cv::Mat fill_holes_2d(const cv::Mat& mask_slice) {
    cv::Mat padded;
    cv::copyMakeBorder(mask_slice, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::floodFill(padded, cv::Point{0, 0}, cv::Scalar(1));  // fundo ligado à borda (4-vizinhança)
    const cv::Mat inner = padded(cv::Rect{1, 1, mask_slice.cols, mask_slice.rows});
    cv::Mat filled = mask_slice.clone();
    filled.setTo(1, inner == 0);
    return filled;
}

Volume segment_brain(const Volume& ct, const Spacing& spacing, const SegmentationParams& params) {
    require_non_empty(ct);
    const Volume thresholded = threshold_hu(ct, params.lower_hu, params.upper_hu);
    const Volume smooth = gaussian_smooth(thresholded, spacing, params.sigma_mm);

    Volume candidate;
    candidate.reserve(ct.size());
    for (std::size_t z = 0; z < ct.size(); ++z) {
        // a suavização "vaza" ~1 voxel para o osso; o voxel original também precisa estar na faixa
        const cv::Mat keep = (smooth[z] >= params.brain_min_hu) & (smooth[z] <= params.upper_hu) &
                             (ct[z] >= params.lower_hu) & (ct[z] <= params.upper_hu);
        candidate.push_back(binary_mask(keep));
    }

    const auto element = structuring_element(spacing, params.erosion_mm);
    Volume eroded = element ? Volume{} : clone_volume(candidate);
    if (element) {
        for (const auto& slice : candidate) {
            cv::Mat e;
            cv::erode(slice, e, *element);
            eroded.push_back(e);
        }
    }

    const Volume core = largest_component_3d(eroded);

    Volume mask;
    mask.reserve(ct.size());
    for (std::size_t z = 0; z < ct.size(); ++z) {
        cv::Mat grown = core[z];
        if (element) {
            cv::dilate(core[z], grown, *element);
        }
        mask.push_back(fill_holes_2d(grown & candidate[z]));
    }
    return mask;
}

}  // namespace ctbrain
