#include "ctbrain/volume.hpp"

namespace ctbrain {

Volume make_volume(int depth, int rows, int cols, int type) {
    Volume volume;
    volume.reserve(static_cast<std::size_t>(depth));
    for (int z = 0; z < depth; ++z) {
        volume.emplace_back(cv::Mat::zeros(rows, cols, type));
    }
    return volume;
}

Volume clone_volume(const Volume& volume) {
    Volume copy;
    copy.reserve(volume.size());
    for (const auto& slice : volume) {
        copy.push_back(slice.clone());
    }
    return copy;
}

}  // namespace ctbrain
