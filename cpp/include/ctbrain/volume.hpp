#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace ctbrain {

/// Espaçamento do voxel em mm (x = colunas, y = linhas, z = fatias).
struct Spacing {
    double x{1.0};
    double y{1.0};
    double z{1.0};
};

/// Volume 3D guardado como uma pilha de fatias axiais (cada cv::Mat é uma fatia y × x).
/// Imagens de TC usam CV_32F (HU); máscaras usam CV_8U com valores {0, 1}.
using Volume = std::vector<cv::Mat>;

/// Cria um volume com `depth` fatias de `rows` × `cols`, preenchido com zero.
[[nodiscard]] Volume make_volume(int depth, int rows, int cols, int type);

/// Cópia profunda (cv::Mat compartilha dados por padrão).
[[nodiscard]] Volume clone_volume(const Volume& volume);

}  // namespace ctbrain
