#pragma once

#include "ctbrain/volume.hpp"

#include <array>
#include <cstdint>
#include <filesystem>

namespace ctbrain {

/// Códigos de datatype do NIfTI-1 usados na escrita.
enum class NiftiType : std::int16_t {
    UInt8 = 2,
    Int16 = 4,
    Float32 = 16,
};

/// Imagem NIfTI-1 lida em memória: dados em HU (float32, scl_slope/inter aplicados) + cabeçalho original.
struct NiftiImage {
    std::array<char, 348> header{};
    bool swap_bytes{false};  ///< true se o arquivo tem endianness diferente da máquina
    Spacing spacing{};
    Volume data;  ///< fatias CV_32F

    [[nodiscard]] int width() const;
    [[nodiscard]] int height() const;
    [[nodiscard]] int depth() const;
};

/// Lê .nii ou .nii.gz (zlib lê os dois formatos de forma transparente).
/// Lança std::runtime_error se o arquivo for inválido ou o datatype não suportado.
[[nodiscard]] NiftiImage read_nifti(const std::filesystem::path& path);

/// Grava `volume` com a geometria (cabeçalho) de `reference`. Comprime se o nome terminar em .gz.
void write_nifti(const std::filesystem::path& path,
                 const Volume& volume,
                 const NiftiImage& reference,
                 NiftiType type);

}  // namespace ctbrain
