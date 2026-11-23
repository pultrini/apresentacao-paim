#include "ctbrain/nifti.hpp"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctbrain {
namespace {

constexpr std::size_t kHeaderSize = 348;
constexpr std::size_t kVoxOffset = 352;  // cabeçalho + 4 bytes de extensão vazia

// Offsets dos campos no cabeçalho NIfTI-1.
constexpr std::size_t kOffDim = 40;
constexpr std::size_t kOffDatatype = 70;
constexpr std::size_t kOffBitpix = 72;
constexpr std::size_t kOffPixdim = 76;
constexpr std::size_t kOffVoxOffset = 108;
constexpr std::size_t kOffSclSlope = 112;
constexpr std::size_t kOffSclInter = 116;
constexpr std::size_t kOffCalMax = 124;
constexpr std::size_t kOffCalMin = 128;
constexpr std::size_t kOffMagic = 344;

struct GzCloser {
    void operator()(gzFile_s* file) const noexcept { gzclose(file); }
};
using GzHandle = std::unique_ptr<gzFile_s, GzCloser>;

template <typename T>
T byteswap(T value) {
    std::array<unsigned char, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    std::reverse(bytes.begin(), bytes.end());
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
}

template <typename T>
T load(const char* base, std::size_t offset, bool swap) {
    T value{};
    std::memcpy(&value, base + offset, sizeof(T));
    return swap ? byteswap(value) : value;
}

template <typename T>
void store(char* base, std::size_t offset, T value, bool swap) {
    if (swap) {
        value = byteswap(value);
    }
    std::memcpy(base + offset, &value, sizeof(T));
}

bool ends_with_gz(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    return name.size() >= 3 && name.compare(name.size() - 3, 3, ".gz") == 0;
}

std::vector<char> read_all(const std::filesystem::path& path) {
    GzHandle file{gzopen(path.string().c_str(), "rb")};
    if (!file) {
        throw std::runtime_error("não foi possível abrir " + path.string());
    }
    std::vector<char> buffer;
    std::array<char, 1 << 16> chunk{};
    int got = 0;
    while ((got = gzread(file.get(), chunk.data(), static_cast<unsigned>(chunk.size()))) > 0) {
        buffer.insert(buffer.end(), chunk.data(), chunk.data() + got);
    }
    if (got < 0) {
        throw std::runtime_error("erro ao descompactar " + path.string());
    }
    return buffer;
}

void write_all(const std::filesystem::path& path, const std::vector<char>& bytes) {
    if (ends_with_gz(path)) {
        GzHandle file{gzopen(path.string().c_str(), "wb6")};
        if (!file || gzwrite(file.get(), bytes.data(), static_cast<unsigned>(bytes.size())) !=
                         static_cast<int>(bytes.size())) {
            throw std::runtime_error("erro ao gravar " + path.string());
        }
        return;
    }
    std::ofstream out{path, std::ios::binary};
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("erro ao gravar " + path.string());
    }
}

template <typename T>
void decode(const char* src, bool swap, float slope, float inter, NiftiImage& image, int nx, int ny) {
    const bool scale = slope != 0.0F && (slope != 1.0F || inter != 0.0F);
    std::size_t index = 0;
    for (auto& slice : image.data) {
        for (int y = 0; y < ny; ++y) {
            auto* row = slice.ptr<float>(y);
            for (int x = 0; x < nx; ++x, ++index) {
                const T raw = load<T>(src, index * sizeof(T), swap);
                const auto value = static_cast<float>(raw);
                row[x] = scale ? value * slope + inter : value;
            }
        }
    }
}

template <typename T>
void encode(const Volume& volume, bool swap, char* dst) {
    std::size_t index = 0;
    for (const auto& slice : volume) {
        cv::Mat as_float;
        slice.convertTo(as_float, CV_32F);
        for (int y = 0; y < as_float.rows; ++y) {
            const auto* row = as_float.ptr<float>(y);
            for (int x = 0; x < as_float.cols; ++x, ++index) {
                store<T>(dst, index * sizeof(T), static_cast<T>(row[x]), swap);
            }
        }
    }
}

int bits_of(NiftiType type) {
    switch (type) {
        case NiftiType::UInt8: return 8;
        case NiftiType::Int16: return 16;
        case NiftiType::Float32: return 32;
    }
    throw std::invalid_argument("tipo NIfTI desconhecido");
}

}  // namespace

int NiftiImage::width() const { return data.empty() ? 0 : data.front().cols; }
int NiftiImage::height() const { return data.empty() ? 0 : data.front().rows; }
int NiftiImage::depth() const { return static_cast<int>(data.size()); }

NiftiImage read_nifti(const std::filesystem::path& path) {
    const std::vector<char> raw = read_all(path);
    if (raw.size() < kHeaderSize) {
        throw std::runtime_error(path.string() + ": arquivo truncado");
    }

    NiftiImage image;
    std::copy_n(raw.begin(), kHeaderSize, image.header.begin());
    const char* hdr = image.header.data();

    if (load<std::int32_t>(hdr, 0, false) == static_cast<std::int32_t>(kHeaderSize)) {
        image.swap_bytes = false;
    } else if (load<std::int32_t>(hdr, 0, true) == static_cast<std::int32_t>(kHeaderSize)) {
        image.swap_bytes = true;
    } else {
        throw std::runtime_error(path.string() + ": não é NIfTI-1");
    }
    if (std::memcmp(hdr + kOffMagic, "n+1", 3) != 0) {
        throw std::runtime_error(path.string() + ": magic inválido (só .nii de arquivo único)");
    }

    const bool swap = image.swap_bytes;
    std::array<std::int16_t, 8> dim{};
    for (std::size_t i = 0; i < dim.size(); ++i) {
        dim[i] = load<std::int16_t>(hdr, kOffDim + 2 * i, swap);
    }
    const int nx = dim[1];
    const int ny = dim[0] >= 2 ? dim[2] : 1;
    const int nz = dim[0] >= 3 ? dim[3] : 1;  // 4D: usa só o primeiro volume
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::runtime_error(path.string() + ": dimensões inválidas");
    }

    const auto datatype = load<std::int16_t>(hdr, kOffDatatype, swap);
    const auto bitpix = load<std::int16_t>(hdr, kOffBitpix, swap);
    const auto vox_offset = static_cast<std::size_t>(load<float>(hdr, kOffVoxOffset, swap));
    const auto slope = load<float>(hdr, kOffSclSlope, swap);
    const auto inter = load<float>(hdr, kOffSclInter, swap);
    image.spacing = {std::abs(load<float>(hdr, kOffPixdim + 4, swap)),
                     std::abs(load<float>(hdr, kOffPixdim + 8, swap)),
                     std::abs(load<float>(hdr, kOffPixdim + 12, swap))};
    for (double* s : {&image.spacing.x, &image.spacing.y, &image.spacing.z}) {
        if (*s == 0.0) {
            *s = 1.0;
        }
    }

    const std::size_t count = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t needed = vox_offset + count * static_cast<std::size_t>(bitpix / 8);
    if (raw.size() < needed) {
        throw std::runtime_error(path.string() + ": dados de voxel truncados");
    }

    image.data = make_volume(nz, ny, nx, CV_32F);
    const char* voxels = raw.data() + vox_offset;
    switch (datatype) {
        case 2: decode<std::uint8_t>(voxels, swap, slope, inter, image, nx, ny); break;
        case 4: decode<std::int16_t>(voxels, swap, slope, inter, image, nx, ny); break;
        case 8: decode<std::int32_t>(voxels, swap, slope, inter, image, nx, ny); break;
        case 16: decode<float>(voxels, swap, slope, inter, image, nx, ny); break;
        case 64: decode<double>(voxels, swap, slope, inter, image, nx, ny); break;
        case 256: decode<std::int8_t>(voxels, swap, slope, inter, image, nx, ny); break;
        case 512: decode<std::uint16_t>(voxels, swap, slope, inter, image, nx, ny); break;
        case 768: decode<std::uint32_t>(voxels, swap, slope, inter, image, nx, ny); break;
        default: throw std::runtime_error(path.string() + ": datatype " + std::to_string(datatype) + " não suportado");
    }
    return image;
}

void write_nifti(const std::filesystem::path& path,
                 const Volume& volume,
                 const NiftiImage& reference,
                 NiftiType type) {
    if (volume.empty()) {
        throw std::invalid_argument("volume vazio");
    }
    const int nx = volume.front().cols;
    const int ny = volume.front().rows;
    const int nz = static_cast<int>(volume.size());
    const int bytes_per_voxel = bits_of(type) / 8;
    const std::size_t count = static_cast<std::size_t>(nx) * ny * nz;

    std::vector<char> out(kVoxOffset + count * static_cast<std::size_t>(bytes_per_voxel), 0);
    std::copy(reference.header.begin(), reference.header.end(), out.begin());
    char* hdr = out.data();
    const bool swap = reference.swap_bytes;

    const std::array<std::int16_t, 8> dim{3, static_cast<std::int16_t>(nx), static_cast<std::int16_t>(ny),
                                          static_cast<std::int16_t>(nz), 1, 1, 1, 1};
    for (std::size_t i = 0; i < dim.size(); ++i) {
        store<std::int16_t>(hdr, kOffDim + 2 * i, dim[i], swap);
    }
    store<std::int16_t>(hdr, kOffDatatype, static_cast<std::int16_t>(type), swap);
    store<std::int16_t>(hdr, kOffBitpix, static_cast<std::int16_t>(bits_of(type)), swap);
    store<float>(hdr, kOffVoxOffset, static_cast<float>(kVoxOffset), swap);
    store<float>(hdr, kOffSclSlope, 1.0F, swap);
    store<float>(hdr, kOffSclInter, 0.0F, swap);
    store<float>(hdr, kOffCalMax, 0.0F, swap);
    store<float>(hdr, kOffCalMin, 0.0F, swap);
    std::memcpy(hdr + kOffMagic, "n+1\0", 4);

    char* voxels = out.data() + kVoxOffset;
    switch (type) {
        case NiftiType::UInt8: encode<std::uint8_t>(volume, swap, voxels); break;
        case NiftiType::Int16: encode<std::int16_t>(volume, swap, voxels); break;
        case NiftiType::Float32: encode<float>(volume, swap, voxels); break;
    }
    write_all(path, out);
}

}  // namespace ctbrain
