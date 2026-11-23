// ctbrain_cv — extração de parênquima cerebral em TC (Muschelli et al., 2015) só com OpenCV.
//
//   ctbrain_cv segment <entrada.nii[.gz]> <mascara.nii[.gz]> [--brain saida] [--lower v] [--upper v]
//                      [--sigma mm] [--brain-min v] [--erosion mm]
//   ctbrain_cv metrics <predita.nii[.gz]> <referencia.nii[.gz]>
//
// Saída: uma linha JSON, no mesmo formato da versão Python.

#include "ctbrain/metrics.hpp"
#include "ctbrain/nifti.hpp"
#include "ctbrain/segmentation.hpp"
#include "ctbrain/timer.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace ctbrain;

constexpr std::string_view kUsage =
    "uso:\n"
    "  ctbrain_cv segment <entrada> <mascara> [--brain saida] [--lower v] [--upper v]\n"
    "                     [--sigma mm] [--brain-min v] [--erosion mm]\n"
    "  ctbrain_cv metrics <predita> <referencia>\n";

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

double parse_number(const std::string& flag, const std::string& value) {
    try {
        std::size_t used = 0;
        const double number = std::stod(value, &used);
        if (used != value.size()) {
            throw std::invalid_argument(value);
        }
        return number;
    } catch (const std::exception&) {
        throw std::invalid_argument("valor inválido para " + flag + ": " + value);
    }
}

struct SegmentArgs {
    fs::path input;
    fs::path output;
    std::optional<fs::path> brain;
    SegmentationParams params;
};

SegmentArgs parse_segment(const std::vector<std::string>& args) {
    SegmentArgs parsed;
    std::vector<std::string> positional;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg.rfind("--", 0) != 0) {
            positional.push_back(arg);
            continue;
        }
        if (i + 1 >= args.size()) {
            throw std::invalid_argument("faltou o valor de " + arg);
        }
        const std::string& value = args[++i];
        if (arg == "--brain") {
            parsed.brain = value;
        } else if (arg == "--lower") {
            parsed.params.lower_hu = parse_number(arg, value);
        } else if (arg == "--upper") {
            parsed.params.upper_hu = parse_number(arg, value);
        } else if (arg == "--sigma") {
            parsed.params.sigma_mm = parse_number(arg, value);
        } else if (arg == "--brain-min") {
            parsed.params.brain_min_hu = parse_number(arg, value);
        } else if (arg == "--erosion") {
            parsed.params.erosion_mm = parse_number(arg, value);
        } else {
            throw std::invalid_argument("opção desconhecida: " + arg);
        }
    }
    if (positional.size() != 2) {
        throw std::invalid_argument("segment espera <entrada> <mascara>");
    }
    parsed.input = positional[0];
    parsed.output = positional[1];
    return parsed;
}

int run_segment(const std::vector<std::string>& args) {
    const SegmentArgs parsed = parse_segment(args);
    Stopwatch watch;
    const NiftiImage image = read_nifti(parsed.input);
    const double read_s = watch.lap();

    const Volume mask = segment_brain(image.data, image.spacing, parsed.params);
    const double segment_s = watch.lap();

    write_nifti(parsed.output, mask, image, NiftiType::UInt8);
    if (parsed.brain) {
        Volume brain;
        brain.reserve(mask.size());
        for (std::size_t z = 0; z < mask.size(); ++z) {
            cv::Mat masked = cv::Mat::zeros(image.data[z].size(), CV_32F);
            image.data[z].copyTo(masked, mask[z]);
            brain.push_back(masked);
        }
        write_nifti(*parsed.brain, brain, image, NiftiType::Float32);
    }
    const double write_s = watch.lap();

    long long brain_voxels = 0;
    for (const auto& slice : mask) {
        brain_voxels += cv::countNonZero(slice);
    }
    const long long voxels = static_cast<long long>(image.width()) * image.height() * image.depth();

    std::ostringstream json;
    json.precision(10);
    json << R"({"impl": "cpp", "input": ")" << json_escape(parsed.input.string()) << R"(", "read_s": )" << read_s
         << R"(, "segment_s": )" << segment_s << R"(, "write_s": )" << write_s
         << R"(, "total_s": )" << (read_s + segment_s + write_s) << R"(, "voxels": )" << voxels
         << R"(, "brain_voxels": )" << brain_voxels << "}";
    std::cout << json.str() << '\n';
    return 0;
}

int run_metrics(const std::vector<std::string>& args) {
    if (args.size() != 2) {
        throw std::invalid_argument("metrics espera <predita> <referencia>");
    }
    const NiftiImage pred = read_nifti(args[0]);
    const NiftiImage ref = read_nifti(args[1]);
    const OverlapMetrics m = compute_overlap(pred.data, ref.data, ref.spacing);

    std::ostringstream json;
    json.precision(10);
    json << R"({"dice": )" << m.dice << R"(, "jaccard": )" << m.jaccard << R"(, "sensitivity": )"
         << m.sensitivity << R"(, "specificity": )" << m.specificity << R"(, "volume_pred_ml": )"
         << m.volume_pred_ml << R"(, "volume_ref_ml": )" << m.volume_ref_ml << R"(, "volume_diff_pct": )"
         << m.volume_diff_pct << R"(, "pred": ")" << json_escape(args[0]) << R"(", "ref": ")"
         << json_escape(args[1]) << R"("})";
    std::cout << json.str() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "-h" || args[0] == "--help") {
        std::cout << kUsage;
        return args.empty() ? 1 : 0;
    }
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    try {
        if (args[0] == "segment") {
            return run_segment(rest);
        }
        if (args[0] == "metrics") {
            return run_metrics(rest);
        }
        std::cerr << "comando desconhecido: " << args[0] << "\n" << kUsage;
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "erro: " << error.what() << '\n';
        return 1;
    }
}
