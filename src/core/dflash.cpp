#include "strata/core/dflash.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>

namespace strata::core {
namespace {

constexpr char kMagic[4] = {'D', 'F', 'L', '2'};

uint32_t read_u32(const unsigned char*& p, const unsigned char* end, bool& ok) {
    if (!ok || end - p < 4) { ok = false; return 0; }
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}

uint64_t read_u64(const unsigned char*& p, const unsigned char* end, bool& ok) {
    if (!ok || end - p < 8) { ok = false; return 0; }
    uint64_t v;
    std::memcpy(&v, p, 8);
    p += 8;
    return v;
}

bool read_file(const std::string& path, std::vector<unsigned char>& out, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "dflash: cannot open " + path; return false; }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    if (n < 0) { err = "dflash: cannot size " + path; return false; }
    in.seekg(0);
    out.resize(static_cast<size_t>(n));
    if (n > 0 && !in.read(reinterpret_cast<char*>(out.data()), n)) {
        err = "dflash: short read " + path;
        return false;
    }
    return true;
}

}  // namespace

bool DFlashPack::load(const std::string& path, std::string& err) {
    std::string file = path;
    if (!file.empty() && file.back() == '/') file.pop_back();
    std::ifstream dir_probe(file + "/weights.bin", std::ios::binary);
    if (dir_probe) file += "/weights.bin";
    dir_probe.close();

    std::vector<unsigned char> bytes;
    if (!read_file(file, bytes, err)) return false;
    if (bytes.size() < 64 || std::memcmp(bytes.data(), kMagic, 4) != 0) {
        err = "dflash: " + file + " is not a DFL2 pack (run tools/dflash_rt.py)";
        return false;
    }
    const unsigned char* p = bytes.data() + 4;
    const unsigned char* end = bytes.data() + bytes.size();
    bool ok = true;
    const uint32_t version = read_u32(p, end, ok);
    const uint32_t n_tensors = read_u32(p, end, ok);
    (void) read_u32(p, end, ok);
    hidden = read_u32(p, end, ok);
    n_layers = read_u32(p, end, ok);
    n_heads = read_u32(p, end, ok);
    n_kv_heads = read_u32(p, end, ok);
    head_dim = read_u32(p, end, ok);
    intermediate = read_u32(p, end, ok);
    vocab = read_u32(p, end, ok);
    block_size = read_u32(p, end, ok);
    selector_rank = read_u32(p, end, ok);
    selector_top_k = read_u32(p, end, ok);
    conv_kernel = read_u32(p, end, ok);
    conv_group = read_u32(p, end, ok);
    sliding_window = read_u32(p, end, ok);
    mask_token = static_cast<int32_t>(read_u32(p, end, ok));
    n_target_layers = read_u32(p, end, ok);
    if (!ok || version != kVersion || n_target_layers > kMaxTargetLayers || selector_top_k == 0 ||
        selector_top_k > 64 || n_tensors > 10000) {
        err = "dflash: bad DFL2 header in " + file;
        return false;
    }
    for (uint32_t i = 0; i < kMaxTargetLayers; ++i) {
        target_layers[i] = static_cast<int32_t>(read_u32(p, end, ok));
    }
    if (!ok) { err = "dflash: truncated header in " + file; return false; }

    tensors.clear();
    tensors.reserve(n_tensors);
    uint64_t weight_end = 0;
    for (uint32_t i = 0; i < n_tensors; ++i) {
        const uint32_t name_len = read_u32(p, end, ok);
        if (!ok || name_len == 0 || name_len > 512 || end - p < static_cast<std::ptrdiff_t>(name_len)) {
            err = "dflash: bad tensor name in " + file;
            return false;
        }
        DFlashTensor t;
        t.name.assign(reinterpret_cast<const char*>(p), name_len);
        p += name_len;
        t.rank = static_cast<int>(read_u32(p, end, ok));
        if (!ok || t.rank < 1 || t.rank > 4) { err = "dflash: bad rank for " + t.name; return false; }
        for (int d = 0; d < 4; ++d) t.dims[d] = static_cast<int64_t>(read_u64(p, end, ok));
        t.offset = read_u64(p, end, ok);
        t.nbytes = read_u64(p, end, ok);
        if (!ok) { err = "dflash: truncated tensor table in " + file; return false; }
        int64_t count = 1;
        for (int d = 0; d < t.rank; ++d) {
            if (t.dims[d] <= 0) { err = "dflash: bad shape for " + t.name; return false; }
            count *= t.dims[d];
        }
        if (t.nbytes != static_cast<uint64_t>(count) * 2 ||
            t.offset > std::numeric_limits<uint64_t>::max() - t.nbytes) {
            err = "dflash: " + t.name + " is not packed BF16";
            return false;
        }
        if (t.offset + t.nbytes > weight_end) weight_end = t.offset + t.nbytes;
        tensors.push_back(std::move(t));
    }
    const auto header_bytes = static_cast<uint64_t>(p - bytes.data());
    if (header_bytes > bytes.size() || bytes.size() - header_bytes < weight_end) {
        err = "dflash: weights truncated in " + file;
        return false;
    }
    weights.assign(p, p + weight_end);
    for (const auto& t : tensors) {
        if (t.offset + t.nbytes > weights.size()) {
            err = "dflash: " + t.name + " runs past the weight blob";
            return false;
        }
    }
    return true;
}

const DFlashTensor* DFlashPack::find(std::string_view name) const {
    for (const auto& t : tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

int dflash2_select(const float* lattice, int n_rows, int row_stride, int top_k, float min_p,
                   int32_t* out, int out_cap) {
    if (lattice == nullptr || out == nullptr || n_rows < 2 || top_k < 1 || out_cap < 1) return 0;
    if (row_stride < top_k + top_k * top_k) return 0;
    int n = 0;
    int predecessor = 0;
    for (int i = 1; i < n_rows && n < out_cap; ++i) {
        const float* row = lattice + static_cast<size_t>(i) * static_cast<size_t>(row_stride);
        const float* scores = row + top_k + static_cast<size_t>(predecessor) * static_cast<size_t>(top_k);
        int best = 0;
        float peak = scores[0];
        for (int k = 1; k < top_k; ++k) {
            if (scores[k] > peak) { peak = scores[k]; best = k; }
        }
        if (min_p > 0.f) {
            float sum = 0.f;
            for (int k = 0; k < top_k; ++k) sum += std::exp(scores[k] - peak);
            if (1.f / sum < min_p) break;
        }
        out[n++] = static_cast<int32_t>(row[best]);
        predecessor = best;
    }
    return n;
}

}  // namespace strata::core
