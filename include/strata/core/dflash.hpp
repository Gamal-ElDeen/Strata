// include/strata/core/dflash.hpp
//
// Opt-in DFlash2 draft pack. The serve path does not draft with this yet: the verify window still
// belongs to MtpDrafter. This loads the pack tools/dflash_rt.py writes from a local
// DFlash2DraftModel snapshot and walks a selector lattice the way llama.cpp's draft-dflash does.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strata::core {

struct DFlashTensor {
    std::string name;
    int rank = 0;
    int64_t dims[4] = {};
    uint64_t offset = 0;   // into weights
    uint64_t nbytes = 0;
};

struct DFlashPack {
    static constexpr uint32_t kVersion = 1;
    static constexpr int kMaxTargetLayers = 8;

    uint32_t hidden = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t intermediate = 0;
    uint32_t vocab = 0;
    uint32_t block_size = 0;
    uint32_t selector_rank = 0;
    uint32_t selector_top_k = 0;
    uint32_t conv_kernel = 0;
    uint32_t conv_group = 0;
    uint32_t sliding_window = 0;
    int32_t mask_token = -1;
    int32_t target_layers[kMaxTargetLayers] = {};
    uint32_t n_target_layers = 0;
    std::vector<DFlashTensor> tensors;
    std::vector<unsigned char> weights;

    /// `path` is a .dfl2 file written by tools/dflash_rt.py.
    bool load(const std::string& path, std::string& err);
    const DFlashTensor* find(std::string_view name) const;
    uint64_t weight_bytes() const { return weights.size(); }
};

/// Walk one DFlash2 lattice. `lattice` holds `n_rows` rows of `row_stride` floats.
/// Row 0 is the anchor slot and is not a draft. Each later row is `top_k` token ids
/// stored as floats, then a `top_k` by `top_k` score matrix. The walk matches
/// llama.cpp's draft-dflash reader: predecessor 0, argmax of that predecessor's
/// scores, stop before a pick whose softmax peak is below `min_p` (`min_p` <= 0
/// keeps every pick). Writes draft token ids to `out` and returns how many.
int dflash2_select(const float* lattice, int n_rows, int row_stride, int top_k, float min_p,
                   int32_t* out, int out_cap);

}  // namespace strata::core
