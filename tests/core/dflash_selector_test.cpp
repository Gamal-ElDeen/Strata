// Host-only check of the DFlash2 lattice walk and the DFL2 reader. No GPU.
#include "strata/core/dflash.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

int fail(const char* what) {
    std::cerr << "dflash_selector_test: " << what << "\n";
    return 1;
}

void put_u32(std::vector<unsigned char>& out, uint32_t v) {
    unsigned char b[4];
    std::memcpy(b, &v, 4);
    out.insert(out.end(), b, b + 4);
}

void put_u64(std::vector<unsigned char>& out, uint64_t v) {
    unsigned char b[8];
    std::memcpy(b, &v, 8);
    out.insert(out.end(), b, b + 8);
}

}  // namespace

int main() {
    const int top_k = 2;
    const int stride = top_k + top_k * top_k;
    float lattice[3 * 6] = {};
    // Row 1, predecessor 0 scores pick candidate 1 (token 11).
    lattice[stride + 0] = 10.f;
    lattice[stride + 1] = 11.f;
    lattice[stride + 2] = 0.1f;
    lattice[stride + 3] = 0.9f;
    // Row 2, predecessor 1 scores pick candidate 1 (token 21).
    lattice[2 * stride + 0] = 20.f;
    lattice[2 * stride + 1] = 21.f;
    lattice[2 * stride + 4] = 0.2f;
    lattice[2 * stride + 5] = 5.f;
    int32_t out[4] = {};
    if (strata::core::dflash2_select(lattice, 3, stride, top_k, 0.f, out, 4) != 2 ||
        out[0] != 11 || out[1] != 21) {
        return fail("walk did not follow the lattice");
    }
    // Peak softmax of [0.1, 0.9] is about 0.69, so a 0.8 floor stops before a pick.
    if (strata::core::dflash2_select(lattice, 3, stride, top_k, 0.8f, out, 4) != 0) {
        return fail("min_p did not stop the first uncertain pick");
    }

    std::vector<unsigned char> pack;
    pack.insert(pack.end(), {'D', 'F', 'L', '2'});
    put_u32(pack, 1);
    put_u32(pack, 1);
    put_u32(pack, 0);
    const uint32_t geom[] = {4, 1, 1, 1, 2, 8, 16, 8, 4, 2, 2, 2, 32};
    for (uint32_t v : geom) put_u32(pack, v);
    put_u32(pack, 7);  // mask token
    put_u32(pack, 1);
    put_u32(pack, 5);
    for (int i = 0; i < 7; ++i) put_u32(pack, 0xFFFFFFFFu);
    const char* name = "fc.weight";
    put_u32(pack, static_cast<uint32_t>(std::strlen(name)));
    pack.insert(pack.end(), name, name + std::strlen(name));
    put_u32(pack, 2);
    put_u64(pack, 2);
    put_u64(pack, 2);
    put_u64(pack, 0);
    put_u64(pack, 0);
    put_u64(pack, 0);
    put_u64(pack, 8);
    unsigned char weights[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    pack.insert(pack.end(), weights, weights + 8);

    const char* path = "/tmp/dflash-selector-test.dfl2";
    {
        std::ofstream out_file(path, std::ios::binary);
        out_file.write(reinterpret_cast<const char*>(pack.data()), static_cast<std::streamsize>(pack.size()));
    }
    strata::core::DFlashPack loaded;
    std::string err;
    if (!loaded.load(path, err)) return fail(err.c_str());
    if (loaded.hidden != 4 || loaded.mask_token != 7 || loaded.target_layers[0] != 5 ||
        loaded.n_target_layers != 1) {
        return fail("header geometry was not read back");
    }
    const auto* tensor = loaded.find("fc.weight");
    if (tensor == nullptr || tensor->nbytes != 8 || loaded.weights[7] != 8) {
        return fail("fc.weight was not stored");
    }
    std::cout << "dflash_selector_test: ok\n";
    return 0;
}
