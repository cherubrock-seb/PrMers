#include "example_common.h"

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const uint32_t device = parse_device(argc, argv);
    const char* tune_dir = parse_tune_dir(argc, argv);
    aevum_engine_handle engine = aevum_engine_create(1362763u, 6, device, 0, "", tune_dir);
    if (!engine) throw std::runtime_error(aevum_engine_last_error());

    try {
        require_ok(aevum_engine_set_u32(engine, 0, 3), "set 3");
        require_ok(aevum_engine_square_mul(engine, 0, 3), "square and multiply by 3");
        if (read_small(engine, 0) != 27) throw std::runtime_error("3^2 * 3 mismatch");

        require_ok(aevum_engine_set_u32(engine, 1, 5), "set 5");
        require_ok(aevum_engine_prepare(engine, 1, 1), "prepare 5");
        require_ok(aevum_engine_set_u32(engine, 0, 7), "set 7");
        require_ok(aevum_engine_mul(engine, 0, 1, 2), "7 * 5 * 2");
        if (read_small(engine, 0) != 70) throw std::runtime_error("prepared multiplication mismatch");

        require_ok(aevum_engine_set_u32(engine, 2, 100), "set 100");
        require_ok(aevum_engine_set_u32(engine, 3, 23), "set 23");
        require_ok(aevum_engine_add(engine, 2, 3), "100 + 23");
        if (read_small(engine, 2) != 123) throw std::runtime_error("add mismatch");
        require_ok(aevum_engine_sub_reg(engine, 2, 3), "123 - 23");
        require_ok(aevum_engine_sub_u32(engine, 2, 7), "100 - 7");
        if (read_small(engine, 2) != 93) throw std::runtime_error("subtract mismatch");

        require_ok(aevum_engine_copy(engine, 4, 0), "copy register");
        if (read_small(engine, 4) != 70) throw std::runtime_error("copy mismatch");

        std::vector<uint32_t> words(aevum_engine_word_count(engine), 0);
        words[0] = 0x12345678u;
        require_ok(aevum_engine_set_words(engine, 5, words.data(), words.size()), "set words");
        if (read_small(engine, 5) != 0x12345678u) throw std::runtime_error("word import mismatch");
    } catch (...) {
        aevum_engine_destroy(engine);
        throw;
    }

    aevum_engine_destroy(engine);
    std::cout << "Aevum register operations passed" << std::endl;
    return 0;
}
