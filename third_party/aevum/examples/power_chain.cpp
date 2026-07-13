#include "example_common.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <iostream>
#include <vector>

using boost::multiprecision::cpp_int;

static cpp_int read_integer(aevum_engine_handle engine, size_t reg) {
    std::vector<uint32_t> words(aevum_engine_word_count(engine));
    require_ok(aevum_engine_get_words(engine, reg, words.data(), words.size()), "get_words");
    cpp_int value = 0;
    for (size_t i = words.size(); i-- > 0;) {
        value <<= 32;
        value += words[i];
    }
    return value;
}

int main(int argc, char** argv) {
    const uint32_t device = parse_device(argc, argv);
    const char* tune_dir = parse_tune_dir(argc, argv);
    aevum_engine_handle engine = aevum_engine_create(1362763u, 3, device, 0, "", tune_dir);
    if (!engine) throw std::runtime_error(aevum_engine_last_error());

    cpp_int expected = 3;
    try {
        require_ok(aevum_engine_set_u32(engine, 0, 3), "set base");
        for (unsigned i = 0; i < 10; ++i) {
            require_ok(aevum_engine_square_mul(engine, 0, 3), "power chain step");
            expected = expected * expected * 3;
        }
        if (read_integer(engine, 0) != expected) throw std::runtime_error("power chain mismatch");

        require_ok(aevum_engine_set_u32(engine, 1, 7), "set multiplier");
        require_ok(aevum_engine_prepare(engine, 1, 1), "prepare multiplier");
        require_ok(aevum_engine_mul(engine, 0, 1, 5), "prepared multiply with factor 5");
        expected *= 35;
        if (read_integer(engine, 0) != expected) throw std::runtime_error("prepared chain mismatch");
    } catch (...) {
        aevum_engine_destroy(engine);
        throw;
    }

    aevum_engine_destroy(engine);
    std::cout << "Aevum power chain passed, bits=" << boost::multiprecision::msb(expected) + 1 << std::endl;
    return 0;
}
