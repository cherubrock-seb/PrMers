#include "marin/engine.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static std::uint64_t value(engine& eng, engine::Reg reg) {
    mpz_t z;
    mpz_init(z);
    eng.get_mpz(z, reg);
    const std::uint64_t out = mpz_get_ui(z);
    mpz_clear(z);
    return out;
}

static void expect(std::uint64_t actual, std::uint64_t expected, const char* label) {
    if (actual != expected) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
                                 ", expected " + std::to_string(expected));
    }
}

int main() {
    constexpr std::uint32_t exponent = 31;
    constexpr std::uint64_t modulus = (std::uint64_t(1) << exponent) - 1;

    engine::configure_gpu_backend(engine::gpu_backend::aevum, "test-shape");
    if (engine::configured_gpu_backend() != engine::gpu_backend::aevum) return 2;
    if (engine::configured_aevum_fft_spec() != "test-shape") return 3;

    std::unique_ptr<engine> eng(engine::create_gpu(exponent, 8, 0, false));
    eng->set(0, 5);
    eng->set(1, 7);
    eng->set_multiplicand(2, 1);
    eng->mul(0, 2);
    expect(value(*eng, 0), 35, "mul");

    eng->square_mul(0, 3);
    expect(value(*eng, 0), 3675, "square_mul");

    eng->add(0, 1);
    eng->sub_reg(0, 1);
    eng->sub(0, 2);
    expect(value(*eng, 0), 3673, "add/sub");

    mpz_t z;
    mpz_init(z);
    mpz_set_ui(z, modulus + 123);
    eng->set_mpz(3, z);
    mpz_clear(z);
    expect(value(*eng, 3), 123, "set_mpz");

    std::vector<char> one(eng->get_register_data_size());
    if (!eng->get_data(one, 3)) return 4;
    eng->set(4, std::uint32_t(0));
    if (!eng->set_data(4, one)) return 5;
    expect(value(*eng, 4), 123, "register data");
    if (!eng->is_equal(3, 4)) return 10;
    eng->set(4, 124);
    if (eng->is_equal(3, 4)) return 11;
    eng->set(4, 123);

    std::vector<char> checkpoint(eng->get_checkpoint_size());
    if (!eng->get_checkpoint(checkpoint)) return 6;
    eng->set(0, 1);
    eng->set(3, 1);
    if (!eng->set_checkpoint(checkpoint)) return 7;
    expect(value(*eng, 0), 3673, "checkpoint r0");
    expect(value(*eng, 3), 123, "checkpoint r3");

    eng->set(5, 9);
    eng->pow(6, 5, 13);
    std::uint64_t expected = 1;
    for (int i = 0; i < 13; ++i) expected = (expected * 9) % modulus;
    expect(value(*eng, 6), expected, "pow");

    engine::digit digits(eng.get(), 3);
    if (digits.get_size() != 8 || digits.res64() != 123) return 8;

    eng->sync();
    engine::configure_gpu_backend(engine::gpu_backend::marin);
    if (engine::configured_gpu_backend() != engine::gpu_backend::marin) return 9;

    std::cout << "Aevum engine::Reg adapter test passed" << std::endl;
    return 0;
}
