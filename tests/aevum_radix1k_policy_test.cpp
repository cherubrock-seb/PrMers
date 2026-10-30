// The backend log must report the 1K radix the Aevum plugin will use, not just the AEVUM_RADIX1K variable.
#include "aevum/Radix1kPolicy.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

static int failures = 0;

static void expect(const char* env, bool apple, bool nvidia, const std::string& want) {
    const std::string got = aevum_policy::radix1k_detail(aevum_policy::radix1k_policy(env, apple, nvidia));
    if (got != want) {
        std::cerr << "FAIL env=" << (env ? env : "(null)") << " apple=" << apple << " nvidia=" << nvidia
                  << ": got '" << got << "', want '" << want << "'" << std::endl;
        ++failures;
    }
}

int main() {
    // No override: radix-8 on NVIDIA only (it used to be reported as 4 safe-default there).
    expect(nullptr, false, true, "radix1k=8 nvidia-default");
    expect("", false, true, "radix1k=8 nvidia-default");
    expect(nullptr, false, false, "radix1k=4 safe-default");
    // Explicit 4/8 wins over the device default.
    expect("8", false, false, "radix1k=8 explicit-override");
    expect("8", false, true, "radix1k=8 explicit-override");
    expect("4", false, true, "radix1k=4 explicit-override");
    expect("4", false, false, "radix1k=4 explicit-override");
    // Anything but exactly "8" is radix-4 (the plugin rejects such values at creation).
    expect("88", false, true, "radix1k=4 explicit-override");
    expect("8x", false, true, "radix1k=4 explicit-override");
    // Apple forces radix-4, whatever the variable says (it used to be reported as 8 explicit-override).
    expect("8", true, false, "radix1k=4 apple-forced");
    expect("8", true, true, "radix1k=4 apple-forced");
    expect(nullptr, true, false, "radix1k=4 apple-forced");

    if (failures) return 1;
    std::cout << "Aevum radix1k policy test: PASS" << std::endl;
    return 0;
}
