// Host test for the Gaussian trial-factoring command-line pre-parser.
//
// tryRunGaussianTrialFactor() runs before the regular option parser, so it sees
// every command line. It must leave unrelated command lines alone and, when
// -gm-tf is present, must not mistake the value of another option for the
// exponent. Every case below stops in request validation (even or oversized
// exponent) or returns before any OpenCL work, so no device is touched.

#include "modes/GaussianTrialFactor.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

struct Outcome {
    bool threw = false;
    std::string message;
    std::optional<int> result;
};

Outcome run(std::vector<std::string> args) {
    args.insert(args.begin(), "prmers");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    Outcome out;
    try {
        out.result = core::tryRunGaussianTrialFactor(static_cast<int>(argv.size()), argv.data());
    } catch (const std::exception& e) {
        out.threw = true;
        out.message = e.what();
    }
    return out;
}

void expectDeclined(const char* name, std::vector<std::string> args) {
    const Outcome o = run(std::move(args));
    if (o.threw || o.result) {
        std::cerr << "FAIL " << name << ": expected the pre-parser to decline, "
                  << (o.threw ? "threw: " + o.message : "returned a result") << "\n";
        ++failures;
    } else {
        std::cout << "PASS " << name << "\n";
    }
}

void expectError(const char* name, std::vector<std::string> args, const std::string& needle) {
    const Outcome o = run(std::move(args));
    if (!o.threw || o.message.find(needle) == std::string::npos) {
        std::cerr << "FAIL " << name << ": expected error containing \"" << needle << "\", got "
                  << (o.threw ? "\"" + o.message + "\"" : "no error") << "\n";
        ++failures;
    } else {
        std::cout << "PASS " << name << " (" << o.message << ")\n";
    }
}

} // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "prmers_gm_tf_preparser_test";
    std::filesystem::create_directories(dir);
    const std::string none = (dir / "missing_worktodo.txt").string();
    const std::string gmtf = (dir / "worktodo.txt").string();
    {
        std::ofstream out(gmtf);
        out << "GMTF=4,20,30\n";  // even p: rejected by validation before any GPU work
    }

    // Command lines without -gm-tf are not this mode's business, whatever the
    // option values look like.
    expectDeclined("user_name_before_worktodo", {"-user", "mark", "-worktodo", none});
    expectDeclined("computer_name_gui", {"-computer", "box1", "-gui"});
    expectDeclined("config_style_user", {"-user", "bob", "-worktodo", none});
    expectDeclined("path_values", {"-f", "out dir", "-kernelpath", "/opt/k", "-worktodo", none});
    expectDeclined("bad_family_without_tf", {"-gm-family", "XYZ", "-worktodo", none});

    // An option value must not be taken for the exponent when -gm-tf is given.
    expectError("user_before_tf", {"-user", "bob", "4", "-gm-tf", "20", "30"}, "odd exponent");
    expectError("user_after_tf", {"-gm-tf", "20", "30", "4", "-user", "bob"}, "odd exponent");
    expectError("computer_value_skipped", {"-computer", "box1", "-gm-tf", "20", "30", "4"},
                "odd exponent");
    // 4294967295 is odd but 4p overflows; the numeric value of -t (4) must not win.
    expectError("numeric_option_value_skipped",
                {"-t", "4", "4294967295", "-gm-tf", "20", "30"}, "4p");
    expectError("password_value_skipped", {"-password", "hunter2", "4", "-gm-tf", "20", "30"},
                "odd exponent");

    // Genuinely bad input is still reported.
    expectError("bad_exponent", {"-gm-tf", "20", "30", "12a7"}, "Invalid Gaussian exponent");
    expectError("missing_exponent", {"-user", "bob", "-gm-tf", "20", "30"},
                "requires the exponent");
    expectError("bad_family_with_tf", {"-gm-tf", "20", "30", "5", "-gm-family", "XYZ"}, "family");

    // A GMTF line in the worktodo is found even when an option value precedes it.
    expectError("worktodo_with_user", {"-user", "mark", "-worktodo", gmtf}, "odd exponent");
    expectError("worktodo_with_computer_gui",
                {"-computer", "box1", "-worktodo", gmtf}, "odd exponent");

    // The optional -pfa / -pfa-auto radix value (3, 7 or 9) must not stop the
    // worktodo GMTF entry from being reached or be taken for the exponent.
    for (const char* radix : {"3", "7", "9"}) {
        for (const char* flag : {"-pfa", "-pfa-auto"}) {
            expectError((std::string("worktodo_with_") + flag + "_" + radix).c_str(),
                        {flag, radix, "-worktodo", gmtf}, "odd exponent");
            expectError((std::string("tf_exponent_after_") + flag + "_" + radix).c_str(),
                        {flag, radix, "4", "-gm-tf", "20", "30"}, "odd exponent");
        }
    }
    // Without a radix value the flag stands alone, and a non-radix number after it is the exponent.
    expectError("worktodo_with_bare_pfa", {"-pfa", "-worktodo", gmtf}, "odd exponent");
    expectError("pfa_without_radix_then_exponent", {"-pfa", "4", "-gm-tf", "20", "30"},
                "odd exponent");

    // Hostile values. An option value that spells a mode flag is still a value, so it does not
    // stop the worktodo entry; a bare exponent on the command line does not either.
    expectError("mode_spelled_value_skipped", {"-user", "-gm", "-worktodo", gmtf}, "odd exponent");
    expectError("bare_exponent_with_worktodo", {"4", "-worktodo", gmtf}, "odd exponent");
    expectDeclined("value_option_at_end", {"-worktodo", none, "-user"});
    expectDeclined("garbage_device_without_tf", {"-d", "abc", "-worktodo", none});
    expectError("garbage_device_with_worktodo", {"-d", "abc", "-worktodo", gmtf}, "Invalid device");
    expectError("garbage_device_with_tf", {"-d", "abc", "-gm-tf", "20", "30", "5"}, "Invalid device");
    expectError("exponent_overflow", {"-gm-tf", "20", "30", "18446744073709551616"},
                "Invalid Gaussian exponent");
    expectError("exponent_leading_zeros", {"-gm-tf", "20", "30", "0004"}, "odd exponent");
    expectError("exponent_hex", {"-gm-tf", "20", "30", "0x5"}, "Invalid Gaussian exponent");
    expectError("tf_bits_missing", {"-user", "bob", "-gm-tf", "20"}, "requires FROM_BITS");

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (failures != 0) {
        std::cerr << failures << " Gaussian TF pre-parser check(s) failed\n";
        return 1;
    }
    std::cout << "Gaussian TF pre-parser test passed\n";
    return 0;
}
