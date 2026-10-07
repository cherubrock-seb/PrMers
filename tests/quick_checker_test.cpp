// QuickChecker must only answer plain command-line Mersenne tests from its table.
// Wagstaff (exponent already doubled by the CLI parser), cofactor and worktodo
// runs have to go through the real test and the normal result/removal path.
#include "core/QuickChecker.hpp"

#include <iostream>

static int failures = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

int main() {
    using core::QuickChecker;

    // Plain Mersenne numbers: table answers.
    {
        io::CliOptions o;
        o.exponent = 61;
        auto r = QuickChecker::run(o, false);
        expect(r && *r == 0, "M61 is prime");
        o.exponent = 67;
        r = QuickChecker::run(o, false);
        expect(r && *r == 1, "M67 is composite");
        o.exponent = 127;
        expect(!QuickChecker::run(o, false), "p >= 127 is not shortcut");
    }

    // Wagstaff: options.exponent is 2p, so (2^31+1)/3 arrives as 62.
    {
        io::CliOptions o;
        o.wagstaff = true;
        o.exponent = 62;
        expect(!QuickChecker::run(o, false), "wagstaff p=31 is not shortcut");
        o.exponent = 122;
        expect(!QuickChecker::run(o, false), "wagstaff p=61 is not shortcut");
    }

    // Cofactor: the table describes 2^p-1, not 2^p-1 divided by known factors.
    {
        io::CliOptions o;
        o.exponent = 67;
        o.knownFactors = {"193707721"};
        expect(!QuickChecker::run(o, false), "cofactor is not shortcut");
    }

    // Worktodo entries must be written to results and removed from the queue.
    {
        io::CliOptions o;
        o.exponent = 61;
        expect(!QuickChecker::run(o, true), "worktodo entry is not shortcut");
    }

    if (failures == 0) std::cout << "quick checker test passed\n";
    return failures == 0 ? 0 : 1;
}
