// Host test: the rule that keeps Lucas-Lehmer off the legacy internal NTT path (-marin).
// The same function is used for the command line (main.cpp) and for a mode that came from a worktodo
// LL entry (App.cpp).
#include "core/LegacyLlGuard.hpp"

#include <cstdio>
#include <string>

static int fails = 0;

static void expect(const bool ok, const std::string& what)
{
	if (!ok) { ++fails; std::printf("FAIL %s\n", what.c_str()); }
}

int main()
{
	// marin == false is what -marin sets: the legacy path.
	expect(!core::legacyLlRejection("ll", false).empty(), "ll on the legacy path is rejected");
	expect(core::legacyLlRejection("ll", false).find("not validated for Lucas-Lehmer") != std::string::npos,
	       "rejection text names the reason");
	expect(core::legacyLlRejection("ll", true).empty(), "ll on an engine backend is allowed");
	for (const char* mode : { "prp", "pm1", "ecm", "llsafe", "llsafe2", "gm-prp", "", "LL", "ll " })
	{
		expect(core::legacyLlRejection(mode, false).empty(), std::string("mode '") + mode + "' is not blocked on the legacy path");
		expect(core::legacyLlRejection(mode, true).empty(), std::string("mode '") + mode + "' is not blocked on an engine backend");
	}
	// -allow-unvalidated-legacy-ll: the same case is let through and warns; nothing else warns.
	expect(core::legacyLlRejection("ll", false, true).empty(), "ll on the legacy path is allowed with the opt-in");
	expect(core::legacyLlRejection("ll", false, false) == core::legacyLlRejection("ll", false),
	       "the opt-in is off by default");
	expect(core::legacyLlWarning("ll", false, true).find("-allow-unvalidated-legacy-ll") != std::string::npos,
	       "the opt-in warning names the flag");
	expect(core::legacyLlWarning("ll", false, true).find("not validated") != std::string::npos,
	       "the opt-in warning says the path is not validated");
	expect(core::legacyLlWarning("ll", false, false).empty(), "no warning without the opt-in");
	expect(core::legacyLlWarning("ll", true, true).empty(), "no warning on an engine backend");
	for (const char* mode : { "prp", "pm1", "ecm", "llsafe", "llsafe2", "gm-prp", "", "LL", "ll " })
	{
		expect(core::legacyLlRejection(mode, false, true).empty(), std::string("mode '") + mode + "' is not blocked with the opt-in");
		expect(core::legacyLlWarning(mode, false, true).empty(), std::string("mode '") + mode + "' does not warn with the opt-in");
	}
	std::printf("Legacy LL guard test: %s\n", fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
