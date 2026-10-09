// The GUI access token must not leak to child processes. It used to be put into PRMERS_GUI_TOKEN for the
// whole life of the process, so Prime95 and every shell command inherited it. Now the environment only
// holds it between WebGuiServer::exportTokenForRestart() and the relaunch.
//
// Build/run: bash tests/test_gui_token_env.sh   (POSIX; uses /bin/sh as the "child")
#include "ui/WebGuiServer.hpp"
#include "util/SelfExe.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

static int fails = 0;
static void expect(const bool ok, const std::string& what)
{
	if (!ok) { ++fails; std::printf("FAIL %s\n", what.c_str()); }
}

// What a child process started now sees in PRMERS_GUI_TOKEN ("<unset>" when absent).
static std::string childSees()
{
	FILE* p = popen("printf '%s' \"${PRMERS_GUI_TOKEN-<unset>}\"", "r");
	std::string out;
	char buf[256];
	size_t n;
	while (p && (n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
	if (p) pclose(p);
	return out;
}

static std::string tokenOf(ui::WebGuiServer& s)
{
	const std::string url = s.url();
	const auto t = url.find("token=");
	return t == std::string::npos ? std::string() : url.substr(t + 6);
}

static ui::WebGuiConfig config()
{
	ui::WebGuiConfig cfg;
	cfg.port = 0;
	cfg.bind_host = "127.0.0.1";
	return cfg;
}

int main(int argc, char** argv)
{
	// Re-executed by the "restart" case below: report what the relaunched program inherited.
	if (argc > 1 && std::strcmp(argv[1], "--child") == 0)
	{
		const char* v = std::getenv("PRMERS_GUI_TOKEN");
		ui::WebGuiServer s(config(), [](const std::string&) {});
		s.start();
		std::printf("CHILD env=%s token=%s\n", v ? v : "<unset>", tokenOf(s).c_str());
		s.stop();
		return 0;
	}

	const std::string valid = "0123456789abcdef0123456789abcdef";

	// 1. A fresh start (no inherited token): a token exists, but the environment stays clean.
	unsetenv("PRMERS_GUI_TOKEN");
	{
		ui::WebGuiServer s(config(), [](const std::string&) {});
		s.start();
		const std::string tok = tokenOf(s);
		expect(tok.size() >= 16, "fresh start has a token");
		expect(std::getenv("PRMERS_GUI_TOKEN") == nullptr, "fresh start does not export the token");
		expect(childSees() == "<unset>", "a child started by a fresh run does not see the token");
		s.stop();
	}

	// 2. A relaunch: the inherited token is reused, then removed from the environment.
	setenv("PRMERS_GUI_TOKEN", valid.c_str(), 1);
	{
		ui::WebGuiServer s(config(), [](const std::string&) {});
		s.start();
		expect(tokenOf(s) == valid, "the inherited token is reused");
		expect(std::getenv("PRMERS_GUI_TOKEN") == nullptr, "the inherited token is removed from the environment");
		expect(childSees() == "<unset>", "a child (Prime95, shell command) of a relaunched run does not see the token");

		// 3. restart_self exports it just before the exec and clears it if the exec fails.
		s.exportTokenForRestart();
		expect(childSees() == valid, "a relaunch gets the token");
		ui::WebGuiServer::clearTokenEnv();
		expect(childSees() == "<unset>", "clearTokenEnv removes it again");
		expect(std::getenv("PRMERS_GUI_TOKEN") == nullptr, "clearTokenEnv leaves nothing behind");
		ui::WebGuiServer::clearTokenEnv();   // idempotent
		s.exportTokenForRestart();
		s.exportTokenForRestart();           // idempotent
		expect(childSees() == valid, "exporting twice is harmless");
		ui::WebGuiServer::clearTokenEnv();
		s.stop();
	}

	// 4. Garbage inherited values are not trusted and do not stay in the environment either.
	const std::vector<std::string> bad = {
		"", "short", "has space in it 0123456789abcdef", "semi;colon0123456789abcdef", std::string(129, 'a'),
		"quote\"0123456789abcdef0123", "new\nline0123456789abcdef0123",
	};
	for (const auto& b : bad)
	{
		setenv("PRMERS_GUI_TOKEN", b.c_str(), 1);
		ui::WebGuiServer s(config(), [](const std::string&) {});
		s.start();
		expect(tokenOf(s) != b && tokenOf(s).size() >= 16, "an invalid inherited token is replaced: '" + b + "'");
		expect(std::getenv("PRMERS_GUI_TOKEN") == nullptr, "an invalid inherited token is not left in the environment: '" + b + "'");
		s.stop();
	}

	// 5. A real exec: the relaunched program (this binary with --child) gets the token the old one had.
	{
		setenv("PRMERS_GUI_TOKEN", valid.c_str(), 1);
		ui::WebGuiServer s(config(), [](const std::string&) {});
		s.exportTokenForRestart();
		std::fflush(stdout);
		util::execSelf({ argv[0], "--child" });
		std::perror("FAIL execSelf");
		return 1;
	}
}
