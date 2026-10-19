// The result JSON must stay valid JSON whatever -user, -computer and the worktodo AID hold.  jsonEscape escaped
// only " \ \b \f \n \r \t, so any other control character (U+0001 to U+001F) went out raw, which RFC 8259
// forbids: a parser rejects the whole results.txt line.  Each must now be written as \u00XX, while UTF-8
// (bytes from 0x80 up) passes through unchanged.  The JSON is printed for tests/test_json_escape_control.sh,
// which parses it with Python's json module.
#include "io/CliParser.hpp"
#include "io/JsonBuilder.hpp"

#include <iostream>
#include <string>

namespace {
int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}
}  // namespace

int main() {
    io::CliOptions opts;
    opts.exponent = 127;
    opts.mode = "prp";
    // Every control character, the two that need a backslash, and UTF-8 (e with acute accent).
    std::string user;
    for (int c = 0x01; c < 0x20; ++c) user += static_cast<char>(c);
    user += "\"\\ caf\xC3\xA9";
    opts.user = user;
    opts.computer_name = std::string("box\x7f\x1b[31m");
    opts.aid = std::string("0123456789ABCDEF0123456789ABCDEF\x01");

    const std::string json = io::JsonBuilder::generate(opts, 8, true, "0000000000000001", std::string(512, '0'));

    for (unsigned char c : json) expect(c >= 0x20, "raw control character " + std::to_string(c) + " in the JSON");
    expect(json.find("\\u0001\\u0002") != std::string::npos, "\\u0001 \\u0002 not escaped");
    expect(json.find("\\u001f") != std::string::npos, "\\u001f not escaped");
    expect(json.find("\\b\\t\\n\\u000b\\f\\r") != std::string::npos, "short escapes \\b \\t \\n \\f \\r changed");
    expect(json.find("\\\"\\\\ caf\xC3\xA9") != std::string::npos, "quote, backslash or UTF-8 not kept");
    expect(json.find("box\x7f\\u001b[31m") != std::string::npos, "computer name not escaped as expected");

    std::cout << json << "\n";
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    return 0;
}
