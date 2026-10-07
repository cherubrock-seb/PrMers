// The GUI access token is printed (terminal) in the URL "http://host:port/?token=<token>" and must not
// reach prmers.log, which is created world-readable by default and appended forever.
#include "util/LogRedact.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void expectEq(const std::string& got, const std::string& want, const std::string& what) {
    if (got != want) {
        std::cerr << "FAIL " << what << ": got [" << got << "], expected [" << want << "]\n";
        ++failures;
    }
}

} // namespace

int main() {
    const std::string tok = "0123456789abcdef0123456789abcdef";
    const std::string url = "http://127.0.0.1:3131/?token=" + tok;

    expectEq(util::redactGuiToken("GUI " + url), "GUI http://127.0.0.1:3131/?token=********", "plain URL");
    expectEq(util::redactGuiToken("a token=" + tok + " b token=x_y-Z c"), "a token=******** b token=******** c", "two tokens");
    expectEq(util::redactGuiToken("no secret here, token= alone"), "no secret here, token= alone", "empty token value");
    expectEq(util::redactGuiToken("Progress: 10%"), "Progress: 10%", "unrelated text");

    // The same through a streambuf, written the way std::cout writes the GUI line.
    {
        std::ostringstream file;
        {
            util::TokenRedactingBuf buf(file.rdbuf());
            std::ostream os(&buf);
            os.setf(std::ios::unitbuf);
            os << "GUI " << url << std::endl;
            os << "line two\n";
        }
        expectEq(file.str(), "GUI http://127.0.0.1:3131/?token=********\nline two\n", "streambuf, unit-buffered");
    }
    // A flush in the middle of the token must not let the token through.
    {
        std::ostringstream file;
        {
            util::TokenRedactingBuf buf(file.rdbuf());
            std::ostream os(&buf);
            os << "GUI http://127.0.0.1:3131/?token=0123456789ab" << std::flush;
            os << "cdef0123456789abcdef" << std::endl;
        }
        expectEq(file.str(), "GUI http://127.0.0.1:3131/?token=********\n", "flush inside the token");
    }
    // A partial line without a token is passed on at flush time (progress lines end in \r, not \n).
    {
        std::ostringstream file;
        util::TokenRedactingBuf buf(file.rdbuf());
        std::ostream os(&buf);
        os << "Progress: 50%\r" << std::flush;
        expectEq(file.str(), "Progress: 50%\r", "partial line flushed");
    }

    if (failures != 0) {
        std::cerr << failures << " log redaction check(s) failed\n";
        return 1;
    }
    std::cout << "Log redaction test passed\n";
    return 0;
}
