// The GUI access token is printed (terminal) in the URL "http://host:port/?token=<token>" and must not
// reach prmers.log, which is created world-readable by default and appended forever.
#include "util/LogRedact.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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

    // Threads writing at once (std::cout is written by the progress spinner and the main loop alike):
    // no line may be lost, torn or crash the buffer. Each line is one operator<< call plus a flush.
    {
        std::ostringstream file;
        constexpr int kThreads = 4, kLines = 3000;
        {
            util::TokenRedactingBuf buf(file.rdbuf());
            std::vector<std::thread> threads;
            for (int t = 0; t < kThreads; ++t)
                threads.emplace_back([&buf, t] {
                    std::ostream os(&buf);
                    for (int i = 0; i < kLines; ++i) {
                        os << "thread " << t << " line " << i << " end" << std::endl;
                        os << "\rProgress " << t << " " << i << "%" << std::flush;
                    }
                });
            for (auto& th : threads) th.join();
        }
        // Fragments of different lines may interleave (that is what unsynchronised stdout does too), but
        // nothing may be lost: the byte and line counts are exact.
        size_t want = 0;
        for (int t = 0; t < kThreads; ++t)
            for (int i = 0; i < kLines; ++i)
                want += std::string("thread " + std::to_string(t) + " line " + std::to_string(i) + " end\n").size() +
                        std::string("\rProgress " + std::to_string(t) + " " + std::to_string(i) + "%").size();
        const std::string out = file.str();
        size_t newlines = 0;
        for (char c : out) newlines += (c == '\n');
        if (newlines != static_cast<size_t>(kThreads) * kLines || out.size() != want) {
            std::cerr << "FAIL concurrent writers: " << newlines << " lines / " << out.size()
                      << " bytes, expected " << static_cast<size_t>(kThreads) * kLines << " / " << want << "\n";
            ++failures;
        }
    }

    if (failures != 0) {
        std::cerr << failures << " log redaction check(s) failed\n";
        return 1;
    }
    std::cout << "Log redaction test passed\n";
    return 0;
}
