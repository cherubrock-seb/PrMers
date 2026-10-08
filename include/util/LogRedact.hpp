#pragma once
#include <cctype>
#include <mutex>
#include <streambuf>
#include <string>

namespace util {

// The web GUI prints its URL "http://host:port/?token=<token>" so the user can open it. The token gates
// every /api call, so it must not be persisted: replace it in anything written to a log file.
inline std::string redactGuiToken(std::string text) {
    static const std::string key = "token=";
    size_t pos = 0;
    while ((pos = text.find(key, pos)) != std::string::npos) {
        const size_t start = pos + key.size();
        size_t end = start;
        while (end < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '-' || text[end] == '_'))
            ++end;
        if (end > start) {
            text.replace(start, end - start, "********");
            pos = start + 8;
        } else {
            pos = start;
        }
    }
    return text;
}

// A streambuf that forwards to `sink` with the GUI token redacted. Text is passed on line by line (and at
// sync(), except for a trailing "token=<chars>" that may still be growing), so a token is never split.
// Several threads write to std::cout (the progress spinner and the main loop): every entry point takes the
// lock, because they all modify the same pending_ buffer.
class TokenRedactingBuf : public std::streambuf {
public:
    explicit TokenRedactingBuf(std::streambuf* sink) : sink_(sink) {}
    ~TokenRedactingBuf() override { std::lock_guard<std::mutex> lock(mutex_); flushAll(); }

protected:
    int overflow(int ch) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ch == traits_type::eof()) return traits_type::not_eof(ch);
        pending_.push_back(static_cast<char>(ch));
        if (ch == '\n') flushUpTo(pending_.size());
        return ch;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.append(s, static_cast<size_t>(n));
        const size_t nl = pending_.rfind('\n');
        if (nl != std::string::npos) flushUpTo(nl + 1);
        return n;
    }
    int sync() override {
        std::lock_guard<std::mutex> lock(mutex_);
        flushUpTo(safePrefix());
        return sink_ ? sink_->pubsync() : 0;
    }

private:
    // Length of the part of pending_ that cannot end inside a token.
    size_t safePrefix() const {
        const size_t k = pending_.rfind("token=");
        if (k == std::string::npos) return pending_.size();
        for (size_t i = k + 6; i < pending_.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(pending_[i]);
            if (!(std::isalnum(c) || c == '-' || c == '_')) return pending_.size();
        }
        return k;
    }
    void flushUpTo(size_t n) {
        if (n == 0 || !sink_) { if (!sink_) pending_.erase(0, n); return; }
        const std::string out = redactGuiToken(pending_.substr(0, n));
        sink_->sputn(out.data(), static_cast<std::streamsize>(out.size()));
        pending_.erase(0, n);
    }
    void flushAll() {
        flushUpTo(pending_.size());
        if (sink_) sink_->pubsync();
    }

    std::mutex mutex_;
    std::streambuf* sink_;
    std::string pending_;
};

} // namespace util
