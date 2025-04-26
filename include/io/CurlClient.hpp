// include/io/CurlClient.hpp
#pragma once

#include <string>

namespace io {

class CurlClient {
public:
    // Logs in using user/password, then POSTs `json` to PrimeNet.
    // Returns true on HTTP-level success.
    static bool submit(const std::string& json, const std::string& user, const std::string& password);

private:
    static bool login(const std::string& user, const std::string& password, std::string& uid);
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    static std::string extractUID(const std::string& html);
};

} // namespace io
