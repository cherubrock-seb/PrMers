#ifndef IO_CURLCLIENT_HPP
#define IO_CURLCLIENT_HPP

#include <string>

namespace io {

class CurlClient {
public:
    static std::string promptHiddenPassword();

     static bool sendManualResultWithLogin(const std::string& jsonResult,
                                          const std::string& username,
                                          const std::string& password);

private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    static std::string extractUID(const std::string& html);
};

} // namespace io

#endif // IO_CURLCLIENT_HPP
