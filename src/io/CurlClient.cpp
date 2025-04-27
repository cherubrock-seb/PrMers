/*
 * Mersenne OpenCL Primality Test Host Code
 *
 * This code is inspired by:
 *   - "mersenne.cpp" by Yves Gallot (Copyright 2020, Yves Gallot) based on
 *     Nick Craig-Wood's IOCCC 2012 entry (https://github.com/ncw/ioccc2012).
 *   - The Armprime project, explained at:
 *         https://www.craig-wood.com/nick/armprime/
 *     and available on GitHub at:
 *         https://github.com/ncw/
 *   - Yves Gallot (https://github.com/galloty), author of Genefer 
 *     (https://github.com/galloty/genefer22), who helped clarify the NTT and IDBWT concepts.
 *   - The GPUOwl project (https://github.com/preda/gpuowl), which performs Mersenne
 *     searches using FFT and double-precision arithmetic.
 * This code performs a Mersenne prime search using integer arithmetic and an IDBWT via an NTT,
 * executed on the GPU through OpenCL.
 *
 * Author: Cherubrock
 *
 * This code is released as free software. 
 */
#include "io/CurlClient.hpp"
#ifndef NO_CURL
#include <curl/curl.h>
#endif
#include <iostream>
#include <sstream>

namespace io {

#ifndef NO_CURL

size_t CurlClient::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

std::string CurlClient::extractUID(const std::string& html) {
    const std::string key = "name=\"was_logged_in_as\" value=\"";
    auto pos = html.find(key);
    if (pos == std::string::npos) return "";
    pos += key.size();
    auto end = html.find('"', pos);
    return html.substr(pos, end - pos);
}

bool CurlClient::login(const std::string& user, const std::string& password, std::string& uid) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.mersenne.org/?page=login");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    std::ostringstream post;
    post << "user=" << curl_easy_escape(curl, user.c_str(), 0)
         << "&password=" << curl_easy_escape(curl, password.c_str(), 0);
    std::string postStr = post.str();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postStr.c_str());

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) return false;

    uid = extractUID(response);
    return !uid.empty();
}

bool CurlClient::submit(const std::string& json, const std::string& user, const std::string& password) {
    std::string uid;
    if (!login(user, password, uid)) {
        std::cerr << "Login failed\n";
        return false;
    }

    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.mersenne.org/results/new");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    std::ostringstream post;
    post << "was_logged_in_as=" << curl_easy_escape(curl, uid.c_str(), 0)
         << "&json="          << curl_easy_escape(curl, json.c_str(), 0);
    std::string postStr = post.str();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postStr.c_str());

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) {
        std::cerr << "Submit failed: " << curl_easy_strerror(res) << "\n";
        return false;
    }
    return true;
}

#else // NO_CURL defined

bool CurlClient::login(const std::string& user, const std::string& password, std::string& uid) {
    std::cerr << "CURL support is disabled. Cannot login.\n";
    return false;
}

bool CurlClient::submit(const std::string& json, const std::string& user, const std::string& password) {
    std::cerr << "CURL support is disabled. Cannot submit results.\n";
    return false;
}

#endif // NO_CURL

} // namespace io
