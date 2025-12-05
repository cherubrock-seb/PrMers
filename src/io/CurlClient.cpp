#include "io/CurlClient.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif
#if defined(HAS_CURL) && HAS_CURL
#include <curl/curl.h>
#endif

namespace io {

size_t CurlClient::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

std::string CurlClient::extractUID(const std::string& html)
{
    const std::string key = "name=\"was_logged_in_as\" value=\"";
    auto pos = html.find(key);
    if (pos == std::string::npos) return "";
    pos += key.size();
    auto end = html.find('"', pos);
    return html.substr(pos, end - pos);
}

std::string CurlClient::promptHiddenPassword() {
    std::string pwd;
#if defined(_WIN32)
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode; GetConsoleMode(hStdin, &mode);
    SetConsoleMode(hStdin, mode & ~(ENABLE_ECHO_INPUT));
    std::cout << "Enter your PrimeNet password: ";
    std::getline(std::cin, pwd);
    SetConsoleMode(hStdin, mode);
    std::cout << std::endl;
#else
    std::cout << "Enter your PrimeNet password: ";
    if (system("stty -echo") != 0) {
        std::cerr << "Warning: failed to disable echo\n";
    }
    std::getline(std::cin, pwd);
    if (system("stty echo") != 0) {
        std::cerr << "Warning: failed to restore echo\n";
    }
    std::cout << std::endl;
#endif
    return pwd;
}

bool CurlClient::sendManualResultWithLogin(const std::string& jsonResult,
                                           const std::string& username,
                                           const std::string& password)
{
#if defined(HAS_CURL) && HAS_CURL
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "❌ Failed to initialize CURL.\n";
        return false;
    }

    FILE* trace = nullptr;
    #ifdef _WIN32
        fopen_s(&trace, "curl_trace.txt", "w");
    #else
        trace = fopen("curl_trace.txt", "w");
    #endif

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8");
    headers = curl_slist_append(headers, "Accept-Language: fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7");
    headers = curl_slist_append(headers, "Origin: https://www.mersenne.org");
    headers = curl_slist_append(headers, "Referer: https://www.mersenne.org/login.php");
    headers = curl_slist_append(headers, "Upgrade-Insecure-Requests: 1");
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_STDERR, trace);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_COOKIEFILE, "");
    curl_easy_setopt(curl, CURLOPT_COOKIEJAR, "cookies.txt");
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");

    std::string loginResponse;
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.mersenne.org/login.php");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &loginResponse);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    std::ostringstream loginData;
    loginData << "user_login=" << username
              << "&user_password=" << password;
    std::string postFields = loginData.str();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postFields.c_str());

    std::cerr << "[TRACE] Sending login with user: " << username << std::endl;
    CURLcode loginRes = curl_easy_perform(curl);

    std::ofstream htmlOut("login_response_debug.html");
    htmlOut << loginResponse;
    htmlOut.close();

    if (loginRes != CURLE_OK) {
        std::cerr << "❌ Login failed: " << curl_easy_strerror(loginRes) << "\n";
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        if (trace) fclose(trace);
        return false;
    }

    struct curl_slist* cookies = nullptr;
    curl_easy_getinfo(curl, CURLINFO_COOKIELIST, &cookies);
    bool hasSession = false;
    for (auto c = cookies; c; c = c->next) {
        if (strstr(c->data, "GIMPSWWW")) {
            hasSession = true;
            break;
        }
    }
    curl_slist_free_all(cookies);

    if (!hasSession) {
        std::cerr << "❌ Login failed: no session cookie received.\n";
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        if (trace) fclose(trace);
        return false;
    }

    std::cerr << "✅ Login successful, session cookie received.\n";

    std::string htmlFormPage;
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.mersenne.org/manual_result/");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, nullptr);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &htmlFormPage);

    CURLcode pageRes = curl_easy_perform(curl);
    if (pageRes != CURLE_OK) {
        std::cerr << "❌ Failed to get manual_result page: " << curl_easy_strerror(pageRes) << "\n";
        curl_easy_cleanup(curl);
        if (trace) fclose(trace);
        return false;
    }

    std::string uid = extractUID(htmlFormPage);
    std::cerr << "[TRACE] Extracted was_logged_in_as UID: " << uid << "\n";
    if (uid.empty()) {
        std::ofstream htmlOut2("login_response_manual_debug.html");
        htmlOut2 << htmlFormPage;
        htmlOut2.close();
        std::cerr << "💡 Login HTML saved to login_response_manual_debug.html\n";
        std::cerr << "❌ Could not find was_logged_in_as value in form page.\n";
        curl_easy_cleanup(curl);
        if (trace) fclose(trace);
        return false;
    }

    curl_mime* form = curl_mime_init(curl);

    curl_mimepart* field = curl_mime_addpart(form);
    curl_mime_name(field, "data_file");
    curl_mime_data(field, "", CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "was_logged_in_as");
    curl_mime_data(field, uid.c_str(), CURL_ZERO_TERMINATED);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "data");
    curl_mime_data(field, jsonResult.c_str(), CURL_ZERO_TERMINATED);

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.mersenne.org/manual_result/");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, nullptr);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    std::cerr << "[TRACE] Sending manual result with was_logged_in_as = " << uid << std::endl;

    CURLcode res = curl_easy_perform(curl);
    if (trace) fflush(trace);
    if (trace) fclose(trace);
    curl_mime_free(form);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "❌ Failed to send result: " << curl_easy_strerror(res) << "\n";
        return false;
    }

    std::cerr << "[TRACE] Server response size: " << response.size() << " bytes\n";
    std::cout << "✅ Server response:...\n";
    std::string startTag = "<h2>Manually check in your results</h2>";
    std::string endTagA = "<a href=\"/manual_result/\">Submit more results</a>";
    std::string endTagB = "Aborting processing.</div>";

    auto startPos = response.find(startTag);
    size_t endPos = std::string::npos;
    std::string chosenEndTag;
    bool usedFallback = false;

    if (startPos != std::string::npos) {
        endPos = response.find(endTagA, startPos);
        chosenEndTag = endTagA;

        if (endPos == std::string::npos) {
            endPos = response.find(endTagB, startPos);
            chosenEndTag = endTagB;
        }

        if (endPos == std::string::npos && response.size() > startPos + 1000) {
            endPos = startPos + 1000;
            usedFallback = true;
        }

        if (endPos != std::string::npos) {
            std::string htmlChunk = response.substr(startPos, endPos - startPos + (usedFallback ? 0 : chosenEndTag.length()));

            std::string readable;
            bool insideTag = false;
            for (char c : htmlChunk) {
                if (c == '<') {
                    insideTag = true;
                    continue;
                }
                if (c == '>') {
                    insideTag = false;
                    readable += ' ';
                    continue;
                }
                if (!insideTag) readable += c;
            }

            std::regex spaceRegex("\\s+");
            readable = std::regex_replace(readable, spaceRegex, " ");

            std::cout << "\n📝 Parsed PrimeNet Result Summary:\n" << readable << "\n";
        } else {
            std::cout << "⚠️ Could not find an end marker. Raw response:\n\n" << response << "\n";
        }
    } else {
        std::cout << "⚠️ Could not find start of results section. Raw response:\n\n" << response << "\n";
    }

    return true;
#else
    (void)jsonResult;
    (void)username;
    (void)password;
    return false;
#endif
}

} // namespace io
