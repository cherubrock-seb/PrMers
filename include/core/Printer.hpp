// core/Printer.hpp
#ifndef CORE_PRINTER_HPP
#define CORE_PRINTER_HPP

#include "io/CliParser.hpp"

namespace core {

class Printer {
public:
    static void banner(const io::CliOptions& opts);
    static bool finalReport(const io::CliOptions& opts,
                          double elapsed,
                          const std::string& jsonResult,
                          bool isPrime);
    static void displayVector(const std::vector<uint64_t>& vec, const std::string& label = "Vector");
    static std::string formatNumber(const io::CliOptions& opts);

};

} // namespace core

#endif
