// core/Printer.hpp
#ifndef CORE_PRINTER_HPP
#define CORE_PRINTER_HPP

#include "io/CliParser.hpp"

namespace core {

class Printer {
public:
    static void banner(const io::CliOptions& opts);
    static bool finalReport(const io::CliOptions& opts,
                          const std::vector<uint64_t>& resultVec,
                          std::string res64,                          
                          uint64_t n,
                          const std::string& timestampBuf,
                          double elapsed, std::string jsonResult);
    static void displayVector(const std::vector<uint64_t>& vec, const std::string& label = "Vector");

};

} // namespace core

#endif
