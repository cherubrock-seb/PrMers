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
#include "core/App.hpp"
#include "io/CliParser.hpp"
#include "aevum/EngineAevum.hpp"
#include <fstream>
#include <streambuf>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <exception>

class TeeBuf : public std::streambuf {
    std::streambuf* a_;
    std::streambuf* b_;
public:
    TeeBuf(std::streambuf* a, std::streambuf* b) : a_(a), b_(b) {}
protected:
    int overflow(int ch) override {
        if (ch == EOF) return !EOF;
        const int r1 = a_ ? a_->sputc(ch) : ch;
        const int r2 = b_ ? b_->sputc(ch) : ch;
        return (r1 == EOF || r2 == EOF) ? EOF : ch;
    }
    std::streamsize xsputn(const char* s, std::streamsize count) override {
        const std::streamsize r1 = a_ ? a_->sputn(s, count) : count;
        const std::streamsize r2 = b_ ? b_->sputn(s, count) : count;
        return (r1 == count && r2 == count) ? count : 0;
    }

    int sync() override {
        const int s1 = a_ ? a_->pubsync() : 0;
        const int s2 = b_ ? b_->pubsync() : 0;
        return (s1 == 0 && s2 == 0) ? 0 : -1;
    }
};

struct LogTee {
    std::ofstream file;
    std::streambuf *oldCout = nullptr, *oldCerr = nullptr, *oldClog = nullptr;
    TeeBuf *teeCout = nullptr, *teeCerr = nullptr, *teeClog = nullptr;

    explicit LogTee(const std::string& path) : file(path, std::ios::app) {
        if (!file) return;
        oldCout = std::cout.rdbuf();
        oldCerr = std::cerr.rdbuf();
        oldClog = std::clog.rdbuf();
        teeCout = new TeeBuf(oldCout, file.rdbuf());
        teeCerr = new TeeBuf(oldCerr, file.rdbuf());
        teeClog = new TeeBuf(oldClog, file.rdbuf());
        std::cout.rdbuf(teeCout);
        std::cerr.rdbuf(teeCerr);
        std::clog.rdbuf(teeClog);
        std::cout.setf(std::ios::unitbuf);
        std::cerr.setf(std::ios::unitbuf);
        std::clog.setf(std::ios::unitbuf);
    }

    ~LogTee() {
        if (teeCout) { std::cout.rdbuf(oldCout); delete teeCout; }
        if (teeCerr) { std::cerr.rdbuf(oldCerr); delete teeCerr; }
        if (teeClog) { std::clog.rdbuf(oldClog); delete teeClog; }
        if (file) file.flush();
    }
};

namespace {

std::vector<std::string> effectiveArguments(int argc, char** argv) {
    std::vector<std::string> out;
    out.emplace_back(argc > 0 && argv[0] ? argv[0] : "prmers");
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-config" && i + 1 < argc) {
            std::ifstream cfg(argv[++i]);
            std::string line;
            while (std::getline(cfg, line)) {
                std::istringstream iss(line);
                std::string token;
                while (iss >> token) out.push_back(token);
            }
        } else {
            out.push_back(arg);
        }
    }
    return out;
}

int validateCompatibilityBeforeApp(int argc, char** argv) {
    auto args = effectiveArguments(argc, argv);
    std::vector<char*> parsed_argv;
    parsed_argv.reserve(args.size() + 1);
    for (auto& arg : args) parsed_argv.push_back(arg.data());
    parsed_argv.push_back(nullptr);

    const auto options = io::CliParser::parse(
        static_cast<int>(args.size()), parsed_argv.data());

    if (options.pm1_ultralowmem && options.aevum) {
        std::cerr << "[Backend Compatibility] -pm1-ultralowmem is a Marin fast3-only "
                     "one-register algorithm and cannot be forced to Aevum. "
                     "Use automatic mode or -engine-marin.\n";
        return 2;
    }

    if (options.mode == "ll" && !options.marin) {
        std::cerr << "[Backend Compatibility] -llunsafe cannot use the legacy internal "
                     "PrMers NTT backend selected by -marin because that path is not "
                     "validated for Lucas-Lehmer. Use automatic mode, -engine-marin, "
                     "or -aevum.\n";
        return 2;
    }

    // A forced backend must be honest.  Resolve the native plan before App
    // constructs OpenCL contexts and large transform tables whenever the
    // exponent is available directly from the command line/config file.
    if (options.aevum && options.aevum_fft_spec.empty() && options.exponent != 0) {
        std::size_t transform = 0;
        std::string spec;
        std::string reason;
        if (!aevum_engine_resolve_auto_fft(
                static_cast<std::uint32_t>(options.exponent),
                &transform, &spec, &reason)) {
            std::cerr << "[Backend Aevum] Forced Aevum request cannot be satisfied for exponent "
                      << options.exponent << ": " << reason << ".\n";
            return 2;
        }
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    LogTee _tee("prmers.log");
    if (const int rc = validateCompatibilityBeforeApp(argc, argv); rc != 0) return rc;
    try {
        return core::App(argc, argv).run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }
}
