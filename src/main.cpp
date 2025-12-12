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
#include <fstream>
#include <streambuf>
#include <iostream>
#include <string>

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

int main(int argc, char** argv) {
    LogTee _tee("prmers.log");
    return core::App(argc, argv).run();
}
