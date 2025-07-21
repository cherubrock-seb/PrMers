#pragma once
#include <cstdint>
#include <vector>
#include <functional>
#include <gmp.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace opencl { class Buffers; }
namespace math { class Carry; }

namespace math {

class GerbiczLiChecker {
public:
    using MulFn = std::function<void(cl_mem, cl_mem)>;
    GerbiczLiChecker(uint64_t L,
                     uint64_t B,
                     cl_context ctx,
                     cl_command_queue q,
                     opencl::Buffers& bufs,
                     size_t limbBytes,
                     uint64_t baseA,
                     Carry& carry,
                     MulFn mul,
                     bool debug=false);
    void init(cl_mem x0Buf, uint64_t resumeIter1);
    void step(uint64_t iter1, cl_mem xBuf);
    bool finalCheck(cl_mem xBuf,
                    const std::vector<int>& digitWidths,
                    const mpz_t& Mp);
    bool finalDebugCheck(cl_mem xBuf,
                         const std::vector<int>& digitWidths,
                         const mpz_t& Mp);
    ~GerbiczLiChecker();
private:
    uint64_t L_;
    uint64_t B_;
    uint64_t q_;
    uint64_t r_;
    cl_context ctx_;
    cl_command_queue qcl_;
    opencl::Buffers& bufs_;
    size_t limbBytes_;
    uint64_t baseA_;
    Carry& carry_;
    MulFn mul_;
    cl_mem dBuf_;
    cl_mem zBuf_;
    bool haveZ_;
    bool debug_;
    void gpuCopy(cl_mem src, cl_mem dst);
    void mpzFromBuf(mpz_t out,
                    cl_mem buf,
                    const std::vector<int>& widths,
                    const mpz_t Mp);
    void pow2pow(mpz_t out, const mpz_t base, uint64_t k, const mpz_t mod);
    uint64_t hash64(const mpz_t x);
};

}
