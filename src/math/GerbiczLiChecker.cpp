#include "math/GerbiczLiChecker.hpp"
#include "math/Carry.hpp"
#include <stdexcept>
#include <vector>
#include <iostream>

namespace core {
void vectToMpz2(mpz_t,
                const std::vector<uint64_t>&,
                const std::vector<int>&,
                const mpz_t);
}

namespace math {

GerbiczLiChecker::GerbiczLiChecker(uint64_t L,
                                   uint64_t B,
                                   cl_context ctx,
                                   cl_command_queue q,
                                   opencl::Buffers& bufs,
                                   size_t limbBytes,
                                   uint64_t baseA,
                                   Carry& carry,
                                   MulFn mul,
                                   bool debug)
    : L_(L)
    , B_(B)
    , q_(L / B)
    , r_(L % B)
    , ctx_(ctx)
    , qcl_(q)
    , bufs_(bufs)
    , limbBytes_(limbBytes)
    , baseA_(baseA)
    , carry_(carry)
    , mul_(std::move(mul))
    , dBuf_(nullptr)
    , zBuf_(nullptr)
    , haveZ_(false)
    , debug_(debug)
{
    cl_int err;
    dBuf_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, limbBytes_, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer dBuf");
    if (r_ > 0) {
        zBuf_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, limbBytes_, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer zBuf");
    }
    if (debug_) {
        std::cout << "[Gerbicz] L=" << L_ << " B=" << B_ << " q=" << q_ << " r=" << r_ << std::endl;
    }
}

void GerbiczLiChecker::gpuCopy(cl_mem src, cl_mem dst)
{
    clEnqueueCopyBuffer(qcl_, src, dst, 0, 0, limbBytes_, 0, nullptr, nullptr);
}

void GerbiczLiChecker::init(cl_mem x0Buf, uint64_t resumeIter1)
{
    gpuCopy(x0Buf, dBuf_);

    if (r_ > 0) {
        uint64_t qB = L_ - r_;
        if (resumeIter1 >= qB) {
            gpuCopy(x0Buf, zBuf_);
            haveZ_ = true;
        }
    }
}

void GerbiczLiChecker::step(uint64_t iter1, cl_mem xBuf)
{
    if (r_ == 0) {
        if (iter1 % B_ == 0 && iter1 < L_) {
            mul_(dBuf_, xBuf);
            if (debug_) std::cout << "[Gerbicz] mult d *= x_" << iter1 << std::endl;
        }
    } else {
        uint64_t qB = L_ - r_;
        if (iter1 % B_ == 0 && iter1 < qB) {
            mul_(dBuf_, xBuf);
            if (debug_) std::cout << "[Gerbicz] mult d *= x_" << iter1 << std::endl;
        }
        if (iter1 == qB && !haveZ_) {
            gpuCopy(xBuf, zBuf_);
            haveZ_ = true;
            if (debug_) std::cout << "[Gerbicz] snapshot z = x_" << iter1 << std::endl;
        }
    }
}

void GerbiczLiChecker::mpzFromBuf(mpz_t out,
                                  cl_mem buf,
                                  const std::vector<int>& widths,
                                  const mpz_t Mp)
{
    std::vector<uint64_t> tmp(limbBytes_ / sizeof(uint64_t));
    clEnqueueReadBuffer(qcl_, buf, CL_TRUE, 0, limbBytes_, tmp.data(), 0, nullptr, nullptr);
    carry_.handleFinalCarry(tmp, widths);
    core::vectToMpz2(out, tmp, widths, Mp);
}

void GerbiczLiChecker::pow2pow(mpz_t out, const mpz_t base, uint64_t k, const mpz_t mod)
{
    mpz_set(out, base);
    for (uint64_t i = 0; i < k; ++i) {
        mpz_mul(out, out, out);
        mpz_mod(out, out, mod);
    }
}

uint64_t GerbiczLiChecker::hash64(const mpz_t x)
{
    size_t n = mpz_size(x);
    const mp_limb_t* p = mpz_limbs_read(x);
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ (uint64_t)n;
    for (size_t i = 0; i < n; ++i) {
        uint64_t v = (uint64_t)p[i];
        h ^= v + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
    }
    return h;
}

bool GerbiczLiChecker::finalCheck(cl_mem xBuf,
                                  const std::vector<int>& widths,
                                  const mpz_t& Mp)
{
    if (debug_) return finalDebugCheck(xBuf, widths, Mp);
    mpz_t m; mpz_init_set(m, Mp);
    mpz_t d; mpz_init(d);
    mpz_t e; mpz_init(e);
    mpzFromBuf(d, dBuf_, widths, m);
    mpzFromBuf(e, xBuf, widths, m);
    mpz_t left; mpz_init(left);
    mpz_mul(left, d, e);
    mpz_mod(left, left, m);
    mpz_t d2B; mpz_init(d2B);
    pow2pow(d2B, d, B_, m);
    mpz_t right; mpz_init(right);
    mpz_mul_ui(right, d2B, baseA_);
    mpz_mod(right, right, m);
    if (r_ > 0) {
        if (!haveZ_) { mpz_clears(m,d,e,left,d2B,right,nullptr); return false; }
        mpz_t z; mpz_init(z);
        mpzFromBuf(z, zBuf_, widths, m);
        mpz_t zinv; mpz_init(zinv);
        if (mpz_invert(zinv, z, m) == 0) { mpz_clears(m,d,e,left,d2B,right,z,zinv,nullptr); return false; }
        mpz_mul(right, right, e);
        mpz_mod(right, right, m);
        mpz_mul(right, right, zinv);
        mpz_mod(right, right, m);
        mpz_clears(z, zinv, nullptr);
    }
    bool ok = (mpz_cmp(left, right) == 0);
    mpz_clears(m,d,e,left,d2B,right,nullptr);
    return ok;
}

bool GerbiczLiChecker::finalDebugCheck(cl_mem xBuf,
                                       const std::vector<int>& widths,
                                       const mpz_t& Mp)
{
    mpz_t m; mpz_init_set(m, Mp);
    mpz_t d; mpz_init(d);
    mpz_t e; mpz_init(e);
    mpzFromBuf(d, dBuf_, widths, m);
    mpzFromBuf(e, xBuf, widths, m);
    mpz_t d2B; mpz_init(d2B);
    pow2pow(d2B, d, B_, m);
    mpz_t z; mpz_init(z);
    if (r_>0) {
        if (!haveZ_) std::cout << "[Gerbicz] missing z snapshot\n";
        else mpzFromBuf(z, zBuf_, widths, m);
    }
    mpz_t A1; mpz_init(A1);
    mpz_t A2; mpz_init(A2);
    mpz_mul(A1, d, e);
    mpz_mod(A1, A1, m);
    mpz_mul_ui(A2, d2B, baseA_);
    mpz_mod(A2, A2, m);
    if (r_>0) {
        mpz_t zinv; mpz_init(zinv);
        mpz_invert(zinv, z, m);
        mpz_mul(A2, A2, e);
        mpz_mod(A2, A2, m);
        mpz_mul(A2, A2, zinv);
        mpz_mod(A2, A2, m);
        mpz_clear(zinv);
    }
    mpz_t deltaB; mpz_init(deltaB);
    mpz_t A2inv; mpz_init(A2inv);
    if (mpz_invert(A2inv, A2, m) == 0) std::cout << "[Gerbicz] invert A2 failed\n";
    else {
        mpz_mul(deltaB, A1, A2inv);
        mpz_mod(deltaB, deltaB, m);
    }
    mpz_t lhsA; mpz_init(lhsA);
    mpz_t rhsA; mpz_init(rhsA);
    if (r_==0 || haveZ_) {
        mpz_mul(lhsA, d, (r_==0)? e : z);
        mpz_mod(lhsA, lhsA, m);
        mpz_mul_ui(rhsA, d2B, baseA_);
        mpz_mod(rhsA, rhsA, m);
        mpz_t rhsAinv; mpz_init(rhsAinv);
        if (mpz_invert(rhsAinv, rhsA, m)) {
            mpz_mul(rhsAinv, lhsA, rhsAinv);
            mpz_mod(rhsAinv, rhsAinv, m);
            std::cout << "[Gerbicz] deltaA hash=" << hash64(rhsAinv) << " is1=" << (mpz_cmp_ui(rhsAinv,1)==0) << std::endl;
        } else {
            std::cout << "[Gerbicz] invert rhsA failed\n";
        }
        mpz_clear(rhsAinv);
    }
    std::cout << "[Gerbicz] hash(d)=" << hash64(d)
              << " hash(e)=" << hash64(e)
              << " hash(d2B)=" << hash64(d2B);
    if (r_>0 && haveZ_) std::cout << " hash(z)=" << hash64(z);
    std::cout << " deltaB_hash=" << hash64(deltaB)
              << " okB=" << (mpz_cmp_ui(deltaB,1)==0) << std::endl;
    bool ok = (mpz_cmp_ui(deltaB,1)==0);
    mpz_clears(m,d,e,d2B,z,A1,A2,deltaB,A2inv,lhsA,rhsA,nullptr);
    return ok;
}

GerbiczLiChecker::~GerbiczLiChecker()
{
    if (dBuf_) clReleaseMemObject(dBuf_);
    if (zBuf_) clReleaseMemObject(zBuf_);
}

}
