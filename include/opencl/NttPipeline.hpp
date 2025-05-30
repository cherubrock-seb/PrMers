#ifndef OPENCL_NTTPIPELINE_HPP
#define OPENCL_NTTPIPELINE_HPP

#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace opencl {

struct NttStage {
    cl_kernel                kernel;
    struct Arg { size_t size; std::vector<uint8_t> data; };
    std::vector<Arg>         args;
    size_t                   globalDiv;
    const size_t*            localSize;
    std::string              name;
};

inline void setStageArgs(const NttStage& s) {
    for (cl_uint i = 0; i < s.args.size(); ++i) {
        const auto& A = s.args[i];
        if (!A.data.empty())
            clSetKernelArg(s.kernel, i, A.size, A.data.data());
    }
}

template<typename T>
static std::vector<uint8_t> toBytes(const T& x) {
    std::vector<uint8_t> b(sizeof(T));
    std::memcpy(b.data(), &x, sizeof(T));
    return b;
}

std::vector<NttStage> buildForwardPipeline(
    cl_uint n,
    cl_command_queue queue,
    cl_mem buf_x,
    cl_kernel k_first,
    cl_kernel k_mm_2,
    cl_kernel k_mm_3,
    cl_kernel k_m4,
    cl_kernel k_m8,
    cl_kernel k_last_m1,
    cl_kernel k_last_m1_n4,
    cl_kernel k_r2_s_r2_r4,
    cl_kernel k_radix2_square_radix2,
    cl_mem buf_w,
    cl_mem buf_dw,
    const size_t* ls0,
    const size_t* ls2,
    const size_t* ls3
) {
    std::vector<NttStage> v;
    if (n == 4) {
        v.push_back(NttStage{
            k_last_m1_n4,
            { { sizeof(cl_mem), toBytes(buf_x) },
              { sizeof(cl_mem), toBytes(buf_w) },
              { sizeof(cl_mem), toBytes(buf_dw) } },
            1,
            ls0,
            "kernel_ntt_last_m1_n4(m=1)"
        });
        return v;
    }
    else{
        cl_uint m0 = n / 4;
        v.push_back(NttStage{
            k_first,
            { { sizeof(cl_mem), toBytes(buf_x) },
            { sizeof(cl_mem), toBytes(buf_w) },
            { sizeof(cl_mem), toBytes(buf_dw) } },
            2,
            ls0,
            "kernel_ntt_radix4_mm_first(m=" + std::to_string(m0) + ")"
        });
        cl_uint mm = m0;
        for (cl_uint m = n / 16; m >= 32; m /= 16) {
            v.push_back(NttStage{
                k_mm_2,
                { { sizeof(cl_mem),  toBytes(buf_x) },
                { sizeof(cl_mem),  toBytes(buf_w) },
                { sizeof(cl_uint), toBytes(m) } },
                4,
                ls2,
                "kernel_ntt_mm_2_steps(m=" + std::to_string(m) + ")"
            });
            mm = m / 4;
        }
        if (mm == 256) {
            v.push_back(NttStage{
                k_mm_2,
                { { sizeof(cl_mem),  toBytes(buf_x) },
                { sizeof(cl_mem),  toBytes(buf_w) },
                { sizeof(cl_uint), toBytes((cl_uint)64) } },
                4,
                ls2,
                "kernel_ntt_mm_2_steps(m=64)"
            });
            v.push_back(NttStage{
                k_m4,
                { { sizeof(cl_mem), toBytes(buf_x) },
                { sizeof(cl_mem), toBytes(buf_w) } },
                2,
                ls0,
                "kernel_ntt_mm(m=4)"
            });
            v.push_back(NttStage{
                k_last_m1,
                { { sizeof(cl_mem), toBytes(buf_x) },
                { sizeof(cl_mem), toBytes(buf_w) } },
                1,
                ls0,
                "kernel_ntt_radix4_last_m1(m=4)"
            });
        }
        else if (mm == 64) {
            v.push_back(NttStage{
                k_mm_2,
                { { sizeof(cl_mem),  toBytes(buf_x) },
                { sizeof(cl_mem),  toBytes(buf_w) },
                { sizeof(cl_uint), toBytes((cl_uint)16) } },
                4,
                ls2,
                "kernel_ntt_mm_2_steps(m=16)"
            });
            v.push_back(NttStage{
                k_last_m1,
                { { sizeof(cl_mem), toBytes(buf_x) },
                { sizeof(cl_mem), toBytes(buf_w) } },
                1,
                ls0,
                "kernel_ntt_last_m1(m=16)"
            });
        }
        else if (mm == 32) {
            v.push_back(NttStage{
                k_m8,
                { { sizeof(cl_mem), toBytes(buf_x) },
                { sizeof(cl_mem), toBytes(buf_w) } },
                2,
                ls0,
                "kernel_ntt_mm(m=8)"
            });
            v.push_back(NttStage{
                k_r2_s_r2_r4,
                { { sizeof(cl_mem), toBytes(buf_x) } },
                2,
                ls3,
                "kernel_radix4_radix2_square_radix2_radix4(m=2)"
            });
        }
        else if (mm == 16) {
            v.push_back(NttStage{
                k_m4,
                { { sizeof(cl_mem), toBytes(buf_x) },
                { sizeof(cl_mem), toBytes(buf_w) } },
                2,
                ls0,
                "kernel_ntt_mm(m=4)"
            });
            v.push_back(NttStage{
                k_last_m1,
                { { sizeof(cl_mem), toBytes(buf_x) },
                { sizeof(cl_mem), toBytes(buf_w) } },
                1,
                ls0,
                "kernel_ntt_radix4_last_m1(m=4)"
            });
        }
        else if (mm == 8) {
            v.push_back(NttStage{
                k_r2_s_r2_r4,
                { { sizeof(cl_mem), toBytes(buf_x) } },
                2,
                ls3,
                "kernel_radix4_radix2_square_radix2_radix4(m=2)"
            });
        }
        else if (mm == 4) {
            v.push_back(NttStage{
                k_last_m1,
                { { sizeof(cl_mem), toBytes(buf_x) },
                { sizeof(cl_mem), toBytes(buf_w) } },
                1,
                ls0,
                "kernel_ntt_radix4_last_m1(m=4)"
            });
        }
        else {
            v.push_back(NttStage{
                k_radix2_square_radix2,
                { { sizeof(cl_mem), toBytes(buf_x) } },
                0,
                ls0,
                "kernel_radix2_square_radix2(m=?)"
            });
        }
    }
    return v;
}

std::vector<NttStage> buildInversePipeline(
    cl_uint n,
    cl_command_queue queue,
    cl_mem buf_x,
    cl_kernel k_i_m1_n4,
    cl_kernel k_i_m1,
    cl_kernel k_i_mm_2,
    cl_kernel k_i_mm,
    cl_kernel k_i_mm_last,
    cl_mem buf_wi,
    cl_mem buf_diw,
    const size_t* ls0,
    const size_t* ls2
) {
    std::vector<NttStage> v;
    if (n == 4) {
        v.push_back(NttStage{
            k_i_m1_n4,
            { { sizeof(cl_mem), toBytes(buf_x) },
              { sizeof(cl_mem), toBytes(buf_wi) },
              { sizeof(cl_mem), toBytes(buf_diw) } },
            1,
            ls0,
            "kernel_inverse_ntt_m1_n4(m=1)"
        });
        return v;
    }
    else{
        cl_uint mmm=n/4;
        for (cl_uint m = n / 16; m >= 32; m /= 16) {
            
            mmm =m/16;
        }
        bool even = !(mmm == 8||mmm==2||mmm==32)||(n==4);
        if (even) {
            v.push_back(NttStage{
                k_i_m1,
                { { sizeof(cl_mem), toBytes(buf_x) },
                { sizeof(cl_mem), toBytes(buf_wi) } },
                1,
                ls0,
                "kernel_inverse_ntt_radix4_m1(m=1)"
            });
            cl_uint mm = 4;
            for (cl_uint m = 4; m < n/16; m *= 16) {
                v.push_back(NttStage{
                    k_i_mm_2,
                    { { sizeof(cl_mem),  toBytes(buf_x) },
                    { sizeof(cl_mem),  toBytes(buf_wi) },
                    { sizeof(cl_uint), toBytes(m) } },
                    4,
                    ls2,
                    "kernel_ntt_inverse_mm_2_steps(m=" + std::to_string(m) + ")"
                });
                mm = m*16;

            }
            if (mm <= n/16 && n > 8) {
                v.push_back(NttStage{
                    k_i_mm,
                    { { sizeof(cl_mem),  toBytes(buf_x) },
                    { sizeof(cl_mem),  toBytes(buf_wi) },
                    { sizeof(cl_uint), toBytes(mm) } },
                    2,
                    ls0,
                    "kernel_inverse_ntt_radix4_mm(m=" + std::to_string(mm) + ")"
                });
            }
        } else {
            cl_uint mm = 8;
            for (cl_uint m = 8; m < n/16; m *= 16) {
                v.push_back(NttStage{
                    k_i_mm_2,
                    { { sizeof(cl_mem),  toBytes(buf_x) },
                    { sizeof(cl_mem),  toBytes(buf_wi) },
                    { sizeof(cl_uint), toBytes(m) } },
                    4,
                    ls2,
                    "kernel_ntt_inverse_mm_2_steps(m=" + std::to_string(m) + ")"
                });
                mm = m*16;
            }
            if (mm <= n/16 && n > 8) {
                v.push_back(NttStage{
                    k_i_mm,
                    { { sizeof(cl_mem),  toBytes(buf_x) },
                    { sizeof(cl_mem),  toBytes(buf_wi) },
                    { sizeof(cl_uint), toBytes(mm) } },
                    2,
                    ls0,
                    "kernel_inverse_ntt_radix4_mm(m=" + std::to_string(mm) + ")"
                });
            }
        }
        cl_uint mm = n / 4;
        v.push_back(NttStage{
            k_i_mm_last,
            { { sizeof(cl_mem),   toBytes(buf_x) },
            { sizeof(cl_mem),   toBytes(buf_wi) },
            { sizeof(cl_mem),   toBytes(buf_diw) },
            { sizeof(cl_uint),  toBytes(mm) } },
            2,
            ls0,
            "kernel_inverse_ntt_radix4_mm_last(m=" + std::to_string(mm) + ")"
        });
    }
    return v;
}

} // namespace opencl

#endif // OPENCL_NTTPIPELINE_HPP
