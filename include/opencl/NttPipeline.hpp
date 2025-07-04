#ifndef OPENCL_NTTPIPELINE_HPP
#define OPENCL_NTTPIPELINE_HPP

#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <functional>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace opencl {
enum class ArgKind {
  BufX,
  BufW,
  BufDW,  
  ParamM,
  BufW4,
  BufW5,
};
struct RadixOp {
    enum Position { First, Any, Last } position;
    int            localFactor;
    int           globalScale;
    cl_kernel      kernel;
    const size_t*  localSize;
    std::string    name;
    std::function<bool(unsigned m, unsigned n)> condition;
    std::vector<ArgKind> argKinds;
    int                      outputInverse;
};
struct NttStage {
    cl_kernel                kernel;
    struct Arg { size_t size; std::vector<uint8_t> data; };
    std::vector<Arg>         args;
    int                      globalScale;
    const size_t*            localSize;
    std::string              name;
    int                      outputInverse;
};

inline void setStageArgs(NttStage& s, cl_mem buf_x) {
    for (cl_uint i = 0; i < s.args.size(); ++i) {
        auto& A = s.args[i];
        if (!A.data.empty()) {
            if (A.size == sizeof(cl_mem)) {
                cl_mem* candidate = reinterpret_cast<cl_mem*>(A.data.data());
                if (*candidate == nullptr) {
                    std::memcpy(A.data.data(), &buf_x, sizeof(cl_mem));
                }
            }
            clSetKernelArg(s.kernel, i, A.size, A.data.data());
        }
    }
}

template<typename T>
static std::vector<uint8_t> toBytes(const T& x) {
    std::vector<uint8_t> b(sizeof(T));
    std::memcpy(b.data(), &x, sizeof(T));
    return b;
}
inline NttStage makeStage(const RadixOp& op,
                          cl_uint m,
                          cl_mem buf_x,
                          cl_mem buf_w4,
                          cl_mem buf_w5,
                          cl_mem buf_dw = nullptr)
{
    NttStage s;
    s.kernel    = op.kernel;
    s.globalScale = op.globalScale;
    s.localSize   = op.localSize;
    s.name      = op.name + "(m=" + std::to_string(m) + ")";
    s.outputInverse = op.outputInverse;
    std::cout << s.name << std::endl;
    for (auto kind : op.argKinds) {
        switch (kind) {
            case ArgKind::BufX:
                s.args.push_back({ sizeof(cl_mem), toBytes(buf_x) });
                break;
            case ArgKind::BufW:
                s.args.push_back({ sizeof(cl_mem), toBytes(buf_w4) });
                break;
            case ArgKind::BufDW:
                s.args.push_back({ sizeof(cl_mem), toBytes(buf_dw) });
                break;
            case ArgKind::ParamM:
                s.args.push_back({ sizeof(cl_uint), toBytes((cl_uint)m) });
                break;            
            case ArgKind::BufW5:
                s.args.push_back({ sizeof(cl_mem), toBytes(buf_w5) });
                break;
            case ArgKind::BufW4:
                s.args.push_back({ sizeof(cl_mem), toBytes(buf_w4) });
                break;
        }
    }
    return s;
}

inline std::vector<NttStage> buildForwardPipeline(
    cl_uint n,
    cl_command_queue queue,
    cl_mem buf_x,
    cl_kernel k_first,
    cl_kernel k_mm_2,
    cl_kernel k_m4,
    cl_kernel k_m8,
    cl_kernel k_m16,
    cl_kernel k_m32,
    cl_kernel k_last_m1,
    cl_kernel k_last_m1_n4,
    cl_kernel k_r2_s_r2_r4,
    cl_kernel k_radix2_square_radix2,
    cl_kernel k_mm_2_first,
    cl_kernel k_r4_s_r4,
    cl_kernel k_r5_first,
    cl_mem buf_dw,
    cl_mem buf_w4,
    cl_mem buf_w5,
    const size_t* ls0,
    const size_t* ls2,
    const size_t* ls3
) {

    std::vector<RadixOp> all = {
    { RadixOp::First,   5, 5,
        k_r5_first,        ls0, "kernel_ntt_radix5_mm_first",
         [](auto m0, auto nn){ return nn%5==0; },
        { ArgKind::BufX,ArgKind::BufW5, ArgKind::BufDW,ArgKind::ParamM } , 0},
      
     { RadixOp::First,   16, 16,
        k_mm_2_first,        ls2, "kernel_ntt_radix4_mm_2steps_first",
         [](auto m0, auto nn){ return m0>32; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::BufDW, ArgKind::ParamM } , 0},
      
      { RadixOp::First,   4, 8,
        k_first,        ls0, "kernel_ntt_radix4_mm_first",
        /*cond*/ [](auto m0, auto){ return m0>=2; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::BufDW, ArgKind::ParamM } , 0},
        
      { RadixOp::Any,     16, 16,
        k_mm_2,         ls2, "kernel_ntt_radix4_mm_2steps",
        [](auto m0, auto){ return m0>=64 && m0%4==0; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::ParamM } ,0},



      { RadixOp::Any,     4, 8,
        k_m4,           ls0, "kernel_ntt_radix4_mm_m4",
        [](auto m0, auto){ return m0==16; },
        { ArgKind::BufX, ArgKind::BufW } ,0},

      { RadixOp::Any,     4, 8,
        k_m8,           ls0, "kernel_ntt_radix4_mm_m8",
        [](auto m0, auto){ return m0==32; },
        { ArgKind::BufX, ArgKind::BufW } ,0},
      { RadixOp::Any,     4, 8,
        k_m16,           ls0, "kernel_ntt_radix4_mm_m16",
        [](auto m0, auto){ return m0==64; },
        { ArgKind::BufX, ArgKind::BufW } ,0},
       { RadixOp::Any,     4, 8,
        k_m32,           ls0, "kernel_ntt_radix4_mm_m32",
        [](auto m0, auto){ return m0==128; },
        { ArgKind::BufX, ArgKind::BufW } ,0},
      { RadixOp::Last,     4, 8,
        k_r2_s_r2_r4,   ls3, "kernel_radix4_radix2_square_radix2_radix4",
        [](auto m0, auto){ return m0==8;},
        { ArgKind::BufX } ,8},

      { RadixOp::Last,    2, 2,
        k_radix2_square_radix2, ls0, "kernel_ntt_radix2_square_radix2",
        [](auto m0, auto){ return m0==2;},
        { ArgKind::BufX } ,2},

        { RadixOp::Last,    4, 4,
        k_r4_s_r4,      ls0, "kernel_ntt_radix4_square_radix4",
        [](auto m0, auto){ return m0==4; },
        { ArgKind::BufX} ,4},

      { RadixOp::Last,    4, 4,
        k_last_m1,      ls0, "kernel_ntt_radix4_last_m1",
        [](auto m0, auto){ return m0==4; },
        { ArgKind::BufX, ArgKind::BufW } ,1},

      { RadixOp::Last,    4, 4,
        k_last_m1_n4,   ls0, "kernel_ntt_radix4_last_m1_n4",
        [](auto, auto nn){ return nn==4; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::BufDW } ,1}
    };


    std::vector<NttStage> v;
    if (n == 4) {
        v.push_back(NttStage{
            k_last_m1_n4,
            { { sizeof(cl_mem), toBytes(buf_x) },
              { sizeof(cl_mem), toBytes(buf_w4) },
              { sizeof(cl_mem), toBytes(buf_dw) } },
            4,
            ls0,
            "kernel_ntt_last_m1_n4(m=1)"
        });
        return v;
    }
    else{
        cl_uint m0 = n;

        auto it = std::find_if(all.begin(), all.end(), [&](auto& op){
                return op.position==RadixOp::First
                    && op.condition(m0,n)
                    && m0 % op.localFactor==0;
            });
            if (it!=all.end()) {
                if(it->name=="kernel_ntt_radix4_mm_2steps_first"){
                        m0 /= 4;
                }
                else{
                    m0 /= it->localFactor;
                }
                v.push_back(makeStage(*it,m0,buf_x,buf_w4, buf_w5, buf_dw));
                if(it->name=="kernel_ntt_radix4_mm_2steps_first"){
                        m0 /= 4;
                }
            }
        
        while (m0 > 8) {
            auto it = std::find_if(all.begin(), all.end(), [&](auto& op){
                return op.position==RadixOp::Any
                    && op.condition(m0,n)
                    && m0 % op.localFactor==0;
            });
            if (it==all.end()) break;
            if(it->name=="kernel_ntt_radix4_mm_2steps"){
                    m0 /= 4;
            }
            else{
                m0 /= it->localFactor;
            }
            v.push_back(makeStage(*it,m0,buf_x,buf_w4, buf_w5,buf_dw));
            if(it->name=="kernel_ntt_radix4_mm_2steps"){
                    m0 /= 4;
            }
        }
        if (m0 >= 1) {
            auto it = std::find_if(all.begin(), all.end(), [&](auto& op){
                return op.position == RadixOp::Last
                    && op.condition(m0, n)
                    && m0 % op.localFactor == 0;
            });
            
            if (it != all.end()) {
                m0 /= it->localFactor;
                v.push_back(makeStage(*it, m0, buf_x,buf_w4, buf_w5, buf_dw));
            } /*else {
                throw std::runtime_error("Pas de RadixOp::Last pour m0=" + std::to_string(m0));
            }*/
        }
        
    }
    return v;
}


inline std::vector<NttStage> buildInversePipeline(
    cl_uint n,
    cl_command_queue queue,
    cl_mem buf_x,
    cl_kernel k_i_m1_n4,
    cl_kernel k_i_m1,
    cl_kernel k_i_mm_2,
    cl_kernel k_i_mm,
    cl_kernel k_i_mm_last,
    cl_kernel k_i_mm_2_last,
    cl_kernel k_i_r5_last,
    cl_mem buf_wi4,
    cl_mem buf_wi5,
    cl_mem buf_diw,
    const size_t* ls0,
    const size_t* ls2,
    int lastOutputInv
) {
    std::vector<RadixOp> allInv = {
      { RadixOp::First,  4, 4,
        k_i_m1,        ls0, "kernel_inverse_ntt_radix4_m1",
        [](auto m0, auto nn){ return m0==1; },
        { ArgKind::BufX },0},
      /*{ RadixOp::First,    16, 16,
        k_i_mm_2,      ls2, "kernel_ntt_inverse_mm_2_steps",
        [](auto m0, auto nn){ return m0=8 && nn>32; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::ParamM },
        -1
      },*/
      { RadixOp::Any,    16, 16,
        k_i_mm_2,      ls2, "kernel_ntt_inverse_mm_2_steps",
        [](auto m0, auto nn){ return m0 < nn/16; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::ParamM },
        /*outputInverse*/ 0
      },
      { RadixOp::Any,    4, 8,
        k_i_mm,        ls0, "kernel_inverse_ntt_radix4_mm",
        [](auto m0, auto nn){ return m0 < nn/4; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::ParamM },
        /*outputInverse*/ 0
      },
      { RadixOp::Last,   5,  5,
        k_i_r5_last,    ls0, "kernel_ntt_inverse_radix5_mm_last",
        [](auto m0, auto nn){ return m0 == nn/5; },
        { ArgKind::BufX,ArgKind::BufW5, ArgKind::BufDW,ArgKind::ParamM },
        /*outputInverse*/ 0
      },
      { RadixOp::Last,   16,  16,
        k_i_mm_2_last,    ls2, "kernel_ntt_radix4_inverse_mm_2steps_last",
        [](auto m0, auto nn){ return m0 == nn/16; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::BufDW,ArgKind::ParamM },
        /*outputInverse*/ 0
      },
      { RadixOp::Last,   4,  8,
        k_i_mm_last,   ls0, "kernel_inverse_ntt_radix4_mm_last",
        [](auto m0, auto nn){ return m0==nn/4; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::BufDW, ArgKind::ParamM },
        /*outputInverse*/ 0
      },
      { RadixOp::Last,   4, 4,
        k_i_m1_n4,     ls0, "kernel_inverse_ntt_m1_n4",
        [](auto m0, auto nn){ return m0==1 && nn==4; },
        { ArgKind::BufX, ArgKind::BufW, ArgKind::BufDW },
        /*outputInverse*/ 0
      }
    };

    std::vector<NttStage> v;
    if (n == 4) {
        v.push_back(NttStage{
            k_i_m1_n4,
            { { sizeof(cl_mem), toBytes(buf_x) },
              { sizeof(cl_mem), toBytes(buf_wi4) },
              { sizeof(cl_mem), toBytes(buf_diw) } },
            4,
            ls0,
            "kernel_inverse_ntt_m1_n4(m=1)"
        });
        return v;
    }
    else{

        unsigned m0 = lastOutputInv;
        
        auto it = std::find_if(allInv.begin(), allInv.end(), [&](auto& op){
                        return op.position == RadixOp::First
                            && op.condition(m0,n);
                    });
        if (it != allInv.end()){
            v.push_back(makeStage(*it, m0, buf_x, buf_wi4, buf_wi5, buf_diw));
            m0 *= it->localFactor;    
        }
        
        
        while(((n%5!=0) && (m0 < (unsigned)(n/16)))||((n%5==0)&&(m0 < (unsigned)(n/10)) ) ){
            auto it = std::find_if(allInv.begin(), allInv.end(), [&](auto& op){
                return op.position == RadixOp::Any
                    && op.condition(m0,n)
                    && (m0 * op.localFactor) <= (unsigned)(n/4);
            });
            if (it == allInv.end()) break;
            v.push_back(makeStage(*it, m0, buf_x, buf_wi4, buf_wi5, buf_diw));
            m0 *= it->localFactor;
                
        }
        if (((n%5!=0)&&(m0 >= n/16 || n == 8)) || m0 == n/5)  {
            auto it = std::find_if(allInv.begin(), allInv.end(), [&](auto& op){
                return op.position == RadixOp::Last
                    && op.condition(m0,n);
            });

            if (it != allInv.end()) {
                
                v.push_back(makeStage(*it, m0, buf_x, buf_wi4, buf_wi5, buf_diw));
                m0 *= it->localFactor;
            }
        }
    }
    return v;
}

} // namespace opencl

#endif // OPENCL_NTTPIPELINE_HPP
