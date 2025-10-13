#include "core/App.hpp"
#include "core/AlgoUtils.hpp"
#define NOMINMAX
#include "core/App.hpp"
#include "core/QuickChecker.hpp"
#include "core/Printer.hpp"
#include "core/ProofSet.hpp"
#include "core/ProofSetMarin.hpp"
#include "math/Carry.hpp"
#include "util/GmpUtils.hpp"
#include "io/WorktodoParser.hpp"
#include "io/WorktodoManager.hpp"
#include "io/CurlClient.hpp"
#include "marin/engine.h"
#include "marin/file.h"
#include "ui/WebGuiServer.hpp"
#include "core/Version.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <map>
#include <future>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/cl.h>
#endif
#ifdef _WIN32
# include <windows.h>
#endif
#include <csignal>
#include <chrono>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <tuple>
#include <atomic>
#include <fstream>
#include <memory>
#include <optional>
#include <cmath>
#include <thread>
#include <gmp.h>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <set>

using namespace core;
using namespace std::chrono;
using core::algo::format_res64_hex;
using core::algo::format_res2048_hex;
using core::algo::helperu;
using core::algo::mod3_words;
using core::algo::delete_checkpoints;
using core::algo::to_uppercase;
using core::algo::div3_words;
using core::algo::pack_words_from_eng_digits;
using core::algo::prp3_div9;
using core::algo::parseConfigFile;
using core::algo::interrupted;
using core::algo::handle_sigint;
using core::algo::writeStageResult;
using core::algo::restart_self;
using core::algo::buildE;
using core::algo::evenGapBound;
using core::algo::primeCountApprox;
using core::algo::read_mers_file;
using core::algo::writeEcmResumeLine;
using core::algo::mpz_to_lower_hex;
using core::algo::ecm_checksum_pminus1;
using core::algo::CHKSUMMOD;
using core::algo::write_prime95_s1_from_bytes;
using core::algo::checksum_prime95_s1;
using core::algo::hex_to_bytes_reversed_pad8;
using core::algo::parse_ecm_resume_line;
using core::algo::read_text_file;
using core::algo::write_u32;
using core::algo::write_i32;
using core::algo::write_u64;
using core::algo::write_u16;
using core::algo::write_f64;
using core::algo::write_u8;
using core::algo::hex_to_le_bytes_pad4;
using core::algo::buildE2;
using core::algo::product_prefix_fit_u64;
using core::algo::product_tree_range_u64;
using core::algo::compute_X_with_dots;
using core::algo::gcd_with_dots;

int core::App::runMemtestOpenCL() {
    if (guiServer_) { std::ostringstream oss; oss << "MEMTEST"; guiServer_->setStatus(oss.str()); }
    cl_uint np = 0; clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> plats(np); if (np) clGetPlatformIDs(np, plats.data(), nullptr);
    std::vector<cl_device_id> devs; std::vector<cl_platform_id> dev_plats;
    for (auto pid : plats) { cl_uint nd = 0; clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd); if (!nd) continue; size_t old = devs.size(); devs.resize(old + nd); dev_plats.resize(old + nd); clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, nd, devs.data() + old, nullptr); for (cl_uint i=0;i<nd;i++) dev_plats[old+i]=pid; }
    if (devs.empty()) { std::cerr << "No OpenCL GPU device found\n"; return -1; }
    size_t idx = (size_t)options.device_id; if (idx >= devs.size()) idx = 0;
    cl_device_id dev = devs[idx]; cl_platform_id plat = dev_plats[idx];
    auto get_str = [&](cl_device_info p){ size_t sz=0; clGetDeviceInfo(dev,p,0,nullptr,&sz); std::string s(sz,'\0'); if(sz) clGetDeviceInfo(dev,p,sz,s.data(),nullptr); if(!s.empty()&&s.back()=='\0') s.pop_back(); return s; };
    std::string gpu_name=get_str(CL_DEVICE_NAME), gpu_vendor=get_str(CL_DEVICE_VENDOR), driver_ver=get_str(CL_DRIVER_VERSION), ver=get_str(CL_DEVICE_VERSION);
    cl_bool ecc_b = CL_FALSE; clGetDeviceInfo(dev, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(ecc_b), &ecc_b, nullptr);
    cl_uint cu=0, freq=0; cl_ulong vram=0, maxalloc=0, gcache=0, lmem=0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram), &vram, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxalloc), &maxalloc, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(gcache), &gcache, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
    std::cout << "OpenCL GPU " << gpu_vendor << " " << gpu_name << " | Driver: " << driver_ver << " | " << ver << " | CUs: " << cu << " | Freq: " << freq << " MHz | VRAM: " << (vram/(1024*1024)) << " MB | MaxAlloc: " << (maxalloc/(1024*1024)) << " MB | GCache: " << (gcache/1024) << " KB | Local: " << (lmem/1024) << " KB | ECC: " << (ecc_b? "yes":"no") << "\n";
    if (guiServer_) { std::ostringstream oss; oss << "OpenCL GPU " << gpu_vendor << " " << gpu_name << " | Driver: " << driver_ver << " | " << ver << " | CUs: " << cu << " | Freq: " << freq << " MHz | VRAM: " << (vram/(1024*1024)) << " MB | MaxAlloc: " << (maxalloc/(1024*1024)) << " MB | GCache: " << (gcache/1024) << " KB | Local: " << (lmem/1024) << " KB | ECC: " << (ecc_b? "yes":"no"); guiServer_->appendLog(oss.str()); }
    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0 };
    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(props, 1, &dev, nullptr, nullptr, &err); if (err != CL_SUCCESS) { std::cerr << "clCreateContext failed: " << err << "\n"; return -1; }
#if defined(CL_VERSION_2_0)
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, &err);
#endif
    if (err != CL_SUCCESS) { std::cerr << "clCreateCommandQueue failed: " << err << "\n"; clReleaseContext(ctx); return -1; }
    const char* src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define TYPE ulong
#define MAX_ERR_RECORD_COUNT 64
#define MOD_SZ 20
inline uint atomic_inc_u32(volatile __global uint* p){ return atomic_inc((volatile __global int*)p); }
#define RECORD_ERR(errc, p, expect, current) do{ uint idx = atomic_inc_u32(errc); idx = idx % MAX_ERR_RECORD_COUNT; err_addr[idx] = (ulong)(p); err_expect[idx] = (ulong)(expect); err_current[idx] = (ulong)(current); err_second_read[idx] = (ulong)(*(p)); }while(0)
__kernel void kernel_write(__global char* ptr, ulong memsize, TYPE p1){
  __global TYPE* buf = (__global TYPE*)ptr; ulong n = memsize/sizeof(TYPE); size_t idx=get_global_id(0), stride=get_global_size(0);
  for (ulong i=idx;i<n;i+=stride) buf[i]=p1;
}
__kernel void kernel_readwrite(__global char* ptr, ulong memsize, TYPE p1, TYPE p2, volatile __global uint* err_count, __global ulong* err_addr, __global ulong* err_expect, __global ulong* err_current, __global ulong* err_second_read){
  __global TYPE* buf = (__global TYPE*)ptr; ulong n = memsize/sizeof(TYPE); size_t idx=get_global_id(0), stride=get_global_size(0);
  for (ulong i=idx;i<n;i+=stride){ TYPE v = buf[i]; if (v!=p1) RECORD_ERR(err_count, &buf[i], p1, v); buf[i]=p2; }
}
__kernel void kernel_read(__global char* ptr, ulong memsize, TYPE p1, volatile __global uint* err_count, __global ulong* err_addr, __global ulong* err_expect, __global ulong* err_current, __global ulong* err_second_read){
  __global TYPE* buf = (__global TYPE*)ptr; ulong n = memsize/sizeof(TYPE); size_t idx=get_global_id(0), stride=get_global_size(0);
  for (ulong i=idx;i<n;i+=stride){ TYPE v = buf[i]; if (v!=p1) RECORD_ERR(err_count, &buf[i], p1, v); }
}
__kernel void kernel1_write(__global char* ptr, ulong memsize){
  __global TYPE* buf = (__global TYPE*)ptr; ulong n = memsize/sizeof(TYPE); size_t idx=get_global_id(0), stride=get_global_size(0);
  for (ulong i=idx;i<n;i+=stride){ buf[i]=(TYPE)(i*sizeof(TYPE)); }
}
__kernel void kernel1_read(__global char* ptr, ulong memsize, volatile __global uint* err_count, __global ulong* err_addr, __global ulong* err_expect, __global ulong* err_current, __global ulong* err_second_read){
  __global TYPE* buf = (__global TYPE*)ptr; ulong n = memsize/sizeof(TYPE); size_t idx=get_global_id(0), stride=get_global_size(0);
  for (ulong i=idx;i<n;i+=stride){ TYPE e=(TYPE)(i*sizeof(TYPE)); TYPE v=buf[i]; if (v!=e) RECORD_ERR(err_count,&buf[i],e,v); }
}
__kernel void kernel_modtest_write(__global char* ptr, ulong memsize, uint offset, TYPE p1, TYPE p2){
  __global TYPE* buf = (__global TYPE*)ptr; ulong n = memsize/sizeof(TYPE); size_t idx=get_global_id(0), stride=get_global_size(0);
  for (ulong i=idx;i<n;i+=stride){ if (((i+MOD_SZ-offset)%MOD_SZ)==0) buf[i]=p1; else buf[i]=p2; }
}
__kernel void kernel_modtest_read(__global char* ptr, ulong memsize, uint offset, TYPE p1, TYPE p2, volatile __global uint* err_count, __global ulong* err_addr, __global ulong* err_expect, __global ulong* err_current, __global ulong* err_second_read){
  __global TYPE* buf = (__global TYPE*)ptr; ulong n = memsize/sizeof(TYPE); size_t idx=get_global_id(0), stride=get_global_size(0);
  for (ulong i=idx;i<n;i+=stride){ TYPE v=buf[i]; int m=((i+MOD_SZ-offset)%MOD_SZ)==0; TYPE e=m?p1:p2; if (v!=e) RECORD_ERR(err_count,&buf[i],e,v); }
}
)CLC";
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err); if (err != CL_SUCCESS) { std::cerr << "clCreateProgramWithSource failed: " << err << "\n"; clReleaseCommandQueue(queue); clReleaseContext(ctx); return -1; }
    const char* opts = "-cl-std=CL1.2";
    err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz=0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::string log(logsz,'\0'); if(logsz) clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << "clBuildProgram failed: " << err << "\n" << log << "\n";
        clReleaseProgram(prog); clReleaseCommandQueue(queue); clReleaseContext(ctx); return -1;
    }
    cl_mem err_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    cl_mem err_addr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_ulong)*64, nullptr, &err);
    cl_mem err_expect = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_ulong)*64, nullptr, &err);
    cl_mem err_current = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_ulong)*64, nullptr, &err);
    cl_mem err_second = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_ulong)*64, nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "error buffers alloc failed\n"; clReleaseProgram(prog); clReleaseCommandQueue(queue); clReleaseContext(ctx); return -1; }
    auto zero_err = [&](){ cl_uint z=0; clEnqueueWriteBuffer(queue, err_count, CL_TRUE, 0, sizeof(z), &z, 0, nullptr, nullptr); };
    cl_kernel k_write = clCreateKernel(prog, "kernel_write", &err);
    cl_kernel k_rw = clCreateKernel(prog, "kernel_readwrite", &err);
    cl_kernel k_read = clCreateKernel(prog, "kernel_read", &err);
    cl_kernel k_a_w = clCreateKernel(prog, "kernel1_write", &err);
    cl_kernel k_a_r = clCreateKernel(prog, "kernel1_read", &err);
    cl_kernel k_m_w = clCreateKernel(prog, "kernel_modtest_write", &err);
    cl_kernel k_m_r = clCreateKernel(prog, "kernel_modtest_read", &err);
    size_t gws[1] = { 64*1024 }, lws[1] = { 64 };
    auto now = std::chrono::high_resolution_clock::now;
    auto dur = [&](auto t0, auto t1){ return std::chrono::duration<double>(t1-t0).count(); };
    auto fmt_time = [&](double s){ int h=(int)(s/3600.0); s-=h*3600.0; int m=(int)(s/60.0); s-=m*60.0; std::ostringstream o; o<<std::setfill('0')<<std::setw(2)<<h<<":"<<std::setw(2)<<m<<":"<<std::setw(2)<<(int)s; return o.str(); };
    uint64_t total_read=0, total_write=0;
    double addr_w_time=0.0, addr_r_time=0.0, inv_w_time=0.0, inv_rw_time=0.0, inv_r_time=0.0, mod_w_time=0.0, mod_r_time=0.0;
    uint64_t addr_err_total=0, inv_err_total=0, mod_err_total=0;
    size_t addr_bytes_w=0, addr_bytes_r=0, inv_bytes_w=0, inv_bytes_rw_r=0, inv_bytes_rw_w=0, inv_bytes_r=0, mod_bytes_w=0, mod_bytes_r=0;
    const size_t align = 1024*1024;
    std::vector<cl_mem> bufs;
    std::vector<size_t> bsz;
    size_t covered = 0;
    size_t target = (size_t)vram;
    size_t margin = 32ull*1024*1024;
    if (target > margin) target -= margin;
    size_t remain = target;
    size_t chunk = (size_t)maxalloc;
    if (chunk == 0) chunk = remain;
    while (remain > 0) {
        size_t try_sz = std::min(remain, chunk);
        try_sz = (try_sz/align)*align;
        if (try_sz == 0) break;
        cl_int e2 = CL_SUCCESS;
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, try_sz, nullptr, &e2);
        if (e2 == CL_SUCCESS && m) { bufs.push_back(m); bsz.push_back(try_sz); covered += try_sz; remain -= try_sz; }
        else {
            if (chunk <= align) break;
            chunk = (chunk/2/align)*align;
            if (chunk == 0) chunk = align;
        }
    }
    uint64_t seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ 0x9E3779B97F4A7C15ull;
    auto splitmix64 = [&](uint64_t& x)->uint64_t{ x += 0x9E3779B97f4A7C15ULL; uint64_t z=x; z^=z>>30; z*=0xBF58476D1CE4E5B9ULL; z^=z>>27; z*=0x94D049BB133111EBULL; z^=z>>31; return z; };
    auto rnd64 = [&](){ return (cl_ulong)splitmix64(seed); };
    const uint32_t MOD_SZ_HOST = 20;
    const int iters = 128;
    size_t total_sectors_all = 0;
    for (size_t bi = 0; bi < bufs.size(); ++bi) {
        size_t want = bsz[bi];
        size_t sector = 256ull*1024*1024; if (sector > want) sector = want; sector = (sector/align)*align; if (sector == 0) sector = align;
        size_t sectors = (want + sector - 1) / sector;
        total_sectors_all += sectors;
    }
    uint64_t ops_per_sector = (uint64_t)iters + 4 + 2*MOD_SZ_HOST;
    uint64_t total_ops = total_sectors_all * ops_per_sector;
    uint64_t steps_done = 0;
    auto t_start = now();
    auto show = [&](const std::string& phase, size_t bi, size_t sectors, size_t sidx, int inner, int innerMax){
        double pct = total_ops ? (100.0 * (double)steps_done / (double)total_ops) : 0.0;
        double elapsed = dur(t_start, now());
        double eta = pct>0.0 ? elapsed * (100.0/pct - 1.0) : 0.0;
        std::ostringstream line;
        line << "\r[" << std::fixed << std::setprecision(1) << pct << "%] buf " << (bi+1) << "/" << bufs.size() << " sec " << (sidx+1) << "/" << sectors << " " << phase;
        if (innerMax>0) line << " " << inner << "/" << innerMax;
        line << " ETA " << fmt_time(eta) << "   ";
        std::cout << line.str() << std::flush;
        if (guiServer_) guiServer_->setStatus(line.str());
    };
    struct ErrRec { uint32_t test; uint32_t buf; uint32_t sec; uint32_t offmod; cl_ulong addr, exp, cur, reread; };
    std::vector<ErrRec> samples; samples.reserve(128);
    static volatile std::sig_atomic_t stop_flag = 0;
    auto onint = +[](int){ stop_flag = 1; };
    std::signal(SIGINT, onint);
#ifdef SIGTERM
    std::signal(SIGTERM, onint);
#endif
    bool interrupted = false;
    for (size_t bi = 0; bi < bufs.size() && !interrupted; ++bi) {
        cl_mem parent = bufs[bi];
        size_t want = bsz[bi];
        size_t sector = 256ull*1024*1024; if (sector > want) sector = want; sector = (sector/align)*align; if (sector == 0) sector = align;
        size_t sectors = (want + sector - 1) / sector;
        std::cout << "Buffer " << (bi+1) << "/" << bufs.size() << " sectors: " << sectors << " x " << (sector/(1024*1024)) << " MB\n";
        if (guiServer_) { std::ostringstream oss; oss << "Buffer " << (bi+1) << "/" << bufs.size() << " sectors: " << sectors << " x " << (sector/(1024*1024)) << " MB"; guiServer_->appendLog(oss.str()); }
        for (size_t s=0; s<sectors && !interrupted; ++s) {
            size_t off = s*sector; size_t sz = std::min(sector, want - off);
            cl_buffer_region reg; reg.origin = off; reg.size = sz;
            cl_int err2=CL_SUCCESS;
            cl_mem sub = clCreateSubBuffer(parent, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err2);
            if (err2 != CL_SUCCESS) { std::cerr << "clCreateSubBuffer failed: " << err2 << "\n"; break; }
            zero_err();
            auto t0 = now();
            err = clSetKernelArg(k_a_w, 0, sizeof(cl_mem), &sub); err|=clSetKernelArg(k_a_w,1,sizeof(cl_ulong),&sz);
            show("addr W", bi, sectors, s, -1, 0);
            err|=clEnqueueNDRangeKernel(queue,k_a_w,1,nullptr,gws,lws,0,nullptr,nullptr);
            err|=clFinish(queue);
            auto t1 = now();
            addr_w_time += dur(t0,t1);
            total_write += sz; addr_bytes_w += sz;
            steps_done++; show("addr W", bi, sectors, s, -1, 0);
            if (stop_flag) { interrupted = true; clReleaseMemObject(sub); break; }
            err|=clSetKernelArg(k_a_r,0,sizeof(cl_mem),&sub); err|=clSetKernelArg(k_a_r,1,sizeof(cl_ulong),&sz);
            err|=clSetKernelArg(k_a_r,2,sizeof(cl_mem),&err_count); err|=clSetKernelArg(k_a_r,3,sizeof(cl_mem),&err_addr);
            err|=clSetKernelArg(k_a_r,4,sizeof(cl_mem),&err_expect); err|=clSetKernelArg(k_a_r,5,sizeof(cl_mem),&err_current);
            err|=clSetKernelArg(k_a_r,6,sizeof(cl_mem),&err_second);
            show("addr R", bi, sectors, s, -1, 0);
            err|=clEnqueueNDRangeKernel(queue,k_a_r,1,nullptr,gws,lws,0,nullptr,nullptr);
            err|=clFinish(queue);
            auto t2 = now();
            addr_r_time += dur(t1,t2);
            total_read += sz; addr_bytes_r += sz;
            cl_uint ec=0; clEnqueueReadBuffer(queue, err_count, CL_TRUE, 0, sizeof(ec), &ec, 0, nullptr, nullptr);
            addr_err_total += ec;
            if (ec) {
                std::vector<cl_ulong> addr(64), ex(64), cuv(64), sec2(64);
                clEnqueueReadBuffer(queue, err_addr, CL_TRUE, 0, addr.size()*sizeof(cl_ulong), addr.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(queue, err_expect, CL_TRUE, 0, ex.size()*sizeof(cl_ulong), ex.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(queue, err_current, CL_TRUE, 0, cuv.size()*sizeof(cl_ulong), cuv.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(queue, err_second, CL_TRUE, 0, sec2.size()*sizeof(cl_ulong), sec2.data(), 0, nullptr, nullptr);
                for (size_t i=0;i<std::min<size_t>(ec,addr.size());++i) if (samples.size()<samples.capacity()) samples.push_back({0,(uint32_t)bi,(uint32_t)s,0,addr[i],ex[i],cuv[i],sec2[i]});
            }
            steps_done++; show("addr R", bi, sectors, s, -1, 0);
            if (stop_flag) { interrupted = true; clReleaseMemObject(sub); break; }
            zero_err();
            cl_ulong p1 = rnd64(); cl_ulong p2 = ~p1;
            err = clSetKernelArg(k_write,0,sizeof(cl_mem),&sub); err|=clSetKernelArg(k_write,1,sizeof(cl_ulong),&sz); err|=clSetKernelArg(k_write,2,sizeof(cl_ulong),&p1);
            auto t3 = now();
            show("inv W", bi, sectors, s, -1, 0);
            err|=clEnqueueNDRangeKernel(queue,k_write,1,nullptr,gws,lws,0,nullptr,nullptr); err|=clFinish(queue);
            auto t4 = now();
            inv_w_time += dur(t3,t4);
            total_write += sz; inv_bytes_w += sz;
            steps_done++; show("inv W", bi, sectors, s, -1, 0);
            if (stop_flag) { interrupted = true; clReleaseMemObject(sub); break; }
            for (int it=0; it<iters; ++it) {
                auto trw0 = now();
                err|=clSetKernelArg(k_rw,0,sizeof(cl_mem),&sub); err|=clSetKernelArg(k_rw,1,sizeof(cl_ulong),&sz);
                err|=clSetKernelArg(k_rw,2,sizeof(cl_ulong),&p1); err|=clSetKernelArg(k_rw,3,sizeof(cl_ulong),&p2);
                err|=clSetKernelArg(k_rw,4,sizeof(cl_mem),&err_count); err|=clSetKernelArg(k_rw,5,sizeof(cl_mem),&err_addr);
                err|=clSetKernelArg(k_rw,6,sizeof(cl_mem),&err_expect); err|=clSetKernelArg(k_rw,7,sizeof(cl_mem),&err_current);
                err|=clSetKernelArg(k_rw,8,sizeof(cl_mem),&err_second);
                if ((it & 7) == 0) show("inv RW", bi, sectors, s, it, iters);
                err|=clEnqueueNDRangeKernel(queue,k_rw,1,nullptr,gws,lws,0,nullptr,nullptr); err|=clFinish(queue);
                auto trw1 = now();
                inv_rw_time += dur(trw0,trw1);
                total_read += sz; total_write += sz; inv_bytes_rw_r += sz; inv_bytes_rw_w += sz;
                cl_uint ec_it=0; clEnqueueReadBuffer(queue, err_count, CL_TRUE, 0, sizeof(ec_it), &ec_it, 0, nullptr, nullptr);
                if (ec_it) {
                    inv_err_total += ec_it;
                    std::vector<cl_ulong> addr(64), ex(64), cuv(64), sec2(64);
                    clEnqueueReadBuffer(queue, err_addr, CL_TRUE, 0, addr.size()*sizeof(cl_ulong), addr.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(queue, err_expect, CL_TRUE, 0, ex.size()*sizeof(cl_ulong), ex.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(queue, err_current, CL_TRUE, 0, cuv.size()*sizeof(cl_ulong), cuv.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(queue, err_second, CL_TRUE, 0, sec2.size()*sizeof(cl_ulong), sec2.data(), 0, nullptr, nullptr);
                    for (size_t i=0;i<std::min<size_t>(ec_it,addr.size());++i) if (samples.size()<samples.capacity()) samples.push_back({1,(uint32_t)bi,(uint32_t)s,0,addr[i],ex[i],cuv[i],sec2[i]});
                    zero_err();
                }
                cl_ulong t = p1; p1 = p2; p2 = t;
                steps_done++; if ((it & 7) == 7) show("inv RW", bi, sectors, s, it+1, iters);
                if (stop_flag) { interrupted = true; break; }
            }
            if (interrupted) { clReleaseMemObject(sub); break; }
            err|=clSetKernelArg(k_read,0,sizeof(cl_mem),&sub); err|=clSetKernelArg(k_read,1,sizeof(cl_ulong),&sz); err|=clSetKernelArg(k_read,2,sizeof(cl_ulong),&p1);
            err|=clSetKernelArg(k_read,3,sizeof(cl_mem),&err_count); err|=clSetKernelArg(k_read,4,sizeof(cl_mem),&err_addr);
            err|=clSetKernelArg(k_read,5,sizeof(cl_mem),&err_expect); err|=clSetKernelArg(k_read,6,sizeof(cl_mem),&err_current);
            err|=clSetKernelArg(k_read,7,sizeof(cl_mem),&err_second);
            auto t5 = now();
            show("inv R", bi, sectors, s, -1, 0);
            err|=clEnqueueNDRangeKernel(queue,k_read,1,nullptr,gws,lws,0,nullptr,nullptr); err|=clFinish(queue);
            auto t6 = now();
            inv_r_time += dur(t5,t6);
            total_read += sz; inv_bytes_r += sz;
            cl_uint ec2=0; clEnqueueReadBuffer(queue, err_count, CL_TRUE, 0, sizeof(ec2), &ec2, 0, nullptr, nullptr);
            inv_err_total += ec2;
            if (ec2) {
                std::vector<cl_ulong> addr(64), ex(64), cuv(64), sec2(64);
                clEnqueueReadBuffer(queue, err_addr, CL_TRUE, 0, addr.size()*sizeof(cl_ulong), addr.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(queue, err_expect, CL_TRUE, 0, ex.size()*sizeof(cl_ulong), ex.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(queue, err_current, CL_TRUE, 0, cuv.size()*sizeof(cl_ulong), cuv.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(queue, err_second, CL_TRUE, 0, sec2.size()*sizeof(cl_ulong), sec2.data(), 0, nullptr, nullptr);
                for (size_t i=0;i<std::min<size_t>(ec2,addr.size());++i) if (samples.size()<samples.capacity()) samples.push_back({1,(uint32_t)bi,(uint32_t)s,0,addr[i],ex[i],cuv[i],sec2[i]});
            }
            steps_done++; show("inv R", bi, sectors, s, -1, 0);
            if (stop_flag) { interrupted = true; clReleaseMemObject(sub); break; }
            zero_err();
            cl_ulong mp1 = 0xAAAAAAAAAAAAAAAAull, mp2 = 0x5555555555555555ull;
            for (uint32_t offmod=0; offmod<MOD_SZ_HOST && !interrupted; ++offmod) {
                err = clSetKernelArg(k_m_w,0,sizeof(cl_mem),&sub); err|=clSetKernelArg(k_m_w,1,sizeof(cl_ulong),&sz); err|=clSetKernelArg(k_m_w,2,sizeof(cl_uint),&offmod); err|=clSetKernelArg(k_m_w,3,sizeof(cl_ulong),&mp1); err|=clSetKernelArg(k_m_w,4,sizeof(cl_ulong),&mp2);
                auto t7 = now();
                show("mod W", bi, sectors, s, static_cast<int>(offmod + 1u), MOD_SZ_HOST);
                err|=clEnqueueNDRangeKernel(queue,k_m_w,1,nullptr,gws,lws,0,nullptr,nullptr); err|=clFinish(queue);
                auto t8 = now();
                mod_w_time += dur(t7,t8);
                total_write += sz; mod_bytes_w += sz;
                steps_done++; show("mod W", bi, sectors, s, static_cast<int>(offmod + 1u), MOD_SZ_HOST);
                if (stop_flag) { interrupted = true; break; }
                err|=clSetKernelArg(k_m_r,0,sizeof(cl_mem),&sub); err|=clSetKernelArg(k_m_r,1,sizeof(cl_ulong),&sz); err|=clSetKernelArg(k_m_r,2,sizeof(cl_uint),&offmod);
                err|=clSetKernelArg(k_m_r,3,sizeof(cl_ulong),&mp1); err|=clSetKernelArg(k_m_r,4,sizeof(cl_ulong),&mp2);
                err|=clSetKernelArg(k_m_r,5,sizeof(cl_mem),&err_count); err|=clSetKernelArg(k_m_r,6,sizeof(cl_mem),&err_addr);
                err|=clSetKernelArg(k_m_r,7,sizeof(cl_mem),&err_expect); err|=clSetKernelArg(k_m_r,8,sizeof(cl_mem),&err_current);
                err|=clSetKernelArg(k_m_r,9,sizeof(cl_mem),&err_second);
                auto t9 = now();
                show("mod R", bi, sectors, s, static_cast<int>(offmod + 1u), MOD_SZ_HOST);
                err|=clEnqueueNDRangeKernel(queue,k_m_r,1,nullptr,gws,lws,0,nullptr,nullptr); err|=clFinish(queue);
                auto t10 = now();
                mod_r_time += dur(t9,t10);
                total_read += sz; mod_bytes_r += sz;
                cl_uint ec3=0; clEnqueueReadBuffer(queue, err_count, CL_TRUE, 0, sizeof(ec3), &ec3, 0, nullptr, nullptr);
                mod_err_total += ec3;
                if (ec3) {
                    std::vector<cl_ulong> addr(64), ex(64), cuv(64), sec2(64);
                    clEnqueueReadBuffer(queue, err_addr, CL_TRUE, 0, addr.size()*sizeof(cl_ulong), addr.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(queue, err_expect, CL_TRUE, 0, ex.size()*sizeof(cl_ulong), ex.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(queue, err_current, CL_TRUE, 0, cuv.size()*sizeof(cl_ulong), cuv.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(queue, err_second, CL_TRUE, 0, sec2.size()*sizeof(cl_ulong), sec2.data(), 0, nullptr, nullptr);
                    for (size_t i=0;i<std::min<size_t>(ec3,addr.size());++i) if (samples.size()<samples.capacity()) samples.push_back({2,(uint32_t)bi,(uint32_t)s,offmod,addr[i],ex[i],cuv[i],sec2[i]});
                    zero_err();
                }
                steps_done++; show("mod R", bi, sectors, s, static_cast<int>(offmod + 1u), MOD_SZ_HOST);
                if (stop_flag) { interrupted = true; break; }
                zero_err();
            }
            clReleaseMemObject(sub);
        }
    }
    std::cout << "\n";
    //double read_time = addr_r_time + inv_rw_time + inv_r_time + mod_r_time;
    //double write_time = addr_w_time + inv_w_time + inv_rw_time + mod_w_time;
    auto gb = [&](double bytes){ return bytes/1073741824.0; };
    double addr_w_bw = addr_w_time>0 ? gb((double)addr_bytes_w)/addr_w_time : 0.0;
    double addr_r_bw = addr_r_time>0 ? gb((double)addr_bytes_r)/addr_r_time : 0.0;
    double inv_w_bw  = inv_w_time>0  ? gb((double)inv_bytes_w)/inv_w_time   : 0.0;
    double inv_rw_bw = inv_rw_time>0 ? gb((double)inv_bytes_rw_r + (double)inv_bytes_rw_w)/inv_rw_time : 0.0;
    double inv_r_bw  = inv_r_time>0  ? gb((double)inv_bytes_r)/inv_r_time   : 0.0;
    double mod_w_bw  = mod_w_time>0  ? gb((double)mod_bytes_w)/mod_w_time   : 0.0;
    double mod_r_bw  = mod_r_time>0  ? gb((double)mod_bytes_r)/mod_r_time   : 0.0;
    size_t tested_mb = covered/(1024*1024);
    size_t vram_mb = (size_t)(vram/(1024*1024));
    double coverage_pct = vram_mb ? (100.0 * (double)tested_mb / (double)vram_mb) : 0.0;
    uint64_t total_err = addr_err_total + inv_err_total + mod_err_total;
    double traffic_gb = gb((double)(total_read + total_write));
    double err_per_gb = traffic_gb>0 ? (double)total_err/traffic_gb : 0.0;
    std::ostringstream rpt;
    rpt << "\n===== GPU Memtest Report =====\n";
    rpt << "Device: " << gpu_vendor << " " << gpu_name << " | Driver " << driver_ver << " | " << ver << " | CUs " << cu << " | " << freq << " MHz | ECC " << (ecc_b? "yes":"no") << "\n";
    rpt << "VRAM: " << vram_mb << " MB | MaxAlloc: " << (size_t)(maxalloc/(1024*1024)) << " MB\n";
    rpt << "Coverage: " << tested_mb << " MB (" << std::fixed << std::setprecision(1) << coverage_pct << "% of VRAM) across " << bufs.size() << " buffers\n";
    rpt << "Plan: address pattern, inversion toggles x" << iters << ", modulo-stride pattern/" << MOD_SZ_HOST << " offsets\n";
    rpt << "Traffic: Read " << std::fixed << std::setprecision(2) << gb((double)total_read) << " GB, Write " << gb((double)total_write) << " GB, Total " << gb((double)(total_read+total_write)) << " GB\n";
    rpt << "Address  W " << std::setprecision(3) << addr_w_bw << " GB/s (" << gb((double)addr_bytes_w) << " GB in " << addr_w_time << " s)  R " << addr_r_bw << " GB/s (" << gb((double)addr_bytes_r) << " GB in " << addr_r_time << " s)  Err " << addr_err_total << "\n";
    rpt << "Invert   W " << inv_w_bw  << " GB/s (" << gb((double)inv_bytes_w)    << " GB in " << inv_w_time  << " s)  RW " << inv_rw_bw << " GB/s (" << gb((double)(inv_bytes_rw_r+inv_bytes_rw_w)) << " GB in " << inv_rw_time << " s)  R " << inv_r_bw << " GB/s (" << gb((double)inv_bytes_r) << " GB in " << inv_r_time << " s)  Err " << inv_err_total << "\n";
    rpt << "Modulo   W " << mod_w_bw  << " GB/s (" << gb((double)mod_bytes_w)    << " GB in " << mod_w_time  << " s)  R " << mod_r_bw  << " GB/s (" << gb((double)mod_bytes_r) << " GB in " << mod_r_time << " s)  Err " << mod_err_total << "\n";
    rpt << "Totals:  Errors " << total_err << "  |  Err/GB " << std::setprecision(4) << err_per_gb << "\n";
    if (!samples.empty()) {
        rpt << "Sample errors (" << samples.size() << "):\n";
        for (size_t i=0;i<std::min<size_t>(samples.size(),16);++i) {
            const auto& e = samples[i];
            const char* t = (e.test==0?"ADDR":(e.test==1?"INVT":"MOD "));
            rpt << "  [" << t;
            if (e.test==2) rpt << " off=" << e.offmod;
            rpt << " buf=" << (e.buf+1) << " sec=" << (e.sec+1) << "] addr=0x" << std::hex << e.addr << " exp=0x" << e.exp << " cur=0x" << e.cur << " reread=0x" << e.reread << std::dec << "\n";
        }
    } else {
        rpt << "No sample errors captured.\n";
    }
    if (stop_flag || interrupted) rpt << "Status: interrupted by signal, partial results shown above.\n";
    rpt << "==============================\n";
    std::cout << rpt.str();
    if (guiServer_) guiServer_->appendLog(rpt.str());
    for (auto m : bufs) clReleaseMemObject(m);
    clReleaseKernel(k_write); clReleaseKernel(k_rw); clReleaseKernel(k_read); clReleaseKernel(k_a_w); clReleaseKernel(k_a_r); clReleaseKernel(k_m_w); clReleaseKernel(k_m_r);
    clReleaseMemObject(err_count); clReleaseMemObject(err_addr); clReleaseMemObject(err_expect); clReleaseMemObject(err_current); clReleaseMemObject(err_second);
    clReleaseProgram(prog); clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return interrupted ? 1 : 0;
}