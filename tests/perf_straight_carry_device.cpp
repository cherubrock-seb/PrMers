// Source-only differential fixture for the legacy four-digit carry experiment.
// IMPLEMENTED — NOT COMPILED — NOT GPU TESTED — NOT BENCHMARKED
// Usage: perf-straight-carry-device [GPU-index] [kernels/prmers.cl]
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(err));
    }
}

int main(int argc, char** argv) {
    try {
        const size_t selected = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 0;
        const char* path = argc > 2 ? argv[2] : "kernels/prmers.cl";
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot read kernel source");
        const std::string source((std::istreambuf_iterator<char>(file)), {});
        const size_t begin = source.find("#ifndef PRMERS_CARRY_STRAIGHT_V1\n");
        const size_t end = source.find("inline ulong4 digit_adc4_last(", begin);
        if (begin == std::string::npos || end == std::string::npos) {
            throw std::runtime_error("Expected guarded carry helper not found");
        }
        const std::string helper = source.substr(begin, end - begin);
        std::string code;
        for (int enabled = 0; enabled <= 1; ++enabled) {
            const std::string name = enabled ? "candidate" : "original";
            code += "#define PRMERS_CARRY_STRAIGHT_V1 " + std::to_string(enabled) + "\n";
            code += "#define digit_adc4 digit_adc4_" + name + "\n";
            code += helper;
            code += "\n#undef digit_adc4\n#undef PRMERS_CARRY_STRAIGHT_V1\n";
            code += "__kernel void eval_" + name +
                "(__global const ulong4* x, __global const int4* w,"
                " __global const ulong* seeds, __global ulong* out) {\n"
                "size_t i=get_global_id(0); ulong c=seeds[i];\n"
                "ulong4 r=digit_adc4_" + name + "(x[i],w[i],&c);\n"
                "vstore4(r,0,out+5*i); out[5*i+4]=c;\n}\n";
        }

        cl_uint platformCount = 0;
        check(clGetPlatformIDs(0, nullptr, &platformCount), "platform count");
        std::vector<cl_platform_id> platforms(platformCount);
        check(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "platform list");
        std::vector<cl_device_id> devices;
        for (cl_platform_id platform : platforms) {
            cl_uint count = 0;
            cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
            if (err == CL_DEVICE_NOT_FOUND) continue;
            check(err, "device count");
            std::vector<cl_device_id> found(count);
            check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, found.data(), nullptr), "device list");
            devices.insert(devices.end(), found.begin(), found.end());
        }
        if (selected >= devices.size()) throw std::runtime_error("GPU index unavailable");
        const cl_device_id device = devices[selected];
        cl_int err = CL_SUCCESS;
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check(err, "create context");
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        check(err, "create queue");
        const char* text = code.c_str();
        const size_t length = code.size();
        cl_program program = clCreateProgramWithSource(context, 1, &text, &length, &err);
        check(err, "create program");
        err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t size = 0;
            check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size), "log size");
            std::vector<char> log(size + 1, 0);
            check(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr), "build log");
            std::cerr << log.data() << '\n';
            check(err, "build program");
        }

        std::vector<cl_ulong> inputs, seeds;
        std::vector<cl_int> widths;
        std::mt19937_64 random(0x8d634ad1ULL);
        const int pairs[][2] = {{1,2}, {8,9}, {26,27}, {31,32}, {62,63}};
        const cl_ulong incoming[] = {0, 1, cl_ulong(1) << 63, ~cl_ulong(0)};
        for (const auto& pair : pairs) {
            for (unsigned pattern = 0; pattern < 16; ++pattern) {
                for (int sample = 0; sample < 13; ++sample) {
                    for (cl_ulong seed : incoming) {
                        seeds.push_back(seed);
                        for (unsigned lane = 0; lane < 4; ++lane) {
                            const int w = pair[(pattern >> lane) & 1];
                            widths.push_back(w);
                            cl_ulong x = 0;
                            if (sample == 1) x = ~cl_ulong(0);
                            else if (sample == 2) x = lane & 1 ? ~cl_ulong(0) : 0;
                            else if (sample == 3) x = (cl_ulong(1) << w) - 1;
                            else if (sample == 4) x = cl_ulong(1) << w;
                            else if (sample >= 5) x = random();
                            inputs.push_back(x);
                        }
                    }
                }
            }
        }
        const size_t count = seeds.size();
        auto buffer = [&](size_t bytes, void* data) {
            cl_mem result = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          bytes, data, &err);
            check(err, "input buffer");
            return result;
        };
        cl_mem x = buffer(inputs.size() * sizeof(cl_ulong), inputs.data());
        cl_mem w = buffer(widths.size() * sizeof(cl_int), widths.data());
        cl_mem c = buffer(seeds.size() * sizeof(cl_ulong), seeds.data());
        const size_t outputBytes = count * 5 * sizeof(cl_ulong);
        cl_mem out = clCreateBuffer(context, CL_MEM_READ_WRITE, outputBytes, nullptr, &err);
        check(err, "output buffer");
        std::vector<cl_ulong> results[2];
        const char* kernels[] = {"eval_original", "eval_candidate"};
        for (int mode = 0; mode < 2; ++mode) {
            cl_kernel kernel = clCreateKernel(program, kernels[mode], &err);
            check(err, "create kernel");
            check(clSetKernelArg(kernel, 0, sizeof(x), &x), "arg x");
            check(clSetKernelArg(kernel, 1, sizeof(w), &w), "arg widths");
            check(clSetKernelArg(kernel, 2, sizeof(c), &c), "arg seeds");
            check(clSetKernelArg(kernel, 3, sizeof(out), &out), "arg output");
            check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &count, nullptr, 0, nullptr, nullptr), "enqueue");
            results[mode].resize(count * 5);
            check(clEnqueueReadBuffer(queue, out, CL_TRUE, 0, outputBytes, results[mode].data(),
                                      0, nullptr, nullptr), "read output");
            check(clReleaseKernel(kernel), "release kernel");
        }
        size_t mismatches = 0;
        for (size_t i = 0; i < results[0].size(); ++i) {
            if (results[0][i] != results[1][i]) {
                if (mismatches < 8) {
                    std::cerr << "case=" << i / 5 << " component=" << i % 5
                              << " original=" << results[0][i] << " candidate=" << results[1][i] << '\n';
                }
                ++mismatches;
            }
        }
        std::cout << "cases=" << count << " compared_words=" << count * 5
                  << " mismatches=" << mismatches << '\n';
        check(clReleaseMemObject(x), "release x");
        check(clReleaseMemObject(w), "release widths");
        check(clReleaseMemObject(c), "release seeds");
        check(clReleaseMemObject(out), "release output");
        check(clReleaseProgram(program), "release program");
        check(clReleaseCommandQueue(queue), "release queue");
        check(clReleaseContext(context), "release context");
        return mismatches ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
