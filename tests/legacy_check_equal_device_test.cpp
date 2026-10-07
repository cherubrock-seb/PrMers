// Device test: legacy Gerbicz-Li check_equal kernel bounds.
//
// The host rounds the global size up to a multiple of the work-group size (n=320 -> 512, n=640 -> 768), so the
// kernel must ignore the tail work-items. The buffers here are larger than n and differ past n, so a kernel
// without the guard reports a false mismatch. Also checks that a real difference is detected, including at n-1,
// and that a failed enqueue is reported instead of ignored.
//
// Build/run: bash tests/test_legacy_check_equal_device.sh [device-index]
#define CL_TARGET_OPENCL_VERSION 120
#include "opencl/Kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef PRMERS_KERNEL_DIR
#define PRMERS_KERNEL_DIR "kernels"
#endif

static std::string check_equal_source()
{
	std::ifstream in(std::string(PRMERS_KERNEL_DIR) + "/prmers.cl");
	std::stringstream ss; ss << in.rdbuf();
	const std::string all = ss.str();
	// The atomic helpers and check_equal are the tail of prmers.cl and do not depend on any build defines.
	const std::string marker = "#if __OPENCL_VERSION__ < 200";
	const size_t pos = all.rfind(marker);
	if (pos == std::string::npos || all.find("__kernel void check_equal", pos) == std::string::npos) {
		std::fprintf(stderr, "cannot find check_equal in prmers.cl\n"); std::exit(2);
	}
	return all.substr(pos);
}

int main(int argc, char ** argv)
{
	const size_t index = (argc > 1) ? (size_t)std::strtoul(argv[1], nullptr, 10) : 0;

	cl_platform_id plats[16]; cl_uint np = 0;
	clGetPlatformIDs(16, plats, &np);
	std::vector<cl_device_id> devs;
	for (cl_uint p = 0; p < np; ++p) {
		cl_device_id d[16]; cl_uint nd = 0;
		if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 16, d, &nd) == CL_SUCCESS)
			for (cl_uint i = 0; i < nd; ++i) devs.push_back(d[i]);
	}
	if (index >= devs.size()) { std::fprintf(stderr, "no OpenCL device %zu\n", index); return 2; }
	cl_device_id dev = devs[index];

	cl_int err;
	cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
	cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
	const std::string src = check_equal_source();
	const char * sp = src.c_str(); size_t sl = src.size();
	cl_program prog = clCreateProgramWithSource(ctx, 1, &sp, &sl, &err);
	if (clBuildProgram(prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
		char log[8192] = {0}; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, nullptr);
		std::fprintf(stderr, "build failed:\n%s\n", log); return 2;
	}

	opencl::Kernels kernels(prog, q);
	kernels.createKernel("check_equal");

	const size_t cap = 2048;
	std::vector<cl_ulong> ha(cap), hb(cap);
	cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, cap * sizeof(cl_ulong), nullptr, &err);
	cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, cap * sizeof(cl_ulong), nullptr, &err);
	cl_mem okb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);

	int fails = 0, total = 0;
	auto run = [&](cl_uint n, int bad_index) {
		for (size_t i = 0; i < cap; ++i) { ha[i] = 0x1234 + i; hb[i] = (i < n) ? ha[i] : ~ha[i]; }
		if (bad_index >= 0) hb[bad_index] ^= 1;
		clEnqueueWriteBuffer(q, a, CL_TRUE, 0, cap * sizeof(cl_ulong), ha.data(), 0, nullptr, nullptr);
		clEnqueueWriteBuffer(q, b, CL_TRUE, 0, cap * sizeof(cl_ulong), hb.data(), 0, nullptr, nullptr);
		cl_uint ok = 1;
		clEnqueueWriteBuffer(q, okb, CL_TRUE, 0, sizeof(ok), &ok, 0, nullptr, nullptr);
		kernels.runCheckEqual(a, b, okb, n);
		clEnqueueReadBuffer(q, okb, CL_TRUE, 0, sizeof(ok), &ok, 0, nullptr, nullptr);
		const cl_uint expect = (bad_index >= 0) ? 0u : 1u;
		++total; if (ok != expect) ++fails;
		std::printf("n=%u bad_index=%d ok=%u expected=%u: %s\n", n, bad_index, ok, expect, ok == expect ? "ok" : "MISMATCH");
	};

	for (cl_uint n : {5u, 64u, 256u, 320u, 640u, 1000u, 1024u, 1536u}) {
		run(n, -1);
		run(n, (int)n - 1);
		run(n, 0);
	}

	// A failed enqueue must be reported, not ignored: a queue from another context than the program is rejected at enqueue time.
	++total;
	cl_context ctx2 = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
	cl_command_queue foreign = clCreateCommandQueue(ctx2, dev, 0, &err);
	opencl::Kernels badKernels(prog, foreign);
	badKernels.createKernel("check_equal");
	try {
		badKernels.runCheckEqual(a, b, okb, 64);
		std::printf("foreign queue: no exception MISMATCH\n"); ++fails;
	} catch (const std::exception & e) {
		std::printf("foreign queue: threw (%s): ok\n", e.what());
	}

	std::printf("legacy check_equal device test: %d/%d mismatches -> %s\n", fails, total, fails ? "FAIL" : "OK");

	// Release the OpenCL objects the test created, so a leak checker sees only real leaks.
	clReleaseMemObject(a); clReleaseMemObject(b); clReleaseMemObject(okb);
	clReleaseCommandQueue(foreign); clReleaseContext(ctx2);
	return fails ? 1 : 0;
}
