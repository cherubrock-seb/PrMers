// Legacy path: the exponent must reach the kernel as an unsigned 64-bit value.
//
// Context::getExponent() used to return an int and Program put it in -DMODULUS_P, so an exponent of 2^31 or
// more (the CLI accepts exponents up to 4294967295) became a negative define and the kernel computed garbage
// digit widths.
//
//  - host part: getExponent() is 64-bit unsigned and the build option spells the exponent out unsigned
//    for every boundary value (0, 1, 2^31 - 1, 2^31, 2^32 - 1);
//  - kernel part: kernels/prmers.cl is compiled on a CPU OpenCL device (never a GPU) with that exact
//    option and get_digit_width / get_digit_width4 must give widths that add up to p, the same way
//    for the scalar and the vector version. Skipped, loudly, when there is no CPU OpenCL device.
//
// Build/run: bash tests/test_legacy_modulus_p.sh
#define CL_TARGET_OPENCL_VERSION 120
#include "opencl/Context.hpp"
#include "opencl/Program.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef PRMERS_KERNEL_DIR
#define PRMERS_KERNEL_DIR "kernels"
#endif

using prmers::ocl::Context;
using prmers::ocl::Program;

static_assert(std::is_same_v<decltype(std::declval<const Context &>().getExponent()), uint64_t>,
              "Context::getExponent must not truncate a 32-bit exponent");

static int fails = 0;

static void expect(const bool ok, const std::string & what)
{
	if (!ok) { ++fails; std::printf("FAIL %s\n", what.c_str()); }
}

static void hostPart()
{
	const struct { uint64_t p; const char * text; } cases[] = {
		{ 0u, "0" }, { 1u, "1" }, { 2147483647u, "2147483647" }, { 2147483648u, "2147483648" },
		{ 3000000019u, "3000000019" }, { 4294967291u, "4294967291" }, { 4294967295u, "4294967295" },
	};
	for (const auto & c : cases)
	{
		const std::string d = Program::modulusDefine(c.p);
		expect(d == std::string("-DMODULUS_P=") + c.text + "UL", "define for p=" + std::string(c.text) + " is '" + d + "'");
		expect(d.find('-', 1) == std::string::npos, "define for p=" + std::string(c.text) + " has no sign");
	}
	// A value past 32 bits must not wrap either (the type is 64-bit all the way).
	expect(Program::modulusDefine(4294967296ull) == "-DMODULUS_P=4294967296UL", "2^32 does not wrap");
}

static std::string readFile(const std::string & path)
{
	std::ifstream in(path, std::ios::binary);
	std::ostringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

static bool findCpuDevice(cl_device_id & dev)
{
	cl_uint np = 0;
	if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return false;
	std::vector<cl_platform_id> ps(np);
	clGetPlatformIDs(np, ps.data(), nullptr);
	for (cl_platform_id p : ps)
	{
		cl_uint nd = 0;
		// CPU devices only: this test must never run on a GPU.
		if (clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &dev, &nd) == CL_SUCCESS && nd > 0) return true;
	}
	return false;
}

static void kernelPart()
{
	cl_device_id dev = nullptr;
	if (!findCpuDevice(dev)) { std::printf("SKIPPED kernel part: no CPU OpenCL device\n"); return; }

	cl_int err = CL_SUCCESS;
	cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
	expect(err == CL_SUCCESS, "clCreateContext");
	cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
	expect(err == CL_SUCCESS, "clCreateCommandQueue");
	if (fails) return;

	const std::string probe = R"(
__kernel void probe(__global int * w1, __global int * w4)
{
	uint i = (uint)get_global_id(0);
	w1[i] = get_digit_width(i);
	if (i % 4u == 0u) { int4 r = get_digit_width4(i); w4[i] = r.s0; w4[i + 1] = r.s1; w4[i + 2] = r.s2; w4[i + 3] = r.s3; }
}
)";
	const std::string src = readFile(std::string(PRMERS_KERNEL_DIR) + "/prmers.cl") + probe;
	expect(src.size() > probe.size(), "kernels/prmers.cl is readable");

	for (const uint64_t n : { 8u, 16u, 20u, 64u })
	{
		for (const uint64_t p : { 2147483647ull, 2147483648ull, 2147483659ull, 3000000019ull, 4294967291ull, 4294967295ull })
		{
			const char * csrc = src.c_str(); const size_t len = src.size();
			cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &len, &err);
			const std::string opts = Program::modulusDefine(p) + " -DTRANSFORM_SIZE_N=" + std::to_string(n);
			err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
			if (err != CL_SUCCESS)
			{
				char log[4096] = {0}; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, nullptr);
				expect(false, "kernel build for p=" + std::to_string(p) + ": " + log);
				clReleaseProgram(prog); continue;
			}
			cl_kernel k = clCreateKernel(prog, "probe", &err);
			cl_mem b1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(int), nullptr, &err);
			cl_mem b4 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(int), nullptr, &err);
			clSetKernelArg(k, 0, sizeof b1, &b1); clSetKernelArg(k, 1, sizeof b4, &b4);
			const size_t g = n;
			expect(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr) == CL_SUCCESS, "enqueue");
			std::vector<int> w1(n, -1), w4(n, -1);
			clEnqueueReadBuffer(q, b1, CL_TRUE, 0, n * sizeof(int), w1.data(), 0, nullptr, nullptr);
			clEnqueueReadBuffer(q, b4, CL_TRUE, 0, n * sizeof(int), w4.data(), 0, nullptr, nullptr);
			uint64_t sum = 0;
			const uint64_t lo = p / n, hi = lo + 1;
			bool range = true, same = true;
			for (uint64_t i = 0; i < n; ++i)
			{
				sum += static_cast<uint64_t>(w1[i] < 0 ? 0 : w1[i]);
				range = range && w1[i] >= 0 && static_cast<uint64_t>(w1[i]) >= lo && static_cast<uint64_t>(w1[i]) <= hi;
				same = same && (n % 4 != 0 || w1[i] == w4[i]);
			}
			const std::string tag = " (p=" + std::to_string(p) + ", N=" + std::to_string(n) + ")";
			expect(sum == p, "digit widths add up to p" + tag + " got " + std::to_string(sum));
			expect(range, "every width is floor(p/N) or ceil(p/N)" + tag);
			expect(same, "get_digit_width4 matches get_digit_width" + tag);
			clReleaseMemObject(b1); clReleaseMemObject(b4); clReleaseKernel(k); clReleaseProgram(prog);
		}
	}
	clReleaseCommandQueue(q); clReleaseContext(ctx);
}

int main()
{
	hostPart();
	kernelPart();
	std::printf("Legacy MODULUS_P test: %s\n", fails ? "FAIL" : "OK");
	return fails ? 1 : 0;
}
