// Marin ocl::device: a failing clSetKernelArg or clEnqueueNDRangeKernel must raise an error instead of
// silently skipping the stage (the fast execution path used to discard both return codes).
// usage: marin-ocl-exec-errors-device-test [device-index]
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "marin/ocl.h"

namespace {

class probe : public ocl::device
{
public:
	probe(const ocl::platform & p, const size_t d) : ocl::device(p, d, false) {}

	cl_kernel make(const char * const name) { return _create_kernel(name); }
	void arg(cl_kernel k, const cl_uint index, const size_t size, const void * value) { _set_kernel_arg(k, index, size, value); }
	void run(cl_kernel k, const size_t global, const size_t local = 0) { _execute_kernel(k, global, local); }
	void finish() { finish_all_queues(); }
	size_t max_wg() const { return get_max_workgroup_size(); }
	cl_mem buffer(const size_t bytes) { return _create_buffer(CL_MEM_READ_WRITE, bytes); }
	void load(const std::string & src) { load_program(src); }
};

int failures = 0;

template<class F> void expect_throw(const char * const what, F && f)
{
	try { f(); }
	catch (const std::runtime_error & e) {
		if (std::string(e.what()).find("opencl error") == std::string::npos) { std::cerr << "FAIL: " << what << " unclear message: " << e.what() << "\n"; ++failures; }
		return;
	}
	std::cerr << "FAIL: " << what << " did not throw\n";
	++failures;
}

}	// namespace

int main(int argc, char ** argv)
{
	const size_t index = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 0;
	const ocl::platform platform;
	if (platform.get_device_count() <= index) { std::cerr << "no OpenCL device " << index << "\n"; return 2; }
	probe dev(platform, index);

	dev.load(
		"__kernel void fill(__global ulong * x, const ulong v) { x[get_global_id(0)] = v; }\n");
	cl_kernel fill = dev.make("fill");
	cl_mem buf = dev.buffer(4096 * sizeof(cl_ulong));
	const cl_ulong v = 7;

	// valid use keeps working
	dev.arg(fill, 0, sizeof(cl_mem), &buf);
	dev.arg(fill, 1, sizeof(cl_ulong), &v);
	dev.run(fill, 4096, 64);
	dev.finish();

	// clSetKernelArg: argument index past the last parameter, wrong size
	expect_throw("set_kernel_arg with index past the last argument", [&] { dev.arg(fill, 2, sizeof(cl_ulong), &v); });
	expect_throw("set_kernel_arg with a wrong argument size", [&] { dev.arg(fill, 1, 3, &v); });
	expect_throw("set_kernel_arg with a null kernel", [&] { dev.arg(nullptr, 0, sizeof(cl_ulong), &v); });

	// clEnqueueNDRangeKernel: global size not a multiple of the local size, local size above the device limit,
	// and a kernel with an argument that was never set
	expect_throw("enqueue with global % local != 0", [&] { dev.run(fill, 100, 64); });
	expect_throw("enqueue with local size above the device limit", [&] { dev.run(fill, 4 * dev.max_wg(), 2 * dev.max_wg()); });
	cl_kernel fresh = dev.make("fill");
	expect_throw("enqueue with an unset kernel argument", [&] { dev.run(fresh, 64); });

	dev.finish();
	clReleaseMemObject(buf);
	if (failures) return EXIT_FAILURE;
	std::cout << "Marin ocl exec error regression: PASS\n";
	return EXIT_SUCCESS;
}
