// Host-only test (no OpenCL device): what an OpenCL error status means to the user.
//
// A GPU that was reset or lost must read as such, not as an out-of-memory or a raw code, while a real allocation
// failure at setup still reads as out of memory.  The rule is in include/util/GpuLost.hpp; this checks it, checks
// that Aevum's separate copy of the table agrees, and drives the Marin and legacy error paths with injected codes.
// Nothing here talks to a driver.
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "marin/ocl.h"
#include "util/GpuLost.hpp"
#include "util/OpenCLError.hpp"
#include "../third_party/aevum/src/GpuLost.h"

namespace {

int failures = 0;

#define EXPECT(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } } while (0)

namespace gl = util::gpulost;

// Reaches the protected ocl_object::fatal() that every Marin OpenCL call goes through.
struct marin_probe : ocl::ocl_object
{
	static void check(const cl_int err, const char * const what, const gl::Phase phase) { fatal(err, what, phase); }
};

enum class Outcome { Nothing, Lost, Oom, Generic };

template<class F> Outcome run(F && f, std::string & message)
{
	try { f(); }
	catch (const gl::GpuLostError & e) { message = e.what(); return Outcome::Lost; }
	catch (const std::runtime_error & e)
	{
		message = e.what();
		return (message.rfind("Out of GPU memory", 0) == 0) ? Outcome::Oom : Outcome::Generic;
	}
	return Outcome::Nothing;
}

const char * name(const Outcome o)
{
	switch (o) { case Outcome::Nothing: return "nothing"; case Outcome::Lost: return "lost"; case Outcome::Oom: return "oom"; default: return "generic"; }
}

struct Row { cl_int code; const char * cl_name; Outcome create; Outcome exec; };

// The documented rule.  -5 is the only code whose meaning depends on the call.
const Row rows[] = {
	{ -2,    "CL_DEVICE_NOT_AVAILABLE",           Outcome::Lost,    Outcome::Lost },
	{ -4,    "CL_MEM_OBJECT_ALLOCATION_FAILURE",  Outcome::Oom,     Outcome::Oom },
	{ -5,    "CL_OUT_OF_RESOURCES",               Outcome::Oom,     Outcome::Lost },
	{ -6,    "CL_OUT_OF_HOST_MEMORY",             Outcome::Oom,     Outcome::Oom },
	{ -34,   "CL_INVALID_CONTEXT",                Outcome::Lost,    Outcome::Lost },
	{ -36,   "CL_INVALID_COMMAND_QUEUE",          Outcome::Lost,    Outcome::Lost },
	{ -9999, "CL_NVIDIA_DRIVER_ERROR",            Outcome::Lost,    Outcome::Lost },
	// anything else keeps the existing generic message
	{ -30,   "CL_INVALID_VALUE",                  Outcome::Generic, Outcome::Generic },
	{ -48,   "CL_INVALID_KERNEL",                 Outcome::Generic, Outcome::Generic },
	{ -54,   "CL_INVALID_WORK_GROUP_SIZE",        Outcome::Generic, Outcome::Generic },
	{ -14,   "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", Outcome::Generic, Outcome::Generic },
	{ -12345, "CL_UNKNOWN_ERROR",                 Outcome::Generic, Outcome::Generic },
};

void expect_lost_text(const std::string & m, const Row & r, const char * const engine)
{
	EXPECT(m.rfind("GPU reset or lost", 0) == 0, engine << " " << r.code << ": must start with the lost prefix: " << m);
	EXPECT(m.find(std::to_string(r.code)) != std::string::npos, engine << " " << r.code << ": must name the code: " << m);
	EXPECT(m.find("last checkpoint") != std::string::npos, engine << " " << r.code << ": must say work is safe to the last checkpoint: " << m);
	EXPECT(m.find("restart PrMers") != std::string::npos, engine << " " << r.code << ": must say restarting resumes: " << m);
	EXPECT(m.find("memory") == std::string::npos, engine << " " << r.code << ": must not mention memory: " << m);
	EXPECT(m.find('\n') == std::string::npos, engine << " " << r.code << ": must be one line: " << m);
}

void expect_oom_text(const std::string & m, const Row & r, const char * const engine)
{
	EXPECT(m.rfind("Out of GPU memory", 0) == 0, engine << " " << r.code << ": must say out of memory: " << m);
	EXPECT(m.find(std::to_string(r.code)) != std::string::npos, engine << " " << r.code << ": must name the code: " << m);
	EXPECT(m.find("reset") == std::string::npos, engine << " " << r.code << ": must not claim a reset: " << m);
}

}	// namespace

int main()
{
	// 1. The classification itself, both phases.
	for (const Row & r : rows)
	{
		const auto kind = [](const gl::Kind k) { return k == gl::Kind::DeviceLost ? Outcome::Lost : (k == gl::Kind::OutOfMemory ? Outcome::Oom : Outcome::Generic); };
		EXPECT(kind(gl::classify(r.code, gl::Phase::Create)) == r.create, "classify(" << r.code << ", Create) = " << name(kind(gl::classify(r.code, gl::Phase::Create))));
		EXPECT(kind(gl::classify(r.code, gl::Phase::Run)) == r.exec, "classify(" << r.code << ", Run) = " << name(kind(gl::classify(r.code, gl::Phase::Run))));
	}

	// 2. Aevum's copy of the table gives the same answer and the same text for every status, both phases.
	for (cl_int code = -80; code <= 2; ++code)
	{
		for (const int p : { 0, 1 })
		{
			const gl::Phase a = p ? gl::Phase::Run : gl::Phase::Create;
			const ::gpulost::Phase b = p ? ::gpulost::Phase::Run : ::gpulost::Phase::Create;
			EXPECT(static_cast<int>(gl::classify(code, a)) == static_cast<int>(::gpulost::classify(code, b)), "Aevum table differs at " << code);
		}
	}
	for (const cl_int code : { -1001, -9998, -9999, -10000 })
	{
		EXPECT(static_cast<int>(gl::classify(code, gl::Phase::Run)) == static_cast<int>(::gpulost::classify(code, ::gpulost::Phase::Run)), "Aevum table differs at " << code);
		EXPECT(static_cast<int>(gl::classify(code, gl::Phase::Create)) == static_cast<int>(::gpulost::classify(code, ::gpulost::Phase::Create)), "Aevum table differs at " << code);
	}
	EXPECT(gl::lost_message("X (-5)", "clFinish") == ::gpulost::lost_message("X (-5)", "clFinish"), "Aevum lost message differs");
	EXPECT(gl::oom_message("X (-5)", "clCreateBuffer") == ::gpulost::oom_message("X (-5)", "clCreateBuffer"), "Aevum out-of-memory message differs");
	EXPECT(std::string(gl::kLostPrefix) == ::gpulost::kLostPrefix, "Aevum prefix differs");
	EXPECT(gl::is_lost_message(gl::lost_message("X (-5)", "")), "the lost message is recognised again");
	EXPECT(!gl::is_lost_message(gl::oom_message("X (-5)", "")), "the out-of-memory message is not a lost message");
	EXPECT(!gl::is_lost_message("opencl error: CL_INVALID_VALUE"), "a generic message is not a lost message");
	EXPECT(!gl::is_lost_message(""), "an empty message is not a lost message");

	// 3. Marin: ocl_object::fatal() with an injected status, as an allocation call and as a run call.
	for (const Row & r : rows)
	{
		for (const gl::Phase phase : { gl::Phase::Create, gl::Phase::Run })
		{
			const Outcome want = (phase == gl::Phase::Create) ? r.create : r.exec;
			std::string m;
			const Outcome got = run([&] { marin_probe::check(r.code, phase == gl::Phase::Create ? "clCreateBuffer" : "clEnqueueNDRangeKernel", phase); }, m);
			EXPECT(got == want, "Marin " << r.code << (phase == gl::Phase::Create ? " create" : " run") << ": got " << name(got) << ", want " << name(want) << ": " << m);
			if (got == Outcome::Lost && got == want) expect_lost_text(m, r, "Marin");
			if (got == Outcome::Oom && got == want) expect_oom_text(m, r, "Marin");
			if (want == Outcome::Generic)
			{
				EXPECT(m.rfind("opencl error: ", 0) == 0, "Marin " << r.code << ": generic text changed: " << m);
				EXPECT(m.find(r.cl_name) != std::string::npos, "Marin " << r.code << ": generic text must name " << r.cl_name << ": " << m);
			}
		}
	}
	{
		std::string m;
		EXPECT(run([] { marin_probe::check(CL_SUCCESS, "clFinish", gl::Phase::Run); }, m) == Outcome::Nothing, "Marin: CL_SUCCESS must not throw");
		run([] { marin_probe::check(-5, "clEnqueueNDRangeKernel", gl::Phase::Run); }, m);
		EXPECT(m.find("CL_OUT_OF_RESOURCES") != std::string::npos && m.find("clEnqueueNDRangeKernel") != std::string::npos, "Marin lost message names the code and the call: " << m);
		run([] { marin_probe::check(-9999, nullptr, gl::Phase::Run); }, m);
		EXPECT(m.find("CL_NVIDIA_DRIVER_ERROR") != std::string::npos, "Marin names the vendor code: " << m);
		run([] { marin_probe::check(-36, nullptr, gl::Phase::Run); }, m);
		EXPECT(m.find(" in ") == std::string::npos, "a lost message without a call name has no dangling 'in': " << m);
	}

	// 4. Legacy NTT: util::throwClError, with the caller's own text for the generic case.
	for (const Row & r : rows)
	{
		for (const gl::Phase phase : { gl::Phase::Create, gl::Phase::Run })
		{
			const Outcome want = (phase == gl::Phase::Create) ? r.create : r.exec;
			std::string m;
			const Outcome got = run([&] { util::throwClError(r.code, phase, "clCreateBuffer", "caller text"); }, m);
			EXPECT(got == want, "legacy " << r.code << (phase == gl::Phase::Create ? " create" : " run") << ": got " << name(got) << ", want " << name(want) << ": " << m);
			if (got == Outcome::Lost && got == want) expect_lost_text(m, r, "legacy");
			if (got == Outcome::Oom && got == want) expect_oom_text(m, r, "legacy");
			if (want == Outcome::Generic) EXPECT(m == "caller text", "legacy " << r.code << ": generic must keep the caller's text: " << m);
		}
	}
	// Setup-time allocation failures that report and exit (RunPrpOrLl.cpp) use the same wording.
	{
		const std::string m5 = util::describeClAllocFailure(-5, "bufd");
		EXPECT(m5.rfind("Out of GPU memory", 0) == 0 && m5.find("bufd") != std::string::npos && m5.find("-5") != std::string::npos, "alloc failure -5 is out of memory: " << m5);
		EXPECT(util::describeClAllocFailure(-4, "r2").rfind("Out of GPU memory", 0) == 0, "alloc failure -4 is out of memory");
		EXPECT(util::describeClAllocFailure(-36, "save").rfind("GPU reset or lost", 0) == 0, "alloc failure -36 is a lost GPU");
		EXPECT(util::describeClAllocFailure(-30, "outOkBuf") == "Failed to allocate outOkBuf: -30", "other codes keep the existing text");
	}
	EXPECT(std::string(util::getCLErrorString(-9999)) == "CL_NVIDIA_DRIVER_ERROR", "getCLErrorString names -9999");
	EXPECT(std::string(util::getCLErrorString(-12345)) == "UNKNOWN ERROR", "getCLErrorString keeps UNKNOWN ERROR");
	EXPECT(std::string(util::getCLErrorString(CL_OUT_OF_RESOURCES)) == "CL_OUT_OF_RESOURCES", "getCLErrorString still names -5");

	if (failures) return EXIT_FAILURE;
	std::cout << "GPU lost message regression: PASS\n";
	return EXIT_SUCCESS;
}
