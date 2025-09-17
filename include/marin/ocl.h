/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#define CL_TARGET_OPENCL_VERSION 110
#if defined(__APPLE__)
	#include <OpenCL/cl.h>
	#include <OpenCL/cl_ext.h>
#else
	#include <CL/cl.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>

// #define ocl_debug		1
#define ocl_fast_exec		1

namespace ocl
{

class ocl_object
{
private:
	static const char * error_string(const cl_int & res)
	{
		switch (res)
		{
	#define ocl_check(err) case err: return #err
			ocl_check(CL_SUCCESS);
			ocl_check(CL_DEVICE_NOT_FOUND);
			ocl_check(CL_DEVICE_NOT_AVAILABLE);
			ocl_check(CL_COMPILER_NOT_AVAILABLE);
			ocl_check(CL_MEM_OBJECT_ALLOCATION_FAILURE);
			ocl_check(CL_OUT_OF_RESOURCES);
			ocl_check(CL_OUT_OF_HOST_MEMORY);
			ocl_check(CL_PROFILING_INFO_NOT_AVAILABLE);
			ocl_check(CL_MEM_COPY_OVERLAP);
			ocl_check(CL_IMAGE_FORMAT_MISMATCH);
			ocl_check(CL_IMAGE_FORMAT_NOT_SUPPORTED);
			ocl_check(CL_BUILD_PROGRAM_FAILURE);
			ocl_check(CL_MAP_FAILURE);
			ocl_check(CL_MISALIGNED_SUB_BUFFER_OFFSET);
			ocl_check(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
			ocl_check(CL_INVALID_VALUE);
			ocl_check(CL_INVALID_DEVICE_TYPE);
			ocl_check(CL_INVALID_PLATFORM);
			ocl_check(CL_INVALID_DEVICE);
			ocl_check(CL_INVALID_CONTEXT);
			ocl_check(CL_INVALID_QUEUE_PROPERTIES);
			ocl_check(CL_INVALID_COMMAND_QUEUE);
			ocl_check(CL_INVALID_HOST_PTR);
			ocl_check(CL_INVALID_MEM_OBJECT);
			ocl_check(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
			ocl_check(CL_INVALID_IMAGE_SIZE);
			ocl_check(CL_INVALID_SAMPLER);
			ocl_check(CL_INVALID_BINARY);
			ocl_check(CL_INVALID_BUILD_OPTIONS);
			ocl_check(CL_INVALID_PROGRAM);
			ocl_check(CL_INVALID_PROGRAM_EXECUTABLE);
			ocl_check(CL_INVALID_KERNEL_NAME);
			ocl_check(CL_INVALID_KERNEL_DEFINITION);
			ocl_check(CL_INVALID_KERNEL);
			ocl_check(CL_INVALID_ARG_INDEX);
			ocl_check(CL_INVALID_ARG_VALUE);
			ocl_check(CL_INVALID_ARG_SIZE);
			ocl_check(CL_INVALID_KERNEL_ARGS);
			ocl_check(CL_INVALID_WORK_DIMENSION);
			ocl_check(CL_INVALID_WORK_GROUP_SIZE);
			ocl_check(CL_INVALID_WORK_ITEM_SIZE);
			ocl_check(CL_INVALID_GLOBAL_OFFSET);
			ocl_check(CL_INVALID_EVENT_WAIT_LIST);
			ocl_check(CL_INVALID_EVENT);
			ocl_check(CL_INVALID_OPERATION);
			ocl_check(CL_INVALID_GL_OBJECT);
			ocl_check(CL_INVALID_BUFFER_SIZE);
			ocl_check(CL_INVALID_MIP_LEVEL);
			ocl_check(CL_INVALID_GLOBAL_WORK_SIZE);
			ocl_check(CL_INVALID_PROPERTY);
	#undef ocl_check
			default: return "CL_UNKNOWN_ERROR";
		}
	}

protected:
	static constexpr bool error(const cl_int res)
	{
		return (res == CL_SUCCESS);
	}

protected:
	static void fatal(const cl_int res, const char * const ext = nullptr)
	{
		if (!error(res))
		{
			std::ostringstream ss; ss << "opencl error: " << error_string(res);
			if (ext != nullptr) ss << " (" << ext << ")";
			throw std::runtime_error(ss.str());
		}
	}
};

class platform : ocl_object
{
private:
	struct device_desc
	{
		cl_platform_id platform_id;
		cl_device_id device_id;
		std::string name;
	};
	std::vector<device_desc> _devices;

protected:
	void find_devices(const bool gpu)
	{
		cl_uint num_platforms;
		cl_platform_id platforms[64];
		fatal(clGetPlatformIDs(64, platforms, &num_platforms));

		for (cl_uint p = 0; p < num_platforms; ++p)
		{
			char platform_name[1024]; fatal(clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, platform_name, nullptr));

			cl_uint num_devices;
			cl_device_id devices[64];
			if (error(clGetDeviceIDs(platforms[p], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL, 64, devices, &num_devices)))
			{
				for (cl_uint d = 0; d < num_devices; ++d)
				{
					char device_name[1024]; fatal(clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, device_name, nullptr));
					char device_vendor[1024]; fatal(clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, 1024, device_vendor, nullptr));

					std::ostringstream ss; ss << "device '" << device_name << "', vendor '" << device_vendor << "', platform '" << platform_name << "'";
					device_desc device;
					device.platform_id = platforms[p];
					device.device_id = devices[d];
					device.name = ss.str();
					_devices.push_back(device);
				}
			}
		}
	}

public:
	platform()
	{
#if defined(ocl_debug)
		std::cout << "Create ocl platform." << std::endl;
#endif
		find_devices(true);
		if (_devices.empty()) find_devices(false);
	}

	platform(const cl_platform_id platform_id, const cl_device_id device_id)
	{
		char platform_name[1024]; fatal(clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 1024, platform_name, nullptr));
		char device_name[1024]; fatal(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, device_name, nullptr));
		char device_vendor[1024]; fatal(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 1024, device_vendor, nullptr));

		std::ostringstream ss; ss << "device '" << device_name << "', vendor '" << device_vendor << "', platform '" << platform_name << "'";
		device_desc device;
		device.platform_id = platform_id;
		device.device_id = device_id;
		device.name = ss.str();
		_devices.push_back(device);
	}

public:
	virtual ~platform()
	{
#if defined(ocl_debug)
		std::cout << "Delete ocl platform." << std::endl;
#endif
	}

public:
	size_t get_device_count() const { return _devices.size(); }

public:
	size_t display_devices() const
	{
		const size_t n = _devices.size();
		for (size_t i = 0; i < n; ++i)
		{
			std::cout << i << " - " << _devices[i].name << "." << std::endl;
		}
		std::cout << std::endl;
		return n;
	}

public:
	cl_platform_id get_platform(const size_t d) const { return _devices[d].platform_id; }
	cl_device_id get_device(const size_t d) const { return _devices[d].device_id; }
};

class device : ocl_object
{
private:
	enum class EVendor { Unknown, NVIDIA, AMD, INTEL };

	const cl_platform_id _platform;
	const cl_device_id _device;
#if defined(ocl_debug)
	const size_t _d;
#endif
	bool _profile = false;
#if defined(__APPLE__)
	bool _is_sync = true;
#else
	bool _is_sync = false;
#endif
	size_t _sync_count = 0;
	cl_ulong _local_mem_size = 0;
	size_t _max_workgroup_size = 0;
	cl_ulong _timer_resolution = 0;
	EVendor _vendor = EVendor::Unknown;
	cl_context _context = nullptr;
	cl_command_queue _queueF = nullptr;
	cl_command_queue _queueP = nullptr;
	cl_command_queue _queue = nullptr;
	cl_program _program = nullptr;

	struct profile
	{
		std::string name;
		size_t count;
		cl_ulong time;

		profile() {}
		profile(const std::string & name) : name(name), count(0), time(0) {}
	};
	std::map<cl_kernel, profile> _profile_map;

public:
	device(const platform & parent, const size_t d, const bool verbose) : _platform(parent.get_platform(d)), _device(parent.get_device(d))
#if defined(ocl_debug)
		, _d(d)
#endif
	{
#if defined(ocl_debug)
		std::cout << "Create ocl device " << d << "." << std::endl;
#endif

		char device_name[1024]; fatal(clGetDeviceInfo(_device, CL_DEVICE_NAME, 1024, device_name, nullptr));
		char device_vendor[1024]; fatal(clGetDeviceInfo(_device, CL_DEVICE_VENDOR, 1024, device_vendor, nullptr));
		char device_version[1024]; fatal(clGetDeviceInfo(_device, CL_DEVICE_VERSION, 1024, device_version, nullptr));
		char driver_version[1024]; fatal(clGetDeviceInfo(_device, CL_DRIVER_VERSION, 1024, driver_version, nullptr));
		_vendor = get_vendor(device_vendor);

		cl_uint compute_units; fatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr));
		cl_uint max_clock_frequency; fatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_frequency), &max_clock_frequency, nullptr));
		cl_ulong mem_size; fatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr));
		cl_ulong mem_cache_size; fatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(mem_cache_size), &mem_cache_size, nullptr));
		cl_uint mem_cache_line_size; fatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(mem_cache_line_size), &mem_cache_line_size, nullptr));
		fatal(clGetDeviceInfo(_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(_local_mem_size), &_local_mem_size, nullptr));
		cl_ulong mem_const_size; fatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_const_size), &mem_const_size, nullptr));
		fatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(_max_workgroup_size), &_max_workgroup_size, nullptr));
		fatal(clGetDeviceInfo(_device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(_timer_resolution), &_timer_resolution, nullptr));

		if (verbose)
		{
			static bool first = true;
			if (first)
			{
				first = false;
				std::cout << "Running on device '" << device_name << "', vendor '" << device_vendor
					<< "', version '" << device_version << "', driver '" << driver_version << "'." << std::endl
					<< compute_units << " Compute Units @ " << max_clock_frequency << " MHz, local memory size = "
					<< (_local_mem_size >> 10) << " kB, max work-group size = " << _max_workgroup_size << "." << std::endl << std::endl;
			}
		}

		const cl_context_properties context_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)_platform, 0 };
		cl_int err_cc;
		_context = clCreateContext(context_properties, 1, &_device, nullptr, nullptr, &err_cc);
		fatal(err_cc);
		cl_int err_ccq;
		_queueF = clCreateCommandQueue(_context, _device, 0, &err_ccq);
		_queueP = clCreateCommandQueue(_context, _device, CL_QUEUE_PROFILING_ENABLE, &err_ccq);
		_queue = _queueF;	// default queue is fast
		fatal(err_ccq);

		if (_vendor != EVendor::NVIDIA) _is_sync = true;
	}

public:
	virtual ~device()
	{
#if defined(ocl_debug)
		std::cout << "Delete ocl device " << _d << "." << std::endl;
#endif
		fatal(clReleaseCommandQueue(_queueP));
		fatal(clReleaseCommandQueue(_queueF));
		fatal(clReleaseContext(_context));
	}

public:
	size_t get_max_workgroup_size() const { return _max_workgroup_size; }
	size_t get_local_mem_size() const { return _local_mem_size; }
	size_t get_max_local_worksize(const size_t type_size) const { return std::min(_max_workgroup_size, size_t(_local_mem_size) / type_size); }
	size_t get_timer_resolution() const { return _timer_resolution; }
	bool isIntel() const { return (_vendor == EVendor::INTEL); }

private:
	static EVendor get_vendor(const std::string & vendor_string)
	{
		std::string l_vendor_string; l_vendor_string.resize(vendor_string.size());
		std::transform(vendor_string.begin(), vendor_string.end(), l_vendor_string.begin(), [](char c){ return std::tolower(c); });

		if (strstr(l_vendor_string.c_str(), "nvidia") != nullptr) return EVendor::NVIDIA;
		if (strstr(l_vendor_string.c_str(), "amd") != nullptr) return EVendor::AMD;
		if (strstr(l_vendor_string.c_str(), "advanced micro devices") != nullptr) return EVendor::AMD;
		if (strstr(l_vendor_string.c_str(), "intel") != nullptr) return EVendor::INTEL;
		// must be tested after 'Intel' because 'ati' is in 'Intel(R) Corporation' string
		if (strstr(l_vendor_string.c_str(), "ati") != nullptr) return EVendor::AMD;
		return EVendor::Unknown;
	}

public:
	void reset_profiles()
	{
		for (auto it : _profile_map)
		{
			profile & prof = _profile_map[it.first];	// it.first is not a reference!
			prof.count = 0;
			prof.time = 0;
		}
	}

public:
	cl_ulong get_profile_time() const
	{
		cl_ulong time = 0;
		for (auto it : _profile_map) time += it.second.time;
		return time;
	}

public:
	void display_profiles(const size_t count) const
	{
		cl_ulong ptime = 0;
		for (auto it : _profile_map) ptime += it.second.time;
		ptime /= count;

		for (auto it : _profile_map)
		{
			const profile & prof = it.second;
			if (prof.count != 0)
			{
				const size_t ncount = prof.count / count;
				const cl_ulong ntime = prof.time / count;
				std::cout << "- " << prof.name << ": " << ncount << ", " << std::setprecision(3)
					<< ntime * 100.0 / ptime << " %, " << ntime << " (" << (ntime / ncount) << ")" << std::endl;
			}
		}
	}

public:
	void set_profiling(const bool enable)
	{
		_profile = enable;
		_queue = enable ? _queueP : _queueF;
		reset_profiles();
	}

public:
	bool read_OpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::ostringstream & src) const
	{
		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;
		
		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2025, Yves Gallot" << std::endl << std::endl;
		hFile << "marin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;

		hFile << "#pragma once" << std::endl << std::endl;
		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"";
			for (char c : line)
			{
				if ((c == '\\') || (c == '\"')) hFile << '\\';
				hFile << c;
			}
			hFile << "\\n\" \\" << std::endl;

			src << line << std::endl;
		}
		hFile << "\"\";" << std::endl;

		hFile.close();
		clFile.close();
		return true;
	}

public:
	void load_program(const std::string & program_src)
	{
#if defined(ocl_debug)
		std::cout <<  "Load ocl program." << std::endl;
#endif
		const char * src[1]; src[0] = program_src.c_str();
		cl_int err_cpws;
		_program = clCreateProgramWithSource(_context, 1, src, nullptr, &err_cpws);
		fatal(err_cpws);

		char pgm_options[1024];
		strcpy(pgm_options, "");
#if defined(ocl_debug)
		if (_vendor == EVendor::NVIDIA) strcat(pgm_options, " -cl-nv-verbose");
		if (_vendor == EVendor::AMD) strcat(pgm_options, " -save-temps=.");
#endif
		const cl_int err = clBuildProgram(_program, 1, &_device, pgm_options, nullptr, nullptr);

#if !defined(ocl_debug)
		if (err != CL_SUCCESS)
#endif
		{
			size_t log_size; clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
			if (log_size > 2)
			{
				std::vector<char> build_log(log_size + 1);
				clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
				build_log[log_size] = '\0';
#if defined(ocl_debug)
				std::ofstream file_out("pgm.log"); 
				file_out << build_log.data() << std::endl;
				file_out.close();
#else
				std::cout <<  build_log.data() << std::endl;
#endif
			}
		}

		fatal(err);

#if defined(ocl_debug)
		size_t bin_size; clGetProgramInfo(_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, nullptr);
		std::vector<char> binary(bin_size);
		clGetProgramInfo(_program, CL_PROGRAM_BINARIES, sizeof(char *), &binary, nullptr);
		std::ofstream file_out((_vendor == EVendor::NVIDIA) ? "pgm.ptx" : "pgm.bin", std::ios::binary);
		file_out.write(binary.data(), std::streamsize(bin_size));
		file_out.close();
#endif
	}

public:
	void clear_program()
	{
#if defined(ocl_debug)
		std::cout << "Clear ocl program." << std::endl;
#endif
		fatal(clReleaseProgram(_program));
		_program = nullptr;
		_profile_map.clear();
	}

private:
	void _sync()
	{
		_sync_count = 0;
		fatal(clFinish(_queue));
	}

public:
	cl_mem _create_buffer(const cl_mem_flags flags, const size_t size, const bool clear = true) const
	{
		cl_int err;
		cl_mem mem = clCreateBuffer(_context, flags, size, nullptr, &err);
		fatal(err);
		if (clear)
		{
			std::vector<uint8_t> ptr(size);
			for (size_t i = 0; i < size; ++i) ptr[i] = 0x00;	// debug 0xff;
			fatal(clEnqueueWriteBuffer(_queue, mem, CL_TRUE, 0, size, ptr.data(), 0, nullptr, nullptr));
		}
		return mem;
	}

public:
	static void _release_buffer(cl_mem & mem)
	{
		if (mem != nullptr)
		{
			fatal(clReleaseMemObject(mem));
			mem = nullptr;
		}
	}

protected:
	void _read_buffer(cl_mem & mem, void * const ptr, const size_t size, const size_t offset = 0)
	{
		// Fill the buffer with random numbers to generate an error even if clEnqueueReadBuffer fails without error.
		char * const cptr = static_cast<char *>(ptr);
		for (size_t i = 0; i < size; ++i) cptr[i] = char(std::rand());
		_sync();
		fatal(clEnqueueReadBuffer(_queue, mem, CL_TRUE, offset, size, ptr, 0, nullptr, nullptr));
	}

protected:
	void _write_buffer(cl_mem & mem, const void * const ptr, const size_t size, const size_t offset = 0)
	{
		_sync();
		fatal(clEnqueueWriteBuffer(_queue, mem, CL_TRUE, offset, size, ptr, 0, nullptr, nullptr));
	}

protected:
	cl_kernel _create_kernel(const char * const kernel_name)
	{
		cl_int err;
		cl_kernel kernel = clCreateKernel(_program, kernel_name, &err);
		fatal(err, kernel_name);
		_profile_map[kernel] = profile(kernel_name);
		return kernel;
	}

protected:
	static void _release_kernel(cl_kernel & kernel)
	{
		if (kernel != nullptr)
		{
			fatal(clReleaseKernel(kernel));
			kernel = nullptr;
		}
	}

protected:
	static void _set_kernel_arg(cl_kernel kernel, const cl_uint arg_index, const size_t arg_size, const void * const arg_value)
	{
#if !defined(ocl_fast_exec) || defined(ocl_debug)
		cl_int err =
#endif
		clSetKernelArg(kernel, arg_index, arg_size, arg_value);
#if !defined(ocl_fast_exec) || defined(ocl_debug)
		fatal(err);
#endif
	}

protected:
	void _execute_kernel(cl_kernel kernel, const size_t global_worksize, const size_t local_worksize = 0)
	{
		if (!_profile)
		{
#if !defined(ocl_fast_exec) || defined(ocl_debug)
			cl_int err =
#endif
			clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &global_worksize, (local_worksize == 0) ? nullptr : &local_worksize, 0, nullptr, nullptr);
#if !defined(ocl_fast_exec) || defined(ocl_debug)
			fatal(err);
#endif
			if (_is_sync)
			{
				++_sync_count;
				if (_sync_count == 16 * 1024) _sync();
			}
		}
		else
		{
			_sync();
			cl_event evt;
			fatal(clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &global_worksize, (local_worksize == 0) ? nullptr : &local_worksize, 0, nullptr, &evt));
			cl_ulong dt = 0;
			if (clWaitForEvents(1, &evt) == CL_SUCCESS)
			{
				cl_ulong start, end;
				cl_int err_s = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
				cl_int err_e = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
				if ((err_s == CL_SUCCESS) && (err_e == CL_SUCCESS)) dt = end - start;
			}
			clReleaseEvent(evt);

			profile & prof = _profile_map[kernel];
			prof.count++;
			prof.time += dt;
		}
	}
};

}