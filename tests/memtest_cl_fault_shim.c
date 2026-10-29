// LD_PRELOAD shim for tests/run_memtest_read_fault.sh: makes one clEnqueueReadBuffer or clFinish of -memtest fail,
// as it does when the GPU is reset or lost part way through a test.  Every other call goes to the real OpenCL library.
//
// -memtest looks for GPU-type devices itself and tests nearly all of the device's global memory, so the shim also
// lets it run on PoCL's CPU device with a small memory:
//   PRMERS_MEMTEST_ANY_DEVICE  if set, a query for GPU devices returns devices of every type
//   PRMERS_MEMTEST_VRAM_MB     if set, the global memory size the device reports, in MB (the test covers it less 32 MB)
//
//   PRMERS_FAULT_READ_BYTES    only reads of exactly this many bytes are counted (unset or 0: every read)
//   PRMERS_FAULT_READ_NTH      the counted read that fails, 1-based (unset or 0: none fails)
//   PRMERS_FAULT_FINISH_NTH    the clFinish that fails, 1-based (unset or 0: none fails)
//   PRMERS_FAULT_WRITE_NTH     the clEnqueueWriteBuffer (an error count reset) that fails, 1-based (unset or 0: none)
//   PRMERS_FAULT_SUBBUF_NTH    the clCreateSubBuffer (a sector) that fails, 1-based (unset or 0: none fails)
//   PRMERS_FAULT_CODE          the status the failing call returns (default -36, CL_INVALID_COMMAND_QUEUE)
//   PRMERS_FAKE_COUNT_NTH      the 4-byte read (an error count), 1-based, that succeeds but returns 1: a memory error
//                              found, so the error report is read (unset or 0: none)
//   PRMERS_FAULT_LOG           a file that gets one line per read ("R <bytes>") and per clFinish ("F")
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef cl_int (*read_fn)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*,
                          cl_event*);
typedef cl_int (*finish_fn)(cl_command_queue);
typedef cl_int (*write_fn)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*,
                           cl_event*);
typedef cl_mem (*subbuf_fn)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void*, cl_int*);
typedef cl_int (*ids_fn)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*info_fn)(cl_device_id, cl_device_info, size_t, void*, size_t*);

static unsigned long long env_u64(const char* name) {
    const char* v = getenv(name);
    return (v && *v) ? strtoull(v, NULL, 10) : 0ULL;
}

static void log_line(const char* line) {
    const char* logPath = getenv("PRMERS_FAULT_LOG");
    if (logPath && *logPath) {
        FILE* f = fopen(logPath, "a");
        if (f) {
            fputs(line, f);
            fclose(f);
        }
    }
}

static cl_int fault_code(void) {
    const char* code = getenv("PRMERS_FAULT_CODE");
    return (code && *code) ? (cl_int)strtol(code, NULL, 10) : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform, cl_device_type type, cl_uint numEntries,
                                               cl_device_id* devices, cl_uint* numDevices) {
    static ids_fn real = NULL;
    if (!real) real = (ids_fn)dlsym(RTLD_NEXT, "clGetDeviceIDs");
    if (getenv("PRMERS_MEMTEST_ANY_DEVICE") && type == CL_DEVICE_TYPE_GPU) type = CL_DEVICE_TYPE_ALL;
    return real(platform, type, numEntries, devices, numDevices);
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info param, size_t size, void* value,
                                                size_t* sizeRet) {
    static info_fn real = NULL;
    if (!real) real = (info_fn)dlsym(RTLD_NEXT, "clGetDeviceInfo");
    const unsigned long long mb = env_u64("PRMERS_MEMTEST_VRAM_MB");
    if (mb != 0 && param == CL_DEVICE_GLOBAL_MEM_SIZE) {
        if (sizeRet) *sizeRet = sizeof(cl_ulong);
        if (value) {
            if (size < sizeof(cl_ulong)) return CL_INVALID_VALUE;
            const cl_ulong bytes = (cl_ulong)mb * 1024ULL * 1024ULL;
            memcpy(value, &bytes, sizeof(bytes));
        }
        return CL_SUCCESS;
    }
    return real(device, param, size, value, sizeRet);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, cl_bool blocking,
                                                    size_t offset, size_t size, void* ptr, cl_uint numEvents,
                                                    const cl_event* waitList, cl_event* event) {
    static read_fn real = NULL;
    static unsigned long long seen = 0;
    if (!real) real = (read_fn)dlsym(RTLD_NEXT, "clEnqueueReadBuffer");

    char line[64];
    snprintf(line, sizeof(line), "R %zu\n", size);
    log_line(line);

    const unsigned long long bytes = env_u64("PRMERS_FAULT_READ_BYTES");
    const unsigned long long nth = env_u64("PRMERS_FAULT_READ_NTH");
    if (nth != 0 && (bytes == 0 || bytes == size) && ++seen == nth) {
        fprintf(stderr, "[fault shim] failing clEnqueueReadBuffer #%llu of %zu bytes\n", seen, size);
        return fault_code();
    }
    const cl_int rc = real(queue, buffer, blocking, offset, size, ptr, numEvents, waitList, event);
    static unsigned long long counts = 0;
    const unsigned long long fake = env_u64("PRMERS_FAKE_COUNT_NTH");
    if (rc == CL_SUCCESS && size == sizeof(cl_uint) && fake != 0 && ++counts == fake) {
        const cl_uint one = 1;
        fprintf(stderr, "[fault shim] error count read #%llu returns 1\n", counts);
        memcpy(ptr, &one, sizeof(one));
    }
    return rc;
}

CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue queue) {
    static finish_fn real = NULL;
    static unsigned long long seen = 0;
    if (!real) real = (finish_fn)dlsym(RTLD_NEXT, "clFinish");

    log_line("F\n");

    const unsigned long long nth = env_u64("PRMERS_FAULT_FINISH_NTH");
    if (nth != 0 && ++seen == nth) {
        fprintf(stderr, "[fault shim] failing clFinish #%llu\n", seen);
        return fault_code();
    }
    return real(queue);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue queue, cl_mem buffer, cl_bool blocking,
                                                     size_t offset, size_t size, const void* ptr, cl_uint numEvents,
                                                     const cl_event* waitList, cl_event* event) {
    static write_fn real = NULL;
    static unsigned long long seen = 0;
    if (!real) real = (write_fn)dlsym(RTLD_NEXT, "clEnqueueWriteBuffer");

    const unsigned long long nth = env_u64("PRMERS_FAULT_WRITE_NTH");
    if (nth != 0 && ++seen == nth) {
        fprintf(stderr, "[fault shim] failing clEnqueueWriteBuffer #%llu\n", seen);
        return fault_code();
    }
    return real(queue, buffer, blocking, offset, size, ptr, numEvents, waitList, event);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type type,
                                                  const void* info, cl_int* err) {
    static subbuf_fn real = NULL;
    static unsigned long long seen = 0;
    if (!real) real = (subbuf_fn)dlsym(RTLD_NEXT, "clCreateSubBuffer");

    const unsigned long long nth = env_u64("PRMERS_FAULT_SUBBUF_NTH");
    if (nth != 0 && ++seen == nth) {
        fprintf(stderr, "[fault shim] failing clCreateSubBuffer #%llu\n", seen);
        if (err) *err = fault_code();
        return NULL;
    }
    return real(buffer, flags, type, info, err);
}
