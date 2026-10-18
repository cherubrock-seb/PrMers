// LD_PRELOAD shim for tests/run_legacy_read_fault.sh: makes one clEnqueueReadBuffer fail, as it does when the GPU
// is reset or lost part way through a run.  Every other call goes to the real OpenCL library.
//
//   PRMERS_FAULT_READ_BYTES  only reads of exactly this many bytes are counted (unset or 0: every read)
//   PRMERS_FAULT_READ_NTH    the counted read that fails, 1-based (unset or 0: none fails)
//   PRMERS_FAULT_READ_CODE   the status it returns (default -36, CL_INVALID_COMMAND_QUEUE)
//   PRMERS_FAULT_READ_LOG    a file that gets one line "<bytes>" per read, to find the read to fail
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef cl_int (*read_fn)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*,
                          cl_event*);

static unsigned long long env_u64(const char* name) {
    const char* v = getenv(name);
    return (v && *v) ? strtoull(v, NULL, 10) : 0ULL;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer, cl_bool blocking,
                                                    size_t offset, size_t size, void* ptr, cl_uint numEvents,
                                                    const cl_event* waitList, cl_event* event) {
    static read_fn real = NULL;
    static unsigned long long seen = 0;
    if (!real) real = (read_fn)dlsym(RTLD_NEXT, "clEnqueueReadBuffer");

    const char* logPath = getenv("PRMERS_FAULT_READ_LOG");
    if (logPath && *logPath) {
        FILE* f = fopen(logPath, "a");
        if (f) {
            fprintf(f, "%zu\n", size);
            fclose(f);
        }
    }

    const unsigned long long bytes = env_u64("PRMERS_FAULT_READ_BYTES");
    const unsigned long long nth = env_u64("PRMERS_FAULT_READ_NTH");
    if (nth != 0 && (bytes == 0 || bytes == size) && ++seen == nth) {
        const char* code = getenv("PRMERS_FAULT_READ_CODE");
        fprintf(stderr, "[fault shim] failing clEnqueueReadBuffer #%llu of %zu bytes\n", seen, size);
        return (code && *code) ? (cl_int)strtol(code, NULL, 10) : CL_INVALID_COMMAND_QUEUE;
    }
    return real(queue, buffer, blocking, offset, size, ptr, numEvents, waitList, event);
}
