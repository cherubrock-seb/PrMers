// LD_PRELOAD shim for the tuner isolation GPU tests: kills one OpenCL context the way a GPU fault or a driver reset of
// a hung context does, and leaves every other context working.  Every call goes to the real OpenCL library unless it
// touches a dead context.
//
//   AEVUM_FAULT_BUILD_OPT  the first program compiled with build options that contain this text arms its context:
//                          the next kernel launched on a queue of that context fails, and the context is dead from
//                          then on.  Only one context is ever armed (unset or empty: none is)
//   AEVUM_FAULT_CODE       the status the failing calls return (default -5, CL_OUT_OF_RESOURCES)
//   AEVUM_FAULT_LOG        a file that gets one line per armed context ("ARM") and per killed context ("KILL")
//
// After the kill, every launch, read, write, copy, fill, marker, flush and finish on a queue of that context, and
// every buffer or program created in it, fails with AEVUM_FAULT_CODE.
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CONTEXTS 64

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static cl_context g_armed[MAX_CONTEXTS];
static cl_context g_dead[MAX_CONTEXTS];
static int g_armed_once = 0;

static cl_int fault_code(void) {
    const char* code = getenv("AEVUM_FAULT_CODE");
    return (code && *code) ? (cl_int)strtol(code, NULL, 10) : CL_OUT_OF_RESOURCES;
}

static void log_line(const char* line) {
    const char* path = getenv("AEVUM_FAULT_LOG");
    if (!path || !*path) return;
    FILE* f = fopen(path, "a");
    if (!f) return;
    fputs(line, f);
    fclose(f);
}

static int contains(cl_context* set, cl_context ctx) {
    for (int i = 0; i < MAX_CONTEXTS; ++i)
        if (set[i] == ctx) return 1;
    return 0;
}

static void insert(cl_context* set, cl_context ctx) {
    if (contains(set, ctx)) return;
    for (int i = 0; i < MAX_CONTEXTS; ++i)
        if (!set[i]) { set[i] = ctx; return; }
}

static cl_context queue_context(cl_command_queue queue) {
    cl_context ctx = NULL;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, NULL) != CL_SUCCESS) return NULL;
    return ctx;
}

static int is_dead(cl_context ctx) {
    pthread_mutex_lock(&g_lock);
    const int dead = ctx && contains(g_dead, ctx);
    pthread_mutex_unlock(&g_lock);
    return dead;
}

static int queue_dead(cl_command_queue queue) { return is_dead(queue_context(queue)); }

#define REAL(name) static __typeof__(&name) real = NULL; if (!real) real = (__typeof__(&name))dlsym(RTLD_NEXT, #name)

CL_API_ENTRY cl_int CL_API_CALL clCompileProgram(cl_program program, cl_uint numDevices, const cl_device_id* devices,
                                                 const char* options, cl_uint numHeaders, const cl_program* headers,
                                                 const char** headerNames,
                                                 void (CL_CALLBACK* notify)(cl_program, void*), void* userData) {
    REAL(clCompileProgram);
    const char* trigger = getenv("AEVUM_FAULT_BUILD_OPT");
    cl_context ctx = NULL;
    clGetProgramInfo(program, CL_PROGRAM_CONTEXT, sizeof(ctx), &ctx, NULL);
    if (is_dead(ctx)) return fault_code();
    if (trigger && *trigger && options && strstr(options, trigger) && ctx) {
        pthread_mutex_lock(&g_lock);
        if (!g_armed_once) {
            g_armed_once = 1;
            insert(g_armed, ctx);
            log_line("ARM\n");
        }
        pthread_mutex_unlock(&g_lock);
    }
    return real(program, numDevices, devices, options, numHeaders, headers, headerNames, notify, userData);
}

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(cl_program program, cl_uint numDevices, const cl_device_id* devices,
                                               const char* options, void (CL_CALLBACK* notify)(cl_program, void*),
                                               void* userData) {
    REAL(clBuildProgram);
    cl_context ctx = NULL;
    clGetProgramInfo(program, CL_PROGRAM_CONTEXT, sizeof(ctx), &ctx, NULL);
    if (is_dead(ctx)) return fault_code();
    return real(program, numDevices, devices, options, notify, userData);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel, cl_uint dim,
                                                       const size_t* offset, const size_t* global, const size_t* local,
                                                       cl_uint nEvents, const cl_event* events, cl_event* event) {
    REAL(clEnqueueNDRangeKernel);
    cl_context ctx = queue_context(queue);
    if (ctx) {
        pthread_mutex_lock(&g_lock);
        int dead = contains(g_dead, ctx);
        if (!dead && contains(g_armed, ctx)) {
            insert(g_dead, ctx);
            dead = 1;
            log_line("KILL\n");
        }
        pthread_mutex_unlock(&g_lock);
        if (dead) return fault_code();
    }
    return real(queue, kernel, dim, offset, global, local, nEvents, events, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue queue, cl_mem buf, cl_bool blocking,
                                                    size_t offset, size_t size, void* ptr, cl_uint nEvents,
                                                    const cl_event* events, cl_event* event) {
    REAL(clEnqueueReadBuffer);
    if (queue_dead(queue)) return fault_code();
    return real(queue, buf, blocking, offset, size, ptr, nEvents, events, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue queue, cl_mem buf, cl_bool blocking,
                                                     size_t offset, size_t size, const void* ptr, cl_uint nEvents,
                                                     const cl_event* events, cl_event* event) {
    REAL(clEnqueueWriteBuffer);
    if (queue_dead(queue)) return fault_code();
    return real(queue, buf, blocking, offset, size, ptr, nEvents, events, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer(cl_command_queue queue, cl_mem src, cl_mem dst, size_t srcOffset,
                                                    size_t dstOffset, size_t size, cl_uint nEvents,
                                                    const cl_event* events, cl_event* event) {
    REAL(clEnqueueCopyBuffer);
    if (queue_dead(queue)) return fault_code();
    return real(queue, src, dst, srcOffset, dstOffset, size, nEvents, events, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer(cl_command_queue queue, cl_mem buf, const void* pattern,
                                                    size_t patternSize, size_t offset, size_t size, cl_uint nEvents,
                                                    const cl_event* events, cl_event* event) {
    REAL(clEnqueueFillBuffer);
    if (queue_dead(queue)) return fault_code();
    return real(queue, buf, pattern, patternSize, offset, size, nEvents, events, event);
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarkerWithWaitList(cl_command_queue queue, cl_uint nEvents,
                                                            const cl_event* events, cl_event* event) {
    REAL(clEnqueueMarkerWithWaitList);
    if (queue_dead(queue)) return fault_code();
    return real(queue, nEvents, events, event);
}

CL_API_ENTRY cl_int CL_API_CALL clFlush(cl_command_queue queue) {
    REAL(clFlush);
    if (queue_dead(queue)) return fault_code();
    return real(queue);
}

CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue queue) {
    REAL(clFinish);
    if (queue_dead(queue)) return fault_code();
    return real(queue);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size, void* host,
                                               cl_int* err) {
    REAL(clCreateBuffer);
    if (is_dead(ctx)) {
        if (err) *err = fault_code();
        return NULL;
    }
    return real(ctx, flags, size, host, err);
}
