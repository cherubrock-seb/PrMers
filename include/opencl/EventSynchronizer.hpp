// src/opencl/EventSynchronizer.hpp
#pragma once

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <cassert>
#include <cstdint>
#include <algorithm>

namespace opencl {

static std::array<uint64_t,3> getEventNanos(cl_event evt) {
    uint64_t tQ=0, tS=0, tSt=0, tE=0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_QUEUED,  sizeof(tQ),  &tQ,  nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_SUBMIT,  sizeof(tS),  &tS,  nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,   sizeof(tSt), &tSt, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,     sizeof(tE),  &tE,  nullptr);
    return { tS - tQ, tSt - tS, tE - tSt };
}

class EventSynchronizer {
public:
    EventSynchronizer() = default;

    void addEvent(cl_event evt) {
        events_.push_back(evt);
    }

    void waitAll(bool update, int waitPercentageFactor) {
        if (update || !profiled_) {
            totalUs_ = std::chrono::microseconds{0};
            lastTotalUs_ = std::chrono::microseconds{0};
            count_ = 0;
            while (!events_.empty()) {
                clearCompleted();
            }
            lastTotalUs_ = totalUs_;
            profiled_    = true;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(lastTotalUs_).count();
            std::cout
                << "[EventSynchronizer] profiled: "
                << lastTotalUs_.count() << " µs"
                << " (" << ms << " ms)\n";
        }
        else {
            int64_t waitCount = lastTotalUs_.count() * waitPercentageFactor / 100;
            std::this_thread::sleep_for(std::chrono::microseconds(waitCount));
            clearAllEvents();    
        }
        assert(events_.empty());
    }

private:
    std::vector<cl_event>       events_;
    bool                        profiled_    = false;
    std::chrono::microseconds   totalUs_{0};
    std::chrono::microseconds   lastTotalUs_{0};
    size_t                      count_{0};

    void clearAllEvents() {
        for (auto &evt : events_) {
            clReleaseEvent(evt);
        }
        events_.clear();
    }

    void clearCompleted() {
        auto it = events_.begin();
        while (it != events_.end()) {
            cl_int status = 0;
            clGetEventInfo(*it, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, nullptr);
            if (status == CL_COMPLETE) {
                auto arr = getEventNanos(*it);
                uint64_t execNs = arr[2];
                totalUs_ += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::nanoseconds(execNs)
                );
                ++count_;
                clReleaseEvent(*it);
                it = events_.erase(it);
            } else {
                ++it;
            }
        }
    }
};

} // namespace opencl
