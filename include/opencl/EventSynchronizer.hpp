// src/opencl/EventSynchronizer.hpp
#pragma once

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#include <vector>
#include <cassert>

namespace opencl {

class EventSynchronizer {
public:
    EventSynchronizer() = default;

    void addEvent(cl_event evt) {
        events_.push_back(evt);
    }

    void clearCompleted() {
        auto it = events_.begin();
        while (it != events_.end()) {
            cl_int status;
            clGetEventInfo(*it,
                           CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status,
                           nullptr);
            if (status == CL_COMPLETE) {
                clReleaseEvent(*it);
                it = events_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void synced() {
        clearCompleted();
        assert(events_.empty() && "Some events still pending after clFinish()");
    }

    void waitAll() {
        synced();
    }

private:
    std::vector<cl_event> events_;
};

} // namespace opencl
