// Host test: a command that terminated abnormally must not make the queue wait for ever.
//
// OpenCL reports the execution status of an errored command as a negative cl_int.  Read as an unsigned value that
// is huge and never equals CL_COMPLETE, so Queue::waitForMarkerEvent() (the non-AMD wait path) used to sleep for
// ever.  A user event completed with a negative status stands in for the errored marker: nothing is enqueued on
// the device, so this is safe on any OpenCL device.  The test is compiled with -fno-access-control to reach the
// Queue's marker state.

#include "Queue.h"
#include "Context.h"
#include "Event.h"
#include "TimeInfo.h"
#include "clwrap.h"
#include "log.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unistd.h>

// The bundled tinycl.h has no user events; these are standard OpenCL 1.1 entry points.
extern "C" {
cl_event clCreateUserEvent(cl_context, int* errcode_ret);
int clSetUserEventStatus(cl_event, int execution_status);
}

static const int CL_OK = 0;

static int failures = 0;

#define EXPECT(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } else { std::printf("ok:   %s\n", msg); } } while (0)

static void onAlarm(int) {
  // Unbounded wait: report it and bail without running destructors.
  const char msg[] = "FAIL: timed out, the wait never returned for an errored event\n";
  (void) !write(1, msg, sizeof(msg) - 1);
  _exit(1);
}

static EventHolder makeUserEvent(cl_context ctx) {
  int err = CL_OK;
  cl_event ev = clCreateUserEvent(ctx, &err);
  CHECK1(err);
  return EventHolder{ev};
}

static EventHolder finishedUserEvent(cl_context ctx, int status) {
  EventHolder ev = makeUserEvent(ctx);
  CHECK1(clSetUserEventStatus(ev.get(), status));
  return ev;
}

int main() {
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) {
    std::printf("SKIP: no OpenCL device\n");
    return 0;
  }
  std::setbuf(stdout, nullptr);
  std::signal(SIGALRM, onAlarm);
  alarm(20);

  Context context{devices[0]};
  const int bad = CL_OUT_OF_RESOURCES;  // negative

  {
    EventHolder ev = finishedUserEvent(context.get(), bad);
    int status = getEventInfo(ev.get());
    EXPECT(status == bad, "getEventInfo returns the negative error status");
  }
  {
    EventHolder ev = makeUserEvent(context.get());
    EXPECT(getEventInfo(ev.get()) >= 0 && getEventInfo(ev.get()) != CL_COMPLETE, "a pending event is not complete");
    CHECK1(clSetUserEventStatus(ev.get(), CL_COMPLETE));
    EXPECT(getEventInfo(ev.get()) == CL_COMPLETE, "a finished event reports CL_COMPLETE");
  }
  {
    TimeInfo tInfo{"test"};
    Event e{finishedUserEvent(context.get(), bad), &tInfo};
    EXPECT(e.isComplete(), "Event::isComplete retires an errored event");
    EXPECT(tInfo.n == 0, "no timestamps are collected for an errored event");
  }
  {
    Queue queue{context, false};
    queue.markerEvent = finishedUserEvent(context.get(), bad);
    queue.markerQueued = true;
    bool threw = false;
    std::string what;
    try {
      queue.waitForMarkerEvent();
    } catch (const std::runtime_error& e) {
      threw = true;
      what = e.what();
    }
    EXPECT(threw, "Queue::waitForMarkerEvent throws on an errored marker");
    EXPECT(what.find(errMes(bad)) != std::string::npos, "the thrown error names the command's status");
    queue.markerQueued = false;
  }
  {
    Queue queue{context, false};
    queue.markerEvent = finishedUserEvent(context.get(), CL_COMPLETE);
    queue.markerQueued = true;
    queue.waitForMarkerEvent();
    EXPECT(!queue.markerQueued, "Queue::waitForMarkerEvent still returns normally for a completed marker");
  }
  alarm(0);
  return failures ? 1 : 0;
}
