// Copyright (C) Mihai Preda.

#include "Signal.h"

#include <csignal>

using namespace std;

static volatile sig_atomic_t signalled = 0;

static void (* volatile oldIntHandler)(int) = 0;
static void (* volatile oldTermHandler)(int) = 0;

// A second Ctrl-C ends the process at once, for when the graceful stop is not coming (or is too slow).  Windows' CRT already behaves this
// way (it resets the handler to SIG_DFL before calling it); POSIX signal() keeps the handler, so do it here.  signal() and raise() are
// async-signal-safe.
static void signalHandler(int sig) {
  if (sig == SIGINT && signalled == SIGINT) {
    signal(SIGINT, SIG_DFL);
    raise(SIGINT);
    return;
  }
  signalled = sig;
}

Signal::Signal() {
  if (!oldIntHandler) {
    oldIntHandler = signal(SIGINT, signalHandler);
    // SIGTERM is what service managers and `kill` send first; unhandled it ends the process at once, without the final savefile.
    // Windows defines SIGTERM but never delivers it from the console, so this is harmless there.
    oldTermHandler = signal(SIGTERM, signalHandler);
    isOwner = true;
  }
}

Signal::~Signal() { release(); }

unsigned Signal::stopRequested() { return signalled; }

void Signal::requestStop() { signalled = SIGTERM; }

void Signal::release() {
  if (isOwner) {
    isOwner = false;
    signal(SIGINT, oldIntHandler);
    signal(SIGTERM, oldTermHandler);
    oldIntHandler = 0;
    oldTermHandler = 0;
  }
}
