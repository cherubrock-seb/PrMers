#pragma once
namespace core { namespace algo {

// Call first thing in main(), before any OpenCL call. Records whether SIGHUP was inherited as ignored
// (`nohup prmers ...`). An OpenCL runtime that embeds LLVM (PoCL, ROCm) replaces the SIGHUP/SIGINT/SIGTERM
// dispositions when it initialises, ignored ones included, so by the time install_stop_handlers() runs the
// original state can no longer be read from the process.
void note_inherited_signals() noexcept;
// True when note_inherited_signals() saw SIGHUP ignored.
bool sighup_inherited_ignored() noexcept;

}} // namespace core::algo
