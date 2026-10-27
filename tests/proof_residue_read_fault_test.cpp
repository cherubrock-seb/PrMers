// A proof residue whose GPU read fails must not be saved.  ProofManager::checkpoint ignored the status of its
// clEnqueueReadBuffer, so a failed read saved the all-zero host buffer as the residue of that iteration (and the
// read-back check passed, since the file held exactly what was written).  It must now throw (a lost GPU as
// util::gpulost::GpuLostError) and leave no residue file.
//
// clEnqueueReadBuffer is defined here, so it replaces the OpenCL library's for the code linked into this test;
// no OpenCL device is used.  usage: proof-residue-read-fault-test <scratch dir>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/ProofManager.hpp"
#include "core/ProofManagerMarin.hpp"
#include "core/ProofSet.hpp"
#include "util/GpuLost.hpp"

namespace {
cl_int g_status = CL_SUCCESS;  // what the next read returns
int g_reads = 0;
int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}
}  // namespace

extern "C" CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t size,
                                                               void* ptr, cl_uint, const cl_event*, cl_event*) {
    ++g_reads;
    if (g_status != CL_SUCCESS) return g_status;
    auto* words = static_cast<uint64_t*>(ptr);
    for (size_t i = 0; i < size / sizeof(uint64_t); ++i) words[i] = (i * 7 + 3) & 0xFFFF;
    return CL_SUCCESS;
}

template <class Manager>
void run(const char* name, const std::filesystem::path& dir) {
    const uint32_t p = 21701, n = 1024;
    const int power = 2;
    std::vector<int> width(n, static_cast<int>(p / n));
    for (uint32_t i = 0; i < p % n; ++i) ++width[i];

    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const core::ProofSet points(p, power, {}, core::ProofLocation(dir.string()));
    std::vector<uint32_t> iters;
    for (uint32_t k = 1; k < p && iters.size() < 2; ++k)
        if (points.shouldCheckpoint(k)) iters.push_back(k);
    if (iters.size() < 2) { expect(false, std::string(name) + ": no checkpoint iterations"); return; }

    Manager pm(p, power, reinterpret_cast<cl_command_queue>(0x1), n, width, {}, dir.string());
    const core::ProofLocation where(dir.string());
    auto cl_buf = reinterpret_cast<cl_mem>(0x2);

    // A good read saves the residue (the test's own read replacement is the one in use).
    g_status = CL_SUCCESS;
    g_reads = 0;
    pm.checkpoint(cl_buf, iters[0]);
    expect(g_reads == 1, std::string(name) + ": the read replacement was not called");
    expect(std::filesystem::exists(where.residueWriteFile(p, iters[0])), std::string(name) + ": good residue not saved");

    // A lost GPU: GpuLostError, nothing saved.
    for (const cl_int status : {cl_int(-36), cl_int(-5), cl_int(-14)}) {
        g_status = status;
        bool lost = false, other = false;
        try {
            pm.checkpoint(cl_buf, iters[1]);
        } catch (const util::gpulost::GpuLostError&) {
            lost = true;
        } catch (const std::exception& e) {
            other = true;
            expect(std::string(e.what()).find("proof residue") != std::string::npos,
                   std::string(name) + ": unclear message: " + e.what());
        }
        const std::string tag = std::string(name) + " status " + std::to_string(status);
        if (status == -14) expect(other, tag + ": a failed read did not throw");
        else expect(lost, tag + ": a lost GPU did not throw GpuLostError");
        expect(!std::filesystem::exists(where.residueWriteFile(p, iters[1])), tag + ": a residue was saved from a failed read");
    }
    std::filesystem::remove_all(dir);
}

int main(int argc, char** argv) {
    const std::filesystem::path base = argc > 1 ? argv[1] : "proof-residue-read-fault";
    run<core::ProofManager>("ProofManager", base / "legacy");
    run<core::ProofManagerMarin>("ProofManagerMarin", base / "marin");
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Proof residue read fault test: PASS\n";
    return 0;
}
