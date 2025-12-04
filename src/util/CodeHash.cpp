#include "util/CodeHash.hpp"
#include "util/Crc32.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdint>

#if defined(__linux__)
  #include <link.h>
  #include <cstring>
#elif defined(_WIN32)
  #include <windows.h>
  #include <winnt.h>
#elif defined(__APPLE__)
  #include <mach-o/loader.h>
  #include <mach-o/dyld.h>
#endif

namespace {

static std::string hex8_upper(unsigned x){
    std::ostringstream o;
    o << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << x;
    return o.str();
}

}

namespace util {

std::string code_hash_crc32_upper8(){
    std::string buf;

#if defined(__linux__)

    struct Ctx { std::string* out; bool done; };
    auto cb = [](struct dl_phdr_info* info, size_t, void* data)->int{
        Ctx* c = reinterpret_cast<Ctx*>(data);
        if (c->done) return 1;
        for (int i = 0; i < info->dlpi_phnum; ++i){
            const ElfW(Phdr)& ph = info->dlpi_phdr[i];
            if (ph.p_type == PT_LOAD && (ph.p_flags & PF_X)){
                const unsigned char* p = reinterpret_cast<const unsigned char*>(info->dlpi_addr + ph.p_vaddr);
                size_t n = static_cast<size_t>(ph.p_memsz);
                c->out->append(reinterpret_cast<const char*>(p), n);
            }
        }
        c->done = true;
        return 1;
    };
    Ctx ctx{ &buf, false };
    dl_iterate_phdr(cb, &ctx);

#elif defined(_WIN32)

    HMODULE h = GetModuleHandleW(nullptr);
    if (h){
        const uint8_t* base = reinterpret_cast<const uint8_t*>(h);
        auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        if (dos->e_magic == IMAGE_DOS_SIGNATURE){
            auto nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
            if (nt->Signature == IMAGE_NT_SIGNATURE){
                const IMAGE_FILE_HEADER& fh = nt->FileHeader;
                const IMAGE_SECTION_HEADER* sec = IMAGE_FIRST_SECTION(nt);
                for (unsigned i = 0; i < fh.NumberOfSections; ++i, ++sec){
                    DWORD ch = sec->Characteristics;
                    if ((ch & IMAGE_SCN_CNT_CODE) || (ch & IMAGE_SCN_MEM_EXECUTE)){
                        const uint8_t* p = base + sec->VirtualAddress;
                        size_t n = (sec->Misc.VirtualSize ? sec->Misc.VirtualSize : sec->SizeOfRawData);
                        buf.append(reinterpret_cast<const char*>(p), n);
                    }
                }
            }
        }
    }

#elif defined(__APPLE__)

    const mach_header* mh = _dyld_get_image_header(0);
    if (!mh) return hex8_upper(0);
#if defined(__LP64__)
    const mach_header_64* mh64 = reinterpret_cast<const mach_header_64*>(mh);
    intptr_t slide = _dyld_get_image_vmaddr_slide(0);
    const uint8_t* cmdp = reinterpret_cast<const uint8_t*>(mh64) + sizeof(mach_header_64);
    for (uint32_t i = 0; i < mh64->ncmds; ++i){
        const load_command* lc = reinterpret_cast<const load_command*>(cmdp);
        if (lc->cmd == LC_SEGMENT_64){
            const segment_command_64* sg = reinterpret_cast<const segment_command_64*>(lc);
            if (std::string(sg->segname) == "__TEXT"){
                const section_64* sec = reinterpret_cast<const section_64*>(sg + 1);
                for (uint32_t j = 0; j < sg->nsects; ++j, ++sec){
                    if (std::string(sec->sectname) == "__text"){
                        const uint8_t* p = reinterpret_cast<const uint8_t*>(sec->addr + slide);
                        size_t n = static_cast<size_t>(sec->size);
                        buf.append(reinterpret_cast<const char*>(p), n);
                    }
                }
            }
        }
        cmdp += lc->cmdsize;
    }
#else
    const mach_header* mh32 = mh;
    intptr_t slide = _dyld_get_image_vmaddr_slide(0);
    const uint8_t* cmdp = reinterpret_cast<const uint8_t*>(mh32) + sizeof(mach_header);
    for (uint32_t i = 0; i < mh32->ncmds; ++i){
        const load_command* lc = reinterpret_cast<const load_command*>(cmdp);
        if (lc->cmd == LC_SEGMENT){
            const segment_command* sg = reinterpret_cast<const segment_command*>(lc);
            if (std::string(sg->segname) == "__TEXT"){
                const section* sec = reinterpret_cast<const section*>(sg + 1);
                for (uint32_t j = 0; j < sg->nsects; ++j, ++sec){
                    if (std::string(sec->sectname) == "__text"){
                        const uint8_t* p = reinterpret_cast<const uint8_t*>(sec->addr + slide);
                        size_t n = static_cast<size_t>(sec->size);
                        buf.append(reinterpret_cast<const char*>(p), n);
                    }
                }
            }
        }
        cmdp += lc->cmdsize;
    }
#endif

#else
    return hex8_upper(0);
#endif

    unsigned crc = computeCRC32(buf);
    return hex8_upper(crc);
}

}
