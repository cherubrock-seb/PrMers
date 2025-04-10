cmake_minimum_required(VERSION 3.10)
project(PrMers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# Define executable
add_executable(prmers
    prmers.cpp
    proof/common.cpp
    proof/proof.cpp
    proof/md5.cpp
    proof/sha3.cpp
)

# Define macro for kernel path
if (WIN32)
    set(DEFAULT_KERNEL_PATH "\"./kernels/\"")
else()
    set(DEFAULT_KERNEL_PATH "\"/usr/local/share/prmers/\"")
endif()
target_compile_definitions(prmers PRIVATE KERNEL_PATH=${DEFAULT_KERNEL_PATH})

# Optional: Enable or disable libcurl
option(USE_CURL "Enable libcurl support" ON)

if (USE_CURL)
    find_package(CURL)
    if (CURL_FOUND)
        message(STATUS "libcurl found, enabling CURL support.")
        include_directories(${CURL_INCLUDE_DIRS})
        target_link_libraries(prmers ${CURL_LIBRARIES})
        target_compile_definitions(prmers PRIVATE HAS_CURL=1)
    else()
        message(WARNING "libcurl not found. Continuing without CURL support.")
    endif()
else()
    message(STATUS "libcurl support is disabled by option.")
endif()

# Link OpenCL last
target_link_libraries(prmers ${OpenCL_LIBRARIES})
