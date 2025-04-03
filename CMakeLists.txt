cmake_minimum_required(VERSION 3.10)
project(PrMers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# Add executable
add_executable(prmers
    prmers.cpp
    proof/common.cpp
    proof/proof.cpp
    proof/md5.cpp
    proof/sha3.cpp
)

target_link_libraries(prmers ${OpenCL_LIBRARIES})

# Define macro for kernel path
target_compile_definitions(prmers PRIVATE KERNEL_PATH="\"/usr/local/share/prmers/\"")
