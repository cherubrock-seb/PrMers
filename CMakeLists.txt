cmake_minimum_required(VERSION 3.10)
project(PrMers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP   REQUIRED IMPORTED_TARGET gmp)
pkg_check_modules(GMPXX REQUIRED IMPORTED_TARGET gmpxx)

option(USE_CURL "Enable libcurl support" ON)
if (USE_CURL)
  find_package(CURL REQUIRED)
endif()

file(GLOB_RECURSE SOURCES src/*.cpp)
add_executable(prmers ${SOURCES})

target_include_directories(prmers PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  $<TARGET_PROPERTY:PkgConfig::GMP,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:PkgConfig::GMPXX,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_options(prmers PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Wall -Wextra -Wsign-conversion -ffinite-math-only -O2 -O3>
)

target_compile_definitions(prmers PRIVATE
  GPU
  $<$<PLATFORM_ID:Windows>:KERNEL_PATH=\"./kernels/\">
  $<$<NOT:$<PLATFORM_ID:Windows>>:KERNEL_PATH=\"/usr/local/share/prmers/\">
)

if (USE_CURL)
  target_compile_definitions(prmers PRIVATE HAS_CURL=1)
endif()

target_link_libraries(prmers PRIVATE
  OpenCL::OpenCL
  PkgConfig::GMP
  PkgConfig::GMPXX
)

if (WIN32)
  target_link_libraries(prmers PRIVATE ws2_32)
endif()

if (USE_CURL)
  target_link_libraries(prmers PRIVATE CURL::libcurl)
endif()
