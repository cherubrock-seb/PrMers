cmake_minimum_required(VERSION 3.10)
project(PrMers LANGUAGES CXX)

# ------------------------------------------------------------------
#  Global compiler settings
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
#  OpenCL
# ------------------------------------------------------------------
find_package(OpenCL REQUIRED)

# ------------------------------------------------------------------
#  GMP (via pkg-config)
#  vcpkg exports .pc files on every platform, so this works on Windows
# ------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP   REQUIRED IMPORTED_TARGET gmp)
pkg_check_modules(GMPXX REQUIRED IMPORTED_TARGET gmpxx)

# ------------------------------------------------------------------
#  libcurl (optional)
# ------------------------------------------------------------------
option(USE_CURL "Enable libcurl support" ON)
if (USE_CURL)
  find_package(CURL REQUIRED)
endif()

# ------------------------------------------------------------------
#  Sources & executable
# ------------------------------------------------------------------
file(GLOB_RECURSE SOURCES src/*.cpp)
add_executable(prmers ${SOURCES})

# ------------------------------------------------------------------
#  Include directories (only what is not carried by imported targets)
# ------------------------------------------------------------------
target_include_directories(prmers PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  $<TARGET_PROPERTY:PkgConfig::GMP,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:PkgConfig::GMPXX,INTERFACE_INCLUDE_DIRECTORIES>
)

# ------------------------------------------------------------------
#  Compile-time definitions
#  (quotes escaped so that the macro expands to a string literal)
# ------------------------------------------------------------------
target_compile_definitions(prmers PRIVATE
  $<$<PLATFORM_ID:Windows>:KERNEL_PATH=\"./kernels/\">
  $<$<NOT:$<PLATFORM_ID:Windows>>:KERNEL_PATH=\"/usr/local/share/prmers/\">
)

if (USE_CURL)
  target_compile_definitions(prmers PRIVATE HAS_CURL=1)
endif()

# ------------------------------------------------------------------
#  Link libraries
# ------------------------------------------------------------------
target_link_libraries(prmers PRIVATE
  OpenCL::OpenCL
  PkgConfig::GMP
  PkgConfig::GMPXX
)

if (USE_CURL)
  target_link_libraries(prmers PRIVATE CURL::libcurl)
endif()
