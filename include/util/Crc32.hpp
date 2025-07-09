#ifndef CRC32_HPP
#define CRC32_HPP
#include <string>
#include <cstdint>
#include <cstddef>
uint32_t computeCRC32(const std::string &data);
uint32_t computeCRC32(const void* data, size_t size);
std::string toLower(const std::string &s);
#endif
