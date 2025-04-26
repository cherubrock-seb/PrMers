#pragma once
#include <string>
#include "common.h"
#include "io/Hash.h"  
namespace io {

struct MD5Context {
  int isInit;
  unsigned buf[4];
  unsigned bits[2];
  unsigned char in[64];
};

void MD5Init(MD5Context*);
void MD5Update(MD5Context*, const unsigned char*, unsigned int);
void MD5Final(unsigned char digest[16], MD5Context*);

class MD5Hash {
  MD5Context context;

public:
  MD5Hash() { MD5Init(&context); }
  void update(const void* data, u32 size) { MD5Update(&context, reinterpret_cast<const unsigned char*>(data), size); }

  std::string finish() && {
    unsigned char digest[16];
    MD5Final(digest, &context);
    std::string s;
    char hex[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) {
      s.push_back(hex[digest[i] >> 4]);
      s.push_back(hex[digest[i] & 0xf]);
    }
    return s;
  }
};

using MD5 = Hash<MD5Hash>;

} // namespace io
