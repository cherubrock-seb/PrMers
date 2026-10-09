#!/usr/bin/env python3
"""Inspect and rewrite a P-1 stage-1 checkpoint (pm1_m_<p>.ckpt) for tests.

  pm1_ckpt_tool.py info FILE             print "<version> <p> <counter> <crc ok 0|1>"
  pm1_ckpt_tool.py to-v3 FILE            rewrite FILE in the old 32-bit-counter format
  pm1_ckpt_tool.py set-counter FILE N    store counter N (version 4) with a valid CRC
  pm1_ckpt_tool.py truncate FILE N       keep only the first N bytes

File layout: int32 version | uint32 p | counter (uint32 in v3, uint64 in v4) |
double et | register image ... | uint32 CRC, where the CRC is
~crc32(all bytes before it) ^ 0xa23777ac.
"""
import struct
import sys
import zlib

MAGIC = 0xA23777AC


def crc_trailer(body: bytes) -> bytes:
    return struct.pack("<I", (~zlib.crc32(body) & 0xFFFFFFFF) ^ MAGIC)


def parse(data: bytes):
    version, p = struct.unpack_from("<iI", data, 0)
    if version == 3:
        (counter,) = struct.unpack_from("<I", data, 8)
        hdr = 12
    elif version == 4:
        (counter,) = struct.unpack_from("<Q", data, 8)
        hdr = 16
    else:
        raise SystemExit("unsupported version %d" % version)
    crc_ok = data[-4:] == crc_trailer(data[:-4])
    return version, p, counter, hdr, crc_ok


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    cmd, path = argv[1], argv[2]
    with open(path, "rb") as f:
        data = f.read()
    if cmd == "truncate":
        with open(path, "wb") as f:
            f.write(data[: int(argv[3])])
        return
    version, p, counter, hdr, crc_ok = parse(data)
    if cmd == "info":
        print(version, p, counter, 1 if crc_ok else 0)
        return
    if not crc_ok:
        raise SystemExit("input checkpoint has a bad CRC")
    body = data[:-4]
    if cmd == "to-v3":
        if version != 4:
            raise SystemExit("not a version 4 file")
        if counter >= 1 << 32:
            raise SystemExit("counter does not fit in 32 bits")
        body = struct.pack("<iII", 3, p, counter) + body[hdr:]
    elif cmd == "set-counter":
        if version != 4:
            raise SystemExit("not a version 4 file")
        body = body[:8] + struct.pack("<Q", int(argv[3])) + body[hdr:]
    else:
        raise SystemExit("unknown command " + cmd)
    with open(path, "wb") as f:
        f.write(body + crc_trailer(body))


if __name__ == "__main__":
    main(sys.argv)
