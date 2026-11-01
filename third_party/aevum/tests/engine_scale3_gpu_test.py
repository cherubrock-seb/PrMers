#!/usr/bin/env python3
"""Full-word differential test of AEVUM_REG_SCALE3 using an existing engine.
Requires a real supported device. Does not build the engine or measure speed.
"""
import argparse
import ctypes as C
import os
import random
import tempfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--exponent", type=int, default=21000029)
    parser.add_argument("--fft", default="1:256:8:256:202")
    parser.add_argument("--expect-capacity-rejection", action="store_true")
    args = parser.parse_args()
    api = C.CDLL(os.path.abspath(args.library))
    handle, size, u32 = C.c_void_p, C.c_size_t, C.c_uint32
    words = C.POINTER(u32)
    declarations = {
        "create": (handle, [u32, size, u32, C.c_int, C.c_char_p, C.c_char_p]),
        "destroy": (None, [handle]),
        "word_count": (size, [handle]),
        "last_error": (C.c_char_p, []),
        "set_words": (C.c_int, [handle, size, words, size]),
        "get_words": (C.c_int, [handle, size, words, size]),
        "set_u32": (C.c_int, [handle, size, u32]),
        "square_mul": (C.c_int, [handle, size, u32]),
        "mul": (C.c_int, [handle, size, size, u32]),
        "prepare": (C.c_int, [handle, size, size]),
        "copy": (C.c_int, [handle, size, size]),
        "sub_u32": (C.c_int, [handle, size, u32]),
        "sync": (C.c_int, [handle]),
    }
    for name, (result, params) in declarations.items():
        fn = getattr(api, "aevum_engine_" + name)
        fn.restype, fn.argtypes = result, params

    def call(name, *values):
        result = getattr(api, "aevum_engine_" + name)(*values)
        if not result:
            raise RuntimeError(name + ": " + api.aevum_engine_last_error().decode())
        return result

    previous = os.environ.get("AEVUM_REG_SCALE3")
    handles = []
    try:
        with tempfile.TemporaryDirectory(prefix="aevum-scale3-") as directory:
            # The flag is captured per engine: both paths coexist in one process.
            for enabled in ("0", "1"):
                os.environ["AEVUM_REG_SCALE3"] = enabled
                h = call("create", args.exponent, 8, args.device, 1,
                         args.fft.encode(), directory.encode())
                handles.append(h)
            count = call("word_count", handles[0])
            if call("word_count", handles[1]) != count:
                raise RuntimeError("word count mismatch")
            array = u32 * count

            def both(name, *values):
                for h in handles:
                    call(name, h, *values)

            def compare(reg=0):
                outputs = []
                for h in handles:
                    out = array()
                    call("get_words", h, reg, out, count)
                    outputs.append(bytes(out))
                if outputs[0] != outputs[1]:
                    raise RuntimeError("full-word mismatch in register " + str(reg))
                return outputs[0]

            if args.expect_capacity_rejection:
                for h in handles:
                    call("set_u32", h, 0, 3)
                    if api.aevum_engine_square_mul(h, 0, 3):
                        raise RuntimeError("unsafe factor-3 plan was accepted")
                    error = api.aevum_engine_last_error().decode()
                    if "small-factor capacity exceeded" not in error:
                        raise RuntimeError("unexpected rejection: " + error)
                print("both paths rejected insufficient factor-3 capacity")
                return

            both("set_u32", 0, 3)
            both("square_mul", 0, 3)
            out = compare()
            expected = array()
            expected[0] = 27
            if out != bytes(expected):
                raise RuntimeError("3 * 3^2 did not produce 27")

            rng = random.Random(0x5343414C4533)
            dense = array.from_buffer_copy(rng.randbytes(count * 4))
            if args.exponent % 32:
                dense[-1] &= (1 << (args.exponent % 32)) - 1
            high = array(*([0xffffffff] * count))
            if args.exponent % 32:
                high[-1] &= (1 << (args.exponent % 32)) - 1
            high[0] -= 1  # 2^E - 2: exercises negative balanced digits.
            alternating = array(*([0xaaaaaaaa, 0x55555555] * ((count + 1) // 2))[:count])
            if args.exponent % 32:
                alternating[-1] &= (1 << (args.exponent % 32)) - 1

            for seed in (array(), high, alternating, dense):
                both("set_words", 0, seed, count)
                both("set_u32", 1, 5)
                both("prepare", 2, 1)
                # Pending pure squares, factor-3 canonical boundary, other factors.
                for factor in (1, 1, 3, 1, 3, 2, 5, 3):
                    both("square_mul", 0, factor)
                compare()
                # The same helper handles prepared multiplication with factor 3.
                both("mul", 0, 2, 3)
                both("copy", 3, 0)
                compare(3)
                # Prepared-cache invalidation and another register boundary.
                both("prepare", 4, 0)
                both("square_mul", 0, 3)
                both("mul", 3, 4, 3)
                compare(0)
                compare(3)
                both("square_mul", 0, 1)
                both("sub_u32", 0, 2)
                both("square_mul", 1, 3)
                both("sync")
                compare(0)
                compare(1)
                # Canonical readback can be uploaded unchanged (checkpoint boundary).
                snapshot = array.from_buffer_copy(compare())
                both("set_words", 0, snapshot, count)
                both("square_mul", 0, 3)
                compare()

            for h in handles:
                if api.aevum_engine_square_mul(h, 0, 0):
                    raise RuntimeError("factor zero was accepted")
                if api.aevum_engine_square_mul(h, 8, 3):
                    raise RuntimeError("out-of-range register was accepted")
            print("scale3 full-word differential and API boundary checks completed")
    finally:
        for h in handles:
            api.aevum_engine_destroy(h)
        if previous is None:
            os.environ.pop("AEVUM_REG_SCALE3", None)
        else:
            os.environ["AEVUM_REG_SCALE3"] = previous

if __name__ == "__main__":
    main()
