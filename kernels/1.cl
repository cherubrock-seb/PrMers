__kernel void kernel_ntt_radix4_inverse_mm_2steps(__global ulong* restrict x,
                                                  __global ulong* restrict wi,
                                                  const uint m) {
    __local ulong shared_mem[LOCAL_SIZE2 * 16];
    __local ulong* local_x = shared_mem + get_local_id(0) * 16;
    int write_index = 0;
    const gid_t gid        = get_global_id(0);
    const gid_t group      = gid / m;
    const gid_t local_id   = gid % m;
    uint k_first          = group * m * 4 + local_id;
    

    uint base              = 4 * (k_first - local_id) + local_id;
    const uint  tw_offset  = 6 * m + 3 * local_id;
    ulong2 tw1_2            = vload2(0, wi + tw_offset);
    ulong tw3              = wi[tw_offset + 2];
    ulong r, r2;

    #pragma unroll 4
    for (int pass = 0; pass < 4; pass++) {
        local_x[write_index    ] = x[base];
        local_x[write_index + 1] = modMul(x[base + m],           tw1_2.s1);
        local_x[write_index + 2] = modMul(x[base + (m << 1)],    tw1_2.s0);
        local_x[write_index + 3] = modMul(x[base + ((m << 1) + m)], tw3);

        r  = modAdd(local_x[write_index    ], local_x[write_index + 1]);
        r2 = modSub(local_x[write_index    ], local_x[write_index + 1]);
        local_x[write_index    ] = r;
        local_x[write_index + 1] = r2;

        r  = modAdd(local_x[write_index + 2], local_x[write_index + 3]);
        r2 = modMuli(modSub(local_x[write_index + 3], local_x[write_index + 2]));
        local_x[write_index + 2] = r;
        local_x[write_index + 3] = r2;

        r  = modAdd(local_x[write_index    ], local_x[write_index + 2]);
        r2 = modSub(local_x[write_index    ], local_x[write_index + 2]);
        local_x[write_index    ] = r;
        local_x[write_index + 2] = r2;
        r  = modAdd(local_x[write_index + 1], local_x[write_index + 3]);
        local_x[write_index + 3] = modSub(local_x[write_index + 1], local_x[write_index + 3]);
        local_x[write_index + 1] = r;

        write_index += 4;
        base        += 4 * m;
        k_first     += m;
    }

    const uint new_m     = m * 4;
    write_index          = 0;
    k_first       = group * m * 4 + local_id;

    #pragma unroll 4
    for (int pass = 0; pass < 4; pass++) {
        const gid_t j2            = k_first & (new_m - 1);
        const gid_t base2   = 4 * (k_first - j2) + j2;
        const gid_t tw_off2 = 6 * new_m + 3 * j2;
        tw1_2 = vload2(0, wi + tw_off2);
        tw3 = wi[tw_off2 + 2];

        local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4] = modMul(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4], tw1_2.s1);
        local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4] = modMul(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4], tw1_2.s0);
        local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4] = modMul(local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4], tw3);

        r  = modAdd(local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4],
                    local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4]);
        r2 = modSub(local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4],
                    local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4]);
        local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4] = r;
        local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4] = r2;

        r  = modAdd(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4],
                    local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4]);
        r2 = modMuli(modSub(local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4],
                            local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4]));

        x[base2]                            = modAdd(local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4], r);
        x[base2 + (new_m << 1)]            = modSub(local_x[((write_index    ) % 4) * 4 + (write_index    ) / 4], r);
        x[base2 + new_m]                    = modAdd(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4], r2);
        x[base2 + ((new_m << 1) + new_m)]  = modSub(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4], r2);

        write_index += 4;
        k_first    += m;
    }
}