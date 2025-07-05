__kernel void kernel_ntt_radix4_mm_2steps(__global ulong* restrict x,
                                          __global ulong* restrict w,
                                          const uint m) {

    const gid_t gid = get_global_id(0);
    const gid_t group = gid / (m / 4);
    const gid_t local_id = gid % (m / 4);
    uint k_first = group * m + local_id;

    //ulong local_x[16];
    __local ulong shared_mem[LOCAL_SIZE2 * 16];
    __local ulong* local_x = shared_mem + get_local_id(0) * 16;
    int write_index = 0;
    ulong2 twiddle1_2;
    ulong twiddle3;

    #pragma unroll 4
    for (uint pass = 0; pass < 4; pass++) {
        const gid_t j = k_first & (m - 1);
        const gid_t i = 4 * (k_first - j) + j;
        const gid_t twiddle_offset = 6 * m + 3 * j;

        twiddle1_2 = vload2(0, w + twiddle_offset);
        twiddle3 = w[twiddle_offset + 2];

        local_x[write_index]    = x[i];
        local_x[write_index+1]  = x[i + m];
        local_x[write_index+2]  = x[i + (m << 1)];
        local_x[write_index+3]  = x[i + ((m << 1) + m)];

        ulong r = modAdd(local_x[write_index], local_x[write_index+2]);
        ulong r2  = modSub(local_x[write_index], local_x[write_index+2]);
        local_x[write_index] = r;
        local_x[write_index + 2] = r2;


        r = modAdd(local_x[write_index+1], local_x[write_index+3]);
        r2 = modMuli(modSub(local_x[write_index+1], local_x[write_index+3]));
        local_x[write_index + 1] = r;
        local_x[write_index+3] = r2;

        r                    =   modAdd(local_x[write_index], local_x[write_index + 1]);
        r2                   =   modSub(local_x[write_index], local_x[write_index + 1]);
        local_x[write_index] = r;
        local_x[write_index + 1] = r2;
        r                          =   modAdd(local_x[write_index + 2], local_x[write_index+3]);
        r2  =   modSub(local_x[write_index + 2], local_x[write_index+3]);
        local_x[write_index + 2] = r;
        local_x[write_index + 3]  = r2;
        
        local_x[write_index+1] = modMul(local_x[write_index+1], twiddle1_2.s1);
        local_x[write_index+2] = modMul(local_x[write_index+2], twiddle1_2.s0);
        local_x[write_index+3] = modMul(local_x[write_index+3], twiddle3);
        write_index += 4;
        k_first += m / 4;
    }

    const uint new_m = m / 4;
    write_index = 0;
    const uint twiddle_offset = 6 * new_m + 3 * local_id;
    k_first = 4 * (group * m) + local_id;
    twiddle1_2 = vload2(0, w + twiddle_offset);
    twiddle3 = w[twiddle_offset + 2];

    #pragma unroll 4
    for (uint pass = 0; pass < 4; pass++) {
        ulong r = modAdd(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+2) % 4) * 4 + (write_index+2) / 4]);
        ulong r2  = modSub(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+2) % 4) * 4 + (write_index+2) / 4]);
        local_x[((write_index) % 4) * 4 + (write_index) / 4] = r;
        local_x[((write_index+2) % 4) * 4 + (write_index+2) / 4]= r2;


        r = modAdd(local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]);
        r2 = modMuli(modSub(local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]));
        local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4] = r;
        local_x[((write_index + 3) % 4) * 4 + (write_index + 3) / 4] = r2;


        x[k_first]                    =   modAdd(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]);
        r2                   =   modSub(local_x[((write_index) % 4) * 4 + (write_index) / 4], local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]);
        local_x[((write_index+1) % 4) * 4 + (write_index+1) / 4]= r2;
        
        r                    =   modAdd(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]);
        r2                   =   modSub(local_x[((write_index + 2) % 4) * 4 + (write_index + 2) / 4], local_x[((write_index+3) % 4) * 4 + (write_index+3) / 4]);

        x[k_first + new_m] = modMul(local_x[((write_index + 1) % 4) * 4 + (write_index + 1) / 4],twiddle1_2.s1);
        x[k_first + (new_m << 1)] = modMul(r,twiddle1_2.s0);
        x[k_first + ((new_m << 1) + new_m)] = modMul(r2,twiddle3);
        write_index += 4;
        k_first += m;
    }

}
