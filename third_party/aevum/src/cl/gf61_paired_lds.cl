// PRPLL 4fe779cab5b6b8b669c37acad71efb037fd4bd5a: factor byte addresses
// and put same-base stores next to each other, allowing paired LDS accesses.
// Only the GF61 wrapper calls this; GF31/FP32 keep their production exchange.
#if NTT_GF61 && WG == 64 && RADIX == 8 && SHUFL_BYTES == 8 && \
    ((DOING_WIDTH && AEVUM_GF61_PAIRED_LDS_W) || (DOING_HEIGHT && AEVUM_GF61_PAIRED_LDS_H))

#if HAS_ASM
#define AEVUM_PAIRED_LDS_OPAQUE(x) __asm volatile("" : "+v"(x))
#else
#define AEVUM_PAIRED_LDS_OPAQUE(x)
#endif

void aevumGf61PairedLds(local GF61 *lds2, GF61 *u, u32 f, u32 numWG, u32 lowMe) {
  // XOR offsets within this transform's partition, not the aggregate buffer.
  // Native LDSPAD can give partitions a 4544-byte stride, not a power of two.
  local char *root = (local char *)lds2;
  if (numWG > 1) root += (get_local_id(0) / WG) * LDS_BYTES;
  if (f == 1) {
    bar(WG);
    u32 wa = ((lowMe * 8) ^ (lowMe & 15)) * 8;
    u32 ra = (lowMe ^ ((lowMe / 8) & 7)) * 8;
    AEVUM_PAIRED_LDS_OPAQUE(wa); AEVUM_PAIRED_LDS_OPAQUE(ra);
    for (u32 i = 0; i < 8; ++i) *(local u64 *)(root + (wa ^ (i * 8))) = u[i].x;
    bar(WG);
    for (u32 i = 0; i < 8; ++i) u[i].x = *(local u64 *)(root + i * WG * 8 + (ra ^ ((i & 1) * 64)));
    bar(WG);
    AEVUM_PAIRED_LDS_OPAQUE(wa); AEVUM_PAIRED_LDS_OPAQUE(ra);
    for (u32 i = 0; i < 8; ++i) *(local u64 *)(root + (wa ^ (i * 8))) = u[i].y;
    bar(WG);
    for (u32 i = 0; i < 8; ++i) u[i].y = *(local u64 *)(root + i * WG * 8 + (ra ^ ((i & 1) * 64)));
  } else {
    // Dispatch admits f=8 only here. Even/odd stores expose constant offsets.
    bar(WG);
    u32 wa = (((lowMe / 8) * 64 + (lowMe & 7)) ^ (lowMe & 8)) * 8;
    u32 ra = lowMe * 8;
    AEVUM_PAIRED_LDS_OPAQUE(wa); AEVUM_PAIRED_LDS_OPAQUE(ra);
    for (u32 i = 0; i < 8; i += 2) *(local u64 *)(root + wa + i * 64) = u[i].x;
    for (u32 i = 1; i < 8; i += 2) *(local u64 *)(root + (wa ^ 64) + (i - 1) * 64) = u[i].x;
    bar(WG);
    for (u32 i = 0; i < 8; ++i) u[i].x = *(local u64 *)(root + i * WG * 8 + (ra ^ ((i & 1) * 64)));
    bar(WG);
    AEVUM_PAIRED_LDS_OPAQUE(wa); AEVUM_PAIRED_LDS_OPAQUE(ra);
    for (u32 i = 0; i < 8; i += 2) *(local u64 *)(root + wa + i * 64) = u[i].y;
    for (u32 i = 1; i < 8; i += 2) *(local u64 *)(root + (wa ^ 64) + (i - 1) * 64) = u[i].y;
    bar(WG);
    for (u32 i = 0; i < 8; ++i) u[i].y = *(local u64 *)(root + i * WG * 8 + (ra ^ ((i & 1) * 64)));
  }
  // As in shufl64, the next LDS user supplies the final barrier.
}
#undef AEVUM_PAIRED_LDS_OPAQUE
#endif
