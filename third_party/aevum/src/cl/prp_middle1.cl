// PRP-only fusion of the MIDDLE=1 forward/tail/inverse boundary.
// Reuse the exact stock arithmetic and addressing; no intermediate global planes.
#if AEVUM_PRP_MIDDLE1
#if MIDDLE != 1 || INPLACE || PFA_RADIX
#error PRP middle-one fusion requires a non-PFA out-of-place MIDDLE=1 plan
#endif
#include "fft-middle.cl"

// Coordinates of a scalar in fftMiddleOut's output, before its optional LDS
// transpose. Writing this scalar directly has the same layout including PAD.
inline u32 prpMiddle1OutLane(u32 x, u32 y) {
  return (x % OUT_SIZEX) * (OUT_WG / OUT_SIZEX) + y % (OUT_WG / OUT_SIZEX);
}

#if FFT_FP32
void OVERLOAD prpReadMiddle1(CP(F2) in, F2* u, u32 line, u32 me, Trig middleTrig) {
  TrigFP32 trig = (TrigFP32) middleTrig;
  for (u32 i = 0; i < NH; ++i) {
    u32 y = me + i * G_H;
    F2 v[1];
    readMiddleInLine(v, in, y, line);
    middleMul2(v, line, y, 1.0f, trig);
    u[i] = v[0];
  }
}
void OVERLOAD prpWriteMiddle1(F2* u, P(F2) out, u32 line, u32 me, Trig middleTrig) {
  TrigFP32 trig = (TrigFP32) middleTrig;
  for (u32 i = 0; i < NH; ++i) {
    u32 x = me + i * G_H;
    F2 v[1] = {u[i]};
    middleMul2(v, line, x, 1.0f / (NWORDS * 2), trig);
    writeMiddleOutLine(out + prpMiddle1OutLane(x, line), v,
                      line / (OUT_WG / OUT_SIZEX), x / OUT_SIZEX);
  }
}
#endif

#if NTT_GF31
void OVERLOAD prpReadMiddle1(CP(GF31) in, GF31* u, u32 line, u32 me, Trig middleTrig) {
  TrigGF31 trig = (TrigGF31) (middleTrig + DISTMTRIGGF31);
  for (u32 i = 0; i < NH; ++i) {
    u32 y = me + i * G_H;
    GF31 v[1];
    readMiddleInLine(v, in, y, line);
    middleMul2(v, line, y, trig);
    u[i] = v[0];
  }
}
void OVERLOAD prpWriteMiddle1(GF31* u, P(GF31) out, u32 line, u32 me, Trig middleTrig) {
  TrigGF31 trig = (TrigGF31) (middleTrig + DISTMTRIGGF31);
  for (u32 i = 0; i < NH; ++i) {
    u32 x = me + i * G_H;
    GF31 v[1] = {u[i]};
    middleMul2(v, line, x, trig);
    writeMiddleOutLine(out + prpMiddle1OutLane(x, line), v,
                      line / (OUT_WG / OUT_SIZEX), x / OUT_SIZEX);
  }
}
#endif

#if NTT_GF61
void OVERLOAD prpReadMiddle1(CP(GF61) in, GF61* u, u32 line, u32 me, Trig middleTrig) {
  TrigGF61 trig = (TrigGF61) (middleTrig + DISTMTRIGGF61);
  for (u32 i = 0; i < NH; ++i) {
    u32 y = me + i * G_H;
    GF61 v[1];
    readMiddleInLine(v, in, y, line);
    middleMul2(v, line, y, trig);
    u[i] = v[0];
  }
}
void OVERLOAD prpWriteMiddle1(GF61* u, P(GF61) out, u32 line, u32 me, Trig middleTrig) {
  TrigGF61 trig = (TrigGF61) (middleTrig + DISTMTRIGGF61);
  for (u32 i = 0; i < NH; ++i) {
    u32 x = me + i * G_H;
    GF61 v[1] = {u[i]};
    middleMul2(v, line, x, trig);
    writeMiddleOutLine(out + prpMiddle1OutLane(x, line), v,
                      line / (OUT_WG / OUT_SIZEX), x / OUT_SIZEX);
  }
}
#endif
#endif
