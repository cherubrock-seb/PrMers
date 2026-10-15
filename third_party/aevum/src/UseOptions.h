#pragma once
#include <string>
// Exact host-side CUDA limit keys; reject misspellings instead of accepting
// arbitrary REG prefixes. -1 means compiler default, 1..16 launch bounds.
inline bool aevumRegisterUseKey(const std::string& key) {
  for (const char* allowed : {"REGCF64","REGCF3161","REGCF3261","REGCF61",
       "REGCF323161","REGCF3231","REGCF6431","REGCF31","REGCF32",
       "REGMI64","REGMI32","REGMI31","REGMI61","REGMO64","REGMO32",
       "REGMO31","REGMO61","REGTS64","REGTS32","REGTS31","REGTS61"})
    if (key == allowed) return true;
  return false;
}
// Single key registry shared with Gpu::makeDefines (-use).
inline bool aevumUseKey(const std::string& key) {
  if (aevumRegisterUseKey(key)) return true;
  if (key == "AEVUM_EXPERIMENT_LEAN_CARRY128") return true;
  if (key == "AEVUM_PFA_GF31_COSBASE") return true;
  if (key == "AEVUM_PFA7_INPLACE") return true;
  if (key == "AEVUM_PFA7_NUSS8") return true;
  if (key == "AEVUM_CUDA_GRAPH_SQUARES") return true;
  if (key == "AEVUM_PFA_FUSED_INVERSE") return true;
  if (key == "AEVUM_GF61_BETTER_ONEPAIRSQ") return true;
  if (key == "AEVUM_PFA7_SEPARATE_CARRYB") return true;
  for(const char* allowed : {"FAST_BARRIER","AEVUM_ROE_SUBGROUP_REDUCE","STATS","IN_SIZEX","IN_WG","OUT_SIZEX","OUT_WG","UNROLL_H","UNROLL_W","LDSPAD_H","ZEROHACK_H","ZEROHACK_W","NO_ASM","DEBUG","CARRY64","BIGLIT","NONTEMPORAL","INPLACE","PAD","MIDDLE_IN_LDS_TRANSPOSE","MIDDLE_OUT_LDS_TRANSPOSE","MULTI_Q","PRP_MIDDLE1","TAIL_KERNELS","TAIL_TRIGS","TAIL_TRIGS31","TAIL_TRIGS32","TAIL_TRIGS61","TABMUL_CHAIN","TABMUL_CHAIN31","TABMUL_CHAIN32","TABMUL_CHAIN61","MODM31","LOADS","STORES","NOREG","WMUL","AEVUM_GF61_CMUL_ALT","AEVUM_GF61_LIMB32","AEVUM_GF61_MIDDLE_BASE_TABLE"}) if(key==allowed)return true;
  return false;
}
