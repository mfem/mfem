// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../raja.hpp"

#if defined(__NVCC__) and defined(__NVVP__)

// *****************************************************************************
static const uint32_t legacy_colors[] = {
  0x000000, 0x000080, 0x00008B, 0x0000CD, 0x0000FF, 0x006400, 0x008000,
  0x008080, 0x008B8B, 0x00BFFF, 0x00CED1, 0x00FA9A, 0x00FF00, 0x00FF00,
  0x00FF7F, 0x00FFFF, 0x00FFFF, 0x191970, 0x1E90FF, 0x20B2AA, 0x228B22,
  0x2E8B57, 0x2F4F4F, 0x32CD32, 0x3CB371, 0x40E0D0, 0x4169E1, 0x4682B4,
  0x483D8B, 0x48D1CC, 0x4B0082, 0x556B2F, 0x5F9EA0, 0x6495ED, 0x663399,
  0x66CDAA, 0x696969, 0x6A5ACD, 0x6B8E23, 0x708090, 0x778899, 0x7B68EE,
  0x7CFC00, 0x7F0000, 0x7F007F, 0x7FFF00, 0x7FFFD4, 0x808000, 0x808080,
  0x87CEEB, 0x87CEFA, 0x8A2BE2, 0x8B0000, 0x8B008B, 0x8B4513, 0x8FBC8F,
  0x90EE90, 0x9370DB, 0x9400D3, 0x98FB98, 0x9932CC, 0x9ACD32, 0xA020F0,
  0xA0522D, 0xA52A2A, 0xA9A9A9, 0xADD8E6, 0xADFF2F, 0xAFEEEE, 0xB03060,
  0xB0C4DE, 0xB0E0E6, 0xB22222, 0xB8860B, 0xBA55D3, 0xBC8F8F, 0xBDB76B,
  0xBEBEBE, 0xC0C0C0, 0xC71585, 0xCD5C5C, 0xCD853F, 0xD2691E, 0xD2B48C,
  0xD3D3D3, 0xD8BFD8, 0xDA70D6, 0xDAA520, 0xDB7093, 0xDC143C, 0xDCDCDC,
  0xDDA0DD, 0xDEB887, 0xE0FFFF, 0xE6E6FA, 0xE9967A, 0xEE82EE, 0xEEE8AA,
  0xF08080, 0xF0E68C, 0xF0F8FF, 0xF0FFF0, 0xF0FFFF, 0xF4A460, 0xF5DEB3,
  0xF5F5DC, 0xF5F5F5, 0xF5FFFA, 0xF8F8FF, 0xFA8072, 0xFAEBD7, 0xFAF0E6,
  0xFAFAD2, 0xFDF5E6, 0xFF0000, 0xFF00FF, 0xFF00FF, 0xFF1493, 0xFF4500,
  0xFF6347, 0xFF69B4, 0xFF7F50, 0xFF8C00, 0xFFA07A, 0xFFA500, 0xFFB6C1,
  0xFFC0CB, 0xFFD700, 0xFFDAB9, 0xFFDEAD, 0xFFE4B5, 0xFFE4C4, 0xFFE4E1,
  0xFFEBCD, 0xFFEFD5, 0xFFF0F5, 0xFFF5EE, 0xFFF8DC, 0xFFFACD, 0xFFFAF0,
  0xFFFAFA, 0xFFFF00, 0xFFFFE0, 0xFFFFF0, 0xFFFFFF };

// *****************************************************************************
static const int nb_colors = sizeof(legacy_colors)/sizeof(uint32_t);

// *****************************************************************************
static int rNvtxAttrPushEx(const char *ascii, const int color){
  if (!mfem::rconfig::Get().Nvvp()) return 0;
  const int color_id = color%nb_colors;
  nvtxEventAttributes_t eAttrib = {0};
  eAttrib.version = NVTX_VERSION;
  eAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eAttrib.colorType = NVTX_COLOR_ARGB;
  eAttrib.color = legacy_colors[color_id];
  eAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eAttrib.message.ascii = ascii;
  return nvtxRangePushEx(&eAttrib);
}

// *****************************************************************************
int rNvtxRangePushEx(const char *function, const char *file, const int line,
                     const int color){
  if (!mfem::rconfig::Get().Nvvp()) return 0;
  const size_t size = 2048;
  static char marker[size];
  const int nb_of_char_printed =
    snprintf(marker,size, "%s@%s:%d",function,file,line);
  assert(nb_of_char_printed>=0);
  return rNvtxAttrPushEx(marker,color);
}

// *****************************************************************************
int rNvtxRangePushEx(const char *ascii, const int color){
  return rNvtxAttrPushEx(ascii,color);
}

// ***************************************************************************
int rNvtxSyncPop(void){ // Enforce Kernel Synchronization
  if (!mfem::rconfig::Get().Nvvp()) return 0;
  rNvtxAttrPushEx("EKS", Yellow);
  cudaStreamSynchronize(0);//*mfem::rconfig::Get().Stream());
  nvtxRangePop();
  return nvtxRangePop();
}
  
#endif // defined(__NVCC__) and defined(__NVVP__)
