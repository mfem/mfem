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
#ifndef LAGHOS_RAJA_NVVP
#define LAGHOS_RAJA_NVVP

// *****************************************************************************
// en.wikipedia.org/wiki/Web_colors#Hex_triplet
typedef enum { 
  Black, NavyBlue, DarkBlue, MediumBlue, Blue, DarkGreen, WebGreen, Teal,
  DarkCyan, DeepSkyBlue, DarkTurquoise, MediumSpringGreen, Green, Lime,
  SpringGreen, Aqua, Cyan, MidnightBlue, DodgerBlue, LightSeaGreen,
  ForestGreen, SeaGreen, DarkSlateGray, LimeGreen, MediumSeaGreen,
  Turquoise, RoyalBlue, SteelBlue, DarkSlateBlue, MediumTurquoise, Indigo,
  DarkOliveGreen, CadetBlue, Cornflower, RebeccaPurple, MediumAquamarine,
  DimGray, SlateBlue, OliveDrab, SlateGray, LightSlateGray,
  MediumSlateBlue, LawnGreen, WebMaroon, WebPurple, Chartreuse,
  Aquamarine, Olive, WebGray, SkyBlue, LightSkyBlue, BlueViolet, DarkRed,
  DarkMagenta, SaddleBrown, DarkSeaGreen, LightGreen, MediumPurple,
  DarkViolet, PaleGreen, DarkOrchid, YellowGreen, Purple, Sienna, Brown,
  DarkGray, LightBlue, GreenYellow, PaleTurquoise, Maroon,
  LightSteelBlue, PowderBlue, Firebrick, DarkGoldenrod, MediumOrchid,
  RosyBrown, DarkKhaki, Gray, Silver, MediumVioletRed, IndianRed, Peru,
  Chocolate, Tan, LightGray, Thistle, Orchid, Goldenrod, PaleVioletRed,
  Crimson, Gainsboro, Plum, Burlywood, LightCyan, Lavender, DarkSalmon,
  Violet, PaleGoldenrod, LightCoral, Khaki, AliceBlue, Honeydew, Azure,
  SandyBrown, Wheat, Beige, WhiteSmoke, MintCream, GhostWhite, Salmon,
  AntiqueWhite, Linen, LightGoldenrod, OldLace, Red, Fuchsia, Magenta,
  DeepPink, OrangeRed, Tomato, HotPink, Coral, DarkOrange, LightSalmon,
  Orange, LightPink, Pink, Gold, PeachPuff, NavajoWhite, Moccasin,
  Bisque, MistyRose, BlanchedAlmond, PapayaWhip, LavenderBlush, Seashell,
  Cornsilk, LemonChiffon, FloralWhite, Snow, Yellow, LightYellow, Ivory,
  White} colors;

// *****************************************************************************
#if defined(__NVCC__) and defined(__NVVP__)

#include <cuda.h>
#include <nvToolsExt.h>
#include <cudaProfiler.h>

// *****************************************************************************
NVTX_DECLSPEC int NVTX_API rNvtxRangePushEx(const char*,const char*,const int,const int);
NVTX_DECLSPEC int NVTX_API rNvtxRangePushEx(const char*, const int);
NVTX_DECLSPEC int NVTX_API rNvtxSyncPop(void);

// *****************************************************************************
#define pop(...) (mfem::rconfig::Get().Sync())?rNvtxSyncPop():nvtxRangePop();

// *****************************************************************************
#define PUSH2(ascii,color) rNvtxRangePushEx(#ascii,color);
#define PUSH1(color) rNvtxRangePushEx(__PRETTY_FUNCTION__, __FILE__, __LINE__,color);
#define PUSH0() rNvtxRangePushEx(__PRETTY_FUNCTION__, __FILE__, __LINE__,Lime);

// *****************************************************************************
#define LPAREN (
#define COMMA_IF_PARENS(...) ,
#define EXPAND(...) __VA_ARGS__
#define PUSH(a0,a1,a2,a3,a4,a5,a,...) a
#define CHOOSE(...) EXPAND(PUSH LPAREN \
      __VA_ARGS__ COMMA_IF_PARENS \
      __VA_ARGS__ COMMA_IF_PARENS __VA_ARGS__ (),  \
      PUSH2, impossible, PUSH2, PUSH1, PUSH0, PUSH1, ))
#define push(...) CHOOSE(__VA_ARGS__)(__VA_ARGS__)

#else // __NVCC__ && _NVVP__ ***************************************************

#define pop(...)
#define push(...) dbg("\n%s",__PRETTY_FUNCTION__)
#define cuProfilerStart(...)
#define cuProfilerStop(...)

#endif // defined(__NVCC__) and defined(__NVVP__)

#endif // LAGHOS_RAJA_NVVP

