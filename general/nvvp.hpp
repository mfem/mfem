// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#ifndef MFEM_OKINA_NVVP_HPP
#define MFEM_OKINA_NVVP_HPP

// *****************************************************************************
void push_flf(const char *file, const int line, const char *func);

// *****************************************************************************
// en.wikipedia.org/wiki/Web_colors#Hex_triplet
typedef enum
{
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
   White
} colors;

// *****************************************************************************
#if defined(__NVCC__) and defined(__NVVP__)

#include <cuda.h>
#include <nvToolsExt.h>
#include <cudaProfiler.h>

// *****************************************************************************
NVTX_DECLSPEC int NVTX_API kNvtxRangePushEx(const char*,const char*,const int,
                                            const int);
NVTX_DECLSPEC int NVTX_API kNvtxRangePushEx(const char*, const int);
NVTX_DECLSPEC int NVTX_API kNvtxSyncPop(void);

// *****************************************************************************
#define pop(...) (mfem::config::Get().Sync())?kNvtxSyncPop():nvtxRangePop();

// *****************************************************************************
#define PUSH2(ascii,color) kNvtxRangePushEx(#ascii,color);
#define PUSH1(color) kNvtxRangePushEx(__PRETTY_FUNCTION__, __FILE__, __LINE__,color);
#define PUSH0() kNvtxRangePushEx(__PRETTY_FUNCTION__, __FILE__, __LINE__,Lime);

// *****************************************************************************
#define LPAREN (
#define COMMA_IF_PARENS(...) ,
#define EXPAND(...) __VA_ARGS__
#define PUSH(a0,a1,a2,a3,a4,a5,a,...) a
#define PUSH_CHOOSE(...) EXPAND(PUSH LPAREN \
      __VA_ARGS__ COMMA_IF_PARENS \
      __VA_ARGS__ COMMA_IF_PARENS __VA_ARGS__ (),  \
      PUSH2, impossible, PUSH2, PUSH1, PUSH0, PUSH1, ))
#define push(...) PUSH_CHOOSE(__VA_ARGS__)(__VA_ARGS__)

#else // __NVCC__ && _NVVP__ ***************************************************

#define pop(...)
#define push(...) //push_flf(__FILENAME__,__LINE__,__FUNCTION__)

#endif // defined(__NVCC__) and defined(__NVVP__)

#endif // MFEM_OKINA_NVVP_HPP
