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

#ifndef MFEM_BACKENDS_PA_UTIL_HPP
#define MFEM_BACKENDS_PA_UTIL_HPP

namespace mfem
{

namespace pa
{

#ifdef __NVCC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif

/**
*  The different operators available for the Kernels
*/
enum PAOp { BtDB, BtDG, GtDB, GtDG };

enum Location {Host, CudaDevice};

struct Empty{};

template <PAOp OpName>
struct QuadDimVal{
	static const int value = 1;
};

template <>
struct QuadDimVal<BtDB>{
	static const int value = 0;
};

template <>
struct QuadDimVal<GtDG>{
	static const int value = 2;
};

template <typename Equation>
struct QuadDim{
	static const int value = QuadDimVal<Equation::OpName>::value;
};

template <typename Equation>
struct EltDim{
	static const int value = QuadDim<Equation>::value + 1;
};

template <typename Equation>
struct TensorDim{
	static const int value = EltDim<Equation>::value + 1;
};

}

}

#endif