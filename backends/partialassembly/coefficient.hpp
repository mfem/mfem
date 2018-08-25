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


#ifndef MFEM_BACKENDS_PA_COEFFICIENT_HPP
#define MFEM_BACKENDS_PA_COEFFICIENT_HPP

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

namespace mfem
{

namespace pa
{

#include "util.hpp"

class Coefficient{
public:
	static const bool uses_coordinates = false;
	static const bool uses_jacobians   = false;
	static const bool uses_attributes  = false;
};

class ConstCoefficient{
private:
	typedef double Result;
	double val;
public:
	ConstCoefficient(const mfem::ConstCoefficient& coeff): val(coeff.constant) {}

	template <typename Info>
	__host__ __device__ Result operator()(const Info& info) const {
		return val;
	}
}

// #ifdef __NVCC__

// template <>
// class ConstCoefficient<CudaDevice>: public Coefficient{
// private:
// 	typedef double Result;
// 	__device__ double* val;
// public:
// 	__host__ ConstCoefficient(const mfem::ConstCoefficient& coeff): val(coeff.constant)
// 	{
// 		cudaMalloc(&val,sizeof(double));
// 		cudaMemcpy(coeff.constant,val,sizeof(double),hosttodevice)
// 		createCoeff<<<>>>(val)
// 	}

// 	__device__ ConstCoefficient(double* _val)
// 	{
// 		*val = *_val;
// 	}

// 	template <typename Info>
// 	__device__ Result operator()(const Info& info) const {
// 		return val;
// 	}
// }

// #endif

}

}

#endif

#endif