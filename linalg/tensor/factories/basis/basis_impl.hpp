// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BASIS_IMPL
#define MFEM_BASIS_IMPL

#include "../../tensor_types.hpp"
#include "../../utilities/pow.hpp"

namespace mfem
{

// ALL THIS SHOULD BE REWRITTEN...
// TODO Maybe remove this class?
// TODO maybe D before Q?
template <int Dim, bool IsTensor, typename TensorType>
class BasisTensor : public TensorType
{
public:
   MFEM_HOST_DEVICE
   BasisTensor(int quads, int dofs): TensorType(quads,dofs) { }

   MFEM_HOST_DEVICE
   BasisTensor(double *shared_mem, int quads, int dofs)
      : TensorType(shared_mem,quads,dofs) { }
};

/// Represent the rank 2 tensor containing B1d or G1d with dynamic sizes
template <int Dim>
using DynamicBasisTensor = BasisTensor<Dim,true,DynamicDTensor<2>>;
template <int Dim>
using DynamicSharedBasisTensor = BasisTensor<Dim,true,DeviceDTensor<2>>;

/// Represent the rank 2 tensor containing B1d or G1d with static sizes
template <int Dim, int Q, int D>
using StaticBasisTensor = BasisTensor<Dim,true,StaticDTensor<Q,D>>;
template <int Dim, int Q, int D>
using StaticSharedBasisTensor = BasisTensor<Dim,true,StaticPointerDTensor<Q,D>>;

/// Represent the rank 2 tensor containing B or G with dynamic sizes
template <int Dim>
using DynamicBasisNonTensor = BasisTensor<
   Dim, false, DynamicDTensor<2,pow(DynamicMaxSize,2*Dim)>>;

/// Represent the rank 2 tensor containing B or G with static sizes
template <int Dim, int Q, int D>
using StaticBasisNonTensor = BasisTensor<Dim,false,StaticDTensor<Q,D>>;

} // mfem namespace

#endif // MFEM_BASIS_IMPL
