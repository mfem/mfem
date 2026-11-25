// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_BATCHED_KERNELS_HPP
#define MFEM_LOR_BATCHED_KERNELS_HPP

#include "lor_batched.hpp"
#include "lor_h1.hpp"
#include "lor_dg.hpp"
#include "lor_nd.hpp"
#include "lor_rt.hpp"
#include "../kernel_dispatch.hpp"
#include "../../general/forall.hpp"

namespace mfem
{
namespace internal
{

template <int ORDER, int SDIM, typename LOR_KERNEL>
void Assemble_(LOR_KERNEL &kernel, int dim)
{
   if (dim == 2) { kernel.template Assemble2D<ORDER,SDIM>(); }
   else if (dim == 3) { kernel.template Assemble3D<ORDER>(); }
   else { MFEM_ABORT("Unsupported dimension"); }
}

template <int ORDER, typename LOR_KERNEL>
void Assemble_(LOR_KERNEL &kernel, int dim, int sdim)
{
   if (sdim == 2) { Assemble_<ORDER,2>(kernel, dim); }
   else if (sdim == 3) { Assemble_<ORDER,3>(kernel, dim); }
   else { MFEM_ABORT("Unsupported space dimension."); }
}

template <typename LOR_KERNEL>
struct BatchedLORAssemblyKernels
{
   using KernelType = void (*)(LOR_KERNEL &, int, int);
   MFEM_REGISTER_KERNELS(AssembleKernels,
                         KernelType,
                         (int));

   BatchedLORAssemblyKernels()
   {
      AssembleKernels::template Specialization<1>::Add();
      AssembleKernels::template Specialization<2>::Add();
      AssembleKernels::template Specialization<3>::Add();
      AssembleKernels::template Specialization<4>::Add();
      AssembleKernels::template Specialization<5>::Add();
   }
};

template <typename LOR_KERNEL>
template <int ORDER>
typename BatchedLORAssemblyKernels<LOR_KERNEL>::KernelType
BatchedLORAssemblyKernels<LOR_KERNEL>::AssembleKernels::Kernel()
{
   return Assemble_<ORDER, LOR_KERNEL>;
}

template <typename LOR_KERNEL>
typename BatchedLORAssemblyKernels<LOR_KERNEL>::KernelType
BatchedLORAssemblyKernels<LOR_KERNEL>::AssembleKernels::Fallback(int order)
{
   MFEM_ABORT("No kernel order " << order << "!");
}

} // namespace internal

} // namespace mfem

#endif
