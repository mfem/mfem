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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "bilininteg_diffusion_kernels.hpp"

namespace mfem
{

namespace internal
{

EADiffusionAssembleKernelsType EADiffusionAssembleKernels::Fallback(int DIM,
                                                                    int, int)
{
   if (DIM == 1) { return EADiffusionAssemble1D<0,0>; }
   if (DIM == 2) { return EADiffusionAssemble2D<0,0>; }
   if (DIM == 3) { return EADiffusionAssemble3D<0,0>; }
   MFEM_ABORT("Unknown kernel.");
}

/// Register the existing specializations once, on first use.
static void RegisterEADiffusionAssembleKernels()
{
   struct Kernels
   {
      Kernels()
      {
         EADiffusionAssembleKernels::Specialization<1,2,2>::Add();
         EADiffusionAssembleKernels::Specialization<1,3,3>::Add();
         EADiffusionAssembleKernels::Specialization<1,4,4>::Add();
         EADiffusionAssembleKernels::Specialization<1,5,5>::Add();
         EADiffusionAssembleKernels::Specialization<1,6,6>::Add();
         EADiffusionAssembleKernels::Specialization<1,7,7>::Add();
         EADiffusionAssembleKernels::Specialization<1,8,8>::Add();
         EADiffusionAssembleKernels::Specialization<1,9,9>::Add();
         EADiffusionAssembleKernels::Specialization<2,2,2>::Add();
         EADiffusionAssembleKernels::Specialization<2,3,3>::Add();
         EADiffusionAssembleKernels::Specialization<2,4,4>::Add();
         EADiffusionAssembleKernels::Specialization<2,5,5>::Add();
         EADiffusionAssembleKernels::Specialization<2,6,6>::Add();
         EADiffusionAssembleKernels::Specialization<2,7,7>::Add();
         EADiffusionAssembleKernels::Specialization<2,8,8>::Add();
         EADiffusionAssembleKernels::Specialization<2,9,9>::Add();
         EADiffusionAssembleKernels::Specialization<3,2,3>::Add();
         EADiffusionAssembleKernels::Specialization<3,3,4>::Add();
         EADiffusionAssembleKernels::Specialization<3,4,5>::Add();
      }
   };
   static Kernels kernels;
}

} // namespace internal

void DiffusionIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                     Vector &ea_data,
                                     const bool add)
{
   AssemblePA(fes);
   MFEM_VERIFY(maps->mode != DofToQuad::RAGGED_TENSOR,
               "AssembleEA not implemented for ragged tensor bases");
   ne = fes.GetMesh()->GetNE();
   const Array<real_t> &B = maps->B;
   const Array<real_t> &G = maps->G;
   internal::RegisterEADiffusionAssembleKernels();
   return internal::EADiffusionAssembleKernels::Run(
             dim, dofs1D, quad1D,
             ne,B,G,pa_data,ea_data,add,dofs1D,quad1D);
}

}
