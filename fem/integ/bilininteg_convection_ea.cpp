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
#include "bilininteg_convection_kernels.hpp"

namespace mfem
{

namespace internal
{

EAConvectionAssembleKernelsType EAConvectionAssembleKernels::Fallback(int DIM,
                                                                      int, int)
{
   if (DIM == 1) { return EAConvectionAssemble1D<0,0>; }
   if (DIM == 2) { return EAConvectionAssemble2D<0,0>; }
   if (DIM == 3) { return EAConvectionAssemble3D<0,0>; }
   MFEM_ABORT("Unknown kernel.");
}

/// Register the existing specializations once, on first use.
static void RegisterEAConvectionAssembleKernels()
{
   struct Kernels
   {
      Kernels()
      {
         EAConvectionAssembleKernels::Specialization<1,2,2>::Add();
         EAConvectionAssembleKernels::Specialization<1,3,3>::Add();
         EAConvectionAssembleKernels::Specialization<1,4,4>::Add();
         EAConvectionAssembleKernels::Specialization<1,5,5>::Add();
         EAConvectionAssembleKernels::Specialization<1,6,6>::Add();
         EAConvectionAssembleKernels::Specialization<1,7,7>::Add();
         EAConvectionAssembleKernels::Specialization<1,8,8>::Add();
         EAConvectionAssembleKernels::Specialization<1,9,9>::Add();
         EAConvectionAssembleKernels::Specialization<2,2,2>::Add();
         EAConvectionAssembleKernels::Specialization<2,3,3>::Add();
         EAConvectionAssembleKernels::Specialization<2,4,4>::Add();
         EAConvectionAssembleKernels::Specialization<2,5,5>::Add();
         EAConvectionAssembleKernels::Specialization<2,6,6>::Add();
         EAConvectionAssembleKernels::Specialization<2,7,7>::Add();
         EAConvectionAssembleKernels::Specialization<2,8,8>::Add();
         EAConvectionAssembleKernels::Specialization<2,9,9>::Add();
         EAConvectionAssembleKernels::Specialization<3,2,3>::Add();
         EAConvectionAssembleKernels::Specialization<3,3,4>::Add();
         EAConvectionAssembleKernels::Specialization<3,4,5>::Add();
         EAConvectionAssembleKernels::Specialization<3,5,6>::Add();
      }
   };
   static Kernels kernels;
}

} // namespace internal

void ConvectionIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                      Vector &ea_data,
                                      const bool add)
{
   AssemblePA(fes);
   ne = fes.GetMesh()->GetNE();
   const Array<real_t> &B = maps->B;
   const Array<real_t> &G = maps->G;
   internal::RegisterEAConvectionAssembleKernels();
   return internal::EAConvectionAssembleKernels::Run(
             dim, dofs1D, quad1D,
             ne,B,G,pa_data,ea_data,add,dofs1D,quad1D);
}

}
