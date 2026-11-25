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
#include "bilininteg_mass_kernels.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

EAMassAssembleKernelsType EAMassAssembleKernels::Fallback(int DIM, int, int)
{
   if (DIM == 1) { return EAMassAssemble1D<0,0>; }
   if (DIM == 2) { return EAMassAssemble2D<0,0>; }
   if (DIM == 3) { return EAMassAssemble3D<0,0>; }
   MFEM_ABORT("Unknown kernel.");
}

/// Register the existing specializations once, on first use.
static void RegisterEAMassAssembleKernels()
{
   struct Kernels
   {
      Kernels()
      {
         EAMassAssembleKernels::Specialization<1,2,2>::Add();
         EAMassAssembleKernels::Specialization<1,3,3>::Add();
         EAMassAssembleKernels::Specialization<1,4,4>::Add();
         EAMassAssembleKernels::Specialization<1,5,5>::Add();
         EAMassAssembleKernels::Specialization<1,6,6>::Add();
         EAMassAssembleKernels::Specialization<1,7,7>::Add();
         EAMassAssembleKernels::Specialization<1,8,8>::Add();
         EAMassAssembleKernels::Specialization<1,9,9>::Add();
         EAMassAssembleKernels::Specialization<2,2,2>::Add();
         EAMassAssembleKernels::Specialization<2,3,3>::Add();
         EAMassAssembleKernels::Specialization<2,4,4>::Add();
         EAMassAssembleKernels::Specialization<2,5,5>::Add();
         EAMassAssembleKernels::Specialization<2,6,6>::Add();
         EAMassAssembleKernels::Specialization<2,7,7>::Add();
         EAMassAssembleKernels::Specialization<2,8,8>::Add();
         EAMassAssembleKernels::Specialization<2,9,9>::Add();
         EAMassAssembleKernels::Specialization<3,2,3>::Add();
         EAMassAssembleKernels::Specialization<3,3,4>::Add();
         EAMassAssembleKernels::Specialization<3,4,5>::Add();
         EAMassAssembleKernels::Specialization<3,5,6>::Add();
         EAMassAssembleKernels::Specialization<3,6,7>::Add();
         EAMassAssembleKernels::Specialization<3,7,8>::Add();
         EAMassAssembleKernels::Specialization<3,8,9>::Add();
      }
   };
   static Kernels kernels;
}

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

void MassIntegrator::AssembleEA_(Vector &ea_data,
                                 const bool add)
{
   MFEM_VERIFY(maps->mode != DofToQuad::RAGGED_TENSOR,
               "AssembleEA not implemented for ragged tensor bases");
   const Array<real_t> &B = maps->B;
   internal::RegisterEAMassAssembleKernels();
   return internal::EAMassAssembleKernels::Run(
             dim, dofs1D, quad1D,
             ne,B,pa_data,ea_data,add,dofs1D,quad1D);
}

void MassIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                Vector &ea_data,
                                const bool add)
{
   AssemblePA(fes);
   if (ne > 0) { AssembleEA_(ea_data, add); }
}

void MassIntegrator::AssembleEABoundary(const FiniteElementSpace &fes,
                                        Vector &ea_data,
                                        const bool add)
{
   AssemblePABoundary(fes);
   if (ne > 0) { AssembleEA_(ea_data, add); }
}

}
