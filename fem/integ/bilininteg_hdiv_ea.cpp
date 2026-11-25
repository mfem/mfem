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
#include "bilininteg_hdiv_kernels.hpp"

namespace mfem
{

// For H(div) mass, Bo and Bc are the basis evaluation operators, and the
// pa_data corresponds to a (potentially symmetric) matrix coefficient.
// coeff_dim must be 3 or 4 depending on symmetry.
//
// For div-div, Bc is the derivative evaluation operator, and pa_data
// corresponds to a scalar coefficient. coeff_dim must be 1.
//
// These two integrators are distinguished using coeff_dim.

// For H(div) mass, Bo and Bc are the basis evaluation operators, and the
// pa_data corresponds to a (potentially symmetric) matrix coefficient.
// coeff_dim must be 6 or 9 depending on symmetry.
//
// For div-div, Bc is the derivative evaluation operator, and pa_data
// corresponds to a scalar coefficient. coeff_dim must be 1.
//
// These two integrators are distinguished using coeff_dim.

namespace internal
{

EAHdivAssembleKernelsType EAHdivAssembleKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2) { return EAHdivAssemble2D<0,0>; }
   if (DIM == 3) { return EAHdivAssemble3D<0,0>; }
   MFEM_ABORT("Unknown kernel.");
}

/// Register the existing specializations once, on first use.
static void RegisterEAHdivAssembleKernels()
{
   struct Kernels
   {
      Kernels()
      {
         EAHdivAssembleKernels::Specialization<2,2,2>::Add();
         EAHdivAssembleKernels::Specialization<2,3,3>::Add();
         EAHdivAssembleKernels::Specialization<2,4,4>::Add();
         EAHdivAssembleKernels::Specialization<2,5,5>::Add();
         EAHdivAssembleKernels::Specialization<3,2,3>::Add();
         EAHdivAssembleKernels::Specialization<3,3,4>::Add();
         EAHdivAssembleKernels::Specialization<3,4,5>::Add();
         EAHdivAssembleKernels::Specialization<3,5,6>::Add();
      }
   };
   static Kernels kernels;
}

} // namespace internal

void VectorFEMassIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                        Vector &ea_data,
                                        const bool add)
{
   AssemblePA(fes);

   if (trial_fetype != mfem::FiniteElement::DIV ||
       test_fetype != mfem::FiniteElement::DIV)
   {
      MFEM_ABORT("Unsupported kernel.");
   }

   const Array<real_t> &Bo = mapsO->B;
   const Array<real_t> &Bc = mapsC->B;

   if (dim == 2)
   {
      const int coeff_dim = symmetric ? 3 : 4;
      internal::RegisterEAHdivAssembleKernels();
      return internal::EAHdivAssembleKernels::Run(
                dim, dofs1D, quad1D,
                ne,Bo,Bc,coeff_dim,pa_data,ea_data,add,dofs1D,quad1D);
   }
   else if (dim == 3)
   {
      const int coeff_dim = symmetric ? 6 : 9;
      internal::RegisterEAHdivAssembleKernels();
      return internal::EAHdivAssembleKernels::Run(
                dim, dofs1D, quad1D,
                ne,Bo,Bc,coeff_dim,pa_data,ea_data,add,dofs1D,quad1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

void DivDivIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                  Vector &ea_data,
                                  const bool add)
{
   AssemblePA(fes);

   const Array<real_t> &Bo = mapsO->B;
   const Array<real_t> &Gc = mapsC->G;

   if (dim == 2)
   {
      internal::RegisterEAHdivAssembleKernels();
      return internal::EAHdivAssembleKernels::Run(
                dim, dofs1D, quad1D,
                ne,Bo,Gc,1,pa_data,ea_data,add,dofs1D,quad1D);
   }
   else if (dim == 3)
   {
      internal::RegisterEAHdivAssembleKernels();
      return internal::EAHdivAssembleKernels::Run(
                dim, dofs1D, quad1D,
                ne,Bo,Gc,1,pa_data,ea_data,add,dofs1D,quad1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

}
