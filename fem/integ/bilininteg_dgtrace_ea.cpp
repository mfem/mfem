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
#include "bilininteg_dgtrace_kernels.hpp"

namespace mfem
{

static void EADGTraceAssemble1DInt(const int NF,
                                   const Array<real_t> &basis,
                                   const Vector &padata,
                                   Vector &eadata_int,
                                   Vector &eadata_ext,
                                   const bool add)
{
   auto D = Reshape(padata.Read(), 2, 2, NF);
   auto A_int = Reshape(eadata_int.ReadWrite(), 2, NF);
   auto A_ext = Reshape(eadata_ext.ReadWrite(), 2, NF);
   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      real_t val_int0, val_int1, val_ext01, val_ext10;
      val_int0  = D(0, 0, f);
      val_ext10 = D(1, 0, f);
      val_ext01 = D(0, 1, f);
      val_int1  = D(1, 1, f);
      if (add)
      {
         A_int(0, f) += val_int0;
         A_int(1, f) += val_int1;
         A_ext(0, f) += val_ext01;
         A_ext(1, f) += val_ext10;
      }
      else
      {
         A_int(0, f) = val_int0;
         A_int(1, f) = val_int1;
         A_ext(0, f) = val_ext01;
         A_ext(1, f) = val_ext10;
      }
   });
}

static void EADGTraceAssemble1DBdr(const int NF,
                                   const Array<real_t> &basis,
                                   const Vector &padata,
                                   Vector &eadata_bdr,
                                   const bool add)
{
   auto D = Reshape(padata.Read(), 2, 2, NF);
   auto A_bdr = Reshape(eadata_bdr.ReadWrite(), NF);
   mfem::forall(NF, [=] MFEM_HOST_DEVICE (int f)
   {
      if (add)
      {
         A_bdr(f) += D(0, 0, f);
      }
      else
      {
         A_bdr(f) = D(0, 0, f);
      }
   });
}

/// \cond DO_NOT_DOCUMENT

namespace internal
{

EADGTraceAssembleIntKernelsType EADGTraceAssembleIntKernels::Fallback(int DIM,
                                                                      int, int)
{
   if (DIM == 2) { return EADGTraceAssemble2DInt<0,0>; }
   if (DIM == 3) { return EADGTraceAssemble3DInt<0,0>; }
   MFEM_ABORT("Unknown kernel.");
}

/// Register the existing specializations once, on first use.
static void RegisterEADGTraceAssembleIntKernels()
{
   struct Kernels
   {
      Kernels()
      {
         EADGTraceAssembleIntKernels::Specialization<2,2,2>::Add();
         EADGTraceAssembleIntKernels::Specialization<2,3,3>::Add();
         EADGTraceAssembleIntKernels::Specialization<2,4,4>::Add();
         EADGTraceAssembleIntKernels::Specialization<2,5,5>::Add();
         EADGTraceAssembleIntKernels::Specialization<2,6,6>::Add();
         EADGTraceAssembleIntKernels::Specialization<2,7,7>::Add();
         EADGTraceAssembleIntKernels::Specialization<2,8,8>::Add();
         EADGTraceAssembleIntKernels::Specialization<2,9,9>::Add();
         EADGTraceAssembleIntKernels::Specialization<3,2,3>::Add();
         EADGTraceAssembleIntKernels::Specialization<3,3,4>::Add();
         EADGTraceAssembleIntKernels::Specialization<3,4,5>::Add();
         EADGTraceAssembleIntKernels::Specialization<3,5,6>::Add();
         EADGTraceAssembleIntKernels::Specialization<3,6,7>::Add();
         EADGTraceAssembleIntKernels::Specialization<3,7,8>::Add();
         EADGTraceAssembleIntKernels::Specialization<3,8,9>::Add();
      }
   };
   static Kernels kernels;
}

EADGTraceAssembleBdrKernelsType EADGTraceAssembleBdrKernels::Fallback(int DIM,
                                                                      int, int)
{
   if (DIM == 2) { return EADGTraceAssemble2DBdr<0,0>; }
   if (DIM == 3) { return EADGTraceAssemble3DBdr<0,0>; }
   MFEM_ABORT("Unknown kernel.");
}

/// Register the existing specializations once, on first use.
static void RegisterEADGTraceAssembleBdrKernels()
{
   struct Kernels
   {
      Kernels()
      {
         EADGTraceAssembleBdrKernels::Specialization<2,2,2>::Add();
         EADGTraceAssembleBdrKernels::Specialization<2,3,3>::Add();
         EADGTraceAssembleBdrKernels::Specialization<2,4,4>::Add();
         EADGTraceAssembleBdrKernels::Specialization<2,5,5>::Add();
         EADGTraceAssembleBdrKernels::Specialization<2,6,6>::Add();
         EADGTraceAssembleBdrKernels::Specialization<2,7,7>::Add();
         EADGTraceAssembleBdrKernels::Specialization<2,8,8>::Add();
         EADGTraceAssembleBdrKernels::Specialization<2,9,9>::Add();
         EADGTraceAssembleBdrKernels::Specialization<3,2,3>::Add();
         EADGTraceAssembleBdrKernels::Specialization<3,3,4>::Add();
         EADGTraceAssembleBdrKernels::Specialization<3,4,5>::Add();
         EADGTraceAssembleBdrKernels::Specialization<3,5,6>::Add();
         EADGTraceAssembleBdrKernels::Specialization<3,6,7>::Add();
         EADGTraceAssembleBdrKernels::Specialization<3,7,8>::Add();
         EADGTraceAssembleBdrKernels::Specialization<3,8,9>::Add();
      }
   };
   static Kernels kernels;
}

} // namespace internal

/// \endcond DO_NOT_DOCUMENT

void DGTraceIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace& fes,
                                                Vector &ea_data_int,
                                                Vector &ea_data_ext,
                                                const bool add)
{
   SetupPA(fes, FaceType::Interior);
   nf = fes.GetNFbyType(FaceType::Interior);
   if (nf==0) { return; }
   const Array<real_t> &B = maps->B;
   if (dim == 1)
   {
      return EADGTraceAssemble1DInt(nf,B,pa_data,ea_data_int,ea_data_ext,add);
   }
   else if (dim == 2)
   {
      internal::RegisterEADGTraceAssembleIntKernels();
      return internal::EADGTraceAssembleIntKernels::Run(
                dim, dofs1D, quad1D,
                nf,B,pa_data,ea_data_int,ea_data_ext,add,dofs1D,quad1D);
   }
   else if (dim == 3)
   {
      internal::RegisterEADGTraceAssembleIntKernels();
      return internal::EADGTraceAssembleIntKernels::Run(
                dim, dofs1D, quad1D,
                nf,B,pa_data,ea_data_int,ea_data_ext,add,dofs1D,quad1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

void DGTraceIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace& fes,
                                                Vector &ea_data_bdr,
                                                const bool add)
{
   SetupPA(fes, FaceType::Boundary);
   nf = fes.GetNFbyType(FaceType::Boundary);
   if (nf==0) { return; }
   const Array<real_t> &B = maps->B;
   if (dim == 1)
   {
      return EADGTraceAssemble1DBdr(nf,B,pa_data,ea_data_bdr,add);
   }
   else if (dim == 2)
   {
      internal::RegisterEADGTraceAssembleBdrKernels();
      return internal::EADGTraceAssembleBdrKernels::Run(
                dim, dofs1D, quad1D,
                nf,B,pa_data,ea_data_bdr,add,dofs1D,quad1D);
   }
   else if (dim == 3)
   {
      internal::RegisterEADGTraceAssembleBdrKernels();
      return internal::EADGTraceAssembleBdrKernels::Run(
                dim, dofs1D, quad1D,
                nf,B,pa_data,ea_data_bdr,add,dofs1D,quad1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

}
