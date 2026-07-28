// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bilininteg_mass_kernels.hpp"
#include "bilininteg_mass_pa_simplex.hpp" // IWYU pragma: keep

namespace mfem
{

MassIntegrator::Kernels::Kernels()
{
   // 2D
   // Q = P, only for simplex
   MassIntegrator::AddSimplexSpecialization<2,2,1>();
   MassIntegrator::AddSimplexSpecialization<2,3,2>();
   MassIntegrator::AddSimplexSpecialization<2,4,3>();
   MassIntegrator::AddSimplexSpecialization<2,5,4>();
   MassIntegrator::AddSimplexSpecialization<2,6,5>();
   MassIntegrator::AddSimplexSpecialization<2,7,6>();
   // Q=P+1
   MassIntegrator::AddSpecialization<2,1,1>();
   MassIntegrator::AddSpecialization<2,2,2>();
   MassIntegrator::AddSpecialization<2,3,3>();
   MassIntegrator::AddSpecialization<2,4,4>();
   MassIntegrator::AddSpecialization<2,5,5>();
   MassIntegrator::AddSpecialization<2,6,6>();
   MassIntegrator::AddSpecialization<2,7,7>();
   MassIntegrator::AddSpecialization<2,8,8>();
   MassIntegrator::AddSpecialization<2,9,9>();
   // Q=P+2
   MassIntegrator::AddSpecialization<2,1,2>();
   MassIntegrator::AddSpecialization<2,2,3>();
   MassIntegrator::AddSpecialization<2,3,4>();
   MassIntegrator::AddSpecialization<2,4,5>();
   MassIntegrator::AddSpecialization<2,5,6>();
   MassIntegrator::AddSpecialization<2,6,7>();
   MassIntegrator::AddSpecialization<2,7,8>();
   MassIntegrator::AddSpecialization<2,8,9>();
   MassIntegrator::AddSpecialization<2,9,10>();
   // others
   MassIntegrator::AddSpecialization<2,2,4>();
   MassIntegrator::AddSpecialization<2,2,5>();
   MassIntegrator::AddSpecialization<2,3,6>();
   MassIntegrator::AddSpecialization<2,4,6>();
   // Stroud / simplex PA extras (pairs not covered above)
   MassIntegrator::AddSpecialization<2,3,5>();
   MassIntegrator::AddSpecialization<2,4,7>();
   MassIntegrator::AddSpecialization<2,5,7>();
   MassIntegrator::AddSpecialization<2,5,8>();
   MassIntegrator::AddSpecialization<2,6,8>();
   MassIntegrator::AddSpecialization<2,6,9>();
   MassIntegrator::AddSpecialization<2,7,9>();
   MassIntegrator::AddSpecialization<2,7,10>();

   // 3D
   // Q = P, only for simplex
   MassIntegrator::AddSimplexSpecialization<3,2,1>();
   MassIntegrator::AddSimplexSpecialization<3,3,2>();
   MassIntegrator::AddSimplexSpecialization<3,4,3>();
   MassIntegrator::AddSimplexSpecialization<3,5,4>();
   MassIntegrator::AddSimplexSpecialization<3,6,5>();
   MassIntegrator::AddSimplexSpecialization<3,7,6>();
   MassIntegrator::AddSimplexSpecialization<3,8,7>();
   // Q=P+1
   MassIntegrator::AddSpecialization<3,1,1>();
   MassIntegrator::AddSpecialization<3,2,2>();
   MassIntegrator::AddSpecialization<3,3,3>();
   MassIntegrator::AddSpecialization<3,4,4>();
   MassIntegrator::AddSpecialization<3,5,5>();
   MassIntegrator::AddSpecialization<3,6,6>();
   MassIntegrator::AddSpecialization<3,7,7>();
   MassIntegrator::AddSpecialization<3,8,8>();
   MassIntegrator::AddSpecialization<3,9,9>();
   // Q=P+2
   MassIntegrator::AddSpecialization<3,1,2>();
   MassIntegrator::AddSpecialization<3,2,3>();
   MassIntegrator::AddSpecialization<3,3,4>();
   MassIntegrator::AddSpecialization<3,3,6>();
   MassIntegrator::AddSpecialization<3,4,5>();
   MassIntegrator::AddSpecialization<3,5,6>();
   MassIntegrator::AddSpecialization<3,6,7>();
   MassIntegrator::AddSpecialization<3,7,8>();
   MassIntegrator::AddSpecialization<3,8,9>();
   // others
   MassIntegrator::AddSpecialization<3,2,4>();
   MassIntegrator::AddSpecialization<3,4,6>();
   MassIntegrator::AddSpecialization<3,5,8>();

   RegisterSimplexMmaKernels();
   RegisterTensorsMmaKernels();
}

namespace internal
{

#ifdef MFEM_USE_OCCA
void OccaPAMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &B,
                       const Array<real_t> &Bt,
                       const Vector &D,
                       const Vector &X,
                       Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply2D_cpu;
      if (OccaMassApply2D_cpu.find(id) == OccaMassApply2D_cpu.end())
      {
         const occa::kernel MassApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_CPU", props);
         OccaMassApply2D_cpu.emplace(id, MassApply2D_CPU);
      }
      OccaMassApply2D_cpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaMassApply2D_gpu;
      if (OccaMassApply2D_gpu.find(id) == OccaMassApply2D_gpu.end())
      {
         const occa::kernel MassApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_GPU", props);
         OccaMassApply2D_gpu.emplace(id, MassApply2D_GPU);
      }
      OccaMassApply2D_gpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
}

void OccaPAMassApply3D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &B,
                       const Array<real_t> &Bt,
                       const Vector &D,
                       const Vector &X,
                       Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply3D_cpu;
      if (OccaMassApply3D_cpu.find(id) == OccaMassApply3D_cpu.end())
      {
         const occa::kernel MassApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_CPU", props);
         OccaMassApply3D_cpu.emplace(id, MassApply3D_CPU);
      }
      OccaMassApply3D_cpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaMassApply3D_gpu;
      if (OccaMassApply3D_gpu.find(id) == OccaMassApply3D_gpu.end())
      {
         const occa::kernel MassApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_GPU", props);
         OccaMassApply3D_gpu.emplace(id, MassApply3D_GPU);
      }
      OccaMassApply3D_gpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
}
#endif // MFEM_USE_OCCA

} // namespace internal

} // namespace mfem
