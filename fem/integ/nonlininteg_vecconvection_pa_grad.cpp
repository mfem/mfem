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

#include <utility>

#include "../ceed/interface/util.hpp"
#include "./nonlininteg_vecconvection_pa_grad.hpp" // IWYU pragma: keep

namespace mfem
{

void VectorConvectionNLFIntegrator::AssembleGradPA(
   const Vector &u, const FiniteElementSpace &fes)
{
   MFEM_VERIFY(!DeviceCanUseCeed(),
               "VectorConvectionNLFIntegrator PA gradients are not supported "
               "with the libCEED backend");

   this->pa_u = u;
   AssemblePA(fes);

   if (static auto done = false; !std::exchange(done, true))
   {
      // 2D
      VectorConvectionNLFAddMultGradPA2D::Specialization<2, 2>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<2, 3>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<3, 4>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<3, 5>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<4, 5>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<4, 6>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<5, 7>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<5, 8>::Add();
      VectorConvectionNLFAddMultGradPA2D::Specialization<6, 8>::Add();
      // 3D
      VectorConvectionNLFAddMultGradPA3D::Specialization<2, 3>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<2, 4>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<2, 5>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<3, 4>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<3, 5>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<3, 6>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<4, 5>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<4, 6>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<4, 7>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<4, 8>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<5, 6>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<5, 7>::Add();
      VectorConvectionNLFAddMultGradPA3D::Specialization<5, 8>::Add();
   }
}

void VectorConvectionNLFIntegrator::AddMultGradPA(const Vector &x,
                                                  Vector &y) const
{
   MFEM_VERIFY(!DeviceCanUseCeed(),
               "VectorConvectionNLFIntegrator PA gradients are not supported "
               "with the libCEED backend");

   if (dim == 2)
   {
      VectorConvectionNLFAddMultGradPA2D::Run(d1d, q1d, ne,
                                              maps->B.Read(),
                                              maps->G.Read(),
                                              pa_adj.Read(),
                                              pa_u.Read(),
                                              x.Read(),
                                              y.ReadWrite(),
                                              d1d, q1d);
   }
   else if (dim == 3)
   {
      VectorConvectionNLFAddMultGradPA3D::Run(d1d, q1d, ne,
                                              maps->B.Read(),
                                              maps->G.Read(),
                                              pa_adj.Read(),
                                              pa_u.Read(),
                                              x.Read(),
                                              y.ReadWrite(),
                                              d1d, q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
   }
}

} // namespace mfem
