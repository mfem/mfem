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
#include "./nonlininteg_vecconvection_pa_diag.hpp" // IWYU pragma: keep

namespace mfem
{

void VectorConvectionNLFIntegrator::AssembleGradDiagonalPA(Vector &de) const
{
   MFEM_VERIFY(!DeviceCanUseCeed(),
               "VectorConvectionNLFIntegrator PA gradients are not supported "
               "with the libCEED backend");

   if (dim == 2)
   {
      if (static auto ini = false; !std::exchange(ini, true))
      {
         VectorConvectionNLFGradDiagPA2D::Specialization<2, 2>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<2, 3>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<3, 4>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<3, 5>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<4, 5>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<4, 6>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<5, 7>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<5, 8>::Add();
         VectorConvectionNLFGradDiagPA2D::Specialization<6, 8>::Add();
      }
      VectorConvectionNLFGradDiagPA2D::Run(d1d, q1d, ne,
                                           maps->B.Read(),
                                           maps->G.Read(),
                                           pa_adj.Read(),
                                           pa_u.Read(),
                                           de.ReadWrite(),
                                           d1d, q1d);
   }
   else if (dim == 3)
   {
      if (static auto ini = false; !std::exchange(ini, true))
      {
         VectorConvectionNLFGradDiagPA3D::Specialization<2, 3>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<2, 4>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<2, 5>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<3, 4>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<3, 5>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<3, 6>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<4, 5>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<4, 6>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<4, 7>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<4, 8>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<5, 6>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<5, 7>::Add();
         VectorConvectionNLFGradDiagPA3D::Specialization<5, 8>::Add();
      }
      VectorConvectionNLFGradDiagPA3D::Run(d1d, q1d, ne,
                                           maps->B.Read(),
                                           maps->G.Read(),
                                           pa_adj.Read(),
                                           pa_u.Read(),
                                           de.ReadWrite(),
                                           d1d, q1d);
   }
   else
   {
      MFEM_ABORT("Unsupported dimension");
   }
}

} // namespace mfem
