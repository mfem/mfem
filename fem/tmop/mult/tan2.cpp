// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../pa.hpp"
#include "../../tmop.hpp"
#include "../../../general/forall.hpp"
#include "../../../linalg/dtensor.hpp"

namespace mfem
{

void TMOP_Integrator::TangentialPushForwardPA_2D(
   const Vector &map_jac, const Vector &param, Vector &phys) const
{
   const int NE = PA.ne;
   const int D1D = PA.maps->ndof;
   const int ndof = D1D*D1D;
   MFEM_VERIFY(param.Size() == 2*ndof*NE, "invalid E-vector size");
   MFEM_VERIFY(map_jac.Size() == 4*ndof*NE,
               "invalid tangential Jacobian size");

   phys.SetSize(param.Size(), Device::GetMemoryType());
   phys.UseDevice(true);
   const auto J = Reshape(map_jac.Read(), 2, 2, D1D, D1D, NE);
   const auto P = Reshape(param.Read(), D1D, D1D, 2, NE);
   auto X = Reshape(phys.Write(), D1D, D1D, 2, NE);

   mfem::forall(ndof*NE, [=] MFEM_HOST_DEVICE(int n)
   {
      const int e = n/ndof;
      const int i = n - e*ndof;
      const int dx = i%D1D;
      const int dy = i/D1D;
      for (int c = 0; c < 2; c++)
      {
         X(dx, dy, c, e) = J(c, 0, dx, dy, e)*P(dx, dy, 0, e) +
                           J(c, 1, dx, dy, e)*P(dx, dy, 1, e);
      }
   });
}

void TMOP_Integrator::TangentialPullBackPA_2D(
   const Vector &map_jac, const Vector &phys, Vector &param) const
{
   const int NE = PA.ne;
   const int D1D = PA.maps->ndof;
   const int ndof = D1D*D1D;
   MFEM_VERIFY(phys.Size() == 2*ndof*NE, "invalid E-vector size");
   MFEM_VERIFY(map_jac.Size() == 4*ndof*NE,
               "invalid tangential Jacobian size");

   param.SetSize(phys.Size(), Device::GetMemoryType());
   param.UseDevice(true);
   const auto J = Reshape(map_jac.Read(), 2, 2, D1D, D1D, NE);
   const auto X = Reshape(phys.Read(), D1D, D1D, 2, NE);
   auto P = Reshape(param.Write(), D1D, D1D, 2, NE);

   mfem::forall(ndof*NE, [=] MFEM_HOST_DEVICE(int n)
   {
      const int e = n/ndof;
      const int i = n - e*ndof;
      const int dx = i%D1D;
      const int dy = i/D1D;
      for (int a = 0; a < 2; a++)
      {
         P(dx, dy, a, e) = J(0, a, dx, dy, e)*X(dx, dy, 0, e) +
                           J(1, a, dx, dy, e)*X(dx, dy, 1, e);
      }
   });
}

void TMOP_Integrator::TangentialPullBackGradPA_2D(
   const Vector &phys_action, const Vector &param_direction,
   Vector &param_action) const
{
   const int NE = PA.ne;
   const int D1D = PA.maps->ndof;
   const int ndof = D1D*D1D;
   MFEM_VERIFY(phys_action.Size() == 2*ndof*NE, "invalid E-vector size");
   MFEM_VERIFY(param_direction.Size() == phys_action.Size(),
               "invalid E-vector size");

   param_action.SetSize(phys_action.Size(), Device::GetMemoryType());
   param_action.UseDevice(true);
   const auto J = Reshape(PA.tan_J.Read(), 2, 2, D1D, D1D, NE);
   const auto K = Reshape(PA.tan_H.Read(), 2, 2, 2, D1D, D1D, NE);
   const auto R = Reshape(PA.tan_R.Read(), D1D, D1D, 2, NE);
   const auto X = Reshape(phys_action.Read(), D1D, D1D, 2, NE);
   const auto D = Reshape(param_direction.Read(), D1D, D1D, 2, NE);
   auto Y = Reshape(param_action.Write(), D1D, D1D, 2, NE);

   mfem::forall(ndof*NE, [=] MFEM_HOST_DEVICE(int n)
   {
      const int e = n/ndof;
      const int i = n - e*ndof;
      const int dx = i%D1D;
      const int dy = i/D1D;
      for (int a = 0; a < 2; a++)
      {
         real_t value = J(0, a, dx, dy, e)*X(dx, dy, 0, e) +
                        J(1, a, dx, dy, e)*X(dx, dy, 1, e);
         for (int b = 0; b < 2; b++)
         {
            const real_t db = D(dx, dy, b, e);
            for (int c = 0; c < 2; c++)
            {
               value += R(dx, dy, c, e)*K(c, a, b, dx, dy, e)*db;
            }
         }
         Y(dx, dy, a, e) = value;
      }
   });
}

} // namespace mfem
