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

void TMOP_Integrator::TangentialPushForwardPA_3D(
   const Vector &map_jac, const Vector &param, Vector &phys) const
{
   const int NE = PA.ne;
   const int D1D = PA.maps->ndof;
   const int ndof = D1D*D1D*D1D;
   MFEM_VERIFY(param.Size() == 3*ndof*NE, "invalid E-vector size");
   MFEM_VERIFY(map_jac.Size() == 9*ndof*NE,
               "invalid tangential Jacobian size");

   phys.SetSize(param.Size(), Device::GetMemoryType());
   phys.UseDevice(true);
   const auto J = Reshape(map_jac.Read(), 3, 3, D1D, D1D, D1D, NE);
   const auto P = Reshape(param.Read(), D1D, D1D, D1D, 3, NE);
   auto X = Reshape(phys.Write(), D1D, D1D, D1D, 3, NE);

   mfem::forall(ndof*NE, [=] MFEM_HOST_DEVICE(int n)
   {
      const int e = n/ndof;
      const int i = n - e*ndof;
      const int dx = i%D1D;
      const int j = i/D1D;
      const int dy = j%D1D;
      const int dz = j/D1D;
      for (int c = 0; c < 3; c++)
      {
         real_t value = 0.0;
         for (int a = 0; a < 3; a++)
         {
            value += J(c, a, dx, dy, dz, e)*P(dx, dy, dz, a, e);
         }
         X(dx, dy, dz, c, e) = value;
      }
   });
}

void TMOP_Integrator::TangentialPullBackPA_3D(
   const Vector &map_jac, const Vector &phys, Vector &param) const
{
   const int NE = PA.ne;
   const int D1D = PA.maps->ndof;
   const int ndof = D1D*D1D*D1D;
   MFEM_VERIFY(phys.Size() == 3*ndof*NE, "invalid E-vector size");
   MFEM_VERIFY(map_jac.Size() == 9*ndof*NE,
               "invalid tangential Jacobian size");

   param.SetSize(phys.Size(), Device::GetMemoryType());
   param.UseDevice(true);
   const auto J = Reshape(map_jac.Read(), 3, 3, D1D, D1D, D1D, NE);
   const auto X = Reshape(phys.Read(), D1D, D1D, D1D, 3, NE);
   auto P = Reshape(param.Write(), D1D, D1D, D1D, 3, NE);

   mfem::forall(ndof*NE, [=] MFEM_HOST_DEVICE(int n)
   {
      const int e = n/ndof;
      const int i = n - e*ndof;
      const int dx = i%D1D;
      const int j = i/D1D;
      const int dy = j%D1D;
      const int dz = j/D1D;
      for (int a = 0; a < 3; a++)
      {
         real_t value = 0.0;
         for (int c = 0; c < 3; c++)
         {
            value += J(c, a, dx, dy, dz, e)*X(dx, dy, dz, c, e);
         }
         P(dx, dy, dz, a, e) = value;
      }
   });
}

void TMOP_Integrator::TangentialPullBackGradPA_3D(
   const Vector &phys_action, const Vector &param_direction,
   Vector &param_action) const
{
   const int NE = PA.ne;
   const int D1D = PA.maps->ndof;
   const int ndof = D1D*D1D*D1D;
   MFEM_VERIFY(phys_action.Size() == 3*ndof*NE, "invalid E-vector size");
   MFEM_VERIFY(param_direction.Size() == phys_action.Size(),
               "invalid E-vector size");

   param_action.SetSize(phys_action.Size(), Device::GetMemoryType());
   param_action.UseDevice(true);
   const auto J = Reshape(PA.tan_J.Read(), 3, 3, D1D, D1D, D1D, NE);
   const auto K = Reshape(PA.tan_H.Read(), 3, 3, 3,
                          D1D, D1D, D1D, NE);
   const auto R = Reshape(PA.tan_R.Read(), D1D, D1D, D1D, 3, NE);
   const auto X = Reshape(phys_action.Read(), D1D, D1D, D1D, 3, NE);
   const auto D = Reshape(param_direction.Read(), D1D, D1D, D1D, 3, NE);
   auto Y = Reshape(param_action.Write(), D1D, D1D, D1D, 3, NE);

   mfem::forall(ndof*NE, [=] MFEM_HOST_DEVICE(int n)
   {
      const int e = n/ndof;
      const int i = n - e*ndof;
      const int dx = i%D1D;
      const int j = i/D1D;
      const int dy = j%D1D;
      const int dz = j/D1D;
      for (int a = 0; a < 3; a++)
      {
         real_t value = 0.0;
         for (int c = 0; c < 3; c++)
         {
            value += J(c, a, dx, dy, dz, e)*X(dx, dy, dz, c, e);
         }
         for (int b = 0; b < 3; b++)
         {
            const real_t db = D(dx, dy, dz, b, e);
            for (int c = 0; c < 3; c++)
            {
               value += R(dx, dy, dz, c, e)*
                        K(c, a, b, dx, dy, dz, e)*db;
            }
         }
         Y(dx, dy, dz, a, e) = value;
      }
   });
}

} // namespace mfem
