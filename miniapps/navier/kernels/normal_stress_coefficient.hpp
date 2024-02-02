// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "navier_solver.hpp"
#include "general/forall.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace navier;

class NormalStressCoefficient : public VectorCoefficient
{
public:
   NormalStressCoefficient(ParGridFunction &u, ParGridFunction &p,
                           const double dynamic_viscosity) :
      VectorCoefficient(u.FESpace()->GetVDim()),
      dim(u.ParFESpace()->GetMesh()->Dimension()),
      mu(dynamic_viscosity),
      ugf(u),
      pgf(p)
   {
      unit_normal.SetSize(dim);
      dudx.SetSize(vdim, dim);
      A.SetSize(vdim, dim);
   }

   using VectorCoefficient::Eval;

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      ugf.GetVectorGradient(T, dudx);

      double p = pgf.GetValue(T);

      CalcOrtho(T.Jacobian(), unit_normal);

      // (-p * I + mu (grad(u) + grad(u)^T)) * n_e
      for (int vd = 0; vd < vdim; vd++)
      {
         for (int d = 0; d < dim; d++)
         {
            A(vd, d) = -p * (d == vd) + mu * dudx(vd, d) + dudx(d, vd);
         }
      }

      for (int vd = 0; vd < vdim; vd++)
      {
         double s = 0.0;
         for (int d = 0; d < dim; d++)
         {
            s += A(vd, d) * unit_normal(d);
         }
         V(vd) = s;
      }
   }

private:
   const int dim;
   const double mu;
   // vdim x dim
   DenseMatrix dudx, A;
   Vector unit_normal;
   ParGridFunction &ugf;
   ParGridFunction &pgf;
};