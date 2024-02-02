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

#include "lib/navier_solver.hpp"
#include "general/forall.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace navier;

class BoundaryNormalPressureIntegrator : public LinearFormIntegrator
{
public:
   BoundaryNormalPressureIntegrator(const GridFunction &pgf) :
      pgf(pgf)
   { }

   virtual bool SupportsDevice() const { return false; }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect)
   {
      int dim = el.GetDim()+1;
      MFEM_ASSERT(dim > 1,
                  "BoundaryNormalPressureIntegrator not implemented for dim == 1");
      int ndofs = el.GetDof();

      shape.SetSize(ndofs);
      nor.SetSize(dim);
      dudx.SetSize(dim, dim);
      A.SetSize(dim, dim);
      elvect.SetSize(ndofs * dim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() + 1);
      }

      for (int qp = 0; qp < ir->GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir->IntPoint(qp);

         Tr.SetIntPoint(&ip);
         CalcOrtho(Tr.Jacobian(), nor);

         double p = pgf.GetValue(Tr, ip);

         el.CalcShape(ip, shape);

         // -p * I
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               A(i, j) = -p * (i == j);
            }
         }

         for (int i = 0; i < dim; i++)
         {
            for (int dof = 0; dof < ndofs; dof++)
            {
               double s = 0.0;
               for (int j = 0; j < dim; j++)
               {
                  s += A(i, j) * nor(j);
               }
               elvect(ndofs * i + dof) += s * ip.weight * shape(dof);
            }
         }
      }
   }

   using LinearFormIntegrator::AssembleRHSElementVect;

   Vector shape, nor;
   DenseMatrix dudx, A;
   const GridFunction &pgf;
};