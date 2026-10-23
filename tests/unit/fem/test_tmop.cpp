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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

// On simplices and prisms the element energy, action and gradient read the
// shape gradients from the DofToQuad table. Check the energy against
// CalcDShape, and the action and gradient against finite differences.
TEST_CASE("TMOP element terms on simplices", "[TMOP]")
{
   const auto type = GENERATE(Element::TRIANGLE, Element::TETRAHEDRON,
                              Element::WEDGE);
   const int order = GENERATE(1, 2, 3);
   CAPTURE(type, order);

   Mesh mesh = (type == Element::TRIANGLE) ?
               Mesh::MakeCartesian2D(2, 2, type) :
               Mesh::MakeCartesian3D(1, 1, 1, type);
   const int dim = mesh.Dimension();
   mesh.SetCurvature(order);
   mesh.Transform([](const Vector &x, Vector &y)
   {
      y = x;
      for (int d = 0; d < x.Size(); d++)
      {
         y(d) += 0.05*sin(3.0*x((d+1) % x.Size()) + d);
      }
   });
   const GridFunction &nodes = *mesh.GetNodes();
   const FiniteElementSpace &fes = *nodes.FESpace();

   TMOP_Metric_002 metric_2d;
   TMOP_Metric_321 metric_3d;
   TMOP_QualityMetric &metric = (dim == 2) ?
                                static_cast<TMOP_QualityMetric &>(metric_2d) :
                                metric_3d;
   TargetConstructor target(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE);
   TMOP_Integrator integ(&metric, &target);

   const real_t h = 1e-6;
   Array<int> vdofs;
   Vector x, xp, elvect, vp, vm;
   DenseMatrix elmat;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const FiniteElement &el = *fes.GetFE(e);
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      fes.GetElementVDofs(e, vdofs);
      nodes.GetSubVector(vdofs, x);
      const int dof = el.GetDof(), nvd = x.Size();

      const IntegrationRule &ir =
         IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3);
      DenseTensor Jtr(dim, dim, ir.GetNPoints());
      target.ComputeElementTargets(e, el, ir, x, Jtr);
      DenseMatrix PMat(x.GetData(), dof, dim), DSh(dof, dim);
      DenseMatrix Jrt(dim), Jpr(dim), Jpt(dim);
      real_t energy = 0.0;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         metric.SetTargetJacobian(Jtr(q));
         CalcInverse(Jtr(q), Jrt);
         el.CalcDShape(ip, DSh);
         MultAtB(PMat, DSh, Jpr);
         Mult(Jpr, Jrt, Jpt);
         energy += ip.weight * Jtr(q).Det() * metric.EvalW(Jpt);
      }
      REQUIRE(integ.GetElementEnergy(el, T, x) == MFEM_Approx(energy));

      integ.AssembleElementVector(el, T, x, elvect);
      integ.AssembleElementGrad(el, T, x, elmat);
      const real_t vtol = 1e-6 * elvect.Normlinf();
      const real_t mtol = 1e-6 * elmat.MaxMaxNorm();
      for (int j = 0; j < nvd; j++)
      {
         xp = x; xp(j) += h;
         const real_t ep = integ.GetElementEnergy(el, T, xp);
         integ.AssembleElementVector(el, T, xp, vp);
         xp(j) = x(j) - h;
         const real_t em = integ.GetElementEnergy(el, T, xp);
         integ.AssembleElementVector(el, T, xp, vm);

         REQUIRE(std::abs((ep - em)/(2*h) - elvect(j)) <= vtol);
         for (int i = 0; i < nvd; i++)
         {
            REQUIRE(std::abs((vp(i) - vm(i))/(2*h) - elmat(i, j)) <= mtol);
         }
      }
   }
}
