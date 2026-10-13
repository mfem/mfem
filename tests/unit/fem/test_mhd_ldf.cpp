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
#include "../../../examples/ex42.hpp"
#include <limits>

using namespace mfem;

namespace
{

// Independently assemble the weak physical divergence operator. For quads,
// use the full nodal Q^p test basis, including its redundant constraint;
// the projector implementation instead uses a reduced Legendre basis.
DenseMatrix AssembleMagneticDivergence(const FiniteElement &fe,
                                       ElementTransformation &Tr)
{
   const int p = fe.GetOrder();
   const int qorder = fe.GetGeomType() == Geometry::TRIANGLE ?
                      std::max(0, p-1) : p;
   L2_FECollection qfec(qorder, 2);
   const FiniteElement &qfe =
      *qfec.FiniteElementForGeometry(fe.GetGeomType());
   const int dof = fe.GetDof();
   DenseMatrix D(qfe.GetDof(), 2*dof);
   D = 0.0;
   Vector qshape(qfe.GetDof());
   DenseMatrix grad(dof, 2);
   const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(), 2*p+4);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetIntPoint(&ip);
      qfe.CalcShape(ip, qshape);
      fe.CalcPhysDShape(Tr, grad);
      const real_t weight = ip.weight * Tr.Weight();
      for (int a = 0; a < qfe.GetDof(); a++)
      {
         for (int j = 0; j < dof; j++)
         {
            D(a,j) += weight * qshape(a) * grad(j,0);
            D(a,dof+j) += weight * qshape(a) * grad(j,1);
         }
      }
   }
   return D;
}

}

TEST_CASE("Magnetic LDF projectors have zero physical divergence",
          "[LDF][MHD]")
{
   for (auto geometry : {Element::TRIANGLE, Element::QUADRILATERAL})
   {
      for (int order = 0; order <= 4; order++)
      {
         Mesh mesh = Mesh::MakeCartesian2D(2, 2, geometry, true);
         DG_FECollection fec(order, 2, BasisType::GaussLobatto);
         FiniteElementSpace fes(&mesh, &fec, 6, Ordering::byNODES);
         IdealMHDFlux flux(2, 5.0/3.0);
         RusanovFlux numerical_flux(flux);
         GPsource source(2);
         RusanovGPsource numerical_source(source);
         DG_MHD_GPsource op(fes,
                            std::unique_ptr<HyperbolicFormIntegrator>(
                               new HyperbolicFormIntegrator(numerical_flux)),
                            std::unique_ptr<GPsourceFormIntegrator>(
                               new GPsourceFormIntegrator(numerical_source)),
                            true, true);
         for (int deformed = 0; deformed < 2; deformed++)
         {
            if (deformed)
            {
               // Triangles remain affine; quads become non-affine bilinear
               // cells, so the updated physical divergence operator changes.
               mesh.Transform([](const Vector &x, Vector &y)
               {
                  y.SetSize(2);
                  y(0) = x(0) + 0.25*x(0)*x(1) + 0.1*x(1);
                  y(1) = x(1) + 0.12*x(0)*x(1);
               });
               op.Update();
            }
            for (int element = 0; element < fes.GetNE(); element++)
            {
               const FiniteElement &fe = *fes.GetFE(element);
               const DenseMatrix D = AssembleMagneticDivergence(
                                        fe, *fes.GetElementTransformation(element));
               const DenseMatrix &P = op.GetLDFProjector(element);
               DenseMatrix DP(D.Height(), P.Width());
               mfem::Mult(D, P, DP);
               const real_t scale = std::max(real_t(1.0), D.MaxMaxNorm());
               const real_t tolerance = 256 *
                                        std::numeric_limits<real_t>::epsilon() * P.Height() * scale;
               CAPTURE(geometry, order, deformed, element, DP.MaxMaxNorm(), tolerance);
               REQUIRE(DP.MaxMaxNorm() <= tolerance);
            }
         }
      }
   }
}
