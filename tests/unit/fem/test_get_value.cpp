// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "catch.hpp"

using namespace mfem;

double func_3D_lin(const Vector &x)
{
   return x[0] + 2.0 * x[1] + 3.0 * x[2];
}

namespace get_value
{

TEST_CASE("3D GetValue",
          "[GridFunction]"
          "[GridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int dim = 3;
   int order = 1;
   double tol = 1e-6;

   Mesh mesh(n, n, n, Element::TETRAHEDRON, 1, 2.0, 3.0, 5.0);

   FunctionCoefficient linCoef(func_3D_lin);

   H1_FECollection h1_fec(order, dim);
   DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                           FiniteElement::VALUE);
   DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                           FiniteElement::INTEGRAL);

   FiniteElementSpace h1_fespace(&mesh, &h1_fec);
   FiniteElementSpace dgv_fespace(&mesh, &dgv_fec);
   FiniteElementSpace dgi_fespace(&mesh, &dgi_fec);

   GridFunction h1_x(&h1_fespace);
   GridFunction dgv_x(&dgv_fespace);
   GridFunction dgi_x(&dgi_fespace);

   GridFunctionCoefficient h1_xCoef(&h1_x);
   GridFunctionCoefficient dgv_xCoef(&dgv_x);
   GridFunctionCoefficient dgi_xCoef(&dgi_x);

   h1_x.ProjectCoefficient(linCoef);
   dgv_x.ProjectCoefficient(linCoef);
   dgi_x.ProjectCoefficient(linCoef);

   int npts = 0;

   SECTION("Domain Evaluation (H1 Context)")
   {
      int e = 1;
      ElementTransformation *T = mesh.GetElementTransformation(e);
      const FiniteElement   *fe = h1_fespace.GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               2*order + 2);

      double h1_err = 0.0;
      double dgv_err = 0.0;
      double dgi_err = 0.0;

      double tip_data[3];
      Vector tip(tip_data, 3);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         npts++;
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         T->Transform(ip, tip);

         double f_val = func_3D_lin(tip);

         double h1_gf_val = h1_xCoef.Eval(*T, ip);
         double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
         double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

         h1_err += fabs(f_val - h1_gf_val);
         dgv_err += fabs(f_val - dgv_gf_val);
         dgi_err += fabs(f_val - dgi_gf_val);

         if (log > 0 && fabs(f_val - h1_gf_val) > tol)
         {
            std::cout << e << ":" << j << " " << f_val << " " << h1_gf_val
                      << " " << fabs(f_val - h1_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
         {
            std::cout << e << ":" << j << " " << f_val << " " << dgv_gf_val
                      << " " << fabs(f_val - dgv_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
         {
            std::cout << e << ":" << j << " " << f_val << " " << dgi_gf_val
                      << " " << fabs(f_val - dgi_gf_val) << std::endl;
         }
      }
      h1_err /= ir.GetNPoints();
      dgv_err /= ir.GetNPoints();
      dgi_err /= ir.GetNPoints();

      REQUIRE(h1_err == Approx(0.0));
      REQUIRE(dgv_err == Approx(0.0));
      REQUIRE(dgi_err == Approx(0.0));
   }

   SECTION("Boundary Evaluation (H1 Context)")
   {
      int be = 1;
      ElementTransformation *T = mesh.GetBdrElementTransformation(be);
      const FiniteElement   *fe = h1_fespace.GetBE(be);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               2*order + 2);

      double h1_err = 0.0;
      double dgv_err = 0.0;
      double dgi_err = 0.0;

      double tip_data[3];
      Vector tip(tip_data, 3);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         npts++;
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         T->Transform(ip, tip);

         double f_val = func_3D_lin(tip);

         double h1_gf_val = h1_xCoef.Eval(*T, ip);
         double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
         double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

         h1_err += fabs(f_val - h1_gf_val);
         dgv_err += fabs(f_val - dgv_gf_val);
         dgi_err += fabs(f_val - dgi_gf_val);

         if (log > 0 && fabs(f_val - h1_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << h1_gf_val
                      << " " << fabs(f_val - h1_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << dgv_gf_val
                      << " " << fabs(f_val - dgv_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << dgi_gf_val
                      << " " << fabs(f_val - dgi_gf_val) << std::endl;
         }
      }
      h1_err /= ir.GetNPoints();
      dgv_err /= ir.GetNPoints();
      dgi_err /= ir.GetNPoints();

      REQUIRE(h1_err == Approx(0.0));
      REQUIRE(dgv_err == Approx(0.0));
      REQUIRE(dgi_err == Approx(0.0));
   }

   SECTION("Domain Evaluation (DG Context)")
   {
      int e = 1;
      ElementTransformation *T = mesh.GetElementTransformation(e);
      const FiniteElement   *fe = dgv_fespace.GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               2*order + 2);

      double h1_err = 0.0;
      double dgv_err = 0.0;
      double dgi_err = 0.0;

      double tip_data[3];
      Vector tip(tip_data, 3);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         npts++;
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         T->Transform(ip, tip);

         double f_val = func_3D_lin(tip);

         double h1_gf_val = h1_xCoef.Eval(*T, ip);
         double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
         double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

         h1_err += fabs(f_val - h1_gf_val);
         dgv_err += fabs(f_val - dgv_gf_val);
         dgi_err += fabs(f_val - dgi_gf_val);

         if (log > 0 && fabs(f_val - h1_gf_val) > tol)
         {
            std::cout << e << ":" << j << " " << f_val << " " << h1_gf_val
                      << " " << fabs(f_val - h1_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
         {
            std::cout << e << ":" << j << " " << f_val << " " << dgv_gf_val
                      << " " << fabs(f_val - dgv_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
         {
            std::cout << e << ":" << j << " " << f_val << " " << dgi_gf_val
                      << " " << fabs(f_val - dgi_gf_val) << std::endl;
         }
      }
      h1_err /= ir.GetNPoints();
      dgv_err /= ir.GetNPoints();
      dgi_err /= ir.GetNPoints();

      REQUIRE(h1_err == Approx(0.0));
      REQUIRE(dgv_err == Approx(0.0));
      REQUIRE(dgi_err == Approx(0.0));
   }

   SECTION("Interior Face Evaluation (DG Context)")
   {
      int be = 2;
      FaceElementTransformations *T = mesh.GetInteriorFaceTransformations(be);
      const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                               2*order + 2);

      double h1_err = 0.0;
      double dgv_err = 0.0;
      double dgi_err = 0.0;

      double tip_data[3];
      Vector tip(tip_data, 3);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         npts++;
         const IntegrationPoint &ip = ir.IntPoint(j);

         T->SetIntPoint(&ip);
         T->Transform(ip, tip);

         double f_val = func_3D_lin(tip);

         double h1_gf_val = h1_xCoef.Eval(*T, ip);
         double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
         double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

         h1_err += fabs(f_val - h1_gf_val);
         dgv_err += fabs(f_val - dgv_gf_val);
         dgi_err += fabs(f_val - dgi_gf_val);

         if (log > 0 && fabs(f_val - h1_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << h1_gf_val
                      << " " << fabs(f_val - h1_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << dgv_gf_val
                      << " " << fabs(f_val - dgv_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << dgi_gf_val
                      << " " << fabs(f_val - dgi_gf_val) << std::endl;
         }
      }
      h1_err /= ir.GetNPoints();
      dgv_err /= ir.GetNPoints();
      dgi_err /= ir.GetNPoints();

      REQUIRE(h1_err == Approx(0.0));
      REQUIRE(dgv_err == Approx(0.0));
      REQUIRE(dgi_err == Approx(0.0));
   }

   SECTION("Boundary Evaluation (DG Context)")
   {
      int be = 1;
      FaceElementTransformations *T = mesh.GetBdrFaceTransformations(be);
      const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                               2*order + 2);

      double h1_err = 0.0;
      double dgv_err = 0.0;
      double dgi_err = 0.0;

      double tip_data[3];
      Vector tip(tip_data, 3);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         npts++;
         const IntegrationPoint &ip = ir.IntPoint(j);

         T->SetIntPoint(&ip);
         T->Transform(ip, tip);

         double f_val = func_3D_lin(tip);

         double h1_gf_val = h1_xCoef.Eval(*T, ip);
         double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
         double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

         h1_err += fabs(f_val - h1_gf_val);
         dgv_err += fabs(f_val - dgv_gf_val);
         dgi_err += fabs(f_val - dgi_gf_val);

         if (log > 0 && fabs(f_val - h1_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << h1_gf_val
                      << " " << fabs(f_val - h1_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << dgv_gf_val
                      << " " << fabs(f_val - dgv_gf_val) << std::endl;
         }
         if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
         {
            std::cout << be << ":" << j << " " << f_val << " " << dgi_gf_val
                      << " " << fabs(f_val - dgi_gf_val) << std::endl;
         }
      }
      h1_err /= ir.GetNPoints();
      dgv_err /= ir.GetNPoints();
      dgi_err /= ir.GetNPoints();

      REQUIRE(h1_err == Approx(0.0));
      REQUIRE(dgv_err == Approx(0.0));
      REQUIRE(dgi_err == Approx(0.0));
   }

   std::cout << "Checked GridFunction::GetValue at "
             << npts << " points" << std::endl;
}

} // namespace get_value
