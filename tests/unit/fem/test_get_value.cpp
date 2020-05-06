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

namespace get_value
{

double func_1D_lin(const Vector &x)
{
   return x[0];
}

double func_2D_lin(const Vector &x)
{
   return x[0] + 2.0 * x[1];
}

double func_3D_lin(const Vector &x)
{
   return x[0] + 2.0 * x[1] + 3.0 * x[2];
}

void Func_2D_lin(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v[0] =  1.234 * x[0] - 2.357 * x[1];
   v[1] =  2.537 * x[0] + 4.321 * x[1];
}

void Func_3D_lin(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] =  1.234 * x[0] - 2.357 * x[1] + 3.572 * x[2];
   v[1] =  2.537 * x[0] + 4.321 * x[1] - 1.234 * x[2];
   v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
}

TEST_CASE("1D GetValue",
          "[GridFunction]"
          "[GridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int dim = 1;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh(n, 2.0);

      FunctionCoefficient linCoef(func_1D_lin);

      SECTION("1D GetValue tests for element type " + std::to_string(type))
      {
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

         SECTION("Domain Evaluation 1D")
         {
            std::cout << "Domain Evaluation 1D" << std::endl;
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[1];
               Vector tip(tip_data, 1);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_1D_lin(tip);

                  double h1_gf_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);
                  dgv_err += fabs(f_val - dgv_gf_val);
                  dgi_err += fabs(f_val - dgi_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 1D (H1 Context)")
         {
            std::cout << "Boundary Evaluation 1D (H1 Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[1];
               Vector tip(tip_data, 1);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_1D_lin(tip);
                  double h1_gf_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);
                  dgv_err += fabs(f_val - dgv_gf_val);
                  dgi_err += fabs(f_val - dgi_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 1D (DG Context)")
         {
            std::cout << "Boundary Evaluation 1D (DG Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[1];
               Vector tip(tip_data, 1);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_1D_lin(tip);

                  double h1_gf_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);
                  dgv_err += fabs(f_val - dgv_gf_val);
                  dgi_err += fabs(f_val - dgi_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }
      }
   }
   std::cout << "Checked GridFunction::GetValue at "
             << npts << " 1D points" << std::endl;
}

TEST_CASE("2D GetValue",
          "[GridFunction]"
          "[GridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int dim = 2;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = (int)Element::TRIANGLE;
        type <= (int)Element::QUADRILATERAL; type++)
   {
      Mesh mesh(n, n, (Element::Type)type, 1, 2.0, 3.0);

      FunctionCoefficient linCoef(func_2D_lin);

      SECTION("2D GetValue tests for element type " + std::to_string(type))
      {
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

         SECTION("Domain Evaluation 2D")
         {
            std::cout << "Domain Evaluation 2D" << std::endl;
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_2D_lin(tip);

                  double h1_gf_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);
                  dgv_err += fabs(f_val - dgv_gf_val);
                  dgi_err += fabs(f_val - dgi_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (H1 Context)")
         {
            std::cout << "Boundary Evaluation 2D (H1 Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_2D_lin(tip);
                  double h1_gf_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);
                  dgv_err += fabs(f_val - dgv_gf_val);
                  dgi_err += fabs(f_val - dgi_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (DG Context)")
         {
            std::cout << "Boundary Evaluation 2D (DG Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_2D_lin(tip);

                  double h1_gf_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gf_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gf_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);
                  dgv_err += fabs(f_val - dgv_gf_val);
                  dgi_err += fabs(f_val - dgi_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Edge Evaluation 2D (H1 Context)")
         {
            std::cout << "Edge Evaluation 2D (H1 Context)" << std::endl;
            for (int e = 0; e < mesh.GetNEdges(); e++)
            {
               ElementTransformation *T = mesh.GetEdgeTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetEdgeElement(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_3D_lin(tip);
                  double h1_gf_val = h1_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
            }
         }
      }
   }
   std::cout << "Checked GridFunction::GetValue at "
             << npts << " 2D points" << std::endl;
}

TEST_CASE("3D GetValue",
          "[GridFunction]"
          "[GridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int dim = 3;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::WEDGE; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      FunctionCoefficient linCoef(func_3D_lin);

      SECTION("3D GetValue tests for element type " + std::to_string(type))
      {
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

         SECTION("Domain Evaluation 3D")
         {
            std::cout << "Domain Evaluation 3D" << std::endl;
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
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
                     std::cout << e << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (H1 Context)")
         {
            std::cout << "Boundary Evaluation 3D (H1 Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
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
                     std::cout << be << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (DG Context)")
         {
            std::cout << "Boundary Evaluation 3D (DG Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
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
                     std::cout << be << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgv_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gf_val << " " << fabs(f_val - dgv_gf_val)
                               << std::endl;
                  }
                  if (log > 0 && fabs(f_val - dgi_gf_val) > tol)
                  {
                     std::cout << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gf_val << " " << fabs(f_val - dgi_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Edge Evaluation 3D (H1 Context)")
         {
            std::cout << "Edge Evaluation 3D (H1 Context)" << std::endl;
            for (int e = 0; e < mesh.GetNEdges(); e++)
            {
               ElementTransformation *T = mesh.GetEdgeTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetEdgeElement(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_3D_lin(tip);
                  double h1_gf_val = h1_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << e << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
            }
         }

         SECTION("Face Evaluation 3D (H1 Context)")
         {
            std::cout << "Face Evaluation 3D (H1 Context)" << std::endl;
            for (int f = 0; f < mesh.GetNFaces(); f++)
            {
               ElementTransformation *T = mesh.GetFaceTransformation(f);
               const FiniteElement   *fe = h1_fespace.GetFaceElement(f);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  double f_val = func_3D_lin(tip);
                  double h1_gf_val = h1_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gf_val);

                  if (log > 0 && fabs(f_val - h1_gf_val) > tol)
                  {
                     std::cout << f << ":" << j << " h1  " << f_val << " "
                               << h1_gf_val << " " << fabs(f_val - h1_gf_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();

               REQUIRE(h1_err == Approx(0.0));
            }
         }
      }
   }
   std::cout << "Checked GridFunction::GetValue at "
             << npts << " 3D points" << std::endl;
}

TEST_CASE("2D GetVectorValue",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int dim = 2;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = (int)Element::TRIANGLE;
        type <= (int)Element::QUADRILATERAL; type++)
   {
      Mesh mesh(n, n, (Element::Type)type, 1, 2.0, 3.0);

      VectorFunctionCoefficient linCoef(dim, Func_2D_lin);

      SECTION("2D GetVectorValue tests for element type " +
              std::to_string(type))
      {
         H1_FECollection  h1_fec(order, dim);
         ND_FECollection  nd_fec(order+1, dim);
         RT_FECollection  rt_fec(order+1, dim);
         L2_FECollection  l2_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);
         DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::INTEGRAL);

         FiniteElementSpace  h1_fespace(&mesh,  &h1_fec, dim);
         FiniteElementSpace  nd_fespace(&mesh,  &nd_fec);
         FiniteElementSpace  rt_fespace(&mesh,  &rt_fec);
         FiniteElementSpace  l2_fespace(&mesh,  &l2_fec, dim);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec, dim);
         FiniteElementSpace dgi_fespace(&mesh, &dgi_fec, dim);

         GridFunction  h1_x( &h1_fespace);
         GridFunction  nd_x( &nd_fespace);
         GridFunction  rt_x( &rt_fespace);
         GridFunction  l2_x( &l2_fespace);
         GridFunction dgv_x(&dgv_fespace);
         GridFunction dgi_x(&dgi_fespace);

         VectorGridFunctionCoefficient  h1_xCoef( &h1_x);
         VectorGridFunctionCoefficient  nd_xCoef( &nd_x);
         VectorGridFunctionCoefficient  rt_xCoef( &rt_x);
         VectorGridFunctionCoefficient  l2_xCoef( &l2_x);
         VectorGridFunctionCoefficient dgv_xCoef(&dgv_x);
         VectorGridFunctionCoefficient dgi_xCoef(&dgi_x);

         h1_x.ProjectCoefficient(linCoef);
         nd_x.ProjectCoefficient(linCoef);
         rt_x.ProjectCoefficient(linCoef);
         l2_x.ProjectCoefficient(linCoef);
         dgv_x.ProjectCoefficient(linCoef);
         dgi_x.ProjectCoefficient(linCoef);

         Vector      f_val(dim);      f_val = 0.0;
         Vector  h1_gf_val(dim);  h1_gf_val = 0.0;
         Vector  nd_gf_val(dim);  nd_gf_val = 0.0;
         Vector  rt_gf_val(dim);  rt_gf_val = 0.0;
         Vector  l2_gf_val(dim);  l2_gf_val = 0.0;
         Vector dgv_gf_val(dim); dgv_gf_val = 0.0;
         Vector dgi_gf_val(dim); dgi_gf_val = 0.0;

         SECTION("Domain Evaluation 2D")
         {
            std::cout << "Domain Evaluation 2D" << std::endl;
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double  rt_err = 0.0;
               double  l2_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_2D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);
                  nd_xCoef.Eval(nd_gf_val, *T, ip);
                  rt_xCoef.Eval(rt_gf_val, *T, ip);
                  l2_xCoef.Eval(l2_gf_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gf_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, 2);
                  double  nd_dist = Distance(f_val,  nd_gf_val, 2);
                  double  rt_dist = Distance(f_val,  rt_gf_val, 2);
                  double  l2_dist = Distance(f_val,  l2_gf_val, 2);
                  double dgv_dist = Distance(f_val, dgv_gf_val, 2);
                  double dgi_dist = Distance(f_val, dgi_gf_val, 2);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ") "
                               << h1_dist << std::endl;
                  }
                  if (log > 0 && nd_dist > tol)
                  {
                     std::cout << e << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << nd_gf_val[0] << "," << nd_gf_val[1] << ") "
                               << nd_dist << std::endl;
                  }
                  if (log > 0 && rt_dist > tol)
                  {
                     std::cout << e << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << rt_gf_val[0] << "," << rt_gf_val[1] << ") "
                               << rt_dist << std::endl;
                  }
                  if (log > 0 && l2_dist > tol)
                  {
                     std::cout << e << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << l2_gf_val[0] << "," << l2_gf_val[1] << ") "
                               << l2_dist << std::endl;
                  }
                  if (log > 0 && dgv_dist > tol)
                  {
                     std::cout << e << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gf_val[0] << "," << dgv_gf_val[1] << ") "
                               << dgv_dist << std::endl;
                  }
                  if (log > 0 && dgi_dist > tol)
                  {
                     std::cout << e << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgi_gf_val[0] << "," << dgi_gf_val[1] << ") "
                               << dgi_dist << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
               REQUIRE( nd_err == Approx(0.0));
               REQUIRE( rt_err == Approx(0.0));
               REQUIRE( l2_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (H1 Context)")
         {
            std::cout << "Boundary Evaluation 2D (H1 Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double  rt_err = 0.0;
               double  l2_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_2D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);
                  nd_xCoef.Eval(nd_gf_val, *T, ip);
                  rt_xCoef.Eval(rt_gf_val, *T, ip);
                  l2_xCoef.Eval(l2_gf_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gf_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, 2);
                  double  nd_dist = Distance(f_val,  nd_gf_val, 2);
                  double  rt_dist = Distance(f_val,  rt_gf_val, 2);
                  double  l2_dist = Distance(f_val,  l2_gf_val, 2);
                  double dgv_dist = Distance(f_val, dgv_gf_val, 2);
                  double dgi_dist = Distance(f_val, dgi_gf_val, 2);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ") "
                               << h1_dist << std::endl;
                  }
                  if (log > 0 && nd_dist > tol)
                  {
                     std::cout << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << nd_gf_val[0] << "," << nd_gf_val[1] << ") "
                               << nd_dist << std::endl;
                  }
                  if (log > 0 && rt_dist > tol)
                  {
                     std::cout << be << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << rt_gf_val[0] << "," << rt_gf_val[1] << ") "
                               << rt_dist << std::endl;
                  }
                  if (log > 0 && l2_dist > tol)
                  {
                     std::cout << be << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << l2_gf_val[0] << "," << l2_gf_val[1] << ") "
                               << l2_dist << std::endl;
                  }
                  if (log > 0 && dgv_dist > tol)
                  {
                     std::cout << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gf_val[0] << "," << dgv_gf_val[1] << ") "
                               << dgv_dist << std::endl;
                  }
                  if (log > 0 && dgi_dist > tol)
                  {
                     std::cout << be << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgi_gf_val[0] << "," << dgi_gf_val[1] << ") "
                               << dgi_dist << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
               REQUIRE( nd_err == Approx(0.0));
               REQUIRE( rt_err == Approx(0.0));
               REQUIRE( l2_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (DG Context)")
         {
            std::cout << "Boundary Evaluation 2D (DG Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double  rt_err = 0.0;
               double  l2_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_2D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);
                  nd_xCoef.Eval(nd_gf_val, *T, ip);
                  rt_xCoef.Eval(rt_gf_val, *T, ip);
                  l2_xCoef.Eval(l2_gf_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gf_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, 2);
                  double  nd_dist = Distance(f_val,  nd_gf_val, 2);
                  double  rt_dist = Distance(f_val,  rt_gf_val, 2);
                  double  l2_dist = Distance(f_val,  l2_gf_val, 2);
                  double dgv_dist = Distance(f_val, dgv_gf_val, 2);
                  double dgi_dist = Distance(f_val, dgi_gf_val, 2);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ") "
                               << h1_dist << std::endl;
                  }
                  if (log > 0 && nd_dist > tol)
                  {
                     std::cout << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << nd_gf_val[0] << "," << nd_gf_val[1] << ") "
                               << nd_dist << std::endl;
                  }
                  if (log > 0 && rt_dist > tol)
                  {
                     std::cout << be << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << rt_gf_val[0] << "," << rt_gf_val[1] << ") "
                               << rt_dist << std::endl;
                  }
                  if (log > 0 && l2_dist > tol)
                  {
                     std::cout << be << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << l2_gf_val[0] << "," << l2_gf_val[1] << ") "
                               << l2_dist << std::endl;
                  }
                  if (log > 0 && dgv_dist > tol)
                  {
                     std::cout << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gf_val[0] << "," << dgv_gf_val[1] << ") "
                               << dgv_dist << std::endl;
                  }
                  if (log > 0 && dgi_dist > tol)
                  {
                     std::cout << be << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgi_gf_val[0] << "," << dgi_gf_val[1] << ") "
                               << dgi_dist << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
               REQUIRE( nd_err == Approx(0.0));
               REQUIRE( rt_err == Approx(0.0));
               REQUIRE( l2_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Edge Evaluation 2D")
         {
            std::cout << "Edge Evaluation 2D" << std::endl;
            for (int e = 0; e < mesh.GetNEdges(); e++)
            {
               ElementTransformation *T = mesh.GetEdgeTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetEdgeElement(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_2D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, 2);

                  h1_err  +=  h1_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ") "
                               << h1_dist << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
            }
         }
      }
   }
   std::cout << "Checked GridFunction::GetVectorValue at "
             << npts << " 2D points" << std::endl;
}

TEST_CASE("3D GetVectorValue",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   int log = 1;
   int n = 1;
   int dim = 3;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient linCoef(dim, Func_3D_lin);

      SECTION("3D GetVectorValue tests for element type " +
              std::to_string(type))
      {
         H1_FECollection  h1_fec(order, dim);
         ND_FECollection  nd_fec(order+1, dim);
         RT_FECollection  rt_fec(order+1, dim);
         L2_FECollection  l2_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);
         DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::INTEGRAL);

         FiniteElementSpace  h1_fespace(&mesh,  &h1_fec, dim);
         FiniteElementSpace  nd_fespace(&mesh,  &nd_fec);
         FiniteElementSpace  rt_fespace(&mesh,  &rt_fec);
         FiniteElementSpace  l2_fespace(&mesh,  &l2_fec, dim);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec, dim);
         FiniteElementSpace dgi_fespace(&mesh, &dgi_fec, dim);

         GridFunction  h1_x( &h1_fespace);
         GridFunction  nd_x( &nd_fespace);
         GridFunction  rt_x( &rt_fespace);
         GridFunction  l2_x( &l2_fespace);
         GridFunction dgv_x(&dgv_fespace);
         GridFunction dgi_x(&dgi_fespace);

         VectorGridFunctionCoefficient  h1_xCoef( &h1_x);
         VectorGridFunctionCoefficient  nd_xCoef( &nd_x);
         VectorGridFunctionCoefficient  rt_xCoef( &rt_x);
         VectorGridFunctionCoefficient  l2_xCoef( &l2_x);
         VectorGridFunctionCoefficient dgv_xCoef(&dgv_x);
         VectorGridFunctionCoefficient dgi_xCoef(&dgi_x);

         h1_x.ProjectCoefficient(linCoef);
         nd_x.ProjectCoefficient(linCoef);
         rt_x.ProjectCoefficient(linCoef);
         l2_x.ProjectCoefficient(linCoef);
         dgv_x.ProjectCoefficient(linCoef);
         dgi_x.ProjectCoefficient(linCoef);

         Vector      f_val(dim);      f_val = 0.0;
         Vector  h1_gf_val(dim);  h1_gf_val = 0.0;
         Vector  nd_gf_val(dim);  nd_gf_val = 0.0;
         Vector  rt_gf_val(dim);  rt_gf_val = 0.0;
         Vector  l2_gf_val(dim);  l2_gf_val = 0.0;
         Vector dgv_gf_val(dim); dgv_gf_val = 0.0;
         Vector dgi_gf_val(dim); dgi_gf_val = 0.0;

         SECTION("Domain Evaluation 3D")
         {
            std::cout << "Domain Evaluation 3D" << std::endl;
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double  rt_err = 0.0;
               double  l2_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_3D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);
                  nd_xCoef.Eval(nd_gf_val, *T, ip);
                  rt_xCoef.Eval(rt_gf_val, *T, ip);
                  l2_xCoef.Eval(l2_gf_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gf_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, dim);
                  double  nd_dist = Distance(f_val,  nd_gf_val, dim);
                  double  rt_dist = Distance(f_val,  rt_gf_val, dim);
                  double  l2_dist = Distance(f_val,  l2_gf_val, dim);
                  double dgv_dist = Distance(f_val, dgv_gf_val, dim);
                  double dgi_dist = Distance(f_val, dgi_gf_val, dim);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ","
                               << h1_gf_val[2] << ") " << h1_dist
                               << std::endl;
                  }
                  if (log > 0 && nd_dist > tol)
                  {
                     std::cout << e << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gf_val[0] << "," << nd_gf_val[1] << ","
                               << nd_gf_val[2] << ") " << nd_dist
                               << std::endl;
                  }
                  if (log > 0 && rt_dist > tol)
                  {
                     std::cout << e << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gf_val[0] << "," << rt_gf_val[1] << ","
                               << rt_gf_val[2] << ") " << rt_dist
                               << std::endl;
                  }
                  if (log > 0 && l2_dist > tol)
                  {
                     std::cout << e << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gf_val[0] << "," << l2_gf_val[1] << ","
                               << l2_gf_val[2] << ") " << l2_dist
                               << std::endl;
                  }
                  if (log > 0 && dgv_dist > tol)
                  {
                     std::cout << e << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gf_val[0] << "," << dgv_gf_val[1] << ","
                               << dgv_gf_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
                  if (log > 0 && dgi_dist > tol)
                  {
                     std::cout << e << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gf_val[0] << "," << dgi_gf_val[1] << ","
                               << dgi_gf_val[2] << ") " << dgi_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
               REQUIRE( nd_err == Approx(0.0));
               REQUIRE( rt_err == Approx(0.0));
               REQUIRE( l2_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (H1 Context)")
         {
            std::cout << "Boundary Evaluation 3D (H1 Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double  rt_err = 0.0;
               double  l2_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_3D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);
                  nd_xCoef.Eval(nd_gf_val, *T, ip);
                  rt_xCoef.Eval(rt_gf_val, *T, ip);
                  l2_xCoef.Eval(l2_gf_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gf_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, dim);
                  double  nd_dist = Distance(f_val,  nd_gf_val, dim);
                  double  rt_dist = Distance(f_val,  rt_gf_val, dim);
                  double  l2_dist = Distance(f_val,  l2_gf_val, dim);
                  double dgv_dist = Distance(f_val, dgv_gf_val, dim);
                  double dgi_dist = Distance(f_val, dgi_gf_val, dim);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ","
                               << h1_gf_val[2] << ") " << h1_dist
                               << std::endl;
                  }
                  if (log > 0 && nd_dist > tol)
                  {
                     std::cout << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gf_val[0] << "," << nd_gf_val[1] << ","
                               << nd_gf_val[2] << ") " << nd_dist
                               << std::endl;
                  }
                  if (log > 0 && rt_dist > tol)
                  {
                     std::cout << be << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gf_val[0] << "," << rt_gf_val[1] << ","
                               << rt_gf_val[2] << ") " << rt_dist
                               << std::endl;
                  }
                  if (log > 0 && l2_dist > tol)
                  {
                     std::cout << be << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gf_val[0] << "," << l2_gf_val[1] << ","
                               << l2_gf_val[2] << ") " << l2_dist
                               << std::endl;
                  }
                  if (log > 0 && dgv_dist > tol)
                  {
                     std::cout << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gf_val[0] << "," << dgv_gf_val[1] << ","
                               << dgv_gf_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
                  if (log > 0 && dgi_dist > tol)
                  {
                     std::cout << be << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gf_val[0] << "," << dgi_gf_val[1] << ","
                               << dgi_gf_val[2] << ") " << dgi_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
               REQUIRE( nd_err == Approx(0.0));
               REQUIRE( rt_err == Approx(0.0));
               REQUIRE( l2_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (DG Context)")
         {
            std::cout << "Boundary Evaluation 3D (DG Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double  rt_err = 0.0;
               double  l2_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_3D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);
                  nd_xCoef.Eval(nd_gf_val, *T, ip);
                  rt_xCoef.Eval(rt_gf_val, *T, ip);
                  l2_xCoef.Eval(l2_gf_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gf_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, dim);
                  double  nd_dist = Distance(f_val,  nd_gf_val, dim);
                  double  rt_dist = Distance(f_val,  rt_gf_val, dim);
                  double  l2_dist = Distance(f_val,  l2_gf_val, dim);
                  double dgv_dist = Distance(f_val, dgv_gf_val, dim);
                  double dgi_dist = Distance(f_val, dgi_gf_val, dim);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ","
                               << h1_gf_val[2] << ") " << h1_dist
                               << std::endl;
                  }
                  if (log > 0 && nd_dist > tol)
                  {
                     std::cout << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gf_val[0] << "," << nd_gf_val[1] << ","
                               << nd_gf_val[2] << ") " << nd_dist
                               << std::endl;
                  }
                  if (log > 0 && rt_dist > tol)
                  {
                     std::cout << be << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gf_val[0] << "," << rt_gf_val[1] << ","
                               << rt_gf_val[2] << ") " << rt_dist
                               << std::endl;
                  }
                  if (log > 0 && l2_dist > tol)
                  {
                     std::cout << be << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gf_val[0] << "," << l2_gf_val[1] << ","
                               << l2_gf_val[2] << ") " << l2_dist
                               << std::endl;
                  }
                  if (log > 0 && dgv_dist > tol)
                  {
                     std::cout << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gf_val[0] << "," << dgv_gf_val[1] << ","
                               << dgv_gf_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
                  if (log > 0 && dgi_dist > tol)
                  {
                     std::cout << be << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gf_val[0] << "," << dgi_gf_val[1] << ","
                               << dgi_gf_val[2] << ") " << dgi_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
               REQUIRE( nd_err == Approx(0.0));
               REQUIRE( rt_err == Approx(0.0));
               REQUIRE( l2_err == Approx(0.0));
               REQUIRE(dgv_err == Approx(0.0));
               REQUIRE(dgi_err == Approx(0.0));
            }
         }

         SECTION("Edge Evaluation 3D")
         {
            std::cout << "Edge Evaluation 3D" << std::endl;
            for (int e = 0; e < mesh.GetNEdges(); e++)
            {
               ElementTransformation *T = mesh.GetEdgeTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetEdgeElement(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_3D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, dim);

                  h1_err  +=  h1_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ","
                               << h1_gf_val[2] << ") " << h1_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
            }
         }

         SECTION("Face Evaluation 3D")
         {
            std::cout << "Face Evaluation 3D" << std::endl;
            for (int f = 0; f < mesh.GetNFaces(); f++)
            {
               ElementTransformation *T = mesh.GetFaceTransformation(f);
               const FiniteElement   *fe = h1_fespace.GetFaceElement(f);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;

               double tip_data[dim];
               Vector tip(tip_data, dim);
               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, tip);

                  Func_3D_lin(tip, f_val);

                  h1_xCoef.Eval(h1_gf_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gf_val, dim);

                  h1_err  +=  h1_dist;

                  if (log > 0 && h1_dist > tol)
                  {
                     std::cout << f << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gf_val[0] << "," << h1_gf_val[1] << ","
                               << h1_gf_val[2] << ") " << h1_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();

               REQUIRE( h1_err == Approx(0.0));
            }
         }
      }
   }
   std::cout << "Checked GridFunction::GetVectorValue at "
             << npts << " 3D points" << std::endl;
}

} // namespace get_value
