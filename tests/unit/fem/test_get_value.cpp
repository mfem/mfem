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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace get_value
{

static int first_1D_et = (int)Element::SEGMENT;
static int  last_1D_et = (int)Element::SEGMENT;
static int first_2D_et = (int)Element::TRIANGLE;
static int  last_2D_et = (int)Element::QUADRILATERAL;
static int first_3D_et = (int)Element::TETRAHEDRON;
static int  last_3D_et = (int)Element::PYRAMID;

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

double func_1D_quad(const Vector &x)
{
   return 2.0 * x[0] + x[0] * x[0];
}

void dfunc_1D_quad(const Vector &x, Vector &v)
{
   v.SetSize(1);
   v[0] = 2.0 + 2.0 * x[0];
}

double func_2D_quad(const Vector &x)
{
   return x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[0] * x[1];
}

void dfunc_2D_quad(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v[0] = 2.0 * x[0] + 3.0 * x[1];
   v[1] = 4.0 * x[1] + 3.0 * x[0];
}

void Func_2D_quad(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v[0] = 1.0 * x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[0] * x[1];
   v[1] = 2.0 * x[1] * x[1] + 3.0 * x[0] * x[0] + 1.0 * x[0] * x[1];
}

void RotFunc_2D_quad(const Vector &x, Vector &v)
{
   v.SetSize(1);
   v[0] = 6.0 * x[0] + 1.0 * x[1] - 4.0 * x[1] - 3.0 * x[0];
}

double DivFunc_2D_quad(const Vector &x)
{
   return 3.0 * x[0] + 7.0 * x[1];
}

double func_3D_quad(const Vector &x)
{
   return x[0] * x[1] + 2.0 * x[1] * x[2] + 3.0 * x[2] * x[0];
}

void dfunc_3D_quad(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] = 1.0 * x[1] + 3.0 * x[2];
   v[1] = 2.0 * x[2] + 1.0 * x[0];
   v[2] = 3.0 * x[0] + 2.0 * x[1];
}

void Func_3D_quad(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] = 1.0 * x[0] * x[1] + 2.0 * x[1] * x[2] + 3.0 * x[2] * x[0];
   v[1] = 2.0 * x[1] * x[2] + 3.0 * x[2] * x[0] + 1.0 * x[0] * x[1];
   v[2] = 3.0 * x[2] * x[0] + 1.0 * x[0] * x[1] + 2.0 * x[1] * x[2];
}

void CurlFunc_3D_quad(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] = 1.0 * x[0] + 2.0 * x[2] - 2.0 * x[1] - 3.0 * x[0];
   v[1] = 2.0 * x[1] + 3.0 * x[0] - 3.0 * x[2] - 1.0 * x[1];
   v[2] = 3.0 * x[2] + 1.0 * x[1] - 1.0 * x[0] - 2.0 * x[2];
}

double DivFunc_3D_quad(const Vector &x)
{
   return 4.0 * x[0] + 3.0 * x[1] + 5.0 * x[2];
}

TEST_CASE("1D GetValue",
          "[GridFunction]"
          "[GridFunctionCoefficient]")
{
   int n = 1;
   int dim = 1;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_1D_et; type <= last_1D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      FunctionCoefficient funcCoef(func_1D_lin);

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

         h1_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         SECTION("Domain Evaluation 1D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double h1_gv_err = 0.0;
               double dgv_gv_err = 0.0;
               double dgi_gv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);

                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  double  h1_gv_val =  h1_x.GetValue(e, ip);
                  double dgv_gv_val = dgv_x.GetValue(e, ip);
                  double dgi_gv_val = dgi_x.GetValue(e, ip);

                  h1_gfc_err += fabs(f_val - h1_gfc_val);
                  dgv_gfc_err += fabs(f_val - dgv_gfc_val);
                  dgi_gfc_err += fabs(f_val - dgi_gfc_val);

                  h1_gv_err += fabs(f_val - h1_gv_val);
                  dgv_gv_err += fabs(f_val - dgv_gv_val);
                  dgi_gv_err += fabs(f_val - dgi_gv_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - h1_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gv " << f_val << " "
                               << h1_gv_val << " " << fabs(f_val - h1_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gv " << f_val << " "
                               << dgv_gv_val << " "
                               << fabs(f_val - dgv_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gv " << f_val << " "
                               << dgi_gv_val << " "
                               << fabs(f_val - dgi_gv_val)
                               << std::endl;
                  }
               }

               h1_gfc_err /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gv_err /= ir.GetNPoints();
               dgv_gv_err /= ir.GetNPoints();
               dgi_gv_err /= ir.GetNPoints();

               REQUIRE(h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE(h1_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 1D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);
                  dgi_err += fabs(f_val - dgi_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 1D (DG Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);
                  dgi_err += fabs(f_val - dgi_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetValue at "
             << npts << " 1D points" << std::endl;
}

#ifdef MFEM_USE_MPI

TEST_CASE("1D GetValue in Parallel",
          "[ParGridFunction]"
          "[GridFunctionCoefficient]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int n = 2 * num_procs;
   int dim = 1;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_1D_et; type <= last_1D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);
      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      FunctionCoefficient funcCoef(func_1D_lin);

      SECTION("1D GetValue tests for element type " + std::to_string(type))
      {
         H1_FECollection h1_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);
         DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::INTEGRAL);

         ParFiniteElementSpace h1_fespace(&pmesh, &h1_fec);
         ParFiniteElementSpace dgv_fespace(&pmesh, &dgv_fec);
         ParFiniteElementSpace dgi_fespace(&pmesh, &dgi_fec);

         ParGridFunction h1_x(&h1_fespace);
         ParGridFunction dgv_x(&dgv_fespace);
         ParGridFunction dgi_x(&dgi_fespace);

         GridFunctionCoefficient h1_xCoef(&h1_x);
         GridFunctionCoefficient dgv_xCoef(&dgv_x);
         GridFunctionCoefficient dgi_xCoef(&dgi_x);

         h1_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         h1_x.ExchangeFaceNbrData();
         dgv_x.ExchangeFaceNbrData();
         dgi_x.ExchangeFaceNbrData();

         SECTION("Shared Face Evaluation 1D")
         {
            for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++)
            {
               FaceElementTransformations *FET =
                  pmesh.GetSharedFaceTransformations(sf);
               ElementTransformation *T = &FET->GetElement2Transformation();
               int e = FET->Elem2No;
               int e_nbr = e - pmesh.GetNE();
               const FiniteElement   *fe = dgv_fespace.GetFaceNbrFE(e_nbr);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double  h1_gv_err = 0.0;
               double dgv_gv_err = 0.0;
               double dgi_gv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double      f_val =   funcCoef.Eval(*T, ip);

                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  double h1_gv_val = h1_x.GetValue(e, ip);
                  double dgv_gv_val = dgv_x.GetValue(e, ip);
                  double dgi_gv_val = dgi_x.GetValue(e, ip);

                  h1_gfc_err  += fabs(f_val -  h1_gfc_val);
                  dgv_gfc_err += fabs(f_val - dgv_gfc_val);
                  dgi_gfc_err += fabs(f_val - dgi_gfc_val);

                  h1_gv_err += fabs(f_val - h1_gv_val);
                  dgv_gv_err += fabs(f_val - dgv_gv_val);
                  dgi_gv_err += fabs(f_val - dgi_gv_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - h1_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gv " << f_val << " "
                               << h1_gv_val << " " << fabs(f_val - h1_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gv " << f_val << " "
                               << dgv_gv_val << " "
                               << fabs(f_val - dgv_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gv " << f_val << " "
                               << dgi_gv_val << " "
                               << fabs(f_val - dgi_gv_val)
                               << std::endl;
                  }
               }

               h1_gfc_err /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gv_err /= ir.GetNPoints();
               dgv_gv_err /= ir.GetNPoints();
               dgi_gv_err /= ir.GetNPoints();

               REQUIRE(h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE(h1_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << my_rank << ": Checked GridFunction::GetValue at "
             << npts << " 1D points" << std::endl;
}

#endif // MFEM_USE_MPI

TEST_CASE("2D GetValue",
          "[GridFunction]"
          "[GridFunctionCoefficient]")
{
   int n = 1;
   int dim = 2;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_2D_et; type <= last_2D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(n, n, (Element::Type)type, 1, 2.0, 3.0);

      FunctionCoefficient funcCoef(func_2D_lin);

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

         h1_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         SECTION("Domain Evaluation 2D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double h1_gv_err = 0.0;
               double dgv_gv_err = 0.0;
               double dgi_gv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);

                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  double h1_gv_val = h1_x.GetValue(e, ip);
                  double dgv_gv_val = dgv_x.GetValue(e, ip);
                  double dgi_gv_val = dgi_x.GetValue(e, ip);

                  h1_gfc_err += fabs(f_val - h1_gfc_val);
                  dgv_gfc_err += fabs(f_val - dgv_gfc_val);
                  dgi_gfc_err += fabs(f_val - dgi_gfc_val);

                  h1_gv_err += fabs(f_val - h1_gv_val);
                  dgv_gv_err += fabs(f_val - dgv_gv_val);
                  dgi_gv_err += fabs(f_val - dgi_gv_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - h1_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gv " << f_val << " "
                               << h1_gv_val << " " << fabs(f_val - h1_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gv " << f_val << " "
                               << dgv_gv_val << " "
                               << fabs(f_val - dgv_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gv " << f_val << " "
                               << dgi_gv_val << " "
                               << fabs(f_val - dgi_gv_val)
                               << std::endl;
                  }
               }

               h1_gfc_err /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gv_err /= ir.GetNPoints();
               dgv_gv_err /= ir.GetNPoints();
               dgi_gv_err /= ir.GetNPoints();

               REQUIRE(h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE(h1_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);
                  dgi_err += fabs(f_val - dgi_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (DG Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);
                  dgi_err += fabs(f_val - dgi_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Edge Evaluation 2D (H1 Context)")
         {
            for (int e = 0; e < mesh.GetNEdges(); e++)
            {
               ElementTransformation *T = mesh.GetEdgeTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetEdgeElement(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetValue at "
             << npts << " 2D points" << std::endl;
}

#ifdef MFEM_USE_MPI

TEST_CASE("2D GetValue in Parallel",
          "[ParGridFunction]"
          "[GridFunctionCoefficient]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int n = (int)ceil(sqrt(2*num_procs));
   int dim = 2;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_2D_et; type <= last_2D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(n, n, (Element::Type)type, 1, 2.0, 3.0);
      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      FunctionCoefficient funcCoef(func_2D_lin);

      SECTION("2D GetValue tests for element type " + std::to_string(type))
      {
         H1_FECollection h1_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);
         DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::INTEGRAL);

         ParFiniteElementSpace h1_fespace(&pmesh, &h1_fec);
         ParFiniteElementSpace dgv_fespace(&pmesh, &dgv_fec);
         ParFiniteElementSpace dgi_fespace(&pmesh, &dgi_fec);

         ParGridFunction h1_x(&h1_fespace);
         ParGridFunction dgv_x(&dgv_fespace);
         ParGridFunction dgi_x(&dgi_fespace);

         GridFunctionCoefficient h1_xCoef(&h1_x);
         GridFunctionCoefficient dgv_xCoef(&dgv_x);
         GridFunctionCoefficient dgi_xCoef(&dgi_x);

         h1_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         h1_x.ExchangeFaceNbrData();
         dgv_x.ExchangeFaceNbrData();
         dgi_x.ExchangeFaceNbrData();

         SECTION("Shared Face Evaluation 2D")
         {
            for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++)
            {
               FaceElementTransformations *FET =
                  pmesh.GetSharedFaceTransformations(sf);
               ElementTransformation *T = &FET->GetElement2Transformation();
               int e = FET->Elem2No;
               int e_nbr = e - pmesh.GetNE();
               const FiniteElement   *fe = dgv_fespace.GetFaceNbrFE(e_nbr);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double  h1_gv_err = 0.0;
               double dgv_gv_err = 0.0;
               double dgi_gv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double      f_val =   funcCoef.Eval(*T, ip);
                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  double h1_gv_val = h1_x.GetValue(e, ip);
                  double dgv_gv_val = dgv_x.GetValue(e, ip);
                  double dgi_gv_val = dgi_x.GetValue(e, ip);

                  h1_gfc_err  += fabs(f_val -  h1_gfc_val);
                  dgv_gfc_err += fabs(f_val - dgv_gfc_val);
                  dgi_gfc_err += fabs(f_val - dgi_gfc_val);

                  h1_gv_err += fabs(f_val - h1_gv_val);
                  dgv_gv_err += fabs(f_val - dgv_gv_val);
                  dgi_gv_err += fabs(f_val - dgi_gv_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - h1_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gv " << f_val << " "
                               << h1_gv_val << " " << fabs(f_val - h1_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gv " << f_val << " "
                               << dgv_gv_val << " "
                               << fabs(f_val - dgv_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gv " << f_val << " "
                               << dgi_gv_val << " "
                               << fabs(f_val - dgi_gv_val)
                               << std::endl;
                  }
               }

               h1_gfc_err /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gv_err /= ir.GetNPoints();
               dgv_gv_err /= ir.GetNPoints();
               dgi_gv_err /= ir.GetNPoints();

               REQUIRE(h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE(h1_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << my_rank << ": Checked GridFunction::GetValue at "
             << npts << " 2D points" << std::endl;
}

#endif // MFEM_USE_MPI

TEST_CASE("3D GetValue",
          "[GridFunction]"
          "[GridFunctionCoefficient]")
{
   int n = 1;
   int dim = 3;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_3D_et; type <= last_3D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      FunctionCoefficient funcCoef(func_3D_lin);

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

         h1_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         SECTION("Domain Evaluation 3D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double h1_gv_err = 0.0;
               double dgv_gv_err = 0.0;
               double dgi_gv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);

                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  double h1_gv_val = h1_x.GetValue(e, ip);
                  double dgv_gv_val = dgv_x.GetValue(e, ip);
                  double dgi_gv_val = dgi_x.GetValue(e, ip);

                  h1_gfc_err += fabs(f_val - h1_gfc_val);
                  dgv_gfc_err += fabs(f_val - dgv_gfc_val);
                  dgi_gfc_err += fabs(f_val - dgi_gfc_val);

                  h1_gv_err += fabs(f_val - h1_gv_val);
                  dgv_gv_err += fabs(f_val - dgv_gv_val);
                  dgi_gv_err += fabs(f_val - dgi_gv_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - h1_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gv " << f_val << " "
                               << h1_gv_val << " " << fabs(f_val - h1_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gv " << f_val << " "
                               << dgv_gv_val << " "
                               << fabs(f_val - dgv_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gv " << f_val << " "
                               << dgi_gv_val << " "
                               << fabs(f_val - dgi_gv_val)
                               << std::endl;
                  }
               }

               h1_gfc_err /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gv_err /= ir.GetNPoints();
               dgv_gv_err /= ir.GetNPoints();
               dgi_gv_err /= ir.GetNPoints();

               REQUIRE(h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE(h1_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);
                  dgi_err += fabs(f_val - dgi_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (DG Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;
               double dgi_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);
                  dgi_err += fabs(f_val - dgi_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgi " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Edge Evaluation 3D (H1 Context)")
         {
            for (int e = 0; e < mesh.GetNEdges(); e++)
            {
               ElementTransformation *T = mesh.GetEdgeTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetEdgeElement(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Face Evaluation 3D (H1 Context)")
         {
            for (int f = 0; f < mesh.GetNFaces(); f++)
            {
               ElementTransformation *T = mesh.GetFaceTransformation(f);
               const FiniteElement   *fe = h1_fespace.GetFaceElement(f);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double f_val = funcCoef.Eval(*T, ip);
                  double h1_gfc_val = h1_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << f << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetValue at "
             << npts << " 3D points" << std::endl;
}

#ifdef MFEM_USE_MPI

TEST_CASE("3D GetValue in Parallel",
          "[ParGridFunction]"
          "[GridFunctionCoefficient]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int n = (int)ceil(pow(2*num_procs, 1.0 / 3.0));
   int dim = 3;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_3D_et; type <= last_3D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);
      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      FunctionCoefficient funcCoef(func_3D_lin);

      SECTION("3D GetValue tests for element type " + std::to_string(type))
      {
         H1_FECollection  h1_fec(order, dim);
         L2_FECollection  l2_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);
         DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::INTEGRAL);

         ParFiniteElementSpace  h1_fespace(&pmesh, &h1_fec);
         ParFiniteElementSpace  l2_fespace(&pmesh,  &l2_fec);
         ParFiniteElementSpace dgv_fespace(&pmesh, &dgv_fec);
         ParFiniteElementSpace dgi_fespace(&pmesh, &dgi_fec);

         ParGridFunction  h1_x( &h1_fespace);
         ParGridFunction  l2_x( &l2_fespace);
         ParGridFunction dgv_x(&dgv_fespace);
         ParGridFunction dgi_x(&dgi_fespace);

         GridFunctionCoefficient  h1_xCoef( &h1_x);
         GridFunctionCoefficient  l2_xCoef( &l2_x);
         GridFunctionCoefficient dgv_xCoef(&dgv_x);
         GridFunctionCoefficient dgi_xCoef(&dgi_x);

         h1_x.ProjectCoefficient(funcCoef);
         l2_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         h1_x.ExchangeFaceNbrData();
         l2_x.ExchangeFaceNbrData();
         dgv_x.ExchangeFaceNbrData();
         dgi_x.ExchangeFaceNbrData();

         SECTION("Shared Face Evaluation 3D")
         {
            for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++)
            {
               FaceElementTransformations *FET =
                  pmesh.GetSharedFaceTransformations(sf);
               ElementTransformation *T = &FET->GetElement2Transformation();
               int e = FET->Elem2No;
               int e_nbr = e - pmesh.GetNE();
               const FiniteElement   *fe = dgv_fespace.GetFaceNbrFE(e_nbr);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double  h1_gv_err = 0.0;
               double dgv_gv_err = 0.0;
               double dgi_gv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double      f_val =   funcCoef.Eval(*T, ip);

                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);
                  double dgi_gfc_val = dgi_xCoef.Eval(*T, ip);

                  double h1_gv_val = h1_x.GetValue(e, ip);
                  double dgv_gv_val = dgv_x.GetValue(e, ip);
                  double dgi_gv_val = dgi_x.GetValue(e, ip);

                  h1_gfc_err  += fabs(f_val -  h1_gfc_val);
                  dgv_gfc_err += fabs(f_val - dgv_gfc_val);
                  dgi_gfc_err += fabs(f_val - dgi_gfc_val);

                  h1_gv_err += fabs(f_val - h1_gv_val);
                  dgv_gv_err += fabs(f_val - dgv_gv_val);
                  dgi_gv_err += fabs(f_val - dgi_gv_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc " << f_val << " "
                               << h1_gfc_val << " " << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc " << f_val << " "
                               << dgi_gfc_val << " "
                               << fabs(f_val - dgi_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - h1_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gv " << f_val << " "
                               << h1_gv_val << " " << fabs(f_val - h1_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gv " << f_val << " "
                               << dgv_gv_val << " "
                               << fabs(f_val - dgv_gv_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgi_gv_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gv " << f_val << " "
                               << dgi_gv_val << " "
                               << fabs(f_val - dgi_gv_val)
                               << std::endl;
                  }
               }

               h1_gfc_err /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gv_err /= ir.GetNPoints();
               dgv_gv_err /= ir.GetNPoints();
               dgi_gv_err /= ir.GetNPoints();

               REQUIRE(h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE(h1_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << my_rank << ": Checked GridFunction::GetValue at "
             << npts << " 3D points" << std::endl;
}

#endif // MFEM_USE_MPI

TEST_CASE("2D GetVectorValue",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 2;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_2D_et; type <= last_2D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(n, n, (Element::Type)type, 1, 2.0, 3.0);

      VectorFunctionCoefficient funcCoef(dim, Func_2D_lin);

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

         h1_x.ProjectCoefficient(funcCoef);
         nd_x.ProjectCoefficient(funcCoef);
         rt_x.ProjectCoefficient(funcCoef);
         l2_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         Vector       f_val(dim);       f_val = 0.0;

         Vector  h1_gfc_val(dim);  h1_gfc_val = 0.0;
         Vector  nd_gfc_val(dim);  nd_gfc_val = 0.0;
         Vector  rt_gfc_val(dim);  rt_gfc_val = 0.0;
         Vector  l2_gfc_val(dim);  l2_gfc_val = 0.0;
         Vector dgv_gfc_val(dim); dgv_gfc_val = 0.0;
         Vector dgi_gfc_val(dim); dgi_gfc_val = 0.0;

         Vector  h1_gvv_val(dim);  h1_gvv_val = 0.0;
         Vector  nd_gvv_val(dim);  nd_gvv_val = 0.0;
         Vector  rt_gvv_val(dim);  rt_gvv_val = 0.0;
         Vector  l2_gvv_val(dim);  l2_gvv_val = 0.0;
         Vector dgv_gvv_val(dim); dgv_gvv_val = 0.0;
         Vector dgi_gvv_val(dim); dgi_gvv_val = 0.0;

         SECTION("Domain Evaluation 2D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_gfc_err = 0.0;
               double  nd_gfc_err = 0.0;
               double  rt_gfc_err = 0.0;
               double  l2_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double  h1_gvv_err = 0.0;
               double  nd_gvv_err = 0.0;
               double  rt_gvv_err = 0.0;
               double  l2_gvv_err = 0.0;
               double dgv_gvv_err = 0.0;
               double dgi_gvv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);

                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  h1_x.GetVectorValue(e, ip, h1_gvv_val);
                  nd_x.GetVectorValue(e, ip, nd_gvv_val);
                  rt_x.GetVectorValue(e, ip, rt_gvv_val);
                  l2_x.GetVectorValue(e, ip, l2_gvv_val);
                  dgv_x.GetVectorValue(e, ip, dgv_gvv_val);
                  dgi_x.GetVectorValue(e, ip, dgi_gvv_val);

                  double  h1_gfc_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_gfc_dist = Distance(f_val,  nd_gfc_val);
                  double  rt_gfc_dist = Distance(f_val,  rt_gfc_val);
                  double  l2_gfc_dist = Distance(f_val,  l2_gfc_val);
                  double dgv_gfc_dist = Distance(f_val, dgv_gfc_val);
                  double dgi_gfc_dist = Distance(f_val, dgi_gfc_val);

                  double  h1_gvv_dist = Distance(f_val,  h1_gvv_val);
                  double  nd_gvv_dist = Distance(f_val,  nd_gvv_val);
                  double  rt_gvv_dist = Distance(f_val,  rt_gvv_val);
                  double  l2_gvv_dist = Distance(f_val,  l2_gvv_val);
                  double dgv_gvv_dist = Distance(f_val, dgv_gvv_val);
                  double dgi_gvv_dist = Distance(f_val, dgi_gvv_val);

                  h1_gfc_err  +=  h1_gfc_dist;
                  nd_gfc_err  +=  nd_gfc_dist;
                  rt_gfc_err  +=  rt_gfc_dist;
                  l2_gfc_err  +=  l2_gfc_dist;
                  dgv_gfc_err += dgv_gfc_dist;
                  dgi_gfc_err += dgi_gfc_dist;

                  h1_gvv_err  +=  h1_gvv_dist;
                  nd_gvv_err  +=  nd_gvv_dist;
                  rt_gvv_err  +=  rt_gvv_dist;
                  l2_gvv_err  +=  l2_gvv_dist;
                  dgv_gvv_err += dgv_gvv_dist;
                  dgi_gvv_err += dgi_gvv_dist;

                  if (verbose_tests && h1_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ") "
                               << h1_gfc_dist << std::endl;
                  }
                  if (verbose_tests && nd_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  gfc ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ") "
                               << nd_gfc_dist << std::endl;
                  }
                  if (verbose_tests && rt_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  gfc ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << rt_gfc_val[0] << "," << rt_gfc_val[1] << ") "
                               << rt_gfc_dist << std::endl;
                  }
                  if (verbose_tests && l2_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " l2  gfc ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << l2_gfc_val[0] << "," << l2_gfc_val[1] << ") "
                               << l2_gfc_dist << std::endl;
                  }
                  if (verbose_tests && dgv_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ") "
                               << dgv_gfc_dist << std::endl;
                  }
                  if (verbose_tests && dgi_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgi_gfc_val[0] << ","
                               << dgi_gfc_val[1] << ") "
                               << dgi_gfc_dist << std::endl;
                  }
                  if (verbose_tests && h1_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gvv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gvv_val[0] << "," << h1_gvv_val[1] << ") "
                               << h1_gvv_dist << std::endl;
                  }
                  if (verbose_tests && nd_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  gvv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << nd_gvv_val[0] << "," << nd_gvv_val[1] << ") "
                               << nd_gvv_dist << std::endl;
                  }
                  if (verbose_tests && rt_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  gvv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << rt_gvv_val[0] << "," << rt_gvv_val[1] << ") "
                               << rt_gvv_dist << std::endl;
                  }
                  if (verbose_tests && l2_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " l2  gvv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << l2_gvv_val[0] << "," << l2_gvv_val[1] << ") "
                               << l2_gvv_dist << std::endl;
                  }
                  if (verbose_tests && dgv_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gvv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gvv_val[0] << ","
                               << dgv_gvv_val[1] << ") "
                               << dgv_gvv_dist << std::endl;
                  }
                  if (verbose_tests && dgi_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gvv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgi_gvv_val[0] << ","
                               << dgi_gvv_val[1] << ") "
                               << dgi_gvv_dist << std::endl;
                  }
               }

               h1_gfc_err  /= ir.GetNPoints();
               nd_gfc_err  /= ir.GetNPoints();
               rt_gfc_err  /= ir.GetNPoints();
               l2_gfc_err  /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gvv_err  /= ir.GetNPoints();
               nd_gvv_err  /= ir.GetNPoints();
               rt_gvv_err  /= ir.GetNPoints();
               l2_gvv_err  /= ir.GetNPoints();
               dgv_gvv_err /= ir.GetNPoints();
               dgi_gvv_err /= ir.GetNPoints();

               REQUIRE( h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE( nd_gfc_err == MFEM_Approx(0.0));
               REQUIRE( rt_gfc_err == MFEM_Approx(0.0));
               REQUIRE( l2_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE( h1_gvv_err == MFEM_Approx(0.0));
               REQUIRE( nd_gvv_err == MFEM_Approx(0.0));
               REQUIRE( rt_gvv_err == MFEM_Approx(0.0));
               REQUIRE( l2_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gvv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (H1 Context)")
         {
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

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double  rt_dist = Distance(f_val,  rt_gfc_val);
                  double  l2_dist = Distance(f_val,  l2_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);
                  double dgi_dist = Distance(f_val, dgi_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ") "
                               << nd_dist << std::endl;
                  }
                  if (verbose_tests && rt_dist > tol)
                  {
                     mfem::out << be << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << rt_gfc_val[0] << "," << rt_gfc_val[1] << ") "
                               << rt_dist << std::endl;
                  }
                  if (verbose_tests && l2_dist > tol)
                  {
                     mfem::out << be << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << l2_gfc_val[0] << "," << l2_gfc_val[1] << ") "
                               << l2_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ") "
                               << dgv_dist << std::endl;
                  }
                  if (verbose_tests && dgi_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgi_gfc_val[0] << ","
                               << dgi_gfc_val[1] << ") "
                               << dgi_dist << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE( rt_err == MFEM_Approx(0.0));
               REQUIRE( l2_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (DG Context)")
         {
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

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double  rt_dist = Distance(f_val,  rt_gfc_val);
                  double  l2_dist = Distance(f_val,  l2_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);
                  double dgi_dist = Distance(f_val, dgi_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gfc_val[0] << ","
                               << h1_gfc_val[1] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << nd_gfc_val[0] << ","
                               << nd_gfc_val[1] << ") "
                               << nd_dist << std::endl;
                  }
                  if (verbose_tests && rt_dist > tol)
                  {
                     mfem::out << be << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << rt_gfc_val[0] << ","
                               << rt_gfc_val[1] << ") "
                               << rt_dist << std::endl;
                  }
                  if (verbose_tests && l2_dist > tol)
                  {
                     mfem::out << be << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << l2_gfc_val[0] << ","
                               << l2_gfc_val[1] << ") "
                               << l2_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ") "
                               << dgv_dist << std::endl;
                  }
                  if (verbose_tests && dgi_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgi_gfc_val[0] << ","
                               << dgi_gfc_val[1] << ") "
                               << dgi_dist << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE( rt_err == MFEM_Approx(0.0));
               REQUIRE( l2_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Edge Evaluation 2D")
         {
            for (int e = 0; e < mesh.GetNEdges(); e++)
            {
               ElementTransformation *T = mesh.GetEdgeTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetEdgeElement(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);

                  h1_err  +=  h1_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ") "
                               << h1_dist << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetVectorValue at "
             << npts << " 2D points" << std::endl;
}

#ifdef MFEM_USE_MPI

TEST_CASE("2D GetVectorValue in Parallel",
          "[ParGridFunction]"
          "[VectorGridFunctionCoefficient]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int n = (int)ceil(sqrt(2*num_procs));
   int dim = 2;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_2D_et; type <= last_2D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(n, n, (Element::Type)type, 1, 2.0, 3.0);
      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      VectorFunctionCoefficient funcCoef(dim, Func_2D_lin);

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

         ParFiniteElementSpace  h1_fespace(&pmesh,  &h1_fec, dim);
         ParFiniteElementSpace  nd_fespace(&pmesh,  &nd_fec);
         ParFiniteElementSpace  rt_fespace(&pmesh,  &rt_fec);
         ParFiniteElementSpace  l2_fespace(&pmesh,  &l2_fec, dim);
         ParFiniteElementSpace dgv_fespace(&pmesh, &dgv_fec, dim);
         ParFiniteElementSpace dgi_fespace(&pmesh, &dgi_fec, dim);

         ParGridFunction  h1_x( &h1_fespace);
         ParGridFunction  nd_x( &nd_fespace);
         ParGridFunction  rt_x( &rt_fespace);
         ParGridFunction  l2_x( &l2_fespace);
         ParGridFunction dgv_x(&dgv_fespace);
         ParGridFunction dgi_x(&dgi_fespace);

         VectorGridFunctionCoefficient  h1_xCoef( &h1_x);
         VectorGridFunctionCoefficient  nd_xCoef( &nd_x);
         VectorGridFunctionCoefficient  rt_xCoef( &rt_x);
         VectorGridFunctionCoefficient  l2_xCoef( &l2_x);
         VectorGridFunctionCoefficient dgv_xCoef(&dgv_x);
         VectorGridFunctionCoefficient dgi_xCoef(&dgi_x);

         h1_x.ProjectCoefficient(funcCoef);
         nd_x.ProjectCoefficient(funcCoef);
         rt_x.ProjectCoefficient(funcCoef);
         l2_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         h1_x.ExchangeFaceNbrData();
         nd_x.ExchangeFaceNbrData();
         rt_x.ExchangeFaceNbrData();
         l2_x.ExchangeFaceNbrData();
         dgv_x.ExchangeFaceNbrData();
         dgi_x.ExchangeFaceNbrData();

         Vector      f_val(dim);      f_val = 0.0;

         Vector  h1_gfc_val(dim);  h1_gfc_val = 0.0;
         Vector  nd_gfc_val(dim);  nd_gfc_val = 0.0;
         Vector  rt_gfc_val(dim);  rt_gfc_val = 0.0;
         Vector  l2_gfc_val(dim);  l2_gfc_val = 0.0;
         Vector dgv_gfc_val(dim); dgv_gfc_val = 0.0;
         Vector dgi_gfc_val(dim); dgi_gfc_val = 0.0;

         Vector  h1_gvv_val(dim);  h1_gvv_val = 0.0;
         Vector  nd_gvv_val(dim);  nd_gvv_val = 0.0;
         Vector  rt_gvv_val(dim);  rt_gvv_val = 0.0;
         Vector  l2_gvv_val(dim);  l2_gvv_val = 0.0;
         Vector dgv_gvv_val(dim); dgv_gvv_val = 0.0;
         Vector dgi_gvv_val(dim); dgi_gvv_val = 0.0;

         SECTION("Shared Face Evaluation 2D")
         {
            for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++)
            {
               FaceElementTransformations *FET =
                  pmesh.GetSharedFaceTransformations(sf);
               ElementTransformation *T = &FET->GetElement2Transformation();
               int e = FET->Elem2No;
               int e_nbr = e - pmesh.GetNE();
               const FiniteElement   *fe = dgv_fespace.GetFaceNbrFE(e_nbr);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_gfc_err = 0.0;
               double  nd_gfc_err = 0.0;
               double  rt_gfc_err = 0.0;
               double  l2_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double  h1_gvv_err = 0.0;
               double  nd_gvv_err = 0.0;
               double  rt_gvv_err = 0.0;
               double  l2_gvv_err = 0.0;
               double dgv_gvv_err = 0.0;
               double dgi_gvv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);

                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  h1_x.GetVectorValue(e, ip, h1_gvv_val);
                  nd_x.GetVectorValue(e, ip, nd_gvv_val);
                  rt_x.GetVectorValue(e, ip, rt_gvv_val);
                  l2_x.GetVectorValue(e, ip, l2_gvv_val);
                  dgv_x.GetVectorValue(e, ip, dgv_gvv_val);
                  dgi_x.GetVectorValue(e, ip, dgi_gvv_val);

                  double  h1_gfc_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_gfc_dist = Distance(f_val,  nd_gfc_val);
                  double  rt_gfc_dist = Distance(f_val,  rt_gfc_val);
                  double  l2_gfc_dist = Distance(f_val,  l2_gfc_val);
                  double dgv_gfc_dist = Distance(f_val, dgv_gfc_val);
                  double dgi_gfc_dist = Distance(f_val, dgi_gfc_val);

                  double  h1_gvv_dist = Distance(f_val,  h1_gvv_val);
                  double  nd_gvv_dist = Distance(f_val,  nd_gvv_val);
                  double  rt_gvv_dist = Distance(f_val,  rt_gvv_val);
                  double  l2_gvv_dist = Distance(f_val,  l2_gvv_val);
                  double dgv_gvv_dist = Distance(f_val, dgv_gvv_val);
                  double dgi_gvv_dist = Distance(f_val, dgi_gvv_val);

                  h1_gfc_err  +=  h1_gfc_dist;
                  nd_gfc_err  +=  nd_gfc_dist;
                  rt_gfc_err  +=  rt_gfc_dist;
                  l2_gfc_err  +=  l2_gfc_dist;
                  dgv_gfc_err += dgv_gfc_dist;
                  dgi_gfc_err += dgi_gfc_dist;

                  h1_gvv_err  +=  h1_gvv_dist;
                  nd_gvv_err  +=  nd_gvv_dist;
                  rt_gvv_err  +=  rt_gvv_dist;
                  l2_gvv_err  +=  l2_gvv_dist;
                  dgv_gvv_err += dgv_gvv_dist;
                  dgi_gvv_err += dgi_gvv_dist;

                  if (verbose_tests && h1_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gfc_val[0] << ","
                               << h1_gfc_val[1] << ") "
                               << h1_gfc_dist << std::endl;
                  }
                  if (verbose_tests && nd_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << nd_gfc_val[0] << ","
                               << nd_gfc_val[1] << ") "
                               << nd_gfc_dist << std::endl;
                  }
                  if (verbose_tests && rt_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << rt_gfc_val[0] << ","
                               << rt_gfc_val[1] << ") "
                               << rt_gfc_dist << std::endl;
                  }
                  if (verbose_tests && l2_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << l2_gfc_val[0] << ","
                               << l2_gfc_val[1] << ") "
                               << l2_gfc_dist << std::endl;
                  }
                  if (verbose_tests && dgv_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ") "
                               << dgv_gfc_dist << std::endl;
                  }
                  if (verbose_tests && dgi_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgi_gfc_val[0] << ","
                               << dgi_gfc_val[1] << ") "
                               << dgi_gfc_dist << std::endl;
                  }
                  if (verbose_tests && h1_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gvv_val[0] << "," << h1_gvv_val[1] << ") "
                               << h1_gvv_dist << std::endl;
                  }
                  if (verbose_tests && nd_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gvv_val[0] << "," << nd_gvv_val[1] << ") "
                               << nd_gvv_dist << std::endl;
                  }
                  if (verbose_tests && rt_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gvv_val[0] << "," << rt_gvv_val[1] << ") "
                               << rt_gvv_dist << std::endl;
                  }
                  if (verbose_tests && l2_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " l2  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gvv_val[0] << "," << l2_gvv_val[1] << ") "
                               << l2_gvv_dist << std::endl;
                  }
                  if (verbose_tests && dgv_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gvv_val[0] << ","
                               << dgv_gvv_val[1] << ") "
                               << dgv_gvv_dist << std::endl;
                  }
                  if (verbose_tests && dgi_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gvv_val[0] << ","
                               << dgi_gvv_val[1] << ") "
                               << dgi_gvv_dist << std::endl;
                  }
               }

               h1_gfc_err  /= ir.GetNPoints();
               nd_gfc_err  /= ir.GetNPoints();
               rt_gfc_err  /= ir.GetNPoints();
               l2_gfc_err  /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gvv_err  /= ir.GetNPoints();
               nd_gvv_err  /= ir.GetNPoints();
               rt_gvv_err  /= ir.GetNPoints();
               l2_gvv_err  /= ir.GetNPoints();
               dgv_gvv_err /= ir.GetNPoints();
               dgi_gvv_err /= ir.GetNPoints();

               REQUIRE( h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE( nd_gfc_err == MFEM_Approx(0.0));
               REQUIRE( rt_gfc_err == MFEM_Approx(0.0));
               REQUIRE( l2_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE( h1_gvv_err == MFEM_Approx(0.0));
               REQUIRE( nd_gvv_err == MFEM_Approx(0.0));
               REQUIRE( rt_gvv_err == MFEM_Approx(0.0));
               REQUIRE( l2_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gvv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << my_rank << ": Checked GridFunction::GetVectorValue at "
             << npts << " 2D points" << std::endl;
}

#endif // MFEM_USE_MPI

TEST_CASE("3D GetVectorValue",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 3;
   int order = 1;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_3D_et; type <= last_3D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, Func_3D_lin);

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

         h1_x.ProjectCoefficient(funcCoef);
         nd_x.ProjectCoefficient(funcCoef);
         rt_x.ProjectCoefficient(funcCoef);
         l2_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         Vector      f_val(dim);      f_val = 0.0;

         Vector  h1_gfc_val(dim);  h1_gfc_val = 0.0;
         Vector  nd_gfc_val(dim);  nd_gfc_val = 0.0;
         Vector  rt_gfc_val(dim);  rt_gfc_val = 0.0;
         Vector  l2_gfc_val(dim);  l2_gfc_val = 0.0;
         Vector dgv_gfc_val(dim); dgv_gfc_val = 0.0;
         Vector dgi_gfc_val(dim); dgi_gfc_val = 0.0;

         Vector  h1_gvv_val(dim);  h1_gvv_val = 0.0;
         Vector  nd_gvv_val(dim);  nd_gvv_val = 0.0;
         Vector  rt_gvv_val(dim);  rt_gvv_val = 0.0;
         Vector  l2_gvv_val(dim);  l2_gvv_val = 0.0;
         Vector dgv_gvv_val(dim); dgv_gvv_val = 0.0;
         Vector dgi_gvv_val(dim); dgi_gvv_val = 0.0;

         Vector  nd_gvf_val(dim);  nd_gvf_val = 0.0;
         Vector  rt_gvf_val(dim);  rt_gvf_val = 0.0;
         DenseMatrix nd_gvf_vals;
         DenseMatrix rt_gvf_vals;
         DenseMatrix tr;

         SECTION("Domain Evaluation 3D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_gfc_err = 0.0;
               double  nd_gfc_err = 0.0;
               double  rt_gfc_err = 0.0;
               double  l2_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double  h1_gvv_err = 0.0;
               double  nd_gvv_err = 0.0;
               double  rt_gvv_err = 0.0;
               double  l2_gvv_err = 0.0;
               double dgv_gvv_err = 0.0;
               double dgi_gvv_err = 0.0;

               double  nd_gvf_err = 0.0;
               double  rt_gvf_err = 0.0;

               nd_x.GetVectorFieldValues(e, ir, nd_gvf_vals, tr);
               rt_x.GetVectorFieldValues(e, ir, rt_gvf_vals, tr);

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);

                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  h1_x.GetVectorValue(e, ip, h1_gvv_val);
                  nd_x.GetVectorValue(e, ip, nd_gvv_val);
                  rt_x.GetVectorValue(e, ip, rt_gvv_val);
                  l2_x.GetVectorValue(e, ip, l2_gvv_val);
                  dgv_x.GetVectorValue(e, ip, dgv_gvv_val);
                  dgi_x.GetVectorValue(e, ip, dgi_gvv_val);

                  nd_gvf_vals.GetRow(j, nd_gvf_val);
                  rt_gvf_vals.GetRow(j, rt_gvf_val);

                  double  h1_gfc_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_gfc_dist = Distance(f_val,  nd_gfc_val);
                  double  rt_gfc_dist = Distance(f_val,  rt_gfc_val);
                  double  l2_gfc_dist = Distance(f_val,  l2_gfc_val);
                  double dgv_gfc_dist = Distance(f_val, dgv_gfc_val);
                  double dgi_gfc_dist = Distance(f_val, dgi_gfc_val);

                  double  h1_gvv_dist = Distance(f_val,  h1_gvv_val);
                  double  nd_gvv_dist = Distance(f_val,  nd_gvv_val);
                  double  rt_gvv_dist = Distance(f_val,  rt_gvv_val);
                  double  l2_gvv_dist = Distance(f_val,  l2_gvv_val);
                  double dgv_gvv_dist = Distance(f_val, dgv_gvv_val);
                  double dgi_gvv_dist = Distance(f_val, dgi_gvv_val);

                  double  nd_gvf_dist = Distance(f_val,  nd_gvf_val);
                  double  rt_gvf_dist = Distance(f_val,  rt_gvf_val);

                  h1_gfc_err  +=  h1_gfc_dist;
                  nd_gfc_err  +=  nd_gfc_dist;
                  rt_gfc_err  +=  rt_gfc_dist;
                  l2_gfc_err  +=  l2_gfc_dist;
                  dgv_gfc_err += dgv_gfc_dist;
                  dgi_gfc_err += dgi_gfc_dist;

                  h1_gvv_err  +=  h1_gvv_dist;
                  nd_gvv_err  +=  nd_gvv_dist;
                  rt_gvv_err  +=  rt_gvv_dist;
                  l2_gvv_err  +=  l2_gvv_dist;
                  dgv_gvv_err += dgv_gvv_dist;
                  dgi_gvv_err += dgi_gvv_dist;

                  nd_gvf_err  +=  nd_gvf_dist;
                  rt_gvf_err  +=  rt_gvf_dist;

                  if (verbose_tests && h1_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") "
                               << h1_gfc_dist << std::endl;
                  }
                  if (verbose_tests && nd_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ","
                               << nd_gfc_val[2] << ") "
                               << nd_gfc_dist << std::endl;
                  }
                  if (verbose_tests && rt_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gfc_val[0] << "," << rt_gfc_val[1] << ","
                               << rt_gfc_val[2] << ") "
                               << rt_gfc_dist << std::endl;
                  }
                  if (verbose_tests && l2_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " l2  gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gfc_val[0] << "," << l2_gfc_val[1] << ","
                               << l2_gfc_val[2] << ") "
                               << l2_gfc_dist << std::endl;
                  }
                  if (verbose_tests && dgv_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") "
                               << dgv_gfc_dist << std::endl;
                  }
                  if (verbose_tests && dgi_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gfc_val[0] << ","
                               << dgi_gfc_val[1] << ","
                               << dgi_gfc_val[2] << ") "
                               << dgi_gfc_dist << std::endl;
                  }
                  if (verbose_tests && h1_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gvv_val[0] << "," << h1_gvv_val[1] << ","
                               << h1_gvv_val[2] << ") "
                               << h1_gvv_dist << std::endl;
                  }
                  if (verbose_tests && nd_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gvv_val[0] << "," << nd_gvv_val[1] << ","
                               << nd_gvv_val[2] << ") "
                               << nd_gvv_dist << std::endl;
                  }
                  if (verbose_tests && rt_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gvv_val[0] << "," << rt_gvv_val[1] << ","
                               << rt_gvv_val[2] << ") "
                               << rt_gvv_dist << std::endl;
                  }
                  if (verbose_tests && l2_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " l2  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gvv_val[0] << "," << l2_gvv_val[1] << ","
                               << l2_gvv_val[2] << ") "
                               << l2_gvv_dist << std::endl;
                  }
                  if (verbose_tests && dgv_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gvv_val[0] << ","
                               << dgv_gvv_val[1] << ","
                               << dgv_gvv_val[2] << ") "
                               << dgv_gvv_dist << std::endl;
                  }
                  if (verbose_tests && dgi_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gvv_val[0] << ","
                               << dgi_gvv_val[1] << ","
                               << dgi_gvv_val[2] << ") "
                               << dgi_gvv_dist << std::endl;
                  }
                  if (verbose_tests && nd_gvf_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  gvf ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gvf_val[0] << ","
                               << nd_gvf_val[1] << ","
                               << nd_gvf_val[2] << ") "
                               << nd_gvf_dist << std::endl;
                  }
                  if (verbose_tests && rt_gvf_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  gvf ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gvf_val[0] << ","
                               << rt_gvf_val[1] << ","
                               << rt_gvf_val[2] << ") "
                               << rt_gvf_dist << std::endl;
                  }
               }

               h1_gfc_err  /= ir.GetNPoints();
               nd_gfc_err  /= ir.GetNPoints();
               rt_gfc_err  /= ir.GetNPoints();
               l2_gfc_err  /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gvv_err  /= ir.GetNPoints();
               nd_gvv_err  /= ir.GetNPoints();
               rt_gvv_err  /= ir.GetNPoints();
               l2_gvv_err  /= ir.GetNPoints();
               dgv_gvv_err /= ir.GetNPoints();
               dgi_gvv_err /= ir.GetNPoints();

               nd_gvf_err  /= ir.GetNPoints();
               rt_gvf_err  /= ir.GetNPoints();

               REQUIRE( h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE( nd_gfc_err == MFEM_Approx(0.0));
               REQUIRE( rt_gfc_err == MFEM_Approx(0.0));
               REQUIRE( l2_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE( h1_gvv_err == MFEM_Approx(0.0));
               REQUIRE( nd_gvv_err == MFEM_Approx(0.0));
               REQUIRE( rt_gvv_err == MFEM_Approx(0.0));
               REQUIRE( l2_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gvv_err == MFEM_Approx(0.0));

               REQUIRE( nd_gvf_err == MFEM_Approx(0.0));
               REQUIRE( rt_gvf_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (H1 Context)")
         {
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

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double  rt_dist = Distance(f_val,  rt_gfc_val);
                  double  l2_dist = Distance(f_val,  l2_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);
                  double dgi_dist = Distance(f_val, dgi_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") " << h1_dist
                               << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ","
                               << nd_gfc_val[2] << ") " << nd_dist
                               << std::endl;
                  }
                  if (verbose_tests && rt_dist > tol)
                  {
                     mfem::out << be << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gfc_val[0] << "," << rt_gfc_val[1] << ","
                               << rt_gfc_val[2] << ") " << rt_dist
                               << std::endl;
                  }
                  if (verbose_tests && l2_dist > tol)
                  {
                     mfem::out << be << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gfc_val[0] << "," << l2_gfc_val[1] << ","
                               << l2_gfc_val[2] << ") " << l2_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << "," << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgi_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gfc_val[0] << "," << dgi_gfc_val[1] << ","
                               << dgi_gfc_val[2] << ") " << dgi_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE( rt_err == MFEM_Approx(0.0));
               REQUIRE( l2_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (DG Context)")
         {
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

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double  rt_dist = Distance(f_val,  rt_gfc_val);
                  double  l2_dist = Distance(f_val,  l2_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);
                  double dgi_dist = Distance(f_val, dgi_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  rt_err  +=  rt_dist;
                  l2_err  +=  l2_dist;
                  dgv_err += dgv_dist;
                  dgi_err += dgi_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") " << h1_dist
                               << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ","
                               << nd_gfc_val[2] << ") " << nd_dist
                               << std::endl;
                  }
                  if (verbose_tests && rt_dist > tol)
                  {
                     mfem::out << be << ":" << j << " rt  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gfc_val[0] << "," << rt_gfc_val[1] << ","
                               << rt_gfc_val[2] << ") " << rt_dist
                               << std::endl;
                  }
                  if (verbose_tests && l2_dist > tol)
                  {
                     mfem::out << be << ":" << j << " l2  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gfc_val[0] << "," << l2_gfc_val[1] << ","
                               << l2_gfc_val[2] << ") " << l2_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << "," << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgi_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgi ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gfc_val[0] << "," << dgi_gfc_val[1] << ","
                               << dgi_gfc_val[2] << ") " << dgi_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               rt_err  /= ir.GetNPoints();
               l2_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();
               dgi_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE( rt_err == MFEM_Approx(0.0));
               REQUIRE( l2_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Edge Evaluation 3D")
         {
            for (int e = 0; e < mesh.GetNEdges(); e++)
            {
               ElementTransformation *T = mesh.GetEdgeTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetEdgeElement(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);

                  h1_err  +=  h1_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") " << h1_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Face Evaluation 3D")
         {
            for (int f = 0; f < mesh.GetNFaces(); f++)
            {
               ElementTransformation *T = mesh.GetFaceTransformation(f);
               const FiniteElement   *fe = h1_fespace.GetFaceElement(f);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  funcCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);

                  h1_err  +=  h1_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << f << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") " << h1_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetVectorValue at "
             << npts << " 3D points" << std::endl;
}

#ifdef MFEM_USE_MPI

TEST_CASE("3D GetVectorValue in Parallel",
          "[ParGridFunction]"
          "[VectorGridFunctionCoefficient]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int n = (int)ceil(pow(2*num_procs, 1.0 / 3.0));
   int dim = 3;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_3D_et; type <= last_3D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);
      ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      VectorFunctionCoefficient funcCoef(dim, Func_3D_lin);

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

         ParFiniteElementSpace  h1_fespace(&pmesh,  &h1_fec, dim);
         ParFiniteElementSpace  nd_fespace(&pmesh,  &nd_fec);
         ParFiniteElementSpace  rt_fespace(&pmesh,  &rt_fec);
         ParFiniteElementSpace  l2_fespace(&pmesh,  &l2_fec, dim);
         ParFiniteElementSpace dgv_fespace(&pmesh, &dgv_fec, dim);
         ParFiniteElementSpace dgi_fespace(&pmesh, &dgi_fec, dim);

         ParGridFunction  h1_x( &h1_fespace);
         ParGridFunction  nd_x( &nd_fespace);
         ParGridFunction  rt_x( &rt_fespace);
         ParGridFunction  l2_x( &l2_fespace);
         ParGridFunction dgv_x(&dgv_fespace);
         ParGridFunction dgi_x(&dgi_fespace);

         VectorGridFunctionCoefficient  h1_xCoef( &h1_x);
         VectorGridFunctionCoefficient  nd_xCoef( &nd_x);
         VectorGridFunctionCoefficient  rt_xCoef( &rt_x);
         VectorGridFunctionCoefficient  l2_xCoef( &l2_x);
         VectorGridFunctionCoefficient dgv_xCoef(&dgv_x);
         VectorGridFunctionCoefficient dgi_xCoef(&dgi_x);

         h1_x.ProjectCoefficient(funcCoef);
         nd_x.ProjectCoefficient(funcCoef);
         rt_x.ProjectCoefficient(funcCoef);
         l2_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);
         dgi_x.ProjectCoefficient(funcCoef);

         h1_x.ExchangeFaceNbrData();
         nd_x.ExchangeFaceNbrData();
         rt_x.ExchangeFaceNbrData();
         l2_x.ExchangeFaceNbrData();
         dgv_x.ExchangeFaceNbrData();
         dgi_x.ExchangeFaceNbrData();

         Vector           x(dim);           x = 0.0;
         Vector       f_val(dim);       f_val = 0.0;

         Vector  h1_gfc_val(dim);  h1_gfc_val = 0.0;
         Vector  nd_gfc_val(dim);  nd_gfc_val = 0.0;
         Vector  rt_gfc_val(dim);  rt_gfc_val = 0.0;
         Vector  l2_gfc_val(dim);  l2_gfc_val = 0.0;
         Vector dgv_gfc_val(dim); dgv_gfc_val = 0.0;
         Vector dgi_gfc_val(dim); dgi_gfc_val = 0.0;

         Vector  h1_gvv_val(dim);  h1_gvv_val = 0.0;
         Vector  nd_gvv_val(dim);  nd_gvv_val = 0.0;
         Vector  rt_gvv_val(dim);  rt_gvv_val = 0.0;
         Vector  l2_gvv_val(dim);  l2_gvv_val = 0.0;
         Vector dgv_gvv_val(dim); dgv_gvv_val = 0.0;
         Vector dgi_gvv_val(dim); dgi_gvv_val = 0.0;

         SECTION("Shared Face Evaluation 3D")
         {
            for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++)
            {
               FaceElementTransformations *FET =
                  pmesh.GetSharedFaceTransformations(sf);
               ElementTransformation *T = &FET->GetElement2Transformation();
               int e = FET->Elem2No;
               int e_nbr = e - pmesh.GetNE();
               const FiniteElement   *fe = dgv_fespace.GetFaceNbrFE(e_nbr);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_gfc_err = 0.0;
               double  nd_gfc_err = 0.0;
               double  rt_gfc_err = 0.0;
               double  l2_gfc_err = 0.0;
               double dgv_gfc_err = 0.0;
               double dgi_gfc_err = 0.0;

               double  h1_gvv_err = 0.0;
               double  nd_gvv_err = 0.0;
               double  rt_gvv_err = 0.0;
               double  l2_gvv_err = 0.0;
               double dgv_gvv_err = 0.0;
               double dgi_gvv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);
                  T->Transform(ip, x);

                  funcCoef.Eval(f_val, *T, ip);

                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  rt_xCoef.Eval(rt_gfc_val, *T, ip);
                  l2_xCoef.Eval(l2_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);
                  dgi_xCoef.Eval(dgi_gfc_val, *T, ip);

                  h1_x.GetVectorValue(e, ip, h1_gvv_val);
                  nd_x.GetVectorValue(e, ip, nd_gvv_val);
                  rt_x.GetVectorValue(e, ip, rt_gvv_val);
                  l2_x.GetVectorValue(e, ip, l2_gvv_val);
                  dgv_x.GetVectorValue(e, ip, dgv_gvv_val);
                  dgi_x.GetVectorValue(e, ip, dgi_gvv_val);

                  double  h1_gfc_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_gfc_dist = Distance(f_val,  nd_gfc_val);
                  double  rt_gfc_dist = Distance(f_val,  rt_gfc_val);
                  double  l2_gfc_dist = Distance(f_val,  l2_gfc_val);
                  double dgv_gfc_dist = Distance(f_val, dgv_gfc_val);
                  double dgi_gfc_dist = Distance(f_val, dgi_gfc_val);

                  double  h1_gvv_dist = Distance(f_val,  h1_gvv_val);
                  double  nd_gvv_dist = Distance(f_val,  nd_gvv_val);
                  double  rt_gvv_dist = Distance(f_val,  rt_gvv_val);
                  double  l2_gvv_dist = Distance(f_val,  l2_gvv_val);
                  double dgv_gvv_dist = Distance(f_val, dgv_gvv_val);
                  double dgi_gvv_dist = Distance(f_val, dgi_gvv_val);

                  h1_gfc_err  +=  h1_gfc_dist;
                  nd_gfc_err  +=  nd_gfc_dist;
                  rt_gfc_err  +=  rt_gfc_dist;
                  l2_gfc_err  +=  l2_gfc_dist;
                  dgv_gfc_err += dgv_gfc_dist;
                  dgi_gfc_err += dgi_gfc_dist;

                  h1_gvv_err  +=  h1_gvv_dist;
                  nd_gvv_err  +=  nd_gvv_dist;
                  rt_gvv_err  +=  rt_gvv_dist;
                  l2_gvv_err  +=  l2_gvv_dist;
                  dgv_gvv_err += dgv_gvv_dist;
                  dgi_gvv_err += dgi_gvv_dist;

                  if (verbose_tests && h1_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") "
                               << h1_gfc_dist << std::endl;
                  }
                  if (verbose_tests && nd_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j
                               << " x = (" << x[0] << "," << x[1] << ","
                               << x[2] << ")\n nd  gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ")\n vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ","
                               << nd_gfc_val[2] << ") "
                               << nd_gfc_dist << std::endl;
                  }
                  if (verbose_tests && rt_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gfc_val[0] << "," << rt_gfc_val[1] << ","
                               << rt_gfc_val[2] << ") "
                               << rt_gfc_dist << std::endl;
                  }
                  if (verbose_tests && l2_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " l2  gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gfc_val[0] << "," << l2_gfc_val[1] << ","
                               << l2_gfc_val[2] << ") "
                               << l2_gfc_dist << std::endl;
                  }
                  if (verbose_tests && dgv_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") "
                               << dgv_gfc_dist << std::endl;
                  }
                  if (verbose_tests && dgi_gfc_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gfc ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gfc_val[0] << ","
                               << dgi_gfc_val[1] << ","
                               << dgi_gfc_val[2] << ") "
                               << dgi_gfc_dist << std::endl;
                  }
                  if (verbose_tests && h1_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gvv_val[0] << "," << h1_gvv_val[1] << ","
                               << h1_gvv_val[2] << ") "
                               << h1_gvv_dist << std::endl;
                  }
                  if (verbose_tests && nd_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gvv_val[0] << "," << nd_gvv_val[1] << ","
                               << nd_gvv_val[2] << ") "
                               << nd_gvv_dist << std::endl;
                  }
                  if (verbose_tests && rt_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " rt  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << rt_gvv_val[0] << "," << rt_gvv_val[1] << ","
                               << rt_gvv_val[2] << ") "
                               << rt_gvv_dist << std::endl;
                  }
                  if (verbose_tests && l2_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " l2  gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << l2_gvv_val[0] << "," << l2_gvv_val[1] << ","
                               << l2_gvv_val[2] << ") "
                               << l2_gvv_dist << std::endl;
                  }
                  if (verbose_tests && dgv_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gvv_val[0] << ","
                               << dgv_gvv_val[1] << ","
                               << dgv_gvv_val[2] << ") "
                               << dgv_gvv_dist << std::endl;
                  }
                  if (verbose_tests && dgi_gvv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgi gvv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgi_gvv_val[0] << ","
                               << dgi_gvv_val[1] << ","
                               << dgi_gvv_val[2] << ") "
                               << dgi_gvv_dist << std::endl;
                  }
               }

               h1_gfc_err  /= ir.GetNPoints();
               nd_gfc_err  /= ir.GetNPoints();
               rt_gfc_err  /= ir.GetNPoints();
               l2_gfc_err  /= ir.GetNPoints();
               dgv_gfc_err /= ir.GetNPoints();
               dgi_gfc_err /= ir.GetNPoints();

               h1_gvv_err  /= ir.GetNPoints();
               nd_gvv_err  /= ir.GetNPoints();
               rt_gvv_err  /= ir.GetNPoints();
               l2_gvv_err  /= ir.GetNPoints();
               dgv_gvv_err /= ir.GetNPoints();
               dgi_gvv_err /= ir.GetNPoints();

               REQUIRE( h1_gfc_err == MFEM_Approx(0.0));
               REQUIRE( nd_gfc_err == MFEM_Approx(0.0));
               REQUIRE( rt_gfc_err == MFEM_Approx(0.0));
               REQUIRE( l2_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gfc_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gfc_err == MFEM_Approx(0.0));

               REQUIRE( h1_gvv_err == MFEM_Approx(0.0));
               REQUIRE( nd_gvv_err == MFEM_Approx(0.0));
               REQUIRE( rt_gvv_err == MFEM_Approx(0.0));
               REQUIRE( l2_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgv_gvv_err == MFEM_Approx(0.0));
               REQUIRE(dgi_gvv_err == MFEM_Approx(0.0));
            }
         }
      }
   }

   mfem::out << my_rank << ": Checked GridFunction::GetVectorValue at "
             << npts << " 3D points" << std::endl;
}

#endif // MFEM_USE_MPI

TEST_CASE("1D GetGradient",
          "[GridFunction]"
          "[GradientGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 1;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_1D_et; type <= last_1D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      FunctionCoefficient funcCoef(func_1D_quad);
      VectorFunctionCoefficient dFuncCoef(dim, dfunc_1D_quad);

      SECTION("1D GetGradient tests for element type " + std::to_string(type))
      {
         H1_FECollection h1_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);

         FiniteElementSpace h1_fespace(&mesh, &h1_fec);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec);

         GridFunction h1_x(&h1_fespace);
         GridFunction dgv_x(&dgv_fespace);

         GradientGridFunctionCoefficient h1_xCoef(&h1_x);
         GradientGridFunctionCoefficient dgv_xCoef(&dgv_x);

         h1_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);

         Vector      f_val(dim);      f_val = 0.0;
         Vector  h1_gfc_val(dim);  h1_gfc_val = 0.0;
         Vector dgv_gfc_val(dim); dgv_gfc_val = 0.0;

         SECTION("Domain Evaluation 1D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  ("
                               << f_val[0] << ") vs. ("
                               << h1_gfc_val[0] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv ("
                               << f_val[0] << ") vs. ("
                               << dgv_gfc_val[0] << ") "
                               << dgv_dist << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 1D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << ") vs. ("
                               << h1_gfc_val[0] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << ") vs. ("
                               << dgv_gfc_val[0] << ") "
                               << dgv_dist << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 1D (DG Context)")
         {
            mfem::out << "Boundary Evaluation 1D (DG Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << ") vs. ("
                               << h1_gfc_val[0] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << ") vs. ("
                               << dgv_gfc_val[0] << ") "
                               << dgv_dist << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetGradient at "
             << npts << " 1D points" << std::endl;
}

TEST_CASE("2D GetGradient",
          "[GridFunction]"
          "[GradientGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 2;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_2D_et; type <= last_2D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(n, n, (Element::Type)type, 1, 2.0, 3.0);

      FunctionCoefficient funcCoef(func_2D_quad);
      VectorFunctionCoefficient dFuncCoef(dim, dfunc_2D_quad);

      SECTION("2D GetGradient tests for element type " + std::to_string(type))
      {
         H1_FECollection h1_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);

         FiniteElementSpace h1_fespace(&mesh, &h1_fec);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec);

         GridFunction h1_x(&h1_fespace);
         GridFunction dgv_x(&dgv_fespace);

         GradientGridFunctionCoefficient h1_xCoef(&h1_x);
         GradientGridFunctionCoefficient dgv_xCoef(&dgv_x);

         h1_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);

         Vector      f_val(dim);      f_val = 0.0;
         Vector  h1_gfc_val(dim);  h1_gfc_val = 0.0;
         Vector dgv_gfc_val(dim); dgv_gfc_val = 0.0;

         SECTION("Domain Evaluation 2D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gfc_val[0] << ","
                               << h1_gfc_val[1] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ") "
                               << dgv_dist << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (H1 Context)")
         {
            mfem::out << "Boundary Evaluation 2D (H1 Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gfc_val[0] << ","
                               << h1_gfc_val[1] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ") "
                               << dgv_dist << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (DG Context)")
         {
            mfem::out << "Boundary Evaluation 2D (DG Context)" << std::endl;
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << h1_gfc_val[0] << ","
                               << h1_gfc_val[1] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ") vs. ("
                               << dgv_gfc_val[0] << ","
                               << dgv_gfc_val[1] << ") "
                               << dgv_dist << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetGradient at "
             << npts << " 2D points" << std::endl;
}

TEST_CASE("3D GetGradient",
          "[GridFunction]"
          "[GradientGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 3;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_3D_et; type <= last_3D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      FunctionCoefficient funcCoef(func_3D_quad);
      VectorFunctionCoefficient dFuncCoef(dim, dfunc_3D_quad);

      SECTION("3D GetGradient tests for element type " + std::to_string(type))
      {
         H1_FECollection h1_fec(order, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);

         FiniteElementSpace h1_fespace(&mesh, &h1_fec);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec);

         GridFunction h1_x(&h1_fespace);
         GridFunction dgv_x(&dgv_fespace);

         GradientGridFunctionCoefficient h1_xCoef(&h1_x);
         GradientGridFunctionCoefficient dgv_xCoef(&dgv_x);

         h1_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);

         Vector      f_val(dim);      f_val = 0.0;
         Vector  h1_gfc_val(dim);  h1_gfc_val = 0.0;
         Vector dgv_gfc_val(dim); dgv_gfc_val = 0.0;

         SECTION("Domain Evaluation 3D")
         {
            mfem::out << "Domain Evaluation 3D for element type "
                      << std::to_string(type) << std::endl;
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << "," << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << "," << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (DG Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") "
                               << h1_dist << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << "," << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetGradient at "
             << npts << " 3D points" << std::endl;
}

TEST_CASE("2D GetCurl",
          "[GridFunction]"
          "[CurlGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 2;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_2D_et; type <= last_2D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(n, n, (Element::Type)type, 1, 2.0, 3.0);

      VectorFunctionCoefficient funcCoef(dim, Func_2D_quad);
      VectorFunctionCoefficient dFuncCoef(1, RotFunc_2D_quad);

      SECTION("2D GetCurl tests for element type " +
              std::to_string(type))
      {
         H1_FECollection  h1_fec(order, dim);
         ND_FECollection  nd_fec(order+1, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);

         FiniteElementSpace  h1_fespace(&mesh,  &h1_fec, dim);
         FiniteElementSpace  nd_fespace(&mesh,  &nd_fec);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec, dim);

         GridFunction  h1_x( &h1_fespace);
         GridFunction  nd_x( &nd_fespace);
         GridFunction dgv_x(&dgv_fespace);

         CurlGridFunctionCoefficient  h1_xCoef( &h1_x);
         CurlGridFunctionCoefficient  nd_xCoef( &nd_x);
         CurlGridFunctionCoefficient dgv_xCoef(&dgv_x);

         h1_x.ProjectCoefficient(funcCoef);
         nd_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);

         Vector      f_val(2*dim-3);      f_val = 0.0;
         Vector  h1_gfc_val(2*dim-3);  h1_gfc_val = 0.0;
         Vector  nd_gfc_val(2*dim-3);  nd_gfc_val = 0.0;
         Vector dgv_gfc_val(2*dim-3); dgv_gfc_val = 0.0;

         SECTION("Domain Evaluation 2D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  ("
                               << f_val[0] << ") vs. ("
                               << h1_gfc_val[0] << ") " << h1_dist
                               << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  ("
                               << f_val[0] << ") vs. ("
                               << nd_gfc_val[0] << ") " << nd_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv ("
                               << f_val[0] << ") vs. ("
                               << dgv_gfc_val[0] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << ") vs. ("
                               << h1_gfc_val[0] << ") " << h1_dist
                               << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd  ("
                               << f_val[0] << ") vs. ("
                               << nd_gfc_val[0] << ") " << nd_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << ") vs. ("
                               << dgv_gfc_val[0] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (DG Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << ") vs. ("
                               << h1_gfc_val[0] << ") " << h1_dist
                               << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd  ("
                               << f_val[0] << ") vs. ("
                               << nd_gfc_val[0] << ") " << nd_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << ") vs. ("
                               << dgv_gfc_val[0] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetCurl at "
             << npts << " 2D points" << std::endl;
}

TEST_CASE("3D GetCurl",
          "[GridFunction]"
          "[CurlGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 3;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_3D_et; type <= last_3D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, Func_3D_quad);
      VectorFunctionCoefficient dFuncCoef(dim, CurlFunc_3D_quad);

      SECTION("3D GetCurl tests for element type " +
              std::to_string(type))
      {
         H1_FECollection  h1_fec(order, dim);
         ND_FECollection  nd_fec(order+1, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);

         FiniteElementSpace  h1_fespace(&mesh,  &h1_fec, dim);
         FiniteElementSpace  nd_fespace(&mesh,  &nd_fec);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec, dim);

         GridFunction  h1_x( &h1_fespace);
         GridFunction  nd_x( &nd_fespace);
         GridFunction dgv_x(&dgv_fespace);

         CurlGridFunctionCoefficient  h1_xCoef( &h1_x);
         CurlGridFunctionCoefficient  nd_xCoef( &nd_x);
         CurlGridFunctionCoefficient dgv_xCoef(&dgv_x);

         h1_x.ProjectCoefficient(funcCoef);
         nd_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);

         Vector      f_val(2*dim-3);      f_val = 0.0;
         Vector  h1_gfc_val(2*dim-3);  h1_gfc_val = 0.0;
         Vector  nd_gfc_val(2*dim-3);  nd_gfc_val = 0.0;
         Vector dgv_gfc_val(2*dim-3); dgv_gfc_val = 0.0;

         SECTION("Domain Evaluation 3D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << e << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") " << h1_dist
                               << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << e << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ","
                               << nd_gfc_val[2] << ") " << nd_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << e << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << "," << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") " << h1_dist
                               << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ","
                               << nd_gfc_val[2] << ") " << nd_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << "," << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (DG Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  nd_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  dFuncCoef.Eval(f_val, *T, ip);
                  h1_xCoef.Eval(h1_gfc_val, *T, ip);
                  nd_xCoef.Eval(nd_gfc_val, *T, ip);
                  dgv_xCoef.Eval(dgv_gfc_val, *T, ip);

                  double  h1_dist = Distance(f_val,  h1_gfc_val);
                  double  nd_dist = Distance(f_val,  nd_gfc_val);
                  double dgv_dist = Distance(f_val, dgv_gfc_val);

                  h1_err  +=  h1_dist;
                  nd_err  +=  nd_dist;
                  dgv_err += dgv_dist;

                  if (verbose_tests && h1_dist > tol)
                  {
                     mfem::out << be << ":" << j << " h1  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << h1_gfc_val[0] << "," << h1_gfc_val[1] << ","
                               << h1_gfc_val[2] << ") " << h1_dist
                               << std::endl;
                  }
                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd  ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << nd_gfc_val[0] << "," << nd_gfc_val[1] << ","
                               << nd_gfc_val[2] << ") " << nd_dist
                               << std::endl;
                  }
                  if (verbose_tests && dgv_dist > tol)
                  {
                     mfem::out << be << ":" << j << " dgv ("
                               << f_val[0] << "," << f_val[1] << ","
                               << f_val[2] << ") vs. ("
                               << dgv_gfc_val[0] << "," << dgv_gfc_val[1] << ","
                               << dgv_gfc_val[2] << ") " << dgv_dist
                               << std::endl;
                  }
               }
               h1_err  /= ir.GetNPoints();
               nd_err  /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE( h1_err == MFEM_Approx(0.0));
               REQUIRE( nd_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetCurl at "
             << npts << " 3D points" << std::endl;
}

TEST_CASE("2D GetDivergence",
          "[GridFunction]"
          "[DivergenceGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 2;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_2D_et; type <= last_2D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(n, n, (Element::Type)type, 1, 2.0, 3.0);

      VectorFunctionCoefficient funcCoef(dim, Func_2D_quad);
      FunctionCoefficient dFuncCoef(DivFunc_2D_quad);

      SECTION("2D GetValue tests for element type " + std::to_string(type))
      {
         H1_FECollection  h1_fec(order, dim);
         RT_FECollection  rt_fec(order+1, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);

         FiniteElementSpace  h1_fespace(&mesh,  &h1_fec, dim);
         FiniteElementSpace  rt_fespace(&mesh,  &rt_fec);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec, dim);

         GridFunction h1_x(&h1_fespace);
         GridFunction rt_x(&rt_fespace);
         GridFunction dgv_x(&dgv_fespace);

         DivergenceGridFunctionCoefficient h1_xCoef(&h1_x);
         DivergenceGridFunctionCoefficient rt_xCoef(&rt_x);
         DivergenceGridFunctionCoefficient dgv_xCoef(&dgv_x);

         h1_x.ProjectCoefficient(funcCoef);
         rt_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);

         SECTION("Domain Evaluation 2D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double  h1_err = 0.0;
               double  rt_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double      f_val = dFuncCoef.Eval(*T, ip);
                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double  rt_gfc_val =  rt_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  rt_err += fabs(f_val - rt_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - rt_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " rt  " << f_val << " "
                               << rt_gfc_val << " "
                               << fabs(f_val - rt_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               rt_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(rt_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double rt_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double      f_val = dFuncCoef.Eval(*T, ip);
                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double  rt_gfc_val =  rt_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  rt_err += fabs(f_val - rt_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - rt_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " rt  " << f_val << " "
                               << rt_gfc_val << " "
                               << fabs(f_val - rt_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               rt_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(rt_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 2D (DG Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double rt_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  double      f_val = dFuncCoef.Eval(*T, ip);
                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double  rt_gfc_val =  rt_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  rt_err += fabs(f_val - rt_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - rt_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " rt  " << f_val << " "
                               << rt_gfc_val << " "
                               << fabs(f_val - rt_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               rt_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(rt_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetDivergence at "
             << npts << " 2D points" << std::endl;
}

TEST_CASE("3D GetDivergence",
          "[GridFunction]"
          "[DivergenceGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 3;
   int order = 2;
   int npts = 0;

   double tol = 1e-6;

   for (int type = first_3D_et; type <= last_3D_et; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, Func_3D_quad);
      FunctionCoefficient dFuncCoef(DivFunc_3D_quad);

      SECTION("3D GetValue tests for element type " + std::to_string(type))
      {
         H1_FECollection h1_fec(order, dim);
         RT_FECollection rt_fec(order+1, dim);
         DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                                 FiniteElement::VALUE);

         FiniteElementSpace h1_fespace(&mesh, &h1_fec, dim);
         FiniteElementSpace rt_fespace(&mesh, &rt_fec);
         FiniteElementSpace dgv_fespace(&mesh, &dgv_fec, dim);

         GridFunction h1_x(&h1_fespace);
         GridFunction rt_x(&rt_fespace);
         GridFunction dgv_x(&dgv_fespace);

         DivergenceGridFunctionCoefficient h1_xCoef(&h1_x);
         DivergenceGridFunctionCoefficient rt_xCoef(&rt_x);
         DivergenceGridFunctionCoefficient dgv_xCoef(&dgv_x);

         h1_x.ProjectCoefficient(funcCoef);
         rt_x.ProjectCoefficient(funcCoef);
         dgv_x.ProjectCoefficient(funcCoef);

         SECTION("Domain Evaluation 3D")
         {
            for (int e = 0; e < mesh.GetNE(); e++)
            {
               ElementTransformation *T = mesh.GetElementTransformation(e);
               const FiniteElement   *fe = h1_fespace.GetFE(e);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double rt_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double      f_val = dFuncCoef.Eval(*T, ip);
                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double  rt_gfc_val =  rt_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  rt_err += fabs(f_val - rt_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - rt_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " rt  " << f_val << " "
                               << rt_gfc_val << " "
                               << fabs(f_val - rt_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << e << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               rt_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(rt_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (H1 Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = h1_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double rt_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  double      f_val = dFuncCoef.Eval(*T, ip);
                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double  rt_gfc_val =  rt_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  rt_err += fabs(f_val - rt_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - rt_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " rt  " << f_val << " "
                               << rt_gfc_val << " "
                               << fabs(f_val - rt_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               rt_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(rt_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }

         SECTION("Boundary Evaluation 3D (DG Context)")
         {
            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               FaceElementTransformations *T =
                  mesh.GetBdrFaceTransformations(be);
               const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                        2*order + 2);

               double h1_err = 0.0;
               double rt_err = 0.0;
               double dgv_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  npts++;
                  const IntegrationPoint &ip = ir.IntPoint(j);

                  T->SetIntPoint(&ip);

                  double      f_val = dFuncCoef.Eval(*T, ip);
                  double  h1_gfc_val =  h1_xCoef.Eval(*T, ip);
                  double  rt_gfc_val =  rt_xCoef.Eval(*T, ip);
                  double dgv_gfc_val = dgv_xCoef.Eval(*T, ip);

                  h1_err += fabs(f_val - h1_gfc_val);
                  rt_err += fabs(f_val - rt_gfc_val);
                  dgv_err += fabs(f_val - dgv_gfc_val);

                  if (verbose_tests && fabs(f_val - h1_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " h1  " << f_val << " "
                               << h1_gfc_val << " "
                               << fabs(f_val - h1_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - rt_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " rt  " << f_val << " "
                               << rt_gfc_val << " "
                               << fabs(f_val - rt_gfc_val)
                               << std::endl;
                  }
                  if (verbose_tests && fabs(f_val - dgv_gfc_val) > tol)
                  {
                     mfem::out << be << ":" << j << " dgv " << f_val << " "
                               << dgv_gfc_val << " "
                               << fabs(f_val - dgv_gfc_val)
                               << std::endl;
                  }
               }
               h1_err /= ir.GetNPoints();
               rt_err /= ir.GetNPoints();
               dgv_err /= ir.GetNPoints();

               REQUIRE(h1_err == MFEM_Approx(0.0));
               REQUIRE(rt_err == MFEM_Approx(0.0));
               REQUIRE(dgv_err == MFEM_Approx(0.0));
            }
         }
      }
   }
   mfem::out << "Checked GridFunction::GetDivergence at "
             << npts << " 3D points" << std::endl;
}

} // namespace get_value
