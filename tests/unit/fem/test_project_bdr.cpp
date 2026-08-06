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

namespace project_bdr
{

void Func_lin(const Vector &x, Vector &v)
{
   const int dim = x.Size();
   v.SetSize(dim);
   v[0] = 1.234 * x[0] - 2.357 * x[1];
   v[1] = 2.537 * x[0] + 4.321 * x[1];
   if (dim == 3)
   {
      v[0] += 3.572 * x[2];
      v[1] -= 1.234 * x[2];
      v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
   }
}

TEST_CASE("3D ProjectBdrCoefficientNormal Vector",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   const int n = 1;
   const int dim = 3;
   const int order = 1;

   const double tol = 1e-6;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, Func_lin);

      SECTION("3D GetVectorValue tests for element type " +
              std::to_string(type))
      {
         RT_FECollection  rt_fec(order+1, dim);

         FiniteElementSpace  rt_fespace(&mesh,  &rt_fec);

         GridFunction  rt_x( &rt_fespace);

         VectorGridFunctionCoefficient  rt_xCoef( &rt_x);

         Array<int> bdr_marker(6);

         Vector normal(dim);
         Vector  f_val(dim);
         Vector rt_val(dim);

         for (int b = 1; b<=6; b++)
         {
            bdr_marker = 0;
            bdr_marker[b-1] = 1;

            rt_x = 0.0;
            rt_x.ProjectBdrCoefficientNormal(funcCoef, bdr_marker);

            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               Element *e = mesh.GetBdrElement(be);
               if (e->GetAttribute() != b) { continue; }

               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = rt_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double rt_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  CalcOrtho(T->Jacobian(), normal);

                  funcCoef.Eval(f_val, *T, ip);
                  rt_xCoef.Eval(rt_val, *T, ip);

                  rt_val -= f_val;

                  double rt_dist = rt_val * normal;

                  rt_err += rt_dist;

                  if (verbose_tests && rt_dist > tol)
                  {
                     mfem::out << be << ":" << j << " rt ("
                               << f_val[0] << "," << f_val[1] << "," << f_val[2]
                               << ") vs. ("
                               << rt_val[0] << "," << rt_val[1] << ","
                               << rt_val[2] << ") " << rt_dist << std::endl;
                  }
               }
               rt_err  /= ir.GetNPoints();

               REQUIRE( rt_err == MFEM_Approx(0.0));
            }
         }
      }
   }
}

TEST_CASE("3D ProjectBdrCoefficientNormal Scalar",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   const int n = 1;
   const int dim = 3;
   const int order = 1;

   const double tol = 1e-6;

   const int bdrs_axis[] = {2, 1, 0, 1, 0, 2};
   const int bdrs_sign[] = {-1, -1, +1, +1, -1, +1};

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, Func_lin);

      SECTION("3D GetVectorValue tests for element type " +
              std::to_string(type))
      {
         RT_FECollection  rt_fec(order+1, dim);

         FiniteElementSpace  rt_fespace(&mesh,  &rt_fec);

         GridFunction  rt_x( &rt_fespace);

         VectorGridFunctionCoefficient  rt_xCoef( &rt_x);

         Array<int> bdr_marker(6);

         Vector normal(dim);
         Vector  f_val(dim);
         Vector rt_val(dim);

         for (int b = 1; b<=6; b++)
         {
            bdr_marker = 0;
            bdr_marker[b-1] = 1;

            rt_x = 0.0;

            normal = 0.;
            normal(bdrs_axis[b-1]) = (bdrs_sign[b-1] > 0)?(+1.):(-1.);
            VectorConstantCoefficient normCoef(normal);
            InnerProductCoefficient prodCoef(funcCoef, normCoef);
            rt_x.ProjectBdrCoefficientNormal(prodCoef, bdr_marker);

            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               Element *e = mesh.GetBdrElement(be);
               if (e->GetAttribute() != b) { continue; }

               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = rt_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double rt_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  CalcOrtho(T->Jacobian(), normal);

                  funcCoef.Eval(f_val, *T, ip);
                  rt_xCoef.Eval(rt_val, *T, ip);

                  rt_val -= f_val;

                  double rt_dist = rt_val * normal;

                  rt_err += rt_dist;

                  if (verbose_tests && rt_dist > tol)
                  {
                     mfem::out << be << ":" << j << " rt ("
                               << f_val[0] << "," << f_val[1] << "," << f_val[2]
                               << ") vs. ("
                               << rt_val[0] << "," << rt_val[1] << ","
                               << rt_val[2] << ") " << rt_dist << std::endl;
                  }
               }
               rt_err  /= ir.GetNPoints();

               REQUIRE( rt_err == MFEM_Approx(0.0));
            }
         }
      }
   }
}

TEST_CASE("3D ProjectBdrCoefficientTangent",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   const int n = 1;
   const int dim = 3;
   const int order = 1;

   const double tol = 1e-6;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, Func_lin);

      SECTION("3D GetVectorValue tests for element type " +
              std::to_string(type))
      {
         ND_FECollection  nd_fec(order+1, dim);

         FiniteElementSpace  nd_fespace(&mesh,  &nd_fec);

         GridFunction  nd_x( &nd_fespace);

         VectorGridFunctionCoefficient  nd_xCoef( &nd_x);

         Array<int> bdr_marker(6);

         Vector normal(dim);
         Vector  f_val(dim);
         Vector nd_val(dim);
         Vector    nxd(dim);

         for (int b = 1; b<=6; b++)
         {
            bdr_marker = 0;
            bdr_marker[b-1] = 1;

            nd_x = 0.0;
            nd_x.ProjectBdrCoefficientTangent(funcCoef, bdr_marker);

            for (int be = 0; be < mesh.GetNBE(); be++)
            {
               Element *e = mesh.GetBdrElement(be);
               if (e->GetAttribute() != b) { continue; }

               ElementTransformation *T = mesh.GetBdrElementTransformation(be);
               const FiniteElement   *fe = nd_fespace.GetBE(be);
               const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                        2*order + 2);

               double nd_err = 0.0;

               for (int j=0; j<ir.GetNPoints(); j++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  T->SetIntPoint(&ip);

                  CalcOrtho(T->Jacobian(), normal);

                  funcCoef.Eval(f_val, *T, ip);
                  nd_xCoef.Eval(nd_val, *T, ip);

                  nd_val -= f_val;

                  nxd[0] = normal[1] * nd_val[2] - normal[2] * nd_val[1];
                  nxd[1] = normal[2] * nd_val[0] - normal[0] * nd_val[2];
                  nxd[2] = normal[0] * nd_val[1] - normal[1] * nd_val[0];

                  double nd_dist = nxd.Norml2();

                  nd_err += nd_dist;

                  if (verbose_tests && nd_dist > tol)
                  {
                     mfem::out << be << ":" << j << " nd ("
                               << f_val[0] << "," << f_val[1] << "," << f_val[2]
                               << ") vs. ("
                               << nd_val[0] << "," << nd_val[1] << ","
                               << nd_val[2] << ") " << nd_dist << std::endl;
                  }
               }
               nd_err  /= ir.GetNPoints();

               REQUIRE( nd_err == MFEM_Approx(0.0));
            }
         }
      }
   }
}

TEST_CASE("ProjectBdrCoefficientTangent with IntegratedGLL",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   const int dim = GENERATE(2, 3);
   CAPTURE(dim);
   Mesh mesh = (dim == 2) ?
               Mesh::MakeCartesian2D(1, 2, Element::QUADRILATERAL,
                                     true, 2.0, 5.0) :
               Mesh::MakeCartesian3D(1, 1, 2, Element::HEXAHEDRON,
                                     2.0, 3.0, 5.0);
   mesh.EnsureNodes();
   mesh.EnsureNCMesh(false);
   VectorFunctionCoefficient func_coef(dim, Func_lin);
   Array<int> all_bdr(mesh.bdr_attributes.Max());
   all_bdr = 1;

   for (int order = 1; order <= 4; order++)
   {
      CAPTURE(order);
      ND_FECollection nd_fec(order, dim, BasisType::GaussLobatto,
                             BasisType::IntegratedGLL);
      FiniteElementSpace nd_fespace(&mesh, &nd_fec);
      GridFunction volume_projection(&nd_fespace);
      GridFunction boundary_projection(&nd_fespace);

      volume_projection.ProjectCoefficient(func_coef);
      boundary_projection = 0.0;
      boundary_projection.ProjectBdrCoefficientTangent(func_coef, all_bdr);

      Array<int> ess_vdofs;
      nd_fespace.GetEssentialVDofs(all_bdr, ess_vdofs);
      real_t max_error = 0.0;
      for (int i = 0; i < ess_vdofs.Size(); i++)
      {
         if (ess_vdofs[i])
         {
            max_error = std::max(max_error, std::abs(
                                    boundary_projection[i] -
                                    volume_projection[i]));
         }
      }
      REQUIRE(max_error == MFEM_Approx(0.0));
   }
}

} // namespace project_bdr
