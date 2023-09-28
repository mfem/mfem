// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

void Func_3D_lin(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] =  1.234 * x[0] - 2.357 * x[1] + 3.572 * x[2];
   v[1] =  2.537 * x[0] + 4.321 * x[1] - 1.234 * x[2];
   v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
}

TEST_CASE("3D ProjectBdrCoefficientTangent",
          "[GridFunction]"
          "[VectorGridFunctionCoefficient]")
{
   int n = 1;
   int dim = 3;
   int order = 1;

   double tol = 1e-6;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type)type, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient funcCoef(dim, Func_3D_lin);

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

} // namespace project_bdr
