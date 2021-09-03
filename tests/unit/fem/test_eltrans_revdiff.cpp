// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

using namespace mfem;

// String used to define a single element mesh, with a c-shaped quad element
std::string mesh_str =
   "MFEM mesh v1.0"                    "\n\n"
   "dimension"                           "\n"
   "2"                                 "\n\n"
   "elements"                            "\n"
   "1"                                   "\n"
   "1 3 0 1 2 3"                       "\n\n"
   "boundary"                            "\n"
   "0"                                 "\n\n"
   "vertices"                            "\n"
   "4"                                 "\n\n"
   "nodes"                               "\n"
   "FiniteElementSpace"                  "\n"
   "FiniteElementCollection: Quadratic"  "\n"
   "VDim: 2"                             "\n"
   "Ordering: 1"                         "\n"
   "0 0"                                 "\n"
   "0 2"                                 "\n"
   "0 6"                                 "\n"
   "0 8"                                 "\n"
   "0 1"                                 "\n"
   "-6 4"                                "\n"
   "0 7"                                 "\n"
   "-8 4"                                "\n"
   "-7 4"                                "\n";

TEST_CASE("IsoparametricTransformation reverse-mode differentiation",
          "[IsoparametricTransformation]")
{
   constexpr double eps_fd = 1e-5;

   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh(meshStr);

   REQUIRE( mesh.GetNE() == 1 );
   REQUIRE( mesh.GetNodes() != NULL );

   bool dumpMesh = false;
   if (dumpMesh)
   {
      std::ofstream mesh_ostream("isoparametric-revdiff-mesh.vtk");
      mesh_ostream.precision(14);
      int refine = 10;
      mesh.PrintVTK(mesh_ostream, refine);
   }

   // Create the transformation and get integration rule
   IsoparametricTransformation trans;
   mesh.GetElementTransformation(0, &trans);
   const int intorder = 5;
   const IntegrationRule *ir = &IntRules.Get(mesh.GetElementBaseGeometry(0),
                                             intorder);
   DenseMatrix &coords = trans.GetPointMat();
   DenseMatrix coords_bar(coords.Height(), coords.Width());

   SECTION("TransformRevDiff")
   {
      // x_bar(i) is the weight on the (i)th entry of the coordinate x;
      // the values are not important for this test.
      double x_bar_data[4] = {2.5, -3.2};
      Vector x_bar(x_bar_data, 2);
      Vector x_fd(2), x_pert(2);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         // reverse-mode differentiation of coordinate transformation
         coords_bar = 0.0;
         trans.TransformRevDiff(ip, x_bar, coords_bar);
         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               trans.Transform(ip, x_fd);
               coords(di, n) -= 2.0*eps_fd;
               trans.Transform(ip, x_pert);
               x_fd -= x_pert;
               x_fd *= 1.0/(2.0*eps_fd);
               coords(di, n) += eps_fd;
               double x_bar_fd = 0.0;
               for (int j = 0; j < x_bar.Size(); ++j)
               {
                  x_bar_fd += x_bar(j)*x_fd(j);
               }
               REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
            }
         }
      }
   }

   SECTION("JacobianRevDiff")
   {
      // dFdx_bar(i,j) is the weight on the (i,j)th entry of the Jacobian;
      // the values are not important for this test.
      double dFdx_bar_data[4] = {2.0, -3.0, 4.0, -1.0};
      DenseMatrix dFdx_bar(dFdx_bar_data, 2, 2);
      DenseMatrix dFdx_fd(2,2);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         // reverse-mode differentiation of Jacobian of mapping
         coords_bar = 0.0;
         trans.JacobianRevDiff(dFdx_bar, coords_bar);
         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               trans.SetIntPoint(&ip); // force re-evaluation of Jacobian
               dFdx_fd = trans.Jacobian();
               coords(di, n) -= 2.0*eps_fd;
               trans.SetIntPoint(&ip); // force re-evaluation of Jacobian
               dFdx_fd -= trans.Jacobian();
               dFdx_fd *= 1.0/(2.0*eps_fd);
               coords(di, n) += eps_fd;
               double dFdx_bar_fd = 0.0;
               for (int j = 0; j < dFdx_bar.Height(); ++j)
               {
                  for (int k = 0; k < dFdx_bar.Width(); ++k)
                  {
                     dFdx_bar_fd += dFdx_bar(j,k)*dFdx_fd(j,k);
                  }
               }
               REQUIRE(coords_bar(di, n) == Approx(dFdx_bar_fd));
            }
         }
      }
   }

   SECTION("AdjugateJacobianRevDiff")
   {
      // adjJ_bar(i,j) is the weight on the (i,j)th entry of the Adjugate;
      // the values are not important for this test.
      double adjJ_bar_data[4] = {2.0, -3.0, 4.0, -1.0};
      DenseMatrix adjJ_bar(adjJ_bar_data, 2, 2);
      DenseMatrix adjJ_fd(2,2);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         // reverse-mode differentiation of Adjugate of mapping
         coords_bar = 0.0;
         trans.AdjugateJacobianRevDiff(adjJ_bar, coords_bar);
         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               trans.SetIntPoint(&ip); // force re-evaluation of Adjugate
               adjJ_fd = trans.AdjugateJacobian();
               coords(di, n) -= 2.0*eps_fd;
               trans.SetIntPoint(&ip); // force re-evaluation of Adjugate
               adjJ_fd -= trans.AdjugateJacobian();
               adjJ_fd *= 1.0/(2.0*eps_fd);
               coords(di, n) += eps_fd;
               double adjJ_bar_fd = 0.0;
               for (int j = 0; j < adjJ_bar.Height(); ++j)
               {
                  for (int k = 0; k < adjJ_bar.Width(); ++k)
                  {
                     adjJ_bar_fd += adjJ_bar(j,k)*adjJ_fd(j,k);
                  }
               }
               REQUIRE(coords_bar(di, n) == Approx(adjJ_bar_fd));
            }
         }
      }
   }

   SECTION("WeightRevDiff")
   {
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         // get the gradient of the Weight() using reverse mode
         coords_bar = 0.0;
         trans.WeightRevDiff(coords_bar);
         // get the gradient of the Weight() using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               trans.SetIntPoint(&ip); // force re-evaluation of Weight
               double dWeight_fd = trans.Weight();
               coords(di, n) -= 2.0*eps_fd;
               trans.SetIntPoint(&ip); // force re-evaluation of Weight
               dWeight_fd -= trans.Weight();
               dWeight_fd /= (2.0*eps_fd);
               coords(di, n) += eps_fd;
               REQUIRE(coords_bar(di, n) == Approx(dWeight_fd));
            }
         }
      }
   }

}