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

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <random>

using namespace mfem;

namespace
{
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

void func(const Vector &x, Vector &y)
{
   y.SetSize(2);
   y(0) = x(0)*x(0) - x(1);
   y(1) = x(0) * exp(x(1));
}

void funcRevDiff(const Vector &x, const Vector &v_bar, Vector &x_bar)
{
   x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1));
   x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1)); 
}

} // anonymous namespace

namespace el_project_revdiff
{

TEST_CASE("FiniteElement::Project_RevDiff reverse-mode differentiation",
          "[FiniteElement]")
{
   constexpr double eps_fd = 1e-5;
   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(-1.0,1.0);

   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh(meshStr);

   // /// Added incase nedelec elements failed because of the curved mesh
   // Mesh mesh(1, 1, Element::QUADRILATERAL,
   //           true /* gen. edges */, 1.0,
   //           1.0, true);
   // mesh.EnsureNodes();

   REQUIRE( mesh.GetNE() == 1 );
   REQUIRE( mesh.GetNodes() != NULL );

   bool dumpMesh = false;
   if (dumpMesh)
   {
      std::ofstream mesh_ostream("finitelement-project-revdiff-mesh.vtk");
      mesh_ostream.precision(14);
      int refine = 10;
      mesh.PrintVTK(mesh_ostream, refine);
   }

   VectorFunctionCoefficient vc(2, func, funcRevDiff);

   for (int p = 1; p <= 4; ++p)
   {
      SECTION("Project_RT_RevDiff for degree p = " + std::to_string(p))
      {
         RT_FECollection fec(p, 2);
         FiniteElementSpace fes(&mesh, &fec);

         const FiniteElement &el = *fes.GetFE(0);
         IsoparametricTransformation trans;
         mesh.GetElementTransformation(0, &trans);

         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());

         const int dof = el.GetDof();

         // P_bar is the vector contrated with the derivative of the projection
         // the values are not important for this test
         Vector P_bar(dof);
         for (int i = 0; i < P_bar.Size(); ++i)
         {
            P_bar(i) = distribution(generator);
         }

         Vector dofs_fd(dof), dofs_pert(dof);
      
         // reverse-mode differentiation of projection
         coords_bar = 0.0;
         el.Project_RevDiff(P_bar, vc, trans, coords_bar);

         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               el.Project(vc, trans, dofs_fd);
               coords(di, n) -= 2.0*eps_fd;
               el.Project(vc, trans, dofs_pert);
               dofs_fd -= dofs_pert;
               dofs_fd *= 1.0/(2.0*eps_fd);
               coords(di, n) += eps_fd;
               double x_bar_fd = P_bar * dofs_fd;

               REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
            }
         }
      }
   }

   for (int p = 1; p <= 4; ++p)
   {
      SECTION("Project_ND_RevDiff for degree p = " + std::to_string(p))
      {
         ND_FECollection fec(p, 2);
         FiniteElementSpace fes(&mesh, &fec);

         const FiniteElement &el = *fes.GetFE(0);
         IsoparametricTransformation trans;
         mesh.GetElementTransformation(0, &trans);

         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());

         const int dof = el.GetDof();

         // P_bar is the vector contrated with the derivative of the projection
         // the values are not important for this test
         Vector P_bar(dof);
         for (int i = 0; i < P_bar.Size(); ++i)
         {
            P_bar(i) = distribution(generator);
         }

         Vector dofs_fd(dof), dofs_pert(dof);
      
         // reverse-mode differentiation of projection
         coords_bar = 0.0;
         el.Project_RevDiff(P_bar, vc, trans, coords_bar);

         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               el.Project(vc, trans, dofs_fd);
               coords(di, n) -= 2.0*eps_fd;
               el.Project(vc, trans, dofs_pert);
               dofs_fd -= dofs_pert;
               dofs_fd *= 1.0/(2.0*eps_fd);
               coords(di, n) += eps_fd;
               double x_bar_fd = P_bar * dofs_fd;

               REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
            }
         }
      }
   }
}

} // namespace el_project_revdiff
