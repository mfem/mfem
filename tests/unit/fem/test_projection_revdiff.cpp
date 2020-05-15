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

std::string one_tet_mesh_str = 
   "MFEM mesh v1.0\n\n"
   "dimension\n"
   "3\n\n"
   "elements\n"
   "1\n"
   "1 4 0 1 2 3\n\n"
   "boundary\n"
   "4\n"
   "1 2 0 1 2\n"
   "2 2 0 1 3\n"
   "3 2 1 2 3\n"
   "4 2 0 2 3\n\n"
   "vertices\n"
   "4\n\n"
   "nodes\n"
   "FiniteElementSpace\n"
   "FiniteElementCollection: H1_3D_P1\n"
   "VDim: 3\n"
   "Ordering: 1\n\n"
   "0 0 0\n"
   "1 0 0\n"
   "0 1 0\n"
   "0 0 1\n";

std::string two_tet_mesh_str = 
   "MFEM mesh v1.0\n\n"
   "dimension\n"
   "3\n\n"
   "elements\n"
   "2\n"
   "1 4 0 1 2 3\n"
   "1 4 1 2 3 4\n\n"
   "boundary\n"
   "7\n"
   "1 2 0 1 2\n"
   "2 2 0 1 3\n"
   "3 2 1 2 3\n"
   "4 2 0 2 3\n"
   "5 2 1 3 4\n"
   "6 2 1 2 4\n"
   "7 2 2 3 4\n\n"
   "vertices\n"
   "5\n\n"
   "nodes\n"
   "FiniteElementSpace\n"
   "FiniteElementCollection: H1_3D_P1\n"
   "VDim: 3\n"
   "Ordering: 1\n\n"
   "0 0 0\n"
   "1 0 0\n"
   "0 1 0\n"
   "0 0 1\n"
   "1 0 1\n";

void func2D(const Vector &x, Vector &y)
{
   y.SetSize(2);
   y(0) = x(0)*x(0) - x(1);
   y(1) = x(0) * exp(x(1));
}

void func2DRevDiff(const Vector &x, const Vector &v_bar, Vector &x_bar)
{
   x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1));
   x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1)); 
}

void func3D(const Vector &x, Vector &y)
{
   y.SetSize(3);
   y(0) = x(0)*x(0) - x(1);
   y(1) = x(0) * exp(x(1));
   y(2) = x(2)*x(0) - x(1);
}

void func3DRevDiff(const Vector &x, const Vector &v_bar, Vector &x_bar)
{
   x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1)) + v_bar(2)*x(2);
   x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1)) - v_bar(2); 
   x_bar(2) = v_bar(2) * x(0); 
}

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(-1.0,1.0);

void randState(const mfem::Vector &x, mfem::Vector &u)
{
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = distribution(generator);
   }
}

} // anonymous namespace

namespace el_project_revdiff
{

TEST_CASE("FiniteElement::Project_RevDiff reverse-mode differentiation",
          "[FiniteElement]")
{
   constexpr double eps_fd = 1e-5;

   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh2D(meshStr);

   REQUIRE( mesh2D.GetNE() == 1 );
   REQUIRE( mesh2D.GetNodes() != NULL );

   bool dumpMesh = false;
   if (dumpMesh)
   {
      std::ofstream mesh_ostream("finitelement-project-revdiff-quad-mesh.vtk");
      mesh_ostream.precision(14);
      int refine = 10;
      mesh2D.PrintVTK(mesh_ostream, refine);
   }

   VectorFunctionCoefficient vc2D(2, func2D, func2DRevDiff);

   for (int p = 1; p <= 4; ++p)
   {
      SECTION("Project_RT_RevDiff (quad) for degree p = " + std::to_string(p))
      {
         RT_FECollection fec(p, 2);
         FiniteElementSpace fes(&mesh2D, &fec);

         const FiniteElement &el = *fes.GetFE(0);
         IsoparametricTransformation trans;
         mesh2D.GetElementTransformation(0, &trans);

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
         el.Project_RevDiff(P_bar, vc2D, trans, coords_bar);

         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               el.Project(vc2D, trans, dofs_fd);
               coords(di, n) -= 2.0*eps_fd;
               el.Project(vc2D, trans, dofs_pert);
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
      SECTION("Project_ND_RevDiff (quad) for degree p = " + std::to_string(p))
      {
         ND_FECollection fec(p, 2);
         FiniteElementSpace fes(&mesh2D, &fec);

         const FiniteElement &el = *fes.GetFE(0);
         IsoparametricTransformation trans;
         mesh2D.GetElementTransformation(0, &trans);

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
         el.Project_RevDiff(P_bar, vc2D, trans, coords_bar);

         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               el.Project(vc2D, trans, dofs_fd);
               coords(di, n) -= 2.0*eps_fd;
               el.Project(vc2D, trans, dofs_pert);
               dofs_fd -= dofs_pert;
               dofs_fd *= 1.0/(2.0*eps_fd);
               coords(di, n) += eps_fd;
               double x_bar_fd = P_bar * dofs_fd;

               REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
            }
         }
      }
   }

   Mesh mesh3D(1, 1, 1, Element::TETRAHEDRON,
             true /* gen. edges */, 1.0,
             1.0, 1.0, true);
   mesh3D.EnsureNodes();

   if (dumpMesh)
   {
      std::ofstream mesh_ostream("finitelement-project-revdiff-tet-mesh.vtk");
      mesh_ostream.precision(14);
      int refine = 10;
      mesh3D.PrintVTK(mesh_ostream, refine);
   }

   VectorFunctionCoefficient vc3D(3, func3D, func3DRevDiff);

   for (int p = 1; p <= 4; ++p)
   {
      SECTION("Project_RT_RevDiff (tet) for degree p = " + std::to_string(p))
      {
         RT_FECollection fec(p, 3);
         FiniteElementSpace fes(&mesh3D, &fec);

         const FiniteElement &el = *fes.GetFE(0);
         IsoparametricTransformation trans;
         mesh3D.GetElementTransformation(0, &trans);

         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());

         const int dof = el.GetDof();

         // P_bar is the vector contracted with the derivative of the projection
         // the values are not important for this test
         Vector P_bar(dof);
         for (int i = 0; i < P_bar.Size(); ++i)
         {
            P_bar(i) = distribution(generator);
         }

         Vector dofs_fd(dof), dofs_pert(dof);

         // reverse-mode differentiation of projection
         coords_bar = 0.0;
         el.Project_RevDiff(P_bar, vc3D, trans, coords_bar);

         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               el.Project(vc3D, trans, dofs_fd);
               coords(di, n) -= 2.0*eps_fd;
               el.Project(vc3D, trans, dofs_pert);
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
      SECTION("Project_ND_RevDiff (tet) for degree p = " + std::to_string(p))
      {
         ND_FECollection fec(p, 3);
         FiniteElementSpace fes(&mesh3D, &fec);

         const FiniteElement &el = *fes.GetFE(0);
         IsoparametricTransformation trans;
         mesh3D.GetElementTransformation(0, &trans);

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
         el.Project_RevDiff(P_bar, vc3D, trans, coords_bar);

         // get the weighted derivatives using finite difference method
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               el.Project(vc3D, trans, dofs_fd);
               coords(di, n) -= 2.0*eps_fd;
               el.Project(vc3D, trans, dofs_pert);
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

TEST_CASE("GridFunction::ProjectCoefficientRevDiff",
          "[GridFunction]")
{
   constexpr double eps_fd = 1e-5;

   // Create tet mesh with single tet
   std::stringstream meshStr;
   meshStr << two_tet_mesh_str;
   // meshStr << one_tet_mesh_str;
   Mesh mesh(meshStr);

   mesh.ReorientTetMesh();
   // mesh.SetCurvature(4);
   mesh.EnsureNodes();

   if (true)
   {
      std::ofstream mesh_ostream("gridfunction-project-revdiff-tet-mesh.vtk");
      mesh_ostream.precision(14);
      int refine = 0;
      mesh.PrintVTK(mesh_ostream, refine);
      // mesh.Print();
   }
   for (int p = 1; p <= 1; ++p)
   {
      SECTION("Project_ND_RevDiff (tet) for degree p = " + std::to_string(p))
      {
         ND_FECollection fec(p, 3);
         FiniteElementSpace fes(&mesh, &fec, Ordering::byVDIM);

         const FiniteElement &el = *fes.GetFE(0);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh.GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // P_bar is the vector contracted with the derivative of the projection
         // the values are not important for this test
         GridFunction P_bar(&fes);
         for (int i = 0; i < P_bar.Size(); ++i)
         {
            P_bar(i) = distribution(generator);
         }
         // P_bar = 1.0;

         VectorFunctionCoefficient vc(3, func3D, func3DRevDiff);

         GridFunction dfdx(mesh_fes);
         dfdx.ProjectCoefficientRevDiff(P_bar, vc);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randState);
         // v.ProjectCoefficient(v_rand);


         // contract dfdx with v
         double dfdx_v = dfdx * v;

         GridFunction gf(&fes);
         gf.ProjectCoefficient(vc);

         /// This is for a directional derivative
         // // now compute the finite-difference approximation...
         // GridFunction x_pert(*x_nodes);
         // x_pert.Add(eps_fd, v);
         // mesh.SetNodes(x_pert);
         // fes.Update();
         // gf.Update();
         // gf.ProjectCoefficient(vc);
         // double dfdx_v_fd = P_bar * gf;
         // x_pert.Add(-2 * eps_fd, v);
         // mesh.SetNodes(x_pert);
         // fes.Update();
         // gf.Update();
         // gf.ProjectCoefficient(vc);
         // dfdx_v_fd -= P_bar * gf;
         // dfdx_v_fd /= (2 * eps_fd);
         // mesh.SetNodes(*x_nodes); // remember to reset the mesh nodes
         // std::cout << "dfdx_v_fd " << dfdx_v_fd << "\n";
         // REQUIRE(dfdx_v == Approx(dfdx_v_fd));

         /// To examine each component
         // // get the weighted derivatives using finite difference method
         for (int n = 0; n < x_nodes->Size(); ++n)
         {
            (*x_nodes)(n) += eps_fd;
            fes.Update();
            gf.Update();
            gf.ProjectCoefficient(vc);
            GridFunction gf_fd(gf);
            (*x_nodes)(n) -= 2.0*eps_fd;
            fes.Update();
            gf.Update();
            gf.ProjectCoefficient(vc);
            gf_fd -= gf;
            gf_fd *= 1.0/(2.0*eps_fd);
            (*x_nodes)(n) += eps_fd;
            double x_bar_fd = P_bar * gf_fd;

            std::cout << "n: " << n << " dfdx(n): " << dfdx(n) << " x_bar_fd: " << x_bar_fd << " diff: " << x_bar_fd - dfdx(n) << "\n"; 
            // REQUIRE(dfdx(n) == Approx(x_bar_fd));
         }
      }
   }
}

} // namespace el_project_revdiff
