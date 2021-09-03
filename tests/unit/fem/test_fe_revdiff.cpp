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

} // anonymous namespace

namespace fe_revdiff
{

template<typename T>
void runProjectRevDiffTest(Mesh &mesh, VectorCoefficient &vc);

template<typename T>
void runCalcVShapeRevDiffTest(Mesh &mesh);

void runCalcPhysCurlShapeRevDiffTest(Mesh &mesh);

constexpr double eps_fd = 1e-5;
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(-1.0,1.0);

TEST_CASE("VectorFiniteElement::ProjectRevDiff - 2D")
{
   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh2D(meshStr);

   REQUIRE(mesh2D.GetNE() == 1);
   REQUIRE(mesh2D.GetNodes() != nullptr);

   VectorFunctionCoefficient vc2D(2, func2D, func2DRevDiff);

   runProjectRevDiffTest<RT_FECollection>(mesh2D, vc2D);
   runProjectRevDiffTest<ND_FECollection>(mesh2D, vc2D);
}

TEST_CASE("VectorFiniteElement::ProjectRevDiff - 3D")
{
   auto mesh3D = Mesh::MakeCartesian3D(2, 1, 1, Element::TETRAHEDRON,
                                       2.0, 1.0, 3.0, true);
   mesh3D.ReorientTetMesh();
   mesh3D.EnsureNodes();

   VectorFunctionCoefficient vc3D(3, func3D, func3DRevDiff);

   runProjectRevDiffTest<RT_FECollection>(mesh3D, vc3D);
   runProjectRevDiffTest<ND_FECollection>(mesh3D, vc3D);
}

TEST_CASE("VectorFiniteElement::CalcVShape_RTRevDiff - 2D")
{
   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh2D(meshStr);
   REQUIRE(mesh2D.GetNE() == 1);
   REQUIRE(mesh2D.GetNodes() != nullptr);
   runCalcVShapeRevDiffTest<RT_FECollection>(mesh2D);
}

TEST_CASE("VectorFiniteElement::CalcVShape_NDRevDiff - 2D")
{
   // Create quadratic mesh with single C-shaped quadrilateral
   std::stringstream meshStr;
   meshStr << mesh_str;
   Mesh mesh2D(meshStr);
   REQUIRE(mesh2D.GetNE() == 1);
   REQUIRE(mesh2D.GetNodes() != nullptr);
   runCalcVShapeRevDiffTest<ND_FECollection>(mesh2D);
}

TEST_CASE("VectorFiniteElement::CalcVShape_RTRevDiff - 3D")
{
   auto mesh3D = Mesh::MakeCartesian3D(2, 1, 1, Element::TETRAHEDRON,
                                       2.0, 1.0, 3.0, true);
   mesh3D.ReorientTetMesh();
   mesh3D.EnsureNodes();
   runCalcVShapeRevDiffTest<RT_FECollection>(mesh3D);
}

TEST_CASE("VectorFiniteElement::CalcVShape_NDRevDiff - 3D")
{
   auto mesh3D = Mesh::MakeCartesian3D(2, 1, 1, Element::TETRAHEDRON,
                                       2.0, 1.0, 3.0, true);
   mesh3D.ReorientTetMesh();
   mesh3D.EnsureNodes();
   runCalcVShapeRevDiffTest<ND_FECollection>(mesh3D);
}

TEST_CASE("FiniteElement::CalcPhysCurlShapeRevDiff - 3D")
{
   auto mesh3D = Mesh::MakeCartesian3D(2, 2, 1, Element::TETRAHEDRON,
                                       2.0, 1.0, 3.0, true);
   mesh3D.ReorientTetMesh();
   mesh3D.EnsureNodes();
   runCalcPhysCurlShapeRevDiffTest(mesh3D);
}

template<typename T>
void runProjectRevDiffTest(Mesh &mesh, VectorCoefficient &vc)
{
   for (int p = 1; p <= 4; ++p)
   {
      const int dim = mesh.Dimension();
      T fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      const FiniteElement &el = *fes.GetFE(0);
      IsoparametricTransformation trans;
      mesh.GetElementTransformation(0, &trans);

      // P_bar is the vector contracted with the derivative of the projection
      // the values are not important for this test
      const int dof = el.GetDof();
      Vector P_bar(dof);
      for (int i = 0; i < P_bar.Size(); ++i)
      {
         P_bar(i) = distribution(generator);
      }

      // reverse-mode differentiation of projection
      DenseMatrix &coords = trans.GetPointMat();
      DenseMatrix coords_bar(coords.Height(), coords.Width());
      coords_bar = 0.0;
      el.ProjectRevDiff(P_bar, vc, trans, coords_bar);

      // get the weighted derivatives using finite difference method
      Vector dofs_fd(dof), dofs_pert(dof);
      for (int n = 0; n < coords.Width(); ++n)
      {
         for (int di = 0; di < coords.Height(); ++di)
         {
            coords(di, n) += eps_fd;
            trans.Reset();
            el.Project(vc, trans, dofs_fd);
            coords(di, n) -= 2.0 * eps_fd;
            trans.Reset();
            el.Project(vc, trans, dofs_pert);
            dofs_fd -= dofs_pert;
            dofs_fd *= 1.0 / (2.0 * eps_fd);
            coords(di, n) += eps_fd;
            double x_bar_fd = P_bar * dofs_fd;

            REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
         }
      }
   }
}

template<typename T>
void runCalcVShapeRevDiffTest(Mesh &mesh)
{
   for (int p = 1; p <= 4; ++p)
   {
      const int dim = mesh.Dimension();
      T fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      const FiniteElement &el = *fes.GetFE(0);
      IsoparametricTransformation trans;
      mesh.GetElementTransformation(0, &trans);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         const int dof = el.GetDof();
         const int el_dim = el.GetDim();
         DenseMatrix vshape_bar(dof, el_dim);
         for (int k = 0; k < vshape_bar.Width(); ++k)
         {
            for (int j = 0; j < vshape_bar.Height(); ++j)
            {
               vshape_bar(j, k) = distribution(generator);
            }
         }

         // reverse-mode differentiation CalcVShape
         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());
         coords_bar = 0.0;
         el.CalcVShapeRevDiff(trans, vshape_bar, coords_bar);

         // get the weighted derivatives using finite difference method
         DenseMatrix vshape_fd(dof, el_dim), vshape_pert(dof, el_dim);
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               trans.Reset();
               el.CalcVShape(trans, vshape_fd);
               coords(di, n) -= 2.0 * eps_fd;
               trans.Reset();
               el.CalcVShape(trans, vshape_pert);
               vshape_fd -= vshape_pert;
               vshape_fd *= 1.0 / (2.0 * eps_fd);
               coords(di, n) += eps_fd;
               double x_bar_fd = vshape_bar * vshape_fd;

               REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
            }
         }
      }
   }
}

void runCalcPhysCurlShapeRevDiffTest(Mesh &mesh)
{
   for (int p = 1; p <= 4; ++p)
   {
      const int dim = mesh.Dimension();
      ND_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec);

      const FiniteElement &el = *fes.GetFE(0);
      IsoparametricTransformation trans;
      mesh.GetElementTransformation(0, &trans);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         const int dof = el.GetDof();
         const int el_dim = el.GetDim();
         DenseMatrix curlshape_bar(dof, el_dim);
         for (int k = 0; k < curlshape_bar.Width(); ++k)
         {
            for (int j = 0; j < curlshape_bar.Height(); ++j)
            {
               curlshape_bar(j, k) = distribution(generator);
            }
         }

         // reverse-mode differentiation CalcPhysCurlShape
         DenseMatrix &coords = trans.GetPointMat();
         DenseMatrix coords_bar(coords.Height(), coords.Width());
         coords_bar = 0.0;
         el.CalcPhysCurlShapeRevDiff(trans, curlshape_bar, coords_bar);

         // get the weighted derivatives using finite difference method
         DenseMatrix curlshape_fd(dof, el_dim), curlshape_pert(dof, el_dim);
         for (int n = 0; n < coords.Width(); ++n)
         {
            for (int di = 0; di < coords.Height(); ++di)
            {
               coords(di, n) += eps_fd;
               trans.Reset();
               el.CalcPhysCurlShape(trans, curlshape_fd);
               coords(di, n) -= 2.0 * eps_fd;
               trans.Reset();
               el.CalcPhysCurlShape(trans, curlshape_pert);
               curlshape_fd -= curlshape_pert;
               curlshape_fd *= 1.0 / (2.0 * eps_fd);
               coords(di, n) += eps_fd;
               double x_bar_fd = curlshape_bar * curlshape_fd;

               REQUIRE(coords_bar(di, n) == Approx(x_bar_fd));
            }
         }
      }
   }
}

} // namespace fe_revdiff
