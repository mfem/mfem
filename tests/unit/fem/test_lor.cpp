// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include <fstream>
#include <sstream>

using namespace mfem;

// Tests the use of refined/LOR grid function coefficients.
//
// Given a space fes, and a refined space fes_refined (either vector or scalar
// spaces), projects coeff_1 onto a grid function in fes, and then creates the
// corresponding grid function coefficient (could be scalar, vector, grad, div,
// or curl grid function coefficients). Then, this grid function coefficient is
// projected onto fes_refined, and compared with the result of projecting
// coeff_2 onto fes_refined.
//
// If coeff_1 can be represented exactly in fes, then these two projections
// should be identical.
template <typename GridFunctionCoeffType=GridFunctionCoefficient,
          typename CoeffType1, typename CoeffType2>
void TestRefinedGridFunctionCoefficient(
   FiniteElementSpace &fes, FiniteElementSpace &fes_refined,
   CoeffType1 &coeff_1, CoeffType2 &coeff_2)
{
   GridFunction gf(&fes);
   gf.ProjectCoefficient(coeff_1);
   GridFunctionCoeffType gf_coeff(&gf);

   GridFunction gf_refined_1(&fes_refined), gf_refined_2(&fes_refined);
   gf_refined_1.ProjectCoefficient(coeff_2);
   gf_refined_2.ProjectCoefficient(gf_coeff);

   gf_refined_2 -= gf_refined_1;
   REQUIRE(gf_refined_2.Normlinf() == MFEM_Approx(0.0));
}

// Forward declarations for functions defined in test_lin_interp.cpp
namespace lin_interp
{
double f2(const Vector & x);
void F2(const Vector & x, Vector & v);
void Grad_f2(const Vector & x, Vector & df);
double curlF2(const Vector & x);
double DivF2(const Vector & x);
double f3(const Vector & x);
void F3(const Vector & x, Vector & v);
void Grad_f3(const Vector & x, Vector & df);
void CurlF3(const Vector & x, Vector & df);
double DivF3(const Vector & x);
}

namespace detail
{
Mesh MakeCartesian(int dim, int nx)
{
   if (dim == 1) { return Mesh::MakeCartesian1D(nx); }
   else if (dim == 2) { return Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL); }
   else { return Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON); }
}
}

TEST_CASE("LOR GridFunction Coefficient", "[LOR][GridFunctionCoefficient]")
{
   auto dim = GENERATE(2, 3);
   Mesh mesh = ::detail::MakeCartesian(dim, 2);
   Mesh mesh_refined = Mesh::MakeRefined(mesh, 3, Quadrature1D::GaussLobatto);

   int order = 1;
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace fes_refined(&mesh_refined, &fec);

   FiniteElementSpace vec_fes(&mesh, &fec, dim);
   FiniteElementSpace vec_fes_refined(&mesh_refined, &fec, dim);

   auto f = (dim == 2)? lin_interp::f2 : lin_interp::f3;
   auto F = (dim == 2)? lin_interp::F2 : lin_interp::F3;
   auto grad = (dim == 2)? lin_interp::Grad_f2 : lin_interp::Grad_f3;
   auto div = (dim == 2)? lin_interp::DivF2 : lin_interp::DivF3;

   FunctionCoefficient f_coeff(f);
   VectorFunctionCoefficient vec_coeff(dim, F);
   VectorFunctionCoefficient grad_coeff(dim, grad);
   FunctionCoefficient div_coeff(div);

   TestRefinedGridFunctionCoefficient<GridFunctionCoefficient>(
      fes, fes_refined, f_coeff, f_coeff);
   TestRefinedGridFunctionCoefficient<VectorGridFunctionCoefficient>(
      vec_fes, vec_fes_refined, vec_coeff, vec_coeff);
   TestRefinedGridFunctionCoefficient<DivergenceGridFunctionCoefficient>(
      vec_fes, fes_refined, vec_coeff, div_coeff);
   TestRefinedGridFunctionCoefficient<GradientGridFunctionCoefficient>(
      fes, vec_fes_refined, f_coeff, grad_coeff);

   // Curl is treated differently for dim = 2 (where it is a scalar quantity)
   // and dim = 3 (where it is a vector quantity)
   if (dim == 2)
   {
      FunctionCoefficient curl_coeff(lin_interp::curlF2);
      TestRefinedGridFunctionCoefficient<CurlGridFunctionCoefficient>(
         vec_fes, fes_refined, vec_coeff, curl_coeff);
   }
   else if (dim == 3)
   {
      VectorFunctionCoefficient curl_coeff(dim, lin_interp::CurlF3);
      TestRefinedGridFunctionCoefficient<CurlGridFunctionCoefficient>(
         vec_fes, vec_fes_refined, vec_coeff, curl_coeff);
   }
}
