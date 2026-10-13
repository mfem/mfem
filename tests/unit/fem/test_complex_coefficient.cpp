// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <type_traits>

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace
{

class TimeDependentComplexCoefficient : public ComplexCoefficient
{
public:
   complex_t Eval(ElementTransformation &, const IntegrationPoint &) override
   {
      return complex_t(GetTime(), -2.0*GetTime());
   }
};

class TimeDependentComplexVectorCoefficient : public ComplexVectorCoefficient
{
public:
   TimeDependentComplexVectorCoefficient() : ComplexVectorCoefficient(2) { }

   void Eval(ComplexVector &V, ElementTransformation &,
             const IntegrationPoint &) override
   {
      V.SetSize(2);
      V(0) = complex_t(GetTime(), -GetTime());
      V(1) = complex_t(2.0*GetTime(), -3.0*GetTime());
   }
};

class TimeDependentComplexMatrixCoefficient : public ComplexMatrixCoefficient
{
public:
   TimeDependentComplexMatrixCoefficient() : ComplexMatrixCoefficient(1) { }

   void Eval(ComplexDenseMatrixData &M, ElementTransformation &,
             const IntegrationPoint &) override
   {
      M.SetSize(1, 1);
      M(0, 0) = complex_t(GetTime(), -2.0*GetTime());
   }
};

static_assert(!std::is_copy_constructible<ComplexCoefficient>::value, "");
static_assert(!std::is_move_constructible<ComplexCoefficient>::value, "");
static_assert(!std::is_copy_constructible<ComplexVectorCoefficient>::value, "");
static_assert(!std::is_move_constructible<ComplexVectorCoefficient>::value, "");
static_assert(!std::is_copy_constructible<ComplexMatrixCoefficient>::value, "");
static_assert(!std::is_move_constructible<ComplexMatrixCoefficient>::value, "");

}

TEST_CASE("Complex coefficient views", "[ComplexCoefficient]")
{
   IsoparametricTransformation T;
   IntegrationPoint ip;

   SECTION("Constant parts")
   {
      ComplexConstantCoefficient coefficient(3.0, -4.0);
      REQUIRE(coefficient.real().Eval(T, ip) == MFEM_Approx(3.0));
      REQUIRE(coefficient.imag().Eval(T, ip) == MFEM_Approx(-4.0));
   }

   SECTION("Time propagation")
   {
      TimeDependentComplexCoefficient scalar;
      scalar.real().SetTime(2.0);
      REQUIRE(scalar.GetTime() == MFEM_Approx(2.0));
      REQUIRE(scalar.imag().Eval(T, ip) == MFEM_Approx(-4.0));

      TimeDependentComplexVectorCoefficient vector;
      vector.imag().SetTime(3.0);
      Vector vector_value;
      vector.real().Eval(vector_value, T, ip);
      REQUIRE(vector.GetTime() == MFEM_Approx(3.0));
      REQUIRE(vector_value(0) == MFEM_Approx(3.0));
      REQUIRE(vector_value(1) == MFEM_Approx(6.0));

      TimeDependentComplexMatrixCoefficient matrix;
      matrix.real().SetTime(4.0);
      DenseMatrix matrix_value;
      matrix.imag().Eval(matrix_value, T, ip);
      REQUIRE(matrix.GetTime() == MFEM_Approx(4.0));
      REQUIRE(matrix_value(0, 0) == MFEM_Approx(-8.0));
   }
}

TEST_CASE("Complex matrix coefficient composition", "[ComplexCoefficient]")
{
   DenseMatrix real_part(2), imag_part(2);
   real_part = 0.0;
   imag_part = 0.0;
   real_part(0, 0) = 1.0;
   real_part(1, 1) = 2.0;
   imag_part(0, 1) = 3.0;
   imag_part(1, 0) = -4.0;

   MatrixConstantCoefficient real_coefficient(real_part);
   MatrixConstantCoefficient imag_coefficient(imag_part);
   ComplexMatrixCoefficient coefficient(real_coefficient, imag_coefficient);

   IsoparametricTransformation T;
   IntegrationPoint ip;
   ComplexDenseMatrixData value;
   coefficient.Eval(value, T, ip, 1.0);

   REQUIRE(value(0, 0) == complex_t(1.0, 0.0));
   REQUIRE(value(1, 1) == complex_t(2.0, 0.0));
   REQUIRE(value(0, 1) == complex_t(0.0, 3.0));
   REQUIRE(value(1, 0) == complex_t(0.0, -4.0));
}

TEST_CASE("ComplexGridFunction Lp errors", "[ComplexGridFunction]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                     false, 1.0, 1.0);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   ComplexGridFunction solution(&fes);
   solution = complex_t(3.0, 4.0);

   ConstantCoefficient zero(0.0);
   ConstantCoefficient weight(2.0);
   REQUIRE(solution.ComputeL2Error(zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeLpError(1.0, zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeLpError(2.0, zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeLpError(infinity(), zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeMaxError(zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeLpError(2.0, zero, zero, &weight) ==
           MFEM_Approx(5.0*sqrt(2.0)));
}

#ifdef MFEM_USE_MPI

TEST_CASE("ParComplexGridFunction Lp errors",
          "[ParComplexGridFunction][Parallel]")
{
   const int nx = std::max(2, Mpi::WorldSize());
   Mesh mesh = Mesh::MakeCartesian2D(nx, 1, Element::QUADRILATERAL,
                                     false, 1.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   H1_FECollection fec(1, pmesh.Dimension());
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParComplexGridFunction solution(&pfes);
   solution = complex_t(3.0, 4.0);
   solution.real().SetTrueVector();
   solution.real().SetFromTrueVector();
   solution.imag().SetTrueVector();
   solution.imag().SetFromTrueVector();

   ConstantCoefficient zero(0.0);
   ConstantCoefficient weight(2.0);
   REQUIRE(solution.ComputeL2Error(zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeLpError(1.0, zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeLpError(2.0, zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeLpError(infinity(), zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeMaxError(zero, zero) == MFEM_Approx(5.0));
   REQUIRE(solution.ComputeLpError(2.0, zero, zero, &weight) ==
           MFEM_Approx(5.0*sqrt(2.0)));
}

#endif // MFEM_USE_MPI
