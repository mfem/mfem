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
static_assert(std::is_abstract<ComplexCoefficient>::value, "");
static_assert(!std::is_copy_constructible<ComplexVectorCoefficient>::value, "");
static_assert(!std::is_move_constructible<ComplexVectorCoefficient>::value, "");
static_assert(std::is_abstract<ComplexVectorCoefficient>::value, "");
static_assert(!std::is_copy_constructible<ComplexMatrixCoefficient>::value, "");
static_assert(!std::is_move_constructible<ComplexMatrixCoefficient>::value, "");
static_assert(std::is_abstract<ComplexMatrixCoefficient>::value, "");

template <typename Form>
void CheckTemplatedLinearIntegratorRegistration(
   Form &form, ComplexCoefficient &complex_coefficient,
   Coefficient &real_coefficient, Array<int> &element_marker,
   Array<int> &boundary_marker)
{
   form.template AddDomainIntegrator<DomainLFIntegrator>(complex_coefficient);
   form.template AddDomainIntegrator<DomainLFIntegrator>(complex_coefficient,
                                                         element_marker);
   form.template AddDomainIntegrator<DomainLFIntegrator>(real_coefficient);
   form.template AddBoundaryIntegrator<DomainLFIntegrator>(complex_coefficient);
   form.template AddBoundaryIntegrator<DomainLFIntegrator>(complex_coefficient,
                                                           boundary_marker);
   form.template AddBdrFaceIntegrator<DomainLFIntegrator>(complex_coefficient);
   form.template AddBdrFaceIntegrator<DomainLFIntegrator>(complex_coefficient,
                                                          boundary_marker);

   REQUIRE(form.real().GetDLFI()->Size() == 3);
   REQUIRE(form.imag().GetDLFI()->Size() == 2);
   REQUIRE(form.real().GetBLFI()->Size() == 2);
   REQUIRE(form.imag().GetBLFI()->Size() == 2);
   REQUIRE(form.real().GetFLFI()->Size() == 2);
   REQUIRE(form.imag().GetFLFI()->Size() == 2);
   REQUIRE((*form.real().GetDLFI_Marker())[1] == &element_marker);
   REQUIRE((*form.imag().GetDLFI_Marker())[1] == &element_marker);
   REQUIRE((*form.real().GetFLFI_Marker())[1] == &boundary_marker);
   REQUIRE((*form.imag().GetFLFI_Marker())[1] == &boundary_marker);
}

template <typename Form>
void CheckTemplatedBilinearIntegratorRegistration(
   Form &form, ComplexCoefficient &complex_coefficient,
   Coefficient &real_coefficient, Array<int> &element_marker,
   Array<int> &boundary_marker)
{
   form.template AddDomainIntegrator<MassIntegrator>(complex_coefficient);
   form.template AddDomainIntegrator<MassIntegrator>(complex_coefficient,
                                                     element_marker);
   form.template AddDomainIntegrator<MassIntegrator>(real_coefficient);
   form.template AddBoundaryIntegrator<MassIntegrator>(complex_coefficient);
   form.template AddBoundaryIntegrator<MassIntegrator>(complex_coefficient,
                                                       boundary_marker);
   form.template AddInteriorFaceIntegrator<MassIntegrator>(complex_coefficient);
   form.template AddBdrFaceIntegrator<MassIntegrator>(complex_coefficient);
   form.template AddBdrFaceIntegrator<MassIntegrator>(complex_coefficient,
                                                      boundary_marker);

   REQUIRE(form.real().GetDBFI()->Size() == 3);
   REQUIRE(form.imag().GetDBFI()->Size() == 2);
   REQUIRE(form.real().GetBBFI()->Size() == 2);
   REQUIRE(form.imag().GetBBFI()->Size() == 2);
   REQUIRE(form.real().GetFBFI()->Size() == 1);
   REQUIRE(form.imag().GetFBFI()->Size() == 1);
   REQUIRE(form.real().GetBFBFI()->Size() == 2);
   REQUIRE(form.imag().GetBFBFI()->Size() == 2);
   REQUIRE((*form.real().GetDBFI_Marker())[1] == &element_marker);
   REQUIRE((*form.imag().GetDBFI_Marker())[1] == &element_marker);
   REQUIRE((*form.real().GetBBFI_Marker())[1] == &boundary_marker);
   REQUIRE((*form.imag().GetBBFI_Marker())[1] == &boundary_marker);
   REQUIRE((*form.real().GetBFBFI_Marker())[1] == &boundary_marker);
   REQUIRE((*form.imag().GetBFBFI_Marker())[1] == &boundary_marker);
}

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

      ConstantCoefficient real_part(1.0), imag_part(-2.0);
      ComplexCoefficientFromRealImag from_parts(real_part, imag_part);
      REQUIRE(from_parts.Eval(T, ip) == complex_t(1.0, -2.0));
      REQUIRE(&from_parts.real() == &real_part);
      REQUIRE(&from_parts.imag() == &imag_part);
   }

   SECTION("Time propagation")
   {
      TimeDependentComplexCoefficient scalar;
      scalar.SetTime(1.0);
      REQUIRE(scalar.GetTime() == MFEM_Approx(1.0));
      REQUIRE(scalar.real().GetTime() == MFEM_Approx(1.0));
      REQUIRE(scalar.imag().GetTime() == MFEM_Approx(1.0));

      scalar.real().SetTime(2.0);
      REQUIRE(scalar.GetTime() == MFEM_Approx(2.0));
      REQUIRE(scalar.real().GetTime() == MFEM_Approx(2.0));
      REQUIRE(scalar.imag().GetTime() == MFEM_Approx(2.0));
      REQUIRE(scalar.imag().Eval(T, ip) == MFEM_Approx(-4.0));

      TimeDependentComplexVectorCoefficient vector;
      vector.SetTime(3.0);
      REQUIRE(vector.GetTime() == MFEM_Approx(3.0));
      REQUIRE(vector.real().GetTime() == MFEM_Approx(3.0));
      REQUIRE(vector.imag().GetTime() == MFEM_Approx(3.0));

      vector.imag().SetTime(4.0);
      Vector vector_value;
      vector.real().Eval(vector_value, T, ip);
      REQUIRE(vector.GetTime() == MFEM_Approx(4.0));
      REQUIRE(vector.real().GetTime() == MFEM_Approx(4.0));
      REQUIRE(vector.imag().GetTime() == MFEM_Approx(4.0));
      REQUIRE(vector_value(0) == MFEM_Approx(4.0));
      REQUIRE(vector_value(1) == MFEM_Approx(8.0));

      TimeDependentComplexMatrixCoefficient matrix;
      matrix.SetTime(5.0);
      REQUIRE(matrix.GetTime() == MFEM_Approx(5.0));
      REQUIRE(matrix.real().GetTime() == MFEM_Approx(5.0));
      REQUIRE(matrix.imag().GetTime() == MFEM_Approx(5.0));

      matrix.real().SetTime(6.0);
      DenseMatrix matrix_value;
      matrix.imag().Eval(matrix_value, T, ip);
      REQUIRE(matrix.GetTime() == MFEM_Approx(6.0));
      REQUIRE(matrix.real().GetTime() == MFEM_Approx(6.0));
      REQUIRE(matrix.imag().GetTime() == MFEM_Approx(6.0));
      REQUIRE(matrix_value(0, 0) == MFEM_Approx(-12.0));

      ConstantCoefficient real_scalar, imag_scalar;
      ComplexCoefficientFromRealImag scalar_from_parts(real_scalar, imag_scalar);
      scalar_from_parts.SetTime(7.0);
      REQUIRE(scalar_from_parts.GetTime() == MFEM_Approx(7.0));
      REQUIRE(real_scalar.GetTime() == MFEM_Approx(7.0));
      REQUIRE(imag_scalar.GetTime() == MFEM_Approx(7.0));

      Vector real_vector(1), imag_vector(1);
      VectorConstantCoefficient real_vector_coef(real_vector);
      VectorConstantCoefficient imag_vector_coef(imag_vector);
      ComplexVectorCoefficientFromRealImag vector_from_parts(real_vector_coef,
                                                             imag_vector_coef);
      vector_from_parts.SetTime(8.0);
      REQUIRE(vector_from_parts.GetTime() == MFEM_Approx(8.0));
      REQUIRE(real_vector_coef.GetTime() == MFEM_Approx(8.0));
      REQUIRE(imag_vector_coef.GetTime() == MFEM_Approx(8.0));

      DenseMatrix real_matrix(1), imag_matrix(1);
      MatrixConstantCoefficient real_matrix_coef(real_matrix);
      MatrixConstantCoefficient imag_matrix_coef(imag_matrix);
      ComplexMatrixCoefficientFromRealImag matrix_from_parts(real_matrix_coef,
                                                             imag_matrix_coef);
      matrix_from_parts.SetTime(9.0);
      REQUIRE(matrix_from_parts.GetTime() == MFEM_Approx(9.0));
      REQUIRE(real_matrix_coef.GetTime() == MFEM_Approx(9.0));
      REQUIRE(imag_matrix_coef.GetTime() == MFEM_Approx(9.0));
   }

   SECTION("Vector output sizing")
   {
      Vector real_values({1.0, -2.0});
      Vector imag_values({3.0, -4.0});
      VectorConstantCoefficient real_coefficient(real_values);
      VectorConstantCoefficient imag_coefficient(imag_values);
      ComplexVectorCoefficientFromRealImag coefficient(real_coefficient,
                                                       imag_coefficient);
      ComplexVector value;

      coefficient.Eval(value, T, ip);

      REQUIRE(value.Size() == 2);
      REQUIRE(value(0) == complex_t(1.0, 3.0));
      REQUIRE(value(1) == complex_t(-2.0, -4.0));
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
   ComplexMatrixCoefficientFromRealImag coefficient(real_coefficient,
                                                    imag_coefficient);

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

TEST_CASE("Complex form templated integrator registration",
          "[ComplexCoefficient]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   Array<int> element_marker(mesh.attributes.Max());
   Array<int> boundary_marker(mesh.bdr_attributes.Max());
   element_marker = 1;
   boundary_marker = 1;
   ComplexConstantCoefficient complex_coefficient(2.0, -3.0);
   ConstantCoefficient real_coefficient(4.0);

   ComplexLinearForm linear_form(&fes);
   CheckTemplatedLinearIntegratorRegistration(
      linear_form, complex_coefficient, real_coefficient,
      element_marker, boundary_marker);

   SesquilinearForm bilinear_form(&fes);
   CheckTemplatedBilinearIntegratorRegistration(
      bilinear_form, complex_coefficient, real_coefficient,
      element_marker, boundary_marker);
}

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel complex form templated integrator registration",
          "[ComplexCoefficient][Parallel]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   H1_FECollection fec(1, pmesh.Dimension());
   ParFiniteElementSpace pfes(&pmesh, &fec);
   Array<int> element_marker(pmesh.attributes.Max());
   Array<int> boundary_marker(pmesh.bdr_attributes.Max());
   element_marker = 1;
   boundary_marker = 1;
   ComplexConstantCoefficient complex_coefficient(2.0, -3.0);
   ConstantCoefficient real_coefficient(4.0);

   ParComplexLinearForm linear_form(&pfes);
   CheckTemplatedLinearIntegratorRegistration(
      linear_form, complex_coefficient, real_coefficient,
      element_marker, boundary_marker);

   ParSesquilinearForm bilinear_form(&pfes);
   CheckTemplatedBilinearIntegratorRegistration(
      bilinear_form, complex_coefficient, real_coefficient,
      element_marker, boundary_marker);
}

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
