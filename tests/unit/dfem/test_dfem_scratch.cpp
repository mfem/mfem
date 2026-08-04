// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.

// Multi-kernel scratch regression for the weak residual
//
//   F(u)_i = int_Omega phi_i c u^3 dx,
//
// evaluated as the scratch chain s = u, s = s*u, y = c*s*u.  The directional
// derivative is
//
//   DF(u)[du]_i = int_Omega phi_i 3 c u^2 du dx.
//
// If the tangent stored in the qfunction scratch shadow is lost between the
// split scratch updates, the final product only sees the direct derivative of
// the last factor and produces int_Omega phi_i c u^2 du dx instead.  This test
// checks both the direct DerivativeAction path and the cached
// DerivativeSetup+DerivativeApply path against the factor-of-3 result.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Test for the dFEM global qfunction with split computation and scratch space.

#include "../unit_tests.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/global_qf/prelude.hpp"

#include "../../../linalg/tensor_arrays.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::future;

using dscalar_t = real_t;

///<--- Q-functions
constexpr int U = 1;
constexpr int Y = 2;
constexpr int COEF = 3;
constexpr int COORDINATES = 4;

// Global qf with splitting and scratch space.
// The user only writes operator(); the shared base handles scratch setup.
struct CubicQFWithScratch : QFWithScratchType
{
   void operator()(tensor_array<const dscalar_t> &x,
                   tensor_array<const dscalar_t> &coef,
                   tensor_array<const real_t, 2, 2> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &y) const
   {
      const int NQ = nq;
      MFEM_ASSERT(NQ == static_cast<int>(x.size()),
                  "unexpected number of quadrature points");

      auto scratch_q = make_tensor_array<>(scratch[0], NQ);

      for (int q = 0; q < NQ; ++q)
      {
         scratch_q(q) = x(q);
      }

      for (int q = 0; q < NQ; ++q)
      {
         scratch_q(q) = scratch_q(q) * x(q);
      }

      for (int q = 0; q < NQ; ++q)
      {
         y(q) = coef(q) * scratch_q(q) * x(q) * det(J(q)) * w(q);
      }
   }
};

template <int DIM>
struct CubicQFWithScratchMultipleSizes : QFWithScratchType
{
   void operator()(tensor_array<const dscalar_t> &x,
                   tensor_array<const dscalar_t> &coef,
                   tensor_array<const real_t, 2, 2> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &y) const
   {
      const int NQ = nq;
      MFEM_ASSERT(NQ == static_cast<int>(x.size()),
                  "unexpected number of quadrature points");

      auto scratch_scalar = make_tensor_array<>(scratch[0], NQ);
      auto scratch_vector = make_tensor_array<DIM>(scratch[1], NQ);

      for (int q = 0; q < NQ; ++q)
      {
         scratch_vector(q)(0) = x(q);
         scratch_vector(q)(1) = x(q) * x(q);
      }

      for (int q = 0; q < NQ; ++q)
      {
         scratch_scalar(q) = scratch_vector(q)(0) * scratch_vector(q)(1);
      }

      for (int q = 0; q < NQ; ++q)
      {
         y(q) = coef(q) * scratch_scalar(q) * det(J(q)) * w(q);
      }
   }
};

struct CubicQFWithGlobalScratch : QFWithGlobalScratchType
{
   void operator()(tensor_array<const dscalar_t> &x,
                   tensor_array<const dscalar_t> &coef,
                   tensor_array<const real_t, 2, 2> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &y) const
   {
      const int NQ = nq;
      MFEM_ASSERT(NQ == static_cast<int>(x.size()),
                  "unexpected number of quadrature points");

      // Unpack the scratch vectors from the scratch bank
      auto scratch_q = make_tensor_array<>(GetScratchPointer(0), NQ);

      // Unpack global scratch
      auto &has_scale = GetGlobalScratch<0>();
      const auto scale = GetGlobalScratch<1>();
      auto &global_vector = GetGlobalScratch<2>();

      has_scale = global_vector.Size() >
                  0;  // If the global vector is non-empty, we will use it to scale the output
      if (has_scale)
      {
         global_vector(0) = 1.0;
      }

      for (int q = 0; q < NQ; ++q)
      {
         scratch_q(q) = x(q) * x(q);
      }

      for (int q = 0; q < NQ; ++q)
      {
         const real_t global_scale = has_scale ? scale * global_vector(0) : 0.0;
         y(q) = global_scale * coef(q) * scratch_q(q) * x(q) * det(J(q)) * w(q);
      }
   }
};

///<--- Utils

/// @param fes
/// @param x
void FillInput(ParFiniteElementSpace &fes, Coefficient &input_coeff, Vector &x)
{
   GridFunction x_gf(&fes);
   x_gf.ProjectCoefficient(input_coeff);
   x_gf.GetTrueDofs(x);
}

void FillQData(FiniteElementSpace &fes, const IntegrationRule &ir,
               Coefficient &coeff_fc, QuadratureFunction &coef)
{
   QuadratureSpace qspace(*fes.GetMesh(), ir);
   QuadratureFunction coef_qf(qspace);
   coeff_fc.Project(coef_qf);
   coef = coef_qf;
}

void CheckResults(ParFiniteElementSpace &fes, const IntegrationRule &ir,
                  Vector &y, Vector &dy)
{
   FunctionCoefficient expected_coeff([](const Vector &p)
   {
      const real_t input = 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0);
      const real_t coeff = 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0);
      return coeff * input * input * input;
   });

   ParLinearForm expected_lf(&fes);
   expected_lf.AddDomainIntegrator(new DomainLFIntegrator(expected_coeff, &ir));
   expected_lf.Assemble();

   Vector expected_y(fes.GetTrueVSize());
   fes.GetProlongationMatrix()->MultTranspose(expected_lf, expected_y);

   y -= expected_y;
   const real_t local_err = y.Normlinf();
   real_t global_err = 0.0;
   MPI_Allreduce(&local_err, &global_err, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, MPI_COMM_WORLD);

   FunctionCoefficient expected_deriv_coeff([](const Vector &p)
   {
      const real_t input = 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0);
      const real_t coeff = 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0);
      return 3.0 * coeff * input * input;
   });

   ParLinearForm expected_deriv_lf(&fes);
   expected_deriv_lf.AddDomainIntegrator(new DomainLFIntegrator(
                                            expected_deriv_coeff, &ir));
   expected_deriv_lf.Assemble();

   Vector expected_dy(fes.GetTrueVSize());
   fes.GetProlongationMatrix()->MultTranspose(expected_deriv_lf, expected_dy);

   dy -= expected_dy;
   const real_t local_deriv_err = dy.Normlinf();
   real_t global_deriv_err = 0.0;
   MPI_Allreduce(&local_deriv_err, &global_deriv_err, 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, MPI_COMM_WORLD);

   if (verbose_tests && Mpi::Root())
   {
      mfem::out << "Primal output max error: " << global_err << endl;
      mfem::out << "Derivative output max error: "
                << global_deriv_err << endl;
   }

   REQUIRE(global_err == MFEM_Approx(0.0));
   REQUIRE(global_deriv_err == MFEM_Approx(0.0));
}

void CheckScratchResults(ParMesh &pmesh, const IntegrationRule &ir,
                         const Vector &scratch, const Vector &scratch_d)
{
   REQUIRE(scratch.Size() == scratch_d.Size());
   REQUIRE(scratch.Size() == pmesh.GetNE() * ir.GetNPoints());

   const real_t *scratch_h = scratch.HostRead();
   const real_t *scratch_d_h = scratch_d.HostRead();

   real_t local_scratch_err = 0.0;
   real_t local_scratch_d_err = 0.0;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      ElementTransformation *T = pmesh.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         Vector p;
         T->Transform(ip, p);
         const real_t u_q = 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0);
         const int idx = q + ir.GetNPoints() * e;
         local_scratch_err = std::max(local_scratch_err,
                                      std::abs(scratch_h[idx] - u_q * u_q));
         local_scratch_d_err = std::max(local_scratch_d_err,
                                        std::abs(scratch_d_h[idx] - 2.0 * u_q));
      }
   }

   real_t global_scratch_err = 0.0;
   real_t global_scratch_d_err = 0.0;
   MPI_Allreduce(&local_scratch_err, &global_scratch_err, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(&local_scratch_d_err, &global_scratch_d_err, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);

   if (verbose_tests && Mpi::Root())
   {
      mfem::out << "Scratch max error: " << global_scratch_err << endl;
      mfem::out << "Scratch derivative max error: "
                << global_scratch_d_err << endl;
   }

   REQUIRE(global_scratch_err == MFEM_Approx(0.0));
   REQUIRE(global_scratch_d_err == MFEM_Approx(0.0));
}

///<--- Test
TEST_CASE("dFEM Scratch scalar", "[Parallel][dFEM][Scratch-Scalar]")
{
   int order = 2;
   int ref_levels = 1;

   ///<--- Mesh and finite element space setup
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0,
                                     1.0);
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.EnsureNodes();
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, &fec);
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *nodes_fes = nodes->ParFESpace();
   Vector nodes_tvec;
   nodes->GetTrueDofs(nodes_tvec);

   ///<--- dFEM setup
   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
   QuadratureSpace qspace(pmesh, ir);
   VectorQuadratureSpace coef_qspace(qspace, 1);
   QuadratureFunction coef(coef_qspace);
   coef.UseDevice(true);
   FunctionCoefficient coeff_fc([](const Vector &p)
   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
   FillQData(fes, ir, coeff_fc, coef);

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;

   const std::vector<FieldDescriptor> inputs
   {
      {U, &fes},
      {COEF, &coef_qspace},
      {COORDINATES, nodes_fes}};
   const std::vector<FieldDescriptor> outputs
   {
      {Y, &fes}};
   DifferentiableOperator dop(inputs, outputs, pmesh);

   // Define the cubic qfunction with scratch space
   // Requesting one scalar scratch vector
   // Equivalent to
   // cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), 1, 1);

   CubicQFWithScratch cubic_qf;
   cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), {1});
   dop.AddDomainIntegrator<GlobalQFBackend>(
      cubic_qf,
      Inputs<Value<U>, Identity<COEF>, Gradient<COORDINATES>, Weight> {},
      Outputs<Value<Y>> {},
      ir, all_domain_attr,
      Derivatives<U> {});


   Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize()), dx(fes.GetTrueVSize()),
          dy(fes.GetTrueVSize());
   x.UseDevice(true);
   y.UseDevice(true);
   dx.UseDevice(true);
   dy.UseDevice(true);
   FunctionCoefficient input_coeff([](const Vector &p)
   { return 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0); });
   FillInput(fes, input_coeff, x);
   ConstantCoefficient direction_coeff(1.0);
   FillInput(fes, direction_coeff, dx);
   y = 0.0;
   dy = 0.0;

   ///<--- Apply the operator
   MultiVector X{x, coef, nodes_tvec};
   MultiVector Y{y};
   dop.Mult(X, Y);

   //<--- Apply derivative operator
   auto dop_deriv = dop.GetDerivative(U, X);
   MultiVector dY{dy};
   dop_deriv->Mult(dx, dY);

   ///<--- Check the result against the expected output
   CheckResults(fes, ir, y, dy);
}


TEST_CASE("dFEM Scratch multiple sizes",
          "[Parallel][dFEM][Scratch-Multiple-Sizes]")
{
   int order = 2;
   int ref_levels = 1;
   int DIM = 2;

   ///<--- Mesh and finite element space setup
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0,
                                     1.0);
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.EnsureNodes();
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, &fec);
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *nodes_fes = nodes->ParFESpace();
   Vector nodes_tvec;
   nodes->GetTrueDofs(nodes_tvec);

   ///<--- dFEM setup
   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
   QuadratureSpace qspace(pmesh, ir);
   VectorQuadratureSpace coef_qspace(qspace, 1);
   QuadratureFunction coef(coef_qspace);
   coef.UseDevice(true);
   FunctionCoefficient coeff_fc([](const Vector &p)
   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
   FillQData(fes, ir, coeff_fc, coef);

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;

   const std::vector<FieldDescriptor> inputs
   {
      {U, &fes},
      {COEF, &coef_qspace},
      {COORDINATES, nodes_fes}};
   const std::vector<FieldDescriptor> outputs
   {
      {Y, &fes}};
   DifferentiableOperator dop(inputs, outputs, pmesh);

   // Define the cubic qfunction with scratch space
   // Requesting one scalar scratch vector per dimension
   CubicQFWithScratch cubic_qf;
   cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), {1, DIM});
   dop.AddDomainIntegrator<GlobalQFBackend>(
      cubic_qf,
      Inputs<Value<U>, Identity<COEF>, Gradient<COORDINATES>, Weight> {},
      Outputs<Value<Y>> {},
      ir, all_domain_attr,
      Derivatives<U> {});


   Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize()), dx(fes.GetTrueVSize()),
          dy(fes.GetTrueVSize());
   x.UseDevice(true);
   y.UseDevice(true);
   dx.UseDevice(true);
   dy.UseDevice(true);
   FunctionCoefficient input_coeff([](const Vector &p)
   { return 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0); });
   FillInput(fes, input_coeff, x);
   ConstantCoefficient direction_coeff(1.0);
   FillInput(fes, direction_coeff, dx);
   y = 0.0;
   dy = 0.0;

   ///<--- Apply the operator
   MultiVector X{x, coef, nodes_tvec};
   MultiVector Y{y};
   dop.Mult(X, Y);

   //<--- Apply derivative operator
   auto dop_deriv = dop.GetDerivative(U, X);
   MultiVector dY{dy};
   dop_deriv->Mult(dx, dY);

   ///<--- Check the result against the expected output
   CheckResults(fes, ir, y, dy);
}

TEST_CASE("dFEM Global Scratch with tuple objects",
          "[Parallel][dFEM][GlobalScratch]")
{
   int order = 2;
   int ref_levels = 1;

   ///<--- Mesh and finite element space setup
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0,
                                     1.0);
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.EnsureNodes();
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, &fec);
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *nodes_fes = nodes->ParFESpace();
   Vector nodes_tvec;
   nodes->GetTrueDofs(nodes_tvec);

   ///<--- dFEM setup
   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
   QuadratureSpace qspace(pmesh, ir);
   VectorQuadratureSpace coef_qspace(qspace, 1);
   QuadratureFunction coef(coef_qspace);
   coef.UseDevice(true);
   FunctionCoefficient coeff_fc([](const Vector &p)
   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
   FillQData(fes, ir, coeff_fc, coef);

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;

   const std::vector<FieldDescriptor> inputs
   {
      {U, &fes},
      {COEF, &coef_qspace},
      {COORDINATES, nodes_fes}};
   const std::vector<FieldDescriptor> outputs
   {
      {Y, &fes}};
   DifferentiableOperator dop(inputs, outputs, pmesh);

   Vector global_vec(1);
   global_vec.UseDevice(true);
   global_vec = 0.0;
   real_t global_scalar = 1.0;
   bool global_flag;

   CubicQFWithGlobalScratch cubic_qf;
   cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), {1});
   cubic_qf.SetGlobalScratch(
      mfem::future::make_tuple(global_flag, global_scalar, global_vec));
   dop.AddDomainIntegrator<GlobalQFBackend>(
      cubic_qf,
      Inputs<Value<U>, Identity<COEF>, Gradient<COORDINATES>, Weight> {},
      Outputs<Value<Y>> {},
      ir, all_domain_attr,
      Derivatives<U> {});

   Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize()), dx(fes.GetTrueVSize()),
          dy(fes.GetTrueVSize());
   x.UseDevice(true);
   y.UseDevice(true);
   dx.UseDevice(true);
   dy.UseDevice(true);
   FunctionCoefficient input_coeff([](const Vector &p)
   { return 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0); });
   FillInput(fes, input_coeff, x);
   ConstantCoefficient direction_coeff(1.0);
   FillInput(fes, direction_coeff, dx);
   y = 0.0;
   dy = 0.0;

   ///<--- Apply the operator
   MultiVector X{x, coef, nodes_tvec};
   MultiVector Y{y};
   dop.Mult(X, Y);

   //<--- Apply derivative operator
   auto dop_deriv = dop.GetDerivative(U, X);
   MultiVector dY{dy};
   dop_deriv->Mult(dx, dY);

   ///<--- Check the result against the expected output
   CheckResults(fes, ir, y, dy);
}


TEST_CASE("dFEM Scratch multi-kernel persists tangents",
          "[Parallel][dFEM][Scratch-MultiKernel]")
{
   // This test checks that the tangent stored in the qfunction scratch shadow is preserved between split scratch updates, and persists to the final product.
   // With the old implementation, the tangent of the temporary scratch vector was used internally but lost after the derivative action.

   int order = 2;
   int ref_levels = 1;

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0,
                                     1.0);
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   pmesh.EnsureNodes();
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fes(&pmesh, &fec);
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *nodes_fes = nodes->ParFESpace();
   Vector nodes_tvec;
   nodes->GetTrueDofs(nodes_tvec);

   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
   QuadratureSpace qspace(pmesh, ir);
   VectorQuadratureSpace coef_qspace(qspace, 1);
   QuadratureFunction coef(coef_qspace);
   coef.UseDevice(true);
   FunctionCoefficient coeff_fc([](const Vector &p)
   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
   FillQData(fes, ir, coeff_fc, coef);

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;

   const std::vector<FieldDescriptor> inputs
   {
      {U, &fes},
      {COEF, &coef_qspace},
      {COORDINATES, nodes_fes}};
   const std::vector<FieldDescriptor> outputs
   {
      {Y, &fes}};
   DifferentiableOperator dop(inputs, outputs, pmesh);

   CubicQFWithScratch cubic_qf;
   cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), {1});
   dop.AddDomainIntegrator<GlobalQFBackend>(
      cubic_qf,
      Inputs<Value<U>, Identity<COEF>, Gradient<COORDINATES>, Weight> {},
      Outputs<Value<Y>> {},
      ir, all_domain_attr,
      Derivatives<U> {});

   Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize()), dx(fes.GetTrueVSize());
   Vector dy_action(fes.GetTrueVSize()), dy_cached(fes.GetTrueVSize());
   x.UseDevice(true);
   y.UseDevice(true);
   dx.UseDevice(true);
   dy_action.UseDevice(true);
   dy_cached.UseDevice(true);
   FunctionCoefficient input_coeff([](const Vector &p)
   { return 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0); });
   FillInput(fes, input_coeff, x);
   ConstantCoefficient direction_coeff(1.0);
   FillInput(fes, direction_coeff, dx);
   y = 0.0;
   dy_action = 0.0;
   dy_cached = 0.0;

   // DifferentiableOperator action
   MultiVector X{x, coef, nodes_tvec};
   MultiVector Y{y};
   dop.Mult(X, Y);

   // Derivative action (non-cached)
   auto dop_deriv_action = dop.GetDerivative(U, X, false);
   MultiVector dY_action{dy_action};
   dop_deriv_action->Mult(dx, dY_action);
   Vector y_action_check(y);
   CheckResults(fes, ir, y_action_check, dy_action);

   auto *stored_qf = dop.GetDerivativeActionQFunction<CubicQFWithScratch>(U);
   auto *stored_qf_shadow =
      dop.GetDerivativeActionShadowQFunction<CubicQFWithScratch>(U);
   REQUIRE(stored_qf != nullptr);
   REQUIRE(stored_qf_shadow != nullptr);

   CheckScratchResults(pmesh, ir, stored_qf->GetScratchVector(0),
                       stored_qf_shadow->GetScratchVector(0));

   // Derivative action (cached)
   auto dop_deriv_cached = dop.GetDerivative(U, X, true);
   MultiVector dY_cached{dy_cached};
   dop_deriv_cached->Mult(dx, dY_cached);

   Vector y_cached_check(y);
   CheckResults(fes, ir, y_cached_check, dy_cached);

   auto *stored_setup_qf = dop.GetDerivativeSetupQFunction<CubicQFWithScratch>(U);
   auto *stored_setup_qf_shadow =
      dop.GetDerivativeSetupShadowQFunction<CubicQFWithScratch>(U);
   REQUIRE(stored_setup_qf != nullptr);
   REQUIRE(stored_setup_qf_shadow != nullptr);
   CheckScratchResults(pmesh, ir, stored_setup_qf->GetScratchVector(0),
                       stored_setup_qf_shadow->GetScratchVector(0));
}

#endif // MFEM_USE_MPI