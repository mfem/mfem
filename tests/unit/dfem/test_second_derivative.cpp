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

#include "../unit_tests.hpp"

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../linalg/test_same_matrices.hpp"

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"
#include "../../../fem/dfem/backends/local_qf/revdiff_transformer.hpp"


using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif


namespace second_derivative_test
{

template <typename dscalar_t, int dim>
struct MinimalSurfaceEnergyFunctional
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const dscalar_t &u,
                   const tensor<dscalar_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &f /* dfdu, dfddudxi */
                  ) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto dx = det(J) * w;
      const auto E = sqrt(1.0_r + sqnorm(dudx));
      f = E * dx;
   }
};

template <typename dscalar_t, int dim>
struct MinimalSurfaceEnergy
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const tensor<dscalar_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &f) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto dx = det(J) * w;
      const auto E = sqrt(1.0_r + sqnorm(dudx));
      f = E * dx;
   }
};

template <typename dscalar_t, int dim>
struct MinimalSurfaceResidual
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const tensor<dscalar_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   tensor<dscalar_t, dim> &dvdx) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto dx = det(J) * w;
      dvdx = dudx / (sqrt(1.0_r + sqnorm(dudx))) * transpose(invJ) * dx;
   }
};

// Hand-coded action of the second derivative of the minimal surface energy,
// i.e. the Hessian-vector product integrand d^2 J(u)[delta_u, v].
template <typename dscalar_t, int dim>
struct MinimalSurfaceHessianAction
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const tensor<real_t, dim> &ddelta_udxi,
                   const tensor<dscalar_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   tensor<real_t, dim> &dvdx) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto ddelta_udx = ddelta_udxi * invJ;
      const auto dx = det(J) * w;
      const auto c = 1.0_r / sqrt(1.0_r + sqnorm(dudx));
      const auto term1 = c * ddelta_udx;
      const auto term2 = c * c * c * dot(dudx, ddelta_udx) * dudx;
      dvdx = (term1 - term2) * transpose(invJ) * dx;
   }
};


// Functional for the mixed problem with two fields u and rho, with the energy functional:
// J(u, rho) = int (rho u^2 + 0.5 rho^2) dx
template <typename dscalar_t, int dim>
struct MixedFunctional
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const dscalar_t &u,
                   const dscalar_t &rho,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &f) const
   {
      f = (rho * u * u + 0.5_r * rho * rho) * det(J) * w;
   }
};

template <typename dscalar_t, int dim>
struct MixedFunctionalUUAction
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const real_t &du,
                   const dscalar_t &rho,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &v) const
   {
      v = 2.0_r * rho * du * det(J) * w;
   }
};

template <typename dscalar_t, int dim>
struct MixedFunctionalURhoAction
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const real_t &drho,
                   const dscalar_t &u,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &v) const
   {
      v = 2.0_r * u * drho * det(J) * w;
   }
};

template <typename dscalar_t, int dim>
struct MixedFunctionalRhoUAction
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const real_t &du,
                   const dscalar_t &u,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &v) const
   {
      v = 2.0_r * u * du * det(J) * w;
   }
};

template <typename dscalar_t, int dim>
struct MixedFunctionalRhoRhoAction
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const real_t &drho,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   real_t &v) const
   {
      v = drho * det(J) * w;
   }
};

template <int dim>
class MyFunctional
{
   static constexpr int U = 0, Coords = 1, Q = 2, DirU = 3;

public:
   MyFunctional(const ParFiniteElementSpace &fes,
                const ParFiniteElementSpace &mfes,
                const IntegrationRule &ir) :
      comm(fes.GetComm()),
      mesh(*mfes.GetParMesh()),
      qspace(*fes.GetParMesh(), ir),
      qspace_vec(qspace, 1),
      q(qspace_vec)
   {

      const auto &pmesh = *fes.GetParMesh();
      Array<int> all_domain_attr;
      if (pmesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(pmesh.attributes.Max());
         all_domain_attr = 1;
      }

      // All three operators below are only ever applied, never assembled.
      constexpr auto kernels = DerivativeKernels::Action;

      // Energy
      {
         const auto in = std::vector
         {
            FieldDescriptor{U, &fes},
            FieldDescriptor{Coords, &mfes}
         };
         const auto out = std::vector
         {
            FieldDescriptor{Q, &qspace_vec}
         };

         functional_dop = std::make_unique<DifferentiableOperator>(in, out, mesh);
         MinimalSurfaceEnergyFunctional<dscalar_t, dim> energy;
         auto derivatives = std::integer_sequence<size_t, U> {};
         auto second_derivatives = SecondDerivatives<DerivativePair<U, U>> {}; // Or equivalently: SecondDerivatives<Pairs::All> {};
         functional_dop->AddDomainIntegrator<LocalQFBackend, kernels>(
            energy,
            Inputs<Value<U>, Gradient<U>, Gradient<Coords>, Weight> {},
            Outputs<FunctionalValue<Q>> {}, /* Value<U>, Gradient<U> */
            ir, all_domain_attr, derivatives, second_derivatives);
      }

      // Manually computed residual
      {
         const auto in = std::vector
         {
            FieldDescriptor{U, &fes},
            FieldDescriptor{Coords, &mfes}
         };
         const auto out = std::vector
         {
            FieldDescriptor{U, &fes}
         };

         residual_dop = std::make_unique<DifferentiableOperator>(in, out, pmesh);
         MinimalSurfaceResidual<dscalar_t, dim> residual;
         auto derivatives = std::integer_sequence<size_t, U> {};
         residual_dop->AddDomainIntegrator<LocalQFBackend, kernels>(
            residual,
            tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Gradient<U>{}},
            ir, all_domain_attr, derivatives);
      }

      // Differentiated energy representing the residual
      {
         const auto in = std::vector
         {
            FieldDescriptor{U, &fes},
            FieldDescriptor{Coords, &mfes}
         };
         const auto out = std::vector
         {
            FieldDescriptor{U, &fes}
         };

         dfunctional_dop = std::make_unique<DifferentiableOperator>(in, out, mesh);
         // Differentiate output f (argument 3) with respect to dudxi
         // (argument 0).
         RevDiff<MinimalSurfaceEnergy<dscalar_t, dim>,
                 tuple<Active, Const, Const>,
                 tuple<Active>,
                 RevDiffDualMode::Derivative>
                 fd;

         auto derivatives = std::integer_sequence<size_t, U> {};
         dfunctional_dop->AddDomainIntegrator<LocalQFBackend, kernels>(
            fd,
            tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Gradient<U>{}},
            ir, all_domain_attr, derivatives);
      }

      // Hand-coded Hessian action with the direction as an explicit field
      {
         const auto in = std::vector
         {
            FieldDescriptor{DirU, &fes},
            FieldDescriptor{U, &fes},
            FieldDescriptor{Coords, &mfes}
         };
         const auto out = std::vector
         {
            FieldDescriptor{U, &fes}
         };

         hessian_dop = std::make_unique<DifferentiableOperator>(in, out, mesh);
         MinimalSurfaceHessianAction<real_t, dim> hessian_action;
         hessian_dop->AddDomainIntegrator<LocalQFBackend>(
            hessian_action,
            tuple{Gradient<DirU>{}, Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Gradient<U>{}},
            ir, all_domain_attr);
      }

      mesh.GetNodes()->GetTrueDofs(coords);
   }

   void gradient_exact(const Vector &u, Vector &g) const
   {
      MultiVector X{u, coords};
      MultiVector Y{g};
      residual_dop->Mult(X, Y);
   }

   void gradient(const Vector &u, Vector &g) const
   {
      MultiVector X{u, coords};
      MultiVector Y{g};
      functional_dop->GetDerivative(U)->Mult(X, Y);
   }

   // Gradient assembled into a Vector from the state captured by
   // GetDerivative, the functional counterpart of Assemble(SparseMatrix *&).
   void gradient_assembled(const Vector &u, Vector &g) const
   {
      MultiVector X{u, coords};
      functional_dop->GetDerivative(U, X)->Assemble(g);
   }

   // Hessian-vector product H(u) v with the hand-coded second derivative.
   void hvp_exact(const Vector &u, const Vector &v, Vector &Hv) const
   {
      MultiVector X{v, u, coords};
      MultiVector Y{Hv};
      hessian_dop->Mult(X, Y);
   }

   // H(u) v as the derivative of the hand-coded residual (single AD).
   void hvp_dresidual(const Vector &u, const Vector &v, Vector &Hv) const
   {
      MultiVector X{u, coords};
      MultiVector Y{Hv};
      residual_dop->GetDerivative(U, X)->Mult(v, Y);
   }

   // H(u) v as the derivative of the differentiated energy
   // (forward-over-reverse AD).
   void hvp(const Vector &u, const Vector &v, Vector &Hv) const
   {
      MultiVector X{u, coords};
      MultiVector Y{Hv};
      dfunctional_dop->GetDerivative(U, X)->Mult(v, Y);
   }

   // H(u) v from the functional's second-derivative interface.
   void hvp_functional(const Vector &u, const Vector &v, Vector &Hv) const
   {
      MultiVector X{u, coords};
      MultiVector Y{Hv};
      functional_dop->GetSecondDerivative(U, X)->Mult(v, Y);
   }

private:
   MPI_Comm comm;
   ParMesh &mesh;
   std::unique_ptr<DifferentiableOperator> functional_dop;
   std::unique_ptr<DifferentiableOperator> dfunctional_dop;
   std::unique_ptr<DifferentiableOperator> residual_dop;
   std::unique_ptr<DifferentiableOperator> hessian_dop;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;
   mutable QuadratureFunction q;
   Vector coords;
};

template <int DIM>
void second_derivative(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   ParGridFunction u_gf(&fes);
   FunctionCoefficient u_coeff(
      [](const auto &x)
   {
      return 2_r * M_PI * x[0] * x[0] * 2_r * M_PI * x[1] * x[1];
   });
   u_gf.ProjectCoefficient(u_coeff);

   Vector u(fes.GetTrueVSize());
   u_gf.GetTrueDofs(u);

   MyFunctional<DIM> functional(fes, *mfes, ir);

   Vector exact_g(fes.GetTrueVSize());
   functional.gradient_exact(u, exact_g);


   Vector g(fes.GetTrueVSize());
   functional.gradient(u, g);

   Vector diff(g);
   diff -= exact_g;
   REQUIRE(diff.Norml2() < 1e-12);

   // The functional derivative assembled into a Vector.
   Vector assembled_g;
   functional.gradient_assembled(u, assembled_g);
   REQUIRE(assembled_g.Size() == fes.GetTrueVSize());

   diff = assembled_g;
   diff -= exact_g;
   REQUIRE(diff.Norml2() < 1e-12);

   // Direction for the Hessian-vector product
   ParGridFunction v_gf(&fes);
   FunctionCoefficient v_coeff(
      [](const auto &x)
   {
      return sin(M_PI * x[0]) * cos(M_PI * x[1]) + 0.5_r * x[0] * x[1];
   });
   v_gf.ProjectCoefficient(v_coeff);

   Vector v(fes.GetTrueVSize());
   v_gf.GetTrueDofs(v);

   Vector exact_Hv(fes.GetTrueVSize());
   functional.hvp_exact(u, v, exact_Hv);

   Vector Hv_dres(fes.GetTrueVSize());
   functional.hvp_dresidual(u, v, Hv_dres);

   diff = Hv_dres;
   diff -= exact_Hv;
   REQUIRE(MFEM_Approx(diff.Norml2()) == 0.0);

   Vector Hv(fes.GetTrueVSize());
   functional.hvp(u, v, Hv);

   diff = Hv;
   diff -= exact_Hv;
   REQUIRE(MFEM_Approx(diff.Norml2()) == 0.0);

   Vector Hv_functional(fes.GetTrueVSize());
   functional.hvp_functional(u, v, Hv_functional);

   diff = Hv_functional;
   diff -= exact_Hv;
   REQUIRE(MFEM_Approx(diff.Norml2()) == 0.0);


   if (verbose_tests)
   {
      mfem::out << "Hessian-vector (functional) product norm: "
                << Hv_functional.Norml2() << std::endl;
      mfem::out << "Hessian-vector (hand-coded residual) product norm: "
                << Hv_dres.Norml2() << std::endl;
      mfem::out << "Hessian-vector (forward over reverse) product norm: "
                << Hv.Norml2() << std::endl;
      mfem::out << "Exact Hessian-vector product norm: "
                << exact_Hv.Norml2() << std::endl;
   }

   // std::cout << "Gradient using FwdDiff<f>:\n";
   // pretty_print(g);

   // std::cout << "Handcoded gradient of f(u):\n";
   // pretty_print(exact_g);

   // std::cout << "Handcoded H(u) v:\n";
   // pretty_print(exact_Hv);

   // std::cout << "H(u) v using the derivative of the handcoded residual:\n";
   // pretty_print(Hv_dres);

   // std::cout << "H(u) v using the derivative of FwdDiff<f>:\n";
   // pretty_print(Hv);
}

template <int DIM>
void mixed_second_derivative(const char *filename, int p)
{
   static constexpr int U = 0, Rho = 1, Coords = 2, Q = 3;
   static constexpr int DU = 4, DRho = 5;
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);
   const int tvsize = fes.GetTrueVSize();

   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   // Use smooth, non-constant fields so all four second-derivative blocks are
   // nontrivial while keeping the exact expressions simple.
   ParGridFunction u_gf(&fes), rho_gf(&fes), du_gf(&fes), drho_gf(&fes);
   FunctionCoefficient u_coeff(
      [](const auto &x)
   {
      return 1.0_r + x[0] + 0.25_r * x[1];
   });
   FunctionCoefficient rho_coeff(
      [](const auto &x)
   {
      return 0.5_r + 0.2_r * x[0] * x[0] + 0.1_r * x[1];
   });
   FunctionCoefficient du_coeff(
      [](const auto &x)
   {
      return cos(M_PI * x[0]) + 0.25_r * x[0] * x[1];
   });
   FunctionCoefficient drho_coeff(
      [](const auto &x)
   {
      return sin(M_PI * x[0]) + 0.5_r * x[1];
   });
   u_gf.ProjectCoefficient(u_coeff);
   rho_gf.ProjectCoefficient(rho_coeff);
   du_gf.ProjectCoefficient(du_coeff);
   drho_gf.ProjectCoefficient(drho_coeff);

   Vector u(tvsize), rho(tvsize), du(tvsize), drho(tvsize), coords;
   u_gf.GetTrueDofs(u);
   rho_gf.GetTrueDofs(rho);
   du_gf.GetTrueDofs(du);
   drho_gf.GetTrueDofs(drho);
   pmesh.GetNodes()->GetTrueDofs(coords);

   const auto functional_in = std::vector
   {
      FieldDescriptor{U, &fes},
      FieldDescriptor{Rho, &fes},
      FieldDescriptor{Coords, mfes}
   };
   QuadratureSpace qspace(pmesh, ir);
   VectorQuadratureSpace qspace_vec(qspace, 1);
   const auto functional_out = std::vector
   {
      FieldDescriptor{Q, &qspace_vec}
   };

   DifferentiableOperator functional_dop(functional_in, functional_out, pmesh);
   MixedFunctional<dscalar_t, DIM> functional;
   constexpr auto kernels =
      DerivativeKernels::Action |
      DerivativeKernels::AssembleMatrix;
   functional_dop.AddDomainIntegrator<LocalQFBackend, kernels>(
      functional,
      Inputs<Value<U>, Value<Rho>, Gradient<Coords>, Weight> {},
      Outputs<FunctionalValue<Q>> {},
      ir, all_domain_attr,
      Derivatives<U, Rho> {},
      SecondDerivatives<Pairs::All> {});

   MultiVector X{u, rho, coords};

   // Every Hessian block of f = \int (rho u^2 + 0.5 rho^2) is a mass matrix,
   // so each one can be assembled with plain MFEM as a reference:
   //
   //    d^2f/du^2      -> 2 rho      d^2f/(du drho) -> 2 u
   //    d^2f/(drho du) -> 2 u        d^2f/drho^2    -> 1
   //
   // The coefficients read the projected grid functions, not the analytic
   // FunctionCoefficients, to match what dFEM evaluates at the quadrature
   // points.
   GridFunctionCoefficient u_gf_coeff(&u_gf), rho_gf_coeff(&rho_gf);
   ProductCoefficient two_u(2.0, u_gf_coeff), two_rho(2.0, rho_gf_coeff);
   ConstantCoefficient one(1.0);

   auto check_block = [&](auto gradient_id,
                          auto direction_id,
                          const Vector &direction,
                          auto exact_qfunc,
                          auto exact_inputs,
                          auto exact_outputs,
                          const std::vector<FieldDescriptor> &exact_in,
                          const std::vector<FieldDescriptor> &exact_out,
                          MultiVector exact_x,
                          Coefficient &hessian_coeff)
   {
      Vector actual(tvsize);
      MultiVector Actual{actual};
      functional_dop.GetSecondDerivative(gradient_id, direction_id, X)->Mult(
         direction, Actual);

      Vector expected(tvsize);
      MultiVector Expected{expected};
      DifferentiableOperator exact_dop(exact_in, exact_out, pmesh);
      exact_dop.AddDomainIntegrator<LocalQFBackend>(
         exact_qfunc, exact_inputs, exact_outputs, ir, all_domain_attr);
      exact_dop.Mult(exact_x, Expected);

      Vector diff(actual);
      diff -= expected;
      REQUIRE(MFEM_Approx(diff.Norml2()) == 0.0);

      HypreParMatrix *actual_mat = nullptr;
      functional_dop.GetSecondDerivative(gradient_id, direction_id, X)->Assemble(
         actual_mat);
      REQUIRE(actual_mat != nullptr);

      // Reference matrix MFEM assembly for the same block
      ParBilinearForm blf(&fes);
      blf.AddDomainIntegrator(new MassIntegrator(hessian_coeff, &ir));
      blf.SetAssemblyLevel(AssemblyLevel::FULL);
      blf.Assemble();
      blf.Finalize();
      HypreParMatrix *expected_mat = blf.ParallelAssemble();
      REQUIRE(expected_mat != nullptr);

      // TestSameMatrices only walks the first matrix' sparsity pattern and
      // reads a missing entry of the second as 0, so compare both ways to
      // cover the union of the two patterns. A structurally deficient
      // actual_mat is only visible from the reference side.
      REQUIRE(actual_mat->Width() == expected_mat->Width());
      TestSameMatrices(*actual_mat, *expected_mat);
      TestSameMatrices(*expected_mat, *actual_mat);

      delete actual_mat;
      delete expected_mat;
   };

   check_block(U, U, du,
               MixedFunctionalUUAction<real_t, DIM> {},
               Inputs<Value<DU>, Value<Rho>, Gradient<Coords>, Weight> {},
               Outputs<Value<U>> {},
               std::vector{FieldDescriptor{DU, &fes},
                           FieldDescriptor{Rho, &fes},
                           FieldDescriptor{Coords, mfes}},
               std::vector{FieldDescriptor{U, &fes}},
               MultiVector{du, rho, coords},
               two_rho);

   check_block(U, Rho, drho,
               MixedFunctionalURhoAction<real_t, DIM> {},
               Inputs<Value<DRho>, Value<U>, Gradient<Coords>, Weight> {},
               Outputs<Value<U>> {},
               std::vector{FieldDescriptor{DRho, &fes},
                           FieldDescriptor{U, &fes},
                           FieldDescriptor{Coords, mfes}},
               std::vector{FieldDescriptor{U, &fes}},
               MultiVector{drho, u, coords},
               two_u);

   check_block(Rho, U, du,
               MixedFunctionalRhoUAction<real_t, DIM> {},
               Inputs<Value<DU>, Value<U>, Gradient<Coords>, Weight> {},
               Outputs<Value<Rho>> {},
               std::vector{FieldDescriptor{DU, &fes},
                           FieldDescriptor{U, &fes},
                           FieldDescriptor{Coords, mfes}},
               std::vector{FieldDescriptor{Rho, &fes}},
               MultiVector{du, u, coords},
               two_u);

   check_block(Rho, Rho, drho,
               MixedFunctionalRhoRhoAction<real_t, DIM> {},
               Inputs<Value<DRho>, Gradient<Coords>, Weight> {},
               Outputs<Value<Rho>> {},
               std::vector{FieldDescriptor{DRho, &fes},
                           FieldDescriptor{Coords, mfes}},
               std::vector{FieldDescriptor{Rho, &fes}},
               MultiVector{drho, coords},
               one);
}

// Registers the mixed functional with the given second derivative request and
// reports which of the four Hessian blocks became available, in the order
// (U,U), (U,Rho), (Rho,U), (Rho,Rho).
template <typename second_derivatives_t>
std::array<bool, 4> registered_blocks(second_derivatives_t second_derivatives)
{
   static constexpr int DIM = 2;
   static constexpr int U = 0, Rho = 1, Coords = 2, Q = 3;

   Mesh smesh("../../data/inline-quad.mesh");
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   H1_FECollection fec(1, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);
   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2);

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   QuadratureSpace qspace(pmesh, ir);
   VectorQuadratureSpace qspace_vec(qspace, 1);

   DifferentiableOperator dop(
      std::vector{FieldDescriptor{U, &fes},
                  FieldDescriptor{Rho, &fes},
                  FieldDescriptor{Coords, mfes}},
      std::vector{FieldDescriptor{Q, &qspace_vec}}, pmesh);

   MixedFunctional<dscalar_t, DIM> functional;
   constexpr auto kernels = DerivativeKernels::Action;
   dop.AddDomainIntegrator<LocalQFBackend, kernels>(
      functional,
      Inputs<Value<U>, Value<Rho>, Gradient<Coords>, Weight> {},
      Outputs<FunctionalValue<Q>> {},
      ir, all_domain_attr,
      Derivatives<U, Rho> {},
      second_derivatives);

   return
   {
      dop.HasSecondDerivative(U, U), dop.HasSecondDerivative(U, Rho),
      dop.HasSecondDerivative(Rho, U), dop.HasSecondDerivative(Rho, Rho)
   };
}

} // namespace second_derivative_test

TEST_CASE("dFEM functional second derivative registration",
          "[Parallel][dFEM][second-derivative]")
{
   using namespace second_derivative_test;
   static constexpr int U = 0, Rho = 1;

   // (U,U), (U,Rho), (Rho,U), (Rho,Rho)
   using blocks_t = std::array<bool, 4>;

   SECTION("none")
   {
      REQUIRE(registered_blocks(SecondDerivatives<Pairs::None> {}) ==
              blocks_t{false, false, false, false});
   }

   SECTION("all")
   {
      REQUIRE(registered_blocks(SecondDerivatives<Pairs::All> {}) ==
              blocks_t{true, true, true, true});
   }

   SECTION("diagonal")
   {
      REQUIRE(registered_blocks(SecondDerivatives<Pairs::Diagonal> {}) ==
              blocks_t{true, false, false, true});
   }

   SECTION("custom")
   {
      // One Hessian block and one mixed block.
      REQUIRE(registered_blocks(
                 SecondDerivatives<DerivativePair<U, U>,
                 DerivativePair<Rho, U>> {}) ==
              blocks_t{true, false, true, false});
   }
}

TEST_CASE("dFEM functional second derivative action matches mfem",
          "[Parallel][dFEM][second-derivative][GPU]")
{
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);

   SECTION("2d")
   {
      const auto f =
         GENERATE(
            // "../../data/star.mesh",
            // "../../data/star-q3.mesh",
            // "../../data/rt-2d-q3.mesh",
            "../../data/inline-quad.mesh"
            // "../../data/periodic-square.mesh"
         );
      second_derivative_test::second_derivative<2>(f, p);
   }

   // SECTION("3d")
   // {
   //    const auto f =
   //       GENERATE(
   //          "../../data/fichera-q3.mesh",
   //          "../../data/inline-hex.mesh",
   //          "../../data/toroid-hex.mesh",
   //          "../../data/periodic-cube.mesh"
   //       );
   //    second_derivative_test::second_derivative<3>(f, p);
   // }
}

TEST_CASE("dFEM functional mixed second derivative action matches exact action",
          "[Parallel][dFEM][second-derivative][GPU]")
{
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);

   SECTION("2d")
   {
      const auto f = GENERATE("../../data/inline-quad.mesh");
      second_derivative_test::mixed_second_derivative<2>(f, p);
   }
}

#endif // MFEM_USE_MPI
