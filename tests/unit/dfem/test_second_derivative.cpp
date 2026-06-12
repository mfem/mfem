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
#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"
#include "../../../fem/dfem/backends/local_qf/fwddiff_transformer.hpp"

#ifdef MFEM_USE_MPI

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
struct MinimalSurfaceEnergy
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<dscalar_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   real_t &f) const
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
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<dscalar_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   tensor<real_t, dim> &dvdx) const
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
   MFEM_HOST_DEVICE inline
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

      const auto &mesh = *fes.GetParMesh();
      Array<int> all_domain_attr;
      if (mesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(mesh.attributes.Max());
         all_domain_attr = 1;
      }

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
         MinimalSurfaceEnergy<real_t, dim> energy;
         auto derivatives = std::integer_sequence<size_t, U> {};
         functional_dop->AddDomainIntegrator<LocalQFBackend>(
            energy,
            tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Identity<Q>{}},
            ir, all_domain_attr, derivatives);
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

         residual_dop = std::make_unique<DifferentiableOperator>(in, out, mesh);
         MinimalSurfaceResidual<real_t, dim> residual;
         auto derivatives = std::integer_sequence<size_t, U> {};
         residual_dop->AddDomainIntegrator<LocalQFBackend>(
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
         FwdDiff<MinimalSurfaceEnergy<real_t, dim>, 0, 3> fd;
         auto derivatives = std::integer_sequence<size_t, U> {};
         dfunctional_dop->AddDomainIntegrator<LocalQFBackend>(
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
      dfunctional_dop->Mult(X, Y);
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

   // H(u) v as the derivative of the differentiated energy FwdDiff<f>
   // (forward-over-forward AD).
   void hvp(const Vector &u, const Vector &v, Vector &Hv) const
   {
      MultiVector X{u, coords};
      MultiVector Y{Hv};
      dfunctional_dop->GetDerivative(U, X)->Mult(v, Y);
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

} // namespace second_derivative_test

TEST_CASE("dFEM functional second derivative action matches mfem",
          "[Parallel][dFEM][second-derivative]")
{
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);

   SECTION("2d")
   {
      const auto f =
         GENERATE(
            "../../data/star.mesh",
            "../../data/star-q3.mesh",
            "../../data/rt-2d-q3.mesh",
            "../../data/inline-quad.mesh",
            "../../data/periodic-square.mesh"
         );
      second_derivative_test::second_derivative<2>(f, p);
   }

   SECTION("3d")
   {
      const auto f =
         GENERATE(
            "../../data/fichera.mesh",
            "../../data/fichera-q3.mesh",
            "../../data/inline-hex.mesh",
            "../../data/toroid-hex.mesh",
            "../../data/periodic-cube.mesh"
         );
      second_derivative_test::second_derivative<3>(f, p);
   }
}

#endif // MFEM_USE_MPI
