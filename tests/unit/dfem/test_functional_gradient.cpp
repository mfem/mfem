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

#ifdef MFEM_USE_ENZYME
#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

namespace functional_gradient_test
{

// Dirichlet-type energy 1/2 \int |grad u|^2 dx. It depends on the solution and
// on the mesh coordinates, so both derivatives are nontrivial. The coordinate
// field enters only through Gradient<Coords>, while the solution enters
// through both Value<U> and Gradient<U>, so the two derivatives have a
// different number of outputs.
template <int dim>
struct EnergyFunctional
{
   MFEM_HOST_DEVICE inline MFEM_FUTURE_ALWAYS_INLINE
   auto operator()(const real_t &u,
                   const tensor<real_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   real_t &f) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      f = (0.5_r * sqnorm(dudx) + 0.25_r * u * u) * det(J) * w;
   }
};

/// Energy functional with derivatives w.r.t. both the solution and the mesh
/// coordinates, plus a finite-difference reference for either one.
template <int dim>
class EnergyWithShapeDerivative
{
   static constexpr int U = 0, Coords = 1, Q = 2;

public:
   EnergyWithShapeDerivative(const ParFiniteElementSpace &fes,
                             const ParFiniteElementSpace &mfes,
                             const IntegrationRule &ir) :
      comm(fes.GetComm()),
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

      const auto in = std::vector
      {
         FieldDescriptor{U, &fes},
         FieldDescriptor{Coords, &mfes}
      };
      const auto out = std::vector
      {
         FieldDescriptor{Q, &qspace_vec}
      };

      dop = std::make_unique<DifferentiableOperator>(in, out, pmesh);
      EnergyFunctional<dim> energy;
      dop->AddDomainIntegrator<LocalQFBackend>(
         energy,
         Inputs<Value<U>, Gradient<U>, Gradient<Coords>, Weight> {},
         Outputs<FunctionalValue<Q>> {},
         ir, all_domain_attr, Derivatives<U, Coords> {});
   }

   real_t Eval(const Vector &u, const Vector &x) const
   {
      MultiVector X{u, x};
      MultiVector Y{q};
      dop->Mult(X, Y);
      const real_t local = q.Sum();
      real_t global;
      MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    comm);
      return global;
   }

   /// Gradient w.r.t. @a derivative_id, assembled into a Vector.
   void AssembleGradient(size_t derivative_id, const Vector &u,
                         const Vector &x, Vector &g) const
   {
      MultiVector X{u, x};
      dop->GetDerivative(derivative_id, X)->Assemble(g);
   }

   /// Central-difference directional derivative of the energy.
   real_t DirectionalFD(const Vector &u, const Vector &x,
                        const Vector &du, const Vector &dx,
                        real_t eps) const
   {
      Vector up(u), um(u), xp(x), xm(x);
      up.Add(eps, du);
      um.Add(-eps, du);
      xp.Add(eps, dx);
      xm.Add(-eps, dx);
      return (Eval(up, xp) - Eval(um, xm)) / (2.0_r * eps);
   }

   /// Global inner product of two T-vectors (true dofs are uniquely owned).
   real_t Dot(const Vector &a, const Vector &b) const
   {
      const real_t local = a * b;
      real_t global;
      MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    comm);
      return global;
   }

private:
   MPI_Comm comm;
   std::unique_ptr<DifferentiableOperator> dop;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;
   mutable QuadratureFunction q;
};

template <int DIM>
void functional_gradient(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   static constexpr int U = 0, Coords = 1;

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p + 2);

   EnergyWithShapeDerivative<DIM> energy(fes, *mfes, ir);

   ParGridFunction u_gf(&fes), du_gf(&fes);
   FunctionCoefficient u_coeff(
   [](const auto &x) { return 1.0_r + x[0] + 0.25_r * x[1] * x[1]; });
   FunctionCoefficient du_coeff(
   [](const auto &x) { return sin(M_PI * x[0]) + 0.5_r * x[0] * x[1]; });
   u_gf.ProjectCoefficient(u_coeff);
   du_gf.ProjectCoefficient(du_coeff);

   ParGridFunction dx_gf(mfes);
   VectorFunctionCoefficient dx_coeff(
      DIM, [](const Vector &x, Vector &v)
   {
      v = 0.0;
      v(0) = 0.1_r * sin(M_PI * x[0]) * x[1];
      v(1) = 0.1_r * cos(M_PI * x[1]) * x[0];
   });
   dx_gf.ProjectCoefficient(dx_coeff);

   Vector u(fes.GetTrueVSize()), du(fes.GetTrueVSize());
   Vector x(mfes->GetTrueVSize()), dx(mfes->GetTrueVSize());
   u_gf.GetTrueDofs(u);
   du_gf.GetTrueDofs(du);
   nodes->GetTrueDofs(x);
   dx_gf.GetTrueDofs(dx);

   Vector zero_u(fes.GetTrueVSize()), zero_x(mfes->GetTrueVSize());
   zero_u = 0.0;
   zero_x = 0.0;

   const real_t eps = 1e-6;

   // Derivative w.r.t. the solution.
   Vector g_u;
   energy.AssembleGradient(U, u, x, g_u);
   REQUIRE(g_u.Size() == fes.GetTrueVSize());

   const real_t dJdu_fd = energy.DirectionalFD(u, x, du, zero_x, eps);
   REQUIRE(energy.Dot(g_u, du) == MFEM_Approx(dJdu_fd, 1e-6, 1e-8));

   // Derivative w.r.t. the mesh coordinates. This is the case that used to
   // assemble into the solution space instead of the coordinate space.
   Vector g_x;
   energy.AssembleGradient(Coords, u, x, g_x);
   REQUIRE(g_x.Size() == mfes->GetTrueVSize());

   const real_t dJdx_fd = energy.DirectionalFD(u, x, zero_u, dx, eps);
   REQUIRE(energy.Dot(g_x, dx) == MFEM_Approx(dJdx_fd, 1e-6, 1e-8));
}

} // namespace functional_gradient_test

TEST_CASE("dFEM functional gradient assembles into a Vector",
          "[Parallel][dFEM][functional-gradient]")
{
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);

   SECTION("2d")
   {
      const auto f = GENERATE("../../data/inline-quad.mesh",
                              "../../data/star.mesh");
      functional_gradient_test::functional_gradient<2>(f, p);
   }

   SECTION("3d")
   {
      const auto f = GENERATE("../../data/inline-hex.mesh");
      functional_gradient_test::functional_gradient<3>(f, p);
   }
}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_ENZYME
