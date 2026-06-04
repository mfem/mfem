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

template <typename dscalar_t, int dim>
struct CubicH1Functional
{
   static constexpr real_t alpha = 0.3;
   static constexpr real_t beta  = 0.7;

   MFEM_HOST_DEVICE inline
   auto operator()(const dscalar_t &u,
                   const tensor<dscalar_t, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &f) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto dx = det(J) * w;

      const auto z =
         + 0.5_r * u * u
         + 0.25_r * alpha * u * u * u * u
         + 0.5_r * beta * dot(dudx, dudx);

      f = z * dx;
   }
};

template <int dim>
class MyFunctional
{
   static constexpr int U = 0, Coords = 1, Q = 2;

public:
   MyFunctional(const ParFiniteElementSpace &fes,
                const ParFiniteElementSpace &mfes,
                const IntegrationRule &ir) :
      comm(fes.GetComm()),
      qspace(*fes.GetParMesh(), ir),
      qspace_vec(qspace, 1),
      q(qspace_vec)
   {
      const auto in_fds = std::vector
      {
         FieldDescriptor{U, &fes},
         FieldDescriptor{Coords, &mfes}
      };
      const auto out_fds = std::vector
      {
         FieldDescriptor{Q, &qspace_vec}
      };

      const auto &mesh = *fes.GetParMesh();
      Array<int> all_domain_attr;
      if (mesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(mesh.attributes.Max());
         all_domain_attr = 1;
      }

      dop = std::make_unique<DifferentiableOperator>(in_fds, out_fds, mesh);
      CubicH1Functional<dscalar_t, dim> apply;
      auto derivatives = std::integer_sequence<size_t, U> {};
      dop->AddDomainIntegrator<LocalQFBackend>(
         apply,
         tuple{Value<U>{}, Gradient<U>{}, Gradient<Coords>{}, Weight{}},
         tuple{Identity<Q>{}},
         ir, all_domain_attr, derivatives);

      mesh.GetNodes()->GetTrueDofs(coords);
   }

   real_t Eval(const Vector &u) const
   {
      real_t local = EvalLocal(u), global;
      MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global;
   }

   // Returns the directional derivative dJ/du · du.
   real_t dJdu_dir(const Vector &u, const Vector &du) const
   {
      real_t local = dJdu_dir_local(u, du), global;
      MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global;
   }

   // Computes the full gradient \nabla J(u) in the trial space via J^T.
   // Since Eval sums raw QF values (J = \sum f_q), the adjoint of the
   // summation is a QF of ones.
   void grad(const Vector &u, Vector &g) const
   {
      MultiVector X{u, coords};
      q = 1.0;
      MultiVector ones{q};
      dop->GetDerivative(U, X)->MultTranspose(ones, g);
   }

   // Computes the full gradient via element-wise central differences.
   void grad_fd(const Vector &u, Vector &g, real_t eps = 1e-5) const
   {
      const int local_size = u.Size();

      // Global offset for this rank's DOFs and total DOF count.
      int offset = 0, global_size = local_size;
      MPI_Exscan(&local_size, &offset, 1, MPITypeMap<int>::mpi_type, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, &global_size, 1, MPITypeMap<int>::mpi_type, MPI_SUM,
                    comm);

      g.SetSize(local_size);
      Vector up(u), um(u);

      // Loop over global DOF indices. Each rank perturbs only when gi falls in
      // its local range [offset, offset+local_size); all ranks call Eval together.
      for (int gi = 0; gi < global_size; ++gi)
      {
         const int li = gi - offset;
         if (li >= 0 && li < local_size) { up(li) += eps; um(li) -= eps; }
         const real_t Jp = Eval(up);
         const real_t Jm = Eval(um);
         if (li >= 0 && li < local_size)
         {
            g(li) = (Jp - Jm) / (2.0 * eps);
            up(li) = u(li);
            um(li) = u(li);
         }
      }
   }

private:
   real_t EvalLocal(const Vector &u) const
   {
      MultiVector X{u, coords};
      MultiVector Y{q};
      dop->Mult(X, Y);
      return q.Sum();
   }

   real_t dJdu_dir_local(const Vector &u, const Vector &du) const
   {
      MultiVector X{u, coords};
      MultiVector dY{q};
      dop->GetDerivative(U, X)->Mult(du, dY);
      return q.Sum();
   }

   MPI_Comm comm;
   std::unique_ptr<DifferentiableOperator> dop;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;
   mutable QuadratureFunction q;
   Vector coords;
};

template <int DIM>
void functional(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   H1_FECollection fec(p, DIM);

   ParFiniteElementSpace fes(&pmesh, &fec);
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   const auto ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   Vector u(fes.GetTrueVSize());
   Vector du(fes.GetTrueVSize());

   u.Randomize(5532);
   du.Randomize(3251);

   MyFunctional<DIM> functional(fes, *mfes, ir);

   const real_t dJ_ad = functional.dJdu_dir(u, du);

   Vector g(fes.GetTrueVSize());
   functional.grad(u, g);
   const real_t dJ_ad_grad = InnerProduct(pmesh.GetComm(), g, du);

   real_t best_error_dir  = infinity();
   real_t best_error_grad = infinity();

   for (real_t eps : {1e-3, 3e-4, 1e-4, 3e-5})
   {
      Vector up(u), um(u);
      up.Add(eps, du);
      um.Add(-eps, du);

      const real_t Jp = functional.Eval(up);
      const real_t Jm = functional.Eval(um);

      const real_t dJ_fd  = (Jp - Jm) / (2.0 * eps);
      const real_t scale  = std::max(real_t(1.0), std::abs(dJ_fd));

      best_error_dir  = std::min(best_error_dir,
                                 std::abs(dJ_ad - dJ_fd) / scale);
      best_error_grad = std::min(best_error_grad,
                                 std::abs(dJ_ad_grad - dJ_fd) / scale);
   }

   REQUIRE(best_error_dir  < 1e-7);
   REQUIRE(best_error_grad < 1e-7);

   // Must match entry-wise FD gradient
   Vector g_fd;
   functional.grad_fd(u, g_fd);

   Vector diff(g);
   diff -= g_fd;
   const real_t scale = std::max(real_t(1.0), g_fd.Normlinf());
   real_t local_norm = diff.Normlinf();
   real_t global_norm;
   MPI_Allreduce(&local_norm, &global_norm, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, pmesh.GetComm());
   REQUIRE(diff.Normlinf() / scale < 1e-5);
}

TEST_CASE("dFEM functional derivative action matches finite differences",
          "[Parallel][dFEM][GPU][functional]")
{
   const auto p = GenAll({1}, {2, 3});
   SECTION("2d")
   {
      const auto meshs = { "../../data/inline-quad.mesh" };
      const auto extra = { "../../data/star.mesh",
                           "../../data/star-q3.mesh",
                           "../../data/rt-2d-q3.mesh",
                           "../../data/periodic-square.mesh"
                         };
      functional<2>(GenAll(meshs, extra), p);
   }

   SECTION("3d")
   {
      const auto meshs = { "../../data/inline-hex.mesh" };
      const auto extra = { "../../data/fichera.mesh",
                           "../../data/fichera-q3.mesh",
                           "../../data/toroid-hex.mesh",
                           "../../data/periodic-cube.mesh"
                         };
      functional<3>(GenAll(meshs, extra), p);
   }
}

#endif // MFEM_USE_MPI
