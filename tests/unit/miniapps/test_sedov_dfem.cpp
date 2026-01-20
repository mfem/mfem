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

#include "test_sedov.hpp"

#include <cstring>
#include "linalg/kernels.hpp"

#if defined(MFEM_SEDOV_DFEM_MPI) && !defined(MFEM_USE_MPI)
#error "Cannot use MFEM_SEDOV_DFEM_MPI without MFEM_USE_MPI!"
#endif

using namespace mfem;
using namespace mfem::future;

namespace mfem
{

template <int DIM, int DIM2 = DIM*DIM>
struct QPoint
{
   using vecd_t = tensor<real_t, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   const real_t gamma, cfl, h0;
   inline static constexpr real_t EPS = 1e-12_r;
   inline static constexpr bool use_viscosity = true;

   QPoint(real_t gamma, real_t cfl, real_t h0):
      gamma(gamma), cfl(cfl), h0(h0) { }

   inline constexpr auto ComputeMaterialProperties(const real_t &rho,
                                                   const real_t &E) const noexcept
   {
      const real_t e = std::fmax(0.0_r, E);
      const real_t p = (gamma - 1.0_r) * rho * e;
      const real_t sound_speed = std::sqrt(gamma * (gamma - 1.0_r) * e);
      return future::tuple{p, sound_speed};
   }

   inline static constexpr real_t smooth_step_01(real_t x) noexcept
   {
      const real_t y = (x + EPS) / (2.0_r * EPS);
      if (y < 0.0_r) { return 0.0_r; }
      if (y > 1.0_r) { return 1.0_r; }
      return (3.0_r - 2.0_r * y) * y * y;
   };

   MFEM_HOST_DEVICE inline auto Update(const matd_t &dvdxi,
                                       const real_t &rho0,
                                       const matd_t &J0,
                                       const matd_t &J,
                                       const real_t &E,
                                       const real_t &w) const noexcept
   {
      const real_t detJ = det(J), detJ0 = det(J0);
      const matd_t invJ = inv(J);

      const real_t rho = rho0 * detJ0 / detJ;
      const auto [p, sound_speed] = ComputeMaterialProperties(rho, E);

      matd_t stress{{{0.0_r}}};
      for (int d = 0; d < DIM; d++) { stress(d, d) = -p; }

      const matd_t dvdx = sym(dvdxi * invJ);
      real_t visc_coeff = 0.0;

      if (use_viscosity)
      {
         real_t eig_val_data[DIM], eig_vec_data[DIM2];
         kernels::CalcEigenvalues<DIM>(flatten(dvdx).values, eig_val_data, eig_vec_data);
         const vecd_t compr_dir = make_tensor<DIM>([&](int i) {return eig_vec_data[i];});
         const vecd_t ph_dir = (J * inv(J0)) * compr_dir;
         const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
         const real_t mu = eig_val_data[0];
         visc_coeff +=
            2.0_r * rho * h * h * fabs(mu) +
            0.5_r * rho * h * sound_speed * (1.0_r - smooth_step_01(mu - 2.0_r * EPS));
         stress += visc_coeff * dvdx;
      }
      return tuple{dvdx, stress, visc_coeff, sound_speed};
   }
};

template <int DIM>
struct QForce: public QPoint<DIM>
{
   using Q = QPoint<DIM>;
   using matd_t = typename Q::matd_t;

   QForce(real_t gamma, real_t cfl, real_t h0): Q(gamma, cfl, h0) { }

   MFEM_HOST_DEVICE inline auto operator()(const matd_t &dvdxi,
                                           const real_t &rho0,
                                           const matd_t &J0,
                                           const matd_t &J,
                                           const real_t &E,
                                           const real_t &w) const
   {
      const auto [dvdx, stress, visc, ss] = Q::Update(dvdxi, rho0, J0, J, E, w);
      return future::tuple{stress * transpose(inv(J)) * det(J) * w};
   }
};

template <int DIM>
struct QForceTranspose: public QPoint<DIM>
{
   using QP = QPoint<DIM>;
   using matd_t = typename QP::matd_t;

   QForceTranspose(real_t gamma, real_t cfl, real_t h0): QP(gamma, cfl, h0) { }

   MFEM_HOST_DEVICE inline auto operator()(const real_t &E,
                                           const matd_t &dvdxi,
                                           const real_t &rho0,
                                           const matd_t &J0,
                                           const matd_t &J,
                                           const real_t &w) const
   {
      const auto [dvdx, stress, visc, ss] = QP::Update(dvdxi, rho0, J0, J, E, w);
      return future::tuple{inner(dvdx, stress) * det(J) * w};
   }
};

template <int DIM>
struct QDeltaTEstimator: public QPoint<DIM>
{
   using QP = QPoint<DIM>;
   using matd_t = typename QP::matd_t;
   using QP::cfl;

   QDeltaTEstimator(real_t gamma, real_t cfl, real_t h0): QP(gamma, cfl, h0) { }

   MFEM_HOST_DEVICE inline auto operator()(const matd_t &dvdxi,
                                           const real_t &rho0,
                                           const matd_t &J0,
                                           const matd_t &J,
                                           const real_t &E,
                                           const real_t &w) const
   {
      const real_t detJ = det(J), detJ0 = det(J0);
      const real_t rho = rho0 * detJ0 / detJ;
      const auto [dvdx, stress, visc, ss] = QP::Update(dvdxi, rho0, J0, J, E, w);

      const real_t sv = kernels::CalcSingularvalue<DIM>(flatten(J).values, DIM-1);
      const real_t h_min = sv / 2.0_r;
      const real_t inv_h_min = 1.0_r / h_min;
      const real_t inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
      const real_t inv_dt = ss * inv_h_min + 2.5_r * visc * inv_rho_inv_h_min_sq;

      const real_t dt_est = det(J) <= 0.0_r ? 0.0_r : cfl / inv_dt;
      return tuple{dt_est};
   }
};

template<int DIM>
class Force: public Operator
{
   typename T::FiniteElementSpace &H1, &L2;
   typename T::Mesh &pmesh;
   const int H1vsize;
   UniformParameterSpace Q1;
   mutable Vector dt_est;
   Array<int> domain_attr;
   mutable typename T::GridFunction rho0, x0;
   std::unique_ptr<DifferentiableOperator> force, force_transpose, dt_estimator;

public:
   Force(typename T::FiniteElementSpace &H1,
         typename T::FiniteElementSpace &L2,
         typename T::Mesh &pmesh,
         const IntegrationRule &ir,
         const typename T::GridFunction &rho0,
         const real_t &gamma,
         const real_t &cfl,
         const real_t &h0) :
      H1(H1), L2(L2),
      pmesh(pmesh),
      H1vsize(H1.GetVSize()),
      Q1(pmesh, ir, 1),
      dt_est(Q1.GetTrueVSize()),
      domain_attr(pmesh.attributes.Max()),
      rho0(rho0),
      x0((pmesh.EnsureNodes(),
          *static_cast<typename T::GridFunction *>(pmesh.GetNodes())))
   {
      domain_attr = 1;

      // Unique field IDs
      constexpr int VELOCITY = 0;
      constexpr int COORDINATES = 1, COORDINATES0 = 2;
      constexpr int RHO0 = 3, ENERGY = 4, DELTA_T = 5;

      // Force operator
      force = std::make_unique<DifferentiableOperator>(
                 std::vector{ FieldDescriptor{VELOCITY, &H1}},
                 std::vector{ FieldDescriptor{RHO0, &L2},
                              FieldDescriptor{COORDINATES0, &H1},
                              FieldDescriptor{COORDINATES, &H1},
                              FieldDescriptor{ENERGY, &L2},
                            }, pmesh);
      QForce<DIM> force_qf(gamma, cfl, h0);
      force->AddDomainIntegrator(force_qf,
                                 future::tuple{ Gradient<VELOCITY>{},
                                                Value<RHO0>{},
                                                Gradient<COORDINATES0>{},
                                                Gradient<COORDINATES>{},
                                                Value<ENERGY>{},
                                                Weight{}
                                              },
                                 future::tuple{Gradient<VELOCITY>{}},
                                 ir, domain_attr);
      force->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);

      // Force transpose operator
      force_transpose = std::make_unique<DifferentiableOperator>(
                           std::vector{ FieldDescriptor{ENERGY, &L2}},
                           std::vector{ FieldDescriptor{VELOCITY, &H1},
                                        FieldDescriptor{RHO0, &L2},
                                        FieldDescriptor{COORDINATES0, &H1},
                                        FieldDescriptor{COORDINATES, &H1},
                                      }, pmesh);
      QForceTranspose<DIM> force_transpose_qf(gamma, cfl,h0);
      force_transpose->AddDomainIntegrator(force_transpose_qf,
                                           future::tuple{ Value<ENERGY>{},
                                                          Gradient<VELOCITY>{},
                                                          Value<RHO0>{},
                                                          Gradient<COORDINATES0>{},
                                                          Gradient<COORDINATES>{},
                                                          Weight{}
                                                        },
                                           future::tuple{Value<ENERGY>{}},
                                           ir, domain_attr);
      force_transpose->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);

      // DeltaT estimator
      dt_estimator = std::make_unique<DifferentiableOperator>(
                        std::vector{ FieldDescriptor{VELOCITY, &H1}},
                        std::vector{ FieldDescriptor{RHO0, &L2},
                                     FieldDescriptor{COORDINATES0, &H1},
                                     FieldDescriptor{COORDINATES, &H1},
                                     FieldDescriptor{ENERGY, &L2},
                                     FieldDescriptor{DELTA_T, &Q1}
                                   }, pmesh);
      QDeltaTEstimator<DIM> dt_estimator_qf(gamma, cfl, h0);
      dt_estimator->AddDomainIntegrator(dt_estimator_qf,
                                        future::tuple{ Gradient<VELOCITY>{},
                                                       Value<RHO0>{},
                                                       Gradient<COORDINATES0>{},
                                                       Gradient<COORDINATES>{},
                                                       Value<ENERGY>{},
                                                       Weight{}
                                                     },
                                        future::tuple{Identity<DELTA_T>{}},
                                        ir, domain_attr);
      dt_estimator->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
   }

   inline auto GetGridFunctions(const Vector &S) const
   {
      auto &s = *const_cast<Vector*>(&S);
      typename T::GridFunction x(&H1, s, 0), v(&H1, s, H1vsize), e(&L2, s, 2*H1vsize);
      return tuple{x, v, e};
   }

   void Mult(const Vector &S, Vector &y) const override
   {
      auto [x, v, e] = GetGridFunctions(S);
      force->SetParameters({&rho0, &x0, &x, &e});
      force->Mult(v, y);
   }

   void MultTranspose(const Vector &S, Vector &y) const override
   {
      auto [x, v, e] = GetGridFunctions(S);
      force_transpose->SetParameters({&v, &rho0, &x0, &x});
      force_transpose->Mult(e, y);
   }

   real_t EstimateDeltaT(const Vector &S) const
   {
      auto [x, v, e] = GetGridFunctions(S);
      dt_estimator->SetParameters({&rho0, &x0, &x, &e, &dt_est});
      dt_estimator->Mult(v, dt_est);
      return dt_est.Min();
   }

   void ResetTimeStepEstimate() { dt_est = std::numeric_limits<real_t>::infinity(); }
};

template <int DIM>
struct QMass
{
   const real_t rho;

   QMass() = delete;

   QMass(real_t rho): rho(rho) { }

   MFEM_HOST_DEVICE inline auto operator()(const real_t &u,
                                           const tensor<real_t, DIM, DIM> &J,
                                           const real_t &w) const
   {
      return future::tuple{rho * u * w * det(J)};
   };
};

template <int DIM>
class Mass : public Operator
{
   typename T::FiniteElementSpace &pfes;
   typename T::Mesh &pmesh;
   Array<int> domain_attr;
   typename T::GridFunction *nodes;
   std::unique_ptr<DifferentiableOperator> mass;
   mutable int ess_tdofs_count;
   mutable Array<int> ess_tdofs;

public:
   Mass(Coefficient &Q,
        typename T::FiniteElementSpace &pfes,
        typename T::Mesh &pmesh,
        const IntegrationRule &ir):
      pfes(pfes),
      pmesh(pmesh),
      domain_attr(pmesh.attributes.Max()),
      nodes(static_cast<typename T::GridFunction *>(pmesh.GetNodes())),
      ess_tdofs_count(0)
   {
      domain_attr = 1;

      constexpr int U = 0, X = 1;
      mass = std::make_unique<DifferentiableOperator>(
                std::vector{ FieldDescriptor{U, &pfes}},
                std::vector{ FieldDescriptor{X, nodes->ParFESpace()}},
                pmesh);

      QMass<DIM> mf_mass_qf{dynamic_cast<ConstantCoefficient*>(&Q)->constant};
      mass->AddDomainIntegrator(mf_mass_qf,
                                tuple{ Value<U>{}, Gradient<X>{}, Weight{} },
                                tuple{ Value<U>{} },
                                ir, domain_attr);
      mass->SetParameters({ nodes });
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      y.SetSize(x.Size());
      mass->Mult(x, y);
      if (ess_tdofs_count) { y.SetSubVector(ess_tdofs, 0.0); }
   }

   void SetEssentialTrueDofs(Array<int> &dofs) const
   {
      ess_tdofs_count = dofs.Size();
      if (ess_tdofs.Size() == 0)
      {
         int global_ess_tdofs_count;
         MPI_Allreduce(&ess_tdofs_count, &global_ess_tdofs_count, 1,
                       MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         MFEM_VERIFY(global_ess_tdofs_count>0, "!(global_ess_tdofs_count>0)");
         ess_tdofs.SetSize(global_ess_tdofs_count);
      }
      if (ess_tdofs_count == 0) { return; }
      ess_tdofs = dofs;
   }

   void EliminateRHS(Vector &b) const
   {
      if (ess_tdofs_count > 0) { b.SetSubVector(ess_tdofs, 0.0); }
   }
};

template<int DIM>
class LagrangianHydroOperator : public LagrangianHydroBase<DIM>
{
   Force<DIM> force;
   Mass<DIM> v_mass, e_mass;

   using B = LagrangianHydroBase<DIM>;

   void UpdateQuadratureData(const Vector &S) const final
   {
      B::dt_est = force.EstimateDeltaT(S);
      B::quad_data_is_current = true;
   }

public:
   LagrangianHydroOperator(Coefficient &rho_coeff, const int size,
                           typename T::FiniteElementSpace &h1,
                           typename T::FiniteElementSpace &l2,
                           typename T::Mesh &pmesh,
                           const Array<int> &essential_tdofs,
                           typename T::GridFunction &rho0,
                           const int source_type,
                           const real_t cfl_,
                           const Coefficient &material,
                           const bool visc,
                           const real_t cgt,
                           const int cgiter,
                           real_t ftz,
                           const int order_q,
                           const real_t gm,
                           int h1_basis_type):
      B(rho_coeff, size, h1, l2, pmesh, essential_tdofs, rho0, source_type,
        cfl_, material, visc, cgt, cgiter, ftz, order_q, gm, h1_basis_type),
      force(B::H1, B::L2, pmesh, B::ir, rho0, B::gamma, B::cfl, B::h0),
      v_mass(rho_coeff, B::H1c, pmesh, B::ir),
      e_mass(rho_coeff, B::L2, pmesh, B::ir)
   {
      B::CG_VMass.SetOperator(v_mass);
      B::CG_EMass.SetOperator(e_mass);
   }

   void ForceMult(const Vector &S) const final
   {
      force.Mult(S, B::rhs);
   }

   void VMassSetup(const int c) const final
   {
      v_mass.SetEssentialTrueDofs(B::c_tdofs[c]);
      v_mass.EliminateRHS(B::B);
   }

   void ForceMultTranspose(const Vector &S, const Vector &v) const final
   {
      force.MultTranspose(S, B::e_rhs);
   }
};

} // namespace mfem

#ifdef MFEM_SEDOV_DFEM_MPI
TEST_CASE("Sedov", "[Sedov][Parallel]")
{
   sedov_tests<LagrangianHydroOperator>(Mpi::WorldRank());
}
#else
TEST_CASE("Sedov", "[Sedov]")
{
   sedov_tests<LagrangianHydroOperator>(0);
}
#endif
