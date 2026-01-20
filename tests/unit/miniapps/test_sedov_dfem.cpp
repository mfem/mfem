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
struct QuadratureData
{
   using vecd_t = tensor<real_t, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   const real_t gamma, cfl, h0;
   inline static constexpr real_t EPS = 1e-12_r;
   inline static constexpr bool use_viscosity = true;

   QuadratureData(real_t gamma, real_t cfl, real_t h0):
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
struct QForce: public QuadratureData<DIM>
{
   using Q = QuadratureData<DIM>;
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
struct QForceTranspose: public QuadratureData<DIM>
{
   using QD = QuadratureData<DIM>;
   using matd_t = typename QD::matd_t;

   QForceTranspose(real_t gamma, real_t cfl, real_t h0): QD(gamma, cfl, h0) { }

   MFEM_HOST_DEVICE inline auto operator()(const real_t &E,
                                           const matd_t &dvdxi,
                                           const real_t &rho0,
                                           const matd_t &J0,
                                           const matd_t &J,
                                           const real_t &w) const
   {
      const auto [dvdx, stress, visc, ss] = QD::Update(dvdxi, rho0, J0, J, E, w);
      return future::tuple{inner(dvdx, stress) * det(J) * w};
   }
};

template <int DIM>
struct QDeltaTEstimator: public QuadratureData<DIM>
{
   using QD = QuadratureData<DIM>;
   using matd_t = typename QD::matd_t;
   using QD::cfl;

   QDeltaTEstimator(real_t gamma, real_t cfl, real_t h0): QD(gamma, cfl, h0) { }

   MFEM_HOST_DEVICE inline auto operator()(const matd_t &dvdxi,
                                           const real_t &rho0,
                                           const matd_t &J0,
                                           const matd_t &J,
                                           const real_t &E,
                                           const real_t &w) const
   {
      const real_t detJ = det(J), detJ0 = det(J0);
      const real_t rho = rho0 * detJ0 / detJ;
      const auto [dvdx, stress, visc, ss] = QD::Update(dvdxi, rho0, J0, J, E, w);

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
   Vector dt_est;
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

   real_t EstimateDeltaT(const Vector &S)
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

template<int DIM, typename TMesh>
static void ComputeVolume(const int NE,
                          const IntegrationRule &ir,
                          TMesh *mesh,
                          real_t &loc_area)
{
   const int NQ = ir.GetNPoints();
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int flags = GeometricFactors::JACOBIANS|GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(ir, flags);
   Vector area(NE*NQ), one(NE*NQ);

   if constexpr (DIM == 2)
   {
      const auto W = Reshape(ir.GetWeights().Read(), Q1D, Q1D);
      const auto detJ = Reshape(geom->detJ.Read(), Q1D, Q1D, NE);
      auto A = Reshape(area.Write(), Q1D, Q1D, NE);
      auto O = Reshape(one.Write(), Q1D, Q1D, NE);
      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
            {
               const real_t det = detJ(qx, qy, e);
               A(qx, qy, e) = W(qx, qy) * det;
               O(qx, qy, e) = 1.0;
            }
         }
      });
   }

   if constexpr (DIM == 3)
   {
      const auto W = Reshape(ir.GetWeights().Read(), Q1D, Q1D, Q1D);
      const auto detJ = Reshape(geom->detJ.Read(), Q1D, Q1D, Q1D, NE);
      auto A = Reshape(area.Write(), Q1D, Q1D, Q1D, NE);
      auto O = Reshape(one.Write(), Q1D, Q1D, Q1D, NE);
      mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD_DIRECT(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx,x,Q1D)
               {
                  const real_t det = detJ(qx, qy, qz, e);
                  A(qx, qy, qz, e) = W(qx, qy, qz) * det;
                  O(qx, qy, qz, e) = 1.0;
               }
            }
         }
      });
   }
   loc_area = area * one;
}

template<int DIM>
class LagrangianHydroOperator : public TimeDependentOperator
{
   typename T::FiniteElementSpace &H1, &L2;
   mutable typename T::FiniteElementSpace H1c;
   const int H1Vsize, H1TVSize, H1cTVSize, L2Vsize, L2TVSize;
   Array<int> block_offsets;
   mutable typename T::GridFunction x_gf;
   const Array<int> &ess_tdofs;
   const int nzones, l2dofs_cnt, h1dofs_cnt, source_type;
   const real_t cfl;
   const bool use_viscosity;
   const real_t cg_rel_tol;
   const int cg_max_iter;
   const real_t ftz_tol;
   const Coefficient &material_pcf;
   const IntegrationRule &ir;
   real_t h0;
   mutable real_t dt_est;
   mutable bool quad_data_is_current;
   std::unique_ptr<Force<DIM>> force;
   Mass<DIM> VMassPA, EMassPA;
   CGSolver CG_VMass, CG_EMass;
   const real_t gamma;
   mutable Vector X, B, rhs, e_rhs;
   mutable typename T::GridFunction rhs_c_gf, dvc_gf;
   mutable Array<int> c_tdofs[3];

   void UpdateQuadratureData(const Vector &S) const
   {
      dt_est = force->EstimateDeltaT(S);
      quad_data_is_current = true;
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
      TimeDependentOperator(size),
      H1(h1), L2(l2),
      H1c(&pmesh, h1.FEColl(), 1),
      H1Vsize(H1.GetVSize()),
      H1TVSize(H1.GetTrueVSize()),
      H1cTVSize(H1c.GetTrueVSize()),
      L2Vsize(L2.GetVSize()),
      L2TVSize(L2.GetTrueVSize()),
      block_offsets(4),
      x_gf(&H1),
      ess_tdofs(essential_tdofs),
      nzones(h1.GetMesh()->GetNE()),
      l2dofs_cnt(l2.GetTypicalFE()->GetDof()),
      h1dofs_cnt(h1.GetTypicalFE()->GetDof()),
      source_type(source_type), cfl(cfl_),
      use_viscosity(visc),
      cg_rel_tol(cgt), cg_max_iter(cgiter),ftz_tol(ftz),
      material_pcf(material),
      ir(IntRules.Get(h1.GetMesh()->GetTypicalElementGeometry(),
                      (order_q > 0) ? order_q :
                      3*h1.GetElementOrder(0) + l2.GetElementOrder(0) - 1)),
      quad_data_is_current(false),
      VMassPA(rho_coeff, H1c, pmesh, ir),
      EMassPA(rho_coeff, L2, pmesh, ir),
      CG_VMass(GetCGSolver()),
      CG_EMass(GetCGSolver()),
      gamma(gm),
      X(H1c.GetTrueVSize()),
      B(H1c.GetTrueVSize()),
      rhs(H1Vsize),
      e_rhs(L2Vsize),
      rhs_c_gf(&H1c),
      dvc_gf(&H1c)
   {
      block_offsets[0] = 0;
      block_offsets[1] = block_offsets[0] + H1Vsize;
      block_offsets[2] = block_offsets[1] + H1Vsize;
      block_offsets[3] = block_offsets[2] + L2Vsize;
      pmesh.GetNodes()->ReadWrite();
      const int bdr_attr_max = pmesh.bdr_attributes.Max();
      Array<int> ess_bdr(bdr_attr_max);
      for (int c = 0; c < DIM; c++)
      {
         ess_bdr = 0; ess_bdr[c] = 1;
         H1c.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
         c_tdofs[c].Read();
      }
      X.UseDevice(true);
      B.UseDevice(true);
      rhs.UseDevice(true);
      e_rhs.UseDevice(true);
      GridFunctionCoefficient rho_coeff_gf(&rho0);
      real_t loc_area = 0.0, glob_area;
      int loc_z_cnt = nzones, glob_z_cnt;
      auto *pm = H1.GetMesh();
      ComputeVolume<DIM>(nzones, ir, H1.GetMesh(), loc_area);
      SumReduce(&loc_area, &glob_area);
      SumReduce(&loc_z_cnt, &glob_z_cnt);
      switch (pm->GetTypicalElementGeometry())
      {
         case Geometry::SQUARE: h0 = sqrt(glob_area / glob_z_cnt); break;
         case Geometry::CUBE: h0 = pow(glob_area / glob_z_cnt, 1.0/3.0); break;
         default: MFEM_ABORT("Unknown zone type!");
      }
      h0 /= (real_t) H1.GetElementOrder(0);

      force = std::make_unique<Force<DIM>>(h1, l2, pmesh, ir, rho0, gamma, cfl, h0);

      CG_VMass.SetOperator(VMassPA);
      CG_VMass.SetRelTol(cg_rel_tol);
      CG_VMass.SetAbsTol(0.0);
      CG_VMass.SetMaxIter(cg_max_iter);
      CG_VMass.SetPrintLevel(0);

      CG_EMass.SetOperator(EMassPA);
      CG_EMass.iterative_mode = false;
      CG_EMass.SetRelTol(1e-8);
      CG_EMass.SetAbsTol(1e-8 * std::numeric_limits<real_t>::epsilon());
      CG_EMass.SetMaxIter(200);
      CG_EMass.SetPrintLevel(-1);
   }

   void Mult(const Vector &S, Vector &dS_dt) const override
   {
      UpdateMesh(S);
      auto *sptr = const_cast<Vector*>(&S);
      typename T::GridFunction v, dx;
      const int VsizeH1 = H1.GetVSize();
      v.MakeRef(&H1, *sptr, VsizeH1);
      dx.MakeRef(&H1, dS_dt, 0);
      dx = v;
      SolveVelocity(S, dS_dt);
      SolveEnergy(S, v, dS_dt);
      quad_data_is_current = false;
   }

   MemoryClass GetMemoryClass() const override  { return Device::GetDeviceMemoryClass(); }

   void SolveVelocity(const Vector &S, Vector &dS_dt) const
   {
      UpdateQuadratureData(S);
      typename T::GridFunction dv(&H1, dS_dt, H1Vsize);
      dv = 0.0;
      force->Mult(S, rhs);
      rhs.Neg();
      const int size = H1c.GetVSize();
      const Operator *Pconf = H1c.GetProlongationMatrix();
      const Operator *Rconf = H1c.GetRestrictionMatrix();
      for (int c = 0; c < DIM; c++)
      {
         dvc_gf.MakeRef(&H1c, dS_dt, H1Vsize + c*size);
         rhs_c_gf.MakeRef(&H1c, rhs, c*size);
         if (Pconf) { Pconf->MultTranspose(rhs_c_gf, B); }
         else { B = rhs_c_gf; }
         if (Rconf) { Rconf->Mult(dvc_gf, X); }
         else { X = dvc_gf; }
         VMassPA.SetEssentialTrueDofs(c_tdofs[c]);
         VMassPA.EliminateRHS(B);
         CG_VMass.Mult(B, X);
         if (Pconf) { Pconf->Mult(X, dvc_gf); }
         else { dvc_gf = X; }
         dvc_gf.GetMemory().SyncAlias(dS_dt.GetMemory(), dvc_gf.Size());
      }
   }

   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const
   {
      UpdateQuadratureData(S);
      typename T::GridFunction de(&L2, dS_dt, H1Vsize*2);
      de = 0.0;
      force->MultTranspose(S, e_rhs);
      CG_EMass.Mult(e_rhs, de);
      de.GetMemory().SyncAlias(dS_dt.GetMemory(), de.Size());
   }

   void UpdateMesh(const Vector &const_S) const
   {
      auto* S = const_cast<Vector*>(&const_S);
      x_gf.MakeRef(&H1, *S, 0);
      H1.GetMesh()->NewNodes(x_gf, false);
   }

   real_t GetTimeStepEstimate(const Vector &S) const
   {
      UpdateMesh(S);
      UpdateQuadratureData(S);
      real_t glob_dt_est;
      MinReduce(&dt_est, &glob_dt_est);
      return glob_dt_est;
   }

   void ResetTimeStepEstimate() const
   {
      dt_est = std::numeric_limits<real_t>::infinity();
   }

   void ResetQuadratureData() const { quad_data_is_current = false; }
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

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   std::cout << "\nThe Sedov unit tests are not supported in single"
             " precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

#ifdef MFEM_SEDOV_DFEM_MPI
   mfem::Mpi::Init();
   mfem::Hypre::Init();
#endif

#ifdef MFEM_SEDOV_DFEM_DEVICE
   Device device(MFEM_SEDOV_DFEM_DEVICE);
#else
   Device device("cpu"); // make sure hypre runs on CPU, if possible
#endif
   device.Print();

#if defined(MFEM_SEDOV_DFEM_MPI) && defined(MFEM_DEBUG) && defined(MFEM_SEDOV_DFEM_DEVICE)
   if (HypreUsingGPU() && !strcmp(MFEM_SEDOV_DFEM_DEVICE, "debug"))
   {
      mfem::out << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this unit test\n"
                << "is NOT supported with the GPU version of hypre.\n\n";
      return MFEM_SKIP_RETURN_VALUE;
   }
#endif

#ifdef MFEM_SEDOV_DFEM_MPI
   return RunCatchSession(argc, argv, {"[Parallel]"}, Root());
#else
   // Exclude parallel tests.
   return RunCatchSession(argc, argv, {"~[Parallel]"});
#endif
}
