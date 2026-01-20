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

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "run_unit_tests.hpp"

#include <cstring>
#include "linalg/kernels.hpp"

#if defined(MFEM_SEDOV_DFEM_MPI) && !defined(MFEM_USE_MPI)
#error "Cannot use MFEM_SEDOV_DFEM_MPI without MFEM_USE_MPI!"
#endif

#if defined(MFEM_USE_MPI) && defined(MFEM_SEDOV_DFEM_MPI)
static auto GetParMesh = [](mfem::Mesh &mesh) { return mfem::ParMesh(MPI_COMM_WORLD, mesh); };
static auto GetCGSolver = []() { return mfem::CGSolver(MPI_COMM_WORLD); };
static MPI_Datatype mpi_real_t = mfem::MPITypeMap<mfem::real_t>::mpi_type;
#else
#define HYPRE_BigInt int
#define GlobalTrueVSize GetVSize
#define ParMesh Mesh
#define GetParMesh GetMesh
#define ParBilinearForm BilinearForm
#define ParGridFunction GridFunction
#define ParFiniteElementSpace FiniteElementSpace
#define MPI_Allreduce(src,dst,...) *dst = *src
static auto GetParMesh = [](mfem::Mesh &mesh) { return mfem::Mesh(mesh); };
static auto GetCGSolver = []() { return mfem::CGSolver(); };
#endif

using namespace mfem;
using namespace mfem::future;

namespace mfem
{

static void v0(const Vector&, Vector &v) { v = 0.0; }
static real_t rho0(const Vector&) { return 1.0; }
static real_t gamma(const Vector&) { return 1.4; }

template <int DIM, int DIM2 = DIM*DIM>
struct QData
{
   using vecd_t = tensor<real_t, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   const real_t gamma, cfl, h0;
   inline static constexpr real_t EPS = 1e-12_r;
   inline static constexpr bool use_viscosity = true;

   QData(real_t gamma, real_t cfl, real_t h0): gamma(gamma), cfl(cfl), h0(h0) { }

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
struct QForce: public QData<DIM>
{
   using Q = QData<DIM>;
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
struct QForceTranspose: public QData<DIM>
{
   using QD = QData<DIM>;
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
struct QDeltaTEstimator: public QData<DIM>
{
   using QD = QData<DIM>;
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
class ForceOperator: public Operator
{
   ParFiniteElementSpace &H1, &L2;
   ParMesh &pmesh;
   const int H1vsize;
   UniformParameterSpace Q1;
   Vector dt_est;
   Array<int> domain_attr;
   mutable ParGridFunction rho0, x0;
   std::unique_ptr<DifferentiableOperator> force, force_transpose, dt_estimator;

public:
   ForceOperator(ParFiniteElementSpace &H1,
                 ParFiniteElementSpace &L2,
                 const IntegrationRule &ir,
                 const ParGridFunction &rho0,
                 const real_t &gamma,
                 const real_t &cfl,
                 const real_t &h0) :
      H1(H1), L2(L2),
      pmesh(*H1.GetParMesh()),
      H1vsize(H1.GetVSize()),
      Q1(pmesh, ir, 1),
      dt_est(Q1.GetTrueVSize()),
      domain_attr(pmesh.attributes.Max()),
      rho0(rho0),
      x0((pmesh.EnsureNodes(), *static_cast<ParGridFunction *>(pmesh.GetNodes())))
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
      ParGridFunction x(&H1, s, 0), v(&H1, s, H1vsize), e(&L2, s, 2*H1vsize);
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
class MassOperator : public Operator
{
   ParFiniteElementSpace &pfes;
   ParMesh *pmesh;
   Array<int> domain_attr;
   ParGridFunction *nodes;
   std::unique_ptr<DifferentiableOperator> mass;
   mutable int ess_tdofs_count;
   mutable Array<int> ess_tdofs;

public:
   MassOperator(Coefficient &Q,
                ParFiniteElementSpace &pfes,
                const IntegrationRule &ir):
      pfes(pfes),
      pmesh(pfes.GetParMesh()),
      domain_attr(pmesh->attributes.Max()),
      nodes(static_cast<ParGridFunction *>(pmesh->GetNodes())),
      ess_tdofs_count(0)
   {
      domain_attr = 1;

      constexpr int U = 0, X = 1;
      mass = std::make_unique<DifferentiableOperator>(
                std::vector{ FieldDescriptor{U, &pfes}},
                std::vector{ FieldDescriptor{X, nodes->ParFESpace()}},
                *pmesh);

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
static void ComputeVolume(const int NE,
                          const IntegrationRule &ir,
                          ParMesh *mesh,
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
protected:
   ParFiniteElementSpace &H1, &L2;
   mutable ParFiniteElementSpace H1c;
   const int H1Vsize, H1TVSize, H1cTVSize, L2Vsize, L2TVSize;
   Array<int> block_offsets;
   mutable ParGridFunction x_gf;
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
   std::unique_ptr<ForceOperator<DIM>> Force;
   MassOperator<DIM> VMassPA, EMassPA;
   CGSolver CG_VMass, CG_EMass;
   const real_t gamma;
   mutable Vector X, B, rhs, e_rhs;
   mutable ParGridFunction rhs_c_gf, dvc_gf;
   mutable Array<int> c_tdofs[3];

   void UpdateQuadratureData(const Vector &S) const
   {
      dt_est = Force->EstimateDeltaT(S);
      quad_data_is_current = true;
   }

public:
   LagrangianHydroOperator(Coefficient &rho_coeff,
                           const int size,
                           ParFiniteElementSpace &h1,
                           ParFiniteElementSpace &l2,
                           const Array<int> &essential_tdofs,
                           ParGridFunction &rho0,
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
      H1c(h1.GetParMesh(), h1.FEColl(), 1),
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
      VMassPA(rho_coeff, H1c, ir),
      EMassPA(rho_coeff, L2, ir),
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
      H1.GetParMesh()->GetNodes()->ReadWrite();
      const int bdr_attr_max = H1.GetMesh()->bdr_attributes.Max();
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
      ParMesh *pm = H1.GetParMesh();
      ComputeVolume<DIM>(nzones, ir, H1.GetParMesh(), loc_area);
      MPI_Allreduce(&loc_area, &glob_area, 1, mpi_real_t, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&loc_z_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      switch (pm->GetTypicalElementGeometry())
      {
         case Geometry::SQUARE: h0 = sqrt(glob_area / glob_z_cnt); break;
         case Geometry::CUBE: h0 = pow(glob_area / glob_z_cnt, 1.0/3.0); break;
         default: MFEM_ABORT("Unknown zone type!");
      }
      h0 /= (real_t) H1.GetElementOrder(0);

      Force = std::make_unique<ForceOperator<DIM>>(h1, l2, ir, rho0, gamma, cfl, h0);

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
      Vector* sptr = const_cast<Vector*>(&S);
      ParGridFunction v;
      const int VsizeH1 = H1.GetVSize();
      v.MakeRef(&H1, *sptr, VsizeH1);
      ParGridFunction dx;
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
      ParGridFunction dv(&H1, dS_dt, H1Vsize);
      dv = 0.0;
      Force->Mult(S, rhs);
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
      ParGridFunction de(&L2, dS_dt, H1Vsize*2);
      de = 0.0;
      Force->MultTranspose(S, e_rhs);
      CG_EMass.Mult(e_rhs, de);
      de.GetMemory().SyncAlias(dS_dt.GetMemory(), de.Size());
   }

   void UpdateMesh(const Vector &const_S) const
   {
      auto* S = const_cast<Vector*>(&const_S);
      x_gf.MakeRef(&H1, *S, 0);
      H1.GetParMesh()->NewNodes(x_gf, false);
   }

   real_t GetTimeStepEstimate(const Vector &S) const
   {
      UpdateMesh(S);
      UpdateQuadratureData(S);
      real_t glob_dt_est;
      MPI_Allreduce(&dt_est, &glob_dt_est, 1, mpi_real_t, MPI_MIN, MPI_COMM_WORLD);
      return glob_dt_est;
   }

   void ResetTimeStepEstimate() const
   {
      dt_est = std::numeric_limits<real_t>::infinity();
   }

   void ResetQuadratureData() const { quad_data_is_current = false; }
};

} // namespace mfem

template <int DIM>
int sedov(int myid, int argc, char *argv[])
{
   int rs_levels = 0;
   int max_tsteps = -1;
   constexpr int rp_levels = 0;
   constexpr int order_v = 2;
   constexpr int order_e = 1;
   constexpr int order_q = -1;
   constexpr real_t t_final = 0.6;
   constexpr real_t cfl = 0.5;
   constexpr real_t cg_tol = 1e-14;
   constexpr real_t ftz_tol = 0.0;
   constexpr int cg_max_iter = 300;
   constexpr int vis_steps = 5;
   constexpr real_t blast_energy = 0.25;
   constexpr real_t blast_position[] = {0.0, 0.0, 0.0};
   constexpr int source = 0;
   constexpr bool visc = true;

   OptionsParser args(argc, argv);
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(mfem::out); }
      return -1;
   }

   Mesh mesh;
   if constexpr (DIM == 2)
   {
      constexpr Element::Type QUAD = Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(2, 2, QUAD, true);
      const int NBE = mesh.GetNBE();
      for (int b = 0; b < NBE; b++)
      {
         Element *bel = mesh.GetBdrElement(b);
         MFEM_ASSERT(bel->GetType() == Element::SEGMENT, "");
         const int attr = (b < NBE/2) ? 2 : 1;
         bel->SetAttribute(attr);
      }
   }

   if constexpr (DIM == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2,Element::HEXAHEDRON);
      const int NBE = mesh.GetNBE();
      MFEM_ASSERT(NBE == 24,"");
      for (int b = 0; b < NBE; b++)
      {
         Element *bel = mesh.GetBdrElement(b);
         MFEM_ASSERT(bel->GetType() == Element::QUADRILATERAL, "");
         const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
         bel->SetAttribute(attr);
      }
   }

   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   ParMesh pmesh = GetParMesh(mesh);
   mesh.Clear();
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   int nzones = pmesh.GetNE(), nzones_min, nzones_max;
   MPI_Allreduce(&nzones, &nzones_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&nzones, &nzones_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   if (nzones_min == 0)
   {
      if (myid == 0) { mfem::out << "Some MPI ranks have no zones. Exiting." << std::endl; }
      return EXIT_SUCCESS;
   }

   L2_FECollection L2fec(order_e, DIM, BasisType::Positive);
   H1_FECollection H1fec(order_v, DIM);
   ParFiniteElementSpace L2(&pmesh, &L2fec);
   ParFiniteElementSpace H1(&pmesh, &H1fec, pmesh.Dimension());
   Array<int> ess_tdofs;
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max()), tdofs1d;
      for (int d = 0; d < pmesh.Dimension(); d++)
      {
         ess_bdr = 0; ess_bdr[d] = 1;
         H1.GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
         ess_tdofs.Append(tdofs1d);
      }
   }

   RK4Solver ode_solver;

   const HYPRE_BigInt H1GTVSize = H1.GlobalTrueVSize(),
                      L2GTVSize = L2.GlobalTrueVSize();
   const int H1Vsize = H1.GetVSize(), L2Vsize = L2.GetVSize();
   if (myid == 0)
   {
      mfem::out << "Number of local/global kinematic (position, velocity) dofs: "
                << H1Vsize << "/" << H1GTVSize << std::endl;
      mfem::out << "Number of local/global specific internal energy dofs: "
                << L2Vsize << "/" << L2GTVSize << std::endl;
   }

   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + H1Vsize;
   true_offset[2] = true_offset[1] + H1Vsize;
   true_offset[3] = true_offset[2] + L2Vsize;
   BlockVector S(true_offset, Device::GetDeviceMemoryType());
   S.UseDevice(true);

   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1, S, true_offset[0]);
   v_gf.MakeRef(&H1, S, true_offset[1]);
   e_gf.MakeRef(&L2, S, true_offset[2]);
   pmesh.SetNodalGridFunction(&x_gf);
   x_gf.SyncAliasMemory(S);

   VectorFunctionCoefficient v_coeff(pmesh.Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);
   v_gf.SyncAliasMemory(S);

   ParGridFunction rho(&L2);
   FunctionCoefficient rho_fct_coeff(rho0);
   ConstantCoefficient rho_coeff(1.0);
   L2_FECollection l2_fec(order_e, pmesh.Dimension());
   ParFiniteElementSpace l2_fes(&pmesh, &l2_fec);
   ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
   l2_rho.ProjectCoefficient(rho_fct_coeff);
   rho.ProjectGridFunction(l2_rho);

   DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                            blast_position[2], blast_energy);
   e_coeff.SetWeight(new ConstantCoefficient(1.0));
   l2_e.ProjectCoefficient(e_coeff);
   e_gf.ProjectGridFunction(l2_e);
   e_gf.SyncAliasMemory(S);

   L2_FECollection mat_fec(0, pmesh.Dimension());
   ParFiniteElementSpace mat_fes(&pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(mfem::gamma);
   mat_gf.ProjectCoefficient(mat_coeff);
   GridFunctionCoefficient mat_gf_coeff(&mat_gf);

   LagrangianHydroOperator<DIM> oper(rho_coeff, S.Size(),
                                     H1, L2, ess_tdofs,
                                     rho, source, cfl,
                                     mat_gf_coeff,
                                     visc, cg_tol, cg_max_iter,
                                     ftz_tol, order_q,
                                     gamma(S),
                                     H1fec.GetBasisType());

   ode_solver.Init(oper);
   oper.ResetTimeStepEstimate();
   real_t t = 0.0, t_old, dt = oper.GetTimeStepEstimate(S);
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   int checks = 0;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      oper.ResetTimeStepEstimate();
      ode_solver.Step(S, t, dt);
      steps++;
      const real_t dt_est = oper.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         dt *= 0.85;
         if (dt < std::numeric_limits<real_t>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         oper.ResetQuadratureData();
         if (myid == 0) { mfem::out << "Repeating step " << ti << std::endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);
      pmesh.NewNodes(x_gf, false);
      if (last_step || (ti % vis_steps) == 0)
      {
         real_t loc_norm = e_gf * e_gf, tot_norm;
         MPI_Allreduce(&loc_norm, &tot_norm, 1, mpi_real_t, MPI_SUM, MPI_COMM_WORLD);
         if (myid == 0)
         {
            const real_t sqrt_tot_norm = sqrt(tot_norm);
            mfem::out << std::fixed;
            mfem::out << "step " << std::setw(5) << ti
                      << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                      << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
                      << ",\t|e| = " << std::setprecision(10)
                      << sqrt_tot_norm;
            mfem::out << std::endl;
         }
      }
      real_t loc_norm = e_gf * e_gf, tot_norm;
      MPI_Allreduce(&loc_norm, &tot_norm, 1, mpi_real_t, MPI_SUM, MPI_COMM_WORLD);
      const real_t stm = sqrt(tot_norm);
      REQUIRE((rs_levels == 0 || rs_levels == 1));
      if constexpr (DIM == 2)
      {
         constexpr real_t p1_05[2] = { 3.508254945225794e+00, 1.403249766367977e+01 };
         constexpr real_t p1_15[2] = { 2.756444596823211e+00, 1.104093401469385e+01 };
         if (ti==5) {checks++; REQUIRE(stm == MFEM_Approx(p1_05[rs_levels]));}
         if (ti==15) {checks++; REQUIRE(stm == MFEM_Approx(p1_15[rs_levels]));}
      }
      if constexpr (DIM == 3)
      {
         constexpr real_t p1_05[2] = { 1.339163718592567e+01, 1.071277540097426e+02 };
         constexpr real_t p1_28[2] = { 7.521073677398005e+00, 5.985720905709158e+01 };
         if (ti==5) {checks++; REQUIRE(stm == MFEM_Approx(p1_05[rs_levels]));}
         if (ti==28) {checks++; REQUIRE(stm == MFEM_Approx(p1_28[rs_levels]));}
      }
   }
   REQUIRE(checks == 2);
   steps *= 4;
   return EXIT_SUCCESS;
}

static inline int argn(const char *argv[], int argc = 0)
{
   while (argv[argc]) { argc+=1; }
   return argc;
}

static void sedov_tests(int rank)
{
   const char *argv2D[]= { "sedov<2>", nullptr };
   REQUIRE(sedov<2>(rank, argn(argv2D), const_cast<char**>(argv2D)) == 0);

   const char *argv2Drs1[]= { "sedov<2>", "-rs", "1", "-ms", "20", nullptr };
   REQUIRE(sedov<2>(rank, argn(argv2Drs1), const_cast<char**>(argv2Drs1)) == 0);

   const char *argv3D[]= { "sedov<3>", nullptr };
   REQUIRE(sedov<3>(rank, argn(argv3D), const_cast<char**>(argv3D)) == 0);

   const char *argv3Drs1[]= { "sedov<3>", "-rs", "1", "-ms", "28", nullptr };
   REQUIRE(sedov<3>(rank, argn(argv3Drs1), const_cast<char**>(argv3Drs1)) == 0);
}

#ifdef MFEM_SEDOV_DFEM_MPI
TEST_CASE("Sedov", "[Sedov][Parallel]")
{
   sedov_tests(Mpi::WorldRank());
}
#else
TEST_CASE("Sedov", "[Sedov]")
{
   sedov_tests(0);
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
