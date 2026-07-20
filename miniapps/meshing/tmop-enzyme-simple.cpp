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
//
//    ---------------------------------------------------------------------
//    Simple Enzyme TMOP Miniapp: Constant and Analytic Targets Only
//    ---------------------------------------------------------------------
//
// This is a standalone Enzyme/dFEM TMOP optimizer for the two target cases
// used by pmesh-optimizer-enzyme:
//   1. constant ideal target,
//   4. analytic full target in physical coordinates.
//
// The implementation intentionally does not include
// pmesh-optimizer-enzyme-common.hpp.
//
// Compile with: make -C miniapps/meshing tmop-enzyme-simple
//
// Sample runs and equivalent pmesh-optimizer runs:
//   Constant ideal target:
//     mpirun -np 4 tmop-enzyme-simple -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8
//     mpirun -np 4 pmesh-optimizer     -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8
//   Constant ideal target with node limiting:
//     mpirun -np 4 tmop-enzyme-simple -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 2 -art 1 -bnd -qt 1 -qo 8 -ex -lc 5000
//     mpirun -np 4 pmesh-optimizer     -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 2 -art 1 -bnd -qt 1 -qo 8 -ex -lc 5000
//   Analytic annular shape target:
//     mpirun -np 4 tmop-enzyme-simple -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -bnd -qt 1 -qo 8
//     mpirun -np 4 pmesh-optimizer     -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -bnd -qt 1 -qo 8
//   Analytic size+alignment target:
//     mpirun -np 4 tmop-enzyme-simple -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 200 -bnd -qt 1 -qo 8
//     mpirun -np 4 pmesh-optimizer     -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 200 -bnd -qt 1 -qo 8
//   Analytic shape+alignment target:
//     mpirun -np 4 tmop-enzyme-simple -m square01.mesh -o 3 -rs 2 -mid 85 -tid 4 -ni 100 -bnd -qt 1 -qo 8 -rtol 1e-6
//     mpirun -np 4 pmesh-optimizer     -m square01.mesh -o 3 -rs 2 -mid 85 -tid 4 -ni 100 -bnd -qt 1 -qo 8 -rtol 1e-6

#include "mfem.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_ENZYME)

#include "../../fem/dfem/backends/local_qf/prelude.hpp"
#include "../../fem/dfem/backends/local_qf/revdiff_transformer.hpp"
#include "../../fem/dfem/doperator.hpp"
#include "mesh-optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace mfem;

namespace future = mfem::future;

using future::Active;
using future::Const;
using future::DerivativeOperator;
using future::DifferentiableOperator;
using future::FieldDescriptor;
using future::Identity;
using future::LocalQFBackend;
using future::RevDiff;
using future::tensor;
using future::Value;
using future::Weight;

namespace
{

static constexpr int X = 0;
static constexpr int Q = 1;
static constexpr int TARGET_W = 2;
static constexpr int REFERENCE_X = 3;
static constexpr int LIMIT_COEFF = 4;

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

template <typename scalar_t, int dim, int metric_id>
MFEM_HOST_DEVICE inline
scalar_t EvaluateTMOPMetric(const tensor<scalar_t, dim, dim> &T,
                            scalar_t shape_weight = 0.5_r)
{
   static_assert(dim == 2, "EvaluateTMOPMetric supports only 2D metrics.");

   const auto tau = det(T);
   const auto norm2 = sqnorm(T);

   if constexpr (metric_id == 2)
   {
      return 0.5_r * norm2 / tau - 1.0_r;
   }
   else if constexpr (metric_id == 14)
   {
      const auto TminusI_00 = T(0,0) - 1.0_r;
      const auto TminusI_01 = T(0,1);
      const auto TminusI_10 = T(1,0);
      const auto TminusI_11 = T(1,1) - 1.0_r;
      return TminusI_00 * TminusI_00 + TminusI_01 * TminusI_01 +
             TminusI_10 * TminusI_10 + TminusI_11 * TminusI_11;
   }
   else if constexpr (metric_id == 80)
   {
      const auto mu2 = 0.5_r * norm2 / tau - 1.0_r;
      const auto tau2 = tau * tau;
      const auto mu77 = 0.5_r * (tau2 + 1.0_r / tau2) - 1.0_r;
      return shape_weight * mu2 + (1.0_r - shape_weight) * mu77;
   }
   else if constexpr (metric_id == 85)
   {
      const auto alpha = sqrt(0.5_r * norm2);
      const auto TminusTp_00 = T(0,0) - alpha;
      const auto TminusTp_01 = T(0,1);
      const auto TminusTp_10 = T(1,0);
      const auto TminusTp_11 = T(1,1) - alpha;
      return TminusTp_00 * TminusTp_00 + TminusTp_01 * TminusTp_01 +
             TminusTp_10 * TminusTp_10 + TminusTp_11 * TminusTp_11;
   }
   else
   {
      static_assert(metric_id == 2 || metric_id == 14 || metric_id == 80 ||
                    metric_id == 85,
                    "Unsupported metric_id");
      return 0.0_r;
   }
}

template <typename scalar_t, int dim, int target_id, int metric_id>
MFEM_HOST_DEVICE inline
tensor<scalar_t, dim, dim>
TargetMatrix(const tensor<scalar_t, dim> &x,
             const tensor<real_t, dim, dim> &constant_W)
{
   tensor<scalar_t, dim, dim> W {};

   if constexpr (target_id == 1)
   {
      MFEM_CONTRACT_VAR(x);
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            W(i,j) = constant_W(i,j);
         }
      }
   }
   else if constexpr (target_id == 4)
   {
      static_assert(dim == 2, "Analytic target id 4 is 2D only.");
      MFEM_CONTRACT_VAR(constant_W);

      if constexpr (metric_id == 14)
      {
         const auto xc = x(0);
         const auto yc = x(1);
         const auto theta = M_PI * yc * (1.0_r - yc) * cos(2.0_r * M_PI * xc);
         const auto alpha_bar = 0.1_r;

         W(0,0) =  alpha_bar * cos(theta);
         W(1,0) =  alpha_bar * sin(theta);
         W(0,1) = -alpha_bar * sin(theta);
         W(1,1) =  alpha_bar * cos(theta);
      }
      else if constexpr (metric_id == 85)
      {
         auto xc = x(0) - 0.5_r;
         auto yc = x(1) - 0.5_r;
         const auto th = 22.5_r * M_PI / 180.0_r;
         const auto xn =  cos(th) * xc + sin(th) * yc;
         const auto yn = -sin(th) * xc + cos(th) * yc;
         xc = xn;
         yc = yn;

         const auto tfac = 20.0_r;
         const auto s1 = 3.0_r;
         const auto s2 = 2.0_r;
         auto wgt = tanh((tfac * yc + s2 * sin(s1 * M_PI * xc)) + 1.0_r)
                    - tanh((tfac * yc + s2 * sin(s1 * M_PI * xc)) - 1.0_r);
         if (wgt > 1.0_r) { wgt = 1.0_r; }
         if (wgt < 0.0_r) { wgt = 0.0_r; }

         xc = x(0);
         yc = x(1);
         const auto theta = M_PI * yc * (1.0_r - yc) * cos(2.0_r * M_PI * xc);
         const auto c = cos(theta);
         const auto s = sin(theta);
         const auto asp_ratio_tar = 0.1_r + (1.0_r - wgt) * (1.0_r - wgt);
         const auto inv_sqrt_asp = 1.0_r / sqrt(asp_ratio_tar);
         const auto sqrt_asp = sqrt(asp_ratio_tar);

         W(0,0) =  c * inv_sqrt_asp;
         W(1,0) =  s * inv_sqrt_asp;
         W(0,1) = -s * sqrt_asp;
         W(1,1) =  c * sqrt_asp;
      }
      else
      {
         const auto xc = x(0) - 0.5_r;
         const auto yc = x(1) - 0.5_r;
         const auto r2 = xc * xc + yc * yc;
         const auto r = (r2 > 0.0_r) ? sqrt(r2) : 0.0_r;
         const auto tan1 = tanh(30.0_r * (r - 0.15_r));
         const auto tan2 = tanh(30.0_r * (r - 0.35_r));

         W(0,0) = 0.5_r + tan1 - tan2;
         W(0,1) = 0.0_r;
         W(1,0) = 0.0_r;
         W(1,1) = 1.0_r;
      }
   }
   else
   {
      static_assert(target_id == 1 || target_id == 4,
                    "Unsupported target_id");
   }

   return W;
}

template <typename scalar_t, int dim, int target_id, int metric_id>
struct TMOPEnergy
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<scalar_t, dim> &x,
                   const tensor<scalar_t, dim, dim> &dxdr,
                   const tensor<real_t, dim> &x0,
                   const tensor<real_t, dim, dim> &constant_W,
                   const real_t &limit_coeff,
                   const real_t &w_q,
                   real_t &f) const
   {
      const auto W = TargetMatrix<scalar_t, dim, target_id, metric_id>(
                        x, constant_W);
      const auto T = dxdr * inv(W);
      const auto weight = det(W) * w_q;
      auto val = EvaluateTMOPMetric<scalar_t, dim, metric_id>(T);
      if (limit_coeff != 0.0_r)
      {
         scalar_t dist2 = 0.0_r;
         for (int d = 0; d < dim; d++)
         {
            const auto diff = x(d) - x0(d);
            dist2 += diff * diff;
         }
         val += 0.5_r * limit_coeff * dist2;
      }
      f = val * weight;
   }
};

template <typename scalar_t, int dim, int metric_id>
struct FrozenTargetTMOPEnergy
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<scalar_t, dim, dim> &dxdr,
                   const tensor<real_t, dim, dim> &W,
                   const real_t &w_q,
                   real_t &f) const
   {
      const auto T = dxdr * inv(W);
      const auto weight = det(W) * w_q;
      f = EvaluateTMOPMetric<scalar_t, dim, metric_id>(T) * weight;
   }
};

template <int dim> class EnzymeTMOPFunctional;

class SingleOutputDerivativeOperator : public Operator
{
public:
   SingleOutputDerivativeOperator(std::shared_ptr<DerivativeOperator> op,
                                  const ParFiniteElementSpace &fes)
      : Operator(fes.GetTrueVSize()),
        derivative(std::move(op))
   { }

   void Mult(const Vector &x, Vector &y) const override
   {
      MultiVector Ymv{y};
      derivative->Mult(x, Ymv);
   }

   void AssembleDiagonal(Vector &diag) const override
   {
      derivative->AssembleDiagonal(diag);
   }

private:
   std::shared_ptr<DerivativeOperator> derivative;
};

template <int dim>
class EnzymeTMOPFunctional
{
public:
   EnzymeTMOPFunctional(ParFiniteElementSpace &fes_,
                        ParMesh &pmesh,
                        const IntegrationRule &ir,
                        int target_id_,
                        int metric_id_,
                        bool exact_action_,
                        const Vector &reference_nodes_,
                        real_t limit_coeff_)
      : comm(fes_.GetComm()),
        mesh(pmesh),
        fes(fes_),
        qspace(pmesh, ir),
        target_qspace_vec(qspace, dim * dim),
        target_w(target_qspace_vec),
        qspace_vec(qspace, 1),
        limit_qdata(qspace_vec),
        q(qspace_vec),
        current_nodes(&fes_),
        reference_nodes(reference_nodes_),
        exact_action(exact_action_),
        limit_coeff(limit_coeff_)
   {
      SetTargetData();
      limit_qdata = limit_coeff;

      Array<int> all_domain_attr;
      if (mesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(mesh.attributes.Max());
         all_domain_attr = 1;
      }

      SetupTMOPOperatorsDispatch(ir, all_domain_attr, target_id_, metric_id_);
   }

   real_t Energy(const Vector &x) const
   {
      q = 0.0;
      MultiVector Xmv{x, reference_nodes, target_w, limit_qdata};
      MultiVector Qmv{q};
      energy_dop->Mult(Xmv, Qmv);

      const real_t local_energy = q.Sum();
      real_t global_energy = 0.0;
      MPI_Allreduce(&local_energy, &global_energy, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global_energy;
   }

   void Gradient(const Vector &x, Vector &g) const
   {
      g = 0.0;
      MultiVector Xmv{x, reference_nodes, target_w, limit_qdata};
      MultiVector Gmv{g};
      residual_dop->Mult(Xmv, Gmv);
   }

   std::unique_ptr<Operator> HessianOperator(const Vector &x) const
   {
      if (!exact_action)
      {
         MFEM_VERIFY(frozen_target_updater != nullptr,
                     "Frozen target updater was not initialized.");
         frozen_target_updater(*this, x);
         MultiVector Xmv{x, target_w};
         return std::make_unique<SingleOutputDerivativeOperator>(
                   hessian_residual_dop->GetDerivative(X, Xmv), fes);
      }

      MultiVector Xmv{x, reference_nodes, target_w, limit_qdata};
      return std::make_unique<SingleOutputDerivativeOperator>(
                residual_dop->GetDerivative(X, Xmv), fes);
   }

private:
   void SetupTMOPOperatorsDispatch(const IntegrationRule &ir,
                                   const Array<int> &all_domain_attr,
                                   int target_id,
                                   int metric_id)
   {
      if constexpr (dim == 2)
      {
         switch (target_id)
         {
            case 1:
               switch (metric_id)
               {
                  case 2:  return SetupTMOPOperators<1, 2>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Target id 1 supports only metric id 2.");
               }
            case 4:
               switch (metric_id)
               {
                  case 2:  return SetupTMOPOperators<4, 2>(ir, all_domain_attr);
                  case 14: return SetupTMOPOperators<4, 14>(ir, all_domain_attr);
                  case 80: return SetupTMOPOperators<4, 80>(ir, all_domain_attr);
                  case 85: return SetupTMOPOperators<4, 85>(ir, all_domain_attr);
                  default:
                     MFEM_ABORT("Unsupported 2D metric id: " << metric_id);
               }
            default:
               MFEM_ABORT("Unsupported target id: " << target_id);
         }
      }
      else if constexpr (dim == 3)
      {
         MFEM_CONTRACT_VAR(ir);
         MFEM_CONTRACT_VAR(all_domain_attr);
         MFEM_CONTRACT_VAR(target_id);
         MFEM_CONTRACT_VAR(metric_id);
         MFEM_ABORT("This miniapp currently supports only 2D target/metric "
                    "pairs.");
      }
   }

   template <int target_id_val, int metric_id_val>
   void SetupTMOPOperators(const IntegrationRule &ir,
                           const Array<int> &all_domain_attr)
   {
      {
         const std::vector<FieldDescriptor> in
         {
            FieldDescriptor{X, &fes},
            FieldDescriptor{REFERENCE_X, &fes},
            FieldDescriptor{TARGET_W, &target_qspace_vec},
            FieldDescriptor{LIMIT_COEFF, &qspace_vec}
         };
         const std::vector<FieldDescriptor> out
         {
            FieldDescriptor{Q, &qspace_vec}
         };

         energy_dop = std::make_unique<DifferentiableOperator>(in, out, mesh);
         TMOPEnergy<real_t, dim, target_id_val, metric_id_val> energy;
         energy_dop->AddDomainIntegrator<LocalQFBackend>(
            energy,
            future::tuple{Value<X>{}, future::Gradient<X>{},
                          Value<REFERENCE_X>{}, Identity<TARGET_W>{},
                          Identity<LIMIT_COEFF>{}, Weight{}},
            future::tuple{Identity<Q>{}},
            ir, all_domain_attr);
      }

      {
         const std::vector<FieldDescriptor> in
         {
            FieldDescriptor{X, &fes},
            FieldDescriptor{REFERENCE_X, &fes},
            FieldDescriptor{TARGET_W, &target_qspace_vec},
            FieldDescriptor{LIMIT_COEFF, &qspace_vec}
         };
         const std::vector<FieldDescriptor> out
         {
            FieldDescriptor{X, &fes}
         };

         residual_dop = std::make_unique<DifferentiableOperator>(in, out, mesh);
         auto derivatives = std::integer_sequence<size_t, X> {};
         const auto inputs =
            future::tuple{Value<X>{}, future::Gradient<X>{},
                          Value<REFERENCE_X>{}, Identity<TARGET_W>{},
                          Identity<LIMIT_COEFF>{}, Weight{}};
         if (exact_action)
         {
            RevDiff<TMOPEnergy<real_t, dim, target_id_val, metric_id_val>,
                    future::tuple<Active, Active, Const, Const, Const, Const>,
                    future::tuple<Active>> denergy;
            residual_dop->AddDomainIntegrator<LocalQFBackend>(
               denergy, inputs,
               future::tuple{Value<X>{}, future::Gradient<X>{}},
               ir, all_domain_attr, derivatives);
         }
         else
         {
            RevDiff<TMOPEnergy<real_t, dim, target_id_val, metric_id_val>,
                    future::tuple<Const, Active, Const, Const, Const, Const>,
                    future::tuple<Active>> denergy;
            residual_dop->AddDomainIntegrator<LocalQFBackend>(
               denergy, inputs,
               future::tuple{future::Gradient<X>{}},
               ir, all_domain_attr, derivatives);
         }
      }

      SetupFrozenTargetHessian<metric_id_val>(ir, all_domain_attr);
      frozen_target_updater =
         &CallUpdateFrozenTargetData<target_id_val, metric_id_val>;
   }

   template <int metric_id_val>
   void SetupFrozenTargetHessian(const IntegrationRule &ir,
                                 const Array<int> &all_domain_attr)
   {
      const std::vector<FieldDescriptor> in
      {
         FieldDescriptor{X, &fes},
         FieldDescriptor{TARGET_W, &target_qspace_vec}
      };
      const std::vector<FieldDescriptor> out
      {
         FieldDescriptor{X, &fes}
      };

      hessian_residual_dop =
         std::make_unique<DifferentiableOperator>(in, out, mesh);
      RevDiff<FrozenTargetTMOPEnergy<real_t, dim, metric_id_val>,
              future::tuple<Active, Const, Const>,
              future::tuple<Active>> denergy;
      auto derivatives = std::integer_sequence<size_t, X> {};
      hessian_residual_dop->AddDomainIntegrator<LocalQFBackend>(
         denergy,
         future::tuple{future::Gradient<X>{}, Identity<TARGET_W>{}, Weight{}},
         future::tuple{future::Gradient<X>{}},
         ir, all_domain_attr, derivatives);
   }

   template <int target_id_val, int metric_id_val>
   void UpdateFrozenTargetData(const Vector &x) const
   {
      current_nodes.SetFromTrueDofs(x);

      real_t *data = target_w.HostWrite();
      Vector pos(dim);
      for (int e = 0; e < qspace.GetNE(); e++)
      {
         const DenseMatrix &Wgeom =
            Geometries.GetGeomToPerfGeomJac(qspace.GetGeometry(e));
         const int offset = qspace.Offset(e);
         const int nq = qspace.GetIntRule(e).GetNPoints();
         for (int qpt = 0; qpt < nq; qpt++)
         {
            const IntegrationPoint &ip = qspace.GetIntRule(e).IntPoint(qpt);
            current_nodes.GetVectorValue(e, ip, pos);

            tensor<real_t, dim> xq {};
            tensor<real_t, dim, dim> constant_W {};
            for (int d = 0; d < dim; d++) { xq(d) = pos(d); }
            for (int i = 0; i < dim; i++)
            {
               for (int j = 0; j < dim; j++)
               {
                  constant_W(i,j) = Wgeom(i,j);
               }
            }

            const auto W =
               TargetMatrix<real_t, dim, target_id_val, metric_id_val>(
                  xq, constant_W);
            real_t *Wq = data + (offset + qpt) * dim * dim;
            for (int i = 0; i < dim; i++)
            {
               for (int j = 0; j < dim; j++)
               {
                  Wq[i * dim + j] = W(i,j);
               }
            }
         }
      }
   }

   using FrozenTargetUpdater =
      void (*)(const EnzymeTMOPFunctional<dim> &, const Vector &);

   template <int target_id_val, int metric_id_val>
   static void CallUpdateFrozenTargetData(
      const EnzymeTMOPFunctional<dim> &self, const Vector &x)
   {
      self.template UpdateFrozenTargetData<target_id_val, metric_id_val>(x);
   }

   void SetTargetData()
   {
      const int vdim = dim * dim;
      real_t *data = target_w.HostWrite();
      for (int e = 0; e < qspace.GetNE(); e++)
      {
         const DenseMatrix &W =
            Geometries.GetGeomToPerfGeomJac(qspace.GetGeometry(e));
         MFEM_VERIFY(W.Height() == dim && W.Width() == dim,
                     "Unexpected target matrix dimension.");
         const int offset = qspace.Offset(e);
         const int nq = qspace.GetIntRule(e).GetNPoints();
         for (int qpt = 0; qpt < nq; qpt++)
         {
            real_t *Wq = data + (offset + qpt) * vdim;
            for (int i = 0; i < dim; i++)
            {
               for (int j = 0; j < dim; j++)
               {
                  Wq[i * dim + j] = W(i, j);
               }
            }
         }
      }
   }

   MPI_Comm comm;
   ParMesh &mesh;
   ParFiniteElementSpace &fes;
   QuadratureSpace qspace;
   VectorQuadratureSpace target_qspace_vec;
   mutable QuadratureFunction target_w;
   VectorQuadratureSpace qspace_vec;
   QuadratureFunction limit_qdata;
   mutable QuadratureFunction q;
   mutable ParGridFunction current_nodes;
   Vector reference_nodes;
   bool exact_action;
   real_t limit_coeff;
   FrozenTargetUpdater frozen_target_updater = nullptr;
   std::unique_ptr<DifferentiableOperator> energy_dop;
   std::unique_ptr<DifferentiableOperator> residual_dop;
   std::unique_ptr<DifferentiableOperator> hessian_residual_dop;
};

template <int dim>
class EnzymeTMOPNonlinearForm : public ParNonlinearForm
{
public:
   EnzymeTMOPNonlinearForm(ParFiniteElementSpace &fes,
                           const EnzymeTMOPFunctional<dim> &functional_)
      : ParNonlinearForm(&fes),
        functional(functional_),
        x_abs(fes.GetTrueVSize())
   {
      reference_true.SetSize(fes.GetTrueVSize());
   }

   void SetReference(const Vector &x0)
   {
      reference_true = x0;
   }

   real_t GetEnergy(const Vector &dx) const override
   {
      add(reference_true, dx, x_abs);
      return functional.Energy(x_abs);
   }

   void Mult(const Vector &dx, Vector &y) const override
   {
      add(reference_true, dx, x_abs);
      functional.Gradient(x_abs, y);
      const Array<int> &ess_tdofs = GetEssentialTrueDofs();
      if (ess_tdofs.Size() > 0) { y.SetSubVector(ess_tdofs, 0.0); }
   }

   Operator &GetGradient(const Vector &dx) const override
   {
      add(reference_true, dx, x_abs);
      hessian = functional.HessianOperator(x_abs);
      const Array<int> &ess_tdofs = GetEssentialTrueDofs();
      constrained_hessian =
         std::make_unique<ConstrainedOperator>(hessian.get(), ess_tdofs, false);
      return *constrained_hessian;
   }

private:
   const EnzymeTMOPFunctional<dim> &functional;
   Vector reference_true;
   mutable Vector x_abs;
   mutable std::unique_ptr<Operator> hessian;
   mutable std::unique_ptr<ConstrainedOperator> constrained_hessian;
};

real_t MinimumDetJ(ParMesh &pmesh,
                   const ParFiniteElementSpace &pfes,
                   IntegrationRules &irules,
                   int quad_order)
{
   real_t min_detJ = infinity();
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const IntegrationRule &ir = irules.Get(pfes.GetFE(e)->GetGeomType(),
                                             quad_order);
      ElementTransformation *trans = pmesh.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         trans->SetIntPoint(&ir.IntPoint(q));
         min_detJ = std::min(min_detJ, trans->Jacobian().Det());
      }
   }

   real_t global_min_detJ = 0.0;
   MPI_Allreduce(&min_detJ, &global_min_detJ, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh.GetComm());
   return global_min_detJ;
}

void SaveMesh(ParMesh &pmesh, const char *filename)
{
   std::ofstream mesh_ofs(filename);
   mesh_ofs.precision(8);
   pmesh.PrintAsOne(mesh_ofs);
}

struct TMOPVisualizationData
{
   std::unique_ptr<TMOP_QualityMetric> metric;
   std::unique_ptr<TargetConstructor> target;
   std::unique_ptr<TMOPMatrixCoefficient> analytic_target_coeff;
};

TMOPVisualizationData MakeVisualizationData(int dim,
                                            int metric_id,
                                            int target_id,
                                            ParGridFunction &nodes)
{
   TMOPVisualizationData data;

   if (dim == 2 && metric_id == 14)
   {
      data.metric = std::make_unique<TMOP_Metric_014>();
   }
   else if (dim == 2 && metric_id == 80)
   {
      data.metric = std::make_unique<TMOP_Metric_080>(0.5);
   }
   else if (dim == 2 && metric_id == 85)
   {
      data.metric = std::make_unique<TMOP_Metric_085>();
   }
   else if (dim == 2)
   {
      data.metric = std::make_unique<TMOP_Metric_002>();
   }
   else
   {
      MFEM_ABORT("Visualization is implemented only for 2D meshes.");
   }

   if (target_id == 1)
   {
      data.target = std::make_unique<TargetConstructor>(
                       TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                       nodes.ParFESpace()->GetComm());
   }
   else if (target_id == 4)
   {
      auto target = std::make_unique<AnalyticAdaptTC>(
                       TargetConstructor::GIVEN_FULL);
      data.analytic_target_coeff =
         std::make_unique<HessianCoefficient>(dim, metric_id);
      target->SetAnalyticTargetSpec(nullptr, nullptr,
                                    data.analytic_target_coeff.get());
      data.target = std::move(target);
   }
   else
   {
      MFEM_ABORT("Unsupported visualization target id: " << target_id);
   }

   data.target->SetNodes(nodes);
   return data;
}

void VisualizeMetricValues(int mesh_poly_deg,
                           TMOPVisualizationData &vis_data,
                           ParMesh &pmesh,
                           ParGridFunction &nodes,
                           const char *title,
                           int position)
{
   vis_data.target->SetNodes(nodes);
   vis_tmop_metric_p(mesh_poly_deg, *vis_data.metric, *vis_data.target,
                     pmesh, const_cast<char *>(title), position);
}

void VisualizeField(ParMesh &pmesh,
                    ParGridFunction &field,
                    const char *title,
                    int x,
                    int y,
                    int w = 600,
                    int h = 600)
{
   socketstream sock;
   if (Mpi::Root())
   {
      sock.open("localhost", 19916);
      sock.precision(8);
      sock << "solution\n";
   }

   pmesh.PrintAsOne(sock);
   field.SaveAsOne(sock);

   if (Mpi::Root())
   {
      sock << "window_title '" << title << "'\n"
           << "window_geometry " << x << " " << y << " " << w << " " << h
           << "\n"
           << "keys jRmclA\n" << std::flush;
   }
}

void GetMeshOptimizerEssentialTrueDofs(const ParFiniteElementSpace &pfes,
                                       bool move_bnd,
                                       Array<int> &ess_tdofs)
{
   ess_tdofs.DeleteAll();
   const ParMesh *pmesh = pfes.GetParMesh();
   if (pmesh->bdr_attributes.Size() == 0) { return; }

   if (!move_bnd)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
      return;
   }

   const int dim = pmesh->Dimension();
   int n = 0;
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      const int nd = pfes.GetBE(i)->GetDof();
      const int attr = pmesh->GetBdrElement(i)->GetAttribute();
      MFEM_VERIFY(!(dim == 2 && attr == 3),
                  "Boundary attribute 3 must be used only for 3D meshes. "
                  "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                  "components, rest for free nodes), or use -fix-bnd.");
      if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
      if (attr == 4) { n += nd * dim; }
   }

   Array<int> vdofs, ess_vdofs(n);
   n = 0;
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      const int nd = pfes.GetBE(i)->GetDof();
      const int attr = pmesh->GetBdrElement(i)->GetAttribute();
      pfes.GetBdrElementVDofs(i, vdofs);
      if (attr == 1)
      {
         for (int j = 0; j < nd; j++) { ess_vdofs[n++] = vdofs[j]; }
      }
      else if (attr == 2)
      {
         for (int j = 0; j < nd; j++) { ess_vdofs[n++] = vdofs[j + nd]; }
      }
      else if (attr == 3)
      {
         for (int j = 0; j < nd; j++) { ess_vdofs[n++] = vdofs[j + 2 * nd]; }
      }
      else if (attr == 4)
      {
         for (int j = 0; j < vdofs.Size(); j++) { ess_vdofs[n++] = vdofs[j]; }
      }
   }

   Array<int> ess_vdof_marker, ess_tdof_marker;
   FiniteElementSpace::ListToMarker(ess_vdofs, pfes.GetVSize(),
                                    ess_vdof_marker);
   ess_tdof_marker.SetSize(pfes.GetTrueVSize());
   pfes.Dof_TrueDof_Matrix()->BooleanMultTranspose(
      1, ess_vdof_marker, 0, ess_tdof_marker);
   FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdofs);
}

IntegrationRules &SelectIntegrationRules(int quad_type)
{
   switch (quad_type)
   {
      case 1: return IntRulesLo;
      case 2: return IntRules;
      case 3: return IntRulesCU;
      default:
         MFEM_ABORT("Unknown quadrature rule type: " << quad_type);
         return IntRules;
   }
}

template <int dim>
int RunOptimizer(ParMesh &pmesh,
                 ParFiniteElementSpace &pfes,
                 ParGridFunction &x,
                 IntegrationRules &irules,
                 int quad_order,
                 const Array<int> &ess_tdofs,
                 real_t min_detJ,
                 int solver_iter,
                 real_t solver_rtol,
                 real_t solver_atol,
                 int lin_solver,
                 int solver_art_type,
                 int max_lin_iter,
                 int target_id,
                 int metric_id,
                 bool exact_action,
                 real_t limit_const,
                 int verbosity)
{
   Vector Xtrue(pfes.GetTrueVSize());
   x.GetTrueDofs(Xtrue);

   const IntegrationRule &ir =
      irules.Get(pmesh.GetTypicalElementGeometry(), quad_order);
   EnzymeTMOPFunctional<dim> functional(pfes, pmesh, ir, target_id, metric_id,
                                        exact_action, Xtrue, limit_const);
   const real_t init_energy = functional.Energy(Xtrue);

   EnzymeTMOPNonlinearForm<dim> oper(pfes, functional);
   oper.SetEssentialTrueDofs(ess_tdofs);
   oper.SetReference(Xtrue);

#ifdef MFEM_USE_SINGLE
   const real_t linsol_rtol = 1e-5;
#else
   const real_t linsol_rtol = 1e-12;
#endif
   IterativeSolver::PrintLevel linear_print;
   if (verbosity > 1) { linear_print.Errors().Warnings().FirstAndLast(); }
   if (verbosity > 2) { linear_print.Errors().Warnings().Iterations(); }
   MFEM_VERIFY(lin_solver == 2 || lin_solver == 3,
               "Only MINRES and MINRES with Jacobi are supported");
   MINRESSolver linear_solver(pfes.GetComm());
   linear_solver.SetMaxIter(max_lin_iter);
   linear_solver.SetRelTol(linsol_rtol);
   linear_solver.SetAbsTol(0.0);
   linear_solver.SetPrintLevel(linear_print);
   OperatorJacobiSmoother jacobi;
   if (lin_solver == 3)
   {
      jacobi.SetPositiveDiagonal(true);
      linear_solver.SetPreconditioner(jacobi);
   }

   TMOPNewtonSolver solver(pfes.GetComm(), ir, 0);
   solver.SetIntegrationRules(irules, quad_order);
   solver.SetMinDetPtr(&min_detJ);
   solver.SetOperator(oper);
   solver.SetPreconditioner(linear_solver);
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(solver_atol);
   if (solver_art_type > 0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   IterativeSolver::PrintLevel newton_print;
   if (verbosity > 0) { newton_print.Errors().Warnings().Iterations(); }
   else               { newton_print.Errors().Warnings(); }
   solver.SetPrintLevel(newton_print);

   Vector zero;
   solver.Mult(zero, Xtrue);

   x.SetFromTrueDofs(Xtrue);
   pmesh.SetNodalGridFunction(&x);
   pmesh.NodesUpdated();
   pmesh.ExchangeFaceNbrData();

   const real_t final_energy = functional.Energy(Xtrue);
   const bool converged = solver.GetConverged();

   if (Mpi::Root())
   {
      std::cout << std::scientific << std::setprecision(4);
      std::cout << "Initial strain energy: " << init_energy << '\n';
      std::cout << "  Final strain energy: " << final_energy << '\n';
      if (init_energy != 0.0)
      {
         std::cout << "The strain energy decreased by: "
                   << (init_energy - final_energy) * 100.0 / init_energy
                   << " %.\n";
      }
   }

   return converged ? 0 : 2;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "square01.mesh";
   int mesh_poly_deg = 2;
   int metric_id = 0;
   int target_id = 1;
   int rs_levels = 0;
   int rp_levels = 0;
   int quad_type = 1;
   int quad_order = 8;
   int solver_iter = 20;
#ifdef MFEM_USE_SINGLE
   real_t solver_rtol = 1e-4;
#else
   real_t solver_rtol = 1e-10;
#endif
   real_t solver_atol = 0.0;
   int max_lin_iter = 100;
   int lin_solver = 2;
   int solver_art_type = 0;
   bool move_bnd = true;
   bool visualization = true;
   bool exact_action = false;
   real_t limit_const = 0.0;
   int mesh_node_order = Ordering::byNODES;
   int verbosity = 1;
   const char *devopt = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Metric id. Use 0 for mu2.\n\t"
                  "Supported pairs: -tid 1 with -mid 2; "
                  "-tid 4 with -mid 2, 14, 80, or 85.");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target type:\n\t"
                  "1: Constant ideal target matrix\n\t"
                  "4: Given full analytic Jacobian in physical space");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad-order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&solver_atol, "-atol", "--newton-abs-tolerance",
                  "Absolute tolerance for the Newton solver.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver:\n\t"
                  "2: MINRES\n\t"
                  "3: MINRES + Jacobi preconditioner.");
   args.AddOption(&solver_art_type, "-art", "--adaptive-rel-tol",
                  "Type of adaptive relative linear solver tolerance:\n\t"
                  "0: None (default)\n\t"
                  "1: Eisenstat-Walker type 1\n\t"
                  "2: Eisenstat-Walker type 2");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary",
                  "-fix-bnd", "--fix-boundary",
                  "Allow boundary motion with component constraints, or fix "
                  "all boundary nodes.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&exact_action, "-ex", "--exact_action", "-no-ex",
                  "--no-exact-action",
                  "Include exact target-coordinate derivative terms for "
                  "physical-space targets.");
   args.AddOption(&limit_const, "-lc", "--limit-const",
                  "Node limiting constant. Requires -ex in this miniapp.");
   args.AddOption(&mesh_node_order, "-mno", "--mesh-node-ordering",
                  "Ordering of mesh nodes: 0 byNODES, 1 byVDIM.");
   args.AddOption(&verbosity, "-vl", "--verbosity-level",
                  "Verbosity level: 0 none, 1 Newton, 2 linear summaries, "
                  "3 linear iterations.");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(std::cout); }

   Device device(devopt);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2, "This miniapp currently supports only 2D meshes.");
   if (mesh_poly_deg <= 0) { mesh_poly_deg = 2; }

   const int active_metric_id = (metric_id == 0) ? 2 : metric_id;

   MFEM_VERIFY(target_id == 1 || target_id == 4,
               "This miniapp supports target ids 1 and 4 only.");
   const bool target_metric_ok =
      (target_id == 1 && active_metric_id == 2) ||
      (target_id == 4 && (active_metric_id == 2 || active_metric_id == 14 ||
                          active_metric_id == 80 ||
                          active_metric_id == 85));
   MFEM_VERIFY(target_metric_ok,
               "Supported pairs: -tid 1 with -mid 2; "
               "-tid 4 with -mid 2, 14, 80, or 85.");
   MFEM_VERIFY(solver_art_type >= 0 && solver_art_type <= 2,
               "Unknown adaptive relative tolerance option: "
               << solver_art_type);
   MFEM_VERIFY(limit_const >= 0.0,
               "Node limiting constant must be nonnegative.");
   MFEM_VERIFY(limit_const == 0.0 || exact_action,
               "Node limiting in tmop-enzyme-simple requires -ex.");

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec, dim, mesh_node_order);

   pmesh.SetNodalFESpace(&pfes);
   ParGridFunction x(&pfes);
   pmesh.SetNodalGridFunction(&x);
   std::unique_ptr<ParGridFunction> x0;
   std::unique_ptr<TMOPVisualizationData> vis_data;
   if (visualization)
   {
      x0 = std::make_unique<ParGridFunction>(x);
      vis_data = std::make_unique<TMOPVisualizationData>(
                    MakeVisualizationData(dim, active_metric_id, target_id, x));
   }

   IntegrationRules &irules = SelectIntegrationRules(quad_type);

   const real_t min_detJ = MinimumDetJ(pmesh, pfes, irules, quad_order);
   if (Mpi::Root())
   {
      std::cout << "Minimum det(J) of the original mesh is "
                << min_detJ << '\n';
      const char *target_descr =
         (target_id == 1) ? "constant ideal target W" :
         (active_metric_id == 14)
         ? "analytic size+alignment target W" :
         (active_metric_id == 85) ? "analytic shape+alignment target W" :
         "analytic annular shape target W";
      std::cout << "Using " << target_descr
                << " and metric mu" << active_metric_id << ".\n";
      if (limit_const != 0.0)
      {
         std::cout << "Using quadratic node limiting with coefficient "
                   << limit_const << ".\n";
      }
   }
   MFEM_VERIFY(min_detJ > 0.0, "The input mesh is inverted.");

   Array<int> ess_tdofs;
   GetMeshOptimizerEssentialTrueDofs(pfes, move_bnd, ess_tdofs);
   if (Mpi::Root())
   {
      std::cout << "Fixed true dofs: " << ess_tdofs.Size() << '\n';
   }

   SaveMesh(pmesh, "perturbed.mesh");
   if (visualization)
   {
      VisualizeMetricValues(mesh_poly_deg, *vis_data, pmesh, x,
                            "Initial metric values", 0);
   }

   const int result = RunOptimizer<2>(pmesh, pfes, x,
                                      irules, quad_order,
                                      ess_tdofs, min_detJ,
                                      solver_iter, solver_rtol, solver_atol,
                                      lin_solver, solver_art_type,
                                      max_lin_iter, target_id,
                                      active_metric_id, exact_action,
                                      limit_const, verbosity);

   SaveMesh(pmesh, "optimized.mesh");
   if (visualization)
   {
      VisualizeMetricValues(mesh_poly_deg, *vis_data, pmesh, x,
                            "Final metric values", 600);
      *x0 -= x;
      VisualizeField(pmesh, *x0, "Displacements", 1200, 0);
   }

   return result;
}

#else

int main(int, char *[])
{
   mfem::err << "tmop-enzyme-simple requires MFEM_USE_MPI=YES and "
             << "MFEM_USE_ENZYME=YES.\n";
   return MFEM_SKIP_RETURN_VALUE;
}

#endif // MFEM_USE_MPI && MFEM_USE_ENZYME
