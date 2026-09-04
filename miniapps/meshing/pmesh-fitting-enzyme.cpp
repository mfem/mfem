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
//    Enzyme Boundary and Interface Fitting Miniapp
//    ---------------------------------------------------------------------
//
// This miniapp is the Enzyme/dFEM counterpart of pmesh-fitting. The TMOP and
// level-set fitting energies are expressed as differentiable operators; their
// residuals and Hessian actions are generated with Enzyme.
//
// Compile with: make pmesh-fitting-enzyme
//
// Sample runs and corresponding pmesh-fitting runs:
// The third command in each case evaluates the level set directly with Enzyme
// and performs no initial- or background-mesh interpolation.
//   Adaptive surface-fitting coefficient (2D):
//     mpirun -np 4 pmesh-fitting         -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -sfa 2 -sft 1e-4 -sfcmax 1e4 -ni 40 -rtol 1e-10 -resid -ae 1
//     mpirun -np 4 pmesh-fitting-enzyme  -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -sfa 2 -sft 1e-4 -sfcmax 1e4 -ni 40 -rtol 1e-10 -resid -dls -dder 1
//     mpirun -np 4 pmesh-fitting-enzyme  -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -sfa 2 -sft 1e-4 -sfcmax 1e4 -ni 40 -rtol 1e-10 -resid -als
//   Initial-mesh interpolation versus direct analytic evaluation (2D):
//     mpirun -np 4 pmesh-fitting         -o 3 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -ae 1
//     mpirun -np 4 pmesh-fitting-enzyme  -o 3 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -dls -dder 1
//     mpirun -np 4 pmesh-fitting-enzyme  -o 3 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -als
// Use -dder 2 instead for element-local discrete derivatives.
//   Background-mesh interpolation versus direct analytic evaluation (2D):
//     mpirun -np 4 pmesh-fitting         -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -rtol 1e-6 -ae 1 -sbgmesh
//     mpirun -np 4 pmesh-fitting-enzyme  -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -rtol 1e-6 -dls -sbgmesh
//     mpirun -np 4 pmesh-fitting-enzyme  -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -rtol 1e-6 -als
//   Spherical level-set fitting (3D):
//     mpirun -np 4 pmesh-fitting         -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -ae 1 -slstype 4
//     mpirun -np 4 pmesh-fitting-enzyme  -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -dls -slstype 4 -dder 1
//     mpirun -np 4 pmesh-fitting-enzyme  -m cube.mesh -rs 2 -o 2 -mid 303 -tid 1 -vl 1 -sfc 5e3 -rtol 1e-5 -als -slstype 4

#include "mfem.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_ENZYME) && \
    defined(MFEM_USE_GSLIB)

#include "../../fem/dfem/backends/local_qf/prelude.hpp"
#include "../../fem/dfem/backends/local_qf/revdiff_transformer.hpp"
#include "../../fem/dfem/doperator.hpp"
#include "mesh-fitting.hpp"

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

using future::DerivativeOperator;
using future::DifferentiableOperator;
using future::FieldDescriptor;
using future::FunctionalValue;
using future::Identity;
using future::LocalQFBackend;
using future::tensor;
using future::Value;
using future::Weight;

namespace pmesh_fitting_enzyme
{

static constexpr int X = 0;
static constexpr int Q = 1;
static constexpr int TARGET_W_INV = 2;
static constexpr int TARGET_DET_W = 3;
static constexpr int SURFACE_FIT_DATA = 4;

template <int dim>
struct SurfaceFitDataLayout
{
   static constexpr int COEFFICIENT = 0;
   static constexpr int VALUE = 1;
   static constexpr int GRADIENT = VALUE + 1;
   static constexpr int HESSIAN = GRADIENT + dim;
   static constexpr int SIZE = HESSIAN + dim * dim;
};

/// Compute the MPI-global Euclidean norm of a distributed vector.
real_t GlobalVectorNorm(MPI_Comm comm, const Vector &x)
{
   const real_t local_norm2 = x * x;
   real_t global_norm2 = 0.0;
   MPI_Allreduce(&local_norm2, &global_norm2, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   return std::sqrt(global_norm2);
}

/// Evaluate the selected TMOP quality metric.
template <typename scalar_t, int dim, int metric_id>
MFEM_HOST_DEVICE inline
scalar_t EvaluateTMOPMetric(const tensor<scalar_t, dim, dim> &T)
{
   const auto tau = det(T);
   const auto norm2 = sqnorm(T);

   if constexpr (dim == 2 && metric_id == 2)
   {
      return 0.5_r * norm2 / tau - 1.0_r;
   }
   else if constexpr (dim == 2 && metric_id == 58)
   {
      const auto i1b = norm2 / tau;
      return i1b * (i1b - 2.0_r);
   }
   else if constexpr (dim == 2 && metric_id == 80)
   {
      const auto mu2 = 0.5_r * norm2 / tau - 1.0_r;
      const auto tau2 = tau * tau;
      const auto mu77 = 0.5_r * (tau2 + 1.0_r / tau2) - 1.0_r;
      return 0.5_r * (mu2 + mu77);
   }
   else if constexpr (dim == 3 && metric_id == 303)
   {
      // mu_303 = |J|^2 / 3 / tau^(2/3) - 1
      return norm2 / (3.0_r * pow(tau, 2.0_r/3.0_r)) - 1.0_r;
   }
   else
   {
      static_assert((dim == 2 &&
                     (metric_id == 2 || metric_id == 58 || metric_id == 80)) ||
                    (dim == 3 && metric_id == 303),
                    "Unsupported TMOP metric/dimension combination");
      return 0.0_r;
   }
}

template <typename scalar_t, int dim, int metric_id>
struct TMOPEnergy
{
   /// Evaluate the weighted TMOP energy at one quadrature point.
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<scalar_t, dim, dim> &dxdr,
                   const tensor<real_t, dim, dim> &W_inv,
                   const real_t &det_W,
                   const real_t &w_q,
                   real_t &f) const
   {
      const auto T = dxdr * W_inv;
      f = EvaluateTMOPMetric<scalar_t, dim, metric_id>(T) * det_W * w_q;
   }
};

struct SurfaceFittingOptions
{
   enum LevelSetSource
   {
      ANALYTIC = 0,
      DISCRETE = 1
   };

   enum AnalyticLevelSet
   {
      CIRCLE = 1,
      SQUIRCLE = 3,
      SPHERE = 4
   };

   enum DiscreteDerivativeMode
   {
      INTERPOLATED_SOURCE = 1,
      ELEMENT_LOCAL = 2
   };

   LevelSetSource source = ANALYTIC;
   AnalyticLevelSet analytic_level_set = CIRCLE;
   DiscreteDerivativeMode discrete_derivative_mode = INTERPOLATED_SOURCE;
   const ParGridFunction *discrete_level_set = nullptr;
   bool discrete_from_background = false;
   const Array<bool> *marker = nullptr;
   real_t coefficient = 0.0;
};

template <int dim>
MFEM_HOST_DEVICE inline
void get_sigma_impl(const real_t *x, const real_t *data, real_t *sigma)
{
   // The volatile zero keeps Enzyme from classifying x as unused before it
   // encounters the custom derivative, without changing the sampled value.
   volatile real_t active_x = x[0];
   const real_t zero = active_x - active_x;
   *sigma = data[SurfaceFitDataLayout<dim>::VALUE] + zero;
}

/// Return a primal value while contributing no derivative to Enzyme.
MFEM_HOST_DEVICE MFEM_ENZYME_INACTIVE
__attribute__((noinline, optnone))
real_t StopGradient(real_t value)
{
   return value;
}

template <int dim>
MFEM_HOST_DEVICE inline
void *get_sigma_aug_impl(const real_t *x, real_t *dx,
                         const real_t *data, real_t *ddata,
                         real_t *sigma, real_t *dsigma)
{
   // These differences are identically zero in the primal. When this rule is
   // forward-differentiated for the Hessian, their tangents are dx.
   *sigma = data[SurfaceFitDataLayout<dim>::VALUE];
   for (int d = 0; d < dim; d++)
   {
      const real_t delta = x[d] - StopGradient(x[d]);
      *sigma += data[SurfaceFitDataLayout<dim>::GRADIENT + d] * delta;
   }
   return nullptr;
}

template <int dim>
MFEM_HOST_DEVICE inline
void get_sigma_rev_impl(const real_t *x, real_t *dx,
                        const real_t *data, real_t *ddata,
                        const real_t *sigma, const real_t *dsigma,
                        void *tape)
{
   // The primal differences vanish, so grad is exactly the sampled gradient.
   // Nested forward differentiation exposes the symmetric Hessian action.
   real_t delta[dim];
   for (int d = 0; d < dim; d++)
   {
      delta[d] = x[d] - StopGradient(x[d]);
   }
   for (int i = 0; i < dim; i++)
   {
      real_t grad = data[SurfaceFitDataLayout<dim>::GRADIENT + i];
      for (int j = 0; j < dim; j++)
      {
         // Match classic TMOP by mirroring the stored upper triangle.
         const int row = i < j ? i : j;
         const int col = i < j ? j : i;
         grad += data[SurfaceFitDataLayout<dim>::HESSIAN + row * dim + col] *
                 delta[j];
      }
      dx[i] += *dsigma * grad;
   }
}

/// Dimension-specific custom-rule entry points must not be inlined before
/// Enzyme discovers their registrations.
MFEM_HOST_DEVICE __attribute__((noinline))
void get_sigma_2d(const real_t *x, const real_t *data, real_t *sigma)
{
   get_sigma_impl<2>(x, data, sigma);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void *get_sigma_2d_aug(const real_t *x, real_t *dx,
                       const real_t *data, real_t *ddata,
                       real_t *sigma, real_t *dsigma)
{
   return get_sigma_aug_impl<2>(x, dx, data, ddata, sigma, dsigma);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void get_sigma_2d_rev(const real_t *x, real_t *dx,
                      const real_t *data, real_t *ddata,
                      const real_t *sigma, const real_t *dsigma,
                      void *tape)
{
   get_sigma_rev_impl<2>(x, dx, data, ddata, sigma, dsigma, tape);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void get_sigma_3d(const real_t *x, const real_t *data, real_t *sigma)
{
   get_sigma_impl<3>(x, data, sigma);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void *get_sigma_3d_aug(const real_t *x, real_t *dx,
                       const real_t *data, real_t *ddata,
                       real_t *sigma, real_t *dsigma)
{
   return get_sigma_aug_impl<3>(x, dx, data, ddata, sigma, dsigma);
}

MFEM_HOST_DEVICE __attribute__((noinline))
void get_sigma_3d_rev(const real_t *x, real_t *dx,
                      const real_t *data, real_t *ddata,
                      const real_t *sigma, const real_t *dsigma,
                      void *tape)
{
   get_sigma_rev_impl<3>(x, dx, data, ddata, sigma, dsigma, tape);
}

template <typename scalar_t, int dim>
struct DiscreteSurfaceFittingEnergy
{
   /// Evaluate the fitting penalty from the sampled level-set value.
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<scalar_t, dim> &x,
                   const tensor<scalar_t, SurfaceFitDataLayout<dim>::SIZE> &data,
                   real_t &f) const
   {
      scalar_t sigma;
      if constexpr (dim == 2)
      {
         get_sigma_2d(&x[0], &data[0], &sigma);
      }
      else
      {
         get_sigma_3d(&x[0], &data[0], &sigma);
      }
      f = data(SurfaceFitDataLayout<dim>::COEFFICIENT) * sigma * sigma;
   }
};

void *__enzyme_register_gradient_get_sigma_2d[3] =
{
   (void *)&get_sigma_2d,
   (void *)&get_sigma_2d_aug,
   (void *)&get_sigma_2d_rev
};

void *__enzyme_register_gradient_get_sigma_3d[3] =
{
   (void *)&get_sigma_3d,
   (void *)&get_sigma_3d_aug,
   (void *)&get_sigma_3d_rev
};

template <typename scalar_t, int level_set, int dim>
struct AnalyticSurfaceFittingEnergy
{
   /// Evaluate the analytic fitting penalty for Enzyme differentiation.
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<scalar_t, dim> &x,
                   const scalar_t &coeff,
                   real_t &f) const
   {
      // Avoid evaluating at unmarked nodes
      if (coeff == 0.0_r)
      {
         f = 0.0_r;
         return;
      }

      const scalar_t xc = x(0) - 0.5_r;
      const scalar_t yc = x(1) - 0.5_r;
      scalar_t sigma;
      if constexpr (level_set == SurfaceFittingOptions::CIRCLE)
      {
         static_assert(dim == 2, "The analytic circle is two-dimensional");
         const scalar_t r = sqrt(xc * xc + yc * yc + 1e-12_r);
         sigma = r - 0.25_r;
      }
      else if constexpr (level_set == SurfaceFittingOptions::SPHERE)
      {
         static_assert(dim == 3, "The analytic sphere is three-dimensional");
         const scalar_t zc = x(2) - 0.5_r;
         const scalar_t r = sqrt(xc * xc + yc * yc + zc * zc + 1e-12_r);
         sigma = r - 0.25_r;
      }
      else
      {
         static_assert(level_set == SurfaceFittingOptions::SQUIRCLE,
                       "Unsupported analytic level set");
         const scalar_t xc2 = xc * xc;
         const scalar_t yc2 = yc * yc;
         sigma = xc2 * xc2 + yc2 * yc2 -
                 0.24_r * 0.24_r * 0.24_r * 0.24_r;
      }
      f = coeff * sigma * sigma;
   }
};

template <int dim>
class SurfaceFittingData
{
public:
   /// Initialize current- and source-mesh fields used by surface fitting.
   SurfaceFittingData(ParMesh &pmesh,
                      ParFiniteElementSpace &mesh_fes,
                      const SurfaceFittingOptions &options)
      : mesh(pmesh),
        order(GetH1Order(mesh_fes)),
        basis(GetH1BasisType(mesh_fes)),
        current_fec(order, dim, basis),
        current_fes(&pmesh, &current_fec),
        current_sigma(&current_fes),
        marker(*options.marker),
        coefficient(options.coefficient),
        source(options.source),
        analytic_level_set(options.analytic_level_set),
        discrete_derivative_mode(options.discrete_derivative_mode),
        discrete_from_background(options.discrete_from_background)
   {
      MFEM_VERIFY(pmesh.Dimension() == dim,
                  "Surface-fitting template dimension does not match mesh.");
      MFEM_VERIFY(options.marker != nullptr,
                  "Surface fitting requires a DOF marker.");
      MFEM_VERIFY(coefficient > 0.0,
                  "Surface fitting requires a positive coefficient.");
      MFEM_VERIFY(marker.Size() == current_fes.GetVSize(),
                  "Surface fitting marker size does not match the scalar "
                  "node space.");

      ParGridFunction counter(&current_fes);
      counter.CountElementsPerVDof(dof_count);

      // Cache marked DOF indices for efficient selective interpolation
      const int ndofs = current_fes.GetVSize();
      marked_dof_indices.Reserve(ndofs);
      for (int i = 0; i < ndofs; i++)
      {
         if (marker[i])
         {
            marked_dof_indices.Append(i);
         }
      }

      if (source == SurfaceFittingOptions::DISCRETE)
      {
         MFEM_VERIFY(options.discrete_level_set != nullptr,
                     "Discrete fitting requires an initial level set.");
         SetupDiscreteLevelSet(*options.discrete_level_set);
      }
   }

   /// Refresh discrete fields and pack their nodal data for the Q-function.
   void FillQuadratureData(const ParGridFunction &nodes,
                           const QuadratureSpace &node_qspace,
                           QuadratureFunction &qdata) const
   {
      UpdateCurrentNodes(nodes);
      if (source == SurfaceFittingOptions::DISCRETE)
      {
         UpdateDiscreteSamples();
      }

      real_t *qdata_ptr = qdata.HostWrite();
      Array<int> dofs;

      for (int e = 0; e < node_qspace.GetNE(); e++)
      {
         const IntegrationRule &ir = node_qspace.GetIntRule(e);
         current_fes.GetElementDofs(e, dofs);
         MFEM_VERIFY(dofs.Size() == ir.GetNPoints(),
                     "Nodal quadrature must match scalar mesh DOFs.");
         const int offset = node_qspace.Offset(e);

         const FiniteElement *fe = current_fes.GetFE(e);
         const auto *nfe = dynamic_cast<const NodalFiniteElement *>(fe);
         const Array<int> *lex_to_native =
            nfe && nfe->GetLexicographicOrdering().Size() > 0 ?
            &nfe->GetLexicographicOrdering() : nullptr;

         DenseMatrix element_grad, element_hess;
         bool use_element_derivatives =
            !discrete_from_background &&
            discrete_derivative_mode == SurfaceFittingOptions::ELEMENT_LOCAL;
         if (use_element_derivatives)
         {
            bool has_marked_dof = false;
            for (int i = 0; i < dofs.Size(); i++)
            {
               has_marked_dof = has_marked_dof || marker[dofs[i]];
            }
            use_element_derivatives = has_marked_dof;
         }
         if (use_element_derivatives)
         {
            ComputeElementDerivatives(e, element_grad, element_hess);
         }

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const int local_dof = lex_to_native ? (*lex_to_native)[q] : q;
            const int dof = dofs[local_dof];
            real_t *data =
               qdata_ptr + (offset + q) * SurfaceFitDataLayout<dim>::SIZE;
            FillNodeData(dof, data);
            if (use_element_derivatives)
            {
               FillElementDerivativeData(local_dof, element_grad,
                                         element_hess, data);
            }
         }
      }
   }

   /// Move the working mesh to the supplied nodal coordinates.
   void UpdateCurrentNodes(const ParGridFunction &nodes) const
   {
      GridFunction *mesh_nodes = mesh.GetNodes();
      MFEM_VERIFY(mesh_nodes && mesh_nodes->Size() == nodes.Size(),
                  "Current mesh nodes are incompatible with the fitting "
                  "space.");
      *mesh_nodes = nodes;
      mesh.NodesUpdated();
      mesh.ExchangeFaceNbrData();
      MFEM_VERIFY(nodes.Size() == dim * current_fes.GetVSize(),
                  "Mesh and level-set nodal spaces must have the same order.");
      current_node_pos = nodes;
      Ordering::Reorder(current_node_pos, dim,
                        nodes.ParFESpace()->GetOrdering(),
                        Ordering::byNODES);
   }

   /// Compute global average and maximum errors on marked fitting nodes.
   void GetErrors(real_t &err_avg, real_t &err_max) const
   {
      if (source == SurfaceFittingOptions::ANALYTIC)
      {
         UpdateAnalyticValues();
      }

      err_avg = 0.0;
      err_max = 0.0;
      int count = 0;
      for (int i = 0; i < marker.Size(); i++)
      {
         if (!marker[i] || current_fes.GetLocalTDofNumber(i) < 0) { continue; }
         const real_t err = std::abs(sigma_samples(i));
         err_avg += err;
         err_max = std::max(err_max, err);
         count++;
      }

      MPI_Allreduce(MPI_IN_PLACE, &err_avg, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &err_max, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX, mesh.GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM,
                    mesh.GetComm());
      if (count > 0) { err_avg /= count; }
   }

   void ScaleCoefficient(real_t factor)
   {
      MFEM_VERIFY(factor >= 1.0,
                  "Surface fitting coefficient scale must be at least one.");
      coefficient *= factor;
   }

   real_t GetCoefficient() const { return coefficient; }

   /// Copy level-set values at the current mesh nodes for visualization.
   void GetCurrentLevelSet(ParGridFunction &level_set) const
   {
      if (source == SurfaceFittingOptions::ANALYTIC)
      {
         UpdateAnalyticValues();
      }
      MFEM_VERIFY(level_set.Size() == sigma_samples.Size(),
                  "Visualization level-set space is incompatible with the "
                  "current fitting state.");
      level_set = sigma_samples;
   }

private:
   /// Return the polynomial order of an H1 mesh space.
   static int GetH1Order(const ParFiniteElementSpace &fes)
   {
      const auto *fec = dynamic_cast<const H1_FECollection *>(fes.FEColl());
      MFEM_VERIFY(fec, "Surface fitting requires an H1 mesh space.");
      return fec->GetOrder();
   }

   /// Return the one-dimensional basis type of an H1 mesh space.
   static int GetH1BasisType(const ParFiniteElementSpace &fes)
   {
      const auto *fec = dynamic_cast<const H1_FECollection *>(fes.FEColl());
      MFEM_VERIFY(fec, "Surface fitting requires an H1 mesh space.");
      return fec->GetBasisType();
   }

   /// Freeze source values and derivatives and initialize their GSLIB finder.
   void SetupDiscreteLevelSet(const ParGridFunction &level_set)
   {
#ifdef MFEM_USE_GSLIB
      const ParFiniteElementSpace *level_fes = level_set.ParFESpace();
      MFEM_VERIFY(level_fes,
                  "The discrete level set must use a parallel FE space.");
      const auto *level_fec =
         dynamic_cast<const H1_FECollection *>(level_fes->FEColl());
      MFEM_VERIFY(level_fec,
                  "The discrete level set must use an H1 space.");

      source_mesh = std::make_unique<ParMesh>(*level_fes->GetParMesh(), true);
      source_fec = std::make_unique<H1_FECollection>(
                      level_fec->GetOrder(), dim, level_fec->GetBasisType());
      source_fes = std::make_unique<ParFiniteElementSpace>(
                      source_mesh.get(), source_fec.get());
      MFEM_VERIFY(level_set.Size() == source_fes->GetVSize(),
                  "The copied source space is incompatible with the "
                  "discrete level set.");
      source_sigma = std::make_unique<ParGridFunction>(source_fes.get());
      *source_sigma = level_set;

      if (discrete_from_background ||
          discrete_derivative_mode ==
          SurfaceFittingOptions::INTERPOLATED_SOURCE)
      {
         source_grad_fes = std::make_unique<ParFiniteElementSpace>(
                              source_mesh.get(), source_fec.get(), dim,
                              Ordering::byNODES);
         source_hess_fes = std::make_unique<ParFiniteElementSpace>(
                              source_mesh.get(), source_fec.get(), dim * dim,
                              Ordering::byNODES);
         source_grad =
            std::make_unique<ParGridFunction>(source_grad_fes.get());
         source_hess =
            std::make_unique<ParGridFunction>(source_hess_fes.get());
         ComputeDerivatives(*source_sigma, *source_grad, *source_hess,
                            *source_fes);
      }

      finder = std::make_unique<FindPointsGSLIB>(source_mesh->GetComm());
      finder->Setup(*source_mesh, 0.1, 1.0e-12, 256);
#else
      MFEM_CONTRACT_VAR(level_set);
      MFEM_ABORT("Discrete surface fitting requires GSLIB.");
#endif
   }

   /// Project a scalar field's gradient and Hessian into nodal H1 fields.
   static void ComputeDerivatives(const ParGridFunction &sigma,
                                  ParGridFunction &gradient,
                                  ParGridFunction &hessian,
                                  ParFiniteElementSpace &scalar_fes)
   {
      const int scalar_size = sigma.Size();
      for (int d = 0; d < dim; d++)
      {
         ParGridFunction grad_comp(
            &scalar_fes, gradient.GetData() + d * scalar_size);
         sigma.GetDerivative(1, d, grad_comp);
      }

      int id = 0;
      for (int d = 0; d < dim; d++)
      {
         ParGridFunction grad_comp(
            &scalar_fes, gradient.GetData() + d * scalar_size);
         for (int idir = 0; idir < dim; idir++)
         {
            ParGridFunction hess_comp(
               &scalar_fes, hessian.GetData() + id * scalar_size);
            grad_comp.GetDerivative(1, idir, hess_comp);
            id++;
         }
      }
   }

   /// Compute derivatives using only the level-set data in one element.
   void ComputeElementDerivatives(int element,
                                  DenseMatrix &gradient,
                                  DenseMatrix &hessian) const
   {
      const FiniteElement &fe = *current_fes.GetFE(element);
      ElementTransformation &trans =
         *current_fes.GetElementTransformation(element);
      const int dof = fe.GetDof();

      Array<int> dofs;
      Vector sigma_e;
      current_fes.GetElementDofs(element, dofs);
      current_sigma.GetSubVector(dofs, sigma_e);

      DenseMatrix grad_phys;
      fe.ProjectGrad(fe, trans, grad_phys);
      gradient.SetSize(dof, dim);
      Vector gradient_data(gradient.GetData(), dof * dim);
      grad_phys.Mult(sigma_e, gradient_data);

      // This reshape reproduces TMOP's element-local second application of
      // ProjectGrad; the final columns store the dim-by-dim Hessian.
      hessian.SetSize(dof * dim, dim);
      Mult(grad_phys, gradient, hessian);
      hessian.SetSize(dof, dim * dim);
   }

   /// Sample discrete fitting data at the current physical node positions.
   void UpdateDiscreteSamples() const
   {
#ifdef MFEM_USE_GSLIB
      const int ndofs = current_fes.GetVSize();
      const int marked_count = marked_dof_indices.Size();

      if (!discrete_from_background &&
          discrete_derivative_mode == SurfaceFittingOptions::ELEMENT_LOCAL)
      {
         // ProjectGrad uses the complete element stencil, so element-local
         // derivatives require sigma at every nodal DOF, including unmarked
         // neighbors of fitting DOFs.
         finder->FindPoints(current_node_pos, Ordering::byNODES);
         finder->Interpolate(*source_sigma, sigma_samples,
                             Ordering::byNODES);
         current_sigma = sigma_samples;
         return;
      }

      // If no marked DOFs, nothing to interpolate
      if (marked_count == 0)
      {
         sigma_samples.SetSize(ndofs);
         sigma_samples = 0.0;
         grad_samples.SetSize(ndofs * dim);
         grad_samples = 0.0;
         hess_samples.SetSize(ndofs * dim * dim);
         hess_samples = 0.0;
         return;
      }

      // Pack marked DOF positions into contiguous array (byNODES ordering)
      Vector marked_positions(marked_count * dim);
      for (int d = 0; d < dim; d++)
      {
         for (int i = 0; i < marked_count; i++)
         {
            marked_positions(i + d * marked_count) =
               current_node_pos(marked_dof_indices[i] + d * ndofs);
         }
      }

      // Interpolate at marked positions only
      Vector marked_sigma, marked_grad, marked_hess;
      finder->FindPoints(marked_positions, Ordering::byNODES);
      finder->Interpolate(*source_sigma, marked_sigma, Ordering::byNODES);

      // Unpack sigma values
      sigma_samples.SetSize(ndofs);
      sigma_samples = 0.0;  // Initialize unmarked DOFs to zero
      for (int i = 0; i < marked_count; i++)
      {
         sigma_samples(marked_dof_indices[i]) = marked_sigma(i);
      }

      finder->Interpolate(*source_grad, marked_grad, Ordering::byNODES);
      finder->Interpolate(*source_hess, marked_hess, Ordering::byNODES);

      // Unpack gradient values
      grad_samples.SetSize(ndofs * dim);
      grad_samples = 0.0;
      for (int d = 0; d < dim; d++)
      {
         for (int i = 0; i < marked_count; i++)
         {
            grad_samples(marked_dof_indices[i] + d * ndofs) =
               marked_grad(i + d * marked_count);
         }
      }

      // Unpack Hessian values
      const int hess_dim = dim * dim;
      hess_samples.SetSize(ndofs * hess_dim);
      hess_samples = 0.0;
      for (int h = 0; h < hess_dim; h++)
      {
         for (int i = 0; i < marked_count; i++)
         {
            hess_samples(marked_dof_indices[i] + h * ndofs) =
               marked_hess(i + h * marked_count);
         }
      }
#else
      MFEM_ABORT("Discrete surface fitting requires GSLIB.");
#endif
   }

   /// Evaluate analytic level-set values for fitting-error reporting.
   void UpdateAnalyticValues() const
   {
      const int ndofs = current_fes.GetVSize();
      sigma_samples.SetSize(ndofs);

      for (int i = 0; i < ndofs; i++)
      {
         const real_t xc = current_node_pos(i) - 0.5;
         const real_t yc = current_node_pos(i + ndofs) - 0.5;
         if (analytic_level_set == SurfaceFittingOptions::SQUIRCLE)
         {
            const real_t radius = 0.24;
            sigma_samples(i) = std::pow(xc, 4.0) + std::pow(yc, 4.0);
            if constexpr (dim == 3)
            {
               const real_t zc = current_node_pos(i + 2 * ndofs) - 0.5;
               sigma_samples(i) += std::pow(zc, 4.0);
            }
            sigma_samples(i) -= std::pow(radius, 4.0);
         }
         else if (analytic_level_set == SurfaceFittingOptions::SPHERE)
         {
            if constexpr (dim == 3)
            {
               const real_t zc = current_node_pos(i + 2 * ndofs) - 0.5;
               const real_t r = std::sqrt(xc * xc + yc * yc + zc * zc);
               sigma_samples(i) = r - 0.25;
            }
         }
         else
         {
            const real_t r = std::sqrt(xc * xc + yc * yc);
            sigma_samples(i) = r - 0.25;
         }
      }
   }

   /// Pack one scalar DOF's sampled value and derivatives into quadrature data.
   void FillNodeData(int dof, real_t *data) const
   {
      for (int j = 0; j < SurfaceFitDataLayout<dim>::SIZE; j++)
      {
         data[j] = 0.0;
      }
      data[SurfaceFitDataLayout<dim>::COEFFICIENT] =
         marker[dof] ? coefficient / dof_count[dof] : 0.0;
      if (source == SurfaceFittingOptions::ANALYTIC) { return; }

      const int ndofs = current_fes.GetVSize();
      data[SurfaceFitDataLayout<dim>::VALUE] = sigma_samples(dof);
      if (discrete_from_background ||
          discrete_derivative_mode ==
          SurfaceFittingOptions::INTERPOLATED_SOURCE)
      {
         for (int d = 0; d < dim; d++)
         {
            data[SurfaceFitDataLayout<dim>::GRADIENT + d] =
               grad_samples(dof + d * ndofs);
         }
         for (int j = 0; j < dim * dim; j++)
         {
            data[SurfaceFitDataLayout<dim>::HESSIAN + j] =
               hess_samples(dof + j * ndofs);
         }
      }
   }

   /// Pack element-local gradient and Hessian data for one nodal point.
   static void FillElementDerivativeData(int local_dof,
                                         const DenseMatrix &gradient,
                                         const DenseMatrix &hessian,
                                         real_t *data)
   {
      for (int d = 0; d < dim; d++)
      {
         data[SurfaceFitDataLayout<dim>::GRADIENT + d] =
            gradient(local_dof, d);
      }
      for (int j = 0; j < dim * dim; j++)
      {
         data[SurfaceFitDataLayout<dim>::HESSIAN + j] =
            hessian(local_dof, j);
      }
   }

   ParMesh &mesh;
   int order;
   int basis;
   H1_FECollection current_fec;
   mutable ParFiniteElementSpace current_fes;
   mutable ParGridFunction current_sigma;
   Array<bool> marker;
   Array<int> dof_count;
   Array<int> marked_dof_indices;  // Cached indices of marked DOFs
   real_t coefficient;
   SurfaceFittingOptions::LevelSetSource source;
   SurfaceFittingOptions::AnalyticLevelSet analytic_level_set;
   SurfaceFittingOptions::DiscreteDerivativeMode discrete_derivative_mode;
   bool discrete_from_background;
   mutable Vector current_node_pos;
   mutable Vector sigma_samples;
   mutable Vector grad_samples;
   mutable Vector hess_samples;

   std::unique_ptr<ParMesh> source_mesh;
   std::unique_ptr<H1_FECollection> source_fec;
   std::unique_ptr<ParFiniteElementSpace> source_fes;
   std::unique_ptr<ParFiniteElementSpace> source_grad_fes;
   std::unique_ptr<ParFiniteElementSpace> source_hess_fes;
   std::unique_ptr<ParGridFunction> source_sigma;
   std::unique_ptr<ParGridFunction> source_grad;
   std::unique_ptr<ParGridFunction> source_hess;
#ifdef MFEM_USE_GSLIB
   std::unique_ptr<FindPointsGSLIB> finder;
#endif
};

/// Return the element's nodal integration rule in lexicographic order.
IntegrationRule MakeTensorNodalIntegrationRule(const FiniteElement &fe)
{
   const IntegrationRule &nodes = fe.GetNodes();
   IntegrationRule lex_nodes(nodes.GetNPoints());
   lex_nodes.SetOrder(nodes.GetOrder());
   const auto *nfe = dynamic_cast<const NodalFiniteElement *>(&fe);
   const Array<int> *lex =
      nfe && nfe->GetLexicographicOrdering().Size() > 0 ?
      &nfe->GetLexicographicOrdering() : nullptr;
   for (int i = 0; i < nodes.GetNPoints(); i++)
   {
      lex_nodes.IntPoint(i) = nodes.IntPoint(lex ? (*lex)[i] : i);
   }
   return lex_nodes;
}

template <int dim>
class SurfaceFittingStateManager
{
public:
   /// Allocate analytic or discrete state needed by the fitting operator.
   SurfaceFittingStateManager(ParMesh &mesh_,
                              ParFiniteElementSpace &fes_,
                              const IntegrationRule &surface_node_ir_,
                              const SurfaceFittingOptions &options)
      : is_analytic(options.source == SurfaceFittingOptions::ANALYTIC),
        surface_qspace(mesh_, surface_node_ir_),
        surface_coeff_qspace(surface_qspace, 1),
        surface_coeff_qdata(surface_coeff_qspace),
        surface_data_qspace(is_analytic ? nullptr :
                           new VectorQuadratureSpace(
                              surface_qspace,
                              SurfaceFitDataLayout<dim>::SIZE)),
        surface_qdata(is_analytic ? nullptr :
                     new QuadratureFunction(*surface_data_qspace)),
        current_nodes(new ParGridFunction(&fes_)),
        surface_data(new SurfaceFittingData<dim>(mesh_, fes_, options))
   {
      // Fill coefficient data once (doesn't change during optimization)
      FillCoefficientData(fes_, options);
   }

   /// Release dynamically allocated discrete fitting state.
   ~SurfaceFittingStateManager()
   {
      delete surface_data;
      delete current_nodes;
      delete surface_qdata;
      delete surface_data_qspace;
   }

   /// Refresh fitting fields after the nonlinear solver changes mesh nodes.
   void UpdateAfterMeshPositionChange(const Vector &x)
   {
      current_nodes->SetFromTrueDofs(x);
      if (is_analytic)
      {
         surface_data->UpdateCurrentNodes(*current_nodes);
      }
      else
      {
         surface_data->FillQuadratureData(*current_nodes, surface_qspace,
                                          *surface_qdata);
      }
   }

   /// Return the active analytic coefficient or sampled discrete data.
   const QuadratureFunction& GetSurfaceQuadratureData() const
   {
      return is_analytic ? surface_coeff_qdata : *surface_qdata;
   }

   /// Return the nodal quadrature space used by surface fitting.
   QuadratureSpace& GetSurfaceQuadratureSpace()
   {
      return surface_qspace;
   }

   /// Return the vector quadrature space describing the active fitting data.
   VectorQuadratureSpace& GetSurfaceDataQuadratureSpace()
   {
      return is_analytic ? surface_coeff_qspace : *surface_data_qspace;
   }

   /// Forward fitting-error evaluation to the surface-data object.
   void GetSurfaceErrors(real_t &err_avg, real_t &err_max) const
   {
      surface_data->GetErrors(err_avg, err_max);
   }

   void ScaleCoefficient(real_t factor)
   {
      surface_data->ScaleCoefficient(factor);
      if (is_analytic)
      {
         surface_coeff_qdata *= factor;
      }
      else
      {
         surface_data->FillQuadratureData(*current_nodes, surface_qspace,
                                          *surface_qdata);
      }
   }

   real_t GetCoefficient() const
   {
      return surface_data->GetCoefficient();
   }

   /// Copy current level-set samples into a nodal grid function.
   void GetCurrentLevelSet(ParGridFunction &level_set) const
   {
      surface_data->GetCurrentLevelSet(level_set);
   }

   /// Report whether fitting uses a directly evaluated analytic level set.
   bool IsAnalytic() const { return is_analytic; }

private:
   /// Pack fixed marker-weight coefficients at nodal quadrature points.
   void FillCoefficientData(ParFiniteElementSpace &fes,
                           const SurfaceFittingOptions &options)
   {
      // Build coefficient data: marker[i] ? coefficient / dof_count[i] : 0
      H1_FECollection current_fec(
         dynamic_cast<const H1_FECollection*>(fes.FEColl())->GetOrder(),
         dim,
         dynamic_cast<const H1_FECollection*>(fes.FEColl())->GetBasisType());
      ParFiniteElementSpace current_fes(fes.GetParMesh(), &current_fec);

      ParGridFunction counter(&current_fes);
      Array<int> dof_count;
      counter.CountElementsPerVDof(dof_count);

      real_t *coeff_ptr = surface_coeff_qdata.HostWrite();
      Array<int> dofs;

      for (int e = 0; e < surface_qspace.GetNE(); e++)
      {
         const IntegrationRule &ir = surface_qspace.GetIntRule(e);
         current_fes.GetElementDofs(e, dofs);
         const int offset = surface_qspace.Offset(e);

         const FiniteElement *fe = current_fes.GetFE(e);
         const auto *nfe = dynamic_cast<const NodalFiniteElement *>(fe);
         const Array<int> *lex_to_native =
            nfe && nfe->GetLexicographicOrdering().Size() > 0 ?
            &nfe->GetLexicographicOrdering() : nullptr;

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const int dof = dofs[lex_to_native ? (*lex_to_native)[q] : q];
            coeff_ptr[offset + q] = (*options.marker)[dof] ?
                                   options.coefficient / dof_count[dof] : 0.0;
         }
      }
   }

   bool is_analytic;
   QuadratureSpace surface_qspace;
   VectorQuadratureSpace surface_coeff_qspace;
   QuadratureFunction surface_coeff_qdata;
   VectorQuadratureSpace *surface_data_qspace;
   QuadratureFunction *surface_qdata;
   ParGridFunction *current_nodes;
   SurfaceFittingData<dim> *surface_data;
};

class SingleOutputDerivativeOperator : public Operator
{
public:
   /// Wrap a one-output dFEM derivative as a conventional MFEM operator.
   SingleOutputDerivativeOperator(std::shared_ptr<DerivativeOperator> op,
                                  const ParFiniteElementSpace &fes)
      : Operator(fes.GetTrueVSize()), derivative(std::move(op)) { }

   /// Apply the wrapped derivative action.
   void Mult(const Vector &x, Vector &y) const override
   {
      MultiVector output{y};
      derivative->Mult(x, output);
   }

   /// Assemble the wrapped derivative's diagonal.
   void AssembleDiagonal(Vector &diag) const override
   {
      derivative->AssembleDiagonal(diag);
   }

private:
   std::shared_ptr<DerivativeOperator> derivative;
};

class SumWithDiagonalOperator : public Operator
{
public:
   /// Construct an operator representing the sum of two compatible operators.
   SumWithDiagonalOperator(std::unique_ptr<Operator> a_,
                           std::unique_ptr<Operator> b_)
      : Operator(a_->Height(), a_->Width()),
        a(std::move(a_)), b(std::move(b_))
   {
      MFEM_VERIFY(a->Height() == b->Height() && a->Width() == b->Width(),
                  "Cannot sum incompatible operators.");
   }

   /// Apply both operators and sum their results.
   void Mult(const Vector &x, Vector &y) const override
   {
      work.SetSize(Height());
      a->Mult(x, work);
      b->Mult(x, y);
      y += work;
   }

   /// Assemble and sum both operator diagonals.
   void AssembleDiagonal(Vector &diag) const override
   {
      a->AssembleDiagonal(diag);
      work.SetSize(diag.Size());
      b->AssembleDiagonal(work);
      diag += work;
   }

private:
   std::unique_ptr<Operator> a;
   std::unique_ptr<Operator> b;
   mutable Vector work;
};

template <int dim>
class EnzymeFittingFunctional
{
public:
   /// Build differentiable TMOP and surface-fitting operators.
   EnzymeFittingFunctional(ParFiniteElementSpace &fes_,
                           ParMesh &mesh_,
                           const IntegrationRule &ir,
                           int metric_id,
                           const SurfaceFittingOptions &surface_options)
      : comm(fes_.GetComm()),
        mesh(mesh_),
        fes(fes_),
        metric_qspace(mesh_, ir),
        surface_node_ir(MakeTensorNodalIntegrationRule(*fes_.GetTypicalFE())),
        target_qspace(metric_qspace, dim * dim),
        target_w_inv(target_qspace),
        metric_scalar_qspace(metric_qspace, 1),
        target_det_w(metric_scalar_qspace),
        metric_values(metric_scalar_qspace),
        surface_state_mgr(mesh_, fes_, surface_node_ir, surface_options),
        surface_scalar_qspace(surface_state_mgr.GetSurfaceQuadratureSpace(), 1),
        surface_values(surface_scalar_qspace)
   {
      Array<int> all_domain_attr;
      if (mesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(mesh.attributes.Max());
         all_domain_attr = 1;
      }
      SetTargetData();
      SetupMetricOperatorDispatch(ir, all_domain_attr, metric_id);
      SetupSurfaceOperator(all_domain_attr, surface_options);
   }

   /// Evaluate and globally sum the TMOP metric energy.
   real_t MetricEnergy(const Vector &x) const
   {
      metric_values = 0.0;
      MultiVector input{x, target_w_inv, target_det_w};
      MultiVector output{metric_values};
      metric_operator->Mult(input, output);
      return GlobalSum(metric_values.Sum());
   }

   /// Evaluate and globally sum the surface-fitting penalty.
   real_t SurfaceEnergy(const Vector &x) const
   {
      surface_values = 0.0;
      MultiVector input{x, surface_state_mgr.GetSurfaceQuadratureData()};
      MultiVector output{surface_values};
      surface_operator->Mult(input, output);
      return GlobalSum(surface_values.Sum());
   }

   /// Evaluate the total metric-plus-fitting energy.
   real_t Energy(const Vector &x) const
   {
      return MetricEnergy(x) + SurfaceEnergy(x);
   }

   /// Apply Enzyme-generated first derivatives of both energy terms.
   void Gradient(const Vector &x, Vector &gradient) const
   {
      gradient = 0.0;
      {
         MultiVector metric_input{x, target_w_inv, target_det_w};
         MultiVector metric_output{gradient};
         metric_operator->GetDerivative(X)->Mult(metric_input, metric_output);
      }

      surface_gradient.SetSize(gradient.Size());
      surface_gradient = 0.0;
      {
         MultiVector surface_input{x, surface_state_mgr.GetSurfaceQuadratureData()};
         MultiVector surface_output{surface_gradient};
         surface_operator->GetDerivative(X)->Mult(surface_input, surface_output);
      }
      gradient += surface_gradient;
   }

   /// Construct the summed Enzyme-generated Hessian at the current state.
   std::unique_ptr<Operator> HessianOperator(const Vector &x) const
   {
      // Cache quadrature-point Hessian data once per Newton state and reuse it
      // for the repeated operator applications in the Krylov solve.
      MultiVector metric_input{x, target_w_inv, target_det_w};
      auto metric_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                               metric_operator->GetSecondDerivative(
                                  X, metric_input, true), fes);

      MultiVector surface_input{x, surface_state_mgr.GetSurfaceQuadratureData()};
      auto surface_hessian =
         std::make_unique<SingleOutputDerivativeOperator>(
            surface_operator->GetSecondDerivative(X, surface_input, true), fes);
      return std::make_unique<SumWithDiagonalOperator>(
                std::move(metric_hessian), std::move(surface_hessian));
   }

   /// Return global fitting errors for the current surface state.
   void GetSurfaceErrors(real_t &err_avg, real_t &err_max) const
   {
      surface_state_mgr.GetSurfaceErrors(err_avg, err_max);
   }

   void ScaleSurfaceFittingCoefficient(real_t factor)
   {
      surface_state_mgr.ScaleCoefficient(factor);
   }

   real_t GetSurfaceFittingCoefficient() const
   {
      return surface_state_mgr.GetCoefficient();
   }

   /// Expose state updates to the nonlinear solver callback.
   SurfaceFittingStateManager<dim>& GetSurfaceStateManager()
   {
      return surface_state_mgr;
   }

private:
   /// Sum a scalar value over the fitting communicator.
   real_t GlobalSum(real_t local_value) const
   {
      real_t global_value = 0.0;
      MPI_Allreduce(&local_value, &global_value, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      return global_value;
   }

   /// Dispatch construction of the selected compile-time TMOP metric.
   void SetupMetricOperatorDispatch(const IntegrationRule &ir,
                                    const Array<int> &all_domain_attr,
                                    int metric_id)
   {
      switch (metric_id)
      {
         case 2:
            if constexpr (dim == 2)
            {
               return SetupMetricOperator<2>(ir, all_domain_attr);
            }
            break;
         case 58:
            if constexpr (dim == 2)
            {
               return SetupMetricOperator<58>(ir, all_domain_attr);
            }
            break;
         case 80:
            if constexpr (dim == 2)
            {
               return SetupMetricOperator<80>(ir, all_domain_attr);
            }
            break;
         case 303:
            if constexpr (dim == 3)
            {
               return SetupMetricOperator<303>(ir, all_domain_attr);
            }
            break;
      }
      MFEM_ABORT("Metric id " << metric_id << " is incompatible with "
                 << dim << "D meshes.");
   }

   /// Construct the differentiable operator for one TMOP metric.
   template <int metric_id>
   void SetupMetricOperator(const IntegrationRule &ir,
                            const Array<int> &all_domain_attr)
   {
      const std::vector<FieldDescriptor> input
      {
         FieldDescriptor{X, &fes},
         FieldDescriptor{TARGET_W_INV, &target_qspace},
         FieldDescriptor{TARGET_DET_W, &metric_scalar_qspace}
      };
      const std::vector<FieldDescriptor> output
      {
         FieldDescriptor{Q, &metric_scalar_qspace}
      };
      metric_operator =
         std::make_unique<DifferentiableOperator>(input, output, mesh);
      auto derivatives = std::integer_sequence<size_t, X> {};

      TMOPEnergy<real_t, dim, metric_id> energy;
      metric_operator->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{future::Gradient<X>{}, Identity<TARGET_W_INV>{},
                       Identity<TARGET_DET_W>{}, Weight{}},
         future::tuple{FunctionalValue<Q>{}},
         ir, all_domain_attr, derivatives);
   }

   /// Dispatch analytic or discrete surface-operator construction.
   void SetupSurfaceOperator(const Array<int> &all_domain_attr,
                             const SurfaceFittingOptions &options)
   {
      if (options.source == SurfaceFittingOptions::DISCRETE)
      {
         SetupDiscreteSurfaceOperator(all_domain_attr);
      }
      else
      {
         if constexpr (dim == 2)
         {
            if (options.analytic_level_set == SurfaceFittingOptions::CIRCLE)
            {
               SetupAnalyticSurfaceOperator<SurfaceFittingOptions::CIRCLE>(
                  all_domain_attr);
            }
            else
            {
               MFEM_VERIFY(options.analytic_level_set ==
                           SurfaceFittingOptions::SQUIRCLE,
                           "Unsupported analytic 2D level set.");
               SetupAnalyticSurfaceOperator<SurfaceFittingOptions::SQUIRCLE>(
                  all_domain_attr);
            }
         }
         else
         {
            MFEM_VERIFY(options.analytic_level_set ==
                        SurfaceFittingOptions::SPHERE,
                        "Unsupported analytic 3D level set.");
            SetupAnalyticSurfaceOperator<SurfaceFittingOptions::SPHERE>(
               all_domain_attr);
         }
      }
   }

   /// Construct a directly differentiated analytic fitting operator.
   template <int level_set>
   void SetupAnalyticSurfaceOperator(const Array<int> &all_domain_attr)
   {
      const std::vector<FieldDescriptor> input
      {
         FieldDescriptor{X, &fes},
         FieldDescriptor{Q, &surface_state_mgr.GetSurfaceDataQuadratureSpace()}
      };
      const std::vector<FieldDescriptor> output
      {
         FieldDescriptor{Q, &surface_scalar_qspace}
      };
      surface_operator =
         std::make_unique<DifferentiableOperator>(input, output, mesh);
      auto derivatives = std::integer_sequence<size_t, X> {};
      AnalyticSurfaceFittingEnergy<real_t, level_set, dim> energy;
      surface_operator->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{Value<X>{}, Identity<Q>{}},
         future::tuple{FunctionalValue<Q>{}},
         surface_node_ir, all_domain_attr, derivatives);
   }

   /// Construct a fitting operator driven by sampled discrete derivative data.
   void SetupDiscreteSurfaceOperator(const Array<int> &all_domain_attr)
   {
      const std::vector<FieldDescriptor> input
      {
         FieldDescriptor{X, &fes},
         FieldDescriptor{SURFACE_FIT_DATA,
                        &surface_state_mgr.GetSurfaceDataQuadratureSpace()}
      };
      const std::vector<FieldDescriptor> output
      {
         FieldDescriptor{Q, &surface_scalar_qspace}
      };
      surface_operator =
         std::make_unique<DifferentiableOperator>(input, output, mesh);
      auto derivatives = std::integer_sequence<size_t, X> {};
      DiscreteSurfaceFittingEnergy<real_t, dim> energy;
      surface_operator->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{Value<X>{}, Identity<SURFACE_FIT_DATA>{}},
         future::tuple{FunctionalValue<Q>{}},
         surface_node_ir, all_domain_attr, derivatives);
   }

   /// Fill ideal target Jacobians at all metric quadrature points.
   void SetTargetData()
   {
      constexpr int vdim = dim * dim;
      real_t *inverse_data = target_w_inv.HostWrite();
      real_t *determinant_data = target_det_w.HostWrite();
      for (int e = 0; e < metric_qspace.GetNE(); e++)
      {
         const DenseMatrix &W =
            Geometries.GetGeomToPerfGeomJac(metric_qspace.GetGeometry(e));
         MFEM_VERIFY(W.Height() == dim && W.Width() == dim,
                     "Unexpected target matrix dimension.");
         DenseMatrix W_inv(dim);
         CalcInverse(W, W_inv);
         const real_t det_W = W.Det();
         const int offset = metric_qspace.Offset(e);
         const int nq = metric_qspace.GetIntRule(e).GetNPoints();
         for (int q = 0; q < nq; q++)
         {
            real_t *Wq_inv = inverse_data + vdim * (offset + q);
            for (int i = 0; i < dim; i++)
            {
               for (int j = 0; j < dim; j++)
               {
                  Wq_inv[dim * i + j] = W_inv(i, j);
               }
            }
            determinant_data[offset + q] = det_W;
         }
      }
   }

   MPI_Comm comm;
   ParMesh &mesh;
   ParFiniteElementSpace &fes;
   QuadratureSpace metric_qspace;
   IntegrationRule surface_node_ir;
   VectorQuadratureSpace target_qspace;
   QuadratureFunction target_w_inv;
   VectorQuadratureSpace metric_scalar_qspace;
   QuadratureFunction target_det_w;
   mutable QuadratureFunction metric_values;
   SurfaceFittingStateManager<dim> surface_state_mgr;
   VectorQuadratureSpace surface_scalar_qspace;
   mutable QuadratureFunction surface_values;
   std::unique_ptr<DifferentiableOperator> metric_operator;
   std::unique_ptr<DifferentiableOperator> surface_operator;
   mutable Vector surface_gradient;
};

template <int dim>
class EnzymeFittingNonlinearForm : public ParNonlinearForm
{
public:
   /// Adapt the absolute-coordinate functional to MFEM's displacement form.
   EnzymeFittingNonlinearForm(ParFiniteElementSpace &fes,
                              EnzymeFittingFunctional<dim> &functional_)
      : ParNonlinearForm(&fes),
        functional(functional_),
        absolute_state(fes.GetTrueVSize())
   {
      reference_state.SetSize(fes.GetTrueVSize());
   }

   /// Store the reference nodes used to convert displacements to positions.
   void SetReference(const Vector &x0) { reference_state = x0; }

   /// Convert a displacement vector to absolute nodal coordinates.
   const Vector& ComputeAbsoluteState(const Vector &dx) const
   {
      add(reference_state, dx, absolute_state);
      return absolute_state;
   }

   /// Refresh discrete fitting data for an absolute nodal state.
   void UpdateSurfaceFittingState(const Vector &x_abs)
   {
      functional.GetSurfaceStateManager().UpdateAfterMeshPositionChange(x_abs);
   }

   void GetSurfaceFittingErrors(real_t &err_avg, real_t &err_max) const
   {
      functional.GetSurfaceErrors(err_avg, err_max);
   }

   void ScaleSurfaceFittingCoefficient(real_t factor)
   {
      functional.ScaleSurfaceFittingCoefficient(factor);
   }

   real_t GetSurfaceFittingCoefficient() const
   {
      return functional.GetSurfaceFittingCoefficient();
   }

   /// Evaluate total energy for a displacement from the reference mesh.
   real_t GetEnergy(const Vector &dx) const override
   {
      add(reference_state, dx, absolute_state);
      return functional.Energy(absolute_state);
   }

   /// Evaluate and constrain the nonlinear residual.
   void Mult(const Vector &dx, Vector &y) const override
   {
      add(reference_state, dx, absolute_state);
      functional.Gradient(absolute_state, y);
      const Array<int> &ess_tdofs = GetEssentialTrueDofs();
      if (ess_tdofs.Size() > 0) { y.SetSubVector(ess_tdofs, 0.0); }
   }

   /// Build and constrain the Hessian used as the residual Jacobian.
   Operator &GetGradient(const Vector &dx) const override
   {
      add(reference_state, dx, absolute_state);
      hessian = functional.HessianOperator(absolute_state);
      constrained_hessian = std::make_unique<ConstrainedOperator>(
                               hessian.get(), GetEssentialTrueDofs(), false);
      return *constrained_hessian;
   }

private:
   EnzymeFittingFunctional<dim> &functional;
   Vector reference_state;
   mutable Vector absolute_state;
   mutable std::unique_ptr<Operator> hessian;
   mutable std::unique_ptr<ConstrainedOperator> constrained_hessian;
};

template <int dim>
class EnzymeFittingNewtonSolver : public TMOPNewtonSolver
{
public:
   /// Attach the fitting nonlinear form to the TMOP Newton solver.
   EnzymeFittingNewtonSolver(MPI_Comm comm,
                             const IntegrationRule &ir,
                             EnzymeFittingNonlinearForm<dim> &nlf_)
      : TMOPNewtonSolver(comm, ir, 0),
        enzyme_nlf(nlf_) { }

   void ConfigureAdaptiveSurfaceFitting(real_t scale_factor,
                                        real_t max_error,
                                        real_t weight_limit,
                                        bool converge_by_error)
   {
      MFEM_VERIFY(scale_factor > 1.0,
                  "Adaptive surface fitting scale must be greater than one.");
      MFEM_VERIFY(weight_limit > 0.0,
                  "Surface fitting weight limit must be positive.");
      MFEM_VERIFY(!converge_by_error || max_error >= 0.0,
                  "Error-based surface fitting requires a nonnegative "
                  "error threshold.");
      surf_fit_scale_factor = scale_factor;
      surf_fit_max_err_limit = max_error;
      surf_fit_weight_limit = weight_limit;
      surf_fit_converge_error = converge_by_error;
   }

   void ResetAdaptiveSurfaceFittingState() const
   {
      previous_surf_fit_avg_error = 10000.0;
      update_surface_fit_coefficient = false;
      surf_fit_adapt_count = 0;
   }

   real_t ComputeScalingFactor(const Vector &dx,
                               const Vector &b) const override
   {
      if (surf_fit_scale_factor > 0.0)
      {
         real_t avg_error = 0.0, max_error = 0.0;
         enzyme_nlf.GetSurfaceFittingErrors(avg_error, max_error);
         if ((surf_fit_converge_error &&
              max_error <= surf_fit_max_err_limit) ||
             surf_fit_adapt_count >= surf_fit_adapt_count_limit)
         {
            return 0.0;
         }
      }

      const real_t scale = TMOPNewtonSolver::ComputeScalingFactor(dx, b);
      if (scale > 0.0 && surf_fit_scale_factor > 0.0)
      {
         update_surface_fit_coefficient = true;
      }
      return scale;
   }

   /// Synchronize fitting fields with each Newton or line-search state.
   void ProcessNewState(const Vector &dx) const override
   {
      // Convert displacement to absolute position
      const Vector &x_abs = enzyme_nlf.ComputeAbsoluteState(dx);

      // Update the discrete fitting fields for every Newton or line-search
      // state before its energy, residual, or Hessian is evaluated.
      enzyme_nlf.UpdateSurfaceFittingState(x_abs);

      if (!update_surface_fit_coefficient) { return; }

      real_t avg_error = 0.0, max_error = 0.0;
      enzyme_nlf.GetSurfaceFittingErrors(avg_error, max_error);
      const real_t coefficient = enzyme_nlf.GetSurfaceFittingCoefficient();
      const real_t relative_change =
         (previous_surf_fit_avg_error - avg_error) /
         previous_surf_fit_avg_error;
      const bool should_increase =
         relative_change < surf_fit_err_rel_change_limit &&
         coefficient < surf_fit_weight_limit &&
         (surf_fit_converge_error || max_error > surf_fit_max_err_limit);

      if (should_increase)
      {
         const real_t factor =
            std::min(surf_fit_scale_factor,
                     surf_fit_weight_limit / coefficient);
         enzyme_nlf.ScaleSurfaceFittingCoefficient(factor);
         surf_fit_adapt_count++;
         if (print_options.iterations && Mpi::Root())
         {
            std::cout << "Adaptive surface fitting: fit_avg=" << avg_error
                      << ", fit_max=" << max_error
                      << ", coefficient="
                      << enzyme_nlf.GetSurfaceFittingCoefficient() << '\n';
         }
      }
      else
      {
         surf_fit_adapt_count = 0;
      }
      previous_surf_fit_avg_error = avg_error;
      update_surface_fit_coefficient = false;
   }

private:
   EnzymeFittingNonlinearForm<dim> &enzyme_nlf;
   real_t surf_fit_scale_factor = 0.0;
   real_t surf_fit_max_err_limit = -1.0;
   real_t surf_fit_err_rel_change_limit = 0.001;
   real_t surf_fit_weight_limit = 1.0e20;
   bool surf_fit_converge_error = false;
   mutable real_t previous_surf_fit_avg_error = 10000.0;
   mutable bool update_surface_fit_coefficient = false;
   mutable int surf_fit_adapt_count = 0;
   int surf_fit_adapt_count_limit = 10;
};

/// Compute the communicator-wide minimum element Jacobian determinant.
real_t MinimumDetJ(ParMesh &pmesh,
                   const ParFiniteElementSpace &fes,
                   IntegrationRules &irules,
                   int quad_order)
{
   real_t min_detJ = infinity();
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const IntegrationRule &ir =
         irules.Get(fes.GetFE(e)->GetGeomType(), quad_order);
      ElementTransformation *trans = pmesh.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         trans->SetIntPoint(&ir.IntPoint(q));
         min_detJ = std::min(min_detJ, trans->Jacobian().Det());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_detJ, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh.GetComm());
   return min_detJ;
}

/// Save a parallel mesh as a single serial mesh file.
void SaveMesh(ParMesh &pmesh, const char *filename)
{
   std::ofstream output(filename);
   output.precision(8);
   pmesh.PrintAsOne(output);
}

/// Select the requested global family of integration rules.
IntegrationRules &SelectIntegrationRules(int quad_type)
{
   switch (quad_type)
   {
      case 1: return IntRulesLo;
      case 2: return IntRules;
      case 3: return IntRulesCU;
      default: MFEM_ABORT("Unknown quadrature rule type: " << quad_type);
   }
}

/// Mark scalar DOFs on an interface or selected boundary attribute.
void MarkSurfaceFittingDofs(ParMesh &pmesh,
                            ParGridFunction &level_set,
                            ParGridFunction &material,
                            int marking_type,
                            Array<bool> &marker,
                            ParGridFunction *marker_vis)
{
   ParFiniteElementSpace *surface_fes = level_set.ParFESpace();
   marker.SetSize(level_set.Size());
   marker = false;
   ParGridFunction local_marker(surface_fes);
   local_marker = 0.0;
   Array<int> dofs;

   if (marking_type == 0)
   {
      material.ExchangeFaceNbrData();
      const Vector &face_nbr_data = material.FaceNbrData();

      for (int i = 0; i < pmesh.GetNumFaces(); i++)
      {
         auto trans = pmesh.GetInteriorFaceTransformations(i);
         if (trans && material(trans->Elem1No) != material(trans->Elem2No))
         {
            surface_fes->GetFaceDofs(i, dofs);
            for (int j = 0; j < dofs.Size(); j++)
            {
               local_marker(dofs[j]) = 1.0;
            }
         }
      }
      for (int i = 0; i < pmesh.GetNSharedFaces(); i++)
      {
         auto trans = pmesh.GetSharedFaceTransformations(i);
         if (!trans) { continue; }
         const real_t first = material(trans->Elem1No);
         const real_t second =
            face_nbr_data(trans->Elem2No - pmesh.GetNE());
         if (first != second)
         {
            surface_fes->GetFaceDofs(pmesh.GetSharedFace(i), dofs);
            for (int j = 0; j < dofs.Size(); j++)
            {
               local_marker(dofs[j]) = 1.0;
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < pmesh.GetNBE(); i++)
      {
         if (pmesh.GetBdrElement(i)->GetAttribute() != marking_type)
         {
            continue;
         }
         surface_fes->GetBdrElementVDofs(i, dofs);
         for (int j = 0; j < dofs.Size(); j++)
         {
            local_marker(dofs[j]) = 1.0;
         }
      }
   }

   local_marker.ExchangeFaceNbrData();
   GroupCommunicator &group_comm = surface_fes->GroupComm();
   Array<real_t> marker_array(local_marker.GetData(), local_marker.Size());
   group_comm.Reduce<real_t>(marker_array, GroupCommunicator::Max);
   group_comm.Bcast(marker_array);
   local_marker.ExchangeFaceNbrData();
   for (int i = 0; i < local_marker.Size(); i++)
   {
      marker[i] = local_marker(i) == 1.0;
   }
   if (marker_vis) { *marker_vis = local_marker; }
}

/// Determine coordinate constraints imposed by the boundary-motion policy.
template <int dim>
void GetMeshOptimizerEssentialTrueDofs(const ParFiniteElementSpace &fes,
                                       bool move_bnd,
                                       Array<int> &ess_tdofs)
{
   ess_tdofs.DeleteAll();
   const ParMesh *pmesh = fes.GetParMesh();
   if (pmesh->bdr_attributes.Size() == 0) { return; }
   if (!move_bnd)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
      return;
   }

   int count = 0;
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      const int ndofs = fes.GetBE(i)->GetDof();
      const int attr = pmesh->GetBdrElement(i)->GetAttribute();
      // Attributes 1/2/3 fix x/y/z; attribute 4 fixes every component.
      if (attr >= 1 && attr <= dim) { count += ndofs; }
      else if (attr == 4) { count += dim * ndofs; }
   }

   Array<int> vdofs, ess_vdofs(count);
   count = 0;
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      const int ndofs = fes.GetBE(i)->GetDof();
      const int attr = pmesh->GetBdrElement(i)->GetAttribute();
      fes.GetBdrElementVDofs(i, vdofs);
      if (attr >= 1 && attr <= dim)
      {
         const int component = attr - 1;
         for (int j = 0; j < ndofs; j++)
         {
            ess_vdofs[count++] = vdofs[j + component * ndofs];
         }
      }
      else if (attr == 4)
      {
         for (int j = 0; j < vdofs.Size(); j++)
         {
            ess_vdofs[count++] = vdofs[j];
         }
      }
   }

   Array<int> vdof_marker, tdof_marker;
   FiniteElementSpace::ListToMarker(ess_vdofs, fes.GetVSize(), vdof_marker);
   tdof_marker.SetSize(fes.GetTrueVSize());
   fes.Dof_TrueDof_Matrix()->BooleanMultTranspose(
      1, vdof_marker, 0, tdof_marker);
   FiniteElementSpace::MarkerToList(tdof_marker, ess_tdofs);
}

/// Determine essential true DOFs while leaving the fitted boundary movable.
template <int dim>
void GetFittingEssentialTrueDofs(const ParFiniteElementSpace &pfes,
                                 bool move_bnd,
                                 int marking_type,
                                 Array<int> &ess_tdofs)
{
   if (move_bnd || marking_type == 0)
   {
      GetMeshOptimizerEssentialTrueDofs<dim>(pfes, move_bnd, ess_tdofs);
      return;
   }

   const ParMesh *pmesh = pfes.GetParMesh();
   if (pmesh->bdr_attributes.Size() == 0)
   {
      ess_tdofs.DeleteAll();
      return;
   }

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   MFEM_VERIFY(marking_type <= ess_bdr.Size(),
               "Surface marking boundary attribute " << marking_type
               << " is not present in the mesh.");
   ess_bdr[marking_type - 1] = 0;
   pfes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
}

/// Configure and run Newton optimization, then report final fitting metrics.
template <int dim>
int RunOptimizer(ParMesh &pmesh,
                 ParFiniteElementSpace &fes,
                 ParGridFunction &nodes,
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
                 int metric_id,
                 int verbosity,
                 const SurfaceFittingOptions &surface_options,
                 real_t surface_fit_tolerance,
                 real_t surface_fit_adapt,
                 real_t surface_fit_threshold,
                 real_t surface_fit_weight_limit,
                 bool surface_fit_converge_error,
                 ParGridFunction &final_level_set)
{
   Vector true_nodes(fes.GetTrueVSize());
   nodes.GetTrueDofs(true_nodes);
   const IntegrationRule &ir =
      irules.Get(pmesh.GetTypicalElementGeometry(), quad_order);
   EnzymeFittingFunctional<dim> functional(fes, pmesh, ir, metric_id,
                                           surface_options);
   functional.GetSurfaceStateManager().UpdateAfterMeshPositionChange(
      true_nodes);

   // Compute the residual norm after eliminating essential true DOFs.
   auto constrained_gradient_norm = [&](const Vector &x)
   {
      Vector gradient(x.Size());
      functional.Gradient(x, gradient);
      if (ess_tdofs.Size() > 0)
      {
         gradient.SetSubVector(ess_tdofs, 0.0);
      }
      return GlobalVectorNorm(fes.GetComm(), gradient);
   };

   const real_t initial_metric_energy = functional.MetricEnergy(true_nodes);
   const real_t initial_surface_energy = functional.SurfaceEnergy(true_nodes);
   const real_t initial_energy =
      initial_metric_energy + initial_surface_energy;

   real_t fit_avg = 0.0;
   real_t fit_max = 0.0;
   if (surface_fit_tolerance > 0.0)
   {
      functional.GetSurfaceErrors(fit_avg, fit_max);
      if (fit_max <= surface_fit_tolerance)
      {
         if (Mpi::Root() && verbosity > 0)
         {
            std::cout << "Surface fitting tolerance reached before Newton: "
                      << "fit_avg=" << fit_avg
                      << ", fit_max=" << fit_max << '\n';
         }
         functional.GetSurfaceStateManager().GetCurrentLevelSet(
            final_level_set);
         return 0;
      }
   }

   EnzymeFittingNonlinearForm<dim> nonlinear_form(fes, functional);
   nonlinear_form.SetEssentialTrueDofs(ess_tdofs);
   nonlinear_form.SetReference(true_nodes);

#ifdef MFEM_USE_SINGLE
   const real_t linear_rtol = 1e-5;
#else
   const real_t linear_rtol = 1e-12;
#endif
   IterativeSolver::PrintLevel linear_print;
   if (verbosity > 1) { linear_print.Errors().Warnings().FirstAndLast(); }
   if (verbosity > 2) { linear_print.Errors().Warnings().Iterations(); }
   MINRESSolver linear_solver(fes.GetComm());
   linear_solver.SetMaxIter(max_lin_iter);
   linear_solver.SetRelTol(linear_rtol);
   linear_solver.SetAbsTol(0.0);
   linear_solver.SetPrintLevel(linear_print);
   OperatorJacobiSmoother jacobi;
   if (lin_solver == 3)
   {
      jacobi.SetPositiveDiagonal(true);
      linear_solver.SetPreconditioner(jacobi);
   }

   EnzymeFittingNewtonSolver<dim> solver(fes.GetComm(), ir, nonlinear_form);
   solver.SetIntegrationRules(irules, quad_order);
   solver.SetMinDetPtr(&min_detJ);
   solver.SetOperator(nonlinear_form);
   solver.SetPreconditioner(linear_solver);
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(solver_atol);
   if (solver_art_type > 0 && surface_fit_adapt <= 0.0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   else if (solver_art_type > 0 && Mpi::Root() && verbosity > 0)
   {
      std::cout << "Disabling adaptive linear tolerance while the surface "
                << "fitting coefficient is adaptive.\n";
   }
   if (surface_fit_adapt > 0.0)
   {
      solver.ConfigureAdaptiveSurfaceFitting(
         surface_fit_adapt, surface_fit_threshold,
         surface_fit_weight_limit, surface_fit_converge_error);
   }
   IterativeSolver::PrintLevel newton_print;
   if (verbosity > 0) { newton_print.Errors().Warnings().Iterations(); }
   solver.SetPrintLevel(newton_print);

   Vector zero;
   solver.Mult(zero, true_nodes);

   if (surface_fit_adapt > 0.0 && surface_fit_converge_error &&
       surface_fit_tolerance >= 0.0)
   {
      for (int stage = 0; stage < 10; stage++)
      {
         functional.GetSurfaceStateManager().UpdateAfterMeshPositionChange(
            true_nodes);
         real_t stage_fit_avg = 0.0, stage_fit_max = 0.0;
         functional.GetSurfaceErrors(stage_fit_avg, stage_fit_max);
         if (stage_fit_max <= surface_fit_tolerance ||
             functional.GetSurfaceFittingCoefficient() >=
             surface_fit_weight_limit)
         {
            break;
         }

         const real_t factor = std::min(
            surface_fit_adapt,
            surface_fit_weight_limit /
            functional.GetSurfaceFittingCoefficient());
         functional.ScaleSurfaceFittingCoefficient(factor);
         if (Mpi::Root() && verbosity > 0)
         {
            std::cout << "Restarting surface fit: fit_avg=" << stage_fit_avg
                      << ", fit_max=" << stage_fit_max
                      << ", coefficient="
                      << functional.GetSurfaceFittingCoefficient() << '\n';
         }
         nonlinear_form.SetReference(true_nodes);
         solver.ResetAdaptiveSurfaceFittingState();
         solver.Mult(zero, true_nodes);
      }
   }

   nodes.SetFromTrueDofs(true_nodes);
   pmesh.SetNodalGridFunction(&nodes);
   pmesh.NodesUpdated();
   pmesh.ExchangeFaceNbrData();
   functional.GetSurfaceStateManager().UpdateAfterMeshPositionChange(
      true_nodes);

   const real_t final_metric_energy = functional.MetricEnergy(true_nodes);
   const real_t final_surface_energy = functional.SurfaceEnergy(true_nodes);
   const real_t final_energy = final_metric_energy + final_surface_energy;
   const real_t final_gradient_norm =
      constrained_gradient_norm(true_nodes);
   functional.GetSurfaceErrors(fit_avg, fit_max);
   functional.GetSurfaceStateManager().GetCurrentLevelSet(final_level_set);

   bool converged = surface_fit_converge_error &&
                    surface_fit_tolerance >= 0.0 ?
                    fit_max <= surface_fit_tolerance :
                    solver.GetConverged();
   if (surface_fit_tolerance > 0.0 && fit_max <= surface_fit_tolerance)
   {
      converged = true;
      if (Mpi::Root() && verbosity > 0)
      {
         std::cout << "Surface fitting tolerance reached after Newton: "
                   << "fit_avg=" << fit_avg
                   << ", fit_max=" << fit_max << '\n';
      }
   }

   if (Mpi::Root() && verbosity > 0)
   {
      std::cout << std::scientific << std::setprecision(4)
                << "Initial strain energy: " << initial_energy
                << " = metrics: " << initial_metric_energy
                << " + extra terms: " << initial_surface_energy << '\n'
                << "  Final strain energy: " << final_energy
                << " = metrics: " << final_metric_energy
                << " + extra terms: " << final_surface_energy << '\n'
                << "Final TMOP gradient norm: "
                << final_gradient_norm << '\n';
      if (initial_energy != 0.0)
      {
         std::cout << "The strain energy decreased by: "
                   << (initial_energy - final_energy) * 100.0 / initial_energy
                   << " %.\n";
      }
      std::cout << "Avg fitting error: " << fit_avg << '\n'
                << "Max fitting error: " << fit_max << '\n';
   }
   return converged ? 0 : 2;
}

} // namespace pmesh_fitting_enzyme

using namespace pmesh_fitting_enzyme;

/// Parse miniapp options, build fitting data, and launch optimization.
int main(int argc, char *argv[])
{
#ifdef HYPRE_USING_GPU
   mfem::out << "\nThis miniapp is NOT supported with the GPU version of "
             << "hypre.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "square01.mesh";
   int mesh_poly_deg = 1;
   int rs_levels = 1;
   int rp_levels = 0;
   int metric_id = 2;
   int target_id = 1;
   real_t surface_fit_const = 10.0;
   int quad_type = 1;
   int quad_order = 8;
   int solver_type = 0;
   int solver_iter = 20;
#ifdef MFEM_USE_SINGLE
   real_t solver_rtol = 1e-4;
#else
   real_t solver_rtol = 1e-10;
#endif
   real_t solver_atol = 0.0;
   int lin_solver = 2;
   int max_lin_iter = 100;
   int solver_art_type = 0;
   bool move_bnd = true;
   bool visualization = false;
   int verbosity = 0;
   bool analytic_level_set = true;
   int discrete_derivative_mode =
      SurfaceFittingOptions::INTERPOLATED_SOURCE;
   const char *devopt = "cpu";
   real_t surface_fit_adapt = 2.0;
   real_t surface_fit_threshold = 1.0e-4;
   real_t surface_fit_const_max = 1.0e4;
   bool adapt_marking = false;
   bool surf_bg_mesh = false;
   bool comp_dist = false;
   int surf_ls_type = SurfaceFittingOptions::CIRCLE;
   int marking_type = 0;
   bool mod_bndr_attr = false;
   bool material = false;
   int mesh_node_ordering = Ordering::byNODES;
   int bg_amr_iters = 0;
   bool conv_residual = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric: 2, 58, or 80.");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target type. This Enzyme miniapp currently supports 1: "
                  "ideal shape, unit size.");
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface fitting coefficient.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature type: 1 Gauss-Lobatto, 2 Gauss-Legendre, "
                  "3 closed uniform.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_type, "-st", "--solver-type",
                  "Solver type. Only 0: Newton is currently supported.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&solver_atol, "-atol", "--newton-abs-tolerance",
                  "Absolute tolerance for the Newton solver.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver: 2 MINRES, 3 MINRES + Jacobi.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&solver_art_type, "-art", "--adaptive-rel-tol",
                  "Adaptive linear relative tolerance: 0 none, "
                  "1 Eisenstat-Walker 1, 2 Eisenstat-Walker 2.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary", "Enable constrained boundary motion.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis output.");
   args.AddOption(&verbosity, "-vl", "--verbosity-level",
                  "Verbosity: 0 none, 1 Newton, 2 linear summaries, "
                  "3 linear iterations.");
   args.AddOption(&analytic_level_set,
                  "-als", "--analytic-level-set",
                  "-dls", "--discrete-level-set",
                  "Use an analytic level-set formula or a discrete FE "
                  "level-set field.");
   args.AddOption(&discrete_derivative_mode,
                  "-dder", "--discrete-derivative-mode",
                  "No-background discrete derivative mode: "
                  "1 interpolate initial-mesh gradient and Hessian, "
                  "2 element-local ProjectGrad.");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Factor (> 1) used to increase a stalled surface fitting "
                  "weight; zero disables adaptation.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Maximum fitting error for error-based termination.");
   args.AddOption(&surface_fit_const_max, "-sfcmax", "--surf-fit-const-max",
                  "Maximum adaptive fitting weight.");
   args.AddOption(&adapt_marking, "-marking", "--adaptive-marking",
                  "-no-amarking", "--no-adaptive-marking",
                  "Adaptive marking (not yet supported).");
   args.AddOption(&surf_bg_mesh, "-sbgmesh", "--surf-bg-mesh",
                  "-no-sbgmesh", "--no-surf-bg-mesh",
                  "Use a background mesh for discrete surface fitting.");
   args.AddOption(&comp_dist, "-dist", "--comp-dist", "-no-dist",
                  "--no-comp-dist", "Convert the background level set to "
                  "a distance field.");
   args.AddOption(&surf_ls_type, "-slstype", "--surf-ls-type",
                  "Level set: 1 circle, 2 reactor, 3 squircle.");
   args.AddOption(&marking_type, "-smtype", "--surf-marking-type",
                  "0 interface, otherwise a boundary attribute.");
   args.AddOption(&mod_bndr_attr, "-mod-bndr-attr",
                  "--modify-boundary-attribute", "-fix-bndr-attr",
                  "--fix-boundary-attribute",
                  "Set boundary attributes from Cartesian alignment.");
   args.AddOption(&material, "-mat", "--mat", "-no-mat", "--no-mat",
                  "Use mesh material attributes (not yet supported).");
   args.AddOption(&mesh_node_ordering, "-mno", "--mesh_node_ordering",
                  "Mesh node ordering: 0 byNODES, 1 byVDIM.");
   args.AddOption(&bg_amr_iters, "-bgamriter", "--amr-iter",
                  "Number of background-mesh AMR iterations.");
   args.AddOption(&conv_residual, "-resid", "--resid", "-no-resid",
                  "--no-resid", "Use residual- or fitting-error-based "
                  "termination.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(std::cout); }

   Device device(devopt);
   if (Mpi::Root()) { device.Print(); }

   MFEM_VERIFY(surface_fit_const > 0.0,
               "This miniapp is for surface fitting only. Use "
               "pmesh-optimizer-enzyme for optimization without fitting.");
   MFEM_VERIFY(target_id == 1,
               "pmesh-fitting-enzyme currently supports target id 1 only.");
   MFEM_VERIFY(metric_id == 2 || metric_id == 58 || metric_id == 80 ||
               metric_id == 303,
               "pmesh-fitting-enzyme supports metric ids 2, 58, 80 (2D) and 303 (3D).");
   MFEM_VERIFY(solver_type == 0,
               "pmesh-fitting-enzyme currently supports Newton (-st 0) only.");
   MFEM_VERIFY(lin_solver == 2 || lin_solver == 3,
               "pmesh-fitting-enzyme supports linear solvers 2 and 3.");
   MFEM_VERIFY(solver_art_type >= 0 && solver_art_type <= 2,
               "Unknown adaptive relative tolerance option: "
               << solver_art_type);
   MFEM_VERIFY(surf_ls_type == SurfaceFittingOptions::CIRCLE ||
               surf_ls_type == 2 ||
               surf_ls_type == SurfaceFittingOptions::SQUIRCLE ||
               surf_ls_type == SurfaceFittingOptions::SPHERE,
               "Supported level sets are 1 (circle), 2 (reactor), "
               "3 (squircle), and 4 (sphere).");
   MFEM_VERIFY(!analytic_level_set || surf_ls_type != 2,
               "The reactor level set is available only as a discrete "
               "level set (-dls).");
   MFEM_VERIFY(!surf_bg_mesh || !analytic_level_set,
               "A background mesh is used only with a discrete level set "
               "(-dls).");
   MFEM_VERIFY(discrete_derivative_mode ==
               SurfaceFittingOptions::INTERPOLATED_SOURCE ||
               discrete_derivative_mode ==
               SurfaceFittingOptions::ELEMENT_LOCAL,
               "Discrete derivative mode must be 1 (interpolated source) or "
               "2 (element local).");
   MFEM_VERIFY(marking_type >= 0,
               "Surface fitting marking must be nonnegative.");
   MFEM_VERIFY(mesh_node_ordering == Ordering::byNODES ||
               mesh_node_ordering == Ordering::byVDIM,
               "Mesh node ordering must be 0 (byNODES) or 1 (byVDIM).");
   MFEM_VERIFY(conv_residual || surface_fit_threshold > 0.0,
               "Error-based convergence (-no-resid) requires a positive "
               "surface fitting threshold (-sft).");
   MFEM_VERIFY(surface_fit_adapt == 0.0 || surface_fit_adapt > 1.0,
               "Adaptive surface fitting factor must be zero (disabled) "
               "or greater than one.");
   MFEM_VERIFY(surface_fit_const_max >= surface_fit_const,
               "Maximum surface fitting coefficient must be at least the "
               "initial coefficient.");
   MFEM_VERIFY(!adapt_marking && !material,
               "Adaptive marking and material-attribute marking are not yet "
               "supported by pmesh-fitting-enzyme.");
   MFEM_VERIFY(!comp_dist || surf_bg_mesh,
               "Distance conversion requires a background mesh.");
   MFEM_VERIFY(bg_amr_iters >= 0,
               "The number of background AMR iterations must be "
               "nonnegative.");
   MFEM_VERIFY(bg_amr_iters == 0 || surf_bg_mesh,
               "Background AMR requires a background mesh.");

   Mesh mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "pmesh-fitting-enzyme supports 2D and 3D meshes.");
   MFEM_VERIFY((dim == 2 &&
                (metric_id == 2 || metric_id == 58 || metric_id == 80)) ||
               (dim == 3 && metric_id == 303),
               "Metric id " << metric_id << " is incompatible with "
               << dim << "D meshes.");
   MFEM_VERIFY((dim == 2 && surf_ls_type != SurfaceFittingOptions::SPHERE) ||
               (dim == 3 && surf_ls_type == SurfaceFittingOptions::SPHERE),
               "Use circle/reactor/squircle level sets in 2D and the sphere "
               "level set in 3D.");
   if (mesh_poly_deg <= 0) { mesh_poly_deg = 2; }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec, dim, mesh_node_ordering);
   pmesh.SetNodalFESpace(&pfes);
   ParGridFunction x(&pfes);
   pmesh.SetNodalGridFunction(&x);
   ParGridFunction x0(x);

   if (mod_bndr_attr)
   {
      ModifyBoundaryAttributesForNodeMovement(&pmesh, x);
      pmesh.SetAttributes();
   }
   pmesh.ExchangeFaceNbrData();

   H1_FECollection surface_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace surface_fes(&pmesh, &surface_fec);
   L2_FECollection material_fec(0, dim);
   ParFiniteElementSpace material_fes(&pmesh, &material_fec);
   ParGridFunction surface_gf0(&surface_fes);
   ParGridFunction surface_marker_vis(&surface_fes);
   ParGridFunction material_vis(&material_fes);

   std::unique_ptr<FunctionCoefficient> level_set_coeff;
   if (surf_ls_type == SurfaceFittingOptions::CIRCLE)
   {
      level_set_coeff = std::make_unique<FunctionCoefficient>(circle_level_set);
   }
   else if (surf_ls_type == 2)
   {
      level_set_coeff = std::make_unique<FunctionCoefficient>(reactor);
   }
   else if (surf_ls_type == SurfaceFittingOptions::SPHERE)
   {
      level_set_coeff = std::make_unique<FunctionCoefficient>(sphere_level_set);
   }
   else
   {
      level_set_coeff =
         std::make_unique<FunctionCoefficient>(squircle_level_set);
   }
   surface_gf0.ProjectCoefficient(*level_set_coeff);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      material_vis(e) = material_id(e, surface_gf0);
   }

   std::unique_ptr<ParMesh> surface_bg_mesh;
   std::unique_ptr<H1_FECollection> surface_bg_fec;
   std::unique_ptr<ParFiniteElementSpace> surface_bg_fes;
   std::unique_ptr<ParGridFunction> surface_bg_level_set;
   if (surf_bg_mesh)
   {
      Mesh serial_bg = dim == 2 ?
                        Mesh::MakeCartesian2D(
                           4, 4, Element::QUADRILATERAL, true) :
                        Mesh::MakeCartesian3D(
                           4, 4, 4, Element::HEXAHEDRON, true);
      serial_bg.EnsureNCMesh();
      surface_bg_mesh =
         std::make_unique<ParMesh>(MPI_COMM_WORLD, serial_bg);
      surface_bg_mesh->SetCurvature(mesh_poly_deg);

      Vector p_min(dim), p_max(dim);
      pmesh.GetBoundingBox(p_min, p_max);
      GridFunction &x_bg = *surface_bg_mesh->GetNodes();
      const int bg_nodes = x_bg.Size() / dim;
      for (int i = 0; i < bg_nodes; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            const real_t length = p_max(d) - p_min(d);
            const real_t extra = 0.2 * length;
            x_bg(i + d * bg_nodes) = p_min(d) - extra +
                                     x_bg(i + d * bg_nodes) *
                                     (length + 2.0 * extra);
         }
      }
      surface_bg_mesh->NodesUpdated();

      surface_bg_fec =
         std::make_unique<H1_FECollection>(mesh_poly_deg + 1, dim);
      surface_bg_fes = std::make_unique<ParFiniteElementSpace>(
                          surface_bg_mesh.get(), surface_bg_fec.get());
      surface_bg_level_set =
         std::make_unique<ParGridFunction>(surface_bg_fes.get());

      OptimizeMeshWithAMRAroundZeroLevelSet(
         *surface_bg_mesh, *level_set_coeff, bg_amr_iters,
         *surface_bg_level_set);
      surface_bg_mesh->Rebalance();
      surface_bg_fes->Update();
      surface_bg_level_set->Update();
      if (comp_dist)
      {
         ComputeScalarDistanceFromLevelSet(
            *surface_bg_mesh, *level_set_coeff, *surface_bg_level_set);
      }
      else
      {
         surface_bg_level_set->ProjectCoefficient(*level_set_coeff);
      }
   }

   Array<bool> surface_marker;
   MarkSurfaceFittingDofs(pmesh, surface_gf0, material_vis, marking_type,
                          surface_marker, &surface_marker_vis);

   SurfaceFittingOptions surface_options;
   surface_options.source = analytic_level_set ?
                            SurfaceFittingOptions::ANALYTIC :
                            SurfaceFittingOptions::DISCRETE;
   surface_options.analytic_level_set =
      surf_ls_type == SurfaceFittingOptions::SQUIRCLE ?
      SurfaceFittingOptions::SQUIRCLE :
      (surf_ls_type == SurfaceFittingOptions::SPHERE ?
       SurfaceFittingOptions::SPHERE : SurfaceFittingOptions::CIRCLE);
   surface_options.discrete_derivative_mode =
      static_cast<SurfaceFittingOptions::DiscreteDerivativeMode>(
         discrete_derivative_mode);
   surface_options.discrete_level_set =
      analytic_level_set ? nullptr :
      (surf_bg_mesh ? surface_bg_level_set.get() : &surface_gf0);
   surface_options.discrete_from_background = surf_bg_mesh;
   surface_options.marker = &surface_marker;
   surface_options.coefficient = surface_fit_const;

   SaveMesh(pmesh, "perturbed.mesh");
   if (visualization)
   {
      socketstream vis1, vis2, vis3, vis4, vis5;
      common::VisualizeField(vis1, "localhost", 19916, surface_gf0,
                             "Level Set", 0, 0, 300, 300);
      common::VisualizeField(vis2, "localhost", 19916, material_vis,
                             "Materials", 300, 0, 300, 300);
      common::VisualizeField(vis3, "localhost", 19916, surface_marker_vis,
                             "Surface DOFs", 600, 0, 300, 300);
      if (surf_bg_mesh)
      {
         common::VisualizeField(vis4, "localhost", 19916,
                                *surface_bg_level_set,
                                "Level Set - Background",
                                0, 400, 300, 300);
      }
   }

   IntegrationRules &irules = SelectIntegrationRules(quad_type);
   if (Mpi::Root())
   {
      std::cout << "Triangle quadrature points: "
                << irules.Get(Geometry::TRIANGLE, quad_order).GetNPoints()
                << "\nQuadrilateral quadrature points: "
                << irules.Get(Geometry::SQUARE, quad_order).GetNPoints()
                << '\n';
      std::cout << "Using "
                << (analytic_level_set ?
                    "analytic Enzyme" :
                    (surf_bg_mesh ? "discrete background-mesh GSLIB" :
                     "discrete initial-mesh GSLIB"))
                << " level-set updates and metric mu" << metric_id << ".\n";
      if (!analytic_level_set && !surf_bg_mesh)
      {
         std::cout << "Using "
                   << (discrete_derivative_mode ==
                       SurfaceFittingOptions::INTERPOLATED_SOURCE ?
                       "initial-mesh interpolated" : "element-local")
                   << " discrete level-set derivatives.\n";
      }
   }

   const real_t min_detJ = MinimumDetJ(pmesh, pfes, irules, quad_order);
   if (Mpi::Root())
   {
      std::cout << "Minimum det(J) of the original mesh is "
                << min_detJ << '\n';
   }
   MFEM_VERIFY(min_detJ > 0.0, "The input mesh is inverted, use "
               "pmesh-optimizer-enzyme first.");

   Array<int> ess_tdofs;
   const auto get_essential_tdofs = dim == 2 ?
                                    &GetFittingEssentialTrueDofs<2> :
                                    &GetFittingEssentialTrueDofs<3>;
   get_essential_tdofs(pfes, move_bnd, marking_type, ess_tdofs);
   if (Mpi::Root())
   {
      std::cout << "Fixed true dofs: " << ess_tdofs.Size() << '\n';
   }

   const real_t fitting_tolerance =
      conv_residual ? -1.0 : surface_fit_threshold;
   const auto run_optimizer = dim == 2 ? &RunOptimizer<2> : &RunOptimizer<3>;
   const int result = run_optimizer(
                         pmesh, pfes, x, irules, quad_order, ess_tdofs,
                         min_detJ, solver_iter, solver_rtol, solver_atol,
                         lin_solver, solver_art_type, max_lin_iter,
                         metric_id, verbosity, surface_options,
                         fitting_tolerance, surface_fit_adapt,
                         surface_fit_threshold, surface_fit_const_max,
                         !conv_residual, surface_gf0);

   SaveMesh(pmesh, "optimized.mesh");
   if (visualization)
   {
      socketstream vis1, vis2, vis3;
      common::VisualizeField(vis1, "localhost", 19916, surface_gf0,
                             "Level Set", 0, 400, 300, 300);
      common::VisualizeField(vis2, "localhost", 19916, material_vis,
                             "Materials", 300, 400, 300, 300);
      common::VisualizeField(vis3, "localhost", 19916, surface_marker_vis,
                             "Surface DOFs", 600, 400, 300, 300);

      x0 -= x;
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, x0,
                             "Displacements", 900, 400, 300, 300,
                             "jRmclA");
   }

   return result;
}

#else

/// Report that required MFEM build features are unavailable.
int main(int, char *[])
{
   mfem::err << "pmesh-fitting-enzyme requires MFEM_USE_MPI=YES and "
             << "MFEM_USE_ENZYME=YES and MFEM_USE_GSLIB=YES.\n";
   return MFEM_SKIP_RETURN_VALUE;
}

#endif // MFEM_USE_MPI && MFEM_USE_ENZYME && MFEM_USE_GSLIB
