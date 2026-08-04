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
//   Initial-mesh interpolation versus direct analytic evaluation:
//     mpirun -np 4 pmesh-fitting         -o 3 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -ae 1
//     mpirun -np 4 pmesh-fitting-enzyme  -o 3 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -dls
//     mpirun -np 4 pmesh-fitting-enzyme  -o 3 -mid 58 -tid 1 -vl 1 -sfc 5e4 -rtol 1e-5 -als
//   Background-mesh interpolation versus direct analytic evaluation:
//     mpirun -np 4 pmesh-fitting         -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -rtol 1e-6 -ae 1 -sbgmesh
//     mpirun -np 4 pmesh-fitting-enzyme  -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -rtol 1e-6 -dls -sbgmesh
//     mpirun -np 4 pmesh-fitting-enzyme  -o 2 -mid 2 -tid 1 -vl 1 -sfc 10 -rtol 1e-6 -als

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
static constexpr int TARGET_W = 2;
static constexpr int SURFACE_FIT_DATA = 3;
static constexpr int SURFACE_FIT_DATA_SIZE = 17;

/// Compute the MPI-global Euclidean norm of a distributed vector.
real_t GlobalVectorNorm(MPI_Comm comm, const Vector &x)
{
   const real_t local_norm2 = x * x;
   real_t global_norm2 = 0.0;
   MPI_Allreduce(&local_norm2, &global_norm2, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   return std::sqrt(global_norm2);
}

/// Evaluate the selected two-dimensional TMOP quality metric.
template <typename scalar_t, int metric_id>
MFEM_HOST_DEVICE inline
scalar_t EvaluateTMOPMetric(const tensor<scalar_t, 2, 2> &T)
{
   const auto tau = det(T);
   const auto norm2 = sqnorm(T);

   if constexpr (metric_id == 2)
   {
      return 0.5_r * norm2 / tau - 1.0_r;
   }
   else if constexpr (metric_id == 58)
   {
      const auto i1b = norm2 / tau;
      return i1b * (i1b - 2.0_r);
   }
   else if constexpr (metric_id == 80)
   {
      const auto mu2 = 0.5_r * norm2 / tau - 1.0_r;
      const auto tau2 = tau * tau;
      const auto mu77 = 0.5_r * (tau2 + 1.0_r / tau2) - 1.0_r;
      return 0.5_r * (mu2 + mu77);
   }
   else
   {
      static_assert(metric_id == 2 || metric_id == 58 || metric_id == 80,
                    "Unsupported metric id");
      return 0.0_r;
   }
}

template <typename scalar_t, int metric_id>
struct TMOPEnergy
{
   /// Evaluate the weighted TMOP energy at one quadrature point.
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<scalar_t, 2, 2> &dxdr,
                   const tensor<real_t, 2, 2> &W,
                   const real_t &w_q,
                   real_t &f) const
   {
      const auto T = dxdr * inv(W);
      f = EvaluateTMOPMetric<scalar_t, metric_id>(T) * det(W) * w_q;
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
      SQUIRCLE = 3
   };

   LevelSetSource source = ANALYTIC;
   AnalyticLevelSet analytic_level_set = CIRCLE;
   const ParGridFunction *discrete_level_set = nullptr;
   bool discrete_from_background = false;
   const Array<bool> *marker = nullptr;
   real_t coefficient = 0.0;
};

template <typename scalar_t>
struct DiscreteSurfaceFittingEnergy
{
   /// Evaluate a local Taylor model of the discrete fitting penalty.
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<scalar_t, 2> &x,
                   const tensor<scalar_t, SURFACE_FIT_DATA_SIZE> &data,
                   real_t &f) const
   {
      constexpr int center_offset = 1;
      constexpr int value_offset = center_offset + 2;
      constexpr int grad_offset = value_offset + 1;
      constexpr int hess_offset = grad_offset + 2;

      scalar_t dx[2];
      scalar_t sigma = data(value_offset);
      for (int d = 0; d < 2; d++)
      {
         dx[d] = x(d) - data(center_offset + d);
         sigma += data(grad_offset + d) * dx[d];
      }
      for (int i = 0; i < 2; i++)
      {
         for (int j = 0; j < 2; j++)
         {
            sigma += 0.5_r * data(hess_offset + 2 * i + j) * dx[i] * dx[j];
         }
      }
      f = data(0) * sigma * sigma;
   }
};

template <typename scalar_t, int level_set>
struct AnalyticSurfaceFittingEnergy
{
   /// Evaluate the analytic fitting penalty for Enzyme differentiation.
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<scalar_t, 2> &x,
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
         const scalar_t r = sqrt(xc * xc + yc * yc + 1e-12_r);
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
        current_fec(order, 2, basis),
        current_fes(&pmesh, &current_fec),
        current_grad_fes(&pmesh, &current_fec, 2, Ordering::byNODES),
        current_hess_fes(&pmesh, &current_fec, 4, Ordering::byNODES),
        current_sigma(&current_fes),
        current_grad(&current_grad_fes),
        current_hess(&current_hess_fes),
        marker(*options.marker),
        coefficient(options.coefficient),
        source(options.source),
        analytic_level_set(options.analytic_level_set),
        discrete_from_background(options.discrete_from_background)
   {
      MFEM_VERIFY(pmesh.Dimension() == 2,
                  "Surface fitting is currently implemented only in 2D.");
      MFEM_VERIFY(options.marker != nullptr,
                  "Surface fitting requires a DOF marker.");
      MFEM_VERIFY(coefficient > 0.0,
                  "Surface fitting requires a positive coefficient.");
      MFEM_VERIFY(marker.Size() == current_fes.GetVSize(),
                  "Surface fitting marker size does not match the scalar "
                  "node space.");

      ParGridFunction counter(&current_fes);
      counter.CountElementsPerVDof(dof_count);

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
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const int dof = dofs[lex_to_native ? (*lex_to_native)[q] : q];
            FillNodeData(dof,
                         qdata_ptr + (offset + q) * SURFACE_FIT_DATA_SIZE);
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
      current_fes.GetNodePositions(*mesh_nodes, current_node_pos,
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

   /// Freeze the discrete source field and initialize its GSLIB finder.
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
                      level_fec->GetOrder(), 2, level_fec->GetBasisType());
      source_fes = std::make_unique<ParFiniteElementSpace>(
                      source_mesh.get(), source_fec.get());
      MFEM_VERIFY(level_set.Size() == source_fes->GetVSize(),
                  "The copied source space is incompatible with the "
                  "discrete level set.");
      source_sigma = std::make_unique<ParGridFunction>(source_fes.get());
      *source_sigma = level_set;

      if (discrete_from_background)
      {
         source_grad_fes = std::make_unique<ParFiniteElementSpace>(
                              source_mesh.get(), source_fec.get(), 2,
                              Ordering::byNODES);
         source_hess_fes = std::make_unique<ParFiniteElementSpace>(
                              source_mesh.get(), source_fec.get(), 4,
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
      for (int d = 0; d < 2; d++)
      {
         ParGridFunction grad_comp(
            &scalar_fes, gradient.GetData() + d * scalar_size);
         sigma.GetDerivative(1, d, grad_comp);
      }

      int id = 0;
      for (int d = 0; d < 2; d++)
      {
         ParGridFunction grad_comp(
            &scalar_fes, gradient.GetData() + d * scalar_size);
         for (int idir = 0; idir < 2; idir++)
         {
            ParGridFunction hess_comp(
               &scalar_fes, hessian.GetData() + id * scalar_size);
            grad_comp.GetDerivative(1, idir, hess_comp);
            id++;
         }
      }
   }

   /// Sample discrete fitting data at the current physical node positions.
   void UpdateDiscreteSamples() const
   {
#ifdef MFEM_USE_GSLIB
      finder->FindPoints(current_node_pos, Ordering::byNODES);
      finder->Interpolate(*source_sigma, sigma_samples, Ordering::byNODES);
      if (discrete_from_background)
      {
         finder->Interpolate(*source_grad, grad_samples, Ordering::byNODES);
         finder->Interpolate(*source_hess, hess_samples, Ordering::byNODES);
      }
      else
      {
         current_sigma = sigma_samples;
         ComputeDerivatives(current_sigma, current_grad, current_hess,
                            current_fes);
         grad_samples = current_grad;
         hess_samples = current_hess;
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
            sigma_samples(i) = std::pow(xc, 4.0) + std::pow(yc, 4.0) -
                               std::pow(radius, 4.0);
         }
         else
         {
            const real_t r = std::sqrt(xc * xc + yc * yc);
            sigma_samples(i) = r - 0.25;
         }
      }
   }

   /// Pack one scalar DOF's coefficient and Taylor data into quadrature data.
   void FillNodeData(int dof, real_t *data) const
   {
      for (int j = 0; j < SURFACE_FIT_DATA_SIZE; j++) { data[j] = 0.0; }
      const int ndofs = current_fes.GetVSize();
      data[0] = marker[dof] ? coefficient / dof_count[dof] : 0.0;
      if (source == SurfaceFittingOptions::ANALYTIC) { return; }

      data[1] = current_node_pos(dof);
      data[2] = current_node_pos(dof + ndofs);
      data[3] = sigma_samples(dof);
      data[4] = grad_samples(dof);
      data[5] = grad_samples(dof + ndofs);
      for (int j = 0; j < 4; j++)
      {
         data[6 + j] = hess_samples(dof + j * ndofs);
      }
   }

   ParMesh &mesh;
   int order;
   int basis;
   H1_FECollection current_fec;
   mutable ParFiniteElementSpace current_fes;
   mutable ParFiniteElementSpace current_grad_fes;
   mutable ParFiniteElementSpace current_hess_fes;
   mutable ParGridFunction current_sigma;
   mutable ParGridFunction current_grad;
   mutable ParGridFunction current_hess;
   Array<bool> marker;
   Array<int> dof_count;
   real_t coefficient;
   SurfaceFittingOptions::LevelSetSource source;
   SurfaceFittingOptions::AnalyticLevelSet analytic_level_set;
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
                              surface_qspace, SURFACE_FIT_DATA_SIZE)),
        surface_qdata(is_analytic ? nullptr :
                     new QuadratureFunction(*surface_data_qspace)),
        current_nodes(new ParGridFunction(&fes_)),
        surface_data(new SurfaceFittingData(mesh_, fes_, options))
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

   /// Return the active analytic coefficient or discrete Taylor data.
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
         2,
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
   SurfaceFittingData *surface_data;
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
        target_qspace(metric_qspace, 4),
        target_w(target_qspace),
        metric_scalar_qspace(metric_qspace, 1),
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
      SetupMetricOperator(ir, all_domain_attr, metric_id);
      SetupSurfaceOperator(all_domain_attr, surface_options);
   }

   /// Evaluate and globally sum the TMOP metric energy.
   real_t MetricEnergy(const Vector &x) const
   {
      metric_values = 0.0;
      MultiVector input{x, target_w};
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
      MultiVector metric_input{x, target_w};
      MultiVector metric_output{gradient};
      metric_operator->GetDerivative(X)->Mult(metric_input, metric_output);

      surface_gradient.SetSize(gradient.Size());
      surface_gradient = 0.0;
      MultiVector surface_input{x, surface_state_mgr.GetSurfaceQuadratureData()};
      MultiVector surface_output{surface_gradient};
      surface_operator->GetDerivative(X)->Mult(surface_input, surface_output);
      gradient += surface_gradient;
   }

   /// Construct the summed Enzyme-generated Hessian at the current state.
   std::unique_ptr<Operator> HessianOperator(const Vector &x) const
   {
      MultiVector metric_input{x, target_w};
      auto metric_hessian = std::make_unique<SingleOutputDerivativeOperator>(
                               metric_operator->GetSecondDerivative(
                                  X, metric_input), fes);

      MultiVector surface_input{x, surface_state_mgr.GetSurfaceQuadratureData()};
      auto surface_hessian =
         std::make_unique<SingleOutputDerivativeOperator>(
            surface_operator->GetSecondDerivative(X, surface_input), fes);
      return std::make_unique<SumWithDiagonalOperator>(
                std::move(metric_hessian), std::move(surface_hessian));
   }

   /// Return global fitting errors for the current surface state.
   void GetSurfaceErrors(real_t &err_avg, real_t &err_max) const
   {
      surface_state_mgr.GetSurfaceErrors(err_avg, err_max);
   }

   /// Expose state updates to the nonlinear solver callback.
   SurfaceFittingStateManager& GetSurfaceStateManager()
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
   void SetupMetricOperator(const IntegrationRule &ir,
                            const Array<int> &all_domain_attr,
                            int metric_id)
   {
      switch (metric_id)
      {
         case 2: return SetupMetricOperator<2>(ir, all_domain_attr);
         case 58: return SetupMetricOperator<58>(ir, all_domain_attr);
         case 80: return SetupMetricOperator<80>(ir, all_domain_attr);
         default: MFEM_ABORT("Unsupported metric id: " << metric_id);
      }
   }

   /// Construct the differentiable operator for one TMOP metric.
   template <int metric_id>
   void SetupMetricOperator(const IntegrationRule &ir,
                            const Array<int> &all_domain_attr)
   {
      const std::vector<FieldDescriptor> input
      {
         FieldDescriptor{X, &fes},
         FieldDescriptor{TARGET_W, &target_qspace}
      };
      const std::vector<FieldDescriptor> output
      {
         FieldDescriptor{Q, &metric_scalar_qspace}
      };
      metric_operator =
         std::make_unique<DifferentiableOperator>(input, output, mesh);
      TMOPEnergy<real_t, metric_id> energy;
      auto derivatives = std::integer_sequence<size_t, X> {};
      metric_operator->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{future::Gradient<X>{}, Identity<TARGET_W>{}, Weight{}},
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
      else if (options.analytic_level_set == SurfaceFittingOptions::CIRCLE)
      {
         SetupAnalyticSurfaceOperator<SurfaceFittingOptions::CIRCLE>(
            all_domain_attr);
      }
      else
      {
         MFEM_VERIFY(options.analytic_level_set ==
                     SurfaceFittingOptions::SQUIRCLE,
                     "Unsupported analytic level set.");
         SetupAnalyticSurfaceOperator<SurfaceFittingOptions::SQUIRCLE>(
            all_domain_attr);
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
      AnalyticSurfaceFittingEnergy<real_t, level_set> energy;
      surface_operator->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{Value<X>{}, Identity<Q>{}},
         future::tuple{FunctionalValue<Q>{}},
         surface_node_ir, all_domain_attr, derivatives);
   }

   /// Construct a fitting operator driven by sampled discrete Taylor data.
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
      DiscreteSurfaceFittingEnergy<real_t> energy;
      surface_operator->AddDomainIntegrator<LocalQFBackend>(
         energy,
         future::tuple{Value<X>{}, Identity<SURFACE_FIT_DATA>{}},
         future::tuple{FunctionalValue<Q>{}},
         surface_node_ir, all_domain_attr, derivatives);
   }

   /// Fill ideal target Jacobians at all metric quadrature points.
   void SetTargetData()
   {
      real_t *data = target_w.HostWrite();
      for (int e = 0; e < metric_qspace.GetNE(); e++)
      {
         const DenseMatrix &W =
            Geometries.GetGeomToPerfGeomJac(metric_qspace.GetGeometry(e));
         MFEM_VERIFY(W.Height() == 2 && W.Width() == 2,
                     "Unexpected target matrix dimension.");
         const int offset = metric_qspace.Offset(e);
         const int nq = metric_qspace.GetIntRule(e).GetNPoints();
         for (int q = 0; q < nq; q++)
         {
            real_t *Wq = data + 4 * (offset + q);
            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++) { Wq[2 * i + j] = W(i, j); }
            }
         }
      }
   }

   MPI_Comm comm;
   ParMesh &mesh;
   ParFiniteElementSpace &fes;
   QuadratureSpace metric_qspace;
   IntegrationRule surface_node_ir;
   VectorQuadratureSpace target_qspace;
   QuadratureFunction target_w;
   VectorQuadratureSpace metric_scalar_qspace;
   mutable QuadratureFunction metric_values;
   SurfaceFittingStateManager surface_state_mgr;
   VectorQuadratureSpace surface_scalar_qspace;
   mutable QuadratureFunction surface_values;
   std::unique_ptr<DifferentiableOperator> metric_operator;
   std::unique_ptr<DifferentiableOperator> surface_operator;
   mutable Vector surface_gradient;
};

class EnzymeFittingNonlinearForm : public ParNonlinearForm
{
public:
   /// Adapt the absolute-coordinate functional to MFEM's displacement form.
   EnzymeFittingNonlinearForm(ParFiniteElementSpace &fes,
                              EnzymeFittingFunctional &functional_)
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
   EnzymeFittingFunctional &functional;
   Vector reference_state;
   mutable Vector absolute_state;
   mutable std::unique_ptr<Operator> hessian;
   mutable std::unique_ptr<ConstrainedOperator> constrained_hessian;
};

class EnzymeFittingNewtonSolver : public TMOPNewtonSolver
{
public:
   /// Attach the fitting nonlinear form to the TMOP Newton solver.
   EnzymeFittingNewtonSolver(MPI_Comm comm,
                             const IntegrationRule &ir,
                             EnzymeFittingNonlinearForm &nlf_)
      : TMOPNewtonSolver(comm, ir, 0),
        enzyme_nlf(nlf_) { }

   /// Synchronize fitting fields with each Newton or line-search state.
   void ProcessNewState(const Vector &dx) const override
   {
      // Convert displacement to absolute position
      const Vector &x_abs = enzyme_nlf.ComputeAbsoluteState(dx);

      // Update the discrete fitting fields for every Newton or line-search
      // state before its energy, residual, or Hessian is evaluated.
      enzyme_nlf.UpdateSurfaceFittingState(x_abs);
   }

private:
   EnzymeFittingNonlinearForm &enzyme_nlf;
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
      MFEM_VERIFY(attr != 3,
                  "Boundary attribute 3 is valid only for 3D meshes.");
      if (attr == 1 || attr == 2) { count += ndofs; }
      if (attr == 4) { count += 2 * ndofs; }
   }

   Array<int> vdofs, ess_vdofs(count);
   count = 0;
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      const int ndofs = fes.GetBE(i)->GetDof();
      const int attr = pmesh->GetBdrElement(i)->GetAttribute();
      fes.GetBdrElementVDofs(i, vdofs);
      if (attr == 1)
      {
         for (int j = 0; j < ndofs; j++) { ess_vdofs[count++] = vdofs[j]; }
      }
      else if (attr == 2)
      {
         for (int j = 0; j < ndofs; j++)
         {
            ess_vdofs[count++] = vdofs[j + ndofs];
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
void GetFittingEssentialTrueDofs(const ParFiniteElementSpace &pfes,
                                 bool move_bnd,
                                 int marking_type,
                                 Array<int> &ess_tdofs)
{
   if (move_bnd || marking_type == 0)
   {
      GetMeshOptimizerEssentialTrueDofs(pfes, move_bnd, ess_tdofs);
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
                 ParGridFunction &final_level_set)
{
   Vector true_nodes(fes.GetTrueVSize());
   nodes.GetTrueDofs(true_nodes);
   const IntegrationRule &ir =
      irules.Get(pmesh.GetTypicalElementGeometry(), quad_order);
   EnzymeFittingFunctional functional(fes, pmesh, ir, metric_id,
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

   EnzymeFittingNonlinearForm nonlinear_form(fes, functional);
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

   EnzymeFittingNewtonSolver solver(fes.GetComm(), ir, nonlinear_form);
   solver.SetIntegrationRules(irules, quad_order);
   solver.SetMinDetPtr(&min_detJ);
   solver.SetOperator(nonlinear_form);
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
   solver.SetPrintLevel(newton_print);

   Vector zero;
   solver.Mult(zero, true_nodes);

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

   bool converged = solver.GetConverged();
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
   real_t surface_fit_const = 100.0;
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
   const char *devopt = "cpu";
   real_t surface_fit_adapt = 0.0;
   real_t surface_fit_threshold = -10.0;
   real_t surface_fit_const_max = 1e20;
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
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Adaptive fitting-weight scaling (not yet supported).");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Maximum fitting error for error-based termination.");
   args.AddOption(&surface_fit_const_max, "-sfcmax", "--surf-fit-const-max",
                  "Maximum adaptive fitting weight (not yet supported).");
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
   MFEM_VERIFY(metric_id == 2 || metric_id == 58 || metric_id == 80,
               "pmesh-fitting-enzyme supports metric ids 2, 58, and 80.");
   MFEM_VERIFY(solver_type == 0,
               "pmesh-fitting-enzyme currently supports Newton (-st 0) only.");
   MFEM_VERIFY(lin_solver == 2 || lin_solver == 3,
               "pmesh-fitting-enzyme supports linear solvers 2 and 3.");
   MFEM_VERIFY(solver_art_type >= 0 && solver_art_type <= 2,
               "Unknown adaptive relative tolerance option: "
               << solver_art_type);
   MFEM_VERIFY(surf_ls_type == SurfaceFittingOptions::CIRCLE ||
               surf_ls_type == 2 ||
               surf_ls_type == SurfaceFittingOptions::SQUIRCLE,
               "Supported level sets are 1 (circle), 2 (reactor), and "
               "3 (squircle).");
   MFEM_VERIFY(!analytic_level_set || surf_ls_type != 2,
               "The reactor level set is available only as a discrete "
               "level set (-dls).");
   MFEM_VERIFY(!surf_bg_mesh || !analytic_level_set,
               "A background mesh is used only with a discrete level set "
               "(-dls).");
   MFEM_VERIFY(marking_type >= 0,
               "Surface fitting marking must be nonnegative.");
   MFEM_VERIFY(mesh_node_ordering == Ordering::byNODES ||
               mesh_node_ordering == Ordering::byVDIM,
               "Mesh node ordering must be 0 or 1.");
   MFEM_VERIFY(conv_residual || surface_fit_threshold > 0.0,
               "Error-based convergence (-no-resid) requires a positive "
               "surface fitting threshold (-sft).");
   MFEM_VERIFY(surface_fit_adapt == 0.0,
               "Adaptive surface fitting weights (-sfa) are not yet "
               "supported by pmesh-fitting-enzyme.");
   MFEM_VERIFY(surface_fit_const_max > 0.0,
               "Maximum surface fitting coefficient must be positive.");
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
   MFEM_VERIFY(dim == 2,
               "pmesh-fitting-enzyme currently supports only 2D meshes.");
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
      Mesh serial_bg(
         Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true));
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
      SurfaceFittingOptions::SQUIRCLE : SurfaceFittingOptions::CIRCLE;
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
   GetFittingEssentialTrueDofs(pfes, move_bnd, marking_type, ess_tdofs);
   if (Mpi::Root())
   {
      std::cout << "Fixed true dofs: " << ess_tdofs.Size() << '\n';
   }

   const real_t fitting_tolerance =
      conv_residual ? -1.0 : surface_fit_threshold;
   const int result = RunOptimizer(
                         pmesh, pfes, x, irules, quad_order, ess_tdofs,
                         min_detJ, solver_iter, solver_rtol, solver_atol,
                         lin_solver, solver_art_type, max_lin_iter,
                         metric_id, verbosity, surface_options,
                         fitting_tolerance, surface_gf0);

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
