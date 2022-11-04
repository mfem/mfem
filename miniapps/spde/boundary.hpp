// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#ifndef BOUNDARY_HPP
#define BOUNDARY_HPP

#include <ostream>
#include <unordered_map>
#include "mfem.hpp"

namespace mfem {
namespace spde {

enum BoundaryType { kNeumann, kDirichlet, kRobin, kPeriodic, kUndefined };

struct Boundary {
  Boundary() = default;

  /// Print the information specifying the boundary conditions.
  void PrintInfo(std::ostream& os = mfem::out) const;

  /// Verify that all defined boundaries are actually defined on the mesh, i.e.
  /// if the keys of boundary attributes appear in the boundary attributes of
  /// the mesh.
  void VerifyDefinedBoundaries(const Mesh& mesh) const;

  /// Computes the error for each defined boundary attribute by calling back to
  /// the IntegrateBC.
  void ComputeBoundaryError(const ParGridFunction& solution);

  /// Helper function to compute the coefficients alpha, beta, gamma (see in
  /// `IntegrateBC`) for a given boundary attribute.
  void UpdateIntegrationCoefficients(int i, double& alpha, double& beta,
                                     double& gamma);

  /// Add a homogeneous boundary condition to the boundary.
  void AddHomogeneousBoundaryCondition(int boundary, BoundaryType type);

  /// Add a inhomogeneous Dirichlet boundary condition to the boundary.
  void AddInhomogeneousDirichletBoundaryCondition(int boundary,
                                                  double coefficient);

  /// Set the robin coefficient for the boundary.
  void SetRobinCoefficient(double coefficient);

  /// Map to assign homogeneous boundary conditions to defined boundary types.
  std::map<int, BoundaryType> boundary_attributes;
  /// Coefficient for inhomogeneous Dirichlet boundary conditions.
  std::map<int, double> dirichlet_coefficients;
  /// Coefficient for Robin boundary conditions (n.grad(u) + coeff u = 0) on
  /// defined boundaries.
  double robin_coefficient = 1.0;
};

/// IntegrateBC function from ex27p.cpp. For boundary verification.
/// Compute the average value of alpha*n.Grad(sol) + beta*sol over the boundary
/// attributes marked in bdr_marker. Also computes the L2 norm of
/// alpha*n.Grad(sol) + beta*sol - gamma over the same boundary.
double IntegrateBC(const ParGridFunction& x, const Array<int>& bdr,
                   double alpha, double beta, double gamma, double& glb_err);

}  // namespace spde
}  // namespace mfem

#endif  // BOUNDARY_HPP
