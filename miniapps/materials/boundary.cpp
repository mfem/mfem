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

#include "boundary.hpp"

namespace mfem {
namespace materials {

void Boundary::PrintInfo(std::ostream &os) const {
  os << "\n<Boundary Info>\n"
     << " Boundary Conditions:\n";
  for (const auto &it : boundary_attributes) {
    os << "  Boundary " << it.first << ": ";
    switch (it.second) {
      case BoundaryType::kNeumann:
        os << "Neumann";
        break;
      case BoundaryType::kDirichlet:
        os << "Dirichlet";
        break;
      case BoundaryType::kRobin:
        os << "Robin, coefficient: " << robin_coefficient;
        break;
      case BoundaryType::kPeriodic:
        os << "Periodic";
        break;
      default:
        os << "Undefined";
        break;
    }
    os << "\n";
  }
  bool first_print_statement = true;
  // If the map is not empty
  if (!dirichlet_coefficients.empty()) {
    os << "  Inhomogeneous Dirichlet defined on ";
    for (const auto &it : dirichlet_coefficients) {
      if (!first_print_statement){
        os << ", ";
      } else {
        first_print_statement = false;
      }
      os << it.first;
    }
    os << "\n";
  }
  os << "<Boundary Info>\n\n";
}

void Boundary::VerifyDefinedBoundaries(const Mesh& mesh) const{
  // Verify that all defined boundaries are actually defined on the 
  // mesh, i.e. if the keys of boundary attributes appear in the boundary 
  // attributes of the mesh.
  mfem::out << "\n<Boundary Verify>\n";
  const Array<int> boundary (mesh.bdr_attributes);
  for (const auto &it : boundary_attributes) {
    if (boundary.Find(it.first) == -1) {
      MFEM_ABORT("  Boundary " 
                    << it.first 
                    << " is not defined on the mesh but in Boundary class."
                    << "Exiting...");
    }
  }

  /// Verify if all boundary attributes appear as keys in the 
  /// boundary attributes, if not let the user know that we use Neumann by 
  /// default.
  std::vector<int> boundary_attributes_keys;
  for (int i = 0; i < boundary.Size(); i++) {
    if (boundary_attributes.find(boundary[i]) == boundary_attributes.end()) {
      boundary_attributes_keys.push_back(boundary[i]);
    }
  }
  if (!boundary_attributes_keys.empty()) {
    mfem::out << "  Boundaries (";
    for (const auto &it : boundary_attributes_keys) {
      mfem::out << it << ", ";
    }  
    mfem::out << ") are defined on the mesh but not in the";
    mfem::out << " boundary attributes (Use Neumann).";
  }

  /// Check if any periodic boundary is registered
  for (const auto &it : boundary_attributes) {
    if (it.second == BoundaryType::kPeriodic) {
      MFEM_ABORT("  Periodic boundaries must be defined on the mesh"
                    << ", not in Boundaries. Exiting...");
    }
  }

  mfem::out << "\n<Boundary Verify>\n\n";
}

void Boundary::AddHomogeneousBoundaryCondition(int boundary, BoundaryType type) {
  boundary_attributes[boundary] = type;
}

void Boundary::AddInhomogeneousDirichletBoundaryCondition(
  int boundary, Coefficient* coefficient){
  boundary_attributes[boundary] = BoundaryType::kDirichlet;
  dirichlet_coefficients[boundary] = coefficient;
}

void Boundary::SetRobinCoefficient(double coefficient){
  robin_coefficient = coefficient;
};

}
}

