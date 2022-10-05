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
      os << it.first << "(=" << it.second << ")";
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

void Boundary::ComputeBoundaryError(const ParGridFunction& solution){
  const ParFiniteElementSpace &fes = *solution.ParFESpace();
  ParMesh &pmesh = *fes.GetParMesh();

  if (Mpi::Root()){
    mfem::out << "<Boundary::ComputeBoundaryError>" << "\n";
    mfem::out << "   GetVDim: " << fes.GetVDim() << "\n";
  }

  double alpha {0.0};
  double beta {1.0};
  double gamma {0.0};

  // Index i needs to be incremented by one to map to the boundary attributes
  // in the mesh.
  for (int i = 0; i < pmesh.bdr_attributes.Max(); i++) {
    double error, avg;
    Array<int> bdr(pmesh.bdr_attributes.Max());
    bdr = 0;
    bdr[i] = 1;

    UpdateIntegrationCoefficients(i+1, alpha, beta, gamma);
    avg = IntegrateBC(solution, bdr, alpha, beta, gamma, error);
    if (Mpi::Root()){
      mfem::out << "->Boundary " << i+1 << "\n";
      mfem::out << "    Alpha   : " << alpha << "\n";
      mfem::out << "    Beta    : " << beta << "\n";
      mfem::out << "    Gamma   : " << gamma << "\n";
      mfem::out << "    Average : " << avg << "\n";
      mfem::out << "    Error   : " << error << "\n\n";
    }
  }

  if (Mpi::Root()){
    mfem::out << "<Boundary::ComputeBoundaryError>" << std::endl;
  }
}

void Boundary::UpdateIntegrationCoefficients(int i, double& alpha, double& beta, 
                                     double& gamma){
  // Check if i is a key in boundary_attributes
  if (boundary_attributes.find(i) != boundary_attributes.end()) {
    switch (boundary_attributes[i]) {
      case BoundaryType::kNeumann:
        alpha = 1.0;
        beta = 0.0;
        gamma = 0.0;
        break;
      case BoundaryType::kDirichlet:
        alpha = 0.0;
        beta = 1.0;
        if (dirichlet_coefficients.find(i) != dirichlet_coefficients.end()) {
          gamma = dirichlet_coefficients[i];
        } else {
          gamma = 0.0;
        }
        break;
      case BoundaryType::kRobin:
        alpha = 1.0;
        beta = robin_coefficient;
        gamma = 0.0;
        break;
      default:
        alpha = 1.0;
        beta = 0.0;
        gamma = 0.0;
        break;
    }
  } else {
    // If i is not a key in boundary_attributes, it corresponds to Neumann.
    alpha = 1.0;
    beta = 0.0;
    gamma = 0.0;
  }
}

void Boundary::AddHomogeneousBoundaryCondition(int boundary, BoundaryType type) {
  boundary_attributes[boundary] = type;
}

void Boundary::AddInhomogeneousDirichletBoundaryCondition(
  int boundary, double coefficient){
  boundary_attributes[boundary] = BoundaryType::kDirichlet;
  dirichlet_coefficients[boundary] = coefficient;
}

void Boundary::SetRobinCoefficient(double coefficient){
  robin_coefficient = coefficient;
};

double IntegrateBC(const ParGridFunction &x, const Array<int> &bdr,
                   double alpha, double beta, double gamma,
                   double &glb_err)
{
   double loc_vals[3];
   double &nrm = loc_vals[0];
   double &avg = loc_vals[1];
   double &error = loc_vals[2];

   nrm = 0.0;
   avg = 0.0;
   error = 0.0;

   const bool a_is_zero = alpha == 0.0;
   const bool b_is_zero = beta == 0.0;

   const ParFiniteElementSpace &fes = *x.ParFESpace();
   MFEM_ASSERT(fes.GetVDim() == 1, "");
   ParMesh &mesh = *fes.GetParMesh();
   Vector shape, loc_dofs, w_nor;
   DenseMatrix dshape;
   Array<int> dof_ids;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (bdr[mesh.GetBdrAttribute(i)-1] == 0) { continue; }

      FaceElementTransformations *FTr = mesh.GetBdrFaceTransformations(i);
      if (FTr == nullptr) { continue; }

      const FiniteElement &fe = *fes.GetFE(FTr->Elem1No);
      MFEM_ASSERT(fe.GetMapType() == FiniteElement::VALUE, "");
      const int int_order = 2*fe.GetOrder() + 3;
      const IntegrationRule &ir = IntRules.Get(FTr->FaceGeom, int_order);

      fes.GetElementDofs(FTr->Elem1No, dof_ids);
      x.GetSubVector(dof_ids, loc_dofs);
      if (!a_is_zero)
      {
         const int sdim = FTr->Face->GetSpaceDim();
         w_nor.SetSize(sdim);
         dshape.SetSize(fe.GetDof(), sdim);
      }
      if (!b_is_zero)
      {
         shape.SetSize(fe.GetDof());
      }
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         IntegrationPoint eip;
         FTr->Loc1.Transform(ip, eip);
         FTr->Face->SetIntPoint(&ip);
         double face_weight = FTr->Face->Weight();
         double val = 0.0;
         if (!a_is_zero)
         {
            FTr->Elem1->SetIntPoint(&eip);
            fe.CalcPhysDShape(*FTr->Elem1, dshape);
            CalcOrtho(FTr->Face->Jacobian(), w_nor);
            val += alpha * dshape.InnerProduct(w_nor, loc_dofs) / face_weight;
         }
         if (!b_is_zero)
         {
            fe.CalcShape(eip, shape);
            val += beta * (shape * loc_dofs);
         }

         // Measure the length of the boundary
         nrm += ip.weight * face_weight;

         // Integrate alpha * n.Grad(x) + beta * x
         avg += val * ip.weight * face_weight;

         // Integrate |alpha * n.Grad(x) + beta * x - gamma|^2
         val -= gamma;
         error += (val*val) * ip.weight * face_weight;
      }
   }

   double glb_vals[3];
   MPI_Allreduce(loc_vals, glb_vals, 3, MPI_DOUBLE, MPI_SUM, fes.GetComm());

   double glb_nrm = glb_vals[0];
   double glb_avg = glb_vals[1];
   glb_err = glb_vals[2];

   // Normalize by the length of the boundary
   if (std::abs(glb_nrm) > 0.0)
   {
      glb_err /= glb_nrm;
      glb_avg /= glb_nrm;
   }

   // Compute l2 norm of the error in the boundary condition (negative
   // quadrature weights may produce negative 'error')
   glb_err = (glb_err >= 0.0) ? sqrt(glb_err) : -sqrt(-glb_err);

   // Return the average value of alpha * n.Grad(x) + beta * x
   return glb_avg;
}

}
}

