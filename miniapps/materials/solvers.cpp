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

#include "solvers.hpp"
#include "examples/ex33.hpp"

namespace mfem {
namespace materials {

SPDESolver::SPDESolver(MatrixConstantCoefficient &diff_coefficient, double nu,
                       const Boundary& bc,
                       ParFiniteElementSpace *fespace)
    : nu_(nu), bc_(bc), fespace_ptr_(fespace),
      k_(fespace), m_(fespace), restriction_matrix_(nullptr),
      prolongation_matrix_(nullptr), Op_(nullptr) {
  if (Mpi::Root()) {
    mfem::out << "<SPDESolver> Initialize Solver .." << std::endl;
  }
  StopWatch sw;
  sw.Start();

  // Resize the marker arrays for the boundary conditions
  dbc_marker.SetSize(fespace_ptr_->GetParMesh()->bdr_attributes.Max());
  nbc_marker.SetSize(fespace_ptr_->GetParMesh()->bdr_attributes.Max());
  rbc_marker.SetSize(fespace_ptr_->GetParMesh()->bdr_attributes.Max());
  dbc_marker = 0;
  nbc_marker = 0;
  rbc_marker = 0;

  // Fill the marker arrays for the boundary conditions
  for (const auto &it : bc_.boundary_attributes) {
    switch (it.second) {
      case BoundaryType::kDirichlet:
        dbc_marker[it.first] = 1;
        break;
      case BoundaryType::kNeumann:
        nbc_marker[it.first] = 1;
        break;
      case BoundaryType::kRobin:
        rbc_marker[it.first] = 1;
        break;
      default:
        break;
    }
  }

  // Handle homogeneous Dirichlet boundary conditions 
  // Note: for non zero DBC we usually need to project the boundary onto the 
  // solution. This is not necessary in this case since the boundary is 
  // homogeneous. For inhomogeneous Dirichlet we consider a lifting scheme.
  fespace_ptr_->GetEssentialTrueDofs(dbc_marker, ess_tdof_list_);

  // Compute the rational approximation coefficients.
  int dim = fespace_ptr_->GetParMesh()->Dimension();
  alpha_ = (nu_ + dim / 2.0) / 2.0; // fractional exponent
  integer_order_of_exponent_ = std::floor(alpha_);
  double exponent_to_approximate = alpha_ - integer_order_of_exponent_;

  // Compute the rational approximation coefficients.
  ComputeRationalCoefficients(exponent_to_approximate);

  // Set the bilinear forms.

  // Assemble stiffness matrix
  k_.AddDomainIntegrator(new DiffusionIntegrator(diff_coefficient));
  ConstantCoefficient robin_coefficient(bc_.robin_coefficient);
  k_.AddBoundaryIntegrator(new MassIntegrator(robin_coefficient), rbc_marker);
  k_.Assemble();

  // Assemble mass matrix
  ConstantCoefficient one(1.0);
  m_.AddDomainIntegrator(new MassIntegrator(one));
  m_.Assemble();

  // Form matrices for the linear system
  Array<int> empty;
  k_.FormSystemMatrix(empty, stiffness_);
  m_.FormSystemMatrix(empty, mass_bc_);

  // Get the restriction and prolongation matrix for transformations
  restriction_matrix_ = fespace->GetRestrictionMatrix();
  prolongation_matrix_ = fespace->GetProlongationMatrix();

  // Resize the the vectors B and X to the appropriate size
  if (prolongation_matrix_) {
    B_.SetSize(prolongation_matrix_->Width());
  } else {
    mfem::err << "<SPDESolver> prolongation matrix is not defined" << std::endl;
  }
  if (restriction_matrix_) {
    X_.SetSize(restriction_matrix_->Height());
  } else {
    mfem::err << "<SPDESolver> restriction matrix is not defined" << std::endl;
  }

  sw.Stop();
  if (Mpi::Root()) {
    mfem::out << "<SPDESolver::Timing> matrix assembly " << sw.RealTime()
              << " [s]" << std::endl;
  }
}

void SPDESolver::Solve(ParLinearForm &b, ParGridFunction &x) {
  // ------------------------------------------------------------------------
  // Solve the PDE (A)^N g = f, i.e. compute g = (A)^{-1}^N f iteratively.
  // ------------------------------------------------------------------------

  StopWatch sw;
  sw.Start();

  ParGridFunction helper_gf(fespace_ptr_);
  k_.EliminateVDofsInRHS(ess_tdof_list_,helper_gf,b);


  if (integer_order_of_exponent_ > 0) {
    if (Mpi::Root()) {
      mfem::out << "<SPDESolver> Solving PDE (A)^" << integer_order_of_exponent_
                << " u = f" << std::endl;
    }
    ActivateRepeatedSolve();
    Solve(b, helper_gf, 1.0, 1.0, integer_order_of_exponent_);
    if (integer_order_) {
      x += helper_gf;
    }
    UpdateRHS(b);
    DeactivateRepeatedSolve();
  }

  // ------------------------------------------------------------------------
  // Solve the (remaining) fractional PDE by solving M integer order PDEs and
  // adding up the solutions.
  // ------------------------------------------------------------------------
  if (!integer_order_) {
    // Iterate over all expansion coefficient that contribute to the
    // solution.
    for (int i = 0; i < coeffs_.Size(); i++) {
      if (Mpi::Root()) {
        mfem::out << "\n<SPDESolver> Solving PDE -Δ u + " << -poles_[i]
                  << " u = " << coeffs_[i] << " g " << std::endl;
      }
      helper_gf = 0.0;
      Solve(b, helper_gf, 1.0 - poles_[i], coeffs_[i]);
      x += helper_gf;
    }
  }

  // Apply the inhomogeneous Dirichlet boundary conditions.
  if (!bc_.dirichlet_coefficients.empty()) {
    LiftSolution(x);
  }

  sw.Stop();
  if (Mpi::Root()) {
    mfem::out << "<SPDESolver::Timing> all PCG solves " << sw.RealTime()
              << " [s]" << std::endl;
  }
}

void SPDESolver::Solve(const ParLinearForm &b, ParGridFunction &x, double alpha,
                       double beta, int exponent) {
  // Form system of equations. This is less general than
  // BilinearForm::FormLinearSystem and kind of resembles the necessary subset
  // of instructions that we need in this case.
  if (prolongation_matrix_) {
    prolongation_matrix_->MultTranspose(b, B_);
  } else {
    B_ = b;
  }
  B_ *= beta;
  // Initialize X_ to zero. Important! Might contain nan/inf -> crash.
  X_ = 0.0;
  delete Op_;
  Op_ = Add(1.0, stiffness_, alpha, mass_bc_); //  construct Operator
  HypreParMatrix *Ae = Op_->EliminateRowsCols(ess_tdof_list_);
  Op_->EliminateBC(*Ae, ess_tdof_list_, X_, B_); // only for homogeneous BC

  for (int i = 0; i < exponent; i++) {
    // Solve the linear system Op_ X_ = B_
    SolveLinearSystem();
    k_.RecoverFEMSolution(X_, b, x);
    if (repeated_solve_) {
      // Prepare for next iteration. X is a primal and B is a dual vector. B_
      // must be updated to represent X_ in the next step. Instead of copying
      // it, we must transform it appropriately.
      GridFunctionCoefficient gfc(&x);
      ParLinearForm previous_solution(fespace_ptr_);
      previous_solution.AddDomainIntegrator(new DomainLFIntegrator(gfc));
      previous_solution.Assemble();
      prolongation_matrix_->MultTranspose(previous_solution, B_);
      Op_->EliminateBC(*Ae, ess_tdof_list_, X_, B_);
    }
  }
  delete Ae;
}

void SPDESolver::LiftSolution(ParGridFunction& x){
  // Lifting of the solution takes care of inhomogeneous boundary conditions.
  // See doi:10.1016/j.jcp.2019.109009; section 2.6
  if (Mpi::Root()) {
    mfem::out << "\n<SPDESolver> Applying inhomogeneous DBC" << std::endl;
  }

  // Define temporary grid function for lifting.
  ParGridFunction helper_gf(fespace_ptr_);
  helper_gf = 0.0;

  // Project the boundary conditions onto the solution space.
  for (const auto &bc : bc_.dirichlet_coefficients) {
    Array<int> marker(fespace_ptr_->GetParMesh()->bdr_attributes.Max());
    marker = 0; marker[bc.first] = 1;
    helper_gf.ProjectBdrCoefficient(*bc.second, marker);
  }

  // Create linear form for the right hand side.
  ParLinearForm b(fespace_ptr_);
  ConstantCoefficient zero(0.0);
  b.AddDomainIntegrator(new DomainLFIntegrator(zero));
  b.Assemble();

  k_.EliminateVDofsInRHS(ess_tdof_list_,helper_gf,b);

  // Solve the PDE for the lifting.
  Solve(b, helper_gf, 1.0, 1.0);

  // Add the lifting to the solution.
  x += helper_gf;
}

void SPDESolver::UpdateRHS(ParLinearForm &b) {
  if (!repeated_solve_) {
    // This function is only relevant for repeated solves.
    return;
  }
  if (restriction_matrix_) {
    // This effectively writes the solution of the previous iteration X_ to the
    // linear form b. Note that at the end of solve we update B_ = Mass * X_.
    restriction_matrix_->MultTranspose(B_, b);
  } else {
    b = B_;
  }
}

void SPDESolver::SolveLinearSystem() {
  HypreBoomerAMG prec(*Op_);
  prec.SetPrintLevel(-1);
  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(3);
  cg.SetPreconditioner(prec);
  cg.SetOperator(*Op_);
  cg.Mult(B_, X_);
}

void SPDESolver::ComputeRationalCoefficients(double exponent) {
  if (abs(exponent) > 1e-12) {
    if (Mpi::Root()) {
      mfem::out << "<SPDESolver> Approximating the fractional exponent "
                << exponent << std::endl;
    }
    ComputePartialFractionApproximation(exponent, coeffs_, poles_);

    // If the example is build without LAPACK, the exponent
    // might be modified by the function call above.
    alpha_ = exponent + integer_order_of_exponent_;
  } else {
    integer_order_ = true;
    if (Mpi::Root()) {
      mfem::out << "<SPDESolver> Treating integer order PDE." << std::endl;
    }
  }
}

} // namespace materials
} // namespace mfem
