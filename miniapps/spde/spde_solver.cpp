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

#include <ctime>

#include "examples/ex33.hpp"
#include "spde_solver.hpp"

namespace mfem {
namespace spde {

SPDESolver::SPDESolver(double nu, const Boundary &bc,
                       ParFiniteElementSpace *fespace, double l1, double l2,
                       double l3, double e1, double e2, double e3)
    : nu_(nu),
      bc_(bc),
      fespace_ptr_(fespace),
      k_(fespace),
      m_(fespace),
      restriction_matrix_(nullptr),
      prolongation_matrix_(nullptr),
      Op_(nullptr),
      l1_(l1),
      l2_(l2),
      l3_(l3),
      e1_(e1),
      e2_(e2),
      e3_(e3) {
  if (Mpi::Root()) {
    mfem::out << "<SPDESolver> Initialize Solver .." << std::endl;
  }
  StopWatch sw;
  sw.Start();

  // ToDo: "_" missing; nbc not filled nor used.
  // Resize the marker arrays for the boundary conditions
  dbc_marker_.SetSize(fespace_ptr_->GetParMesh()->bdr_attributes.Max());
  rbc_marker_.SetSize(fespace_ptr_->GetParMesh()->bdr_attributes.Max());
  dbc_marker_ = 0;
  rbc_marker_ = 0;

  // Fill the marker arrays for the boundary conditions. We decrement the number
  // it.first by one because the boundary attributes in the mesh start at 1 and
  // the marker arrays start at 0.
  for (const auto &it : bc_.boundary_attributes) {
    switch (it.second) {
      case BoundaryType::kDirichlet:
        dbc_marker_[it.first - 1] = 1;
        break;
      case BoundaryType::kRobin:
        rbc_marker_[it.first - 1] = 1;
        break;
      default:
        break;
    }
  }

  // Handle homogeneous Dirichlet boundary conditions
  // Note: for non zero DBC we usually need to project the boundary onto the
  // solution. This is not necessary in this case since the boundary is
  // homogeneous. For inhomogeneous Dirichlet we consider a lifting scheme.
  fespace_ptr_->GetEssentialTrueDofs(dbc_marker_, ess_tdof_list_);

  // Compute the rational approximation coefficients.
  int dim = fespace_ptr_->GetParMesh()->Dimension();
  alpha_ = (nu_ + dim / 2.0) / 2.0;  // fractional exponent
  integer_order_of_exponent_ = std::floor(alpha_);
  double exponent_to_approximate = alpha_ - integer_order_of_exponent_;

  // Compute the rational approximation coefficients.
  ComputeRationalCoefficients(exponent_to_approximate);

  // Set the bilinear forms.

  // Assemble stiffness matrix
  auto diffusion_tensor =
      ConstructMatrixCoefficient(l1_, l2_, l3_, e1_, e2_, e3_, nu_,
                                 fespace_ptr_->GetParMesh()->Dimension());
  MatrixConstantCoefficient diffusion_coefficient(diffusion_tensor);
  k_.AddDomainIntegrator(new DiffusionIntegrator(diffusion_coefficient));
  ConstantCoefficient robin_coefficient(bc_.robin_coefficient);
  k_.AddBoundaryIntegrator(new MassIntegrator(robin_coefficient), rbc_marker_);
  k_.Assemble(0);

  // Assemble mass matrix
  ConstantCoefficient one(1.0);
  m_.AddDomainIntegrator(new MassIntegrator(one));
  m_.Assemble(0);

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

  // Zero initialize x to avoid touching uninitialized memory
  x = 0.0;

  ParGridFunction helper_gf(fespace_ptr_);
  helper_gf = 0.0;

  if (integer_order_of_exponent_ > 0) {
    // if (Mpi::Root()) {
    //   mfem::out << "<SPDESolver> Solving PDE (A)^" << integer_order_of_exponent_
    //             << " u = f" << std::endl;
    // }
    ActivateRepeatedSolve();
    Solve(b, helper_gf, 1.0, 1.0, integer_order_of_exponent_);
    if (integer_order_) {
      // If the exponent is an integer, we can directly add the solution to the
      // final solution and return.
      x += helper_gf;
      if (!bc_.dirichlet_coefficients.empty()) {
        LiftSolution(x);
      }
      return;
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

void SPDESolver::SetupRandomFieldGenerator(int seed) {
  delete b;
  delete integ;
  integ = new WhiteGaussianNoiseDomainLFIntegrator(seed);
  b = new ParLinearForm(fespace_ptr_);
  b->AddDomainIntegrator(integ);
};

void SPDESolver::GenerateRandomField(ParGridFunction &x) {
  if (!integ) MFEM_ABORT("Need to call SPDESolver::SetupRandomFieldGenerator(...) first");
  // Create the stochastic load
  b->Assemble();
  double normalization = ConstructNormalizationCoefficient(
      nu_, l1_, l2_, l3_, fespace_ptr_->GetParMesh()->Dimension());
  (*b) *= normalization;

  // Call back to solve to generate the random field
  Solve(*b, x);
};

double SPDESolver::ConstructNormalizationCoefficient(double nu, double l1,
                                                     double l2, double l3,
                                                     int dim) {
  // Computation considers squaring components, computing determinant, and
  // squaring
  double det = 0;
  if (dim == 1) {
    det = l1;
  } else if (dim == 2) {
    det = l1 * l2;
  } else if (dim == 3) {
    det = l1 * l2 * l3;
  }
  const double gamma1 = tgamma(nu + static_cast<double>(dim) / 2.0);
  const double gamma2 = tgamma(nu);
  return sqrt(pow(2 * M_PI, dim / 2.0) * det * gamma1 /
              (gamma2 * pow(nu, dim / 2.0)));
}

DenseMatrix SPDESolver::ConstructMatrixCoefficient(double l1, double l2,
                                                   double l3, double e1,
                                                   double e2, double e3,
                                                   double nu, int dim) {
  if (dim == 3) {
    // Compute cosine and sine of the angles e1, e2, e3
    const double c1 = cos(e1);
    const double s1 = sin(e1);
    const double c2 = cos(e2);
    const double s2 = sin(e2);
    const double c3 = cos(e3);
    const double s3 = sin(e3);

    // Fill the rotation matrix R with the Euler angles.
    DenseMatrix R(3, 3);
    R(0, 0) = c1 * c3 - c2 * s1 * s3;
    R(0, 1) = -c1 * s3 - c2 * c3 * s1;
    R(0, 2) = s1 * s2;
    R(1, 0) = c3 * s1 + c1 * c2 * s3;
    R(1, 1) = c1 * c2 * c3 - s1 * s3;
    R(1, 2) = -c1 * s2;
    R(2, 0) = s2 * s3;
    R(2, 1) = c3 * s2;
    R(2, 2) = c2;

    // Multiply the rotation matrix R with the translation vector.
    Vector l(3);
    l(0) = std::pow(l1, 2);
    l(1) = std::pow(l2, 2);
    l(2) = std::pow(l3, 2);
    l *= (1 / (2.0 * nu));

    // Compute result = R^t diag(l) R
    DenseMatrix res(3, 3);
    R.Transpose();
    MultADBt(R, l, R, res);
    return res;
  } else if (dim == 2) {
    DenseMatrix res(2, 2);
    res(0, 0) = std::pow(l1, 2) / (2.0 * nu);
    res(1, 1) = std::pow(l2, 2) / (2.0 * nu);
    res(0, 1) = 0;
    res(1, 0) = 0;
    return res;
  } else {
    DenseMatrix res(1, 1);
    res(0, 0) = std::pow(l1, 2) / (2.0 * nu);
    return res;
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

  if (!apply_lift_) {
    // Initialize X_ to zero. Important! Might contain nan/inf -> crash.
    X_ = 0.0;
  } else {
    restriction_matrix_->Mult(x, X_);
  }

  delete Op_;
  Op_ = Add(1.0, stiffness_, alpha, mass_bc_);  //  construct Operator
  HypreParMatrix *Ae = Op_->EliminateRowsCols(ess_tdof_list_);
  Op_->EliminateBC(*Ae, ess_tdof_list_, X_, B_);  // only for homogeneous BC

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

void SPDESolver::LiftSolution(ParGridFunction &x) {
  // Set lifting flag
  apply_lift_ = true;

  // Lifting of the solution takes care of inhomogeneous boundary conditions.
  // See doi:10.1016/j.jcp.2019.109009; section 2.6
  // if (Mpi::Root()) {
  //   mfem::out << "\n<SPDESolver> Applying inhomogeneous DBC" << std::endl;
  // }

  // Define temporary grid function for lifting.
  ParGridFunction helper_gf(fespace_ptr_);
  helper_gf = 0.0;

  // Project the boundary conditions onto the solution space.
  for (const auto &bc : bc_.dirichlet_coefficients) {
    Array<int> marker(fespace_ptr_->GetParMesh()->bdr_attributes.Max());
    marker = 0;
    marker[bc.first - 1] = 1;
    ConstantCoefficient cc(bc.second);
    helper_gf.ProjectBdrCoefficient(cc, marker);
  }

  // Create linear form for the right hand side.
  ParLinearForm b(fespace_ptr_);
  ConstantCoefficient zero(0.0);
  b.AddDomainIntegrator(new DomainLFIntegrator(zero));
  b.Assemble();

  // Solve the PDE for the lifting.
  Solve(b, helper_gf, 1.0, 1.0);

  // Add the lifting to the solution.
  x += helper_gf;

  // Reset the lifting flag.
  apply_lift_ = false;
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
  cg.SetPrintLevel(0);
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

SPDESolver::~SPDESolver() {   
  delete Op_; 
  delete b;
  delete integ;
};

}  // namespace spde
}  // namespace mfem
