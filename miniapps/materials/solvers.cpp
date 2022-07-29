#include "solvers.hpp"
#include "rational_approximation.hpp"

namespace mfem {
namespace materials {

SPDESolver::SPDESolver(MatrixConstantCoefficient &diff_coefficient, double nu,
                       const Array<int> &ess_tdof_list,
                       ParFiniteElementSpace *fespace)
    : nu_(nu), ess_tdof_list_(ess_tdof_list), fespace_ptr_(fespace),
      k_(fespace), m_(fespace), restriction_matrix_(nullptr),
      prolongation_matrix_(nullptr), Op_(nullptr) {
  if (Mpi::Root()) {
    mfem::out << "<SPDESolver> Initialize Solver .." << std::endl;
  }
  StopWatch sw;
  sw.Start();

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
  k_.Assemble();

  // Assemble mass matrix
  ConstantCoefficient one(1.0);
  m_.AddDomainIntegrator(new MassIntegrator(one));
  m_.Assemble();

  // Form matrices for the linear system
  Array<int> empty;
  k_.FormSystemMatrix(ess_tdof_list_, stiffness_);
  m_.FormSystemMatrix(ess_tdof_list_, mass_bc_);
  m_.FormSystemMatrix(empty, mass_0_);

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

void SPDESolver::Solve(LinearForm &b, GridFunction &x) {
  // ------------------------------------------------------------------------
  // Solve the PDE (A)^N g = f, i.e. compute g = (A)^{-1}^N f iteratively.
  // ------------------------------------------------------------------------

  StopWatch sw;
  sw.Start();

  ParGridFunction helper_gf(fespace_ptr_);

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
  sw.Stop();
  if (Mpi::Root()) {
    mfem::out << "<SPDESolver::Timing> all PCG solves " << sw.RealTime()
              << " [s]" << std::endl;
  }
}

void SPDESolver::Solve(const LinearForm &b, GridFunction &x, double alpha,
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

  for (int i = 0; i < exponent; i++) {
    // Solve the linear system Op_ X_ = B_
    SolveLinearSystem();

    if (i == exponent - 1) {
      // Update x to contain the solution of the problem in the last step.
      k_.RecoverFEMSolution(X_, b, x);
    }
    if (repeated_solve_) {
      // Prepare for next iteration. X is a primal and B is a dual vector. B_
      // must be updated to represent X_ in the next step. Instead of copying
      // it, we must transform it appropriately with the mass matrix.
      mass_0_.Mult(X_, B_);
      X_.SetSubVectorComplement(ess_tdof_list_, 0.0);
    }
  }
}

void SPDESolver::UpdateRHS(LinearForm &b) {
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
