#include "solvers.hpp"

namespace mfem {
namespace materials {

PDESolver::PDESolver(MatrixConstantCoefficient& diff_coefficient,
                     const Array<int>& ess_tdof_list,
                     ParFiniteElementSpace* fespace)
  : ess_tdof_list_(ess_tdof_list), fespace_ptr_(fespace), k_(fespace), 
    m_(fespace) {
  StopWatch sw;
  if (Mpi::Root()){
    mfem::out << "Initialize Solver .." << std::endl;
  }
  sw.Start();
  
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
  if (prolongation_matrix_){
    B_.SetSize(prolongation_matrix_->Width());
  } else {
    mfem::err << "PDESolver::Solve: prolongation matrix is not defined" 
              << std::endl;
  }
  if (restriction_matrix_){
    X_.SetSize(restriction_matrix_->Height());
  } else {
    mfem::err << "PDESolver::Solve: restriction matrix is not defined" 
              << std::endl;
  }

  sw.Stop();
  if (Mpi::Root()){
    mfem::out << "Done. (" << sw.RealTime() << " sec)" << std::endl;
  }
}

void PDESolver::Solve(const LinearForm &b, GridFunction &x, 
                      double alpha, double beta, 
                      int exponent) {

  // Form system of equations. This is less general than 
  // BilinearForm::FormLinearSystem and kind of resembles the necessary subset 
  // of instructions that we need in this case.
  if (prolongation_matrix_){
    prolongation_matrix_->MultTranspose(b, B_);
  } else {
    B_ = b;
  }
  B_ *= beta;
  X_ = 0.0; // Initialize X_ to zero. Important! Might contain nan/inf -> crash.
  Op_ = Add(1.0, stiffness_, alpha, mass_bc_); //  construct Operator

  for (int i = 0; i < exponent; i++){
    // Solve the linear system Op_ X_ = B_
    SolveLinearSystem();

    if (i == exponent - 1)
    {
      // Update x to contain the solution of the problem in the last step.
      k_.RecoverFEMSolution(X_, b, x);
    } 
    if (repeated_solve_) {
      // Prepare for next iteration. X is a primal and B is a dual vector. B_ 
      // must be updated to represent X_ in the next step. Instead of copying
      // it, we must transform it appropriately with the mass matrix.
      mass_0_.Mult(X_, B_);
      X_.SetSubVectorComplement(ess_tdof_list_,0.0);
    }
  }

}

void PDESolver::UpdateRHS(LinearForm&b){
  if (!repeated_solve_){
    // This function is only relevant for repeated solves.
    return;
  }
  if (restriction_matrix_){
    // This effectively writes the solution of the previous iteration X_ to the 
    // linear form b. Note that at the end of solve we update B_ = Mass * X_.
    restriction_matrix_->MultTranspose(B_,b);
  } else {
      b = B_;
  }
}

void PDESolver::SolveLinearSystem(){
  HypreBoomerAMG prec(*Op_);
  prec.SetPrintLevel(-1);
  CGSolver cg(MPI_COMM_WORLD);
  prec.SetPrintLevel(-1);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(3);
  cg.SetPreconditioner(prec);
  cg.SetOperator(*Op_);
  cg.Mult(B_, X_);
}

} // namespace materials
} // namespace mfem
