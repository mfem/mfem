#include "mfem.hpp"

namespace mfem {
namespace materials {

// ===========================================================================
// Header interface
// ===========================================================================

/// PDE solver for equations of type (div \Theta grad + \nu I) u = f.
class PDESolver {
 public:
  /// Constructor. PDE solver for equations of type 
  /// (div \Theta grad + c I) u = f.
  /// @param c The coefficient c.
  /// @param diff_coefficient The diffusion coefficient \Theta.
  /// @param ess_tdof_list Boundary conditions.
  /// @param fespace Finite element space.
  /// @param integer_exp_of_operator Apply solver N times.
  PDESolver(double c, 
            MatrixConstantCoefficient& diff_coefficient,
            const Array<int>& ess_tdof_list,
            ParFiniteElementSpace* fespace,
            int integer_exp_of_operator = 1);
  void Solve(Vector &b, Vector &x);
 
 private:
  ParBilinearForm k_;
  HypreParMatrix mass_;
  Vector X_;
  Vector B_;
  OperatorPtr Op_;
  const Array<int>& ess_tdof_list_;
  ParFiniteElementSpace* fespace_ptr_;
  int integer_exp_of_operator_;
};


// ===========================================================================
// Implementation details
// ===========================================================================

PDESolver::PDESolver(double c, 
                                       MatrixConstantCoefficient& diff_coefficient,
                                       const Array<int>& ess_tdof_list,
                                       ParFiniteElementSpace* fespace,
                                       int integer_exp_of_operator)
  : integer_exp_of_operator_(integer_exp_of_operator), 
    ess_tdof_list_(ess_tdof_list), fespace_ptr_(fespace), k_(fespace) {
  // Consruct the System Matrix
  ConstantCoefficient cc(c);
  k_.AddDomainIntegrator(new DiffusionIntegrator(diff_coefficient));
  k_.AddDomainIntegrator(new MassIntegrator(cc));
  k_.Assemble();
}

void PDESolver::Solve(Vector &b, Vector &x){

  bool repeated_solve = (integer_exp_of_operator_ > 1);

  if (repeated_solve){
    // Construct the mass matrix
    ParBilinearForm m(fespace_ptr_);
    ConstantCoefficient one(1.0);
    m.AddDomainIntegrator(new MassIntegrator(one));
    m.Assemble();
    Array<int> empty;
    m.FormSystemMatrix(empty, mass_);
  }

  // Form system of equations
  k_.FormLinearSystem(ess_tdof_list_, x, b, Op_, X_, B_);
  HypreBoomerAMG prec;
  prec.SetPrintLevel(-1);
  CGSolver cg(MPI_COMM_WORLD);
  prec.SetPrintLevel(-1);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(3);
  cg.SetPreconditioner(prec);
  cg.SetOperator(*Op_);

  for (int i = 0; i < integer_exp_of_operator_; i++){
    // Solve the linear system A X = B (integer_exp_of_operator_ times).
    cg.Mult(B_, X_);
    // Visualize the solution g of -Δ ^ N g = f in the last step
    if (i == integer_exp_of_operator_ - 1)
    {
      // Needed for visualization and solution verification.
      k_.RecoverFEMSolution(X_, b, x);
    } 
    if (repeated_solve) {
      // Prepare for next iteration (primal / dual space). Unne
      mass_.Mult(X_, B_);
      X_.SetSubVectorComplement(ess_tdof_list_,0.0);
    }
  }

  if (repeated_solve) {
    // Extract solution for the next step. The b now corresponds to the
    //      function g in the PDE.
    const SparseMatrix* rm = fespace_ptr_->GetRestrictionMatrix();
    rm->MultTranspose(B_, b);
  }

}

} // namespace materials
} // namespace mfem
