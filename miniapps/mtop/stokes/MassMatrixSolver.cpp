#include "MassMatrixSolver.hpp"

namespace mfem
{

// ---------------------------------------------------------------------------
// Constructor 1 – general Coefficient* (caller owns coeff)
// ---------------------------------------------------------------------------
MassMatrixSolver::MassMatrixSolver(ParFiniteElementSpace *pfes,
                                   Coefficient           *coeff,
                                   real_t                 rel_tol,
                                   int                    max_iter,
                                   int                    print_level)
   : owns_coeff_(false), owned_coeff_ptr_(nullptr)
{
   Init(pfes, coeff, rel_tol, max_iter, print_level);
}

// ---------------------------------------------------------------------------
// Constructor 2 – constant scalar coefficient (solver owns the coeff)
// ---------------------------------------------------------------------------
MassMatrixSolver::MassMatrixSolver(ParFiniteElementSpace *pfes,
                                   real_t                 coeff_val,
                                   real_t                 rel_tol,
                                   int                    max_iter,
                                   int                    print_level)
   : owns_coeff_(true),
     owned_coeff_ptr_(new ConstantCoefficient(coeff_val))
{
   Init(pfes, owned_coeff_ptr_, rel_tol, max_iter, print_level);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
MassMatrixSolver::~MassMatrixSolver()
{
   // Delete in reverse construction order.
   delete cg_;
   delete prec_;
   delete pblf_;                          // integrators (and their data) freed here
   if (owns_coeff_) delete owned_coeff_ptr_;
}

// ---------------------------------------------------------------------------
// Init – shared setup
// ---------------------------------------------------------------------------
void MassMatrixSolver::Init(ParFiniteElementSpace *pfes,
                            Coefficient           *coeff,
                            real_t                 rel_tol,
                            int                    max_iter,
                            int                    print_level)
{
   MFEM_VERIFY(pfes != nullptr, "MassMatrixSolver: pfes must not be null.");

   // ------------------------------------------------------------------
   // 1. Partially assembled mass bilinear form
   // ------------------------------------------------------------------
   pblf_ = new ParBilinearForm(pfes);
   pblf_->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   // MassIntegrator does NOT take ownership of the coefficient.
   if (coeff)
      pblf_->AddDomainIntegrator(new MassIntegrator(*coeff));
   else
      pblf_->AddDomainIntegrator(new MassIntegrator());

   pblf_->Assemble();

   // ------------------------------------------------------------------
   // 2. Form the true-dof system operator
   //    (empty essential-BC list – pure mass solve has no Dirichlet BCs)
   // ------------------------------------------------------------------
   Array<int> empty_ess;
   pblf_->FormSystemMatrix(empty_ess, mass_oper_);

   // ------------------------------------------------------------------
   // 3. Jacobi (diagonal) preconditioner from the PA diagonal
   // ------------------------------------------------------------------
   prec_ = new OperatorJacobiSmoother(*pblf_, empty_ess);

   // ------------------------------------------------------------------
   // 4. CG solver
   // ------------------------------------------------------------------
   cg_ = new CGSolver(pfes->GetComm());
   cg_->SetRelTol(rel_tol);
   cg_->SetMaxIter(max_iter);
   cg_->SetPrintLevel(print_level);
   cg_->SetOperator(*mass_oper_);
   cg_->SetPreconditioner(*prec_);
   cg_->iterative_mode = true; // honour x as initial guess
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------
void MassMatrixSolver::Solve(const Vector &b, Vector &x) const
{
   cg_->Mult(b, x);
   last_num_iter_  = cg_->GetNumIterations();
   last_final_res_ = cg_->GetFinalNorm();
   last_converged_ = cg_->GetConverged();
}

} // namespace mfem
