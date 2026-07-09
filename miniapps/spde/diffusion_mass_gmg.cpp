#include "diffusion_mass_gmg.hpp"
#include "frac_noise.hpp"

#include <algorithm>
#include <memory>

namespace mfem
{
namespace
{

HypreParMatrix &RequireHypreParMatrix(OperatorHandle &handle)
{
   MFEM_VERIFY(handle.Ptr() != nullptr, "Coarse AMG matrix is null.");
   HypreParMatrix *matrix = handle.Is<HypreParMatrix>();
   MFEM_VERIFY(matrix != nullptr, "Coarse AMG matrix is not a HypreParMatrix.");
   return *matrix;
}

class OwnedSymmetrizedSmoother : public SymmetrizedSmoother
{
public:
   OwnedSymmetrizedSmoother(Solver *smoother, Operator *op)
      : SymmetrizedSmoother(smoother, op), owned_smoother_(smoother)
   { }

private:
   std::unique_ptr<Solver> owned_smoother_;
};

class OwnedBoomerAMGCoarseSolver : public Solver
{
public:
   OwnedBoomerAMGCoarseSolver(ParBilinearForm *form,
                              OperatorHandle *matrix_handle)
      : Solver(RequireHypreParMatrix(*matrix_handle).Height(),
               RequireHypreParMatrix(*matrix_handle).Width()),
        form_(form),
        matrix_handle_(matrix_handle),
        matrix_(&RequireHypreParMatrix(*matrix_handle)),
        amg_(new HypreBoomerAMG),
        cg_(new CGSolver(matrix_->GetComm()))
   {
      amg_->SetPrintLevel(0);
      amg_->SetRelaxType(6); // symmetric Gauss-Seidel
      amg_->SetOperator(*matrix_);
      cg_->SetRelTol(1e-12);
      cg_->SetAbsTol(0.0);
      cg_->SetMaxIter(std::max(50, matrix_->Height()));
      cg_->SetPrintLevel(-1);
      cg_->SetPreconditioner(*amg_);
      cg_->SetOperator(*matrix_);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      cg_->Mult(x, y);
   }

   void SetOperator(const Operator &op) override
   {
      MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
                  "Coarse solver operator size mismatch.");
   }

private:
   std::unique_ptr<ParBilinearForm> form_;
   std::unique_ptr<OperatorHandle> matrix_handle_;
   HypreParMatrix *matrix_;
   std::unique_ptr<HypreBoomerAMG> amg_;
   std::unique_ptr<CGSolver> cg_;
};

} // namespace

DiffusionMassGeometricMultigrid::DiffusionMassGeometricMultigrid(
   ParFiniteElementSpaceHierarchy &hierarchy,
   const Array<int> &ess_bdr,
   Coefficient &diffusion_coefficient,
   Coefficient &mass_coefficient,
   real_t jacobi_damping,
   CoarseSolver coarse_solver)
   : GeometricMultigrid(hierarchy, ess_bdr),
     hierarchy_(hierarchy),
     diffusion_coefficient_(&diffusion_coefficient),
     mass_coefficient_(&mass_coefficient),
     jacobi_damping_(jacobi_damping),
     coarse_solver_(coarse_solver),
     level_operator_owner_(true)
{
   Rebuild();
}

void DiffusionMassGeometricMultigrid::SetDiffusionCoefficient(
   Coefficient &coefficient)
{
   diffusion_coefficient_ = &coefficient;
   Rebuild();
}

void DiffusionMassGeometricMultigrid::SetMassCoefficient(
   Coefficient &coefficient)
{
   mass_coefficient_ = &coefficient;
   Rebuild();
}

ParFiniteElementSpaceHierarchy &
DiffusionMassGeometricMultigrid::GetHierarchy()
{
   return hierarchy_;
}

const ParFiniteElementSpaceHierarchy &
DiffusionMassGeometricMultigrid::GetHierarchy() const
{
   return hierarchy_;
}

ParBilinearForm &
DiffusionMassGeometricMultigrid::GetLevelBilinearForm(int level)
{
   MFEM_VERIFY(level >= 0 && level < bfs.Size(), "Invalid multigrid level.");
   return *static_cast<ParBilinearForm *>(bfs[level]);
}

const ParBilinearForm &
DiffusionMassGeometricMultigrid::GetLevelBilinearForm(int level) const
{
   MFEM_VERIFY(level >= 0 && level < bfs.Size(), "Invalid multigrid level.");
   return *static_cast<const ParBilinearForm *>(bfs[level]);
}

const Array<int> &
DiffusionMassGeometricMultigrid::GetEssentialTrueDofsAtLevel(int level) const
{
   return LevelEssentialTrueDofs(level);
}

void DiffusionMassGeometricMultigrid::Rebuild()
{
   ClearLevelData();

   const int nlevels = hierarchy_.GetNumLevels();
   MFEM_VERIFY(nlevels > 0, "Expected at least one multigrid level.");

   for (int level = 0; level < nlevels; level++)
   {
      ConstructLevelOperatorAndSmoother(hierarchy_.GetFESpaceAtLevel(level),
                                        level);
   }
}

void DiffusionMassGeometricMultigrid::SetCoarseSolver(
   CoarseSolver coarse_solver)
{
   coarse_solver_ = coarse_solver;
   Rebuild();
}

void DiffusionMassGeometricMultigrid::SetOperator(const Operator &op)
{
   MFEM_VERIFY(operators.Size() > 0, "The multigrid hierarchy is empty.");
   MFEM_VERIFY(op.Height() == Height() && op.Width() == Width(),
               "The supplied operator does not match the finest GMG level.");
}

void DiffusionMassGeometricMultigrid::FormFineLinearSystem(
   Vector &x, Vector &b, OperatorHandle &A, Vector &X, Vector &B)
{
   GetLevelBilinearForm(bfs.Size() - 1).FormLinearSystem(
      LevelEssentialTrueDofs(bfs.Size() - 1), x, b, A, X, B);
}

void DiffusionMassGeometricMultigrid::RecoverFineFEMSolution(
   const Vector &X, const Vector &b, Vector &x)
{
   GetLevelBilinearForm(bfs.Size() - 1).RecoverFEMSolution(X, b, x);
}

const Array<int> &
DiffusionMassGeometricMultigrid::LevelEssentialTrueDofs(int level) const
{
   if (essentialTrueDofs.Size() == 0)
   {
      return empty_ess_tdofs_;
   }
   MFEM_VERIFY(level >= 0 && level < essentialTrueDofs.Size(),
               "Invalid multigrid level.");
   return *essentialTrueDofs[level];
}

void DiffusionMassGeometricMultigrid::ClearLevelData()
{
   for (int i = 0; i < operators.Size(); i++)
   {
      if (ownedOperators[i])
      {
         delete operators[i];
      }
      if (ownedSmoothers[i])
      {
         delete smoothers[i];
      }
   }

   operators.SetSize(0);
   smoothers.SetSize(0);
   ownedOperators.SetSize(0);
   ownedSmoothers.SetSize(0);

   for (int i = 0; i < bfs.Size(); i++)
   {
      delete bfs[i];
   }
   bfs.SetSize(0);

   for (int i = 0; i < X.NumRows(); i++)
   {
      for (int j = 0; j < X.NumCols(); j++)
      {
         delete X(i, j);
         delete Y(i, j);
         delete R(i, j);
         delete Z(i, j);
      }
   }
   X.SetSize(0, 0);
   Y.SetSize(0, 0);
   R.SetSize(0, 0);
   Z.SetSize(0, 0);
}

void DiffusionMassGeometricMultigrid::ConstructLevelOperatorAndSmoother(
   ParFiniteElementSpace &fespace, int level)
{
   ParBilinearForm *form = new ParBilinearForm(&fespace);
   form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form->AddDomainIntegrator(new DiffusionIntegrator(*diffusion_coefficient_));
   form->AddDomainIntegrator(new MassIntegrator(*mass_coefficient_));
   form->Assemble();
   bfs.Append(form);

   OperatorPtr level_operator;
   level_operator.SetType(Operator::ANY_TYPE);
   form->FormSystemMatrix(LevelEssentialTrueDofs(level), level_operator);
   level_operator.SetOperatorOwner(false);

   Solver *smoother = nullptr;
   if (level == 0 && coarse_solver_ == CoarseSolver::BoomerAMG)
   {
      smoother = ConstructBoomerAMGCoarseSolver(fespace,
                                                LevelEssentialTrueDofs(level));
   }
   else
   {
      smoother = ConstructSymmetrizedJacobiSmoother(
         *form, *level_operator.Ptr(), LevelEssentialTrueDofs(level));
   }

   AddLevel(level_operator.Ptr(), smoother, level_operator_owner_, true);
}

Solver *DiffusionMassGeometricMultigrid::ConstructSymmetrizedJacobiSmoother(
   ParBilinearForm &form, Operator &op, const Array<int> &ess_tdofs)
{
   Solver *jacobi = new OperatorJacobiSmoother(form, ess_tdofs,
                                               jacobi_damping_);
   return new OwnedSymmetrizedSmoother(jacobi, &op);
}

Solver *DiffusionMassGeometricMultigrid::ConstructBoomerAMGCoarseSolver(
   ParFiniteElementSpace &fespace, const Array<int> &ess_tdofs)
{
   ParBilinearForm *coarse_form = new ParBilinearForm(&fespace);
   coarse_form->AddDomainIntegrator(
      new DiffusionIntegrator(*diffusion_coefficient_));
   coarse_form->AddDomainIntegrator(new MassIntegrator(*mass_coefficient_));
   coarse_form->Assemble();

   OperatorHandle *coarse_matrix_handle = new OperatorHandle;
   coarse_matrix_handle->SetType(Operator::Hypre_ParCSR);
   coarse_form->FormSystemMatrix(ess_tdofs, *coarse_matrix_handle);
   RequireHypreParMatrix(*coarse_matrix_handle);
   return new OwnedBoomerAMGCoarseSolver(coarse_form, coarse_matrix_handle);
}

} // namespace mfem
