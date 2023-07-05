#include "mfem.hpp"
#include "proximalGalerkin.hpp"
namespace mfem
{

void BlockLinearSystem::Assemble(BlockVector *x)
{
   Array<int> trial_ess_bdr;
   Array<int> test_ess_bdr;
   *b = 0.0;
   for (int row = 0; row < numSpaces; row++)
   {
      b_forms[row]->Update(spaces[row], b->GetBlock(row), 0);
      b_forms[row]->Assemble();
      test_ess_bdr.MakeRef(ess_bdr.GetRow(row), ess_bdr.NumCols());
      if (A_forms(row, row))
      {
         if (A_forms(row, row))
         {
            BilinearForm* bilf = this->GetDiagBlock(row);
            
            bilf->SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
            bilf->Assemble();
            bilf->EliminateEssentialBC(test_ess_bdr, x->GetBlock(row), b->GetBlock(row));
            bilf->Finalize();
            A->SetBlock(row, row, &(bilf->SpMat()));
         }
      }
      for (int col = 0; col < numSpaces; col++)
      {
         trial_ess_bdr.MakeRef(ess_bdr.GetRow(col), ess_bdr.NumCols());
         if (A_forms(row, col))
         {
            MixedBilinearForm* bilf = this->GetBlock(row, col);
            bilf->Assemble();
            bilf->EliminateTrialDofs(trial_ess_bdr, x->GetBlock(col), b->GetBlock(col));
            bilf->EliminateTestDofs(test_ess_bdr);
            bilf->Finalize();
            A->SetBlock(row, col, &(bilf->SpMat()));
         }
      }
   }
}

int BlockLinearSystem::GMRES(BlockVector *x)
{
   mfem::GMRES(*A, *prec, *b, *x, 0, 200, 50, 1e-12, 0.0);
}

} // end of namespace mfem