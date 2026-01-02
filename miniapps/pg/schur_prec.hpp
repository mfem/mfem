#pragma once
#include "mfem.hpp"

namespace mfem
{
class SchurPrec : public Solver
{
   mutable int apply_count = 0;
   const BlockOperator * op=nullptr;
   Array<int> offsets;
   GSSmoother A00_prec;
   std::unique_ptr<SparseMatrix> BtDB;
   std::unique_ptr<SparseMatrix> negS;
   GSSmoother negS_prec;
   Vector diag_inv;
   void setup(bool full_update)
   {
      MFEM_VERIFY(op != nullptr, "SchurPrec::setup - operator not set");
      // out << "Setting up Schur complement preconditioner..." << std::endl;
      if (full_update)
      {
         const SparseMatrix &A00 = static_cast<const SparseMatrix&>(op->GetBlock(0,0));
         const SparseMatrix &A01 = static_cast<const SparseMatrix&>(op->GetBlock(0,1));

         A00.GetDiag(diag_inv);
         diag_inv.Reciprocal();

         BtDB.reset(Mult_AtDA(A01, diag_inv));
         BtDB->Finalize(0);

         A00_prec.SetOperator(A00);
      }
      if (!op->IsZeroBlock(1,1))
      {
         // out << "  Non-zero A11 block detected." << std::endl;
         const SparseMatrix *A11 =
            dynamic_cast<const SparseMatrix*>(&op->GetBlock(1, 1));
         MFEM_VERIFY(A11 != nullptr, "SchurPrec::setup - A11 is not a SparseMatrix");
         // out << "  Computing Schur complement S = -B^T diag(A)^{-1} B + A11" <<
         // std::endl;
         negS.reset(Add(-1.0, *A11, 1.0, *BtDB));
         negS->Finalize(0);
         // out << "  Schur complement matrix S = -B^T D^{-1} B + B11" << std::endl;
         negS_prec.SetOperator(*negS);
         // out << "  Schur complement preconditioner using S." << std::endl;
      }
      else
      {
         negS.reset(new SparseMatrix(*BtDB));
         negS->Finalize(0);
         negS_prec.SetOperator(*negS);
      }
      // out << "Schur complement preconditioner setup complete." << std::endl;
   }

public:
   SchurPrec()
   {
      A00_prec.iterative_mode=false;
      negS_prec.iterative_mode=false;
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      // out << "Applying Schur complement preconditioner..." << std::endl;
      MFEM_ASSERT(offsets.Last() == x.Size(), "SchurPrec::Mult - invalid input size");
      MFEM_ASSERT(offsets.Last() == y.Size(), "SchurPrec::Mult - invalid input size");
      BlockVector bx(const_cast<real_t*>(x.GetData()), offsets);
      BlockVector by(y, offsets);
      // out << "  Solving with A00_prec..." << std::endl;
      A00_prec.Mult(bx.GetBlock(0), by.GetBlock(0));
      // out << "  Solving with S_prec..." << std::endl;
      negS_prec.Mult(bx.GetBlock(1), by.GetBlock(1));
      by.GetBlock(1).Neg();
      apply_count++;
   }
   void Update(int block)
   {
      apply_count = 0;
      if (block == 1) { setup(false); }
      else { setup(true); }
   }
   int GetApplyCount() const { return apply_count; }
   void SetOperator(const Operator &new_op) override
   {
      // out << "Setting Schur complement preconditioner operator..." << std::endl;
      op = dynamic_cast<const BlockOperator*>(&new_op);
      MFEM_VERIFY(op != nullptr, "SchurPrec::SetOperator - not a BlockOperator");
      offsets = op->ColOffsets();
      setup(true);
      // out << "Schur complement preconditioner is ready." << std::endl;
   }
};
} // namespace mfem
