#include "schurlsc.hpp"
#include "_hypre_parcsr_ls.h"

namespace mfem {

SchurLSC::SchurLSC(BlockOperator *op)
   : Operator(op->GetBlock(1, 0).Height(), op->GetBlock(0, 1).Width()),
   op_(op)
{
   B_ = static_cast<HypreParMatrix *>(&op->GetBlock(0, 1));
   C_ = static_cast<HypreParMatrix *>(&op->GetBlock(1, 0));
   CB_ = ParMult(C_, B_);
   amgCB_ = new HypreBoomerAMG(*CB_);
   amgCB_->SetPrintLevel(0);

   HYPRE_Solver amg_precond = static_cast<HYPRE_Solver>(*amgCB_);
   HYPRE_BoomerAMGSetCoarsenType(amg_precond, 6);
   HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
   HYPRE_BoomerAMGSetRelaxType(amg_precond, 6);
   HYPRE_BoomerAMGSetInterpType(amg_precond, 0);
   HYPRE_BoomerAMGSetPMaxElmts(amg_precond, 0);

   x0.SetSize(B_->Height());
   y0.SetSize(B_->Height());
   x1.SetSize(C_->Height());
}

void SchurLSC::Mult(const Vector &x, Vector &y) const
{
   amgCB_->Mult(x, x1);
   B_->Mult(x1, x0);
   op_->GetBlock(0, 0).Mult(x0, y0);
   C_->Mult(y0, x1);
   amgCB_->Mult(x1, y);
}

SchurLSC::~SchurLSC()
{
   delete CB_;
   delete amgCB_;
}

} // namespace mfem
