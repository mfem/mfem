#include "mfem.hpp"

using namespace mfem;

void reconstructH1Field(const ParGridFunction &src, ParGridFunction &dst)
{
   Vector dst_dofs;
   dst.GetTrueDofs(dst_dofs);

   OperatorPtr M_op;
   ParBilinearForm M(dst.ParFESpace());
   M.AddDomainIntegrator(new MassIntegrator());
   M.Assemble();
   M.FormSystemMatrix({}, M_op);

   OperatorPtr Rhs_op;
   Vector src_dofs;
   Vector rhs(dst_dofs.Size());
   ParMixedBilinearForm Rhs(src.ParFESpace(), dst.ParFESpace());
   Rhs.AddDomainIntegrator(new MassIntegrator());
   Rhs.Assemble();
   Rhs.FormRectangularSystemMatrix({}, {}, Rhs_op);
   src.GetTrueDofs(src_dofs);
   Rhs_op->Mult(src_dofs, rhs);

   CGSolver solver(dst.ParFESpace()->GetComm());
   solver.iterative_mode = false;
   solver.SetRelTol(1e-5);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(30);
   solver.SetPrintLevel(0);
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi);
   solver.SetPreconditioner(prec);
   solver.SetOperator(*M_op);
   solver.Mult(rhs, dst_dofs);
   MFEM_VERIFY(solver.GetConverged(), "Solver did not converge.");
   dst.SetFromTrueDofs(dst_dofs);
}
