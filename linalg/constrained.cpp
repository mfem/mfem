// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "constrained.hpp"

#include <set>

namespace mfem
{

/// convenience function, convert a SparseMatrix to a (serial)
/// HypreParMatrix so you can use hypre solvers
HypreParMatrix* SerialHypreMatrix(SparseMatrix& mat, bool transfer_ownership=true)
{
   HYPRE_Int row_starts[3];
   row_starts[0] = 0;
   row_starts[1] = mat.Height();
   row_starts[2] = mat.Height();
   HYPRE_Int col_starts[3];
   col_starts[0] = 0;
   col_starts[1] = mat.Width();
   col_starts[2] = mat.Width();
   HypreParMatrix * out = new HypreParMatrix(
      MPI_COMM_WORLD, mat.Height(), mat.Width(),
      row_starts, col_starts, &mat);
   out->CopyRowStarts();
   out->CopyColStarts();

   /// 3 gives MFEM full ownership of i, j, data
   if (transfer_ownership)
   {
      out->SetOwnerFlags(3, out->OwnsOffd(), out->OwnsColMap());
      mat.LoseData();
   }
   else
   {
      out->SetOwnerFlags(0, out->OwnsOffd(), out->OwnsColMap());
   }

   return out;
}

EliminationProjection::EliminationProjection(SparseMatrix& A, SparseMatrix& B,
                                             Array<int>& master_contact_dofs,
                                             Array<int>& slave_contact_dofs)
   :
   Operator(A.Height(),
            A.Height() - slave_contact_dofs.Size()),
   A_(A),
   B_(B),
   master_contact_dofs_(master_contact_dofs),
   slave_contact_dofs_(slave_contact_dofs)
{
   Array<int> lm_dofs;
   for (int i = 0; i < B.Height(); ++i)
   {
      lm_dofs.Append(i);
   }
   Bm_.SetSize(B_.Height(), master_contact_dofs.Size());
   B_.GetSubMatrix(lm_dofs, master_contact_dofs, Bm_);

   Bs_.SetSize(B_.Height(), slave_contact_dofs.Size());
   B_.GetSubMatrix(lm_dofs, slave_contact_dofs, Bs_);
   BsT_.Transpose(Bs_);

   ipiv_.SetSize(Bs_.Height());
   Bsinverse_.data = Bs_.GetData();
   Bsinverse_.ipiv = ipiv_.GetData();
   Bsinverse_.Factor(Bs_.Height());

   ipivT_.SetSize(Bs_.Height());
   BsTinverse_.data = BsT_.GetData();
   BsTinverse_.ipiv = ipivT_.GetData();
   BsTinverse_.Factor(Bs_.Height());
}

/*
   Return this projector as an assembled matrix.

   It *may* be possible to implement this with Bm_ as a findpts call
   rather than a matrix; the hypre assembly will not be so great, though
*/
SparseMatrix * EliminationProjection::AssembleExact() const
{
   int num_elim_dofs = slave_contact_dofs_.Size();

   SparseMatrix * out = new SparseMatrix(
      A_.Height(), A_.Height() - num_elim_dofs);

   int column_dof = 0;
   Array<int> mapped_master_contact_dofs;
   for (int i = 0; i < A_.Height(); ++i)
   {
      if (slave_contact_dofs_.FindSorted(i) >= 0)
      {
         // if dof is a slave, it doesn't show up in reduced system
         ;
      }
      else
      {
         // otherwise, dof exists in reduced system (identity)
         out->Set(i, column_dof, 1.0);
         if (master_contact_dofs_.FindSorted(i) >= 0)
         {
            // in addition, mapped_master_contact_dofs[reduced_id] = larger_id
            mapped_master_contact_dofs.Append(column_dof);
         }
         column_dof++;
      }
   }
   MFEM_ASSERT(mapped_master_contact_dofs.Size() == master_contact_dofs_.Size(),
               "Unable to map master contact dofs!");

   DenseMatrix block(Bm_);
   std::cout << "        inverting matrix of size " << Bs_.Height() << std::endl;
   Bsinverse_.Solve(Bs_.Height(), Bm_.Width(), block.GetData());
   std::cout << "        ...done." << std::endl;

   for (int iz = 0; iz < slave_contact_dofs_.Size(); ++iz)
   {
      int i = slave_contact_dofs_[iz];
      for (int jz = 0; jz < mapped_master_contact_dofs.Size(); ++jz)
      {
         int j = mapped_master_contact_dofs[jz];
         out->Add(i, j, -block(iz, jz));
      }
   }
   out->Finalize();
   return out;
}

void EliminationProjection::Mult(const Vector& in, Vector& out) const
{
   int num_elim_dofs = slave_contact_dofs_.Size();
   MFEM_ASSERT(in.Size() == A_.Height() - num_elim_dofs, "Sizes don't match!");
   MFEM_ASSERT(out.Size() == A_.Height(), "Sizes don't match!");

   out = 0.0;
   int column_dof = 0;
   Array<int> mapped_master_contact_dofs;
   for (int i = 0; i < A_.Height(); ++i)
   {
      if (slave_contact_dofs_.FindSorted(i) >= 0)
      {
         // if dof is a slave, it doesn't show up in reduced system
         ;
      }
      else
      {
         // otherwise, dof exists in reduced system (identity)
         out(i) += in(column_dof);
         if (master_contact_dofs_.FindSorted(i) >= 0)
         {
            // in addition, mapped_master_contact_dofs[reduced_id] = larger_id
            mapped_master_contact_dofs.Append(column_dof);
         }
         column_dof++;
      }
   }
   MFEM_ASSERT(mapped_master_contact_dofs.Size() == master_contact_dofs_.Size(),
               "Unable to map master contact dofs!");

   Vector subvecin;
   Vector subvecout(slave_contact_dofs_.Size());
   in.GetSubVector(mapped_master_contact_dofs, subvecin);
   Bm_.Mult(subvecin, subvecout);
   Bsinverse_.Solve(Bs_.Height(), 1, subvecout);
   subvecout *= -1.0;
   out.AddElementVector(slave_contact_dofs_, subvecout);
}

void EliminationProjection::MultTranspose(const Vector& in, Vector& out) const
{
   int num_elim_dofs = slave_contact_dofs_.Size();
   MFEM_ASSERT(out.Size() == A_.Height() - num_elim_dofs, "Sizes don't match!");
   MFEM_ASSERT(in.Size() == A_.Height(), "Sizes don't match!");

   out = 0.0;
   int row_dof = 0;
   Array<int> mapped_master_contact_dofs;
   for (int i = 0; i < A_.Height(); ++i)
   {
      if (slave_contact_dofs_.FindSorted(i) >= 0)
      {
         ;
      }
      else
      {
         out(row_dof) += in(i);
         if (master_contact_dofs_.FindSorted(i) >= 0)
         {
            mapped_master_contact_dofs.Append(row_dof);
         }
         row_dof++;
      }
   }
   MFEM_ASSERT(mapped_master_contact_dofs.Size() == master_contact_dofs_.Size(),
               "Unable to map master contact dofs!");

   Vector subvecin;
   Vector subvecout(Bm_.Width());

   in.GetSubVector(slave_contact_dofs_, subvecin);
   BsTinverse_.Solve(Bs_.Height(), 1, subvecin);
   Bm_.MultTranspose(subvecin, subvecout);
   subvecout *= -1.0;
   out.AddElementVector(mapped_master_contact_dofs, subvecout);
}

void EliminationProjection::BuildGTilde(const Vector& g, Vector& gtilde) const
{
   // int num_elim_dofs = slave_contact_dofs_.Size();
   MFEM_ASSERT(g.Size() == B_.Height(), "Sizes don't match!");
   MFEM_ASSERT(gtilde.Size() == A_.Height(), "Sizes don't match!");

   gtilde = 0.0;
   Vector cinvg(g);
   Bsinverse_.Solve(Bs_.Height(), 1, cinvg);
   gtilde.AddElementVector(slave_contact_dofs_, cinvg);
}

void EliminationProjection::RecoverPressure(const Vector& disprhs, const Vector& disp,
                                            Vector& pressure) const
{
   MFEM_ASSERT(pressure.Size() == B_.Height(), "Sizes don't match!");
   MFEM_ASSERT(disp.Size() == A_.Height(), "Sizes don't match!");

   Vector fullrhs(A_.Height());
   A_.Mult(disp, fullrhs);
   fullrhs -= disprhs;
   fullrhs *= -1.0;
   fullrhs.GetSubVector(slave_contact_dofs_, pressure);
   BsTinverse_.Solve(Bs_.Height(), 1, pressure);
}



EliminationCGSolver::~EliminationCGSolver()
{
   delete h_explicit_operator_;
   delete projector_;
   delete prec_;
}

void EliminationCGSolver::BuildSeparatedInterfaceDofs(int firstblocksize)
{
   std::set<int> first_interface;
   std::set<int> second_interface;
   int * I = B_.GetI();
   int * J = B_.GetJ();
   double * data = B_.GetData();
   const double tol = 1.e-14;
   for (int i = 0; i < B_.Height(); ++i)
   {
      for (int jidx = I[i]; jidx < I[i + 1]; ++jidx)
      {
         int j = J[jidx];
         double v = data[jidx];
         if (fabs(v) < tol)
         {
            continue;
         }
         if (j < firstblocksize)
         {
            first_interface.insert(j);
         }
         else
         {
            second_interface.insert(j);
         }
      }
   }

   for (auto i : first_interface)
   {
      first_interface_dofs_.Append(i);
   }
   first_interface_dofs_.Sort();
   int second_interface_size = 0;
   for (auto i : second_interface)
   {
      second_interface_size++;
      second_interface_dofs_.Append(i);
   }
   second_interface_dofs_.Sort();

   if (second_interface_size != B_.Height())
   {
      // is this really expected? equating nodes with dofs in some weird way on a manifold?
      std::cerr << "B_.Height() = " << B_.Height()
                << ", slave_interface_size = " << second_interface_size << std::endl;
      MFEM_VERIFY(false, "I don't understand how this matrix is constructed!");
   }
}

void EliminationCGSolver::BuildPreconditioner()
{
   // first_interface_dofs = master_dofs, column indices corresponding to nonzeros in constraint
   // rectangular B_1 = B_m has lagrange_dofs rows, first_interface_dofs columns
   // square B_2 = B_s has lagrange_dofs rows, second_interface_dofs columns
   projector_ = new EliminationProjection(A_, B_, first_interface_dofs_,
                                          second_interface_dofs_);

   SparseMatrix * explicit_projector = projector_->AssembleExact();
   HypreParMatrix * h_explicit_projector = SerialHypreMatrix(*explicit_projector);
   HypreParMatrix * h_A = SerialHypreMatrix(A_, false);
   h_explicit_operator_ = RAP(h_A, h_explicit_projector);
   h_explicit_operator_->CopyRowStarts();
   h_explicit_operator_->CopyColStarts();
   prec_ = new HypreBoomerAMG(*h_explicit_operator_);
   prec_->SetPrintLevel(0);

   delete explicit_projector;
   delete h_explicit_projector;
   delete h_A;
   // delete explicit_operator;

   // so kind of want to do this, use systems version of AMG, but we've eliminated
   // just some of the dofs associated with a particular direction, so the sizes
   // don't work out! (the *correct* way to do this reorders again or uses rigid body modes
   // or something)
   /*
      SparseMatrix * explicit_operator = RAPWithP(A, *explicit_projector);
      std::cout << "A.Height() = " << A.Height() << ", explicit_operator->Height() = "
                << explicit_operator->Height() << std::endl;
      const int dim = 3;
      HypreBoomerAMGReordered prec(*explicit_operator, dim);
   */

   // next line doesn't really belong here
   rel_tol = 1.e-8;
}

EliminationCGSolver::EliminationCGSolver(SparseMatrix& A, SparseMatrix& B,
                                         Array<int>& master_dofs,
                                         Array<int>& slave_dofs)
   :
   A_(A),
   B_(B),
   first_interface_dofs_(master_dofs),
   second_interface_dofs_(slave_dofs)
{
   BuildPreconditioner();
}

EliminationCGSolver::EliminationCGSolver(SparseMatrix& A, SparseMatrix& B,
                                         int firstblocksize)
   :
   A_(A),
   B_(B)
{
   // identify interface dofs to eliminate via nonzero structure
   BuildSeparatedInterfaceDofs(firstblocksize);

   BuildPreconditioner();
}

void EliminationCGSolver::Mult(const Vector& rhs, Vector& sol) const
{
   RAPOperator reducedoperator(*projector_, A_, *projector_);
   CGSolver krylov;
   krylov.SetOperator(reducedoperator);
   krylov.SetPreconditioner(*prec_);
   krylov.SetMaxIter(max_iter);
   krylov.SetRelTol(rel_tol);
   krylov.SetAbsTol(abs_tol);
   krylov.SetPrintLevel(print_level);

   Vector displacementrhs(A_.Height());
   for (int i = 0; i < displacementrhs.Size(); ++i)
   {
      displacementrhs(i) = rhs(i);
   }
   Vector lagrangerhs(B_.Height());
   for (int i = 0; i < lagrangerhs.Size(); ++i)
   {
      lagrangerhs(i) = rhs(displacementrhs.Size() + i);
   }
   Vector displacementsol(displacementrhs.Size());

   Vector gtilde(displacementrhs.Size());
   projector_->BuildGTilde(lagrangerhs, gtilde);
   A_.AddMult(gtilde, displacementrhs, -1.0);

   Vector reducedrhs(reducedoperator.Height());
   projector_->MultTranspose(displacementrhs, reducedrhs);
   Vector reducedsol(reducedoperator.Height());
   reducedsol = 0.0;
   krylov.Mult(reducedrhs, reducedsol);
   projector_->Mult(reducedsol, displacementsol);

   Vector pressure(lagrangerhs.Size());
   projector_->RecoverPressure(displacementrhs, displacementsol, pressure);

   displacementsol += gtilde;
   for (int i = 0; i < displacementsol.Size(); ++i)
   {
      sol(i) = displacementsol(i);
   }

   for (int i = 0; i < pressure.Size(); ++i)
   {
      sol(displacementsol.Size() + i) = pressure(i);
   }
}

void PenaltyConstrainedSolver::Initialize(HypreParMatrix& A, HypreParMatrix& B)
{
   HypreParMatrix * hBT = B.Transpose();
   HypreParMatrix * hBTB = ParMult(hBT, &B, true);
   // this matrix doesn't get cleanly deleted?
   // (hypre comm pkg)
   penalized_mat = Add(1.0, A, penalty, *hBTB);
   prec = new HypreBoomerAMG(*penalized_mat);
   prec->SetPrintLevel(0);
   // prec->SetSystemsOptions(2); // ???
   delete hBTB;
   delete hBT;
}

PenaltyConstrainedSolver::PenaltyConstrainedSolver(HypreParMatrix& A, SparseMatrix& B,
                                                   double penalty_)
   :
   penalty(penalty_),
   constraintB(B)
{
   HYPRE_Int hB_row_starts[2] = {0, B.Height()};
   HYPRE_Int hB_col_starts[2] = {0, B.Width()};
   HypreParMatrix hB(MPI_COMM_WORLD, B.Height(), B.Width(),
                     hB_row_starts, hB_col_starts, &B);
   Initialize(A, hB);
}

PenaltyConstrainedSolver::PenaltyConstrainedSolver(HypreParMatrix& A, HypreParMatrix& B,
                                                   double penalty_)
   :
   penalty(penalty_),
   constraintB(B)
{
   // TODO: check column starts of A and B are compatible?
   // (probably will happen in ParMult later)

   Initialize(A, B);
}

PenaltyConstrainedSolver::~PenaltyConstrainedSolver()
{
   delete penalized_mat;
   delete prec;
}

void PenaltyConstrainedSolver::Mult(const Vector& b, Vector& x) const
{
   const int disp_size = penalized_mat->Height();
   const int lm_size = b.Size() - disp_size;

   // form penalized right-hand side
   Vector rhs_disp, rhs_lm;
   rhs_disp.MakeRef(const_cast<Vector&>(b), 0, disp_size);
   rhs_lm.MakeRef(const_cast<Vector&>(b), disp_size, lm_size);
   Vector temp(disp_size);
   constraintB.MultTranspose(rhs_lm, temp);
   temp *= penalty;
   Vector penalized_rhs(rhs_disp);
   penalized_rhs += temp;

   // actually solve
   Vector penalized_sol(disp_size);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetOperator(*penalized_mat);
   cg.SetRelTol(rel_tol);
   cg.SetAbsTol(abs_tol);
   cg.SetMaxIter(max_iter);
   cg.SetPrintLevel(print_level);
   cg.SetPreconditioner(*prec);

   penalized_sol = 0.0;
   cg.Mult(penalized_rhs, penalized_sol);

   // recover Lagrange multiplier
   Vector lmtemp(rhs_lm.Size());
   constraintB.Mult(penalized_sol, lmtemp);
   lmtemp -= rhs_lm;
   lmtemp *= penalty;
    
   // put solution in x
   for (int i = 0; i < disp_size; ++i)
   {
      x(i) = penalized_sol(i);
   }
   for (int i = disp_size; i < disp_size + lm_size; ++i)
   {
      x(i) = lmtemp(i - disp_size);
   }
}

/// because IdentityOperator isn't a Solver
class IdentitySolver : public Solver
{
public:
   IdentitySolver(int size) : Solver(size) { }
   void Mult(const Vector& x, Vector& y) const { y = x; }
   void SetOperator(const Operator& op) { }
};

SchurConstrainedSolver::SchurConstrainedSolver(BlockOperator& block_op_,
                                               Solver& primal_pc_)
   :
   block_op(block_op_),
   primal_pc(primal_pc_),
   block_pc(block_op.RowOffsets()),
   dual_pc(NULL)
{
   primal_pc.SetOperator(block_op.GetBlock(0, 0));
   block_pc.SetDiagonalBlock(0, &primal_pc);
   dual_pc = new IdentitySolver(block_op.RowOffsets()[2] -
                                block_op.RowOffsets()[1]);
   block_pc.SetDiagonalBlock(1, dual_pc);
   rel_tol = 1.e-6;
}

SchurConstrainedSolver::~SchurConstrainedSolver()
{
   delete dual_pc;
}

void SchurConstrainedSolver::Mult(const Vector& x, Vector& y) const
{
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetOperator(block_op);
   gmres.SetRelTol(rel_tol);
   gmres.SetAbsTol(abs_tol);
   gmres.SetMaxIter(max_iter);
   gmres.SetPrintLevel(print_level);
   gmres.SetPreconditioner(const_cast<BlockDiagonalPreconditioner&>(block_pc));

   gmres.Mult(x, y);
}

ConstrainedSolver::ConstrainedSolver(Operator& A, Operator& B)
   :
   // Solver(A.Height() + B.Height()),
   Solver(A.Height()),  // not sure conceptually what the size should be!
   offsets(3),
   subsolver(NULL)
{
   offsets[0] = 0;
   offsets[1] = A.Height();
   offsets[2] = A.Height() + B.Height();

   block_op = new BlockOperator(offsets);
   block_op->SetBlock(0, 0, &A);
   block_op->SetBlock(1, 0, &B);
   tr_B = new TransposeOperator(&B);
   block_op->SetBlock(0, 1, tr_B);

   workb.SetSize(A.Height() + B.Height());
   workx.SetSize(A.Height() + B.Height());
   dual_rhs.SetSize(0);
   dual_sol.SetSize(B.Height());
}

ConstrainedSolver::~ConstrainedSolver()
{
   delete block_op;
   delete tr_B;
   delete subsolver;
}

void ConstrainedSolver::SetSchur(Solver& pc)
{
   subsolver = new SchurConstrainedSolver(*block_op, pc);
}

/// @todo consistency in primary/secondary notation (think it's fine
void ConstrainedSolver::SetElimination(Array<int>& primary_dofs,
                                       Array<int>& secondary_dofs)
{
   HypreParMatrix& A = dynamic_cast<HypreParMatrix&>(block_op->GetBlock(0, 0));
   SparseMatrix& B = dynamic_cast<SparseMatrix&>(block_op->GetBlock(1, 0));

   MFEM_VERIFY(secondary_dofs.Size() == B.Height(),
               "Wrong number of dofs for elimination!");

   //// TODO ugly ugly hack (should work in parallel in principle)
   A.GetDiag(hypre_diag);

/*
   Array<int> primary_dofs;
   secondary_dofs.Sort();

   // could go through nonzeros in B, nonzero columns not in secondary_dofs
   // are assumed to be primary dofs, not sure it is worth the effort
*/

   subsolver = new EliminationCGSolver(hypre_diag, B, primary_dofs, secondary_dofs);
}

void ConstrainedSolver::SetPenalty(double penalty)
{
   HypreParMatrix& A = dynamic_cast<HypreParMatrix&>(block_op->GetBlock(0, 0));
   SparseMatrix& B = dynamic_cast<SparseMatrix&>(block_op->GetBlock(1, 0));

   subsolver = new PenaltyConstrainedSolver(A, B, penalty);
}

void ConstrainedSolver::SetDualRHS(const Vector& r)
{
   MFEM_VERIFY(r.Size() == block_op->GetBlock(1, 0).Height(),
               "Vector is wrong size!");
   dual_rhs = r;
}

void ConstrainedSolver::Mult(const Vector& b, Vector& x) const
{
   workb = 0.0;
   workx = 0.0;
   for (int i = 0; i < b.Size(); ++i)
   {
      workb(i) = b(i);
      workx(i) = x(i);
   }
   for (int i = 0; i < dual_rhs.Size(); ++i)
   {
      workb(b.Size() + i) = dual_rhs(i);
   }

   /// note that for EliminationCGSolver, the extension to workb, workx
   /// is promptly undone. We could have a block / reduced option?
   /// (or better, rewrite EliminatioNCGSolver to handle lagrange multiplier blocks)
   subsolver->Mult(workb, workx);

   for (int i = 0; i < b.Size(); ++i)
   {
      x(i) = workx(i);
   }
   for (int i = 0; i < dual_sol.Size(); ++i)
   {
      dual_sol(i) = workx(b.Size() + i);
   }
}

}
