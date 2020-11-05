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
                                             Array<int>& primary_contact_dofs,
                                             Array<int>& secondary_contact_dofs)
   :
   Operator(A.Height(),
            A.Height() - secondary_contact_dofs.Size()),
   A_(A),
   B_(B),
   primary_contact_dofs_(primary_contact_dofs),
   secondary_contact_dofs_(secondary_contact_dofs)
{
   Array<int> lm_dofs;
   for (int i = 0; i < B.Height(); ++i)
   {
      lm_dofs.Append(i);
   }
   Bp_.SetSize(B_.Height(), primary_contact_dofs.Size());
   B_.GetSubMatrix(lm_dofs, primary_contact_dofs, Bp_);

   Bs_.SetSize(B_.Height(), secondary_contact_dofs.Size());
   B_.GetSubMatrix(lm_dofs, secondary_contact_dofs, Bs_);
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

   It *may* be possible to implement this with Bp_ as a findpts call
   rather than a matrix; the hypre assembly will not be so great, though
*/
SparseMatrix * EliminationProjection::AssembleExact() const
{
   int num_elim_dofs = secondary_contact_dofs_.Size();

   SparseMatrix * out = new SparseMatrix(
      A_.Height(), A_.Height() - num_elim_dofs);

   int column_dof = 0;
   Array<int> mapped_primary_contact_dofs;
   for (int i = 0; i < A_.Height(); ++i)
   {
      if (secondary_contact_dofs_.FindSorted(i) >= 0)
      {
         // if dof is a secondary, it doesn't show up in reduced system
         ;
      }
      else
      {
         // otherwise, dof exists in reduced system (identity)
         out->Set(i, column_dof, 1.0);
         if (primary_contact_dofs_.FindSorted(i) >= 0)
         {
            // in addition, mapped_primary_contact_dofs[reduced_id] = larger_id
            mapped_primary_contact_dofs.Append(column_dof);
         }
         column_dof++;
      }
   }
   MFEM_ASSERT(mapped_primary_contact_dofs.Size() == primary_contact_dofs_.Size(),
               "Unable to map primary contact dofs!");

   DenseMatrix block(Bp_);
   // the following line may be expensive in the general case
   Bsinverse_.Solve(Bs_.Height(), Bp_.Width(), block.GetData());

   for (int iz = 0; iz < secondary_contact_dofs_.Size(); ++iz)
   {
      int i = secondary_contact_dofs_[iz];
      for (int jz = 0; jz < mapped_primary_contact_dofs.Size(); ++jz)
      {
         int j = mapped_primary_contact_dofs[jz];
         out->Add(i, j, -block(iz, jz));
      }
   }
   out->Finalize();
   return out;
}

void EliminationProjection::Mult(const Vector& in, Vector& out) const
{
   int num_elim_dofs = secondary_contact_dofs_.Size();
   MFEM_ASSERT(in.Size() == A_.Height() - num_elim_dofs, "Sizes don't match!");
   MFEM_ASSERT(out.Size() == A_.Height(), "Sizes don't match!");

   out = 0.0;
   int column_dof = 0;
   Array<int> mapped_primary_contact_dofs;
   for (int i = 0; i < A_.Height(); ++i)
   {
      if (secondary_contact_dofs_.FindSorted(i) >= 0)
      {
         // if dof is a secondary, it doesn't show up in reduced system
         ;
      }
      else
      {
         // otherwise, dof exists in reduced system (identity)
         out(i) += in(column_dof);
         if (primary_contact_dofs_.FindSorted(i) >= 0)
         {
            // in addition, mapped_primary_contact_dofs[reduced_id] = larger_id
            mapped_primary_contact_dofs.Append(column_dof);
         }
         column_dof++;
      }
   }
   MFEM_ASSERT(mapped_primary_contact_dofs.Size() == primary_contact_dofs_.Size(),
               "Unable to map primary contact dofs!");

   Vector subvecin;
   Vector subvecout(secondary_contact_dofs_.Size());
   in.GetSubVector(mapped_primary_contact_dofs, subvecin);
   Bp_.Mult(subvecin, subvecout);
   Bsinverse_.Solve(Bs_.Height(), 1, subvecout);
   subvecout *= -1.0;
   out.AddElementVector(secondary_contact_dofs_, subvecout);
}

void EliminationProjection::MultTranspose(const Vector& in, Vector& out) const
{
   int num_elim_dofs = secondary_contact_dofs_.Size();
   MFEM_ASSERT(out.Size() == A_.Height() - num_elim_dofs, "Sizes don't match!");
   MFEM_ASSERT(in.Size() == A_.Height(), "Sizes don't match!");

   out = 0.0;
   int row_dof = 0;
   Array<int> mapped_primary_contact_dofs;
   for (int i = 0; i < A_.Height(); ++i)
   {
      if (secondary_contact_dofs_.FindSorted(i) >= 0)
      {
         ;
      }
      else
      {
         out(row_dof) += in(i);
         if (primary_contact_dofs_.FindSorted(i) >= 0)
         {
            mapped_primary_contact_dofs.Append(row_dof);
         }
         row_dof++;
      }
   }
   MFEM_ASSERT(mapped_primary_contact_dofs.Size() == primary_contact_dofs_.Size(),
               "Unable to map primary contact dofs!");

   Vector subvecin;
   Vector subvecout(Bp_.Width());

   in.GetSubVector(secondary_contact_dofs_, subvecin);
   BsTinverse_.Solve(Bs_.Height(), 1, subvecin);
   Bp_.MultTranspose(subvecin, subvecout);
   subvecout *= -1.0;
   out.AddElementVector(mapped_primary_contact_dofs, subvecout);
}

void EliminationProjection::BuildGTilde(const Vector& g, Vector& gtilde) const
{
   // int num_elim_dofs = secondary_contact_dofs_.Size();
   MFEM_ASSERT(g.Size() == B_.Height(), "Sizes don't match!");
   MFEM_ASSERT(gtilde.Size() == A_.Height(), "Sizes don't match!");

   gtilde = 0.0;
   Vector cinvg(g);
   Bsinverse_.Solve(Bs_.Height(), 1, cinvg);
   gtilde.AddElementVector(secondary_contact_dofs_, cinvg);
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
   fullrhs.GetSubVector(secondary_contact_dofs_, pressure);
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
   int * I = spB_.GetI();
   int * J = spB_.GetJ();
   double * data = spB_.GetData();
   const double tol = 1.e-14;
   for (int i = 0; i < spB_.Height(); ++i)
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

   if (second_interface_size != spB_.Height())
   {
      // is this really expected? equating nodes with dofs in some weird way on a manifold?
      std::cerr << "spB_.Height() = " << spB_.Height()
                << ", secondary_interface_size = " << second_interface_size << std::endl;
      MFEM_VERIFY(false, "I don't understand how this matrix is constructed!");
   }
}

void EliminationCGSolver::BuildPreconditioner()
{
   // first_interface_dofs = primary_dofs, column indices corresponding to nonzeros in constraint
   // rectangular B_1 = B_m has lagrange_dofs rows, first_interface_dofs columns
   // square B_2 = B_s has lagrange_dofs rows, second_interface_dofs columns
   projector_ = new EliminationProjection(spA_, spB_, first_interface_dofs_,
                                          second_interface_dofs_);

   SparseMatrix * explicit_projector = projector_->AssembleExact();
   HypreParMatrix * h_explicit_projector = SerialHypreMatrix(*explicit_projector);
   h_explicit_operator_ = RAP(&hA_, h_explicit_projector);
   h_explicit_operator_->CopyRowStarts();
   h_explicit_operator_->CopyColStarts();
   prec_ = new HypreBoomerAMG(*h_explicit_operator_);
   prec_->SetPrintLevel(0);

   delete explicit_projector;
   delete h_explicit_projector;

   // kind of want to do this, use systems version of AMG, but we've eliminated
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

EliminationCGSolver::EliminationCGSolver(HypreParMatrix& A, SparseMatrix& B,
                                         Array<int>& primary_dofs,
                                         Array<int>& secondary_dofs)
   :
   ConstrainedSolver(MPI_COMM_SELF, A, B),
   hA_(A),
   spB_(B),
   first_interface_dofs_(primary_dofs),
   second_interface_dofs_(secondary_dofs)
{
   MFEM_VERIFY(secondary_dofs.Size() == B.Height(),
               "Wrong number of dofs for elimination!");
   A.GetDiag(spA_);
   BuildPreconditioner();
}

EliminationCGSolver::EliminationCGSolver(HypreParMatrix& A, SparseMatrix& B,
                                         int firstblocksize)
   :
   ConstrainedSolver(MPI_COMM_SELF, A, B),
   hA_(A),
   spB_(B)
{
   A.GetDiag(spA_);

   // identify interface dofs to eliminate via nonzero structure
   BuildSeparatedInterfaceDofs(firstblocksize);

   BuildPreconditioner();
}

void EliminationCGSolver::Mult(const Vector& rhs, Vector& sol) const
{
   RAPOperator reducedoperator(*projector_, spA_, *projector_);
   CGSolver krylov;
   krylov.SetOperator(reducedoperator);
   krylov.SetPreconditioner(*prec_);
   krylov.SetMaxIter(max_iter);
   krylov.SetRelTol(rel_tol);
   krylov.SetAbsTol(abs_tol);
   krylov.SetPrintLevel(print_level);

   Vector gtilde(rhs.Size());
   if (constraint_rhs.Size() > 0)
   {
      projector_->BuildGTilde(constraint_rhs, gtilde);
   }
   else
   {
      gtilde = 0.0;
   }
   Vector temprhs(rhs);
   spA_.AddMult(gtilde, temprhs, -1.0);

   Vector reducedrhs(reducedoperator.Height());
   projector_->MultTranspose(temprhs, reducedrhs);
   Vector reducedsol(reducedoperator.Height());
   reducedsol = 0.0;
   krylov.Mult(reducedrhs, reducedsol);
   projector_->Mult(reducedsol, sol);

   projector_->RecoverPressure(temprhs, sol, multiplier_sol);

   sol += gtilde;
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

PenaltyConstrainedSolver::PenaltyConstrainedSolver(
   MPI_Comm comm, HypreParMatrix& A, SparseMatrix& B, double penalty_)
   :
   ConstrainedSolver(comm, A, B),
   penalty(penalty_),
   constraintB(B)
{
   HYPRE_Int hB_row_starts[2] = {0, B.Height()};
   HYPRE_Int hB_col_starts[2] = {0, B.Width()};
   HypreParMatrix hB(MPI_COMM_WORLD, B.Height(), B.Width(),
                     hB_row_starts, hB_col_starts, &B);
   Initialize(A, hB);
}

PenaltyConstrainedSolver::PenaltyConstrainedSolver(
   MPI_Comm comm, HypreParMatrix& A, HypreParMatrix& B, double penalty_)
   :
   ConstrainedSolver(comm, A, B),
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
   // form penalized right-hand side
   Vector penalized_rhs(b);
   if (constraint_rhs.Size() > 0)
   {
      Vector temp(x.Size());
      constraintB.MultTranspose(constraint_rhs, temp);
      temp *= penalty;
      penalized_rhs += temp;
   }

   // actually solve
   CGSolver cg(GetComm());
   cg.SetOperator(*penalized_mat);
   cg.SetRelTol(rel_tol);
   cg.SetAbsTol(abs_tol);
   cg.SetMaxIter(max_iter);
   cg.SetPrintLevel(print_level);
   cg.SetPreconditioner(*prec);
   cg.Mult(penalized_rhs, x);

   constraintB.Mult(x, multiplier_sol);
   if (constraint_rhs.Size() > 0)
   {
      multiplier_sol -= constraint_rhs;
   }
   multiplier_sol *= penalty;
}

/// because IdentityOperator isn't a Solver
class IdentitySolver : public Solver
{
public:
   IdentitySolver(int size) : Solver(size) { }
   void Mult(const Vector& x, Vector& y) const { y = x; }
   void SetOperator(const Operator& op) { }
};

void SchurConstrainedSolver::Initialize()
{
   offsets[0] = 0;
   offsets[1] = A.Height();
   offsets[2] = A.Height() + B.Height();

   block_op = new BlockOperator(offsets);
   block_op->SetBlock(0, 0, &A);
   block_op->SetBlock(1, 0, &B);
   tr_B = new TransposeOperator(&B);
   block_op->SetBlock(0, 1, tr_B);

   block_pc = new BlockDiagonalPreconditioner(block_op->RowOffsets()),
   rel_tol = 1.e-6;
}

SchurConstrainedSolver::SchurConstrainedSolver(MPI_Comm comm,
                                               Operator& A_, Operator& B_,
                                               Solver& primal_pc_)
   :
   ConstrainedSolver(comm, A_, B_),
   offsets(3),
   primal_pc(&primal_pc_),
   dual_pc(nullptr)
{
   Initialize();
   primal_pc->SetOperator(block_op->GetBlock(0, 0));
   dual_pc = new IdentitySolver(block_op->RowOffsets()[2] -
                                block_op->RowOffsets()[1]);
   block_pc->SetDiagonalBlock(0, primal_pc);
   block_pc->SetDiagonalBlock(1, dual_pc);
}

// protected constructor
SchurConstrainedSolver::SchurConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_)
   :
   ConstrainedSolver(comm, A_, B_),
   offsets(3),
   primal_pc(nullptr),
   dual_pc(nullptr)
{
   Initialize();
}

SchurConstrainedSolver::~SchurConstrainedSolver()
{
   delete block_op;
   delete tr_B;
   delete block_pc;
   delete dual_pc;
}

void SchurConstrainedSolver::SaddleMult(const Vector& x, Vector& y) const
{
   GMRESSolver gmres(GetComm());
   gmres.SetOperator(*block_op);
   gmres.SetRelTol(rel_tol);
   gmres.SetAbsTol(abs_tol);
   gmres.SetMaxIter(max_iter);
   gmres.SetPrintLevel(print_level);
   gmres.SetPreconditioner(const_cast<BlockDiagonalPreconditioner&>(*block_pc));

   gmres.Mult(x, y);
}

SchurConstrainedHypreSolver::SchurConstrainedHypreSolver(MPI_Comm comm, HypreParMatrix& hA_,
                                                         HypreParMatrix& hB_)
   :
   SchurConstrainedSolver(comm, hA_, hB_),
   hA(hA_),
   hB(hB_)
{
   auto h_primal_pc = new HypreBoomerAMG(hA);
   h_primal_pc->SetPrintLevel(0);
   primal_pc = h_primal_pc;
   
   HypreParMatrix * scaledB = new HypreParMatrix(hB);
   Vector diagA;
   hA.GetDiag(diagA);
   HypreParMatrix * scaledBT = scaledB->Transpose();
   scaledBT->InvScaleRows(diagA);
   schur_mat = ParMult(scaledB, scaledBT);
   schur_mat->CopyRowStarts();
   schur_mat->CopyColStarts();
   auto h_dual_pc = new HypreBoomerAMG(*schur_mat);
   h_dual_pc->SetPrintLevel(0);
   dual_pc = h_dual_pc;
   delete scaledB;
   delete scaledBT;

   block_pc->SetDiagonalBlock(0, primal_pc);
   block_pc->SetDiagonalBlock(1, dual_pc);
}

SchurConstrainedHypreSolver::~SchurConstrainedHypreSolver()
{
   delete schur_mat;
   delete primal_pc;
}

ConstrainedSolver::ConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_)
   :
   IterativeSolver(comm), A(A_), B(B_)
{
   height = A.Height();
   width = A.Width();

   workb.SetSize(A.Height() + B.Height());
   workx.SetSize(A.Height() + B.Height());
   constraint_rhs.SetSize(0);
   multiplier_sol.SetSize(B.Height());
}

ConstrainedSolver::~ConstrainedSolver()
{
}

void ConstrainedSolver::SetConstraintRHS(const Vector& r)
{
   MFEM_VERIFY(r.Size() == multiplier_sol.Size(), "Vector is wrong size!");
   constraint_rhs = r;
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
   for (int i = 0; i < constraint_rhs.Size(); ++i)
   {
      workb(b.Size() + i) = constraint_rhs(i);
   }

   SaddleMult(workb, workx);

   for (int i = 0; i < b.Size(); ++i)
   {
      x(i) = workx(i);
   }
   for (int i = 0; i < multiplier_sol.Size(); ++i)
   {
      multiplier_sol(i) = workx(b.Size() + i);
   }
}

}
