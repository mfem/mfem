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

Eliminator::Eliminator(const SparseMatrix& B, const Array<int>& lagrange_tdofs,
                       const Array<int>& primary_tdofs,
                       const Array<int>& secondary_tdofs)
   :
   lagrange_tdofs_(lagrange_tdofs),
   primary_tdofs_(primary_tdofs),
   secondary_tdofs_(secondary_tdofs)
{
   MFEM_VERIFY(lagrange_tdofs.Size() == secondary_tdofs.Size(),
               "Dof sizes don't match!");

   Bp_.SetSize(lagrange_tdofs.Size(), primary_tdofs.Size());
   B.GetSubMatrix(lagrange_tdofs, primary_tdofs, Bp_);

   Bs_.SetSize(lagrange_tdofs.Size(), secondary_tdofs.Size());
   B.GetSubMatrix(lagrange_tdofs, secondary_tdofs, Bs_);
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

void Eliminator::Eliminate(const Vector& in, Vector& out) const
{
   Bp_.Mult(in, out);
   Bsinverse_.Solve(Bs_.Height(), 1, out);
   out *= -1.0;
}

void Eliminator::EliminateTranspose(const Vector& in, Vector& out) const
{
   Vector work(in);
   BsTinverse_.Solve(Bs_.Height(), 1, work);
   Bp_.MultTranspose(work, out);
   out *= -1.0;
}

void Eliminator::LagrangeSecondary(const Vector& in, Vector& out) const
{
   out = in;
   Bsinverse_.Solve(Bs_.Height(), 1, out);
}

void Eliminator::LagrangeSecondaryTranspose(const Vector& in, Vector& out) const
{
   out = in;
   BsTinverse_.Solve(Bs_.Height(), 1, out);
}

void Eliminator::ExplicitAssembly(DenseMatrix& mat) const
{
   mat.SetSize(Bp_.Height(), Bp_.Width());
   mat = Bp_;
   Bsinverse_.Solve(Bs_.Height(), Bp_.Width(), mat.GetData());
   mat *= -1.0;
}

EliminationProjection::EliminationProjection(const Operator& A,
                                             Array<Eliminator*>& eliminators)
   :
   Operator(A.Height()),
   A_(A),
   eliminators_(eliminators)
{
}

void EliminationProjection::Mult(const Vector& in, Vector& out) const
{
   MFEM_ASSERT(in.Size() == width, "Wrong vector size!");
   MFEM_ASSERT(out.Size() == height, "Wrong vector size!");

   out = in;

   for (int k = 0; k < eliminators_.Size(); ++k)
   {
      Eliminator* elim = eliminators_[k];
      Vector subvecin;
      Vector subvecout(elim->SecondaryDofs().Size());
      in.GetSubVector(elim->PrimaryDofs(), subvecin);
      elim->Eliminate(subvecin, subvecout);
      out.SetSubVector(elim->SecondaryDofs(), subvecout);
   }
}

void EliminationProjection::MultTranspose(const Vector& in, Vector& out) const
{
   MFEM_ASSERT(in.Size() == height, "Wrong vector size!");
   MFEM_ASSERT(out.Size() == width, "Wrong vector size!");

   out = in;

   for (int k = 0; k < eliminators_.Size(); ++k)
   {
      Eliminator* elim = eliminators_[k];
      Vector subvecin;
      Vector subvecout(elim->PrimaryDofs().Size());
      in.GetSubVector(elim->SecondaryDofs(), subvecin);
      elim->EliminateTranspose(subvecin, subvecout);
      out.AddElementVector(elim->PrimaryDofs(), subvecout);
      out.SetSubVector(elim->SecondaryDofs(), 0.0);
   }
}

SparseMatrix * EliminationProjection::AssembleExact() const
{
   SparseMatrix * out = new SparseMatrix(height, width);

   for (int i = 0; i < height; ++i)
   {
      out->Add(i, i, 1.0);
   }

   for (int k = 0; k < eliminators_.Size(); ++k)
   {
      Eliminator* elim = eliminators_[k];
      DenseMatrix mat;
      elim->ExplicitAssembly(mat);
      for (int iz = 0; iz < elim->SecondaryDofs().Size(); ++iz)
      {
         int i = elim->SecondaryDofs()[iz];
         for (int jz = 0; jz < elim->PrimaryDofs().Size(); ++jz)
         {
            int j = elim->PrimaryDofs()[jz];
            out->Add(i, j, mat(iz, jz));
         }
         out->Set(i, i, 0.0);
      }
   }

   out->Finalize();
   return out;
}

void EliminationProjection::BuildGTilde(const Vector& r, Vector& rtilde) const
{
   // MFEM_ASSERT(r.Size() == B_.Height(), "Sizes don't match!");
   MFEM_ASSERT(rtilde.Size() == A_.Height(), "Sizes don't match!");

   rtilde = 0.0;
   for (int k = 0; k < eliminators_.Size(); ++k)
   {
      Eliminator* elim = eliminators_[k];
      Vector subr;
      r.GetSubVector(elim->LagrangeDofs(), subr);
      Vector bsinvr(subr.Size());
      elim->LagrangeSecondary(subr, bsinvr);
      rtilde.AddElementVector(elim->SecondaryDofs(), bsinvr);
   }
}

void EliminationProjection::RecoverMultiplier(
   const Vector& disprhs, const Vector& disp, Vector& lagrangem) const
{
   // MFEM_ASSERT(lagrangem.Size() == B_.Height(), "Sizes don't match!");
   MFEM_ASSERT(disp.Size() == A_.Height(), "Sizes don't match!");

   Vector fullrhs(A_.Height());
   A_.Mult(disp, fullrhs);
   fullrhs -= disprhs;
   fullrhs *= -1.0;
   for (int k = 0; k < eliminators_.Size(); ++k)
   {
      Eliminator* elim = eliminators_[k];
      Vector localsec;
      fullrhs.GetSubVector(elim->SecondaryDofs(), localsec);
      Vector locallagrange(localsec.Size());
      elim->LagrangeSecondaryTranspose(localsec, locallagrange);
      lagrangem.AddElementVector(elim->LagrangeDofs(), locallagrange);
   }
}

EliminationCGSolver::~EliminationCGSolver()
{
   delete h_explicit_operator_;
   for (auto elim : elims_)
   {
      delete elim;
   }
   delete projector_;
   delete prec_;
}

void EliminationCGSolver::BuildPreconditioner()
{
   SparseMatrix * explicit_projector = projector_->AssembleExact();
   HypreParMatrix * h_explicit_projector =
      new HypreParMatrix(hA_.GetComm(), hA_.GetGlobalNumRows(),
                         hA_.GetRowStarts(), explicit_projector);
   h_explicit_projector->CopyRowStarts();
   h_explicit_projector->CopyColStarts();

   h_explicit_operator_ = RAP(&hA_, h_explicit_projector);   
   /// next line because of square projector
   h_explicit_operator_->EliminateZeroRows();
   h_explicit_operator_->CopyRowStarts();
   h_explicit_operator_->CopyColStarts();

   prec_ = new HypreBoomerAMG(*h_explicit_operator_);
   prec_->SetPrintLevel(0);

   delete explicit_projector;
   delete h_explicit_projector;

   // kind of want to do this, use systems version of AMG, but we've eliminated
   // just some of the dofs associated with a particular direction, so the
   // sizes don't work out! (the *correct* way to do this reorders again or
   // uses rigid body modes or something)
   /*
      SparseMatrix * explicit_operator = RAPWithP(A, *explicit_projector);
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
   ConstrainedSolver(A.GetComm(), A, B),
   hA_(A)
{
   MFEM_VERIFY(secondary_dofs.Size() == B.Height(),
               "Wrong number of dofs for elimination!");
   Array<int> lagrange_dofs(secondary_dofs.Size());
   for (int i = 0; i < lagrange_dofs.Size(); ++i)
   {
      lagrange_dofs[i] = i;
   }
   elims_.Append(new Eliminator(B, lagrange_dofs, primary_dofs,
                                secondary_dofs));
   projector_ = new EliminationProjection(hA_, elims_);
   BuildPreconditioner();
}

EliminationCGSolver::EliminationCGSolver(HypreParMatrix& A, SparseMatrix& B,
                                         Array<int>& lagrange_rowstarts)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   hA_(A)
{
   // if (B.Height() > 0)
   if (!B.Empty())
   {
      int * I = B.GetI();
      int * J = B.GetJ();
      double * data = B.GetData();

      for (int k = 0; k < lagrange_rowstarts.Size() - 1; ++k)
      {
         int constraint_size = lagrange_rowstarts[k + 1] -
                               lagrange_rowstarts[k];
         Array<int> lagrange_dofs(constraint_size);
         Array<int> primary_dofs;
         Array<int> secondary_dofs(constraint_size);
         for (int i = lagrange_rowstarts[k]; i < lagrange_rowstarts[k + 1]; ++i)
         {
            lagrange_dofs[i - lagrange_rowstarts[k]] = i;
            int j = J[I[i]];
            double val = data[I[i]];
            secondary_dofs[i  - lagrange_rowstarts[k]] = j;
            // could actually deal with following issue, for now we are lazy
            MFEM_VERIFY(std::abs(val) > 1.e-16,
                        "Explicit zero in leading position in B matrix!");
            for (int jptr = I[i] + 1; jptr < I[i + 1]; ++jptr)
            {
               j = J[jptr];
               val = data[jptr];
               primary_dofs.Append(j);
            }
         }
         primary_dofs.Sort();
         primary_dofs.Unique();
         elims_.Append(new Eliminator(B, lagrange_dofs, primary_dofs,
                                      secondary_dofs));
      }
   }
   projector_ = new EliminationProjection(hA_, elims_);
   BuildPreconditioner();
}

void EliminationCGSolver::Mult(const Vector& rhs, Vector& sol) const
{
   // with square projector, need to add ones on diagonal
   // for quasi-eliminated dofs...
   // RAPOperator reducedoperator(*projector_, spA_, *projector_);

   CGSolver krylov(GetComm());
   krylov.SetOperator(*h_explicit_operator_);
   krylov.SetPreconditioner(*prec_);
   krylov.SetMaxIter(max_iter);
   krylov.SetRelTol(rel_tol);
   krylov.SetAbsTol(abs_tol);
   krylov.SetPrintLevel(print_level);

   Vector rtilde(rhs.Size());
   if (constraint_rhs.Size() > 0)
   {
      projector_->BuildGTilde(constraint_rhs, rtilde);
   }
   else
   {
      rtilde = 0.0;
   }
   Vector temprhs(rhs);
   // hA_.AddMult(rtilde, temprhs, -1.0);
   hA_.Mult(-1.0, rtilde, 1.0, temprhs);

   Vector reducedrhs(rhs.Size());
   projector_->MultTranspose(temprhs, reducedrhs);
   Vector reducedsol(rhs.Size());
   reducedsol = 0.0;
   krylov.Mult(reducedrhs, reducedsol);
   projector_->Mult(reducedsol, sol);

   projector_->RecoverMultiplier(temprhs, sol, multiplier_sol);

   sol += rtilde;
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
   HypreParMatrix hB(comm, B.Height(), B.Width(),
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
SchurConstrainedSolver::SchurConstrainedSolver(MPI_Comm comm, Operator& A_,
                                               Operator& B_)
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
   gmres.SetPreconditioner(
      const_cast<BlockDiagonalPreconditioner&>(*block_pc));

   gmres.Mult(x, y);
}

SchurConstrainedHypreSolver::SchurConstrainedHypreSolver(MPI_Comm comm,
                                                         HypreParMatrix& hA_,
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
