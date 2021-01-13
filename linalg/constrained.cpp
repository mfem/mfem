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

#include "../fem/pfespace.hpp"

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

EliminationSolver::~EliminationSolver()
{
   delete h_explicit_operator_;
   for (auto elim : elims_)
   {
      delete elim;
   }
   delete projector_;
   delete prec_;
}

void EliminationSolver::BuildExplicitOperator()
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

   delete explicit_projector;
   delete h_explicit_projector;
}

EliminationSolver::EliminationSolver(HypreParMatrix& A, SparseMatrix& B,
                                     Array<int>& primary_dofs,
                                     Array<int>& secondary_dofs)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   hA_(A),
   prec_(nullptr)
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
}

EliminationSolver::EliminationSolver(HypreParMatrix& A, SparseMatrix& B,
                                     Array<int>& lagrange_rowstarts)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   hA_(A),
   prec_(nullptr)
{
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
         secondary_dofs = -1;
         // loop through rows, identify one secondary dof for each row
         for (int i = lagrange_rowstarts[k]; i < lagrange_rowstarts[k + 1]; ++i)
         {
            lagrange_dofs[i - lagrange_rowstarts[k]] = i;
            for (int jptr = I[i]; jptr < I[i + 1]; ++jptr)
            {
               int j = J[jptr];
               double val = data[jptr];
               if (std::abs(val) > 1.e-12 && secondary_dofs.Find(j) == -1)
               {
                  secondary_dofs[i - lagrange_rowstarts[k]] = j;
                  break;
               }
            }
         }
         // loop through rows again, assigning non-secondary dofs as primary
         for (int i = lagrange_rowstarts[k]; i < lagrange_rowstarts[k + 1]; ++i)
         {
            MFEM_ASSERT(secondary_dofs[i - lagrange_rowstarts[k]] >= 0,
                        "Secondary dofs don't match rows!");
            for (int jptr = I[i]; jptr < I[i + 1]; ++jptr)
            {
               int j = J[jptr];
               if (secondary_dofs.Find(j) == -1)
               {
                  primary_dofs.Append(j);
               }
            }
         }
         primary_dofs.Sort();
         primary_dofs.Unique();
         elims_.Append(new Eliminator(B, lagrange_dofs, primary_dofs,
                                      secondary_dofs));
      }
   }
   projector_ = new EliminationProjection(hA_, elims_);
}

void EliminationSolver::PrimalMult(const Vector& rhs, Vector& sol) const
{
   if (!prec_)
   {
      prec_ = BuildPreconditioner();
   }
   IterativeSolver * krylov = BuildKrylov();
   krylov->SetOperator(*h_explicit_operator_);
   krylov->SetPreconditioner(*prec_);
   krylov->SetMaxIter(max_iter);
   krylov->SetRelTol(rel_tol);
   krylov->SetAbsTol(abs_tol);
   krylov->SetPrintLevel(print_level);

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
   hA_.Mult(-1.0, rtilde, 1.0, temprhs);

   Vector reducedrhs(rhs.Size());
   projector_->MultTranspose(temprhs, reducedrhs);
   Vector reducedsol(rhs.Size());
   reducedsol = 0.0;
   krylov->Mult(reducedrhs, reducedsol);
   final_iter = krylov->GetNumIterations();
   final_norm = krylov->GetFinalNorm();
   converged = krylov->GetConverged();
   delete krylov;

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
   delete hBTB;
   delete hBT;
}

PenaltyConstrainedSolver::PenaltyConstrainedSolver(
   HypreParMatrix& A, SparseMatrix& B, double penalty_)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   penalty(penalty_),
   constraintB(B),
   prec(nullptr)
{
   HYPRE_Int hB_row_starts[2] = {0, B.Height()};
   HYPRE_Int hB_col_starts[2] = {0, B.Width()};
   HypreParMatrix hB(A.GetComm(), B.Height(), B.Width(),
                     hB_row_starts, hB_col_starts, &B);
   Initialize(A, hB);
}

PenaltyConstrainedSolver::PenaltyConstrainedSolver(
   HypreParMatrix& A, HypreParMatrix& B, double penalty_)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   penalty(penalty_),
   constraintB(B),
   prec(nullptr)
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

void PenaltyConstrainedSolver::PrimalMult(const Vector& b, Vector& x) const
{
   if (!prec)
   {
      prec = BuildPreconditioner();
   }
   IterativeSolver * krylov = BuildKrylov();

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
   krylov->SetOperator(*penalized_mat);
   krylov->SetRelTol(rel_tol);
   krylov->SetAbsTol(abs_tol);
   krylov->SetMaxIter(max_iter);
   krylov->SetPrintLevel(print_level);
   krylov->SetPreconditioner(*prec);
   krylov->Mult(penalized_rhs, x);
   final_iter = krylov->GetNumIterations();
   final_norm = krylov->GetFinalNorm();
   converged = krylov->GetConverged();
   delete krylov;

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

void SchurConstrainedSolver::Mult(const Vector& x, Vector& y) const
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
   final_iter = gmres.GetNumIterations();
}

SchurConstrainedHypreSolver::SchurConstrainedHypreSolver(MPI_Comm comm,
                                                         HypreParMatrix& hA_,
                                                         HypreParMatrix& hB_,
                                                         int dimension,
                                                         bool reorder)
   :
   SchurConstrainedSolver(comm, hA_, hB_),
   hA(hA_),
   hB(hB_)
{
   auto h_primal_pc = new HypreBoomerAMG(hA);
   h_primal_pc->SetPrintLevel(0);
   if (dimension > 0)
   {
      h_primal_pc->SetSystemsOptions(dimension, reorder);
   }
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
   height = A.Height() + B.Height();
   width = A.Width() + B.Height();

   workb.SetSize(A.Height());
   workx.SetSize(A.Height());
   constraint_rhs.SetSize(B.Height());
   constraint_rhs = 0.0;
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

void ConstrainedSolver::PrimalMult(const Vector& f, Vector &x) const
{
   Vector pworkb(A.Height() + B.Height());
   Vector pworkx(A.Height() + B.Height());
   pworkb = 0.0;
   pworkx = 0.0;
   for (int i = 0; i < f.Size(); ++i)
   {
      pworkb(i) = f(i);
      pworkx(i) = x(i);
   }
   for (int i = 0; i < B.Height(); ++i)
   {
      pworkb(f.Size() + i) = constraint_rhs(i);
   }

   Mult(pworkb, pworkx);

   for (int i = 0; i < f.Size(); ++i)
   {
      x(i) = pworkx(i);
   }
   for (int i = 0; i < B.Height(); ++i)
   {
      multiplier_sol(i) = pworkx(f.Size() + i);
   }
}

void ConstrainedSolver::Mult(const Vector& f_and_r, Vector& x_and_lambda) const
{
   /// TODO: use GetData, Vector constructor that just wraps doubles, etc?
   for (int i = 0; i < A.Height(); ++i)
   {
      workx(i) = x_and_lambda(i);
      workb(i) = f_and_r(i);
   }
   Vector ref_constraint_rhs(f_and_r.GetData() + A.Height(), B.Height());
   constraint_rhs = ref_constraint_rhs;
   PrimalMult(workb, workx);
   for (int i = 0; i < A.Height(); ++i)
   {
      x_and_lambda(i) = workx(i);
   }
   for (int i = 0; i < B.Height(); ++i)
   {
      x_and_lambda(A.Height() + i) = multiplier_sol(i);
   }
}

}
