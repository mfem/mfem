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

NodalEliminationProjection::NodalEliminationProjection(const SparseMatrix& A, const SparseMatrix& B)
   :
   Operator(A.Height(),
            A.Height() - B.Height()),
   A_(A),
   B_(B),
   secondary_inv_(B.Height())
{
   const int * I = B.GetI();
   const int * J = B.GetJ();
   const double * data = B.GetData();
   for (int i = 0; i < B.Height(); ++i)
   {
      const int jidx = I[i];
      const int j = J[jidx];
      const double val = data[jidx];
      MFEM_VERIFY(std::abs(val) > 1.e-14, "Cannot eliminate!");
      secondary_inv_(i) = 1.0 / val;
      secondary_dofs_.Append(j);
      for (int jidx = I[i] + 1; jidx < I[i + 1]; ++jidx)
      {
         const int j = J[jidx];
         primary_dofs_.Append(j);
      }
   }
   primary_dofs_.Sort();
   // should probably either sort or store some kind of map?
   // secondary_dofs_.Sort(); // ATB veteran

   int column_dof = 0;
   for (int i = 0; i < A.Height(); ++i)
   {
      if (secondary_dofs_.Find(i) == -1)
      {
         // otherwise, dof exists in reduced system (identity)
         if (primary_dofs_.FindSorted(i) >= 0)
         {
            // in addition, mapped_primary_contact_dofs[reduced_id] = larger_id
            mapped_primary_dofs_.Append(column_dof);
         }
         column_dof++;
      }
   }
}

void NodalEliminationProjection::Mult(const Vector& x, Vector& y) const
{
   MFEM_VERIFY(x.Size() == width, "Vector size doesn't match!");
   MFEM_VERIFY(y.Size() == height, "Vector size doesn't match!");

   y = 0.0;

   int column_dof = 0;
   int sequence = 0;
   const int * BI = B_.GetI();
   // const int * BJ = B_.GetJ();
   const double * Bdata = B_.GetData();
   for (int i = 0; i < A_.Height(); ++i)
   {
      // const int brow = secondary_dofs_.FindSorted(i); // ATB veteran
      const int brow = secondary_dofs_.Find(i);
      if (brow >= 0)
      {
         double val = 0.0;
         for (int jidx = BI[brow] + 1; jidx < BI[brow + 1]; ++jidx)
         {
            // const int bcol = BJ[jidx];
            const double bval = Bdata[jidx];
            val += secondary_inv_(brow) * bval * x(mapped_primary_dofs_[sequence]);
            sequence++;
         }
         y(i) += val;
      }
      else
      {
         y(i) += -x(column_dof);
         column_dof++;
      }
   }
}

void NodalEliminationProjection::MultTranspose(const Vector& in, Vector& out) const
{
   int num_elim_dofs = secondary_dofs_.Size();
   MFEM_ASSERT(out.Size() == A_.Height() - num_elim_dofs, "Sizes don't match!");
   MFEM_ASSERT(in.Size() == A_.Height(), "Sizes don't match!");

   out = 0.0;

   int row_dof = 0;
   int sequence = 0;
   const int * BI = B_.GetI();
   // const int * BJ = B_.GetJ();
   const double * Bdata = B_.GetData();
   for (int i = 0; i < A_.Height(); ++i)
   {
      const int brow = secondary_dofs_.Find(i);
      if (brow >= 0)
      {
         for (int jidx = BI[brow] + 1; jidx < BI[brow + 1]; ++jidx)
         {
            const double bval = Bdata[jidx];
            out(mapped_primary_dofs_[sequence]) += secondary_inv_(brow) * bval * in(i);
            sequence++;
         }
      }
      else
      {
         out(row_dof) += -in(i);
         row_dof++;
      }
   }
}

Eliminator::Eliminator(const SparseMatrix& B, const Array<int>& lagrange_tdofs,
                       const Array<int>& primary_tdofs, const Array<int>& secondary_tdofs)
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

EliminationProjection::EliminationProjection(const Operator& A, Array<Eliminator*>& eliminators)
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

// drafted (untested)
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

// drafted (untested)
void EliminationProjection::RecoverPressure(
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
      // BsTinverse_.Solve(Bs_.Height(), 1, lagrangem);
      elim->LagrangeSecondaryTranspose(localsec, locallagrange);
      lagrangem.AddElementVector(elim->LagrangeDofs(), locallagrange);
   }
}

EliminationCGSolver::~EliminationCGSolver()
{
   delete h_explicit_operator_;
   delete elim_;
   delete projector_;
   delete prec_;
}

void EliminationCGSolver::BuildPreconditioner(SparseMatrix& spB)
{
   // first_interface_dofs = primary_dofs, column indices corresponding to nonzeros in constraint
   // rectangular B_1 = B_p has lagrange_dofs rows, first_interface_dofs columns
   // square B_2 = B_s has lagrange_dofs rows, second_interface_dofs columns
   Array<int> lagrange_dofs(second_interface_dofs_.Size());
   for (int i = 0; i < lagrange_dofs.Size(); ++i)
   {
      lagrange_dofs[i] = i;
   }
   elim_ = new Eliminator(spB, lagrange_dofs, first_interface_dofs_,
                          second_interface_dofs_);
   Array<Eliminator*> elims;
   elims.Append(elim_);
   projector_ = new EliminationProjection(hA_, elims);

   SparseMatrix * explicit_projector = projector_->AssembleExact();
   HypreParMatrix * h_explicit_projector = SerialHypreMatrix(*explicit_projector);
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
   BuildPreconditioner(B);
}

void EliminationCGSolver::Mult(const Vector& rhs, Vector& sol) const
{
   // with square projector, need to add ones on diagonal
   // for quasi-eliminated dofs...
   // RAPOperator reducedoperator(*projector_, spA_, *projector_);

   CGSolver krylov;
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

   projector_->RecoverPressure(temprhs, sol, multiplier_sol);

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
