// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "constraints.hpp"

#include "../fem/fespace.hpp"
#include "../fem/pfespace.hpp"

#include <set>

namespace mfem
{

Eliminator::Eliminator(const SparseMatrix& B, const Array<int>& lagrange_tdofs_,
                       const Array<int>& primary_tdofs_,
                       const Array<int>& secondary_tdofs_)
   :
   lagrange_tdofs(lagrange_tdofs_),
   primary_tdofs(primary_tdofs_),
   secondary_tdofs(secondary_tdofs_)
{
   MFEM_VERIFY(lagrange_tdofs.Size() == secondary_tdofs.Size(),
               "Dof sizes don't match!");

   Bp.SetSize(lagrange_tdofs.Size(), primary_tdofs.Size());
   B.GetSubMatrix(lagrange_tdofs, primary_tdofs, Bp);

   Bs.SetSize(lagrange_tdofs.Size(), secondary_tdofs.Size());
   B.GetSubMatrix(lagrange_tdofs, secondary_tdofs, Bs);
   BsT.Transpose(Bs);

   ipiv.SetSize(Bs.Height());
   Bsinverse.data = Bs.HostReadWrite();
   Bsinverse.ipiv = ipiv.HostReadWrite();
   Bsinverse.Factor(Bs.Height());

   ipivT.SetSize(Bs.Height());
   BsTinverse.data = BsT.HostReadWrite();
   BsTinverse.ipiv = ipivT.HostReadWrite();
   BsTinverse.Factor(Bs.Height());
}

void Eliminator::Eliminate(const Vector& vin, Vector& vout) const
{
   Bp.Mult(vin, vout);
   Bsinverse.Solve(Bs.Height(), 1, vout.GetData());
   vout *= -1.0;
}

void Eliminator::EliminateTranspose(const Vector& vin, Vector& vout) const
{
   Vector work(vin);
   BsTinverse.Solve(Bs.Height(), 1, work.GetData());
   Bp.MultTranspose(work, vout);
   vout *= -1.0;
}

void Eliminator::LagrangeSecondary(const Vector& vin, Vector& vout) const
{
   vout = vin;
   Bsinverse.Solve(Bs.Height(), 1, vout.GetData());
}

void Eliminator::LagrangeSecondaryTranspose(const Vector& vin,
                                            Vector& vout) const
{
   vout = vin;
   BsTinverse.Solve(Bs.Height(), 1, vout.GetData());
}

void Eliminator::ExplicitAssembly(DenseMatrix& mat) const
{
   mat.SetSize(Bp.Height(), Bp.Width());
   mat = Bp;
   Bsinverse.Solve(Bs.Height(), Bp.Width(), mat.GetData());
   mat *= -1.0;
}

EliminationProjection::EliminationProjection(const Operator& A,
                                             Array<Eliminator*>& eliminators_)
   :
   Operator(A.Height()),
   Aop(A),
   eliminators(eliminators_)
{
}

void EliminationProjection::Mult(const Vector& vin, Vector& vout) const
{
   MFEM_ASSERT(vin.Size() == width, "Wrong vector size!");
   MFEM_ASSERT(vout.Size() == height, "Wrong vector size!");

   vout = vin;

   for (int k = 0; k < eliminators.Size(); ++k)
   {
      Eliminator* elim = eliminators[k];
      Vector subvec_in;
      Vector subvec_out(elim->SecondaryDofs().Size());
      vin.GetSubVector(elim->PrimaryDofs(), subvec_in);
      elim->Eliminate(subvec_in, subvec_out);
      vout.SetSubVector(elim->SecondaryDofs(), subvec_out);
   }
}

void EliminationProjection::MultTranspose(const Vector& vin, Vector& vout) const
{
   MFEM_ASSERT(vin.Size() == height, "Wrong vector size!");
   MFEM_ASSERT(vout.Size() == width, "Wrong vector size!");

   vout = vin;

   for (int k = 0; k < eliminators.Size(); ++k)
   {
      Eliminator* elim = eliminators[k];
      Vector subvec_in;
      Vector subvec_out(elim->PrimaryDofs().Size());
      vin.GetSubVector(elim->SecondaryDofs(), subvec_in);
      elim->EliminateTranspose(subvec_in, subvec_out);
      vout.AddElementVector(elim->PrimaryDofs(), subvec_out);
      vout.SetSubVector(elim->SecondaryDofs(), 0.0);
   }
}

SparseMatrix * EliminationProjection::AssembleExact() const
{
   SparseMatrix * mat = new SparseMatrix(height, width);

   for (int i = 0; i < height; ++i)
   {
      mat->Add(i, i, 1.0);
   }

   for (int k = 0; k < eliminators.Size(); ++k)
   {
      Eliminator* elim = eliminators[k];
      DenseMatrix mat_k;
      elim->ExplicitAssembly(mat_k);
      for (int iz = 0; iz < elim->SecondaryDofs().Size(); ++iz)
      {
         int i = elim->SecondaryDofs()[iz];
         for (int jz = 0; jz < elim->PrimaryDofs().Size(); ++jz)
         {
            int j = elim->PrimaryDofs()[jz];
            mat->Add(i, j, mat_k(iz, jz));
         }
         mat->Set(i, i, 0.0);
      }
   }

   mat->Finalize();
   return mat;
}

void EliminationProjection::BuildGTilde(const Vector& r, Vector& rtilde) const
{
   MFEM_ASSERT(rtilde.Size() == Aop.Height(), "Sizes don't match!");

   rtilde = 0.0;
   for (int k = 0; k < eliminators.Size(); ++k)
   {
      Eliminator* elim = eliminators[k];
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
   lagrangem = 0.0;
   MFEM_ASSERT(disp.Size() == Aop.Height(), "Sizes don't match!");

   Vector fullrhs(Aop.Height());
   Aop.Mult(disp, fullrhs);
   fullrhs -= disprhs;
   fullrhs *= -1.0;
   for (int k = 0; k < eliminators.Size(); ++k)
   {
      Eliminator* elim = eliminators[k];
      Vector localsec;
      fullrhs.GetSubVector(elim->SecondaryDofs(), localsec);
      Vector locallagrange(localsec.Size());
      elim->LagrangeSecondaryTranspose(localsec, locallagrange);
      lagrangem.AddElementVector(elim->LagrangeDofs(), locallagrange);
   }
}

#ifdef MFEM_USE_MPI

EliminationSolver::~EliminationSolver()
{
   delete h_explicit_operator;
   for (auto elim : eliminators)
   {
      delete elim;
   }
   delete projector;
   delete prec;
   delete krylov;
}

void EliminationSolver::BuildExplicitOperator()
{
   SparseMatrix * explicit_projector = projector->AssembleExact();
   HypreParMatrix * h_explicit_projector =
      new HypreParMatrix(hA.GetComm(), hA.GetGlobalNumRows(),
                         hA.GetRowStarts(), explicit_projector);
   h_explicit_projector->CopyRowStarts();
   h_explicit_projector->CopyColStarts();

   h_explicit_operator = RAP(&hA, h_explicit_projector);
   // next line because of square projector
   h_explicit_operator->EliminateZeroRows();
   h_explicit_operator->CopyRowStarts();
   h_explicit_operator->CopyColStarts();

   delete explicit_projector;
   delete h_explicit_projector;
}

EliminationSolver::EliminationSolver(HypreParMatrix& A, SparseMatrix& B,
                                     Array<int>& primary_dofs,
                                     Array<int>& secondary_dofs)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   hA(A),
   krylov(nullptr),
   prec(nullptr)
{
   MFEM_VERIFY(secondary_dofs.Size() == B.Height(),
               "Wrong number of dofs for elimination!");
   Array<int> lagrange_dofs(secondary_dofs.Size());
   for (int i = 0; i < lagrange_dofs.Size(); ++i)
   {
      lagrange_dofs[i] = i;
   }
   eliminators.Append(new Eliminator(B, lagrange_dofs, primary_dofs,
                                     secondary_dofs));
   projector = new EliminationProjection(hA, eliminators);
   BuildExplicitOperator();
}

EliminationSolver::EliminationSolver(HypreParMatrix& A, SparseMatrix& B,
                                     Array<int>& constraint_rowstarts)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   hA(A),
   krylov(nullptr),
   prec(nullptr)
{
   if (!B.Empty())
   {
      int * I = B.GetI();
      int * J = B.GetJ();
      real_t * data = B.GetData();

      for (int k = 0; k < constraint_rowstarts.Size() - 1; ++k)
      {
         int constraint_size = constraint_rowstarts[k + 1] -
                               constraint_rowstarts[k];
         Array<int> lagrange_dofs(constraint_size);
         Array<int> primary_dofs;
         Array<int> secondary_dofs(constraint_size);
         secondary_dofs = -1;
         // loop through rows, identify one secondary dof for each row
         for (int i = constraint_rowstarts[k]; i < constraint_rowstarts[k + 1]; ++i)
         {
            lagrange_dofs[i - constraint_rowstarts[k]] = i;
            for (int jptr = I[i]; jptr < I[i + 1]; ++jptr)
            {
               int j = J[jptr];
               real_t val = data[jptr];
               if (std::abs(val) > 1.e-12 && secondary_dofs.Find(j) == -1)
               {
                  secondary_dofs[i - constraint_rowstarts[k]] = j;
                  break;
               }
            }
         }
         // loop through rows again, assigning non-secondary dofs as primary
         for (int i = constraint_rowstarts[k]; i < constraint_rowstarts[k + 1]; ++i)
         {
            MFEM_ASSERT(secondary_dofs[i - constraint_rowstarts[k]] >= 0,
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
         eliminators.Append(new Eliminator(B, lagrange_dofs, primary_dofs,
                                           secondary_dofs));
      }
   }
   projector = new EliminationProjection(hA, eliminators);
   BuildExplicitOperator();
}

void EliminationSolver::Mult(const Vector& rhs, Vector& sol) const
{
   if (!prec)
   {
      prec = BuildPreconditioner();
   }
   else
   {
      prec->SetOperator(*h_explicit_operator);
   }
   if (!krylov)
   {
      krylov = BuildKrylov();
      krylov->SetOperator(*h_explicit_operator);
      krylov->SetPreconditioner(*prec);
   }
   krylov->SetMaxIter(max_iter);
   krylov->SetRelTol(rel_tol);
   krylov->SetAbsTol(abs_tol);
   krylov->SetPrintLevel(print_options);

   Vector rtilde(rhs.Size());
   if (constraint_rhs.Size() > 0)
   {
      projector->BuildGTilde(constraint_rhs, rtilde);
   }
   else
   {
      rtilde = 0.0;
   }
   Vector temprhs(rhs);
   hA.Mult(-1.0, rtilde, 1.0, temprhs);

   Vector reducedrhs(rhs.Size());
   projector->MultTranspose(temprhs, reducedrhs);
   Vector reducedsol(rhs.Size());
   reducedsol = 0.0;
   krylov->Mult(reducedrhs, reducedsol);
   final_iter = krylov->GetNumIterations();
   initial_norm = krylov->GetInitialNorm();
   final_norm = krylov->GetFinalNorm();
   converged = krylov->GetConverged();

   projector->Mult(reducedsol, sol);
   projector->RecoverMultiplier(temprhs, sol, multiplier_sol);
   sol += rtilde;
}

void PenaltyConstrainedSolver::Initialize(HypreParMatrix& A, HypreParMatrix& B,
                                          HypreParMatrix& D)
{
   HypreParMatrix * hBTB = RAP(&D, &B);
   // this matrix doesn't get cleanly deleted?
   // (hypre comm pkg)
   penalized_mat = ParAdd(&A, hBTB);
   delete hBTB;
}

PenaltyConstrainedSolver::PenaltyConstrainedSolver(
   HypreParMatrix& A, SparseMatrix& B, real_t penalty_)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   penalty(B.Height()),
   constraintB(B),
   krylov(nullptr),
   prec(nullptr)
{
   int rank, size;
   MPI_Comm_rank(A.GetComm(), &rank);
   MPI_Comm_size(A.GetComm(), &size);

   int constraint_running_total = 0;
   int local_constraints = B.Height();
   MPI_Scan(&local_constraints, &constraint_running_total, 1, MPI_INT,
            MPI_SUM, A.GetComm());
   int global_constraints = 0;
   if (rank == size - 1) { global_constraints = constraint_running_total; }
   MPI_Bcast(&global_constraints, 1, MPI_INT, size - 1, A.GetComm());

   HYPRE_BigInt glob_num_rows = global_constraints;
   HYPRE_BigInt glob_num_cols = A.N();
   HYPRE_BigInt row_starts[2] = { constraint_running_total - local_constraints,
                                  constraint_running_total
                                };
   HYPRE_BigInt col_starts[2] = { A.ColPart()[0], A.ColPart()[1] };
   HypreParMatrix hB(A.GetComm(), glob_num_rows, glob_num_cols,
                     row_starts, col_starts, &B);
   hB.CopyRowStarts();
   hB.CopyColStarts();
   penalty=penalty_;
   SparseMatrix D(penalty);
   HypreParMatrix hD(hB.GetComm(), hB.M(), hB.RowPart(), &D);
   hD.CopyRowStarts();
   hD.CopyColStarts();
   Initialize(A, hB, hD);
}

PenaltyConstrainedSolver::PenaltyConstrainedSolver(
   HypreParMatrix& A, HypreParMatrix& B, real_t penalty_)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   penalty(B.Height()),
   constraintB(B),
   krylov(nullptr),
   prec(nullptr)
{
   penalty=penalty_;
   SparseMatrix D(penalty);
   HypreParMatrix hD(B.GetComm(), B.M(), B.RowPart(), &D);
   hD.CopyRowStarts();
   hD.CopyColStarts();
   Initialize(A, B, hD);
}

PenaltyConstrainedSolver::PenaltyConstrainedSolver(
   HypreParMatrix& A, HypreParMatrix& B, Vector& penalty_)
   :
   ConstrainedSolver(A.GetComm(), A, B),
   penalty(penalty_),
   constraintB(B),
   krylov(nullptr),
   prec(nullptr)
{
   SparseMatrix D(penalty_);
   HypreParMatrix hD(B.GetComm(), B.M(), B.RowPart(), &D);
   hD.CopyRowStarts();
   hD.CopyColStarts();
   Initialize(A, B, hD);
}

PenaltyConstrainedSolver::~PenaltyConstrainedSolver()
{
   delete penalized_mat;
   delete prec;
   delete krylov;
}

void PenaltyConstrainedSolver::Mult(const Vector& b, Vector& x) const
{
   if (!prec)
   {
      prec = BuildPreconditioner();
   }
   else
   {
      prec->SetOperator(*penalized_mat);
   }
   if (!krylov)
   {
      krylov = BuildKrylov();
      krylov->SetOperator(*penalized_mat);
      krylov->SetPreconditioner(*prec);
   }

   // form penalized right-hand side
   Vector penalized_rhs(b);
   if (constraint_rhs.Size() > 0)
   {
      Vector temp_rhs(constraint_rhs.Size());
      SparseMatrix D(penalty);
      D.Mult(constraint_rhs, temp_rhs);
      Vector temp(x.Size());
      constraintB.MultTranspose(temp_rhs, temp);
      penalized_rhs += temp;
   }

   // actually solve
   krylov->SetRelTol(rel_tol);
   krylov->SetAbsTol(abs_tol);
   krylov->SetMaxIter(max_iter);
   krylov->SetPrintLevel(print_options);
   krylov->Mult(penalized_rhs, x);
   final_iter = krylov->GetNumIterations();
   initial_norm = krylov->GetInitialNorm();
   final_norm = krylov->GetFinalNorm();
   converged = krylov->GetConverged();

   constraintB.Mult(x, multiplier_sol);
   if (constraint_rhs.Size() > 0)
   {
      multiplier_sol -= constraint_rhs;
   }
   multiplier_sol *= penalty;
}

#endif

/// because IdentityOperator isn't a Solver
class IdentitySolver : public Solver
{
public:
   IdentitySolver(int size) : Solver(size) { }
   void Mult(const Vector& x, Vector& y) const override { y = x; }
   void SetOperator(const Operator& op) override { }
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

#ifdef MFEM_USE_MPI
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
#endif

SchurConstrainedSolver::SchurConstrainedSolver(Operator& A_, Operator& B_,
                                               Solver& primal_pc_)
   :
   ConstrainedSolver(A_, B_),
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

#ifdef MFEM_USE_MPI
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
#endif

// protected constructor
SchurConstrainedSolver::SchurConstrainedSolver(Operator& A_, Operator& B_)
   :
   ConstrainedSolver(A_, B_),
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

void SchurConstrainedSolver::LagrangeSystemMult(const Vector& x,
                                                Vector& y) const
{
   GMRESSolver * gmres;
#ifdef MFEM_USE_MPI
   if (GetComm() != MPI_COMM_NULL)
   {
      gmres = new GMRESSolver(GetComm());
   }
   else
#endif
   {
      gmres = new GMRESSolver;
   }
   gmres->SetOperator(*block_op);
   gmres->SetRelTol(rel_tol);
   gmres->SetAbsTol(abs_tol);
   gmres->SetMaxIter(max_iter);
   gmres->SetPrintLevel(print_options);
   gmres->SetPreconditioner(
      const_cast<BlockDiagonalPreconditioner&>(*block_pc));

   gmres->Mult(x, y);
   final_iter = gmres->GetNumIterations();
   converged = gmres->GetConverged();
   initial_norm = gmres->GetInitialNorm();
   final_norm = gmres->GetFinalNorm();
   delete gmres;
}

#ifdef MFEM_USE_MPI
SchurConstrainedHypreSolver::SchurConstrainedHypreSolver(MPI_Comm comm,
                                                         HypreParMatrix& hA_,
                                                         HypreParMatrix& hB_,
                                                         Solver * prec,
                                                         int dimension,
                                                         bool reorder)
   :
   SchurConstrainedSolver(comm, hA_, hB_),
   hA(hA_),
   hB(hB_)
{
   if (prec == nullptr)
   {
      auto h_primal_pc = new HypreBoomerAMG(hA);
      h_primal_pc->SetPrintLevel(0);
      if (dimension > 0)
      {
         h_primal_pc->SetSystemsOptions(dimension, reorder);
      }
      primal_pc = h_primal_pc;
   }
   else
   {
      primal_pc = prec;
   }

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
#endif

void ConstrainedSolver::Initialize()
{
   height = A.Height() + B.Height();
   width = A.Width() + B.Height();

   workb.SetSize(A.Height());
   workx.SetSize(A.Height());
   constraint_rhs.SetSize(B.Height());
   constraint_rhs = 0.0;
   multiplier_sol.SetSize(B.Height());
}

#ifdef MFEM_USE_MPI
ConstrainedSolver::ConstrainedSolver(MPI_Comm comm, Operator& A_, Operator& B_)
   :
   IterativeSolver(comm), A(A_), B(B_)
{
   Initialize();
}
#endif

ConstrainedSolver::ConstrainedSolver(Operator& A_, Operator& B_)
   :
   A(A_), B(B_)
{
   Initialize();
}

void ConstrainedSolver::SetConstraintRHS(const Vector& r)
{
   MFEM_VERIFY(r.Size() == multiplier_sol.Size(), "Vector is wrong size!");
   constraint_rhs = r;
}

void ConstrainedSolver::Mult(const Vector& f, Vector &x) const
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

   LagrangeSystemMult(pworkb, pworkx);

   for (int i = 0; i < f.Size(); ++i)
   {
      x(i) = pworkx(i);
   }
   for (int i = 0; i < B.Height(); ++i)
   {
      multiplier_sol(i) = pworkx(f.Size() + i);
   }
}

void ConstrainedSolver::LagrangeSystemMult(const Vector& f_and_r,
                                           Vector& x_and_lambda) const
{
   workb.MakeRef(const_cast<Vector&>(f_and_r), 0);
   workx.MakeRef(x_and_lambda, 0);
   Vector ref_constraint_rhs(f_and_r.GetData() + A.Height(), B.Height());
   constraint_rhs = ref_constraint_rhs;
   Mult(workb, workx);
   Vector ref_constraint_sol(x_and_lambda.GetData() + A.Height(), B.Height());
   GetMultiplierSolution(ref_constraint_sol);
}

/* Helper routine to reduce code duplication - given a node (which MFEM
   sometimes calls a "dof"), this returns what normal people call a dof but
   which MFEM sometimes calls a "vdof" - note that MFEM's naming conventions
   regarding this are not entirely consistent. In parallel, this always
   returns the "truedof" in parallel numbering. */
int CanonicalNodeNumber(FiniteElementSpace& fespace,
                        int node, bool parallel, int d=0)
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      ParFiniteElementSpace* pfespace =
         dynamic_cast<ParFiniteElementSpace*>(&fespace);
      if (pfespace)
      {
         const int vdof = pfespace->DofToVDof(node, d);
         return pfespace->GetLocalTDofNumber(vdof);
      }
      else
      {
         MFEM_ABORT("Asked for parallel form of serial object!");
         return -1;
      }
   }
   else
#endif
   {
      return fespace.DofToVDof(node, d);
   }
}

SparseMatrix * BuildNormalConstraints(FiniteElementSpace& fespace,
                                      Array<int>& constrained_att,
                                      Array<int>& constraint_rowstarts,
                                      bool parallel)
{
   int dim = fespace.GetVDim();

   // dof_constraint maps a dof (column of the constraint matrix) to
   // a block-constraint
   // the indexing is by tdof, but a single tdof uniquely identifies a node
   // so we only store one tdof independent of dimension
   std::map<int, int> dof_bconstraint;
   // constraints[j] is a map from attribute to row number,
   //   the j itself is the index of a block-constraint
   std::vector<std::map<int, int> > constraints;
   int n_bconstraints = 0;
   int n_rows = 0;
   for (int att : constrained_att)
   {
      // identify tdofs on constrained boundary
      std::set<int> constrained_tdofs;
      for (int i = 0; i < fespace.GetNBE(); ++i)
      {
         if (fespace.GetBdrAttribute(i) == att)
         {
            Array<int> nodes;
            // get nodes on boundary (MFEM sometimes calls these dofs, what
            // we call dofs it calls vdofs)
            fespace.GetBdrElementDofs(i, nodes);
            for (auto k : nodes)
            {
               // get the (local) dof number corresponding to
               // the x-coordinate dof for node k
               int tdof = CanonicalNodeNumber(fespace, k, parallel);
               if (tdof >= 0) { constrained_tdofs.insert(tdof); }
            }
         }
      }
      // fill in the maps identifying which constraints (rows) correspond to
      // which tdofs
      for (auto k : constrained_tdofs)
      {
         auto it = dof_bconstraint.find(k);
         if (it == dof_bconstraint.end())
         {
            // build new block constraint
            dof_bconstraint[k] = n_bconstraints++;
            constraints.emplace_back();
            constraints.back()[att] = n_rows++;
         }
         else
         {
            // add tdof to existing block constraint
            constraints[it->second][att] = n_rows++;
         }
      }
   }

   // reorder so block-constraints eliminated together are grouped together in
   // adjacent rows
   {
      std::map<int, int> reorder_rows;
      int new_row = 0;
      constraint_rowstarts.DeleteAll();
      constraint_rowstarts.Append(0);
      for (auto& it : dof_bconstraint)
      {
         int bconstraint_index = it.second;
         bool nconstraint = false;
         for (auto& att_it : constraints[bconstraint_index])
         {
            auto rrit = reorder_rows.find(att_it.second);
            if (rrit == reorder_rows.end())
            {
               nconstraint = true;
               reorder_rows[att_it.second] = new_row++;
            }
         }
         if (nconstraint) { constraint_rowstarts.Append(new_row); }
      }
      MFEM_VERIFY(new_row == n_rows, "Remapping failed!");
      for (auto& constraint_map : constraints)
      {
         for (auto& it : constraint_map)
         {
            it.second = reorder_rows[it.second];
         }
      }
   }

   SparseMatrix * mout = new SparseMatrix(n_rows, fespace.GetTrueVSize());

   // fill in constraint matrix with normal vector information
   Vector nor(dim);
   // how many times we have seen a node (key is truek)
   std::map<int, int> node_visits;
   for (int i = 0; i < fespace.GetNBE(); ++i)
   {
      int att = fespace.GetBdrAttribute(i);
      if (constrained_att.FindSorted(att) != -1)
      {
         ElementTransformation * Tr = fespace.GetBdrElementTransformation(i);
         const FiniteElement * fe = fespace.GetBE(i);
         const IntegrationRule& nodes = fe->GetNodes();

         Array<int> dofs;
         fespace.GetBdrElementDofs(i, dofs);
         MFEM_VERIFY(dofs.Size() == nodes.Size(),
                     "Something wrong in finite element space!");

         for (int j = 0; j < dofs.Size(); ++j)
         {
            Tr->SetIntPoint(&nodes[j]);
            // the normal returned in the next line is scaled by h, which is
            // probably what we want in most applications
            CalcOrtho(Tr->Jacobian(), nor);

            int k = dofs[j];
            int truek = CanonicalNodeNumber(fespace, k, parallel);
            if (truek >= 0)
            {
               auto nv_it = node_visits.find(truek);
               if (nv_it == node_visits.end())
               {
                  node_visits[truek] = 1;
               }
               else
               {
                  node_visits[truek]++;
               }
               int visits = node_visits[truek];
               int bconstraint = dof_bconstraint[truek];
               int row = constraints[bconstraint][att];
               for (int d = 0; d < dim; ++d)
               {
                  int inner_truek = CanonicalNodeNumber(fespace, k,
                                                        parallel, d);
                  if (visits == 1)
                  {
                     mout->Add(row, inner_truek, nor[d]);
                  }
                  else
                  {
                     mout->SetColPtr(row);
                     const real_t pv = mout->SearchRow(inner_truek);
                     const real_t scaling = ((real_t) (visits - 1)) /
                                            ((real_t) visits);
                     // incremental average, based on how many times
                     // this node has been visited
                     mout->Set(row, inner_truek,
                               scaling * pv + (1.0 / visits) * nor[d]);
                  }

               }
            }
         }
      }
   }
   mout->Finalize();

   return mout;
}

#ifdef MFEM_USE_MPI
SparseMatrix * ParBuildNormalConstraints(ParFiniteElementSpace& fespace,
                                         Array<int>& constrained_att,
                                         Array<int>& constraint_rowstarts)
{
   return BuildNormalConstraints(fespace, constrained_att,
                                 constraint_rowstarts, true);
}
#endif

}
