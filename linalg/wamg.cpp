// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#include <cassert>
#include <iomanip>
#include <iostream>
using namespace std;

#include "wamg.hpp"
#include "wavelets.hpp"

#define MFEM_DEBUG_COLOR 82
#include "../general/debug.hpp"

#define MFEM_NVTX_COLOR Cornflower
#include "../general/nvtx.hpp"

#include "../general/forall.hpp"
#include "../general/socketstream.hpp"

namespace mfem
{

////////////////////////////////////////////////////////////////////////////////
WaveletRecursiveLevel::WaveletRecursiveLevel(Wavelet::Type &wavelet,
                                             const bool &lowpass,
                                             const Operator &A):
   W(Wavelet::New(wavelet, A.Height(), lowpass)),
   Wt(new TransposeOperator(W)),
   WAWt(new RAPOperator(*Wt,A,*Wt))
{
   MFEM_NVTX;
   MFEM_VERIFY((wavelet == Wavelet::HAAR && lowpass) ||
               (wavelet == Wavelet::DAUBECHIES && lowpass),
               "Wavelet spec error!");
   dbg("A: %dx%d", A.Height(), A.Width());
   dbg("W: %dx%d", W->Height(), W->Width());
   dbg("WAWt: %dx%d", WAWt->Height(), WAWt->Width());
   MFEM_VERIFY(A.Height() == A.Width(), "Operator should be square!");
   MFEM_VERIFY(WAWt->Height() == WAWt->Width(), "WAWt operator should be square!");

   MFEM_VERIFY(W->Width() == A.Width(),
               "Dimensions error: " << WAWt->Height() <<"x"<< WAWt->Width());
}

WaveletRecursiveLevel::~WaveletRecursiveLevel()
{
   delete W;
   delete Wt;
   delete WAWt;
}

////////////////////////////////////////////////////////////////////////////////
WaveletRecursiveLevelFA::WaveletRecursiveLevelFA(FiniteElementSpace &fes,
                                                 Wavelet::Type &wavelet,
                                                 const OperatorHandle &Ah)
{
   MFEM_NVTX;
   MFEM_VERIFY((wavelet == Wavelet::HAAR) ||
               (wavelet == Wavelet::DAUBECHIES), "Wavelet argument error!");

   switch (wavelet)
   {
      case Wavelet::HAAR:
         W = new HaarWavelet(Ah->Height(), true); break;
      case Wavelet::DAUBECHIES:
         W = new DaubechiesWavelet(Ah->Height(), true); break;
      default: assert(false);
   }
   Wt = new TransposeOperator(W); // the prolongator

   // Get the forward and backward matrices
   M = mfem::Transpose(*W->GetMatrix()); // Prepare for the Rt of mfem::RAP
   tM = W->GetTransposedMatrix();

#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
   assert(pfes);

   OperatorHandle H(Operator::Hypre_ParCSR),
                  tH(Operator::Hypre_ParCSR);

   // Compute the row and colum offsets
   HYPRE_BigInt locals[2];
   Array<HYPRE_BigInt> row_offsets;
   Array<HYPRE_BigInt> col_offsets;
   int glob_num_rows, glob_num_cols;
   assert(HYPRE_AssumedPartitionCheck());
   Array<HYPRE_BigInt> *offsets[2] = { &row_offsets, &col_offsets };

   auto CreateRectangularHypreMatrix = [&] (OperatorHandle &H,
                                            SparseMatrix *A)
   {
      NVTX("CreateRectangularHypreMatrix");
      locals[0] = A->Height();
      locals[1] = A->Width();
      pfes->GetParMesh()->GenerateOffsets(2, locals, offsets);
      glob_num_rows = row_offsets[row_offsets.Size()-1];
      glob_num_cols = col_offsets[col_offsets.Size()-1];
      dbg("glob_num_rows:%d glob_num_cols:%d", glob_num_rows, glob_num_cols);
      H.MakeRectangularBlockDiag(pfes->GetComm(),
                                 glob_num_rows, glob_num_cols,
                                 row_offsets, col_offsets, A);
   };

   CreateRectangularHypreMatrix(H, M);
   H.SetOperatorOwner(false);
   assert(H.Ptr());
   assert(H.Is<HypreParMatrix>());

   CreateRectangularHypreMatrix(tH, tM);
   tH.SetOperatorOwner(false);
   assert(tH.Ptr());
   assert(tH.Is<HypreParMatrix>());

   const HypreParMatrix *A = Ah.As<HypreParMatrix>();
   {
      NVTX("mfem::RAP");
      assert(A);
      assert(*H.As<HypreParMatrix>());
      assert(*tH.As<HypreParMatrix>());
      MAMt.Reset(mfem::RAP(H.As<HypreParMatrix>(),
                           A,
                           tH.As<HypreParMatrix>()),
                 false);
   }
#else
   assert(false);
   MFEM_CONTRACT_VAR(fes);
   const SparseMatrix *A = Ah.As<SparseMatrix>();
   MAMt.Reset(new RAPOperator(*Wt,*A,*Wt), false);
#endif // MFEM_USE_MPI

   MFEM_VERIFY(A->Height() == A->Width(), "A should be square!");
   MFEM_VERIFY(MAMt->Height() == MAMt->Width(), "MAMt should be square!");
   MFEM_VERIFY(W->Width() == A->Width(),
               "Dimensions error: " << MAMt->Height() <<"x"<< MAMt->Width());
}

OperatorHandle WaveletRecursiveLevelFA::OpHandle()
{
   return MAMt;
   //return OperatorHandle(MAMt, false);
}

Operator *WaveletRecursiveLevelFA::Prolongator() { return Wt; }

WaveletRecursiveLevelFA::~WaveletRecursiveLevelFA()
{
   delete W;
   delete Wt;
   //delete MAMt;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief The IdentitySolver class
class IdentitySolver: public Solver
{
public:
   IdentitySolver(int s): Solver(s) { }
   void SetOperator(const Operator&) { }
   virtual void Mult(const Vector &x, Vector &y) const { y = x; }
   virtual void MultTranspose(const Vector &x, Vector &y) const  { y = x; }
};

////////////////////////////////////////////////////////////////////////////////
/// \brief WAMGR solver with DIAGONAL
WAMGRSolver::WAMGRSolver(FiniteElementSpace &fes,
                         Wavelet::Type wavelet,
                         const bool lowpass,
                         wargs_t args,
                         const bool to_full) :
   Multigrid()
{
   MFEM_NVTX;
   const int n = args.op_h->Height();
   const int max_depth = args.mg_depth;
   const int max_ndofs = args.mg_ndofs;
   dbg("COARSE WAMGRSolver n:%d, %s wavelets max_levels:%d",
       n, Wavelet::GetType(wavelet).c_str(),  max_depth);

   Array<Solver*> Smoothers;
   Array<Operator*> Operators;
   Array<Operator*> Prolongators;

   const bool round_up = true;
   int m = (round_up ? (n+(n%2&1)) : (n-(n%2&1))) >> 1;
   dbg("n:%d m:%d",n,m);

   OperatorHandle Op_h;
   Op_h.Reset(args.op_h.Ptr(),false);

   for (int depth = 1; true; depth+=1)
   {
      NVTX("DEPTH %d",depth);
      dbg("\033[31mDEPTH:%d",depth);
      WaveletLevel *L = nullptr;
      if (!to_full)
      {
         L = new WaveletRecursiveLevel(wavelet, lowpass, *Op_h);
      }
      else
      {
         assert(lowpass);
         L = new WaveletRecursiveLevelFA(fes, wavelet, Op_h);
      }
      assert(L);
      assert(L->OpHandle().Ptr());
      assert(L->OpHandle()->Height() == m && L->OpHandle()->Width() == m);

      dbg("Smoother: (%dx%d)", Op_h->Height(), Op_h->Width());
      Solver* smoother = nullptr;
      if (depth==1)
      {
         NVTX("Smoother @ 1");
         if (!args.smoother_order)
         {
            dbg("Jacobi smoother");
            smoother = new OperatorJacobiSmoother(args.diag,
                                                  args.ess_tdof_list);
         }
         else
         {
            dbg("Chebyshev smoother");
            OperatorChebyshevSmoother *chebyshev_smoother =
               new OperatorChebyshevSmoother(*Op_h.Ptr(),
                                             args.diag,
                                             args.ess_tdof_list,
                                             args.smoother_order
#ifdef MFEM_USE_MPI
                                             ,MPI_COMM_WORLD
#endif // MFEM_USE_MPI
                                            );
            smoother = chebyshev_smoother;
         }
      }
      else
      {
         NVTX("Smoother @ %d",depth);
         Vector diag(Op_h->Width());
         Op_h->AssembleDiagonal(diag);
         if (!args.smoother_order)
         {
            dbg("Jacobi smoother");
            smoother = new OperatorJacobiSmoother(diag, Array<int>());
         }
         else
         {
            dbg("Chebyshev smoother");
            OperatorChebyshevSmoother *chebyshev_smoother =
               new OperatorChebyshevSmoother(*Op_h.Ptr(),
                                             diag,
                                             Array<int>(),
                                             args.smoother_order
#ifdef MFEM_USE_MPI
                                             ,MPI_COMM_WORLD
#endif // MFEM_USE_MPI
                                            );
            smoother = chebyshev_smoother;
         }
      }
      Smoothers.Append(smoother);
      Operators.Append(Op_h.Ptr());

      Operator *P = L->Prolongator();
      Prolongators.Append(P);

      OperatorHandle LOp_h;
      LOp_h.Reset(L->OpHandle().Ptr(),false);
      dbg("max_ndofs set to %d", max_ndofs);
      const bool coarse_enough = m < max_ndofs; // stop when too small
      const bool reached_max_levels = depth >= max_depth;
      const bool local_coarse_enough_OR_reached_max_levels =
         coarse_enough || reached_max_levels;
      bool coarse_enough_OR_reached_max_levels;

#ifdef MFEM_USE_MPI
      {
         NVTX("MPI_Allreduce");
         MPI_Allreduce(&local_coarse_enough_OR_reached_max_levels,
                       &coarse_enough_OR_reached_max_levels,
                       1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
      }
#else
      coarse_enough_OR_reached_max_levels =
         local_coarse_enough_OR_reached_max_levels;
#endif // MFEM_USE_MPI

      if (coarse_enough_OR_reached_max_levels)
      {
         NVTX("coarse_enough_OR_reached_max_levels");
         if (to_full)
         {
            dbg("Sparse & Full level 0: %dx%d", LOp_h->Height(), LOp_h->Width());
            CGSolver* wcg =  new CGSolver(
#ifdef MFEM_USE_MPI
               MPI_COMM_WORLD
#endif
            );
            wcg->SetMaxIter(args.max_iter);
            wcg->SetRelTol(1e-8);
            wcg->SetAbsTol(1e-8);
            wcg->SetPrintLevel(args.print_level);
            wcg->SetOperator(*LOp_h);
            wcg->iterative_mode = false;

            Smoothers.Append(wcg);
            Operators.Append(LOp_h.Ptr());
         }
         else
         {
            const int depth = Prolongators.Size();
            dbg("Coarse solver level 0 (%dx%d), depth:%d",
                LOp_h->Height(), LOp_h->Width(), depth);
            CGSolver *wcg = new CGSolver(
#ifdef MFEM_USE_MPI
               MPI_COMM_WORLD
#endif // MFEM_USE_MPI
            );
            wcg->SetMaxIter(args.max_iter);
            wcg->SetRelTol(1e-8);
            wcg->SetAbsTol(1e-8);
            wcg->SetPrintLevel(args.print_level);
            wcg->SetOperator(*LOp_h);
            wcg->iterative_mode = false;

            Smoothers.Append(wcg);
            Operators.Append(LOp_h.Ptr());
         }
         break;
      }

      Op_h.Reset(LOp_h.Ptr(), false);
      m = (round_up ? (m+(m%2&1)) : (m-(m%2&1))) >> 1;
   }

   const int depth = Prolongators.Size();

   // Coarse solver
   NVTX("AddLevels");
   AddLevel(Operators[depth], Smoothers[depth], true, true);

   for (int level = 1; level <= depth; level+=1)
   {
      NVTX("level %d", level);
      const int idx = depth-level;
      assert(idx>=0);
      const int Hop = Operators[idx]->Height();
      const int Wop = Operators[idx]->Width();
      assert(Smoothers[idx]->Height() == Hop);
      assert(Smoothers[idx]->Width() == Wop);
      dbg("@%d %dx%d",level,Hop,Wop);
      AddLevel(Operators[idx], Smoothers[idx], false, true);
      // Add the prolongation operator associated with this level
      prolongations.Append(Prolongators[idx]);
      ownedProlongations.Append(false);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief WAMG solver
WAMG::WAMG(FiniteElementSpace &fes,
           Wavelet::Type wavelet, wargs_t args):
   wavelet_solver(fes, wavelet, lowpass, args, to_full)
{
   dbg();
}

void WAMG::Mult(const Vector &x, Vector &y) const
{
   wavelet_solver.Mult(x,y);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief faWAMG solver
faWAMG::faWAMG(FiniteElementSpace &fes,
               Wavelet::Type wavelet,
               wargs_t args):
   wavelet_solver(fes, wavelet, lowpass, args, to_full)
{
   dbg();
}

void faWAMG::Mult(const Vector &x, Vector &y) const {wavelet_solver.Mult(x,y);}

} // namespace mfem

