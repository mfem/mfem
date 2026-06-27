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

#include "darcyop.hpp"
#include "../../general/tic_toc.hpp"
#include <fstream>

#define USE_DIRECT_SOLVER_HYBRIDIZATION
#define USE_DIRECT_SOLVER_REDUCTION
#define USE_DIRECT_SOLVER_SCHUR

namespace mfem
{
namespace hdg
{
void DarcyOperator::SetupLinearSolver(real_t rtol_, real_t atol_,
                                      int max_iters_)
{
   if (darcy->GetHybridization())
   {
#ifdef MFEM_USE_MPI
      if (pdarcy)
      {
         auto *amg = new HypreBoomerAMG();
         amg->SetAdvectiveOptions();
         prec.reset(amg);
         prec_str = "HypreAMG";
      }
      else
#endif
      {
#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER_HYBRIDIZATION)
         prec.reset(new GSSmoother());
         prec_str = "GS";
#else
         prec.reset(new UMFPackSolver());
         prec_str = "UMFPack";
#endif
      }
   }
   else if (darcy->GetReduction())
   {
#ifdef MFEM_USE_MPI
      if (pdarcy)
      {
         auto *amg = new HypreBoomerAMG();
         amg->SetAdvectiveOptions();
         prec.reset(amg);
         prec_str = "HypreAMG";
      }
      else
#endif
      {
#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER_REDUCTION)
         prec.reset(new GSSmoother());
         prec_str = "GS";
#else
         prec.reset(new UMFPackSolver());
         prec_str = "UMFPack";
#endif
      }
   }
   else
   {
      SchurPreconditioner *schur;
#ifdef MFEM_USE_MPI
      if (pdarcy)
      {
         schur = new SchurPreconditioner(pdarcy);
      }
      else
#endif
         schur = new SchurPreconditioner(darcy);
      prec.reset(schur);
      prec_str = schur->GetString();
   }

#ifdef MFEM_USE_MPI
   if (pdarcy)
   {
      MPI_Comm comm = pdarcy->ParFluxFESpace()->GetComm();
      solver.reset(new GMRESSolver(comm));
   }
   else
#endif
      solver.reset(new GMRESSolver());
   solver_str = "GMRES";
   solver->SetAbsTol(atol_);
   solver->SetRelTol(rtol_);
   solver->SetMaxIter(max_iters_);
   if (prec) { solver->SetPreconditioner(*prec); }
   solver->SetPrintLevel((btime_u || btime_p)?0:1);
   solver->iterative_mode = true;
}

DarcyOperator::DarcyOperator(const Array<int> &ess_flux_tdofs_list_,
                             DarcyForm *darcy_,
                             std::vector<LinearForm*> rhs,
                             std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs_,
                             bool btime_u_, bool btime_p_)
   : TimeDependentOperator(0, 0., IMPLICIT),
     ess_flux_tdofs_list(ess_flux_tdofs_list_), darcy(darcy_),
     coeffs(coeffs_), btime_u(btime_u_), btime_p(btime_p_)
{
   offsets = ConstructOffsets(*darcy);
   width = height = offsets.Last();

   if (rhs.size() >= 1 && rhs[0])
   {
      g = rhs[0];
   }
   else
   {
      g = darcy->GetFluxRHS();
   }

   if (rhs.size() >= 2 && rhs[1])
   {
      f = rhs[1];
   }
   else
   {
      f = darcy->GetPotentialRHS();
   }

   if (darcy->GetHybridization())
   {
      trace_space = darcy->GetHybridization()->ConstraintFESpace();
      if (rhs.size() >= 3) { h = rhs[2]; }
   }

   if (btime_u || btime_p)
      idtcoeff.reset(new FunctionCoefficient([&](const Vector &) { return idt; }));

   if (btime_u)
   {
      BilinearForm *Mq = const_cast<BilinearForm*>(
                            (const_cast<const DarcyForm*>(darcy))->GetFluxMassForm());
      const int dim = darcy->FluxFESpace()->GetMesh()->Dimension();
      const int vdim = darcy->FluxFESpace()->GetVDim();
      const bool dg = (darcy->FluxFESpace()->FEColl()->GetRangeType(
                          dim) == FiniteElement::SCALAR);

      if (dg)
      {
         auto *bfi = new VectorMassIntegrator(*idtcoeff);
         bfi->SetVDim(vdim);
         if (Mq) { Mq->AddDomainIntegrator(bfi); }
      }
      else
      {
         MFEM_VERIFY(vdim == 1, "Unsupported case");
         auto *bfi = new VectorFEMassIntegrator(*idtcoeff);
         if (Mq) { Mq->AddDomainIntegrator(bfi); }
      }

      Mq0.reset(new BilinearForm(darcy->FluxFESpace()));
      if (dg)
      {
         auto *bfi = new VectorMassIntegrator(*idtcoeff);
         bfi->SetVDim(vdim);
         Mq0->AddDomainIntegrator(bfi);
      }
      else
      {
         MFEM_VERIFY(vdim == 1, "Unsupported case");
         Mq0->AddDomainIntegrator(new VectorFEMassIntegrator(*idtcoeff));
      }
   }

   if (btime_p)
   {
      const int vdim = darcy->PotentialFESpace()->GetVDim();

      BilinearForm *Mt = const_cast<BilinearForm*>(
                            (const_cast<const DarcyForm*>(darcy))->GetPotentialMassForm());

      auto *bfi = new VectorMassIntegrator(*idtcoeff);
      bfi->SetVDim(vdim);
      if (Mt) { Mt->AddDomainIntegrator(bfi); }

      Mt0.reset(new BilinearForm(darcy->PotentialFESpace()));
      auto *bfi0 = new VectorMassIntegrator(*idtcoeff);
      bfi0->SetVDim(vdim);
      Mt0->AddDomainIntegrator(bfi0);
   }
}

#ifdef MFEM_USE_MPI
DarcyOperator::DarcyOperator(const Array<int> &ess_flux_tdofs_list,
                             ParDarcyForm *darcy_,
                             std::vector<ParLinearForm*> rhs,
                             std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs,
                             bool bflux_u, bool btime_p)
   : DarcyOperator(ess_flux_tdofs_list, (DarcyForm*) darcy_,
                   std::vector<LinearForm*>(rhs.begin(), rhs.end()), coeffs,
                   bflux_u, btime_p)
{
   pdarcy = darcy_;

   pg = static_cast<ParLinearForm*>(g);
   pf = static_cast<ParLinearForm*>(f);
   ph = static_cast<ParLinearForm*>(h);
}
#endif //MFEM_USE_MPI

DarcyOperator::~DarcyOperator()
{
}

Array<int> DarcyOperator::ConstructOffsets(const DarcyForm &darcy)
{
   if (!darcy.GetHybridization())
   {
      return darcy.GetOffsets();
   }

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = darcy.FluxFESpace()->GetVSize();
   offsets[2] = darcy.PotentialFESpace()->GetVSize();
   offsets[3] = darcy.GetHybridization()->ConstraintFESpace()->GetVSize();
   offsets.PartialSum();

   return offsets;
}

void DarcyOperator::ImplicitSolve(const real_t dt, const Vector &x_v,
                                  Vector &dx_v)
{
#ifdef MFEM_USE_MPI
   const bool verbose = (pdarcy)?(Mpi::Root()):(true);
#else
   const bool verbose = true;
#endif

   //form the linear system

   rhs.Update(g->GetData(), darcy->GetOffsets());
   x.Update(dx_v, darcy->GetOffsets());
   dx_v = x_v;

   //set time

   for (const auto &coeff : coeffs)
   {
      if (coeff.index() == 0)
      {
         std::get<0>(coeff)->SetTime(t);
      }
      else if (coeff.index() == 1)
      {
         std::get<1>(coeff)->SetTime(t);
      }
   }

   //assemble rhs

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

#ifdef MFEM_USE_MPI
   if (pdarcy)
   {
      pg->Assemble();
      pf->Assemble();
      if (ph) { ph->Assemble(); }
   }
   else
#endif //MFEM_USE_MPI
   {
      g->Assemble();
      f->Assemble();
      if (h) { h->Assemble(); }
   }

   //check if the operator has to be reassembled

   bool reassemble = (idt != 1./dt);

   if (reassemble)
   {
      idt = 1./dt;

      //reset the operator

      darcy->Update();

      //assemble the system
#ifdef MFEM_USE_MPI
      if (pdarcy)
      {
         pdarcy->Assemble();
      }
      else
#endif //MFEM_USE_MPI
      {
         darcy->Assemble();
      }

      if (Mq0)
      {
         Mq0->Update();
         Mq0->Assemble();
         //Mq0->Finalize();
      }
      if (Mt0)
      {
         Mt0->Update();
         Mt0->Assemble();
         //Mt0->Finalize();
      }
   }

   if (Mq0)
   {
      GridFunction u_h;
      u_h.MakeRef(darcy->FluxFESpace(), x.GetBlock(0), 0);
      Mq0->AddMult(u_h, *g, +1.);
   }

   if (Mt0)
   {
      GridFunction p_h;
      p_h.MakeRef(darcy->PotentialFESpace(), x.GetBlock(1), 0);
      Mt0->AddMult(p_h, *f, -1.);
   }
#if 0
   if (Mq0 && Mt0)
   {
      GridFunction u_h, p_h;
      u_h.MakeRef(darcy->FluxFESpace(), x.GetBlock(0), 0);
      p_h.MakeRef(darcy->PotentialFESpace(), x.GetBlock(1), 0);
      darcy->GetFluxDivForm()->AddMultTranspose(p_h, *g, -1.);
      darcy->GetFluxDivForm()->AddMult(u_h, *f, +1.);
   }
#endif
   //form the reduced system

   OperatorHandle op;
   Vector X, RHS;
   if (trace_space)
   {
#ifdef MFEM_USE_MPI
      if (pdarcy)
      {
         if (ph)
         {
            RHS.SetSize(trace_space->GetTrueVSize());
            ph->ParallelAssemble(RHS);
         }
      }
      else
#endif
      {
         if (trace_space->GetMesh()->Nonconforming())
         {
            auto *cP = trace_space->GetConformingProlongation();
            RHS.SetSize(cP->Width());
            cP->MultTranspose(*h, RHS);

            auto *cR = trace_space->GetConformingRestriction();
            X.SetSize(cR->Height());
            const Vector x_r(dx_v, offsets[2], trace_space->GetVSize());
            cR->Mult(x_r, X);
         }
         else
         {
            X.MakeRef(dx_v, offsets[2], trace_space->GetVSize());
            if (h) { RHS.MakeRef(*h, 0, trace_space->GetVSize()); }
         }
      }
   }

   darcy->FormLinearSystem(ess_flux_tdofs_list, x, rhs,
                           op, X, RHS, true);


   chrono.Stop();
   if (verbose) { std::cout << "Assembly took " << chrono.RealTime() << "s.\n"; }

   if (reassemble)
   {
      // 10. Construct the preconditioner and solver

      chrono.Clear();
      chrono.Start();

      SetupLinearSolver(rtol, atol, max_iters);
      solver->SetOperator(*op);

      chrono.Stop();
      if (verbose) { std::cout << "Preconditioner took " << chrono.RealTime() << "s.\n"; }
   }

   // 11. Solve the linear system with GMRES.
   //     Check the norm of the unpreconditioned residual.

   chrono.Clear();
   chrono.Start();

   solver->Mult(RHS, X);

   darcy->RecoverFEMSolution(X, rhs, x);

   if (trace_space)
   {
#ifdef MFEM_USE_MPI
      if (pdarcy)
      {
         Vector x_r(dx_v, offsets[2], trace_space->GetVSize());
         trace_space->GetProlongationMatrix()->Mult(X, x_r);
      }
      else
#endif
         if (trace_space->GetMesh()->Nonconforming())
         {
            Vector x_r(dx_v, offsets[2], trace_space->GetVSize());
            auto *cP = trace_space->GetConformingProlongation();
            cP->Mult(X, x_r);
         }
   }

   chrono.Stop();

   if (verbose)
   {
      std::cout << solver_str;
      if (!prec_str.empty()) { std::cout << "+" << prec_str; }
      if (solver->GetConverged())
      {
         std::cout << " converged in " << solver->GetNumIterations()
                   << " iterations with a residual norm of " << solver->GetFinalNorm()
                   << ".\n";
      }
      else
      {
         std::cout << " did not converge in " << solver->GetNumIterations()
                   << " iterations. Residual norm is " << solver->GetFinalNorm()
                   << ".\n";
      }
      std::cout << "Solver took " << chrono.RealTime() << "s.\n";
   }

   dx_v -= x_v;
   dx_v *= idt;
}

void DarcyOperator::Update()
{
   offsets = ConstructOffsets(*darcy);
   width = height = offsets.Last();

   if (Mt0) { Mt0->Update(); }
   if (Mq0) { Mq0->Update(); }

   idt = 0.;
}

DarcyOperator::SchurPreconditioner::SchurPreconditioner(const DarcyForm *darcy_)
   : Solver(darcy_->Height()), darcy(darcy_)
{
   Construct();
}

#ifdef MFEM_USE_MPI
DarcyOperator::SchurPreconditioner::SchurPreconditioner(
   const ParDarcyForm *darcy_)
   : Solver(darcy_->Height()), darcy(darcy_), pdarcy(darcy_)
{
   ConstructPar();
}
#endif //MFEM_USE_MPI

void DarcyOperator::SchurPreconditioner::Mult(const Vector &x_,
                                              Vector &y_) const
{
   darcyPrec->Mult(x_,y_);
}

void DarcyOperator::SchurPreconditioner::Construct()

{
   const Array<int> &block_offsets = darcy->GetTrueOffsets();

   // Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     temperature Schur Complement

   const bool pa = (darcy->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   const BilinearForm *Mq = darcy->GetFluxMassForm();
   const MixedBilinearForm *B = darcy->GetFluxDivForm();
   const BilinearForm *Mt = darcy->GetPotentialMassForm();

   const int a_tsize = block_offsets[1] - block_offsets[0];
   const int d_tsize = block_offsets[2] - block_offsets[1];
   Vector Md(a_tsize);
   Solver *invM, *invS;

   if (pa)
   {
      Mq->AssembleDiagonal(Md);
      auto Md_host = Md.HostRead();
      Vector invMd(Md.Size());
      for (int i=0; i < invMd.Size(); ++i)
      {
         invMd(i) = 1.0 / Md_host[i];
      }

      Vector BMBt_diag(d_tsize);
      B->AssembleDiagonal_ADAt(invMd, BMBt_diag);

      Array<int> ess_tdof_list;  // empty

      invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
      invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
      prec_str = "OperJacobi";
   }
   else
   {
      // get diagonal
      const SparseMatrix *Mqm;
      if (Mq)
      {
         Mqm = &Mq->SpMat();
      }
      else
      {
         MFEM_ABORT("No flux diagonal!");
      }

      Mqm->GetDiag(Md);
      invM = new DSmoother(*Mqm);

      Md.HostReadWrite();

      const SparseMatrix &Bm(B->SpMat());
      SparseMatrix *MinvBt = Transpose(Bm);

      for (int i = 0; i < Md.Size(); i++)
      {
         MinvBt->ScaleRow(i, 1./Md(i));
      }

      S.reset(mfem::Mult(Bm, *MinvBt));
      delete MinvBt;

      if (Mt)
      {
         S.reset(Add(Mt->SpMat(), *S));
      }

#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER_SCHUR)
      invS = new GSSmoother(*S);
      prec_str = "GS";
#else
      invS = new UMFPackSolver(*S);
      prec_str = "UMFPack";
#endif
   }

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   darcyPrec.reset(new BlockDiagonalPreconditioner(block_offsets));
   darcyPrec->owns_blocks = true;
   darcyPrec->SetDiagonalBlock(0, invM);
   darcyPrec->SetDiagonalBlock(1, invS);
}

#ifdef MFEM_USE_MPI
void DarcyOperator::SchurPreconditioner::ConstructPar()
{
   const Array<int> &block_offsets = pdarcy->GetTrueOffsets();

   // Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     temperature Schur Complement

   const bool pa = (darcy->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   const ParBilinearForm *Mq = pdarcy->GetParFluxMassForm();
   const ParMixedBilinearForm *B = pdarcy->GetParFluxDivForm();
   const ParBilinearForm *Mt = pdarcy->GetParPotentialMassForm();

   const int a_tsize = block_offsets[1] - block_offsets[0];
   const int d_tsize = block_offsets[2] - block_offsets[1];
   Vector Md(a_tsize);
   darcyPrec.reset(new BlockDiagonalPreconditioner(block_offsets));
   darcyPrec->owns_blocks = true;
   Solver *invM, *invS;

   if (pa)
   {
      Mq->AssembleDiagonal(Md);
      auto Md_host = Md.HostRead();
      Vector invMd(Md.Size());
      for (int i=0; i < invMd.Size(); ++i)
      {
         invMd(i) = 1.0 / Md_host[i];
      }

      Vector BMBt_diag(d_tsize);
      B->AssembleDiagonal_ADAt(invMd, BMBt_diag);

      Array<int> ess_tdof_list;  // empty

      invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
      invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
      prec_str = "OperJacobi";
   }
   else
   {
      // get diagonal
      const HypreParMatrix *Mqm;
      if (Mq)
      {
         Mqm = const_cast<ParBilinearForm*>(Mq)->ParallelAssembleInternalMatrix();
      }
      else
      {
         MFEM_ABORT("No flux diagonal!");
      }

      Mqm->GetDiag(Md);
      invM = new HypreDiagScale(*Mqm);

      Md.HostReadWrite();

      const HypreParMatrix *Bm;
      if (B)
      {
         Bm = const_cast<ParMixedBilinearForm*>(B)->ParallelAssembleInternalMatrix();
      }
      else
      {
         MFEM_ABORT("No flux divergence!");
      }
      HypreParMatrix *MinvBt = Bm->Transpose();
      MinvBt->InvScaleRows(Md);

      hS.reset(mfem::ParMult(Bm, MinvBt, true));
      delete MinvBt;

      if (Mt)
      {
         const HypreParMatrix *Mtm =
            const_cast<ParBilinearForm*>(Mt)->ParallelAssembleInternalMatrix();
         hS.reset(ParAdd(Mtm, hS.get()));
      }

      {
         auto *amg = new HypreBoomerAMG(*hS);
         amg->SetAdvectiveOptions();
         amg->SetPrintLevel(0);
         invS = amg;
         prec_str = "HypreAMG";
      }
   }

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   darcyPrec->SetDiagonalBlock(0, invM);
   darcyPrec->SetDiagonalBlock(1, invS);
}
#endif
} // namespace hdg
} // namespace mfem
