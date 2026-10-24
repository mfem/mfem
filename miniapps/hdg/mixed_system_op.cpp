// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mixed_system_op.hpp"
#include "../../general/tic_toc.hpp"
#include <fstream>

#define USE_DIRECT_SOLVER_HYBRIDIZATION
#define USE_DIRECT_SOLVER_REDUCTION
#define USE_DIRECT_SOLVER_SCHUR

namespace mfem
{
namespace hdg
{
void MixedSystemOperator::SetupLinearSolver(real_t rtol_, real_t atol_,
                                            int max_iters_)
{
   if (mixed_system->GetHybridization())
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
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
   else if (mixed_system->GetReduction())
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
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
      if (pmixed_system)
      {
         schur = new SchurPreconditioner(pmixed_system);
      }
      else
#endif
         schur = new SchurPreconditioner(mixed_system);
      prec.reset(schur);
      prec_str = schur->GetString();
   }

#ifdef MFEM_USE_MPI
   if (pmixed_system)
   {
      MPI_Comm comm = pmixed_system->ParFluxFESpace()->GetComm();
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

MixedSystemOperator::MixedSystemOperator(const Array<int> &ess_flux_tdofs_list_,
                                         MixedSystemForm *mixed_system_,
                                         std::vector<LinearForm*> rhs,
                                         std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs_,
                                         bool btime_u_, bool btime_p_)
   : TimeDependentOperator(0, 0., IMPLICIT),
     ess_flux_tdofs_list(ess_flux_tdofs_list_), mixed_system(mixed_system_),
     coeffs(coeffs_), btime_u(btime_u_), btime_p(btime_p_)
{
   offsets = ConstructOffsets(*mixed_system);
   width = height = offsets.Last();

   if (rhs.size() >= 1 && rhs[0])
   {
      g = rhs[0];
   }
   else
   {
      g = mixed_system->GetFluxRHS();
   }

   if (rhs.size() >= 2 && rhs[1])
   {
      f = rhs[1];
   }
   else
   {
      f = mixed_system->GetPotentialRHS();
   }

   if (mixed_system->GetHybridization())
   {
      trace_space = mixed_system->GetHybridization()->ConstraintFESpace();
      if (rhs.size() >= 3) { h = rhs[2]; }
   }

   if (btime_u || btime_p)
      idtcoeff.reset(new FunctionCoefficient([&](const Vector &) { return idt; }));

   if (btime_u)
   {
      BilinearForm *Mq = const_cast<BilinearForm*>(
                            (const_cast<const MixedSystemForm*>(mixed_system))->GetFluxMassForm());
      const int dim = mixed_system->FluxFESpace()->GetMesh()->Dimension();
      const int vdim = mixed_system->FluxFESpace()->GetVDim();
      const bool dg = (mixed_system->FluxFESpace()->FEColl()->GetRangeType(
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

      Mq0.reset(new BilinearForm(mixed_system->FluxFESpace()));
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
      const int vdim = mixed_system->PotentialFESpace()->GetVDim();

      BilinearForm *Mt = const_cast<BilinearForm*>(
                            (const_cast<const MixedSystemForm*>(mixed_system))->GetPotentialMassForm());

      auto *bfi = new VectorMassIntegrator(*idtcoeff);
      bfi->SetVDim(vdim);
      if (Mt) { Mt->AddDomainIntegrator(bfi); }

      Mt0.reset(new BilinearForm(mixed_system->PotentialFESpace()));
      auto *bfi0 = new VectorMassIntegrator(*idtcoeff);
      bfi0->SetVDim(vdim);
      Mt0->AddDomainIntegrator(bfi0);
   }
}

#ifdef MFEM_USE_MPI
MixedSystemOperator::MixedSystemOperator(const Array<int> &ess_flux_tdofs_list,
                                         ParMixedSystemForm *mixed_system_,
                                         std::vector<ParLinearForm*> rhs,
                                         std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs,
                                         bool bflux_u, bool btime_p)
   : MixedSystemOperator(ess_flux_tdofs_list, (MixedSystemForm*) mixed_system_,
                         std::vector<LinearForm*>(rhs.begin(), rhs.end()), coeffs,
                         bflux_u, btime_p)
{
   pmixed_system = mixed_system_;

   pg = static_cast<ParLinearForm*>(g);
   pf = static_cast<ParLinearForm*>(f);
   ph = static_cast<ParLinearForm*>(h);
}
#endif //MFEM_USE_MPI

MixedSystemOperator::~MixedSystemOperator()
{
}

Array<int> MixedSystemOperator::ConstructOffsets(const MixedSystemForm
                                                 &mixed_system)
{
   if (!mixed_system.GetHybridization())
   {
      return mixed_system.GetOffsets();
   }

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = mixed_system.FluxFESpace()->GetVSize();
   offsets[2] = mixed_system.PotentialFESpace()->GetVSize();
   offsets[3] = mixed_system.GetHybridization()->ConstraintFESpace()->GetVSize();
   offsets.PartialSum();

   return offsets;
}

void MixedSystemOperator::ImplicitSolve(const real_t dt, const Vector &x_v,
                                        Vector &dx_v)
{
#ifdef MFEM_USE_MPI
   const bool verbose = (pmixed_system)?(Mpi::Root()):(true);
#else
   const bool verbose = true;
#endif

   //form the linear system

   rhs.Update(g->GetData(), mixed_system->GetOffsets());
   x.Update(dx_v, mixed_system->GetOffsets());
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
   if (pmixed_system)
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

      mixed_system->Update();

      //assemble the system
#ifdef MFEM_USE_MPI
      if (pmixed_system)
      {
         pmixed_system->Assemble();
      }
      else
#endif //MFEM_USE_MPI
      {
         mixed_system->Assemble();
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
      u_h.MakeRef(mixed_system->FluxFESpace(), x.GetBlock(0), 0);
      Mq0->AddMult(u_h, *g, +1.);
   }

   if (Mt0)
   {
      GridFunction p_h;
      p_h.MakeRef(mixed_system->PotentialFESpace(), x.GetBlock(1), 0);
      Mt0->AddMult(p_h, *f, -1.);
   }
#if 0
   if (Mq0 && Mt0)
   {
      GridFunction u_h, p_h;
      u_h.MakeRef(mixed_system->FluxFESpace(), x.GetBlock(0), 0);
      p_h.MakeRef(mixed_system->PotentialFESpace(), x.GetBlock(1), 0);
      mixed_system->GetFluxDivForm()->AddMultTranspose(p_h, *g, -1.);
      mixed_system->GetFluxDivForm()->AddMult(u_h, *f, +1.);
   }
#endif
   //form the reduced system

   OperatorHandle op;
   Vector X, RHS;
   if (trace_space)
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
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

   mixed_system->FormLinearSystem(ess_flux_tdofs_list, x, rhs,
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

   mixed_system->RecoverFEMSolution(X, rhs, x);

   if (trace_space)
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
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

void MixedSystemOperator::Update()
{
   offsets = ConstructOffsets(*mixed_system);
   width = height = offsets.Last();

   if (Mt0) { Mt0->Update(); }
   if (Mq0) { Mq0->Update(); }

   idt = 0.;
}

MixedSystemOperator::SchurPreconditioner::SchurPreconditioner(
   const MixedSystemForm *mixed_system_)
   : Solver(mixed_system_->Height()), mixed_system(mixed_system_)
{
   Construct();
}

#ifdef MFEM_USE_MPI
MixedSystemOperator::SchurPreconditioner::SchurPreconditioner(
   const ParMixedSystemForm *mixed_system_)
   : Solver(mixed_system_->Height()), mixed_system(mixed_system_),
     pmixed_system(mixed_system_)
{
   ConstructPar();
}
#endif //MFEM_USE_MPI

void MixedSystemOperator::SchurPreconditioner::Mult(const Vector &x_,
                                                    Vector &y_) const
{
   schur_prec->Mult(x_,y_);
}

void MixedSystemOperator::SchurPreconditioner::Construct()

{
   const Array<int> &block_offsets = mixed_system->GetTrueOffsets();

   // Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     temperature Schur Complement

   const bool pa = (mixed_system->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   const BilinearForm *Mq = mixed_system->GetFluxMassForm();
   const MixedBilinearForm *B = mixed_system->GetFluxDivForm();
   const BilinearForm *Mt = mixed_system->GetPotentialMassForm();

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

   schur_prec.reset(new BlockDiagonalPreconditioner(block_offsets));
   schur_prec->owns_blocks = true;
   schur_prec->SetDiagonalBlock(0, invM);
   schur_prec->SetDiagonalBlock(1, invS);
}

#ifdef MFEM_USE_MPI
void MixedSystemOperator::SchurPreconditioner::ConstructPar()
{
   const Array<int> &block_offsets = pmixed_system->GetTrueOffsets();

   // Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     temperature Schur Complement

   const bool pa = (mixed_system->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   const ParBilinearForm *Mq = pmixed_system->GetParFluxMassForm();
   const ParMixedBilinearForm *B = pmixed_system->GetParFluxDivForm();
   const ParBilinearForm *Mt = pmixed_system->GetParPotentialMassForm();

   const int a_tsize = block_offsets[1] - block_offsets[0];
   const int d_tsize = block_offsets[2] - block_offsets[1];
   Vector Md(a_tsize);
   schur_prec.reset(new BlockDiagonalPreconditioner(block_offsets));
   schur_prec->owns_blocks = true;
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

   schur_prec->SetDiagonalBlock(0, invM);
   schur_prec->SetDiagonalBlock(1, invS);
}
#endif
} // namespace hdg
} // namespace mfem
