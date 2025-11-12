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

#include "div_free_solver.hpp"

using namespace std;

namespace mfem::blocksolvers
{

static HypreParMatrix* TwoStepsRAP(const HypreParMatrix *Rt,
                                   const HypreParMatrix *A,
                                   const HypreParMatrix *P)
{
   OperatorPtr R(Rt->Transpose());
   OperatorPtr RA(ParMult(R.As<HypreParMatrix>(), A));
   return ParMult(RA.As<HypreParMatrix>(), P, true);
}

void GetRowColumnsRef(const SparseMatrix& A, int row, Array<int>& cols)
{
   cols.MakeRef(const_cast<int*>(A.GetRowColumns(row)), A.RowSize(row));
}

SparseMatrix ElemToDof(const ParFiniteElementSpace& fes)
{
   int* I = new int[fes.GetNE()+1];
   copy_n(fes.GetElementToDofTable().GetI(), fes.GetNE()+1, I);
   Array<int> J(new int[I[fes.GetNE()]], I[fes.GetNE()]);
   copy_n(fes.GetElementToDofTable().GetJ(), J.Size(), J.begin());
   fes.AdjustVDofs(J);
   real_t* D = new real_t[J.Size()];
   fill_n(D, J.Size(), 1.0);
   return SparseMatrix(I, J, D, fes.GetNE(), fes.GetVSize());
}

DFSSpaces::DFSSpaces(int order, int num_refine, ParMesh *mesh,
                     const Array<int>& ess_attr, const DFSParameters& param)
   : hdiv_fec_(order, mesh->Dimension()), l2_fec_(order, mesh->Dimension()),
     l2_0_fec_(0, mesh->Dimension()), ess_bdr_attr_(ess_attr), level_(0)
{
   if (mesh->GetNE() > 0)
   {
      if (mesh->GetElement(0)->GetType() == Element::TETRAHEDRON && order)
      {
         MFEM_ABORT("DFSDataCollector: High order spaces on tetrahedra are not supported");
      }
   }

   data_.param = param;

   if (mesh->Dimension() == 3)
   {
      hcurl_fec_ = std::make_unique<ND_FECollection>(order+1, mesh->Dimension());
   }
   else
   {
      hcurl_fec_ = std::make_unique<H1_FECollection>(order+1, mesh->Dimension());
   }

   all_bdr_attr_.SetSize(ess_attr.Size(), 1);
   hdiv_fes_ = std::make_unique<ParFiniteElementSpace>(mesh, &hdiv_fec_);
   l2_fes_ = std::make_unique<ParFiniteElementSpace>(mesh, &l2_fec_);
   coarse_hdiv_fes_ = std::make_unique<ParFiniteElementSpace>(*hdiv_fes_);
   coarse_l2_fes_ = std::make_unique<ParFiniteElementSpace>(*l2_fes_);
   l2_0_fes_ = std::make_unique<ParFiniteElementSpace>(mesh, &l2_0_fec_);
   l2_0_fes_->SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
   el_l2dof_.reserve(num_refine+1);
   el_l2dof_.push_back(ElemToDof(*coarse_l2_fes_));

   data_.agg_hdivdof.resize(num_refine);
   data_.agg_l2dof.resize(num_refine);
   data_.P_hdiv.resize(num_refine);
   data_.P_l2.resize(num_refine);

   data_.Q_l2.resize(num_refine);
   hdiv_fes_->GetEssentialTrueDofs(ess_attr, data_.coarsest_ess_hdivdofs);
   data_.C.resize(num_refine+1);
   data_.Ae.resize(num_refine+1);

   hcurl_fes_ = std::make_unique<ParFiniteElementSpace>(mesh, hcurl_fec_.get());
   coarse_hcurl_fes_ = std::make_unique<ParFiniteElementSpace>(*hcurl_fes_);
   data_.P_hcurl.resize(num_refine);
}

SparseMatrix* AggToInteriorDof(const Array<int>& bdr_truedofs,
                               const SparseMatrix& agg_elem,
                               const SparseMatrix& elem_dof,
                               const HypreParMatrix& dof_truedof,
                               Array<HYPRE_BigInt>& agg_starts)
{
   OperatorPtr agg_dof(Mult(agg_elem, elem_dof));
   SparseMatrix& agg_dof_ref = *agg_dof.As<SparseMatrix>();
   OperatorPtr agg_tdof(dof_truedof.LeftDiagMult(agg_dof_ref, agg_starts));
   OperatorPtr agg_tdof_T(agg_tdof.As<HypreParMatrix>()->Transpose());
   SparseMatrix tdof_agg, is_shared;
   HYPRE_BigInt* trash;
   agg_tdof_T.As<HypreParMatrix>()->GetDiag(tdof_agg);
   agg_tdof_T.As<HypreParMatrix>()->GetOffd(is_shared, trash);

   int *I = new int[tdof_agg.NumRows()+1]();
   int *J = new int[tdof_agg.NumNonZeroElems()];

   Array<int> is_bdr;
   FiniteElementSpace::ListToMarker(bdr_truedofs, tdof_agg.NumRows(), is_bdr);

   int counter = 0;
   for (int i = 0; i < tdof_agg.NumRows(); ++i)
   {
      bool agg_bdr = is_bdr[i] || is_shared.RowSize(i) || tdof_agg.RowSize(i)>1;
      if (agg_bdr) { I[i+1] = I[i]; continue; }
      I[i+1] = I[i] + 1;
      J[counter++] = tdof_agg.GetRowColumns(i)[0];
   }

   auto *D = new real_t[I[tdof_agg.NumRows()]];
   std::fill_n(D, I[tdof_agg.NumRows()], 1.0);

   SparseMatrix intdof_agg(I, J, D, tdof_agg.NumRows(), tdof_agg.NumCols());
   return Transpose(intdof_agg);
}

void DFSSpaces::MakeDofRelationTables(int level)
{
   Array<HYPRE_BigInt> agg_starts(Array<HYPRE_BigInt>(l2_0_fes_->GetDofOffsets(),
                                                      2));
   auto& elem_agg = (const SparseMatrix&)*l2_0_fes_->GetUpdateOperator();
   OperatorPtr agg_elem(Transpose(elem_agg));
   SparseMatrix& agg_el = *agg_elem.As<SparseMatrix>();

   el_l2dof_.push_back(ElemToDof(*l2_fes_));
   data_.agg_l2dof[level].Reset(Mult(agg_el, el_l2dof_[level+1]));

   Array<int> bdr_tdofs;
   hdiv_fes_->GetEssentialTrueDofs(all_bdr_attr_, bdr_tdofs);
   auto tmp = AggToInteriorDof(bdr_tdofs, agg_el, ElemToDof(*hdiv_fes_),
                               *hdiv_fes_->Dof_TrueDof_Matrix(), agg_starts);
   data_.agg_hdivdof[level].Reset(tmp);
}

void DFSSpaces::CollectDFSData()
{
   auto GetP = [&](std::unique_ptr<OperatorPtr> &P,
                   std::unique_ptr<ParFiniteElementSpace> &cfes,
                   ParFiniteElementSpace& fes, const bool remove_zero)
   {
      fes.Update();
      auto T = new OperatorHandle(Operator::Hypre_ParCSR);
      fes.GetTrueTransferOperator(*cfes, *T);
      P.reset(T);
      if (remove_zero) { P->As<HypreParMatrix>()->DropSmallEntries(1e-16); }
      (level_ < (int)data_.P_l2.size()-1) ? cfes->Update() : cfes.reset();
   };

   GetP(data_.P_hdiv[level_], coarse_hdiv_fes_, *hdiv_fes_, true);
   GetP(data_.P_l2[level_], coarse_l2_fes_, *l2_fes_, false);

   MakeDofRelationTables(level_);

   GetP(data_.P_hcurl[level_], coarse_hcurl_fes_, *hcurl_fes_, true);

   ParDiscreteLinearOperator curl(hcurl_fes_.get(), hdiv_fes_.get());
   curl.AddDomainInterpolator(new CurlInterpolator);
   curl.Assemble();
   curl.Finalize();
   data_.C[level_+1].Reset(curl.ParallelAssemble());
   mfem::Array<int> ess_hcurl_tdof;
   hcurl_fes_->GetEssentialTrueDofs(ess_bdr_attr_, ess_hcurl_tdof);
   data_.Ae[level_+1].reset(
      data_.C[level_+1].As<HypreParMatrix>()
      ->EliminateCols(ess_hcurl_tdof));

   ++level_;

   if (level_ == (int)data_.P_l2.size()) { DataFinalize(); }
}

void DFSSpaces::DataFinalize()
{
   ParBilinearForm mass(l2_fes_.get());
   mass.AddDomainIntegrator(new MassIntegrator());
   mass.Assemble();
   mass.Finalize();
   OperatorPtr W(mass.LoseMat());

   SparseMatrix P_l2;
   for (int l = (int)data_.P_l2.size()-1; l >= 0; --l)
   {
      data_.P_l2[l]->As<HypreParMatrix>()->GetDiag(P_l2);
      OperatorPtr PT_l2(Transpose(P_l2));
      auto PTW = Mult(*PT_l2.As<SparseMatrix>(), *W.As<SparseMatrix>());
      auto cW = Mult(*PTW, P_l2);
      auto cW_inv = new SymDirectSubBlockSolver(*cW, el_l2dof_[l]);
      data_.Q_l2[l].Reset(new ProductOperator(cW_inv, PTW, true, true));
      W.Reset(cW);
   }

   l2_0_fes_.reset();
}

BBTSolver::BBTSolver(const HypreParMatrix& B, IterSolveParameters param)
   : Solver(B.NumRows()), BBT_solver_(B.GetComm())
{
   OperatorPtr BT(B.Transpose());
   BBT_.Reset(ParMult(&B, BT.As<HypreParMatrix>()));
   BBT_.As<HypreParMatrix>()->CopyColStarts();

   BBT_prec_.Reset(new HypreBoomerAMG(*BBT_.As<HypreParMatrix>()));
   BBT_prec_.As<HypreBoomerAMG>()->SetPrintLevel(0);

   SetOptions(BBT_solver_, param);
   BBT_solver_.SetOperator(*BBT_);
   BBT_solver_.SetPreconditioner(*BBT_prec_.As<HypreBoomerAMG>());
}

LocalSolver::LocalSolver(const DenseMatrix& M, const DenseMatrix& B)
   : Solver(M.NumRows()+B.NumRows()), local_system_(height), offset_(M.NumRows())
{
   local_system_.CopyMN(M, 0, 0);
   local_system_.CopyMN(B, offset_, 0);
   local_system_.CopyMNt(B, 0, offset_);

   local_system_.SetRow(offset_, 0.0);
   local_system_.SetCol(offset_, 0.0);
   local_system_(offset_, offset_) = -1.0;
   local_solver_.SetOperator(local_system_);
}

void LocalSolver::Mult(const Vector &x, Vector &y) const
{
   const real_t x0 = x[offset_];
   const_cast<Vector&>(x)[offset_] = 0.0;

   y.SetSize(local_system_.NumRows());
   local_solver_.Mult(x, y);

   const_cast<Vector&>(x)[offset_] = x0;
}

SaddleSchwarzSmoother::SaddleSchwarzSmoother(const HypreParMatrix& M,
                                             const HypreParMatrix& B,
                                             const SparseMatrix& agg_hdivdof,
                                             const SparseMatrix& agg_l2dof,
                                             const HypreParMatrix& P_l2,
                                             const ProductOperator& Q_l2)
   : Solver(M.NumRows() + B.NumRows()), agg_hdivdof_(agg_hdivdof),
     agg_l2dof_(agg_l2dof), solvers_loc_(agg_l2dof.NumRows())
{
   coarse_l2_projector_.Reset(new ProductOperator(&P_l2, &Q_l2, false, false));

   offsets_loc_.SetSize(3, 0);
   offsets_.SetSize(3, 0);
   offsets_[1] = M.NumRows();
   offsets_[2] = M.NumRows() + B.NumRows();

   SparseMatrix M_diag, B_diag;
   M.GetDiag(M_diag);
   B.GetDiag(B_diag);

   DenseMatrix B_loc, M_loc;

   for (int agg = 0; agg < (int)solvers_loc_.size(); agg++)
   {
      GetRowColumnsRef(agg_hdivdof_, agg, hdivdofs_loc_);
      GetRowColumnsRef(agg_l2dof_, agg, l2dofs_loc_);
      M_loc.SetSize(hdivdofs_loc_.Size(), hdivdofs_loc_.Size());
      B_loc.SetSize(l2dofs_loc_.Size(), hdivdofs_loc_.Size());
      M_diag.GetSubMatrix(hdivdofs_loc_, hdivdofs_loc_, M_loc);
      B_diag.GetSubMatrix(l2dofs_loc_, hdivdofs_loc_, B_loc);
      solvers_loc_[agg].Reset(new LocalSolver(M_loc, B_loc));
   }
}

void SaddleSchwarzSmoother::Mult(const Vector & x, Vector & y) const
{
   y.SetSize(offsets_[2]);
   y = 0.0;

   BlockVector blk_y(y.GetData(), offsets_);
   BlockVector Pi_x(offsets_); // aggregate-wise average free projection of x
   static_cast<Vector&>(Pi_x) = x;

   // Right hand side: F_l = F - W_l P_l2[l] (W_{l+1})^{-1} P_l2[l]^T F
   // This ensures the existence of solutions to the local problems
   Vector coarse_l2_projection(Pi_x.BlockSize(1));
   coarse_l2_projector_->MultTranspose(Pi_x.GetBlock(1), coarse_l2_projection);

   Pi_x.GetBlock(1) -= coarse_l2_projection;

   for (int agg = 0; agg < (int)solvers_loc_.size(); agg++)
   {
      GetRowColumnsRef(agg_hdivdof_, agg, hdivdofs_loc_);
      GetRowColumnsRef(agg_l2dof_, agg, l2dofs_loc_);

      offsets_loc_[1] = hdivdofs_loc_.Size();
      offsets_loc_[2] = offsets_loc_[1]+l2dofs_loc_.Size();

      BlockVector rhs_loc(offsets_loc_), sol_loc(offsets_loc_);
      Pi_x.GetBlock(0).GetSubVector(hdivdofs_loc_, rhs_loc.GetBlock(0));
      Pi_x.GetBlock(1).GetSubVector(l2dofs_loc_, rhs_loc.GetBlock(1));

      solvers_loc_[agg]->Mult(rhs_loc, sol_loc);

      blk_y.GetBlock(0).AddElementVector(hdivdofs_loc_, sol_loc.GetBlock(0));
      blk_y.GetBlock(1).AddElementVector(l2dofs_loc_, sol_loc.GetBlock(1));
   }

   coarse_l2_projector_->Mult(blk_y.GetBlock(1), coarse_l2_projection);
   blk_y.GetBlock(1) -= coarse_l2_projection;
}

DivFreeSolver::DivFreeSolver(const HypreParMatrix &M,
                             const HypreParMatrix &B,
                             const DFSData& data)
   : DarcySolver(M.NumRows(), B.NumRows()), data_(data), param_(data.param),
     BT_(B.Transpose()),
     BBT_solver_(B, param_.BBT_solve_param),
     ops_offsets_(data.P_l2.size()+1),
     ops_(ops_offsets_.size()),
     blk_Ps_(ops_.size()-1),
     smoothers_(ops_.size())
{
   ops_offsets_.back().MakeRef(DarcySolver::offsets_);
   ops_.back() = std::make_unique<BlockOperator>(ops_offsets_.back());
   ops_.back()->SetBlock(0, 0, const_cast<HypreParMatrix*>(&M));
   ops_.back()->SetBlock(1, 0, const_cast<HypreParMatrix*>(&B));
   ops_.back()->SetBlock(0, 1, BT_.Ptr());

   for (int l = data.P_l2.size(); l >= 0; --l)
   {
      auto &M_f = static_cast<const HypreParMatrix&>(ops_[l]->GetBlock(0, 0));
      auto &B_f = static_cast<const HypreParMatrix&>(ops_[l]->GetBlock(1, 0));

      if (l == 0)
      {
         SparseMatrix M_f_diag, B_f_diag;
         M_f.GetDiag(M_f_diag);
         B_f.GetDiag(B_f_diag);
         for (int dof : data.coarsest_ess_hdivdofs)
         {
            M_f_diag.EliminateRowCol(dof);
            B_f_diag.EliminateCol(dof);
         }

         const IterSolveParameters& param = param_.coarse_solve_param;
         auto coarse_solver = new BDPMinresSolver(M_f, B_f, param);
         if (ops_.size() > 1)
         {
            coarse_solver->SetEssZeroDofs(data.coarsest_ess_hdivdofs);
         }
         smoothers_[l].reset(coarse_solver);
         continue;
      }

      auto P_hdiv_l = data.P_hdiv[l-1]->As<HypreParMatrix>();
      auto P_l2_l = data.P_l2[l-1]->As<HypreParMatrix>();
      SparseMatrix& agg_hdivdof_l = *data.agg_hdivdof[l-1].As<SparseMatrix>();
      SparseMatrix& agg_l2dof_l = *data.agg_l2dof[l-1].As<SparseMatrix>();
      ProductOperator& Q_l2_l = *data.Q_l2[l-1].As<ProductOperator>();
      auto* C_l = data.C[l].As<HypreParMatrix>();

      auto S0 = new SaddleSchwarzSmoother(M_f, B_f, agg_hdivdof_l,
                                          agg_l2dof_l, *P_l2_l, Q_l2_l);
      if (param_.coupled_solve)
      {
         auto S1 = new BlockDiagonalPreconditioner(ops_offsets_[l]);
         S1->SetDiagonalBlock(0, new AuxSpaceSmoother(M_f, C_l));
         S1->owns_blocks = 1;
         smoothers_[l] =
            std::make_unique<ProductSolver>(ops_[l].get(), S0, S1, false, true, true);
      }
      else
      {
         smoothers_[l].reset(S0);
      }

      HypreParMatrix* M_c = TwoStepsRAP(P_hdiv_l, &M_f, P_hdiv_l);
      HypreParMatrix* B_c = TwoStepsRAP(P_l2_l, &B_f, P_hdiv_l);

      ops_offsets_[l-1].SetSize(3, 0);
      ops_offsets_[l-1][1] = M_c->NumRows();
      ops_offsets_[l-1][2] = M_c->NumRows() + B_c->NumRows();

      blk_Ps_[l-1] =
         std::make_unique<BlockOperator>(ops_offsets_[l], ops_offsets_[l-1]);
      blk_Ps_[l-1]->SetBlock(0, 0, P_hdiv_l);
      blk_Ps_[l-1]->SetBlock(1, 1, P_l2_l);

      ops_[l-1] =
         std::make_unique<BlockOperator>(ops_offsets_[l-1]);
      ops_[l-1]->SetBlock(0, 0, M_c);
      ops_[l-1]->SetBlock(1, 0, B_c);
      ops_[l-1]->SetBlock(0, 1, B_c->Transpose());
      ops_[l-1]->owns_blocks = 1;
   }

   if (data_.P_l2.size() == 0) { return; }

   Array<bool> own_ops(ops_.size());
   Array<bool> own_smoothers(smoothers_.size());
   Array<bool> own_blk_Ps(blk_Ps_.size());
   own_ops = false, own_smoothers = false, own_blk_Ps = false;

   Array<Solver*> smoothers(smoothers_.size());

   if (param_.coupled_solve)
   {
      solver_.Reset(new GMRESSolver(B.GetComm()));
      solver_.As<GMRESSolver>()->SetOperator(*(ops_.back()));
      Array<BlockOperator*> ops(ops_.size()), blk_Ps(blk_Ps_.size());
      for (size_t i = 0; i < ops_.size(); ++i) { ops[i] = ops_[i].get(); }
      for (size_t i = 0; i < blk_Ps_.size(); ++i) { blk_Ps[i] = blk_Ps_[i].get(); }
      for (size_t i = 0; i < smoothers_.size(); ++i) { smoothers[i] = smoothers_[i].get(); }
      prec_.Reset(new Multigrid(ops, smoothers, blk_Ps,
                                own_ops, own_smoothers, own_blk_Ps));
   }
   else
   {
      Array<HypreParMatrix*> ops(data_.P_hcurl.size()+1);
      Array<HypreParMatrix*> Ps(data_.P_hcurl.size());
      auto C_finest = data.C.back().As<HypreParMatrix>();
      ops.Last() = TwoStepsRAP(C_finest, &M, C_finest);
      ops.Last()->EliminateZeroRows();
      ops.Last()->DropSmallEntries(1e-14);
      solver_.Reset(new CGSolver(B.GetComm()));
      solver_.As<CGSolver>()->SetOperator(*ops.Last());
      smoothers.Last() = new HypreSmoother(*ops.Last());
      static_cast<HypreSmoother*>(smoothers.Last())->SetOperatorSymmetry(true);
      for (int l = Ps.Size()-1; l >= 0; --l)
      {
         Ps[l] = data_.P_hcurl[l]->As<HypreParMatrix>();
         ops[l] = TwoStepsRAP(Ps[l], ops[l+1], Ps[l]);
         ops[l]->DropSmallEntries(1e-14);
         smoothers[l] = new HypreSmoother(*ops[l]);
         static_cast<HypreSmoother*>(smoothers[l])->SetOperatorSymmetry(true);
      }
      own_ops = true, own_smoothers = true;
      prec_.Reset(new Multigrid(ops, smoothers, Ps,
                                own_ops, own_smoothers, own_blk_Ps));
   }

   solver_.As<IterativeSolver>()->SetPreconditioner(*prec_.As<Solver>());
   SetOptions(*solver_.As<IterativeSolver>(), param_);
}

void DivFreeSolver::SolveParticular(const Vector& rhs, Vector& sol) const
{
   std::vector<Vector> rhss(smoothers_.size()), sols(smoothers_.size());
   rhss.back().SetDataAndSize(const_cast<real_t*>(rhs.HostRead()), rhs.Size());
   sols.back().SetDataAndSize(sol.HostWrite(), sol.Size());

   for (int l = blk_Ps_.size()-1; l >= 0; --l)
   {
      rhss[l].SetSize(blk_Ps_[l]->NumCols());
      sols[l].SetSize(blk_Ps_[l]->NumCols());

      sols[l] = 0.0;
      rhss[l] = 0.0;

      blk_Ps_[l]->MultTranspose(rhss[l+1], rhss[l]);
   }

   for (size_t l = 0; l < smoothers_.size(); ++l)
   {
      smoothers_[l]->Mult(rhss[l], sols[l]);
   }

   for (size_t l = 0; l < blk_Ps_.size(); ++l)
   {
      Vector P_sol(blk_Ps_[l]->NumRows());
      blk_Ps_[l]->Mult(sols[l], P_sol);
      sols[l+1] += P_sol;
   }
}

void DivFreeSolver::SolveDivFree(const Vector &rhs, Vector& sol) const
{
   Vector rhs_divfree(data_.C.back()->NumCols());
   data_.C.back()->MultTranspose(rhs, rhs_divfree);

   Vector potential_divfree(rhs_divfree.Size());
   potential_divfree = 0.0;
   solver_->Mult(rhs_divfree, potential_divfree);

   data_.C.back()->Mult(potential_divfree, sol);
}

void DivFreeSolver::SolvePotential(const Vector& rhs, Vector& sol) const
{
   Vector rhs_p(BT_->NumCols());
   BT_->MultTranspose(rhs, rhs_p);
   BBT_solver_.Mult(rhs_p, sol);
}

void DivFreeSolver::Mult(const Vector & x, Vector & y) const
{
   MFEM_VERIFY(x.Size() == offsets_[2], "MLDivFreeSolver: x size is invalid");
   MFEM_VERIFY(y.Size() == offsets_[2], "MLDivFreeSolver: y size is invalid");

   if (ops_.size() == 1) { smoothers_[0]->Mult(x, y); return; }

   BlockVector blk_y(y, offsets_);

   BlockVector resid(offsets_);
   ops_.back()->Mult(y, resid);
   add(1.0, x, -1.0, resid, resid);

   BlockVector correction(offsets_);
   correction = 0.0;

   if (param_.coupled_solve)
   {
      solver_->Mult(resid, correction);
      y += correction;
   }
   else
   {
      StopWatch ch;
      ch.Start();

      SolveParticular(resid, correction);
      blk_y += correction;

      if (param_.verbose)
      {
         cout << "Particular solution found in " << ch.RealTime() << "s.\n";
      }

      ch.Clear();
      ch.Start();

      ops_.back()->Mult(y, resid);
      add(1.0, x, -1.0, resid, resid);

      SolveDivFree(resid.GetBlock(0), correction.GetBlock(0));
      blk_y.GetBlock(0) += correction.GetBlock(0);

      if (param_.verbose)
      {
         cout << "Divergence free solution found in " << ch.RealTime() << "s.\n";
      }

      ch.Clear();
      ch.Start();

      auto& M = dynamic_cast<const HypreParMatrix&>(ops_.back()->GetBlock(0, 0));
      M.Mult(-1.0, correction.GetBlock(0), 1.0, resid.GetBlock(0));
      SolvePotential(resid.GetBlock(0), correction.GetBlock(1));
      blk_y.GetBlock(1) += correction.GetBlock(1);

      if (param_.verbose)
      {
         cout << "Scalar potential found in " << ch.RealTime() << "s.\n";
      }
   }
}

int DivFreeSolver::GetNumIterations() const
{
   if (ops_.size() == 1)
   {
      return static_cast<BDPMinresSolver*>
             (smoothers_.at(0).get())->GetNumIterations();
   }
   return solver_.As<IterativeSolver>()->GetNumIterations();
}

} // namespace mfem::blocksolvers
