// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "coupledop.hpp"

namespace mfem
{
CoupledOperator::CoupledOperator(const Array<int> &bdr_u_is_ess,
                                 const Array<int> &bdr_E_is_ess, Coefficient *sigma_,
                                 const Array<LinearForm*> &lfs_, const Array<Coefficient*> &coeffs_,
                                 FiniteElementSpace *u_space_, FiniteElementSpace *n_space_,
                                 FiniteElementSpace *E_space_, FiniteElementSpace *B_space_,
                                 FiniteElementSpace *tr_space_, real_t td)
   : TimeDependentOperator(0, IMPLICIT), sigma(sigma_), lfs(lfs_),
     coeffs(coeffs_), u_space(u_space_), n_space(n_space_),
     E_space(E_space_), B_space(B_space_), tr_space(tr_space_)
{
   offsets = ConstructOffsets(u_space, n_space, E_space, B_space, tr_space);

   width = height = offsets.Last();

   //plasma

   const bool dg = u_space->FEColl()->GetContType() ==
                   FiniteElementCollection::DISCONTINUOUS;

   if (!dg)
   {
      u_space->GetEssentialTrueDofs(bdr_u_is_ess, ess_u_tdofs_list);
   }

   darcy = new DarcyForm(u_space, n_space);

   Mu = darcy->GetFluxMassForm();
   Mn = darcy->GetPotentialMassForm();
   Du = darcy->GetFluxDivForm();

   idtcoeff = new FunctionCoefficient([&](const Vector &) { return idt; });
   dtcoeff = new FunctionCoefficient([&](const Vector &) { return 1./idt; });

   if (dg)
   {
      Mu->AddDomainIntegrator(new VectorMassIntegrator(*idtcoeff));
   }
   else
   {
      Mu->AddDomainIntegrator(new VectorFEMassIntegrator(idtcoeff));
   }

   if (dg)
   {
      Du->AddDomainIntegrator(new VectorDivergenceIntegrator());
      Du->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                       new DGNormalTraceIntegrator(-1.)));
      Du->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                          -1.)), const_cast<Array<int>&>(bdr_u_is_ess));
   }
   else
   {
      Du->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   }

   if (dg && td > 0.)
   {
      Mn->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(*dtcoeff, td));
      Mn->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(*dtcoeff, td),
                               const_cast<Array<int>&>(bdr_u_is_ess));
   }

   Mn->AddDomainIntegrator(new MassIntegrator(*idtcoeff));

   if (tr_space)
   {
      darcy->EnableHybridization(tr_space,
                                 new NormalTraceJumpIntegrator(),
                                 ess_u_tdofs_list);
   }

   //Maxwell

   E_space->GetEssentialTrueDofs(bdr_E_is_ess, ess_E_tdofs_list);

   ME = new BilinearForm(E_space);
   ME->AddDomainIntegrator(new VectorFEMassIntegrator());
   CE = new MixedBilinearForm(E_space, B_space);
   CE->AddDomainIntegrator(new MixedCurlIntegrator());
   CdE = new DiscreteLinearOperator(E_space, B_space);
   CdE->AddDomainInterpolator(new CurlInterpolator());

   ME->Assemble();
   ME->Finalize();

   CE->Assemble();
   CE->Finalize();

   CdE->Assemble();
   CdE->Finalize();
}

CoupledOperator::~CoupledOperator()
{
   delete ME;
   delete CE;
   delete CdE;
   delete darcy;
   delete idtcoeff;
   delete dtcoeff;
}

Array<int> CoupledOperator::ConstructOffsets(const FiniteElementSpace *u_space,
                                             const FiniteElementSpace *n_space, const FiniteElementSpace *E_space,
                                             const FiniteElementSpace *B_space, const FiniteElementSpace *trace_space)
{
   Array<int> offsets((trace_space)?(6):(5));
   int i = 0;
   offsets[i++] = 0;
   offsets[i++] = u_space->GetVSize();
   offsets[i++] = n_space->GetVSize();
   if (trace_space)
   {
      offsets[i++] = trace_space->GetVSize();
   }
   offsets[i++] = E_space->GetVSize();
   offsets[i++] = B_space->GetVSize();
   offsets.PartialSum();

   return offsets;
}

void CoupledOperator::ImplicitSolve(const double dt, const Vector &x, Vector &y)
{
   const bool time_track = false;

   BlockVector bx(const_cast<Vector&>(x), offsets);
   BlockVector by(y, offsets);

   int i = 0;
   const Vector &un = bx.GetBlock(i++);
   const Vector &nn = bx.GetBlock(i++);
   if (tr_space) { i++; }
   const Vector &En = bx.GetBlock(i++);
   const Vector &Bn = bx.GetBlock(i++);

   i = 2;
   //Vector &u = by.GetBlock(i++);
   //Vector &n = by.GetBlock(i++);
   if (tr_space) { i++; }
   Vector &E = by.GetBlock(i++);
   Vector &B = by.GetBlock(i++);

   //offsets

   Array<int> X_offsets(3);
   X_offsets[0] = 0;
   if (tr_space)
   {
      X_offsets[1] = tr_space->GetVSize();
   }
   else
   {
      X_offsets[1] = darcy->Width();
   }
   X_offsets[2] = E_space->GetVSize();
   X_offsets.PartialSum();

   //solution & rhs vectors

   BlockVector X(X_offsets), RHS(X_offsets);
   BlockDiagonalPreconditioner bprec(X_offsets);
   bprec.owns_blocks = true;

   //plasma

   LinearForm *g = lfs[0];
   LinearForm *f = lfs[1];
   LinearForm *h = lfs[2];

   //set time

   for (Coefficient *coeff : coeffs)
   {
      coeff->SetTime(t);
   }

   //assemble rhs

   StopWatch chrono;
   if (time_track)
   {
      chrono.Clear();
      chrono.Start();
   }

   g->Assemble();
   f->Assemble();
   if (h) { h->Assemble(); }

   //check if the operator has to be reassembled

   bool reassemble = (idt != 1./dt);

   if (reassemble)
   {
      idt = 1./dt;

      //reset the operator

      darcy->Update();

      //assemble the system
      darcy->Assemble();

      if (Mu && tr_space)
      {
         Mu->Update();
         Mu->Assemble();
         //Mq0->Finalize();
      }
      if (Mn && tr_space)
      {
         Mn->Update();
         Mn->Assemble();
         //Mt0->Finalize();
      }
   }

   if (Mu)
   {
      GridFunction u_h;
      u_h.MakeRef(darcy->FluxFESpace(), const_cast<Vector&>(un), 0);
      Mu->AddMult(u_h, *g, +1.);
   }

   if (Mn)
   {
      GridFunction p_h;
      p_h.MakeRef(darcy->PotentialFESpace(), const_cast<Vector&>(nn), 0);
      Mn->AddMult(p_h, *f, -1.);
   }

   //form the reduced system

   OperatorHandle op_pl;
   BlockVector darcy_x(const_cast<Vector&>(x), darcy->GetOffsets());
   BlockVector darcy_rhs(g->GetData(), darcy->GetOffsets());
   BlockVector X_pl;
   BlockVector RHS_pl;
   Array<int> tr_offsets(2);

   if (tr_space)
   {
      tr_offsets[0] = 0;
      tr_offsets[1] = tr_space->GetVSize();
      X_pl.Update(X, tr_offsets);
      RHS_pl.Update(RHS, tr_offsets);
      if (h)
      {
         RHS_pl.Vector::operator=(*h);
      }
      else
      {
         RHS_pl = 0.;
      }
      darcy->FormLinearSystem(ess_u_tdofs_list, darcy_x, darcy_rhs,
                              op_pl, X_pl, RHS_pl);
   }
   else
   {
      X_pl.Update(X, darcy->GetOffsets());
      RHS_pl.Update(RHS, darcy->GetOffsets());
      X_pl = darcy_x;
      RHS_pl = darcy_rhs;
      BlockVector darcy_x_tmp(X_pl, darcy->GetOffsets());
      BlockVector darcy_rhs_tmp(RHS_pl, darcy->GetOffsets());
      darcy->FormLinearSystem(ess_u_tdofs_list, darcy_x_tmp, darcy_rhs_tmp,
                              op_pl, X_pl, RHS_pl);
   }

   if (time_track)
   {
      chrono.Stop();
      std::cout << "Assembly took " << chrono.RealTime() << "s.\n";
   }

   if (tr_space)
   {
      bprec.SetDiagonalBlock(0, new DSmoother(*op_pl.As<SparseMatrix>()));
   }
   else
   {
      BlockDiagonalPreconditioner *darcy_prec = new BlockDiagonalPreconditioner(
         darcy->GetOffsets());
      darcy_prec->owns_blocks = true;;
      darcy_prec->SetDiagonalBlock(0, new GSSmoother(Mu->SpMat()));
      darcy_prec->SetDiagonalBlock(1, new DSmoother(Mn->SpMat()));
      bprec.SetDiagonalBlock(0, darcy_prec);
   }

   //Maxwell

   TransposeOperator Ct(CE);
   ProductOperator CtCd(&Ct, CdE, false, false);
   SumOperator MECtCd(ME, 1., &CtCd, dt*dt, false, false);

   Vector &X_max = X.GetBlock(1);
   X_max = En;

   Vector &RHS_max = RHS.GetBlock(1);
   ME->Mult(En, RHS_max);
   //MECtCd.Mult(En, rhs);
   CE->AddMultTranspose(Bn, RHS_max, dt);
   //CtCd.AddMult(En, rhs, +dt*dt/2.);

   //ME - dt*c2/2*Ct(Bn-dt*CE) = MEn + dt*c2/2CtBn
   //(M + dt2*c2/2*Ct*C)E = MEn + dt*c2*CtBn
   //ME - dt*c2*Ct(Bn-dt/2*CE-dt/2*CEn) = MEn
   //(M + dt2*c2/2*Ct*C)E = MEn + dt*c2CtBn - dt2*c2/2*CtCEn
   //-1/dt B - CE = 0

   //ME->EliminateVDofsInRHS(ess_tdof_list, En, rhs);

   ConstrainedOperator op_max(&MECtCd, ess_E_tdofs_list);

   //op_max.EliminateRHS(En, RHS_max);

   bprec.SetDiagonalBlock(1, new DSmoother(ME->SpMat()));

   //coupling

   /*const Mesh *mesh = n_space->GetMesh();
   const int NE = mesh->GetNE();
   for (int el = 0; el < NE; el++)
   {

   }*/

   ReducedOperator bop(sigma, darcy, E_space, *op_pl.Ptr(), op_max);
   if (tr_space)
   {
      Array<int> ess_tr_tdofs_list;//dummy
      bop.SetEssentialTDOFs(ess_tr_tdofs_list, ess_E_tdofs_list);
   }
   else
   {
      bop.SetEssentialTDOFs(ess_u_tdofs_list, ess_E_tdofs_list);
   }
   bop.EliminateRHS(X, RHS);

   //solve

   GMRESSolver solver;
   solver.SetMaxIter(1000);
   solver.SetAbsTol(0.);
   solver.SetRelTol(1e-6);
   solver.SetOperator(bop);
   solver.SetPreconditioner(bprec);
   solver.SetPrintLevel(0);

   NewtonSolver newton;
   newton.SetMaxIter(1000);
   newton.SetAbsTol(0.);
   newton.SetRelTol(1e-5);
   newton.SetOperator(bop);
   newton.SetSolver(solver);

   if (time_track)
   {
      chrono.Clear();
      chrono.Start();
   }

   newton.Mult(RHS, X);

   if (time_track)
   {
      chrono.Stop();
   }

   if (newton.GetConverged())
   {
      std::cout << " converged in " << newton.GetNumIterations()
                << " iterations with a residual norm of " << newton.GetFinalNorm()
                << ".\n";
   }
   else
   {
      std::cout << " did not converge in " << newton.GetNumIterations()
                << " iterations. Residual norm is " << newton.GetFinalNorm()
                << ".\n";
   }
   if (time_track) { std::cout << "Solver took " << chrono.RealTime() << "s.\n"; }

   //recover solution

   BlockVector darcy_y(y, darcy->GetOffsets());
   darcy_y = darcy_x;

   if (tr_space)
   {
      darcy->RecoverFEMSolution(X_pl, darcy_rhs, darcy_y);
      Vector &darcy_tr = by.GetBlock(2);
      darcy_tr = X_pl;
      darcy_tr -= bx.GetBlock(2);
      darcy_tr *= idt;
   }
   else
   {
      darcy->RecoverFEMSolution(X_pl, RHS_pl, X_pl);
      darcy_y = X_pl;
   }

   darcy_y -= darcy_x;
   darcy_y *= idt;

   E = X_max;

   E -= En;
   E *= idt;

   CdE->Mult(En, B);
   //CdE->AddMult(E, B);
   //B *= -1./2.;
   B.Neg();
}
CoupledOperator::ReducedOperator::ReducedOperator(Coefficient *sigma_,
                                                  DarcyForm *darcy_, FiniteElementSpace *fes_E_, Operator &pl, Operator &max)
   : sigma(sigma_), darcy(darcy_), fes_E(fes_E_)
{
   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = pl.Width();
   offsets[2] = max.Width();
   offsets.PartialSum();

   width = height = offsets.Last();

   if (darcy->GetHybridization())
   {
      offsets_x = offsets;
   }
   else
   {
      offsets_x.SetSize(4);
      offsets_x[0] = 0;
      offsets_x[1] = darcy->FluxFESpace()->GetVSize();
      offsets_x[2] = darcy->PotentialFESpace()->GetVSize();
      offsets_x[3] = fes_E->GetVSize();
      offsets_x.PartialSum();
   }

   BlockOperator *bop = new BlockOperator(offsets);
   bop->SetDiagonalBlock(0, &pl);
   bop->SetDiagonalBlock(1, &max);

   op.Reset(bop);
}

CoupledOperator::ReducedOperator::~ReducedOperator()
{
}

void CoupledOperator::ReducedOperator::SetEssentialTDOFs(
   const Array<int> &u_tdofs_list, const Array<int> &E_tdofs_list)
{
   ess_tdofs_list.DeleteAll();
   ess_tdofs_list.Append(u_tdofs_list);
   ess_tdofs_list.Append(E_tdofs_list);
   const int size = ess_tdofs_list.Size();
   for (int i = 0; i < E_tdofs_list.Size(); i++)
   {
      int &tdof = ess_tdofs_list[size - 1 - i];
      if (tdof >= 0)
      {
         tdof += offsets[1];
      }
      else
      {
         tdof -= offsets[1];
      }
   }
}

void CoupledOperator::ReducedOperator::EliminateRHS(const Vector &x,
                                                    Vector &b) const
{
   if (ess_tdofs_list.Size() <= 0) { return; }

   Vector w(x.Size()), z(b.Size());

   w = 0.;

   for (int tdof : ess_tdofs_list)
   {
      w(tdof) = x(tdof);
   }

   MultUnconstrained(w, z);
   b -= z;

   for (int tdof : ess_tdofs_list)
   {
      b(tdof) = x(tdof);
   }
}

void CoupledOperator::ReducedOperator::Mult(const Vector &x, Vector &y) const
{
   Vector z(x.Size());
   z = x;

   for (int tdof : ess_tdofs_list)
   {
      z(tdof) = 0.;
   }

   MultUnconstrained(z, y);

   for (int tdof : ess_tdofs_list)
   {
      y(tdof) = x(tdof);
   }
}

void CoupledOperator::ReducedOperator::MultUnconstrained(const Vector &x,
                                                         Vector &y) const
{
   op->Mult(x, y);

   //TODO: hybridization
   if (darcy->GetHybridization()) { return; }

   BlockVector bx(const_cast<Vector&>(x), offsets_x);
   BlockVector by(const_cast<Vector&>(y), offsets_x);

   Mesh *mesh = fes_E->GetMesh();
   FiniteElementSpace *fes_u = darcy->FluxFESpace();
   FiniteElementSpace *fes_n = darcy->PotentialFESpace();
   Array<int> vdofs_u, dofs_n, vdofs_E;
   DenseMatrix vshape_u, vshape_E;
   Vector shape_u, shape_E, shape_n, n_z, E_z, E, bu_z, bE_z;

   for (int z = 0; z < mesh->GetNE(); z++)
   {
      ElementTransformation *Tr = mesh->GetElementTransformation(z);
      const FiniteElement *fe_u = fes_u->GetFE(z);
      const FiniteElement *fe_n = fes_n->GetFE(z);
      const FiniteElement *fe_E = fes_E->GetFE(z);
      const int sdim = Tr->GetSpaceDim();
      const int ndof_u = fe_u->GetDof();
      const int ndof_n = fe_n->GetDof();

      fes_u->GetElementVDofs(z, vdofs_u);
      fes_n->GetElementDofs(z, dofs_n);
      fes_E->GetElementVDofs(z, vdofs_E);

      if (fe_u->GetRangeType() == FiniteElement::VECTOR)
      {
         vshape_u.SetSize(vdofs_u.Size(), sdim);
      }
      shape_u.SetSize(ndof_u);
      shape_n.SetSize(ndof_n);
      vshape_E.SetSize(vdofs_E.Size(), sdim);
      shape_E.SetSize(vdofs_E.Size());
      E.SetSize(sdim);

      bu_z.SetSize(vdofs_u.Size());
      bE_z.SetSize(vdofs_E.Size());
      bu_z = 0.;
      bE_z = 0.;

      bx.GetBlock(1).GetSubVector(dofs_n, n_z);
      bx.GetBlock(2).GetSubVector(vdofs_E, E_z);

      const int order = std::max(fe_E->GetOrder(), fe_n->GetOrder()) * 2 + 1;
      const IntegrationRule &ir = IntRules.Get(fe_n->GetGeomType(), order);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         fe_n->CalcShape(ip, shape_n);
         const real_t n = n_z * shape_n;

         fe_E->CalcVShape(*Tr, vshape_E);
         vshape_E.MultTranspose(E_z, E);

         real_t w = n * Tr->Weight();
         if (sigma) { w *= sigma->Eval(*Tr, ip); }

         if (fe_u->GetRangeType() == FiniteElement::VECTOR)
         {
            fe_u->CalcVShape(*Tr, vshape_u);
            vshape_u.Mult(E, shape_u);
            bu_z.Add(w, shape_u);
         }
         else
         {
            fe_u->CalcShape(ip, shape_u);
            for (int d = 0; d < sdim; d++)
               for (int i = 0; i < ndof_u; i++)
               {
                  bu_z(i+d*ndof_u) += w * E(d) * shape_u(i);
               }
         }
         vshape_E.Mult(E, shape_E);
         bE_z.Add(w, shape_E);
      }

      by.GetBlock(0).AddElementVector(vdofs_u, bu_z);
      by.GetBlock(2).AddElementVector(vdofs_E, bE_z);
   }
}

Operator &CoupledOperator::ReducedOperator::GetGradient(const Vector &x) const
{
   MFEM_ASSERT(!darcy->GetHybridization(), "Not implemented");
   grad.Clear();

   FiniteElementSpace *fes_u = darcy->FluxFESpace();
   FiniteElementSpace *fes_n = darcy->PotentialFESpace();

   SparseMatrix *BEE, *BEu, *BnE, *Bnu;

   BEE = new SparseMatrix(fes_E->GetVSize());
   BEu = new SparseMatrix(fes_u->GetVSize(), fes_E->GetVSize());
   BnE = new SparseMatrix(fes_E->GetVSize(), fes_n->GetVSize());
   Bnu = new SparseMatrix(fes_u->GetVSize(), fes_n->GetVSize());

   BlockVector bx(const_cast<Vector&>(x), offsets_x);

   Mesh *mesh = fes_E->GetMesh();

   Array<int> vdofs_u, dofs_n, vdofs_E;
   DenseMatrix vshape_u, vshape_E;
   DenseMatrix BEE_z, BEu_z, BnE_z, Bnu_z;
   Vector shape_u, shape_E, shape_n, n_z, E_z, E;

   for (int z = 0; z < mesh->GetNE(); z++)
   {
      ElementTransformation *Tr = mesh->GetElementTransformation(z);
      const FiniteElement *fe_u = fes_u->GetFE(z);
      const FiniteElement *fe_n = fes_n->GetFE(z);
      const FiniteElement *fe_E = fes_E->GetFE(z);
      const int sdim = Tr->GetSpaceDim();
      const int ndof_u = fe_u->GetDof();
      const int ndof_n = fe_n->GetDof();

      fes_u->GetElementVDofs(z, vdofs_u);
      fes_n->GetElementDofs(z, dofs_n);
      fes_E->GetElementVDofs(z, vdofs_E);

      if (fe_u->GetRangeType() == FiniteElement::VECTOR)
      {
         vshape_u.SetSize(vdofs_u.Size(), sdim);
      }
      shape_u.SetSize(ndof_u);
      shape_n.SetSize(ndof_n);
      vshape_E.SetSize(vdofs_E.Size(), sdim);
      shape_E.SetSize(vdofs_E.Size());
      E.SetSize(sdim);

      BEE_z.SetSize(vdofs_E.Size());
      BEu_z.SetSize(vdofs_u.Size(), vdofs_E.Size());
      BnE_z.SetSize(vdofs_E.Size(), dofs_n.Size());
      Bnu_z.SetSize(vdofs_u.Size(), dofs_n.Size());
      BEE_z = 0.;
      BEu_z = 0.;
      BnE_z = 0.;
      Bnu_z = 0.;

      bx.GetBlock(1).GetSubVector(dofs_n, n_z);
      bx.GetBlock(2).GetSubVector(vdofs_E, E_z);

      const int order = std::max(fe_E->GetOrder(), fe_n->GetOrder()) * 2 + 1;
      const IntegrationRule &ir = IntRules.Get(fe_n->GetGeomType(), order);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         fe_n->CalcShape(ip, shape_n);
         const real_t n = n_z * shape_n;

         fe_E->CalcVShape(*Tr, vshape_E);
         vshape_E.MultTranspose(E_z, E);

         real_t w = Tr->Weight();
         if (sigma) { w *= sigma->Eval(*Tr, ip); }

         if (fe_u->GetRangeType() == FiniteElement::VECTOR)
         {
            fe_u->CalcVShape(*Tr, vshape_u);
            vshape_u.Mult(E, shape_u);
            AddMult_a_VWt(w, shape_u, shape_n, Bnu_z);
            AddMult_a_ABt(w * n, vshape_u, vshape_E, BEu_z);
         }
         else
         {
            fe_u->CalcShape(ip, shape_u);
            for (int d = 0; d < sdim; d++)
               for (int i = 0; i < ndof_u; i++)
               {
                  for (int j = 0; j < ndof_n; j++)
                  {
                     Bnu_z(i+d*ndof_u, j) += w * E(d) * shape_u(i) * shape_n(j);
                  }

                  for (int j = 0; j < vdofs_E.Size(); j++)
                  {
                     BEu_z(i+d*ndof_u, j) += w * n * shape_u(i) * vshape_E(j, d);
                  }
               }
         }
         vshape_E.Mult(E, shape_E);

         AddMult_a_AAt(w * n, vshape_E, BEE_z);
         AddMult_a_VWt(w, shape_E, shape_n, BnE_z);
      }

      BEE->AddSubMatrix(vdofs_E, vdofs_E, BEE_z);
      BEu->AddSubMatrix(vdofs_u, vdofs_E, BEu_z);
      BnE->AddSubMatrix(vdofs_E, dofs_n, BnE_z);
      Bnu->AddSubMatrix(vdofs_u, dofs_n, Bnu_z);
   }

   BEE->Finalize();
   BEu->Finalize();
   BnE->Finalize();
   Bnu->Finalize();

   BlockOperator *bcouple = new BlockOperator(offsets_x);
   bcouple->owns_blocks = true;
   bcouple->SetBlock(2, 2, BEE);
   bcouple->SetBlock(0, 2, BEu);
   bcouple->SetBlock(2, 1, BnE);
   bcouple->SetBlock(0, 1, Bnu);

   grad.Reset(new SumOperator(op.Ptr(), 1., bcouple, 1., false, true));

   if (ess_tdofs_list.Size() > 0)
   {
      grad.SetOperatorOwner(false);
      grad.Reset(new ConstrainedOperator(grad.Ptr(), ess_tdofs_list));
   }

   return *grad;
}
}
