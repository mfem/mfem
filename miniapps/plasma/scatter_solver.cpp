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

#include "scatter_solver.hpp"

namespace mfem
{
namespace plasma
{

CoupledOperator::CoupledOperator(std::vector<Coefficient*> kappa_,
                                 Coefficient *sigma_,
                                 std::vector<std::pair<BCType,VectorCoefficient*>> bcs_plasma,
                                 std::vector<std::pair<BCType,VectorCoefficient*>> bcs_maxwell,
                                 FiniteElementSpace *q_space_, FiniteElementSpace *p_space_,
                                 FiniteElementSpace *E_space_, FiniteElementSpace *B_space_,
                                 FiniteElementSpace *tr_space_, real_t cs, real_t td,
                                 bool imex_)
   : TimeDependentOperator(0, IMPLICIT), kappa(kappa_), sigma(sigma_),
     q_space(q_space_), p_space(p_space_),
     E_space(E_space_), B_space(B_space_), tr_space(tr_space_),
     imex_plasma(imex_)
{
   offsets = ConstructOffsets(q_space, p_space, E_space, B_space, tr_space);

   width = height = offsets.Last();

   //plasma

   const Mesh *mesh = q_space->GetMesh();
   const int dim = mesh->Dimension();
   const int num_equations = 1 + dim;
   const bool fec_disc = (q_space->FEColl()->GetContType() ==
                          FiniteElementCollection::DISCONTINUOUS);
   const bool fec_vec = (q_space->FEColl()->GetRangeType(dim) ==
                         FiniteElement::VECTOR);
   const bool dg = (fec_disc && !fec_vec);
   const bool brt = (fec_disc && fec_vec);

   /*if (!dg && !brt)
   {
      q_space->GetEssentialTrueDofs(bdr_q_is_ess, ess_q_tdofs_list);
   }*/

   //bcs
   const int nbdrs = (mesh->bdr_attributes.Size() > 0)?
                     (mesh->bdr_attributes.Max()):(1);
   MFEM_ASSERT(bcs_plasma.size() == (size_t)nbdrs, "Wrong number of BCs");
   bdr_is_dirichlet_plasma.resize(nbdrs);
   bdr_coeffs_plasma.resize(nbdrs);
   bdr_gcoeffs_plasma.resize(nbdrs);

   for (int b = 0; b < nbdrs; b++)
   {
      switch (bcs_plasma[b].first)
      {
         case BCType::Zero: continue;
         case BCType::Dirichlet:
            bdr_is_dirichlet_plasma[b].SetSize(nbdrs);
            bdr_is_dirichlet_plasma[b] = 0;
            bdr_is_dirichlet_plasma[b][b] = -1;
            bdr_coeffs_plasma[b] = bcs_plasma[b].second;
            MFEM_ASSERT(bdr_coeffs_plasma[b], "Dirichlet BC is null.");
            bdr_gcoeffs_plasma[b].reset(new ScalarVectorProductCoefficient(
                                           -1., *bdr_coeffs_plasma[b]));
            break;
         default:
            MFEM_ABORT("Not implemented");
      }
   }

   //(bi)linear forms
   darcy = new DarcyForm(q_space, p_space);

   Mq = darcy->GetFluxMassForm();
   Mpnl = darcy->GetPotentialMassNonlinearForm();
   Kp = (imex_plasma)?(new NonlinearForm(p_space)):(Mpnl);
   Dq = darcy->GetFluxDivForm();

   bq = darcy->GetFluxRHS();
   bp = darcy->GetPotentialRHS();

   MFEM_ASSERT(kappa.size() == (size_t)num_equations,
               "Not the right number of coefficients");
   ikappa.resize(num_equations);
   for (int i = 0; i < num_equations; i++)
   {
      ikappa[i] = std::make_unique<RatioCoefficient>(1., *kappa[i]);
   }

   // flux mass
   {
      std::vector<BilinearFormIntegrator*> bfis(num_equations);
      if (dg)
      {
         for (int i = 0; i < num_equations; i++)
         {
            bfis[i] = new VectorMassIntegrator(*ikappa[i]);
         }
      }
      else
      {
         for (int i = 0; i < num_equations; i++)
         {
            bfis[i] = new VectorFEMassIntegrator(*ikappa[i]);
         }
      }
      Mq->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(bfis));
   }

   // flux divergence
   if (dg)
   {
      Dq->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                                new VectorDivergenceIntegrator()));
   }
   else
   {
      Dq->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                                new VectorFEDivergenceIntegrator()));
   }

   if (dg || brt)
   {
      Dq->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                       num_equations, new TransposeIntegrator(new DGNormalTraceIntegrator(-1.))));
   }

   // linear diffusion stabilization
   if (dg && td > 0.)
   {
      std::vector<BilinearFormIntegrator*> bfis(num_equations);
      for (int i = 0; i < num_equations; i++)
      {
         bfis[i] = new HDGDiffusionIntegrator(*kappa[i], td);
      }
      Mpnl->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(bfis));
   }

   // potential mass term
   idtcoeff = std::make_unique<FunctionCoefficient>([&](const Vector &) { return idt; });
   Mpnl->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                               new MassIntegrator(*idtcoeff)));

   Mpdt = new BilinearForm(p_space);
   Mpdt->AddDomainIntegrator(new VectorBlockDiagonalIntegrator(num_equations,
                                                               new MassIntegrator()));
   Mpdt->Assemble();
   Mpdt->Finalize();

   // nonlinear convection
   flux = std::make_unique<IsothermalFlux>(dim, cs);
   numericalFlux = std::make_unique<RusanovFlux>(*flux);
   Kp->AddDomainIntegrator(new HyperbolicFormIntegrator(*numericalFlux, 0, -1));
   Kp->AddInteriorFaceIntegrator(new HyperbolicFormIntegrator(*numericalFlux, 0,
                                                              -1));
   for (int b = 0; b < nbdrs; b++)
   {
      if (!bdr_is_dirichlet_plasma[b].Size()) { continue; }
      Kp->AddBdrFaceIntegrator(new BdrHyperbolicDirichletIntegrator(
                                  *numericalFlux, *bdr_coeffs_plasma[b], 0, -1.),
                               bdr_is_dirichlet_plasma[b]);
   }

   //dirichlet BC
   for (int b = 0; b < nbdrs; b++)
   {
      if (!bdr_is_dirichlet_plasma[b].Size()) { continue; }
      if (dg)
      {
         bq->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(
                                     *bdr_gcoeffs_plasma[b]), bdr_is_dirichlet_plasma[b]);
      }
      else if (brt)
      {
         bq->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(
                                     *bdr_gcoeffs_plasma[b]), bdr_is_dirichlet_plasma[b]);
      }
      else
      {
         bq->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(
                                      *bdr_gcoeffs_plasma[b]), bdr_is_dirichlet_plasma[b]);
      }

   }

   //set hybridization

   if (tr_space)
   {
      darcy->EnableHybridization(tr_space,
                                 new VectorBlockDiagonalIntegrator(
                                    num_equations, new NormalTraceJumpIntegrator()),
                                 ess_q_tdofs_list);
   }

   darcy->Assemble();

   //Maxwell

   //bcs
   MFEM_ASSERT(bcs_maxwell.size() == (size_t)nbdrs, "Wrong number of BCs");
   //bdr_is_neumann_maxwell.resize(nbdrs);
   //bdr_coeffs_maxwell.resize(nbdrs);
   Array<int> bdr_E_is_ess(nbdrs);
   bdr_E_is_ess = 0;

   for (int b = 0; b < nbdrs; b++)
   {
      switch (bcs_maxwell[b].first)
      {
         case BCType::Free:
            bdr_E_is_ess[b] = -1;
            break;
         case BCType::Neumann:
            //bdr_is_neumann_maxwell[b].SetSize(nbdrs);
            //bdr_is_neumann_maxwell[b] = 0;
            //bdr_is_neumann_maxwell[b][b] = -1;
            bdr_E_is_ess[b] = -1;
            //bdr_coeffs_maxwell[b] = bcs_maxwell[b].second;
            break;
         case BCType::Zero:
            break;
         default:
            MFEM_ABORT("Not implemented");
      }
   }
   E_space->GetEssentialTrueDofs(bdr_E_is_ess, ess_E_tdofs_list);

   //(bi)linear forms
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
   delete Mpdt;
   if (imex_plasma) { delete Kp; }
   delete ME;
   delete CE;
   delete CdE;
   delete darcy;
}

Array<int> CoupledOperator::ConstructOffsets(
   const FiniteElementSpace *q_space, const FiniteElementSpace *p_space,
   const FiniteElementSpace *E_space, const FiniteElementSpace *B_space,
   const FiniteElementSpace *trace_space)
{
   Array<int> offsets((trace_space)?(6):(5));
   int i = 0;
   offsets[i++] = 0;
   offsets[i++] = q_space->GetVSize();
   offsets[i++] = p_space->GetVSize();
   if (trace_space)
   {
      offsets[i++] = trace_space->GetVSize();
   }
   offsets[i++] = E_space->GetVSize();
   offsets[i++] = B_space->GetVSize();
   offsets.PartialSum();

   return offsets;
}

void CoupledOperator::ProjectIC(BlockVector &x, VectorCoefficient &p) const
{
   GridFunction p_h(darcy->PotentialFESpace(), x.GetBlock(1), 0);
   p_h.ProjectCoefficient(p); //initial condition
   if (tr_space)
   {
      darcy->GetHybridization()->ProjectSolution(x, x.GetBlock(2));
   }
}

void CoupledOperator::ImplicitSolve(const double dt, const Vector &x, Vector &y)
{
   const bool time_track = false;

   BlockVector bx(const_cast<Vector&>(x), offsets);
   BlockVector by(y, offsets);

   int i = 1;
   //const Vector &qn = bx.GetBlock(i++);
   const Vector &pn = bx.GetBlock(i++);
   if (tr_space) { i++; }
   const Vector &En = bx.GetBlock(i++);
   const Vector &Bn = bx.GetBlock(i++);

   i = 2;
   //Vector &q = by.GetBlock(i++);
   //Vector &p = by.GetBlock(i++);
   if (tr_space) { i++; }
   Vector &E = by.GetBlock(i++);
   Vector &B = by.GetBlock(i++);

   //reduced offsets

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

   //plasma

   //set time

   for (int b = 0; b < (int)bdr_coeffs_plasma.size(); b++)
   {
      if (bdr_coeffs_plasma[b]) { bdr_coeffs_plasma[b]->SetTime(t); }
   }

   /*for (int b = 0; b < (int)bdr_coeffs_maxwell.size(); b++)
   {
      if (bdr_coeffs_maxwell[b]) { bdr_coeffs_maxwell[b]->SetTime(t); }
   }*/

   idt = 1./dt;

   //assemble rhs

   StopWatch chrono;
   if (time_track)
   {
      chrono.Clear();
      chrono.Start();
   }

   bq->Assemble();

   if (Mpdt)
   {
      GridFunction p_h;
      p_h.MakeRef(darcy->PotentialFESpace(), const_cast<Vector&>(pn), 0);
      if (imex_plasma)
      {
         Kp->Mult(p_h, *bp);
         Mpdt->AddMult(p_h, *bp, -idt);
      }
      else
      {
         Mpdt->Mult(p_h, *bp);
         *bp *= -idt;
      }
   }

   //form the system operator

   OperatorHandle op_pl;
   BlockVector darcy_x(const_cast<Vector&>(x), darcy->GetOffsets());
   Vector X_pl;
   Vector RHS_pl;

   if (tr_space)
   {
      const Vector darcy_tr(const_cast<Vector&>(bx.GetBlock(2)), 0,
                            bx.GetBlock(2).Size());
      X_pl.SetSize(darcy_tr.Size());
      X_pl = darcy_tr;
   }

   darcy->FormLinearSystem(ess_q_tdofs_list, darcy_x,
                           op_pl, X_pl, RHS_pl, true);

   //solution & rhs vectors

   BlockVector X(X_offsets), RHS(X_offsets);
   X.GetBlock(0) = X_pl;
   RHS.GetBlock(0) = RHS_pl;
   X_pl.MakeRef(X.GetBlock(0), 0);
   RHS_pl.MakeRef(RHS.GetBlock(0), 0);

   if (time_track)
   {
      chrono.Stop();
      std::cout << "Assembly took " << chrono.RealTime() << "s.\n";
   }

   // preconditioning
   BlockDiagonalPreconditioner bprec(X_offsets);
   bprec.owns_blocks = true;

   //SparseMatrix sp_grad;
   if (tr_space)
   {
      //sp_grad = static_cast<SparseMatrix&>(op_pl->GetGradient(X_pl));
      //bprec.SetDiagonalBlock(0, new GSSmoother(sp_grad));
      //const SparseMatrix &sp_grad = static_cast<const SparseMatrix&>
      //                              (op_pl->GetGradient(X_pl));
      //bprec.SetDiagonalBlock(0, new GSSmoother(sp_grad));
   }
   else
   {
      BlockDiagonalPreconditioner *darcy_prec = new BlockDiagonalPreconditioner(
         darcy->GetOffsets());
      darcy_prec->owns_blocks = true;
      darcy_prec->SetDiagonalBlock(0, new GSSmoother(Mq->SpMat()));
      darcy_prec->SetDiagonalBlock(1, new DSmoother(Mpdt->SpMat(), 0, idt));
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

   ConstrainedOperator op_max(&MECtCd, ess_E_tdofs_list);

   bprec.SetDiagonalBlock(1, new DSmoother(ME->SpMat()));

   //coupling

   ReducedOperator bop(sigma, darcy, E_space, *op_pl, op_max);
   if (tr_space)
   {
      Array<int> ess_tr_tdofs_list;//dummy
      bop.SetEssentialTDOFs(ess_tr_tdofs_list, ess_E_tdofs_list);
   }
   else
   {
      bop.SetEssentialTDOFs(ess_q_tdofs_list, ess_E_tdofs_list);
   }
   //bop.EliminateRHS(X, RHS);

   //solve

   constexpr real_t rtol = 1e-6;

   if (tr_space)
   {
      darcy->GetHybridization()->SetLocalNLSolver(
         DarcyHybridization::LSsolveType::Newton,
         1000, rtol * 1e-3);
      darcy->GetHybridization()->SetLocalNLPreconditioner(
         DarcyHybridization::LPrecType::GMRES);
   }

   GMRESSolver solver;
   solver.SetMaxIter(1000);
   solver.SetAbsTol(0.);
   solver.SetRelTol(rtol * 1e-1);
   solver.SetOperator(bop);
   solver.SetPreconditioner(bprec);
   solver.SetPrintLevel(0);

   NewtonSolver newton;
   newton.SetMaxIter(1000);
   newton.SetAbsTol(0.);
   newton.SetRelTol(rtol);
   newton.SetOperator(bop);
   newton.SetSolver(solver);
   newton.SetPrintLevel(1);

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
      darcy->RecoverFEMSolution(X_pl, darcy_y);
      Vector &darcy_tr = by.GetBlock(2);
      darcy_tr = X_pl;
      darcy_tr -= bx.GetBlock(2);
      darcy_tr *= idt;
   }
   else
   {
      BlockVector bX_pl(X_pl, darcy->GetOffsets());
      darcy->RecoverFEMSolution(X_pl, bX_pl);
      darcy_y = bX_pl;
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
CoupledOperator::ReducedOperator::ReducedOperator(
   Coefficient *sigma_, DarcyForm *darcy_, FiniteElementSpace *fes_E_,
   Operator &pl_, Operator &max_)
   : sigma(sigma_), darcy(darcy_), fes_E(fes_E_), op_pl(pl_), op_max(max_)
{
   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = op_pl.Width();
   offsets[2] = op_max.Width();
   offsets.PartialSum();

   width = height = offsets.Last();

   if (darcy->GetHybridization())
   {
      offsets_x = offsets;
      darcy_bp = darcy->GetPotentialRHS();
      darcy_bp_lin = *darcy_bp;
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
   bop->SetDiagonalBlock(0, &op_pl);
   bop->SetDiagonalBlock(1, &op_max);

   op.Reset(bop);
}

CoupledOperator::ReducedOperator::~ReducedOperator()
{
}

void CoupledOperator::ReducedOperator::SetEssentialTDOFs(
   const Array<int> &u_tdofs_list, const Array<int> &E_tdofs_list)
{
   ess_tdofs_list.DeleteAll();
   //ess_tdofs_list.Append(u_tdofs_list);
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
   /*Vector z(x.Size());
   z = x;

   for (int tdof : ess_tdofs_list)
   {
      z(tdof) = 0.;
   }*/

   MultUnconstrained(x, y);

   for (int tdof : ess_tdofs_list)
   {
      y(tdof) = x(tdof);
   }
}

void CoupledOperator::ReducedOperator::MultUnconstrained(const Vector &x,
                                                         Vector &y) const
{
   const bool hybr = darcy->GetHybridization() != NULL;

   if (hybr)
   {
      y = 0.;
   }
   else
   {
      op->Mult(x, y);
   }

   BlockVector bx(const_cast<Vector&>(x), offsets_x);
   BlockVector by(const_cast<Vector&>(y), offsets_x);
   BlockVector darcy_x;

   if (hybr)
   {
      darcy_x.Update(darcy->GetOffsets());
      darcy->RecoverFEMSolution(bx.GetBlock(0), darcy_x);
      *darcy_bp = darcy_bp_lin;
   }

   const Vector &xp = ((hybr)?(darcy_x):(bx)).GetBlock(1);
   const int dim = darcy->PotentialFESpace()->GetMesh()->Dimension();
   const int vdim = darcy->PotentialFESpace()->GetVDim();
   MFEM_ASSERT(vdim == dim+1, "Wrong number of dimensions");
   const int size = darcy->PotentialFESpace()->GetNDofs();
   const Vector xn(const_cast<Vector&>(xp), 0, size);
   const Vector &xE = bx.GetBlock((hybr)?(1):(2));
   Vector &yp = ((hybr)?(*darcy_bp):(by.GetBlock(1)));
   Vector &yE = by.GetBlock((hybr)?(1):(2));

   Mesh *mesh = fes_E->GetMesh();
   FiniteElementSpace *fes_p = darcy->PotentialFESpace();
   Array<int> vdofs_u, dofs_p, vdofs_E;
   DenseMatrix vshape_E;
   Vector shape_E, shape_p, n_z, E_z, E, bu_z, bE_z;

   for (int z = 0; z < mesh->GetNE(); z++)
   {
      ElementTransformation *Tr = mesh->GetElementTransformation(z);
      const FiniteElement *fe_p = fes_p->GetFE(z);
      const FiniteElement *fe_E = fes_E->GetFE(z);
      const int sdim = Tr->GetSpaceDim();
      const int ndof_p = fe_p->GetDof();

      fes_p->GetElementDofs(z, dofs_p);
      fes_E->GetElementVDofs(z, vdofs_E);

      shape_p.SetSize(ndof_p);
      vshape_E.SetSize(vdofs_E.Size(), sdim);
      shape_E.SetSize(vdofs_E.Size());
      E.SetSize(sdim);

      bu_z.SetSize(dofs_p.Size() * dim);
      bE_z.SetSize(vdofs_E.Size());
      bu_z = 0.;
      bE_z = 0.;

      xn.GetSubVector(dofs_p, n_z);
      xE.GetSubVector(vdofs_E, E_z);

      const int order = std::max(fe_E->GetOrder(), fe_p->GetOrder()) * 2 + 1;
      const IntegrationRule &ir = IntRules.Get(fe_p->GetGeomType(), order);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         fe_p->CalcShape(ip, shape_p);
         const real_t n = n_z * shape_p;

         fe_E->CalcVShape(*Tr, vshape_E);
         vshape_E.MultTranspose(E_z, E);

         real_t w = n * Tr->Weight();
         if (sigma) { w *= sigma->Eval(*Tr, ip); }

         for (int d = 0; d < sdim; d++)
            for (int i = 0; i < ndof_p; i++)
            {
               bu_z(i+d*ndof_p) += w * E(d) * shape_p(i);
            }
         vshape_E.Mult(E, shape_E);
         bE_z.Add(w, shape_E);
      }

      if (hybr) { bu_z.Neg(); }

      for (int d = 0; d < dim; d++)
      {
         vdofs_u = dofs_p;
         fes_p->DofsToVDofs(d+1, vdofs_u);
         const Vector bu_zd(bu_z, d*ndof_p, ndof_p);
         yp.AddElementVector(vdofs_u, bu_zd);
      }
      yE.AddElementVector(vdofs_E, bE_z);
   }

   if (hybr)
   {
      BlockVector darcy_Xrhs(darcy->GetFluxRHS()->GetData(), darcy->GetOffsets());
      darcy->GetHybridization()->ReduceRHS(darcy_Xrhs, by.GetBlock(0));
      op->AddMult(x, y);
   }
}

Operator &CoupledOperator::ReducedOperator::GetGradient(const Vector &x) const
{

   const bool hybr = darcy->GetHybridization() != NULL;

   const BlockVector bx(const_cast<Vector&>(x), offsets_x);

   BlockOperator *bgrad = new BlockOperator(offsets);
   if (hybr)
   {
      bgrad->SetBlock(0, 0, &op_pl.GetGradient(bx.GetBlock(0)));
   }
   else
   {
      const BlockVector darcy_x(const_cast<Vector&>(x), darcy->GetOffsets());
      bgrad->SetBlock(0, 0, &op_pl.GetGradient(darcy_x));
   }
   bgrad->SetBlock(1, 1, &op_max);
   grad.Reset(bgrad);
   return *grad;

   //if (darcy->GetHybridization())
   /*{
      //TODO: coupling terms
      //return const_cast<Operator&>(*op);
      return op->GetGradient(x);
   }*/

   /*grad.Clear();

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

   return *grad;*/
}
} // namespace plasma
} // namespace mfem
