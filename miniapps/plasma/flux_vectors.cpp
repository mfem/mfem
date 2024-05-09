// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "flux_vectors.hpp"

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem::plasma;

ParFluxVectors::ParFluxVectors(ParMesh& pmesh, int order, bool pa)
   : df_active_(false)
   , af_active_(false)
   , m_rt_assm_(false)
   , pa_(pa)
   , dCoef_(NULL)
   , DCoef_(NULL)
   , VCoef_(NULL)
   , fes_rt_(&pmesh, order, pmesh.Dimension())
   , m_rt_(&fes_rt_)
{
   m_rt_.AddDomainIntegrator(new VectorFEMassIntegrator);
}

void ParFluxVectors::Update()
{
   fes_rt_.Update();
   m_rt_.Update();
   m_rt_assm_ = false;
   if (df_active_)
   {
      DGradU_lf_.Update();
      df_gf_.Update();
   }
   if (af_active_)
   {
      VProdU_lf_.Update();
      af_gf_.Update();
   }
}

void ParFluxVectors::Assemble()
{
   if (m_rt_assm_) { return; }

   if (df_active_ || af_active_)
   {
      m_rt_.Assemble();
      if (!pa_) { m_rt_.Finalize(); }
      m_rt_.FormSystemMatrix(ess_tdof_, M_rt_);

      X_.SetSize(fes_rt_.GetProlongationMatrix()->Width());
      RHS_.SetSize(X_.Size());
   }
   m_rt_assm_ = true;
}

void ParFluxVectors::SetDiffusionCoef(Coefficient &D)
{
   dCoef_ = &D;
   DCoef_ = NULL;
}

void ParFluxVectors::SetDiffusionCoef(MatrixCoefficient &D)
{
   dCoef_ = NULL;
   DCoef_ = &D;
}

void ParFluxVectors::SetAdvectionCoef(VectorCoefficient &V)
{
   VCoef_ = &V;
}
/*
void ParFluxVectors::Activate()
{
  m_rt_.AddDomainIntegrator(new VectorFEMassIntegrator);
  this->Assemble();
}
*/
void ParFluxVectors::ComputeDiffusiveFlux(const ParGridFunction& u)
{
   if (!df_active_)
   {
      DGradU_lf_.Update(&fes_rt_);

      if (df_gf_.Size() != fes_rt_.GetVSize())
      {
         df_gf_.SetSpace(&fes_rt_);
      }

      df_active_ = true;
   }

   df_gf_ = 0.0;

   if (dCoef_ == NULL && DCoef_ == NULL) { return; }

   Array<int> vdofs;
   ElementTransformation *Tr;
   DofTransformation *doftrans;
   const FiniteElement *el;

   DenseMatrix vshape, DMat;
   Vector elvec, GradU, DGradU;

   GradientGridFunctionCoefficient duCoef(&u);

   DGradU_lf_ = 0.0;

   for (int i = 0; i < fes_rt_.GetNE(); i++)
   {
      doftrans = fes_rt_.GetElementVDofs (i, vdofs);
      Tr = fes_rt_.GetElementTransformation (i);
      el = fes_rt_.GetFE(i);

      int dof = el->GetDof();
      int spaceDim = Tr->GetSpaceDim();
      int vdim = std::max(spaceDim, el->GetRangeDim());

      vshape.SetSize(dof, vdim);
      elvec.SetSize(dof);
      GradU.SetSize(vdim);
      DGradU.SetSize(vdim);

      elvec = 0.0;

      const IntegrationRule *ir = &IntRules.Get(el->GetGeomType(),
                                                2 * el->GetOrder() +
                                                Tr->OrderW());

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr->SetIntPoint (&ip);

         el->CalcVShape(*Tr, vshape);

         duCoef.Eval(GradU, *Tr, ip);
         double dval = (dCoef_) ? dCoef_->Eval(*Tr, ip) : 0.0;
         if (DCoef_)
         {
            DCoef_->Eval(DMat, *Tr, ip);
            DMat.Mult(GradU, DGradU);
         }
         else
         {
            DGradU.Set(dval, GradU);
         }
         double w = Tr->Weight() * ip.weight;

         vshape.AddMult(DGradU, elvec, w);
      }
      if (doftrans)
      {
         doftrans->TransformDual(elvec);
      }
      DGradU_lf_.AddElementVector (vdofs, elvec);
   }

   Operator *diag = NULL;
   Operator *pcg = NULL;
   this->Assemble();

   if (pa_)
   {
      diag = new OperatorJacobiSmoother(m_rt_, ess_tdof_);
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetOperator(*M_rt_);
      cg->SetPreconditioner(static_cast<OperatorJacobiSmoother&>(*diag));
      cg->SetRelTol(1e-12);
      cg->SetMaxIter(1000);
      pcg = cg;
   }
   else
   {
      diag = new HypreDiagScale(*M_rt_.As<HypreParMatrix>());
      HyprePCG *cg = new HyprePCG(*M_rt_.As<HypreParMatrix>());
      cg->SetPreconditioner(static_cast<HypreDiagScale&>(*diag));
      cg->SetTol(1e-12);
      cg->SetMaxIter(1000);
      pcg = cg;
   }

   DGradU_lf_.ParallelAssemble(RHS_);
   X_ = 0.0;
   pcg->Mult(RHS_, X_);
   X_ *= -1.0;

   df_gf_.Distribute(X_);

   delete diag;
   delete pcg;
}

void ParFluxVectors::ComputeAdvectiveFlux(const ParGridFunction& u)
{
   if (!af_active_)
   {
      VProdU_lf_.Update(&fes_rt_);

      if (af_gf_.Size() != fes_rt_.GetVSize())
      {
         af_gf_.SetSpace(&fes_rt_);
      }

      af_active_ = true;
   }

   af_gf_ = 0.0;

   if (VCoef_ == NULL) { return; }

   Array<int> vdofs;
   ElementTransformation *Tr;
   DofTransformation *doftrans;
   const FiniteElement *el;

   DenseMatrix vshape;
   Vector elvec, Vval;

   GridFunctionCoefficient uCoef(&u);

   VProdU_lf_ = 0.0;

   for (int i = 0; i < fes_rt_.GetNE(); i++)
   {
      doftrans = fes_rt_.GetElementVDofs (i, vdofs);
      Tr = fes_rt_.GetElementTransformation (i);
      el = fes_rt_.GetFE(i);

      int dof = el->GetDof();
      int spaceDim = Tr->GetSpaceDim();
      int vdim = std::max(spaceDim, el->GetRangeDim());

      vshape.SetSize(dof, vdim);
      elvec.SetSize(dof);
      Vval.SetSize(vdim);

      elvec = 0.0;

      const IntegrationRule *ir = &IntRules.Get(el->GetGeomType(),
                                                2 * el->GetOrder() +
                                                Tr->OrderW());

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr->SetIntPoint (&ip);

         el->CalcVShape(*Tr, vshape);

         VCoef_->Eval(Vval, *Tr, ip);
         double uval = uCoef.Eval(*Tr, ip);

         double w = Tr->Weight() * ip.weight;

         vshape.AddMult(Vval, elvec, w * uval);
      }
      if (doftrans)
      {
         doftrans->TransformDual(elvec);
      }
      VProdU_lf_.AddElementVector (vdofs, elvec);
   }

   Operator *diag = NULL;
   Operator *pcg = NULL;
   if (pa_)
   {
      diag = new OperatorJacobiSmoother(m_rt_, ess_tdof_);
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetOperator(*M_rt_);
      cg->SetPreconditioner(static_cast<OperatorJacobiSmoother&>(*diag));
      cg->SetRelTol(1e-12);
      cg->SetMaxIter(1000);
      pcg = cg;
   }
   else
   {
      diag = new HypreDiagScale(*M_rt_.As<HypreParMatrix>());
      HyprePCG *cg = new HyprePCG(*M_rt_.As<HypreParMatrix>());
      cg->SetPreconditioner(static_cast<HypreDiagScale&>(*diag));
      cg->SetTol(1e-12);
      cg->SetMaxIter(1000);
      pcg = cg;
   }

   VProdU_lf_.ParallelAssemble(RHS_);
   X_ = 0.0;
   pcg->Mult(RHS_, X_);

   af_gf_.Distribute(X_);

   delete diag;
   delete pcg;
}

ParGridFunction & ParFluxVectors::GetDiffusiveFlux()
{
   if (df_gf_.Size() != fes_rt_.GetVSize())
   {
      df_gf_.SetSpace(&fes_rt_);
   }
   return df_gf_;
}

ParGridFunction & ParFluxVectors::GetAdvectiveFlux()
{
   if (af_gf_.Size() != fes_rt_.GetVSize())
   {
      af_gf_.SetSpace(&fes_rt_);
   }
   return af_gf_;
}

#endif
