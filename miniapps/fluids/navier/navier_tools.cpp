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

#include "navier_tools.hpp"

#ifdef MFEM_USE_GSLIB

using namespace std;

namespace mfem
{
namespace navier
{

      void ComputeCurl3D_particle(ParGridFunction &u, ParGridFunction &cu)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         curl.SetSize(3);
         curl(0) = grad(2, 1) - grad(1, 2);
         curl(1) = grad(0, 2) - grad(2, 0);
         curl(2) = grad(1, 0) - grad(0, 1);

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof = vdofs[j];
         cu(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}

DiffusionSolver_particle::DiffusionSolver_particle(Mesh * mesh_, int order_,
                                 Coefficient * diffcf_, Coefficient * rhscf_)
   : mesh(mesh_), order(order_), diffcf(diffcf_), rhscf(rhscf_)
{

#ifdef MFEM_USE_MPI
   pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh) { parallel = true; }
#endif

   SetupFEM();
}

void DiffusionSolver_particle::SetupFEM()
{
   dim = mesh->Dimension();
   fec = new H1_FECollection(order, dim);

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes = new ParFiniteElementSpace(pmesh, fec);
      u = new ParGridFunction(pfes);
      b = new ParLinearForm(pfes);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, fec);
      u = new GridFunction(fes);
      b = new LinearForm(fes);
   }
#else
   fes = new FiniteElementSpace(mesh, fec);
   u = new GridFunction(fes);
   b = new LinearForm(fes);
#endif
   *u=0.0;

   if (!ess_bdr.Size())
   {
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
   }
}

void DiffusionSolver_particle::UpdateEssentialTDofs()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
   else
   {
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
#else
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
#endif
}

void DiffusionSolver_particle::AssembleDiffusionBilinear(bool update_ess_tdofs)
{
   if (update_ess_tdofs)
   {
      UpdateEssentialTDofs();
   }
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      a = new ParBilinearForm(pfes);
   }
   else
   {
      a = new BilinearForm(fes);
   }
#else
   a = new BilinearForm(fes);
#endif
   a->AddDomainIntegrator(new DiffusionIntegrator(*diffcf));
   if (masscf)
   {
      a->AddDomainIntegrator(new MassIntegrator(*masscf));
   }
   a->Assemble();
   a->FormSystemMatrix(ess_tdof_list, A);
}

void DiffusionSolver_particle::Solve()
{
   Vector B, X;

   if (b)
   {
      delete b;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         b = new ParLinearForm(pfes);
      }
      else
      {
         b = new LinearForm(fes);
      }
#else
      b = new LinearForm(fes);
#endif
   }
   if (rhscf)
   {
      b->AddDomainIntegrator(new DomainLFIntegrator(*rhscf));
   }
   if (neumann_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryLFIntegrator(*neumann_cf),neumann_bdr);
   }
   else if (gradient_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*gradient_cf),
                               neumann_bdr);
   }

   b->Assemble();

   *u=0.0;
   if (essbdr_cf)
   {
      u->ProjectBdrCoefficient(*essbdr_cf,ess_bdr);
   }

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      X.SetSize(pfes->TrueVSize());
      B.SetSize(pfes->TrueVSize());
      dynamic_cast<ParGridFunction*>(u)->ParallelAssemble(X);
      dynamic_cast<ParLinearForm*>(b)->ParallelAssemble(B);
      dynamic_cast<ParBilinearForm*>(a)->ParallelEliminateTDofsInRHS(
         ess_tdof_list, X, B);
   }
   else
   {
      X.NewDataAndSize(u->GetData(), u->Size());
      B.NewDataAndSize(b->GetData(), b->Size());
      a->EliminateVDofsInRHS(ess_tdof_list, X, B);
   }
#else
   X.NewDataAndSize(u->GetData(), u->Size());
   B.NewDataAndSize(b->GetData(), b->Size());
   a->EliminateVDofsInRHS(ess_tdof_list, X, B);
#endif

   CGSolver * cg = nullptr;
   Solver * M = nullptr;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M = new HypreBoomerAMG;
      dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
      cg = new CGSolver(pmesh->GetComm());
   }
   else
   {
      M = new GSSmoother((SparseMatrix&)(*A));
      cg = new CGSolver;
   }
#else
   M = new GSSmoother((SparseMatrix&)(*A));
   cg = new CGSolver;
#endif
   cg->SetRelTol(1e-12);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   delete M;
   delete cg;
   a->RecoverFEMSolution(X, *b, *u);
}

GridFunction * DiffusionSolver_particle::GetFEMSolution()
{
   return u;
}

#ifdef MFEM_USE_MPI
ParGridFunction * DiffusionSolver_particle::GetParFEMSolution()
{
   if (parallel)
   {
      return dynamic_cast<ParGridFunction*>(u);
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
      return nullptr;
   }
}
#endif

DiffusionSolver_particle::~DiffusionSolver_particle()
{
   delete u; u = nullptr;
   delete fes; fes = nullptr;
#ifdef MFEM_USE_MPI
   delete pfes; pfes=nullptr;
#endif
   delete fec; fec = nullptr;
   delete b;
   A.Clear();
   delete a;
}

} // namespace navier
} // namespace mfem
#endif // MFEM_USE_GSLIB
