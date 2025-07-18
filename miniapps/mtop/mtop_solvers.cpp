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

#include <memory>

#include "mtop_solvers.hpp"

using namespace mfem;

IsoLinElasticSolver::IsoLinElasticSolver(ParMesh *mesh_, int vorder)
{
   pmesh = mesh_;
   const int dim = pmesh->Dimension();
   vfec = new H1_FECollection(vorder, dim);
   vfes =
      new ParFiniteElementSpace(pmesh, vfec, dim, Ordering::byVDIM);

   fdisp.SetSpace(vfes);
   fdisp = 0.0;
   adisp.SetSpace(vfes);
   adisp = 0.0;

   sol.SetSize(vfes->GetTrueVSize());
   sol = 0.0;
   rhs.SetSize(vfes->GetTrueVSize());
   rhs = 0.0;
   adj.SetSize(vfes->GetTrueVSize());
   adj = 0.0;

   SetLinearSolver();

   prec = nullptr;
   ls = nullptr;

   Operator::width = vfes->GetTrueVSize();
   Operator::height = vfes->GetTrueVSize();

   lvforce = nullptr;
   volforce = nullptr;

   E = nullptr;
   nu = nullptr;

   lambda = nullptr;
   mu = nullptr;

   bf = nullptr;
   lf = nullptr;

   lcsurf_load = std::make_unique<SurfaceLoad>(dim, load_coeff);
   glsurf_load = std::make_unique<SurfaceLoad>(dim, surf_loads);
}

IsoLinElasticSolver::~IsoLinElasticSolver()
{
   delete prec;
   delete ls;

   delete bf;
   delete lf;

   delete vfes;
   delete vfec;

   delete lvforce;

   for (auto it = load_coeff.begin(); it != load_coeff.end(); it++)
   {
      delete it->second;
   }

   delete lambda;
   delete mu;
}

void IsoLinElasticSolver::SetLinearSolver(real_t rtol, real_t atol,
                                          int miter)
{
   linear_rtol = rtol;
   linear_atol = atol;
   linear_iter = miter;
}

void IsoLinElasticSolver::AddDispBC(int id, int dir, real_t val)
{
   if (dir == 0)
   {
      bcx[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcx[id]);
   }
   if (dir == 1)
   {
      bcy[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcy[id]);
   }
   if (dir == 2)
   {
      bcz[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcz[id]);
   }
   if (dir == 4)
   {
      bcx[id] = ConstantCoefficient(val);
      bcy[id] = ConstantCoefficient(val);
      bcz[id] = ConstantCoefficient(val);
      AddDispBC(id, 0, bcx[id]);
      AddDispBC(id, 1, bcy[id]);
      AddDispBC(id, 2, bcz[id]);
   }
}

void IsoLinElasticSolver::DelDispBC()
{
   bccx.clear();
   bccy.clear();
   bccz.clear();

   bcx.clear();
   bcy.clear();
   bcz.clear();

   ess_tdofv.DeleteAll();
}

void IsoLinElasticSolver::AddDispBC(int id, int dir, Coefficient &val)
{
   if (dir == 0) { bccx[id] = &val; }
   if (dir == 1) { bccy[id] = &val; }
   if (dir == 2) { bccz[id] = &val; }
   if (dir == 4)
   {
      bccx[id] = &val;
      bccy[id] = &val;
      bccz[id] = &val;
   }
   if (pmesh->Dimension() == 2) { bccz.clear(); }
}

void IsoLinElasticSolver::SetVolForce(real_t fx, double fy, double fz)
{
   delete lvforce;
   int dim = pmesh->Dimension();
   Vector ff(dim);
   ff(0) = fx;
   ff(1) = fy;
   if (dim == 3) { ff(2) = fz; }
   lvforce = new VectorConstantCoefficient(ff);
   volforce = lvforce;
}

void IsoLinElasticSolver::SetVolForce(VectorCoefficient &fv)
{
   volforce = &fv;
}

void IsoLinElasticSolver::SetEssTDofs(Vector &bsol,
                                      Array<int> &ess_dofs)
{
   // Set the BC

   ess_tdofv.DeleteAll();

   Array<int> ess_tdofx;
   Array<int> ess_tdofy;
   Array<int> ess_tdofz;

   int dim = pmesh->Dimension();

   {
      for (auto it = bccx.begin(); it != bccx.end(); it++)
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 0;
         ess_bdr[it->first - 1] = 1;
         Array<int> ess_tdof_list;
         vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 0);
         ess_tdofx.Append(ess_tdof_list);

         VectorArrayCoefficient pcoeff(dim);
         pcoeff.Set(0, it->second, false);
         fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
      }
      // copy tdofsx from velocity grid function
      {
         Vector &vc = fdisp.GetTrueVector();
         for (int ii = 0; ii < ess_tdofx.Size(); ii++)
         {
            bsol[ess_tdofx[ii]] = vc[ess_tdofx[ii]];
         }
      }
      ess_dofs.Append(ess_tdofx);
      ess_tdofx.DeleteAll();

      for (auto it = bccy.begin(); it != bccy.end(); it++)
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 0;
         ess_bdr[it->first - 1] = 1;
         Array<int> ess_tdof_list;
         vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 1);
         ess_tdofy.Append(ess_tdof_list);

         VectorArrayCoefficient pcoeff(dim);
         pcoeff.Set(1, it->second, false);
         fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
      }
      // copy tdofsy from velocity grid function
      {
         Vector &vc = fdisp.GetTrueVector();
         for (int ii = 0; ii < ess_tdofy.Size(); ii++)
         {
            bsol[ess_tdofy[ii]] = vc[ess_tdofy[ii]];
         }
      }
      ess_dofs.Append(ess_tdofy);
      ess_tdofy.DeleteAll();

      if (dim == 3)
      {
         for (auto it = bccz.begin(); it != bccz.end(); it++)
         {
            Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr = 0;
            ess_bdr[it->first - 1] = 1;
            Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 2);
            ess_tdofz.Append(ess_tdof_list);

            VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(2, it->second, false);
            fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
         }

         // copy tdofsz from velocity grid function
         {
            Vector &vc = fdisp.GetTrueVector();
            for (int ii = 0; ii < ess_tdofz.Size(); ii++)
            {
               bsol[ess_tdofz[ii]] = vc[ess_tdofz[ii]];
            }
         }
         ess_dofs.Append(ess_tdofz);
         ess_tdofz.DeleteAll();
      }
   }
}

void IsoLinElasticSolver::Mult(const Vector &x, Vector &y) const
{
   // the rhs x is assumed to have the contribution of the BC set in advance
   // the BC values are not modified here
   ls->Mult(x, y);

   int N = ess_tdofv.Size();
   real_t *yp = y.ReadWrite();
   const real_t *sp = sol.Read();
   const int *ep = ess_tdofv.Read();
   forall(N, [=] MFEM_HOST_DEVICE(int i) { yp[ep[i]] = sp[ep[i]]; });
}

void IsoLinElasticSolver::MultTranspose(const Vector &x,
                                        Vector &y) const
{
   // the adjoint rhs is assumed to be corrected for the BC
   // K is symmetric
   ls->Mult(x, y);

   int N = ess_tdofv.Size();
   ess_tdofv.Read();

   real_t *yp = y.ReadWrite();
   const int *ep = ess_tdofv.Read();

   forall(N,
   [=] MFEM_HOST_DEVICE(int i) { yp[ep[i]] = real_t(0.0); });
}

void IsoLinElasticSolver::Assemble()
{
   if (bf == nullptr) { return; }

   // set BC
   sol = real_t(0.0);
   SetEssTDofs(sol, ess_tdofv);

   double nr = ParNormlp(sol, 2, pmesh->GetComm());
   if (pmesh->GetMyRank() == 0) { std::cout << "sol=" << nr << std::endl; }

   bf->Assemble(0);
   bf->Finalize();
   // bf->FormSystemMatrix(ess_tdofv,K);
   K.reset(bf->ParallelAssemble());
   Ke.reset(K->EliminateRowsCols(ess_tdofv));

   if (ls == nullptr)
   {
      ls = new CGSolver(pmesh->GetComm());
      ls->SetAbsTol(linear_atol);
      ls->SetRelTol(linear_rtol);
      ls->SetMaxIter(linear_iter);
      prec = new HypreBoomerAMG();
      // set the rigid body modes
      prec->SetElasticityOptions(vfes);
      prec->SetPrintLevel(1);
      ls->SetPreconditioner(*prec);
      ls->SetOperator(*K);
      ls->SetPrintLevel(1);
   }
   else { ls->SetOperator(*K); }

   std::cout << pmesh->GetMyRank() << " LSW=" << ls->Width()
             << " LSH=" << ls->Height() << " KFW=" << K->Width()
             << " KFH=" << K->Height() << std::endl;
}

void IsoLinElasticSolver::FSolve()
{
   if (lf == nullptr)
   {
      lf = new ParLinearForm(vfes);
      if (volforce != nullptr)
      {
         lf->AddDomainIntegrator(new VectorDomainLFIntegrator(*volforce));
      }
      // add surface loads
      lf->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*lcsurf_load));
   }

   (*lf) = real_t(0.0);

   lf->Assemble();
   lf->ParallelAssemble(rhs);

   /*
   Ke->AddMult(sol,rhs,1.0);
   for(int i=0;i<ess_tdofv.Size();i++){
       rhs[ess_tdofv[i]]=-sol[ess_tdofv[i]];
   }*/

   K->EliminateBC(*Ke, ess_tdofv, sol, rhs);

   {
      double nr = ParNormlp(rhs, 2, pmesh->GetComm());
      if (pmesh->GetMyRank() == 0) { std::cout << "rhs=" << nr << std::endl; }
   }

   ls->Mult(rhs, sol);

   delete lf;
   lf = nullptr;
}
