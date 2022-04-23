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

#include "dgmassinv.hpp"
#include "bilinearform.hpp"
#include "dgmassinv_kernels.hpp"
#include "../general/forall.hpp"

namespace mfem
{

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_orig, Coefficient *coeff,
                             int btype)
   : Solver(fes_orig.GetTrueVSize()),
     fec(fes_orig.GetMaxElementOrder(), fes_orig.GetMesh()->Dimension(), btype),
     fes(fes_orig.GetMesh(), &fec)
{
   MFEM_VERIFY(fes.IsDGSpace(), "Space must be DG.");
   MFEM_VERIFY(!fes.IsVariableOrder(), "Variable orders not supported.");

   int btype_orig = static_cast<const L2_FECollection*>
                    (fes_orig.FEColl())->GetBasisType();

   if (btype_orig == btype)
   {
      // No change of basis required
      d2q = nullptr;
   }
   else
   {
      const IntegrationRule &nodal_ir = fes.GetFE(0)->GetNodes();
      d2q = &fes_orig.GetFE(0)->GetDofToQuad(nodal_ir, DofToQuad::TENSOR);
   }

   if (coeff) { m = new MassIntegrator(*coeff); }
   else { m = new MassIntegrator; }

   BilinearForm M(&fes);
   M.AddDomainIntegrator(m);
   M.UseExternalIntegrators();
   M.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   M.Assemble();

   diag_inv.SetSize(height);
   M.AssembleDiagonal(diag_inv);

   // TODO: need to move this FORALL into its own function, can't have a
   // FORALL loop in constructor.
   // auto dinv = diag_inv.ReadWrite();
   //MFEM_FORALL(i, height, dinv[i] = 1.0/dinv[i]; );
   auto dinv = diag_inv.HostReadWrite();
   for (int i = 0; i < height; ++i) { dinv[i] = 1.0/dinv[i]; }

   r_.SetSize(height);
   d_.SetSize(height);
   z_.SetSize(height);
}

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_, int btype)
   : DGMassInverse(fes_, nullptr, btype) { }

void DGMassInverse::SetOperator(const Operator &op)
{
   MFEM_ABORT("SetOperator not supported with DGMassInverse.")
}

void DGMassInverse::SetRelTol(const double rel_tol_) { rel_tol = rel_tol_; }

void DGMassInverse::SetAbsTol(const double abs_tol_) { abs_tol = abs_tol_; }

void DGMassInverse::SetMaxIter(const double max_iter_) { max_iter = max_iter_; }

DGMassInverse::~DGMassInverse()
{
   delete m;
}

template<int DIM, int D1D, int Q1D>
void DGMassInverse::DGMassCGIteration(const Vector &b_, Vector &u_) const
{
   const int NE = fes.GetNE();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;

   const int ND = pow(d1d, DIM);

   const auto B = m->maps->B.Read();
   const auto Bt = m->maps->Bt.Read();
   const auto pa_data = m->pa_data.Read();
   const auto dinv = diag_inv.Read();
   auto r = r_.Write();
   auto d = d_.Write();
   auto z = z_.Write();
   auto u = u_.ReadWrite();

   const double RELTOL = rel_tol;
   const double ABSTOL = abs_tol;
   const double MAXIT = max_iter;

   // const bool change_basis = (d2q != nullptr);

   // const double *b, *b_orig;
   // if (change_basis)
   // {
   //    b = b2_.Write();
   //    b_orig = b_.Read();
   // }
   // else
   // {
   //    b = b_.Read();
   //    b_orig = nullptr;
   // }

   const double *b = b_.Read();

   const int NB = Q1D ? Q1D : 1; // block size

   // printf("  El.     It.    (Br,r)\n");
   // printf("=============================\n");
   MFEM_FORALL_2D(e, NE, NB, NB, 1,
   {
      const int tid = MFEM_THREAD_ID(x) + NB*MFEM_THREAD_ID(y);
      // int final_iter;
      // double final_norm;
      // bool converged;

      DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, u, r, d1d, q1d);
      DGMassAxpy(e, NE, ND, 1.0, b, -1.0, r, r); // r = b - r

      // TODO: get rid of extra memory usage for z
      DGMassPreconditioner(e, NE, ND, dinv, r, z);
      DGMassAxpy(e, NE, ND, 1.0, z, 0.0, z, d); // d = z

      double nom0 = DGMassDot<NB>(e, NE, ND, d, r);
      double nom = nom0;
      // MFEM_ASSERT(IsFinite(nom), "nom = " << nom);

      if (nom < 0.0)
      {
         return; // Not positive definite...
      }
      double r0 = fmax(nom*RELTOL*RELTOL, ABSTOL*ABSTOL);
      if (nom <= r0)
      {
         // converged = true;
         // final_iter = 0;
         // final_norm = sqrt(nom);
         return;
      }

      DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d);
      double den = DGMassDot<NB>(e, NE, ND, z, d);
      if (den <= 0.0)
      {
         const double d2 = DGMassDot<NB>(e, NE, ND, d, d);
         if (d2 > 0.0 && tid == 0) { printf("Not positive definite.\n"); }
         if (den == 0.0)
         {
            // converged = false;
            // final_iter = 0;
            // final_norm = sqrt(nom);
            return;
         }
      }

      // start iteration
      int i = 1;
      while (true)
      {
         const double alpha = nom/den;
         DGMassAxpy(e, NE, ND, 1.0, u, alpha, d, u); // u = u + alpha*d
         DGMassAxpy(e, NE, ND, 1.0, r, -alpha, z, r); // r = r - alpha*A*d

         DGMassPreconditioner(e, NE, ND, dinv, r, z);

         double betanom = DGMassDot<NB>(e, NE, ND, r, z);
         if (betanom < 0.0)
         {
            if (tid == 0) { printf("Not positive definite.\n"); }
            // converged = false;
            // final_iter = i;
            return;
         }

         // if (tid == 0) { printf(" %4d    %4d    %10.6e\n", e, i, betanom); }

         if (betanom <= r0)
         {
            // converged = true;
            // final_iter = i;
            return;
         }

         if (++i > MAXIT) { return; }

         const double beta = betanom/nom;
         DGMassAxpy(e, NE, ND, 1.0, z, beta, d, d); // d = z + beta*d
         DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d); // z = A d
         den = DGMassDot<NB>(e, NE, ND, d, z);
         if (den <= 0.0)
         {
            const double d2 = DGMassDot<NB>(e, NE, ND, d, d);
            if (d2 > 0.0 && tid == 0) { printf("Not positive definite.\n"); }
            if (den == 0.0)
            {
               // final_iter = i;
               return;
            }
         }
         nom = betanom;
      }
   });
}

void DGMassInverse::Mult(const Vector &Mu, Vector &u) const
{
   const int dim = fes.GetMesh()->Dimension();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;

   const int id = (d1d << 4) | q1d;

   printf("dim = %d id = 0x%x\n", dim, id);
   if (dim == 2)
   {
      switch (id)
      {
         case 0x22: return DGMassCGIteration<2,2,2>(Mu, u);
         case 0x33: return DGMassCGIteration<2,3,3>(Mu, u);
         case 0x35: return DGMassCGIteration<2,3,5>(Mu, u);
         case 0x44: return DGMassCGIteration<2,4,4>(Mu, u);
         case 0x55: return DGMassCGIteration<2,5,5>(Mu, u);
         case 0x66: return DGMassCGIteration<2,6,6>(Mu, u);
         default:
            // printf("id = 0x%x\n", id);
            // MFEM_ABORT("Fallback");
            return DGMassCGIteration<2>(Mu, u);
      }
   }
   else if (dim == 3)
   {
      switch (id)
      {
         case 0x22: return DGMassCGIteration<3,2,2>(Mu, u);
         case 0x23: return DGMassCGIteration<3,2,3>(Mu, u);
         case 0x33: return DGMassCGIteration<3,3,3>(Mu, u);
         case 0x34: return DGMassCGIteration<3,3,4>(Mu, u);
         case 0x44: return DGMassCGIteration<3,4,4>(Mu, u);
         case 0x45: return DGMassCGIteration<3,4,5>(Mu, u);
         case 0x55: return DGMassCGIteration<3,5,5>(Mu, u);
         case 0x56: return DGMassCGIteration<3,5,6>(Mu, u);
         case 0x58: return DGMassCGIteration<3,5,8>(Mu, u);
         case 0x66: return DGMassCGIteration<3,6,6>(Mu, u);
         case 0x67: return DGMassCGIteration<3,6,7>(Mu, u);
         default:
            // printf("id = 0x%x\n", id);
            // MFEM_ABORT("Fallback");
            return DGMassCGIteration<3>(Mu, u);
      }
   }
}

} // namespace mfem
