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

#include "darcyreduction.hpp"

namespace mfem
{

DarcyReduction::DarcyReduction(FiniteElementSpace *fes_u_,
                               FiniteElementSpace *fes_p_, bool bsym_)
   : fes_u(fes_u_), fes_p(fes_p_), bsym(bsym_)
{
   m_nlfi_u = NULL;
   m_nlfi_p = NULL;
   own_m_nlfi_u = false;
   own_m_nlfi_p = false;

   Af_data = NULL;
   Ae_data = NULL;
   Bf_data = NULL;
   Be_data = NULL;
   Df_data = NULL;

   S = NULL;
}

DarcyReduction::~DarcyReduction()
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   if (own_m_nlfi_p) { delete m_nlfi_p; }

   delete[] Af_data;
   delete[] Ae_data;
   delete[] Bf_data;
   delete[] Be_data;
   delete[] Df_data;

   delete S;
}

void DarcyReduction::SetFluxMassNonlinearIntegrator(NonlinearFormIntegrator
                                                    *flux_integ, bool own)
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   own_m_nlfi_u = own;
   m_nlfi_u = flux_integ;
}

void DarcyReduction::SetPotMassNonlinearIntegrator(NonlinearFormIntegrator
                                                   *pot_integ, bool own)
{
   if (own_m_nlfi_p) { delete m_nlfi_p; }
   own_m_nlfi_p = own;
   m_nlfi_p = pot_integ;
}

void DarcyReduction::Init(const Array<int> &ess_flux_tdof_list)
{
   const int NE = fes_u->GetNE();

   // count the number of dofs in the discontinuous version of fes:
   Array<int> vdofs;
   int num_hat_dofs = 0;
   hat_offsets.SetSize(NE+1);
   hat_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      fes_u->GetElementVDofs(i, vdofs);
      num_hat_dofs += vdofs.Size();
      hat_offsets[i+1] = num_hat_dofs;
   }

   // Define the "free" (0) and "essential" (1) hat_dofs.
   // The "essential" hat_dofs are those that depend only on essential cdofs;
   // all other hat_dofs are "free".
   hat_dofs_marker.SetSize(num_hat_dofs);
   Array<int> free_tdof_marker;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>(fes_u);
   free_tdof_marker.SetSize(pfes ? pfes->TrueVSize() :
                            fes_u->GetConformingVSize());
#else
   free_tdof_marker.SetSize(fes_u->GetConformingVSize());
#endif
   free_tdof_marker = 1;
   for (int i = 0; i < ess_flux_tdof_list.Size(); i++)
   {
      free_tdof_marker[ess_flux_tdof_list[i]] = 0;
   }
   Array<int> free_vdofs_marker;
#ifdef MFEM_USE_MPI
   if (!pfes)
   {
      const SparseMatrix *cP = fes_u->GetConformingProlongation();
      if (!cP)
      {
         free_vdofs_marker.MakeRef(free_tdof_marker);
      }
      else
      {
         free_vdofs_marker.SetSize(fes_u->GetVSize());
         cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
      }
   }
   else
   {
      HypreParMatrix *P = pfes->Dof_TrueDof_Matrix();
      free_vdofs_marker.SetSize(fes_u->GetVSize());
      P->BooleanMult(1, free_tdof_marker, 0, free_vdofs_marker);
   }
#else
   const SparseMatrix *cP = fes_u->GetConformingProlongation();
   if (!cP)
   {
      free_vdofs_marker.MakeRef(free_tdof_marker);
   }
   else
   {
      free_vdofs_marker.SetSize(fes_u->GetVSize());
      cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
   }
#endif
   for (int i = 0; i < NE; i++)
   {
      fes_u->GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         hat_dofs_marker[hat_offsets[i]+j] = ! free_vdofs_marker[vdofs[j]];
      }
   }
#ifndef MFEM_DEBUG
   // In DEBUG mode this array is used below.
   free_tdof_marker.DeleteAll();
#endif
   free_vdofs_marker.DeleteAll();
   // Split the "free" (0) hat_dofs into "internal" (0) or "boundary" (-1).
   // The "internal" hat_dofs are those "free" hat_dofs for which the
   // corresponding column in C is zero; otherwise the free hat_dof is
   // "boundary".
   /*for (int i = 0; i < num_hat_dofs; i++)
   {
      // skip "essential" hat_dofs and empty rows in Ct
      if (hat_dofs_marker[i] == 1) { continue; }
      //CT row????????

      //hat_dofs_marker[i] = -1; // mark this hat_dof as "boundary"
   }*/

   // Define Af_offsets and Af_f_offsets
   Af_offsets.SetSize(NE+1);
   Af_offsets[0] = 0;
   Af_f_offsets.SetSize(NE+1);
   Af_f_offsets[0] = 0;

   for (int i = 0; i < NE; i++)
   {
      int f_size = 0; // count the "free" hat_dofs in element i
      for (int j = hat_offsets[i]; j < hat_offsets[i+1]; j++)
      {
         if (hat_dofs_marker[j] != 1) { f_size++; }
      }
      Af_offsets[i+1] = Af_offsets[i] + f_size*f_size;
      Af_f_offsets[i+1] = Af_f_offsets[i] + f_size;
   }

   Af_data = new real_t[Af_offsets[NE]];

   // Define Bf_offsets, Df_offsets and Df_f_offsets
   Bf_offsets.SetSize(NE+1);
   Bf_offsets[0] = 0;
   Df_offsets.SetSize(NE+1);
   Df_offsets[0] = 0;
   Df_f_offsets.SetSize(NE+1);
   Df_f_offsets[0] = 0;
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   Ae_offsets.SetSize(NE+1);
   Ae_offsets[0] = 0;
   Be_offsets.SetSize(NE+1);
   Be_offsets[0] = 0;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   for (int i = 0; i < NE; i++)
   {
      int f_size = Af_f_offsets[i+1] - Af_f_offsets[i];
      int d_size = fes_p->GetFE(i)->GetDof();
      Bf_offsets[i+1] = Bf_offsets[i] + f_size*d_size;
      Df_offsets[i+1] = Df_offsets[i] + d_size*d_size;
      Df_f_offsets[i+1] = Df_f_offsets[i] + d_size;
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
      int a_size = hat_offsets[i+1] - hat_offsets[i];
      int e_size = a_size - f_size;
      Ae_offsets[i+1] = Ae_offsets[i] + e_size*a_size;
      Be_offsets[i+1] = Be_offsets[i] + e_size*d_size;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   }

   Bf_data = new real_t[Bf_offsets[NE]]();//init by zeros
   if (!m_nlfi_p)
   {
      Df_data = new real_t[Df_offsets[NE]]();//init by zeros
   }
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   Ae_data = new real_t[Ae_offsets[NE]];
   Be_data = new real_t[Be_offsets[NE]]();//init by zeros
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
}

void DarcyReduction::AssembleFluxMassMatrix(int el, const DenseMatrix &A)
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   real_t *Af_el_data = Af_data + Af_offsets[el];
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   real_t *Ae_el_data = Ae_data + Ae_offsets[el];
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   for (int j = 0; j < s; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
         for (int i = 0; i < s; i++)
         {
            *(Ae_el_data++) = A(i, j);
         }
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < s; i++)
      {
         if (hat_dofs_marker[o + i] == 1) { continue; }
         *(Af_el_data++) = A(i, j);
      }
   }
   MFEM_ASSERT(Af_el_data == Af_data + Af_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   MFEM_ASSERT(Ae_el_data == Ae_data + Ae_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
}

void DarcyReduction::AssemblePotMassMatrix(int el, const DenseMatrix &D)
{
   const int s = Df_f_offsets[el+1] - Df_f_offsets[el];
   DenseMatrix D_i(Df_data + Df_offsets[el], s, s);
   MFEM_ASSERT(D.Size() == s, "Incompatible sizes");

   D_i += D;
}

void DarcyReduction::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int o = hat_offsets[el];
   const int w = hat_offsets[el+1] - o;
   const int h = Df_f_offsets[el+1] - Df_f_offsets[el];
   real_t *Bf_el_data = Bf_data + Bf_offsets[el];
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   real_t *Be_el_data = Be_data + Be_offsets[el];
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   for (int j = 0; j < w; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
         for (int i = 0; i < h; i++)
         {
            *(Be_el_data++) += B(i, j);
         }
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < h; i++)
      {
         *(Bf_el_data++) += B(i, j);
      }
   }
   MFEM_ASSERT(Bf_el_data == Bf_data + Bf_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   MFEM_ASSERT(Be_el_data == Be_data + Be_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
}

void DarcyReduction::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                         const BlockVector &x, BlockVector &b)
{
   const int NE = fes_u->GetNE();
   Vector u_e, bu_e, bp_e;
   Array<int> u_vdofs, p_dofs, edofs;

   const Vector &xu = x.GetBlock(0);
   Vector &bu = b.GetBlock(0);
   Vector &bp = b.GetBlock(1);

   for (int el = 0; el < NE; el++)
   {
      GetEDofs(el, edofs);
      xu.GetSubVector(edofs, u_e);
      u_e.Neg();

      //bu -= A_e u_e
      const int a_size = hat_offsets[el+1] - hat_offsets[el];
      DenseMatrix Ae(Ae_data + Ae_offsets[el], a_size, edofs.Size());

      bu_e.SetSize(a_size);
      Ae.Mult(u_e, bu_e);

      fes_u->GetElementVDofs(el, u_vdofs);
      bu.AddElementVector(u_vdofs, bu_e);

      //bp -= B_e u_e
      const int d_size = Df_f_offsets[el+1] - Df_f_offsets[el];
      DenseMatrix Be(Be_data + Be_offsets[el], d_size, edofs.Size());

      bp_e.SetSize(d_size);
      Be.Mult(u_e, bp_e);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         bp_e.Neg();
      }

      fes_p->GetElementDofs(el, p_dofs);
      bp.AddElementVector(p_dofs, bp_e);
   }

   for (int vdof : vdofs_flux)
   {
      bu(vdof) = xu(vdof);//<--can be arbitrary as it is ignored
   }
}

void DarcyReduction::Mult(const Vector &x, Vector &y) const
{
   S->Mult(x, y);
}

void DarcyReduction::GetFDofs(int el, Array<int> &fdofs) const
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   Array<int> vdofs;
   fes_u->GetElementVDofs(el, vdofs);
   MFEM_ASSERT(vdofs.Size() == s, "Incompatible DOF sizes");
   fdofs.DeleteAll();
   fdofs.Reserve(s);
   for (int i = 0; i < s; i++)
   {
      if (hat_dofs_marker[i + o] != 1)
      {
         fdofs.Append(vdofs[i]);
      }
   }
}

void DarcyReduction::GetEDofs(int el, Array<int> &edofs) const
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   Array<int> vdofs;
   fes_u->GetElementVDofs(el, vdofs);
   MFEM_ASSERT(vdofs.Size() == s, "Incompatible DOF sizes");
   edofs.DeleteAll();
   edofs.Reserve(s);
   for (int i = 0; i < s; i++)
   {
      if (hat_dofs_marker[i + o] == 1)
      {
         edofs.Append(vdofs[i]);
      }
   }
}

void DarcyReduction::Finalize()
{
   if (!S) { ComputeS(); }
}

void DarcyReduction::Reset()
{
   delete S;
   S = NULL;

   const int NE = fes_u->GetMesh()->GetNE();
   memset(Bf_data, 0, Bf_offsets[NE] * sizeof(real_t));
   if (Df_data)
   {
      memset(Df_data, 0, Df_offsets[NE] * sizeof(real_t));
   }
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   memset(Be_data, 0, Be_offsets[NE] * sizeof(real_t));
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
}

DarcyPotentialReduction::DarcyPotentialReduction(FiniteElementSpace *fes_u,
                                                 FiniteElementSpace *fes_p, bool bsym)
   : DarcyReduction(fes_u, fes_p, bsym)
{
   width = height = fes_u->GetVSize();

   Df_ipiv = NULL;
}

DarcyPotentialReduction::~DarcyPotentialReduction()
{
   delete[] Df_ipiv;
}

void DarcyPotentialReduction::Init(const Array<int> &ess_flux_tdof_list)
{
   DarcyReduction::Init(ess_flux_tdof_list);

   const int NE = fes_p->GetNE();
   Df_ipiv = new int[Df_f_offsets[NE]];
}

void DarcyPotentialReduction::ComputeS()
{
   MFEM_ASSERT(!m_nlfi_u && !m_nlfi_p,
               "Cannot assemble S matrix in the non-linear regime");

   const int skip_zeros = 1;
   const int NE = fes_u->GetNE();

   if (!S) { S = new SparseMatrix(fes_u->GetVSize()); }

   DenseMatrix DiB;
   Array<int> a_dofs;

   for (int el = 0; el < NE; el++)
   {
      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

      DenseMatrix A(Af_data + Af_offsets[el], a_dofs_size, a_dofs_size);
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);

      // Decompose D
      LUFactors LU_D(Df_data + Df_offsets[el], Df_ipiv + Df_f_offsets[el]);

      LU_D.Factor(d_dofs_size);

      // Schur complement
      DiB = B;
      if (!bsym) { DiB.Neg(); }
      LU_D.Solve(DiB.Height(), DiB.Width(), DiB.GetData());
      mfem::AddMultAtB(B, DiB, A);

      GetFDofs(el, a_dofs);

      S->AddSubMatrix(a_dofs, a_dofs, A, skip_zeros);

      // Complete the diagonal
      GetEDofs(el, a_dofs);
      FiniteElementSpace::AdjustVDofs(a_dofs);
      for (int i = 0; i < a_dofs.Size(); i++)
      {
         S->Set(a_dofs[i], a_dofs[i], 1.);
      }
   }

   S->Finalize();
}

void DarcyPotentialReduction::ReduceRHS(const BlockVector &b, Vector &b_r) const
{
   const int NE = fes_u->GetNE();
   Vector bu_l, bp_l;
   Array<int> u_vdofs, p_dofs;

   const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);

   b_r = bu;

   for (int el = 0; el < NE; el++)
   {
      // Load RHS

      GetFDofs(el, u_vdofs);
      bu_l.SetSize(u_vdofs.Size());

      fes_p->GetElementDofs(el, p_dofs);
      bp.GetSubVector(p_dofs, bp_l);

      // -B^T D^-1 bp

      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);
      LUFactors LU_D(Df_data + Df_offsets[el], Df_ipiv + Df_f_offsets[el]);

      LU_D.Solve(d_dofs_size, 1, bp_l.GetData());
      bp_l.Neg();
      B.MultTranspose(bp_l, bu_l);

      b_r.AddElementVector(u_vdofs, bu_l);
   }
}

void DarcyPotentialReduction::ComputeSolution(const BlockVector &b,
                                              const Vector &sol_r,
                                              BlockVector &sol) const
{
   const int NE = fes_u->GetNE();
   Vector bp_l, u_l;
   Array<int> u_vdofs, p_dofs;

   //const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);
   Vector &u = sol.GetBlock(0);
   Vector &p = sol.GetBlock(1);

   u = sol_r;

   for (int el = 0; el < NE; el++)
   {
      //Load RHS

      GetFDofs(el, u_vdofs);
      u.GetSubVector(u_vdofs, u_l);

      fes_p->GetElementDofs(el, p_dofs);
      bp.GetSubVector(p_dofs, bp_l);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         bp_l.Neg();
      }

      // D^-1 (F - B u)

      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);
      LUFactors LU_D(Df_data + Df_offsets[el], Df_ipiv + Df_f_offsets[el]);

      B.AddMult(u_l, bp_l, -1.);

      LU_D.Solve(d_dofs_size, 1, bp_l.GetData());

      p.SetSubVector(p_dofs, bp_l);
   }
}

}
