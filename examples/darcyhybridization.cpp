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

#include "darcyhybridization.hpp"

#define MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
#define MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY

namespace mfem
{

DarcyHybridization::DarcyHybridization(FiniteElementSpace *fes_u_,
                                       FiniteElementSpace *fes_p_,
                                       FiniteElementSpace *fes_c_,
                                       bool bsymmetrize)
   : Hybridization(fes_u_, fes_c_), Operator(c_fes->GetVSize()),
     fes_p(fes_p_), bsym(bsymmetrize)
{
   SetLocalNLSolver(LSsolveType::LBFGS);
   SetLocalNLPreconditioner(LPrecType::GMRES);
}

DarcyHybridization::~DarcyHybridization()
{
   delete c_bfi_p;
   delete c_nlfi_p;
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   if (own_m_nlfi_p) { delete m_nlfi_p; }
   if (own_m_nlfi) { delete m_nlfi; }
   if (!extern_bdr_constr_pot_integs)
   {
      for (int k=0; k < boundary_constraint_pot_integs.Size(); k++)
      { delete boundary_constraint_pot_integs[k]; }
      for (int k=0; k < boundary_constraint_pot_nonlin_integs.Size(); k++)
      { delete boundary_constraint_pot_nonlin_integs[k]; }
   }

   delete[] Af_lin_data;
   delete[] Ae_data;
   delete[] Bf_data;
   delete[] Be_data;
   delete[] Df_data;
   delete[] Df_lin_data;
   delete[] Df_ipiv;
   delete[] Ct_data;
   delete[] E_data;
   delete[] G_data;
   delete[] H_data;
}

void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, BilinearFormIntegrator *c_pot_integ)
{
   MFEM_VERIFY(!m_nlfi_p, "Linear constraint cannot work with a non-linear mass");

   delete c_bfi;
   c_bfi = c_flux_integ;
   delete c_bfi_p;
   c_bfi_p = c_pot_integ;
   delete c_nlfi_p;
   c_nlfi_p = NULL;

   bnl = false;
}

void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, NonlinearFormIntegrator *c_pot_integ)
{
   delete c_bfi;
   c_bfi = c_flux_integ;
   delete c_bfi_p;
   c_bfi_p = NULL;
   delete c_nlfi_p;
   c_nlfi_p = c_pot_integ;

   bnl = true;
}

void DarcyHybridization::SetFluxMassNonlinearIntegrator(
   NonlinearFormIntegrator *flux_integ, bool own)
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   own_m_nlfi_u = own;
   m_nlfi_u = flux_integ;

   bnl = true;
}

void DarcyHybridization::SetPotMassNonlinearIntegrator(NonlinearFormIntegrator
                                                       *pot_integ, bool own)
{
   MFEM_VERIFY(!c_bfi_p, "Non-linear mass cannot work with a linear constraint");

   if (own_m_nlfi_p) { delete m_nlfi_p; }
   own_m_nlfi_p = own;
   m_nlfi_p = pot_integ;

   bnl = true;
}

void DarcyHybridization::SetBlockNonlinearIntegrator(
   BlockNonlinearFormIntegrator *block_integ, bool own)
{
   if (own_m_nlfi) { delete m_nlfi; }
   own_m_nlfi = own;
   m_nlfi = block_integ;

   bnl = true;
}

void DarcyHybridization::Init(const Array<int> &ess_flux_tdof_list)
{
   const int NE = fes->GetNE();

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   if (Ct_data) { return; }

   // count the number of dofs in the discontinuous version of fes:
   Array<int> vdofs;
   int num_hat_dofs = 0;
   hat_offsets.SetSize(NE+1);
   hat_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      num_hat_dofs += vdofs.Size();
      hat_offsets[i+1] = num_hat_dofs;
   }

   // Define the "free" (0) and "essential" (1) hat_dofs.
   // The "essential" hat_dofs are those that depend only on essential cdofs;
   // all other hat_dofs are "free".
   hat_dofs_marker.SetSize(num_hat_dofs);
   Array<int> free_tdof_marker;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
   free_tdof_marker.SetSize(pfes ? pfes->TrueVSize() :
                            fes->GetConformingVSize());
#else
   free_tdof_marker.SetSize(fes->GetConformingVSize());
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
      const SparseMatrix *cP = fes->GetConformingProlongation();
      if (!cP)
      {
         free_vdofs_marker.MakeRef(free_tdof_marker);
      }
      else
      {
         free_vdofs_marker.SetSize(fes->GetVSize());
         cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
      }
   }
   else
   {
      HypreParMatrix *P = pfes->Dof_TrueDof_Matrix();
      free_vdofs_marker.SetSize(fes->GetVSize());
      P->BooleanMult(1, free_tdof_marker, 0, free_vdofs_marker);
   }
#else
   const SparseMatrix *cP = fes->GetConformingProlongation();
   if (!cP)
   {
      free_vdofs_marker.MakeRef(free_tdof_marker);
   }
   else
   {
      free_vdofs_marker.SetSize(fes->GetVSize());
      cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
   }
#endif
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
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
   Af_ipiv = new int[Af_f_offsets[NE]];

   // Assemble the constraint matrix C
   ConstructC();
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   if (Ct) { return; }

   Hybridization::Init(ess_flux_tdof_list);
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY

   // Define Bf_offsets, Df_offsets and Df_f_offsets
   Bf_offsets.SetSize(NE+1);
   Bf_offsets[0] = 0;
   Df_offsets.SetSize(NE+1);
   Df_offsets[0] = 0;
   Df_f_offsets.SetSize(NE+1);
   Df_f_offsets[0] = 0;
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Ae_offsets.SetSize(NE+1);
   Ae_offsets[0] = 0;
   Be_offsets.SetSize(NE+1);
   Be_offsets[0] = 0;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   for (int i = 0; i < NE; i++)
   {
      int f_size = Af_f_offsets[i+1] - Af_f_offsets[i];
      int d_size = fes_p->GetFE(i)->GetDof();
      Bf_offsets[i+1] = Bf_offsets[i] + f_size*d_size;
      Df_offsets[i+1] = Df_offsets[i] + d_size*d_size;
      Df_f_offsets[i+1] = Df_f_offsets[i] + d_size;
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
      int a_size = hat_offsets[i+1] - hat_offsets[i];
      int e_size = a_size - f_size;
      Ae_offsets[i+1] = Ae_offsets[i] + e_size*a_size;
      Be_offsets[i+1] = Be_offsets[i] + e_size*d_size;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   }

   Bf_data = new real_t[Bf_offsets[NE]]();//init by zeros
   if (!m_nlfi_p)
   {
      Df_data = new real_t[Df_offsets[NE]]();//init by zeros
      Df_ipiv = new int[Df_f_offsets[NE]];
   }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Ae_data = new real_t[Ae_offsets[NE]];
   Be_data = new real_t[Be_offsets[NE]]();//init by zeros
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   if (c_bfi_p)
   {
      AllocEG();
      if (bnl)
      {
         AllocH();
      }
   }
}

void DarcyHybridization::AssembleFluxMassMatrix(int el, const DenseMatrix &A)
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   real_t *Af_el_data = Af_data + Af_offsets[el];
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   real_t *Ae_el_data = Ae_data + Ae_offsets[el];
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   for (int j = 0; j < s; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         for (int i = 0; i < s; i++)
         {
            *(Ae_el_data++) = A(i, j);
         }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < s; i++)
      {
         if (hat_dofs_marker[o + i] == 1) { continue; }
         *(Af_el_data++) = A(i, j);
      }
   }
   MFEM_ASSERT(Af_el_data == Af_data + Af_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   MFEM_ASSERT(Ae_el_data == Ae_data + Ae_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   A_empty = false;
}

void DarcyHybridization::AssemblePotMassMatrix(int el, const DenseMatrix &D)
{
   const int s = Df_f_offsets[el+1] - Df_f_offsets[el];
   DenseMatrix D_i(Df_data + Df_offsets[el], s, s);
   MFEM_ASSERT(D.Size() == s, "Incompatible sizes");

   D_i += D;

   D_empty = false;
}

void DarcyHybridization::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int o = hat_offsets[el];
   const int w = hat_offsets[el+1] - o;
   const int h = Df_f_offsets[el+1] - Df_f_offsets[el];
   real_t *Bf_el_data = Bf_data + Bf_offsets[el];
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   real_t *Be_el_data = Be_data + Be_offsets[el];
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   for (int j = 0; j < w; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         for (int i = 0; i < h; i++)
         {
            *(Be_el_data++) += B(i, j);
         }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < h; i++)
      {
         *(Bf_el_data++) += B(i, j);
      }
   }
   MFEM_ASSERT(Bf_el_data == Bf_data + Bf_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   MFEM_ASSERT(Be_el_data == Be_data + Be_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
}

void DarcyHybridization::ComputeAndAssemblePotFaceMatrix(
   int face, DenseMatrix &elmat1, DenseMatrix &elmat2, Array<int> &vdofs1,
   Array<int> &vdofs2)
{
   Mesh *mesh = fes_p->GetMesh();
   const FiniteElement *tr_fe, *fe1, *fe2;
   DenseMatrix elmat, h_elmat;
   int ndof1, ndof2;
   Array<int> c_dofs;

   tr_fe = c_fes->GetFaceElement(face);
   c_fes->GetFaceDofs(face, c_dofs);
   const int c_dof = c_dofs.Size();

   FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(face);
   fes_p->GetElementVDofs(ftr->Elem1No, vdofs1);
   fe1 = fes_p->GetFE(ftr->Elem1No);
   ndof1 = fe1->GetDof();

   if (ftr->Elem2No >= 0)
   {
      fes_p->GetElementVDofs(ftr->Elem2No, vdofs2);
      fe2 = fes_p->GetFE(ftr->Elem2No);
      ndof2 = fe2->GetDof();
   }
   else
   {
      vdofs2.SetSize(0);
      fe2 = fe1;
      ndof2 = 0;
   }

   c_bfi_p->AssembleHDGFaceMatrix(*tr_fe, *fe1, *fe2, *ftr, elmat);

   MFEM_ASSERT(elmat.Width() == ndof1+ndof2+c_dof &&
               elmat.Height() == ndof1+ndof2+c_dof,
               "Size mismatch");

   // assemble D element matrices
   elmat1.CopyMN(elmat, ndof1, ndof1, 0, 0);
   AssemblePotMassMatrix(ftr->Elem1No, elmat1);
   if (ndof2)
   {
      elmat2.CopyMN(elmat, ndof2, ndof2, ndof1, ndof1);
      AssemblePotMassMatrix(ftr->Elem2No, elmat2);
   }

   // assemble E constraint
   DenseMatrix E_f_1(E_data + E_offsets[face], ndof1, c_dof);
   E_f_1.CopyMN(elmat, ndof1, c_dof, 0, ndof1+ndof2);
   if (ndof2)
   {
      DenseMatrix E_f_2(E_data + E_offsets[face] + c_dof*ndof1, ndof2, c_dof);
      E_f_2.CopyMN(elmat, ndof2, c_dof, ndof1, ndof1+ndof2);
   }

   // assemble G constraint
   DenseMatrix G_f(G_data + G_offsets[face], c_dof, ndof1+ndof2);
   G_f.CopyMN(elmat, c_dof, ndof1+ndof2, ndof1+ndof2, 0);

   // assemble H matrix
   if (bnl)
   {
      DenseMatrix H_f(H_data + H_offsets[face], c_dof, c_dof);
      H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
   }
   else
   {
      if (!H) { H = new SparseMatrix(c_fes->GetVSize()); }
      h_elmat.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
      H->AddSubMatrix(c_dofs, c_dofs, h_elmat);
   }
}

void DarcyHybridization::ComputeAndAssemblePotBdrFaceMatrix(
   int bface, DenseMatrix &elmat1, Array<int> &vdofs)
{
   Mesh *mesh = fes_p->GetMesh();
   const FiniteElement *tr_fe, *fe;
   DenseMatrix elmat, elmat_aux, h_elmat;
   Array<int> c_dofs;

   const int face = mesh->GetBdrElementFaceIndex(bface);
   tr_fe = c_fes->GetFaceElement(face);
   c_fes->GetFaceDofs(face, c_dofs);
   const int c_dof = c_dofs.Size();

   FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(face);
   fes_p->GetElementVDofs(ftr->Elem1No, vdofs);
   fe = fes_p->GetFE(ftr->Elem1No);
   const int ndof = fe->GetDof();

   MFEM_ASSERT(boundary_constraint_pot_integs.Size() > 0,
               "No boundary constraint integrators");

   const int bdr_attr = mesh->GetBdrAttribute(bface);
   for (int i = 0; i < boundary_constraint_pot_integs.Size(); i++)
   {
      if (boundary_constraint_pot_integs_marker[i]
          && (*boundary_constraint_pot_integs_marker[i])[bdr_attr-1] == 0) { continue; }

      boundary_constraint_pot_integs[i]->AssembleHDGFaceMatrix(*tr_fe, *fe, *fe, *ftr,
                                                               elmat_aux);

      if (elmat.Size() > 0)
      { elmat += elmat_aux; }
      else
      { elmat = elmat_aux; }
   }

   if (elmat.Size() == 0) { return; }

   MFEM_ASSERT(elmat.Width() == ndof+c_dof &&
               elmat.Height() == ndof+c_dof,
               "Size mismatch");

   // assemble D element matrices
   elmat1.CopyMN(elmat, ndof, ndof, 0, 0);
   AssemblePotMassMatrix(ftr->Elem1No, elmat1);

   // assemble E constraint
   DenseMatrix E_f_1(E_data + E_offsets[face], ndof, c_dof);
   E_f_1.CopyMN(elmat, ndof, c_dof, 0, ndof);

   // assemble G constraint
   DenseMatrix G_f(G_data + G_offsets[face], c_dof, ndof);
   G_f.CopyMN(elmat, c_dof, ndof, ndof, 0);

   // assemble H matrix
   if (bnl)
   {
      DenseMatrix H_f(H_data + H_offsets[face], c_dof, c_dof);
      H_f.CopyMN(elmat, c_dof, c_dof, ndof, ndof);
   }
   else
   {
      if (!H) { H = new SparseMatrix(c_fes->GetVSize()); }
      h_elmat.CopyMN(elmat, c_dof, c_dof, ndof, ndof);
      H->AddSubMatrix(c_dofs, c_dofs, h_elmat);
   }
}

void DarcyHybridization::GetFDofs(int el, Array<int> &fdofs) const
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   Array<int> vdofs;
   fes->GetElementVDofs(el, vdofs);
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

void DarcyHybridization::GetEDofs(int el, Array<int> &edofs) const
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   Array<int> vdofs;
   fes->GetElementVDofs(el, vdofs);
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

void DarcyHybridization::AssembleCtFaceMatrix(int face, int el1, int el2,
                                              const DenseMatrix &elmat)
{
   const int hat_size_1 = hat_offsets[el1+1] - hat_offsets[el1];
   const int f_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
   const int c_size = c_fes->GetFaceElement(face)->GetDof() * c_fes->GetVDim();

   //el1
   DenseMatrix Ct_face_1(Ct_data + Ct_offsets[face], f_size_1, c_size);
   AssembleCtSubMatrix(el1, elmat, Ct_face_1);

   //el2
   if (el2 >= 0)
   {
      //const int hat_size_2 = hat_offsets[el2+1] - hat_offsets[el2];
      const int f_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];

      DenseMatrix Ct_face_2(Ct_data + Ct_offsets[face] + f_size_1*c_size,
                            f_size_2, c_size);
      AssembleCtSubMatrix(el2, elmat, Ct_face_2, hat_size_1);
   }
}

void DarcyHybridization::AssembleCtSubMatrix(int el, const DenseMatrix &elmat,
                                             DenseMatrix &Ct, int ioff)
{
   const int hat_offset = hat_offsets[el];
   const int hat_size = hat_offsets[el+1] - hat_offset;

   int row = 0;
   for (int i = 0; i < hat_size; i++)
   {
      if (hat_dofs_marker[hat_offset + i] == 1) { continue; }
      bool bzero = true;
      for (int j = 0; j < Ct.Width(); j++)
      {
         const real_t val = elmat(i + ioff, j);
         if (val == 0.) { continue; }
         Ct(row, j) = val;
         bzero = false;
      }
      if (!bzero)
      {
         //mark the hat dof as "boundary" if the row is non-zero
         hat_dofs_marker[hat_offset + i] = -1;
      }
      row++;
   }
   MFEM_ASSERT(row == Af_f_offsets[el+1] - Af_f_offsets[el], "Internal error.");
}

void DarcyHybridization::ConstructC()
{
#ifndef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   Hybridization::ConstructC();
   return;
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY

   FaceElementTransformations *FTr;
   Mesh *mesh = fes->GetMesh();
   int num_faces = mesh->GetNumFaces();

#if defined(MFEM_USE_DOUBLE)
   constexpr real_t mtol = 1e-12;
#elif defined(MFEM_USE_SINGLE)
   constexpr real_t mtol = 4e-6;
#else
#error "Only single and double precision are supported!"
   constexpr real_t mtol = 1.;
#endif

   // Define Ct_offsets and allocate Ct_data
   Ct_offsets.SetSize(num_faces+1);
   Ct_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      FTr = mesh->GetFaceElementTransformations(f, 0);

      int f_size = Af_f_offsets[FTr->Elem1No+1] - Af_f_offsets[FTr->Elem1No];
      if (FTr->Elem2No >= 0)
      {
         f_size += Af_f_offsets[FTr->Elem2No+1] - Af_f_offsets[FTr->Elem2No];
      }
      const int c_size = c_fes->GetFaceElement(f)->GetDof() * c_fes->GetVDim();
      Ct_offsets[f+1] = Ct_offsets[f] + c_size * f_size;
   }

   Ct_data = new real_t[Ct_offsets[num_faces]]();//init by zeros

   // Assemble the constraint element matrices
   if (c_bfi)
   {
      DenseMatrix elmat;

      for (int f = 0; f < num_faces; f++)
      {
         FTr = mesh->GetInteriorFaceTransformations(f);
         if (!FTr) { continue; }

         const FiniteElement *fe1 = fes->GetFE(FTr->Elem1No);
         const FiniteElement *fe2 = fes->GetFE(FTr->Elem2No);

         c_bfi->AssembleFaceMatrix(*c_fes->GetFaceElement(f),
                                   *fe1, *fe2, *FTr, elmat);
         // zero-out small elements in elmat
         elmat.Threshold(mtol * elmat.MaxMaxNorm());

         // assemble the matrix
         AssembleCtFaceMatrix(f, FTr->Elem1No, FTr->Elem2No, elmat);
      }

      if (boundary_constraint_integs.Size())
      {
         const FiniteElement *fe1, *fe2;
         const FiniteElement *face_el;

         // Which boundary attributes need to be processed?
         Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
         bdr_attr_marker = 0;
         for (int k = 0; k < boundary_constraint_integs.Size(); k++)
         {
            if (boundary_constraint_integs_marker[k] == NULL)
            {
               bdr_attr_marker = 1;
               break;
            }
            Array<int> &bdr_marker = *boundary_constraint_integs_marker[k];
            MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                        "invalid boundary marker for boundary face integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < bdr_attr_marker.Size(); i++)
            {
               bdr_attr_marker[i] |= bdr_marker[i];
            }
         }

         for (int i = 0; i < fes->GetNBE(); i++)
         {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            FTr = mesh->GetBdrFaceTransformations(i);
            if (!FTr) { continue; }

            int iface = mesh->GetBdrElementFaceIndex(i);
            face_el = c_fes->GetFaceElement(iface);
            fe1 = fes -> GetFE (FTr -> Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < boundary_constraint_integs.Size(); k++)
            {
               if (boundary_constraint_integs_marker[k] &&
                   (*boundary_constraint_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               boundary_constraint_integs[k]->AssembleFaceMatrix(*face_el, *fe1, *fe2, *FTr,
                                                                 elmat);
               // zero-out small elements in elmat
               elmat.Threshold(mtol * elmat.MaxMaxNorm());

               // assemble the matrix
               AssembleCtFaceMatrix(iface, FTr->Elem1No, FTr->Elem2No, elmat);
            }
         }
      }
   }
   else
   {
      // Check if c_fes is really needed here.
      MFEM_ABORT("TODO: algebraic definition of C");
   }
}

void DarcyHybridization::AllocD() const
{
   Df_data = new real_t[Df_offsets.Last()]();//init by zeros
   Df_ipiv = new int[Df_f_offsets.Last()];
}

void DarcyHybridization::AllocEG() const
{
   FaceElementTransformations *FTr;
   Mesh *mesh = fes->GetMesh();
   int num_faces = mesh->GetNumFaces();

   // Define E_offsets and allocate E_data and G_data
   E_offsets.SetSize(num_faces+1);
   E_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      FTr = mesh->GetFaceElementTransformations(f, 0);

      int d_size = Df_f_offsets[FTr->Elem1No+1] - Df_f_offsets[FTr->Elem1No];
      if (FTr->Elem2No >= 0)
      {
         d_size += Df_f_offsets[FTr->Elem2No+1] - Df_f_offsets[FTr->Elem2No];
      }
      const int c_size = c_fes->GetFaceElement(f)->GetDof() * c_fes->GetVDim();
      E_offsets[f+1] = E_offsets[f] + c_size * d_size;
   }

   E_data = new real_t[E_offsets[num_faces]]();//init by zeros
   G_data = new real_t[G_offsets[num_faces]]();//init by zeros
}

void DarcyHybridization::AllocH() const
{
   Mesh *mesh = fes->GetMesh();
   int num_faces = mesh->GetNumFaces();

   // Define E_offsets and allocate E_data and G_data
   H_offsets.SetSize(num_faces+1);
   H_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      const int c_size = c_fes->GetFaceElement(f)->GetDof() * c_fes->GetVDim();
      H_offsets[f+1] = H_offsets[f] + c_size * c_size;
   }

   H_data = new real_t[H_offsets[num_faces]]();//init by zeros
}

void DarcyHybridization::InvertA()
{
   const int NE = fes->GetNE();

   for (int el = 0; el < NE; el++)
   {
      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];

      // Decompose A

      LUFactors LU_A(Af_data + Af_offsets[el], Af_ipiv + Af_f_offsets[el]);

      LU_A.Factor(a_dofs_size);
   }
}

void DarcyHybridization::InvertD()
{
   const int NE = fes->GetNE();

   for (int el = 0; el < NE; el++)
   {
      int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

      // Decompose D

#ifdef MFEM_DEBUG
      DenseMatrix D(Df_data + Df_offsets[el], d_dofs_size, d_dofs_size);
      const double norm = D.MaxMaxNorm();
      if (norm == 0.)
      {
         MFEM_ABORT("Inverting an empty matrix!");
      }
      if (D.Rank(norm * 1e-12) < d_dofs_size)
      {
         MFEM_ABORT("Inverting a singular matrix!");
      }
#endif

      LUFactors LU_D(Df_data + Df_offsets[el], Df_ipiv + Df_f_offsets[el]);

      LU_D.Factor(d_dofs_size);
   }
}

void DarcyHybridization::ComputeH()
{
   MFEM_ASSERT(!bnl, "Cannot assemble H matrix in the non-linear regime");

   const int skip_zeros = 1;
   const int NE = fes->GetNE();
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   const int dim = fes->GetMesh()->Dimension();
   DenseMatrix AiBt, AiCt, BAiCt, CAiBt, H_l;
   DenseMatrix Ct_1_el_1, Ct_1_el_2, Ct_2_el_1, Ct_2_el_2;
   DenseMatrix E_el_1, E_el_2, Gt_el_1, Gt_el_2;
   Array<int> c_dofs_1, c_dofs_2;
   Array<int> faces, oris;
   if (!H) { H = new SparseMatrix(c_fes->GetVSize()); }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   MFEM_ASSERT(!c_bfi_p,
               "Potential constraint is not supported in non-block assembly!");
   DenseMatrix AiBt, BAi, Hb_l;
   Array<int> a_dofs;
   SparseMatrix *Hb = new SparseMatrix(Ct->Height());
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK

   for (int el = 0; el < NE; el++)
   {
      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

      // Decompose A

      LUFactors LU_A(Af_data + Af_offsets[el], Af_ipiv + Af_f_offsets[el]);

      LU_A.Factor(a_dofs_size);

      // Construct Schur complement
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);
      DenseMatrix D(Df_data + Df_offsets[el], d_dofs_size, d_dofs_size);
      AiBt.SetSize(a_dofs_size, d_dofs_size);

      AiBt.Transpose(B);
      if (!bsym) { AiBt.Neg(); }
      LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());
      mfem::AddMult(B, AiBt, D);

      // Decompose Schur complement
      LUFactors LU_S(D.GetData(), Df_ipiv + Df_f_offsets[el]);

      LU_S.Factor(d_dofs_size);
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      switch (dim)
      {
         case 1:
            fes->GetMesh()->GetElementVertices(el, faces);
            break;
         case 2:
            fes->GetMesh()->GetElementEdges(el, faces, oris);
            break;
         case 3:
            fes->GetMesh()->GetElementFaces(el, faces, oris);
            break;
      }

      // Mult C^T
      for (int f1 = 0; f1 < faces.Size(); f1++)
      {
         FaceElementTransformations *FTr = GetCtFaceMatrix(faces[f1], Ct_1_el_1,
                                                           Ct_1_el_2);
         if (!FTr) { continue; }

         DenseMatrix &Ct_1 = (FTr->Elem1No == el)?(Ct_1_el_1):(Ct_1_el_2);

         //A^-1 C^T
         AiCt.SetSize(Ct_1.Height(), Ct_1.Width());
         AiCt = Ct_1;
         LU_A.Solve(Ct_1.Height(), Ct_1.Width(), AiCt.GetData());

         //S^-1 (B A^-1 C^T - E)
         BAiCt.SetSize(B.Height(), Ct_1.Width());
         mfem::Mult(B, AiCt, BAiCt);

         if (c_bfi_p)
         {
            if (GetEFaceMatrix(faces[f1], E_el_1, E_el_2))
            {
               DenseMatrix &E = (FTr->Elem1No == el)?(E_el_1):(E_el_2);
               BAiCt -= E;
            }
         }

         LU_S.Solve(BAiCt.Height(), BAiCt.Width(), BAiCt.GetData());

         for (int f2 = 0; f2 < faces.Size(); f2++)
         {
            FaceElementTransformations *FTr = GetCtFaceMatrix(faces[f2], Ct_2_el_1,
                                                              Ct_2_el_2);
            if (!FTr) { continue; }

            DenseMatrix &Ct_2 = (FTr->Elem1No == el)?(Ct_2_el_1):(Ct_2_el_2);

            //- C A^-1 C^T
            H_l.SetSize(Ct_2.Width(), Ct_1.Width());
            mfem::MultAtB(Ct_2, AiCt, H_l);
            H_l.Neg();

            //(C A^-1 B^T + G) S^-1 (B A^-1 C^T - E)
            CAiBt.SetSize(Ct_2.Width(), B.Height());
            mfem::MultAtB(Ct_2, AiBt, CAiBt);

            if (c_bfi_p)
            {
               if (GetGFaceMatrix(faces[f2], Gt_el_1, Gt_el_2))
               {
                  DenseMatrix &G = (FTr->Elem1No == el)?(Gt_el_1):(Gt_el_2);
                  CAiBt += G;
               }
            }

            mfem::AddMult(CAiBt, BAiCt, H_l);

            c_fes->GetFaceVDofs(faces[f1], c_dofs_1);
            if (f1 == f2)
            {
               H->AddSubMatrix(c_dofs_1, c_dofs_1, H_l, skip_zeros);
            }
            else
            {
               c_fes->GetFaceVDofs(faces[f2], c_dofs_2);
               H->AddSubMatrix(c_dofs_2, c_dofs_1, H_l, skip_zeros);
            }
         }

      }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      Hb_l.SetSize(B.Width());

      //-A^-1
      LU_A.GetInverseMatrix(B.Width(), Hb_l.GetData());
      Hb_l.Neg();

      //B A^-1
      BAi.SetSize(B.Height(), B.Width());
      mfem::Mult(B, Hb_l, BAi);
      BAi.Neg();

      //S^-1 B A^-1
      LU_S.Solve(BAi.Height(), BAi.Width(), BAi.GetData());

      //A^-1 B^T S^-1 B A^-1
      mfem::AddMult(AiBt, BAi, Hb_l);

      a_dofs.SetSize(a_dofs_size);
      for (int i = 0; i < a_dofs_size; i++)
      {
         a_dofs[i] = hat_offsets[el] + i;
      }

      Hb->AddSubMatrix(a_dofs, a_dofs, Hb_l, skip_zeros);
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   }

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   if (diag_policy == DIAG_ONE)
   {
      // Complete the diagonal by ones
      for (int i = 0; i < H->Size(); i++)
      {
         if (!H->RowIsEmpty(i)) { continue; }
         H->Set(i, i, 1.);
      }
   }
   H->Finalize();
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Hb->Finalize();
   if (H)
   {
      SparseMatrix *rap = RAP(*Ct, *Hb, *Ct);
      *H += *rap;
      delete rap;
   }
   else
   {
      H = RAP(*Ct, *Hb, *Ct);
   }
   delete Hb;
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
}

FaceElementTransformations *DarcyHybridization::GetCtFaceMatrix(
   int f, DenseMatrix &Ct_1, DenseMatrix &Ct_2) const
{
   FaceElementTransformations *FTr =
      fes->GetMesh()->GetFaceElementTransformations(f, 0);

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   const int c_size = c_fes->GetFaceElement(f)->GetDof() * c_fes->GetVDim();

   const int f_size_1 = Af_f_offsets[FTr->Elem1No+1] - Af_f_offsets[FTr->Elem1No];
   Ct_1.Reset(Ct_data + Ct_offsets[f], f_size_1, c_size);
   if (FTr->Elem2No >= 0)
   {
      const int f_size_2 = Af_f_offsets[FTr->Elem2No+1] - Af_f_offsets[FTr->Elem2No];
      Ct_2.Reset(Ct_data + Ct_offsets[f] + f_size_1*c_size,
                 f_size_2, c_size);
   }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   Array<int> c_dofs;
   c_fes->GetFaceVDofs(f, c_dofs);
   GetCtSubMatrix(FTr->Elem1No, c_dofs, Ct_1);
   if (FTr->Elem2No >= 0)
   {
      GetCtSubMatrix(FTr->Elem2No, c_dofs, Ct_2);
   }
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   return FTr;
}

FaceElementTransformations *DarcyHybridization::GetEFaceMatrix(
   int f, DenseMatrix &E_1, DenseMatrix &E_2) const
{
   FaceElementTransformations *FTr =
      fes->GetMesh()->GetFaceElementTransformations(f, 3);

   const int c_size = c_fes->GetFaceElement(f)->GetDof() * c_fes->GetVDim();

   const int d_size_1 = Df_f_offsets[FTr->Elem1No+1] - Df_f_offsets[FTr->Elem1No];
   E_1.Reset(E_data + E_offsets[f], d_size_1, c_size);
   if (FTr->Elem2No >= 0)
   {
      const int d_size_2 = Df_f_offsets[FTr->Elem2No+1] - Df_f_offsets[FTr->Elem2No];
      E_2.Reset(E_data + E_offsets[f] + d_size_1*c_size, d_size_2, c_size);
   }
   return FTr;
}

FaceElementTransformations *DarcyHybridization::GetGFaceMatrix(
   int f, DenseMatrix &G_1, DenseMatrix &G_2) const
{
   FaceElementTransformations *FTr =
      fes->GetMesh()->GetFaceElementTransformations(f, 0);

   const int c_size = c_fes->GetFaceElement(f)->GetDof() * c_fes->GetVDim();

   const int d_size_1 = Df_f_offsets[FTr->Elem1No+1] - Df_f_offsets[FTr->Elem1No];
   G_1.Reset(G_data + G_offsets[f], c_size, d_size_1);
   if (FTr->Elem2No >= 0)
   {
      const int d_size_2 = Df_f_offsets[FTr->Elem2No+1] - Df_f_offsets[FTr->Elem2No];
      G_2.Reset(G_data + G_offsets[f] + d_size_1*c_size, c_size, d_size_2);
   }
   return FTr;
}

void DarcyHybridization::GetHFaceMatrix(int f, DenseMatrix &H) const
{
   const int c_size = c_fes->GetFaceElement(f)->GetDof() * c_fes->GetVDim();

   H.Reset(H_data + H_offsets[f], c_size, c_size);
}

void DarcyHybridization::GetCtSubMatrix(int el, const Array<int> &c_dofs,
                                        DenseMatrix &Ct_l) const
{
   const int hat_offset = hat_offsets[el  ];
   const int hat_size = hat_offsets[el+1] - hat_offset;
   const int f_size = Af_f_offsets[el+1] - Af_f_offsets[el];

   Array<int> vdofs;
   fes->GetElementVDofs(el, vdofs);

   Ct_l.SetSize(f_size, c_dofs.Size());
   Ct_l = 0.;

   int i = 0;
   for (int row = hat_offset; row < hat_offset + hat_size; row++)
   {
      if (hat_dofs_marker[row] == 1) { continue; }
      const int ncols = Ct->RowSize(row);
      const int *cols = Ct->GetRowColumns(row);
      const real_t *vals = Ct->GetRowEntries(row);
      for (int j = 0; j < c_dofs.Size(); j++)
      {
         const int cdof = (c_dofs[j]>=0)?(c_dofs[j]):(-1-c_dofs[j]);
         for (int col = 0; col < ncols; col++)
            if (cols[col] == cdof)
            {
               real_t val = vals[col];
               Ct_l(i,j) = (c_dofs[j] >= 0)?(+val):(-val);
               break;
            }
      }
      i++;
   }
}

void DarcyHybridization::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(bfin, "DarcyHybridization must be finalized");

   if (H)
   {
      H->Mult(x, y);
      return;
   }

   MultNL(MultNlMode::Mult, darcy_rhs, x, y);
}

Operator &DarcyHybridization::GetGradient(const Vector &x) const
{
   MFEM_VERIFY(bfin, "DarcyHybridization must be finalized");

   if (H) { return *H; }

   if (!Df_data) { AllocD(); }// D is resetted in ConstructGrad()
   if (!E_data || !G_data) { AllocEG(); }// E and G are rewritten
   if (!H_data) { AllocH(); }
   else if (c_nlfi_p)
   {
      // H is resetted here for additive double side integration
      memset(H_data, 0, H_offsets.Last() * sizeof(real_t));
   }

   Vector y;//dummy
   MultNL(MultNlMode::Grad, darcy_rhs, x, y);

   pGrad.Reset(new Gradient(*this));
   return *pGrad;
}

void DarcyHybridization::MultNL(MultNlMode mode, const BlockVector &b,
                                const Vector &x, Vector &y) const
{
   const int NE = fes->GetNE();
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   const int dim = fes->GetMesh()->Dimension();
   DenseMatrix Ct_1, Ct_2, E_1, E_2, G_1, G_2, H;
   BlockVector x_l;
   Array<int> c_dofs;
   Array<int> c_offsets;
   Array<int> faces, oris;
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   MFEM_ASSERT(!c_nlfi_p,
               "Potential constraint is not supported in non-block assembly!");
   Vector hat_bu(hat_offsets.Last());
   Vector hat_u;
   if (mode == MultNlMode::Mult)
   {
      hat_u.SetSize(hat_offsets.Last());
      hat_u = 0.;//essential vdofs?!
   }
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Vector bu_l, bp_l, u_l, p_l, y_l;
   Array<int> u_vdofs, p_dofs;

   const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);
   BlockVector yb;
   if (mode == MultNlMode::Sol)
   {
      yb.Update(y, darcy_offsets);
   }
   else
   {
      y = 0.0;
   }

   if (f_2_b.Size() == 0)
   {
      f_2_b = fes->GetMesh()->GetFaceToBdrElMap();
   }

#ifndef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   //C^T sol_r
   Ct->Mult(x, hat_bu);
#endif //!MFEM_DARCY_HYBRIDIZATION_CT_BLOCK

   for (int el = 0; el < NE; el++)
   {
      //Load RHS

      if (mode != MultNlMode::GradMult)
      {
         GetFDofs(el, u_vdofs);
         bu.GetSubVector(u_vdofs, bu_l);

         fes_p->GetElementDofs(el, p_dofs);
         bp.GetSubVector(p_dofs, bp_l);
         if (bsym)
         {
            //In the case of the symmetrized system, the sign is oppposite!
            bp_l.Neg();
         }
      }
      else
      {
         bu_l.SetSize(Af_f_offsets[el+1] - Af_f_offsets[el]);
         bu_l = 0.;
         bp_l.SetSize(Df_f_offsets[el+1] - Df_f_offsets[el]);
         bp_l = 0.;
      }

      switch (dim)
      {
         case 1:
            fes->GetMesh()->GetElementVertices(el, faces);
            break;
         case 2:
            fes->GetMesh()->GetElementEdges(el, faces, oris);
            break;
         case 3:
            fes->GetMesh()->GetElementFaces(el, faces, oris);
            break;
      }

      c_offsets.SetSize(faces.Size()+1);
      c_offsets[0] = 0;
      for (int f = 0; f < faces.Size(); f++)
      {
         const int c_size = c_fes->GetFaceElement(faces[f])->GetDof() * c_fes->GetVDim();
         c_offsets[f+1] = c_offsets[f] + c_size;
      }

      x_l.Update(c_offsets);
      for (int f = 0; f < faces.Size(); f++)
      {
         c_fes->GetFaceVDofs(faces[f], c_dofs);
         x.GetSubVector(c_dofs, x_l.GetBlock(f));
      }

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      // bu - C^T x
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = GetCtFaceMatrix(faces[f], Ct_1, Ct_2);
         if (!FTr) { continue; }

         const Vector &x_f = x_l.GetBlock(f);
         DenseMatrix &Ct = (FTr->Elem1No == el)?(Ct_1):(Ct_2);
         Ct.AddMult_a(-1., x_f, bu_l);

         //bp - E x
         if (c_bfi_p || mode == MultNlMode::GradMult)
         {
            if (GetEFaceMatrix(faces[f], E_1, E_2))
            {
               DenseMatrix &E = (FTr->Elem1No == el)?(E_1):(E_2);
               E.AddMult_a(-1., x_f, bp_l);
            }
         }
      }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      // bu - C^T sol
      for (int dof = hat_offsets[el], i = 0; dof < hat_offsets[el+1]; dof++)
      {
         if (hat_dofs_marker[dof] == 1) { continue; }
         bu_l[i++] -= hat_bu[dof];
      }
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK

      if (mode != MultNlMode::GradMult)
      {
         //local u
         if (darcy_u.Size() > 0)
         {
            //load the initial guess from the non-reduced solution vector
            darcy_u.GetSubVector(u_vdofs, u_l);
         }
         else
         {
            u_l.SetSize(u_vdofs.Size());
            u_l = 0.;//initial guess?

         }

         //local p
         if (darcy_p.Size() > 0)
         {
            //load the initial guess from the non-reduced solution vector
            darcy_p.GetSubVector(p_dofs, p_l);
         }
         else
         {
            p_l.SetSize(p_dofs.Size());
            p_l = 0.;//initial guess?
         }

         //(A^-1 - A^-1 B^T S^-1 B A^-1) (bu - C^T sol)
         MultInvNL(el, bu_l, bp_l, x_l, u_l, p_l);

         if (mode == MultNlMode::Sol)
         {
            yb.GetBlock(0).SetSubVector(u_vdofs, u_l);
            yb.GetBlock(1).SetSubVector(p_dofs, p_l);
            continue;
         }
         else if (mode == MultNlMode::Grad)
         {
            ConstructGrad(el, faces, x_l, u_l, p_l);
            continue;
         }
      }
      else
      {
         //(A^-1 - A^-1 B^T S^-1 B A^-1) (bu - C^T sol)
         MultInv(el, bu_l, bp_l, u_l, p_l);
      }

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      // C u_l
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = GetCtFaceMatrix(faces[f], Ct_1, Ct_2);
         if (!FTr) { continue; }

         const Vector &x_f = x_l.GetBlock(f);

         DenseMatrix &Ct = (FTr->Elem1No == el)?(Ct_1):(Ct_2);
         y_l.SetSize(x_f.Size());
         Ct.MultTranspose(u_l, y_l);

         //G p_l + H x_l
         if (c_bfi_p || mode == MultNlMode::GradMult)
         {
            //linear
            if (GetGFaceMatrix(faces[f], G_1, G_2))
            {
               DenseMatrix &G = (FTr->Elem1No == el)?(G_1):(G_2);
               G.AddMult(p_l, y_l);
            }

            //integrate the face contrbution only on one (first) side
            if (FTr->Elem1No == el)
            {
               GetHFaceMatrix(faces[f], H);
               H.AddMult(x_f, y_l);
            }
         }
         else
         {
            //nonlinear
            Vector GpHx_l;
            int type = NonlinearFormIntegrator::HDGFaceType::CONSTR
                       | NonlinearFormIntegrator::HDGFaceType::FACE;
            if (FTr->Elem2No >= 0)
            {
               //interior
               if (c_nlfi_p)
               {
                  if (FTr->Elem1No != el) { type |= 1; }

                  c_nlfi_p->AssembleHDGFaceVector(type,
                                                  *c_fes->GetFaceElement(faces[f]),
                                                  *fes_p->GetFE(el),
                                                  *fes->GetMesh()->GetInteriorFaceTransformations(faces[f]),
                                                  x_f, p_l, GpHx_l);

                  y_l += GpHx_l;
               }
            }
            else
            {
               //boundary

               const int bdr_attr = fes->GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

               for (int i = 0; i < boundary_constraint_pot_nonlin_integs.Size(); i++)
               {
                  if (boundary_constraint_pot_nonlin_integs_marker[i]
                      && (*boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

                  boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                                  *c_fes->GetFaceElement(faces[f]),
                                                                                  *fes_p->GetFE(el),
                                                                                  *fes->GetMesh()->GetFaceElementTransformations(faces[f]),
                                                                                  x_f, p_l, GpHx_l);

                  y_l += GpHx_l;
               }
            }
         }

         c_fes->GetFaceVDofs(faces[f], c_dofs);
         y.AddElementVector(c_dofs, y_l);
      }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      // hat_u += u_l
      for (int dof = hat_offsets[el], i = 0; dof < hat_offsets[el+1]; dof++)
      {
         if (hat_dofs_marker[dof] == 1) { continue; }
         hat_u[dof] += u_l[i++];
      }
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   }
#ifndef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   if (mode == 0)
   {
      //C u
      Ct->MultTranspose(hat_u, y);
   }
#endif //!MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
}

void DarcyHybridization::Finalize()
{
   if (bfin) { return; }

   if (!bnl)
   {
      ComputeH();
   }
   else
   {
      if (!m_nlfi_u && !m_nlfi)
      {
         lop_type = LocalOpType::PotNL;
         InvertA();
      }
      else if (!m_nlfi_p && !c_nlfi_p && !D_empty && !m_nlfi)
      {
         lop_type = LocalOpType::FluxNL;
         InvertD();
      }
      else
      {
         lop_type = LocalOpType::FullNL;
         if (!A_empty)
         {
            std::swap(Af_data, Af_lin_data);
            if (!Af_data)
            {
               Af_data = new real_t[Af_offsets.Last()]();
            }
         }

         if (!D_empty)
         {
            std::swap(Df_data, Df_lin_data);
            if (!Df_data)
            {
               Df_data = new real_t[Df_offsets.Last()]();
            }
         }
      }
   }

   bfin = true;
}

void DarcyHybridization::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                             const BlockVector &x, BlockVector &b)
{
   if (bnl)
   {
      //save the rhs for initial guess in the iterative local solve
      darcy_u = x.GetBlock(0);
      darcy_p = x.GetBlock(1);
   }

   const int NE = fes->GetNE();
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

      fes->GetElementVDofs(el, u_vdofs);
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

void DarcyHybridization::MultInvNL(int el, const Vector &bu_l,
                                   const Vector &bp_l, const BlockVector &x_l,
                                   Vector &u_l, Vector &p_l) const
{
   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

   MFEM_ASSERT(bu_l.Size() == a_dofs_size &&
               bp_l.Size() == d_dofs_size, "Incompatible size");

   //prepare vector of local traces

   Array<int> faces, oris;
   const int dim = fes->GetMesh()->Dimension();
   switch (dim)
   {
      case 1:
         fes->GetMesh()->GetElementVertices(el, faces);
         break;
      case 2:
         fes->GetMesh()->GetElementEdges(el, faces, oris);
         break;
      case 3:
         fes->GetMesh()->GetElementFaces(el, faces, oris);
         break;
   }

   //construct the local operator

   LocalNLOperator *lop;

   switch (lop_type)
   {
      case LocalOpType::FluxNL:
         lop = new LocalFluxNLOperator(*this, el, bp_l, x_l, faces);
         break;
      case LocalOpType::PotNL:
         lop = new LocalPotNLOperator(*this, el, bu_l, x_l, faces);
         break;
      case LocalOpType::FullNL:
         lop = new LocalNLOperator(*this, el, x_l, faces);
         break;
   }

   //solve the local system

   IterativeSolver *lsolver;
   bool use_prec;
   switch (lsolve.type)
   {
      case LSsolveType::LBFGS:
         lsolver = new LBFGSSolver();
         use_prec = false;
         break;
      case LSsolveType::LBB:
         lsolver = new LBBSolver();
         use_prec = false;
         break;
      case LSsolveType::Newton:
         lsolver = new NewtonSolver();
         use_prec = true;
         break;
      default:
         MFEM_ABORT("Unknown local solver");
   }

   IterativeSolver *prec = NULL;
   if (use_prec)
   {
      switch (lsolve.prec.type)
      {
         case LPrecType::GMRES:
            prec = new GMRESSolver();
            break;
         default:
            MFEM_ABORT("Unknown local preconditioner");
      }

      prec->SetMaxIter(lsolve.prec.iters);
      prec->SetRelTol((lsolve.prec.rtol >= 0)?
                      (lsolve.prec.rtol):(lsolve.rtol));
      prec->SetAbsTol((lsolve.prec.atol >= 0)?
                      (lsolve.prec.atol):(lsolve.atol));
   }

   lsolver->SetOperator(*lop);
   if (prec) { lsolver->SetPreconditioner(*prec); }
   lsolver->SetMaxIter(lsolve.iters);
   lsolver->SetRelTol(lsolve.rtol);
   lsolver->SetAbsTol(lsolve.atol);
   lsolver->SetPrintLevel(lsolve.print_lvl);

   switch (lop_type)
   {
      case LocalOpType::FluxNL:
      {
         //solve the flux
         lsolver->Mult(bu_l, u_l);

         //solve the potential
         static_cast<LocalFluxNLOperator*>(lop)->SolveP(u_l, p_l);
      }
      break;
      case LocalOpType::PotNL:
      {
         //solve the potential
         lsolver->Mult(bp_l, p_l);

         //solve the flux
         static_cast<LocalPotNLOperator*>(lop)->SolveU(p_l, u_l);
      }
      break;
      case LocalOpType::FullNL:
      {
         //rhs
         BlockVector b(lop->GetOffsets());
         b.GetBlock(0) = bu_l;
         b.GetBlock(1) = bp_l;

         //x
         BlockVector x(lop->GetOffsets());
         x.GetBlock(0) = u_l;
         x.GetBlock(1) = p_l;

         //solve the flux and potential
         lsolver->Mult(b, x);

         u_l = x.GetBlock(0);
         p_l = x.GetBlock(1);
      }
      break;
   }

   if (lsolver->GetConverged())
   {
      if (lsolve.print_lvl >= 0)
         std::cout << "el: " << el
                   << " iters: " << lsolver->GetNumIterations()
                   << " rel. norm: " << lsolver->GetFinalRelNorm()
                   << std::endl;
   }
   else
   {
      std::cout << "el: " << el
                << " not convered in " << lsolver->GetNumIterations() << " iters"
                << " rel. norm: " << lsolver->GetFinalRelNorm()
                << std::endl;
   }

   delete lsolver;
   delete prec;
   delete lop;
}

void DarcyHybridization::MultInv(int el, const Vector &bu, const Vector &bp,
                                 Vector &u, Vector &p) const
{
   Vector AiBtSiBAibu, AiBtSibp;

   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

   MFEM_ASSERT(bu.Size() == a_dofs_size &&
               bp.Size() == d_dofs_size, "Incompatible size");

   // Load LU decomposition of A and Schur complement

   LUFactors LU_A(Af_data + Af_offsets[el], Af_ipiv + Af_f_offsets[el]);
   LUFactors LU_S(Df_data + Df_offsets[el], Df_ipiv + Df_f_offsets[el]);

   // Load B

   DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);

   //u = A^-1 bu
   u.SetSize(bu.Size());
   u = bu;
   LU_A.Solve(u.Size(), 1, u.GetData());

   //p = -S^-1 (B A^-1 bu - bp)
   p.SetSize(bp.Size());
   B.Mult(u, p);

   p -= bp;

   LU_S.Solve(p.Size(), 1, p.GetData());
   p.Neg();

   //u += -A^-1 B^T S^-1 (B A^-1 bu - bp)
   AiBtSiBAibu.SetSize(B.Width());
   B.MultTranspose(p, AiBtSiBAibu);

   LU_A.Solve(AiBtSiBAibu.Size(), 1, AiBtSiBAibu.GetData());

   if (bsym) { u += AiBtSiBAibu; }
   else { u -= AiBtSiBAibu; }
}

void DarcyHybridization::ConstructGrad(int el, const Array<int> &faces,
                                       const BlockVector &x_l,
                                       const Vector &u_l, const Vector &p_l) const
{
   const FiniteElement *fe_u = fes->GetFE(el);
   const FiniteElement *fe_p = fes_p->GetFE(el);
   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   ElementTransformation *Tr = fes->GetElementTransformation(el);

   DenseMatrix A(Af_data + Af_offsets[el], a_dofs_size, a_dofs_size);
   DenseMatrix D(Df_data + Df_offsets[el], d_dofs_size, d_dofs_size);
   LUFactors LU_A(A.GetData(), Af_ipiv + Af_f_offsets[el]);

   if (m_nlfi)
   {
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array2D<DenseMatrix*> grad_arr(2,2);
      grad_arr(0,0) = &A;
      grad_arr(1,0) = NULL;
      grad_arr(0,1) = NULL;
      grad_arr(1,1) = &D;
      m_nlfi->AssembleElementGrad(fe_arr, *Tr, x_arr, grad_arr);
   }
   else
   {
      A = 0.;
      D = 0.;
   }

   if (m_nlfi_u)
   {
      DenseMatrix grad_A;
      m_nlfi_u->AssembleElementGrad(*fe_u, *Tr, u_l, grad_A);
      A += grad_A;
   }
   else if (!A_empty)
   {
      DenseMatrix A_lin(Af_lin_data + Af_offsets[el], a_dofs_size, a_dofs_size);
      A += A_lin;
   }

   if (m_nlfi_p)
   {
      DenseMatrix grad_D;
      m_nlfi_p->AssembleElementGrad(*fe_p, *Tr, p_l, grad_D);
      D += grad_D;
   }
   else if (!D_empty)
   {
      DenseMatrix D_lin(Df_lin_data + Df_offsets[el], d_dofs_size, d_dofs_size);
      D += D_lin;
   }

   if (c_nlfi_p)
   {
      //bp += E x
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = fes->GetMesh()->GetFaceElementTransformations(
                                              faces[f], 0);

         const Vector &x_f = x_l.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior
            AssembleHDGGrad(el, faces[f], *c_nlfi_p, x_f, p_l);
         }
         else
         {
            //boundary
            const int bdr_attr = fes->GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

            for (int i = 0; i < boundary_constraint_pot_nonlin_integs.Size(); i++)
            {
               if (boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               AssembleHDGGrad(el, faces[f], *boundary_constraint_pot_nonlin_integs[i], x_f,
                               p_l);
            }
         }
      }
   }

   if (m_nlfi_u || m_nlfi)
   {
      // Decompose A
      LU_A.Factor(a_dofs_size);
   }

   // Construct Schur complement
   DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);
   DenseMatrix AiBt(a_dofs_size, d_dofs_size);

   AiBt.Transpose(B);
   if (!bsym) { AiBt.Neg(); }
   LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());
   mfem::AddMult(B, AiBt, D);

   // Decompose Schur complement
   LUFactors LU_S(D.GetData(), Df_ipiv + Df_f_offsets[el]);

   LU_S.Factor(d_dofs_size);
}

void DarcyHybridization::AssembleHDGGrad(int el, int f,
                                         NonlinearFormIntegrator &nlfi, const Vector &x_f, const Vector &p_l) const
{
   const FiniteElement *fe_c = c_fes->GetFaceElement(f);
   const FiniteElement *fe_p = fes_p->GetFE(el);
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   const int c_dofs_size = x_f.Size();

   FaceElementTransformations *FTr =
      fes->GetMesh()->GetFaceElementTransformations(f);

   int type = NonlinearFormIntegrator::HDGFaceType::ELEM
              | NonlinearFormIntegrator::HDGFaceType::TRACE
              | NonlinearFormIntegrator::HDGFaceType::CONSTR
              | NonlinearFormIntegrator::HDGFaceType::FACE;

   if (FTr->Elem1No != el) { type |= 1; }

   DenseMatrix elmat;

   nlfi.AssembleHDGFaceGrad(type, *fe_c, *fe_p, *FTr, x_f, p_l, elmat);

   // assemble D element matrices
   DenseMatrix D(Df_data + Df_offsets[el], d_dofs_size, d_dofs_size);
   DenseMatrix elmat_D;
   elmat_D.CopyMN(elmat, d_dofs_size, d_dofs_size, 0, 0);
   D += elmat_D;

   // assemble E constraint
   const int E_off = (FTr->Elem1No == el)?(0):(c_dofs_size*d_dofs_size);
   DenseMatrix E_f(E_data + E_offsets[f] + E_off, d_dofs_size, c_dofs_size);
   E_f.CopyMN(elmat, d_dofs_size, c_dofs_size, 0, d_dofs_size);

   // assemble G constraint
   const int G_off = E_off;
   DenseMatrix G_f(G_data + G_offsets[f] + G_off, c_dofs_size, d_dofs_size);
   G_f.CopyMN(elmat, c_dofs_size, d_dofs_size, d_dofs_size, 0);

   // assemble H matrix
   DenseMatrix H_f(H_data + H_offsets[f], c_dofs_size, c_dofs_size);
   DenseMatrix elmat_H;
   elmat_H.CopyMN(elmat, c_dofs_size, c_dofs_size, d_dofs_size, d_dofs_size);
   H_f += elmat_H;
}

void DarcyHybridization::ReduceRHS(const BlockVector &b, Vector &b_r) const
{
   if (bnl)
   {
      //store RHS for Mult
      if (!darcy_offsets.Size())
      {
         darcy_offsets.SetSize(3);
         darcy_offsets[0] = 0;
         darcy_offsets[1] = fes->GetVSize();
         darcy_offsets[2] = fes_p->GetVSize();
         darcy_offsets.PartialSum();

         darcy_rhs.Update(darcy_offsets);
      }
      darcy_rhs = b;
      if (b_r.Size() != Height())
      {
         b_r.SetSize(Height());
         b_r = 0.;
      }
      return;
   }

   const int NE = fes->GetNE();
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   const int dim = fes->GetMesh()->Dimension();
   DenseMatrix Ct_1, Ct_2, G_1, G_2;
   Vector b_rl;
   Array<int> c_dofs;
   Array<int> faces, oris;
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   MFEM_ASSERT(!c_bfi_p,
               "Potential constraint is not supported in non-block assembly!");
   Vector hat_u(hat_offsets.Last());
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Vector bu_l, bp_l, u_l, p_l;
   Array<int> u_vdofs, p_dofs;

   if (b_r.Size() != H->Height())
   {
      b_r.SetSize(H->Height());
      b_r = 0.;
   }

   const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);

   for (int el = 0; el < NE; el++)
   {
      // Load RHS

      GetFDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      fes_p->GetElementDofs(el, p_dofs);
      bp.GetSubVector(p_dofs, bp_l);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         bp_l.Neg();
      }

      //-A^-1 bu - A^-1 B^T S^-1 B A^-1 bu
      MultInv(el, bu_l, bp_l, u_l, p_l);
      u_l.Neg();
      p_l.Neg();

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      switch (dim)
      {
         case 1:
            fes->GetMesh()->GetElementVertices(el, faces);
            break;
         case 2:
            fes->GetMesh()->GetElementEdges(el, faces, oris);
            break;
         case 3:
            fes->GetMesh()->GetElementFaces(el, faces, oris);
            break;
      }

      // Mult C u + G p
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = GetCtFaceMatrix(faces[f], Ct_1, Ct_2);
         if (!FTr) { continue; }

         DenseMatrix &Ct = (FTr->Elem1No == el)?(Ct_1):(Ct_2);
         b_rl.SetSize(Ct.Width());
         Ct.MultTranspose(u_l, b_rl);

         if (c_bfi_p)
         {
            if (GetGFaceMatrix(faces[f], G_1, G_2))
            {
               DenseMatrix &G = (FTr->Elem1No == el)?(G_1):(G_2);
               G.AddMult(p_l, b_rl);
            }
         }

         c_fes->GetFaceVDofs(faces[f], c_dofs);
         b_r.AddElementVector(c_dofs, b_rl);
      }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      int i = 0;
      for (int dof = hat_offsets[el]; dof < hat_offsets[el+1]; dof++)
      {
         if (hat_dofs_marker[dof] == 1) { continue; }
         hat_u[dof] = u_l[i++];
      }
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   }

#ifndef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Ct->MultTranspose(hat_u, b_r);
#endif //!MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
}

void DarcyHybridization::ComputeSolution(const BlockVector &b,
                                         const Vector &sol_r, BlockVector &sol) const
{
   if (bnl)
   {
      MultNL(MultNlMode::Sol, b, sol_r, sol);
      return;
   }

   const int NE = fes->GetNE();
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   const int dim = fes->GetMesh()->Dimension();
   DenseMatrix Ct_1, Ct_2, E_1, E_2;
   Vector sol_rl;
   Array<int> c_dofs;
   Array<int> faces, oris;
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   MFEM_ASSERT(!c_bfi_p,
               "Potential constraint is not supported in non-block assembly!");
   Vector hat_bu(hat_offsets.Last());
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Vector bu_l, bp_l, u_l, p_l;
   Array<int> u_vdofs, p_dofs;

   const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);
   Vector &u = sol.GetBlock(0);
   Vector &p = sol.GetBlock(1);

#ifndef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Ct->Mult(sol_r, hat_bu);
#endif //!MFEM_DARCY_HYBRIDIZATION_CT_BLOCK

   for (int el = 0; el < NE; el++)
   {
      //Load RHS

      GetFDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      fes_p->GetElementDofs(el, p_dofs);
      bp.GetSubVector(p_dofs, bp_l);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         bp_l.Neg();
      }

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      switch (dim)
      {
         case 1:
            fes->GetMesh()->GetElementVertices(el, faces);
            break;
         case 2:
            fes->GetMesh()->GetElementEdges(el, faces, oris);
            break;
         case 3:
            fes->GetMesh()->GetElementFaces(el, faces, oris);
            break;
      }

      // bu - C^T sol
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = GetCtFaceMatrix(faces[f], Ct_1, Ct_2);
         if (!FTr) { continue; }

         c_fes->GetFaceVDofs(faces[f], c_dofs);
         sol_r.GetSubVector(c_dofs, sol_rl);
         DenseMatrix &Ct = (FTr->Elem1No == el)?(Ct_1):(Ct_2);
         Ct.AddMult_a(-1., sol_rl, bu_l);

         //bp - E sol
         if (c_bfi_p)
         {
            if (GetEFaceMatrix(faces[f], E_1, E_2))
            {
               DenseMatrix &E = (FTr->Elem1No == el)?(E_1):(E_2);
               E.AddMult_a(-1., sol_rl, bp_l);
            }
         }
      }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      // bu - C^T sol
      int i = 0;
      for (int dof = hat_offsets[el]; dof < hat_offsets[el+1]; dof++)
      {
         if (hat_dofs_marker[dof] == 1) { continue; }
         bu_l[i++] -= hat_bu[dof];
      }
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK

      //(A^-1 - A^-1 B^T S^-1 B A^-1) (bu - C^T sol)
      MultInv(el, bu_l, bp_l, u_l, p_l);

      u.SetSubVector(u_vdofs, u_l);
      p.SetSubVector(p_dofs, p_l);
   }
}

void DarcyHybridization::Reset()
{
   Hybridization::Reset();
   bfin = false;

   A_empty = true;
   memset(Bf_data, 0, Bf_offsets.Last() * sizeof(real_t));
   if (Df_data)
   {
      memset(Df_data, 0, Df_offsets.Last() * sizeof(real_t));
      D_empty = true;
   }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   memset(Be_data, 0, Be_offsets.Last() * sizeof(real_t));
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
}

void DarcyHybridization::Gradient::Mult(const Vector &x, Vector &y) const
{
   //note that rhs is not used, it is only a dummy
   dh.MultNL(MultNlMode::GradMult, dh.darcy_rhs, x, y);
}

DarcyHybridization::LocalNLOperator::LocalNLOperator(
   const DarcyHybridization &dh_, int el_, const BlockVector &trps_,
   const Array<int> &faces_)
   : dh(dh_), el(el_), trps(trps_), faces(faces_),
     a_dofs_size(dh.Af_f_offsets[el+1] - dh.Af_f_offsets[el]),
     d_dofs_size(dh.Df_f_offsets[el+1] - dh.Df_f_offsets[el]),
     B(dh.Bf_data + dh.Bf_offsets[el], d_dofs_size, a_dofs_size),
     Bt(B), offsets({0, a_dofs_size, a_dofs_size+d_dofs_size}), grad(offsets)
{
   width = height = a_dofs_size + d_dofs_size;

   fe_u = dh.fes->GetFE(el);
   fe_p = dh.fes_p->GetFE(el);

   // element transformation
   Tr = new IsoparametricTransformation();
   if (faces.Size() <= 0)
   {
      dh.fes_p->GetMesh()->GetElementTransformation(el, Tr);
   }

   // face transformations
   FTrs.SetSize(faces.Size());
   NbrTrs.SetSize(faces.Size());
   for (int f = 0; f < faces.Size(); f++)
   {
      FaceElementTransformations *&FTr = FTrs[f];
      FTr = new FaceElementTransformations();
      dh.fes_p->GetMesh()->GetFaceElementTransformations(faces[f], *FTr, *Tr, *Tr, 0);
      IsoparametricTransformation *Tr1, *Tr2;
      if (FTr->Elem2No >= 0)
      {
         NbrTrs[f] = new IsoparametricTransformation();
         if (FTr->Elem1No == el)
         {
            Tr1 = Tr;
            Tr2 = NbrTrs[f];
         }
         else
         {
            Tr1 = NbrTrs[f];
            Tr2 = Tr;
         }
      }
      else
      {
         NbrTrs[f] = NULL;
         Tr1 = Tr2 = Tr;
      }

      dh.fes_p->GetMesh()->GetFaceElementTransformations(faces[f], *FTr, *Tr1, *Tr2);
   }
}

DarcyHybridization::LocalNLOperator::~LocalNLOperator()
{
   delete Tr;
   for (int f = 0; f < faces.Size(); f++)
   {
      delete FTrs[f];
      delete NbrTrs[f];
   }
}

void DarcyHybridization::LocalNLOperator::AddMultA(const Vector &u_l,
                                                   Vector &bu) const
{
   //bu += A u_l
   if (dh.m_nlfi_u)
   {
      dh.m_nlfi_u->AssembleElementVector(*fe_u, *Tr, u_l, Au);
      bu += Au;
   }
   else if (!dh.A_empty)
   {
      DenseMatrix A(dh.Af_lin_data + dh.Af_offsets[el], a_dofs_size, a_dofs_size);
      A.AddMult(u_l, bu);
   }
}

void DarcyHybridization::LocalNLOperator::AddMultDE(const Vector &p_l,
                                                    Vector &bp) const
{
   //bp += D p_l
   if (dh.m_nlfi_p)
   {
      dh.m_nlfi_p->AssembleElementVector(*fe_p, *Tr, p_l, Dp);
      bp += Dp;
   }
   else if (!dh.D_empty)
   {
      DenseMatrix D(dh.Df_lin_data + dh.Df_offsets[el], d_dofs_size, d_dofs_size);
      D.AddMult(p_l, bp);
   }

   if (dh.c_nlfi_p)
   {
      //bp += E x
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = FTrs[f];

         int type = NonlinearFormIntegrator::HDGFaceType::ELEM
                    | NonlinearFormIntegrator::HDGFaceType::TRACE;

         const Vector &trp_f = trps.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior
            if (FTr->Elem1No != el) { type |= 1; }

            dh.c_nlfi_p->AssembleHDGFaceVector(type, *dh.c_fes->GetFaceElement(faces[f]),
                                               *fe_p, *FTr, trp_f, p_l, DpEx);

            bp += DpEx;
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes->GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (int i = 0; i < dh.boundary_constraint_pot_nonlin_integs.Size(); i++)
            {
               if (dh.boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               dh.boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                                  *dh.c_fes->GetFaceElement(faces[f]),
                                                                                  *fe_p, *FTr, trp_f, p_l, DpEx);

               bp += DpEx;
            }
         }
      }
   }
}

void DarcyHybridization::LocalNLOperator::AddGradA(const Vector &u_l,
                                                   DenseMatrix &grad) const
{
   //grad += A
   if (dh.m_nlfi_u)
   {
      DenseMatrix grad_A;
      dh.m_nlfi_u->AssembleElementGrad(*fe_u, *Tr, u_l, grad_A);
      grad += grad_A;
   }
   else if (!dh.A_empty)
   {
      DenseMatrix A(dh.Af_lin_data + dh.Af_offsets[el], a_dofs_size, a_dofs_size);
      grad += A;
   }
}

void DarcyHybridization::LocalNLOperator::AddGradDE(const Vector &p_l,
                                                    DenseMatrix &grad) const
{
   //grad += D
   if (dh.m_nlfi_p)
   {
      DenseMatrix grad_D;
      dh.m_nlfi_p->AssembleElementGrad(*fe_p, *Tr, p_l, grad_D);
      grad += grad_D;
   }
   else if (!dh.D_empty)
   {
      DenseMatrix D(dh.Df_lin_data + dh.Df_offsets[el], d_dofs_size, d_dofs_size);
      grad += D;
   }

   if (dh.c_nlfi_p)
   {
      DenseMatrix grad_Df;

      //grad += D_f
      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = FTrs[f];

         int type = NonlinearFormIntegrator::HDGFaceType::ELEM;

         const Vector &trp_f = trps.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior
            if (FTr->Elem1No != el) { type |= 1; }

            dh.c_nlfi_p->AssembleHDGFaceGrad(type, *dh.c_fes->GetFaceElement(faces[f]),
                                             *fe_p, *FTr, trp_f, p_l, grad_Df);

            grad += grad_Df;
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes->GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (int i = 0; i < dh.boundary_constraint_pot_nonlin_integs.Size(); i++)
            {
               if (dh.boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               dh.boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceGrad(type,
                                                                                *dh.c_fes->GetFaceElement(faces[f]),
                                                                                *fe_p, *FTr, trp_f, p_l, grad_Df);

               grad += grad_Df;
            }
         }
      }
   }
}

void DarcyHybridization::LocalNLOperator::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width() && y.Size() == Height(), "Incompatible size");

   const BlockVector x_l(const_cast<Vector&>(x), offsets);
   const Vector &u_l = x_l.GetBlock(0);
   const Vector &p_l = x_l.GetBlock(1);
   BlockVector b(y, offsets);
   Vector &bu = b.GetBlock(0);
   Vector &bp = b.GetBlock(1);

   //bu = B^T p
   B.MultTranspose(p_l, bu);
   if (dh.bsym) { bu.Neg(); }

   //bu += A u
   AddMultA(u_l, bu);

   //bp = B u
   B.Mult(u_l, bp);

   //bp += D p
   AddMultDE(p_l, bp);

   //bu += A u_l - B^T p_l
   //bp += B u_l + D p_l
   if (dh.m_nlfi)
   {
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array<Vector*> y_arr({&Au, &Dp});
      dh.m_nlfi->AssembleElementVector(fe_arr, *Tr, x_arr, y_arr);
      bu += Au;
      bp += Dp;
   }
}

Operator &DarcyHybridization::LocalNLOperator::GetGradient(
   const Vector &x) const
{
   MFEM_ASSERT(x.Size() == Width(), "Incompatible size");

   const BlockVector x_l(const_cast<Vector&>(x), offsets);
   const Vector &u_l = x_l.GetBlock(0);
   const Vector &p_l = x_l.GetBlock(1);

   grad_A.SetSize(a_dofs_size);
   grad_D.SetSize(d_dofs_size);

   if (dh.m_nlfi)
   {
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array2D<DenseMatrix*> grad_arr(2,2);
      grad_arr(0,0) = &grad_A;
      grad_arr(1,0) = NULL;
      grad_arr(0,1) = NULL;
      grad_arr(1,1) = &grad_D;
      dh.m_nlfi->AssembleElementGrad(fe_arr, *Tr, x_arr, grad_arr);
   }
   else
   {
      grad_A = 0.;
      grad_D = 0.;
   }

   //A
   AddGradA(u_l, grad_A);
   grad.SetDiagonalBlock(0, &grad_A);

   //B
   grad.SetBlock(1, 0, &const_cast<DenseMatrix&>(B));

   //B^T
   grad.SetBlock(0, 1, &const_cast<TransposeOperator&>(Bt), (dh.bsym)?(-1.):(+1.));

   //D
   AddGradDE(p_l, grad_D);
   grad.SetDiagonalBlock(1, &grad_D);

   return grad;
}

DarcyHybridization::LocalFluxNLOperator::LocalFluxNLOperator(
   const DarcyHybridization &dh_, int el_, const Vector &bp_,
   const BlockVector &trps_, const Array<int> &faces_)
   : LocalNLOperator(dh_, el_, trps_, faces_), bp(bp_),
     LU_D(dh.Df_data + dh.Df_offsets[el], dh.Df_ipiv + dh.Df_f_offsets[el])
{
   MFEM_ASSERT(bp.Size() == d_dofs_size, "Incompatible size");

   width = height = a_dofs_size;
}

void DarcyHybridization::LocalFluxNLOperator::SolveP(const Vector &u_l,
                                                     Vector &p_l) const
{
   p_l = bp;

   //bp - E x - B^T p
   B.AddMult(u_l, p_l, -1.);

   //p = D^-1 rp
   LU_D.Solve(d_dofs_size, 1, p_l.GetData());
}

void DarcyHybridization::LocalFluxNLOperator::Mult(const Vector &u_l,
                                                   Vector &bu) const
{
   MFEM_ASSERT(u_l.Size() == a_dofs_size &&
               bu.Size() == a_dofs_size, "Incompatible size");

   SolveP(u_l, p_l);

   //bu = B^T p
   B.MultTranspose(p_l, bu);
   if (dh.bsym) { bu.Neg(); }

   AddMultA(u_l, bu);
}

Operator &DarcyHybridization::LocalFluxNLOperator::GetGradient(
   const Vector &u_l) const
{
   MFEM_ASSERT(u_l.Size() == a_dofs_size, "Incompatible size");

   SolveP(u_l, p_l);

   //grad = B^T D^-1 B
   DenseMatrix DiB = B;

   LU_D.Solve(d_dofs_size, a_dofs_size, DiB.GetData());
   grad_A.SetSize(a_dofs_size);
   MultAtB(B, DiB, grad_A);
   if (!dh.bsym) { grad_A.Neg(); }

   //grad += A
   AddGradA(u_l, grad_A);

   return grad_A;
}

DarcyHybridization::LocalPotNLOperator::LocalPotNLOperator(
   const DarcyHybridization &dh_, int el_, const Vector &bu_,
   const BlockVector &trps_, const Array<int> &faces_)
   : LocalNLOperator(dh_, el_, trps_, faces_), bu(bu_),
     LU_A(dh.Af_data + dh.Af_offsets[el], dh.Af_ipiv + dh.Af_f_offsets[el])
{
   MFEM_ASSERT(bu.Size() == a_dofs_size, "Incompatible size");

   width = height = d_dofs_size;
}

void DarcyHybridization::LocalPotNLOperator::SolveU(const Vector &p_l,
                                                    Vector &u_l) const
{
   u_l = bu;

   //bu - C^T x + B^T p
   B.AddMultTranspose(p_l, u_l, (dh.bsym)?(+1.):(-1.));

   //u = A^-1 ru
   LU_A.Solve(a_dofs_size, 1, u_l.GetData());
}

void DarcyHybridization::LocalPotNLOperator::Mult(const Vector &p_l,
                                                  Vector &bp) const
{
   MFEM_ASSERT(p_l.Size() == d_dofs_size &&
               bp.Size() == d_dofs_size, "Incompatible size");

   SolveU(p_l, u_l);

   //bp = B u
   B.Mult(u_l, bp);

   AddMultDE(p_l, bp);
}

Operator &DarcyHybridization::LocalPotNLOperator::GetGradient(
   const Vector &p_l) const
{
   MFEM_ASSERT(p_l.Size() == d_dofs_size, "Incompatible size");

   SolveU(p_l, u_l);

   //grad = B A^-1 B^T
   DenseMatrix BAi = B;

   LU_A.RightSolve(a_dofs_size, d_dofs_size, BAi.GetData());
   grad_D.SetSize(d_dofs_size);
   MultABt(BAi, B, grad_D);
   if (!dh.bsym) { grad_D.Neg(); }

   //grad += D
   AddGradDE(p_l, grad_D);

   return grad_D;
}

}
