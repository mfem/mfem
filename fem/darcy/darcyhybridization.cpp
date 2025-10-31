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

#include "../../mesh/segment.hpp"
#include "../../mesh/triangle.hpp"
#include "../../mesh/quadrilateral.hpp"

#define MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
#define MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY

namespace mfem
{

DarcyHybridization::DarcyHybridization(FiniteElementSpace *fes_u_,
                                       FiniteElementSpace *fes_p_,
                                       FiniteElementSpace *fes_c_,
                                       bool bsymmetrize)
   : Hybridization(fes_u_, fes_c_), Operator(c_fes.GetVSize()),
     fes_p(*fes_p_), bsym(bsymmetrize)
{
#ifdef MFEM_USE_MPI
   pfes = dynamic_cast<ParFiniteElementSpace*>(&fes);
   pfes_p = dynamic_cast<ParFiniteElementSpace*>(&fes_p);
   c_pfes = dynamic_cast<ParFiniteElementSpace*>(&c_fes);
#endif
   SetLocalNLSolver(LSsolveType::LBFGS);
   SetLocalNLPreconditioner(LPrecType::GMRES);
}

DarcyHybridization::~DarcyHybridization()
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   if (own_m_nlfi_p) { delete m_nlfi_p; }
   if (own_m_nlfi) { delete m_nlfi; }
   if (!extern_bdr_constr_pot_integs)
   {
      for (size_t k=0; k < boundary_constraint_pot_integs.size(); k++)
      { delete boundary_constraint_pot_integs[k]; }
      for (size_t k=0; k < boundary_constraint_pot_nonlin_integs.size(); k++)
      { delete boundary_constraint_pot_nonlin_integs[k]; }
      for (size_t k=0; k < boundary_constraint_nonlin_integs.size(); k++)
      { delete boundary_constraint_nonlin_integs[k]; }
   }
}

void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, BilinearFormIntegrator *c_pot_integ)
{
   MFEM_VERIFY(!m_nlfi_p, "Linear constraint cannot work with a non-linear mass");

   c_bfi.reset(c_flux_integ);
   c_bfi_p.reset(c_pot_integ);
   c_nlfi_p.reset();
   c_nlfi.reset();
}

void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, NonlinearFormIntegrator *c_pot_integ)
{
   c_bfi.reset(c_flux_integ);
   c_bfi_p.reset();
   c_nlfi_p.reset(c_pot_integ);
   c_nlfi.reset();
}

void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, BlockNonlinearFormIntegrator *c_integ)
{
   c_bfi.reset(c_flux_integ);
   c_bfi_p.reset();
   c_nlfi_p.reset();
   c_nlfi.reset(c_integ);
}

void DarcyHybridization::SetFluxMassNonlinearIntegrator(
   NonlinearFormIntegrator *flux_integ, bool own)
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   own_m_nlfi_u = own;
   m_nlfi_u = flux_integ;
}

void DarcyHybridization::SetPotMassNonlinearIntegrator(NonlinearFormIntegrator
                                                       *pot_integ, bool own)
{
   MFEM_VERIFY(!c_bfi_p, "Non-linear mass cannot work with a linear constraint");

   if (own_m_nlfi_p) { delete m_nlfi_p; }
   own_m_nlfi_p = own;
   m_nlfi_p = pot_integ;
}

void DarcyHybridization::SetBlockNonlinearIntegrator(
   BlockNonlinearFormIntegrator *block_integ, bool own)
{
   if (own_m_nlfi) { delete m_nlfi; }
   own_m_nlfi = own;
   m_nlfi = block_integ;
}

void DarcyHybridization::Init(const Array<int> &ess_flux_tdof_list)
{
   const int NE = fes.GetNE();

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   if (Ct_data.Size()) { return; }

   // count the number of dofs in the discontinuous version of fes:
   Array<int> vdofs;
   int num_hat_dofs = 0;
   hat_offsets.SetSize(NE+1);
   hat_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      fes.GetElementVDofs(i, vdofs);
      num_hat_dofs += vdofs.Size();
      hat_offsets[i+1] = num_hat_dofs;
   }

   // Define the "free" (0) and "essential" (1) hat_dofs.
   // The "essential" hat_dofs are those that depend only on essential cdofs;
   // all other hat_dofs are "free".
   hat_dofs_marker.SetSize(num_hat_dofs);
   Array<int> free_tdof_marker;
   if (ParallelU())
   {
#ifdef MFEM_USE_MPI
      free_tdof_marker.SetSize(pfes->TrueVSize());
#endif
   }
   else
   {
      free_tdof_marker.SetSize(fes.GetConformingVSize());
   }

   free_tdof_marker = 1;
   for (int i = 0; i < ess_flux_tdof_list.Size(); i++)
   {
      free_tdof_marker[ess_flux_tdof_list[i]] = 0;
   }
   Array<int> free_vdofs_marker;
   if (ParallelU())
   {
#ifdef MFEM_USE_MPI
      HypreParMatrix *P = pfes->Dof_TrueDof_Matrix();
      free_vdofs_marker.SetSize(fes.GetVSize());
      P->BooleanMult(1, free_tdof_marker, 0, free_vdofs_marker);
#endif
   }
   else
   {
      const SparseMatrix *cP = fes.GetConformingProlongation();
      if (!cP)
      {
         free_vdofs_marker.MakeRef(free_tdof_marker);
      }
      else
      {
         free_vdofs_marker.SetSize(fes.GetVSize());
         cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
      }
   }

   for (int i = 0; i < NE; i++)
   {
      fes.GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         hat_dofs_marker[hat_offsets[i]+j] = ! free_vdofs_marker[vdofs[j]];
      }
   }

   free_tdof_marker.DeleteAll();
   free_vdofs_marker.DeleteAll();

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

   Af_data.SetSize(Af_offsets[NE]);
   Af_ipiv.SetSize(Af_f_offsets[NE]);

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
      int d_size = fes_p.GetFE(i)->GetDof() * fes_p.GetVDim();
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

   Bf_data.SetSize(Bf_offsets[NE]); Bf_data = 0.;
   if (!m_nlfi_p)
   {
      AllocD();
   }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Ae_data.SetSize(Ae_offsets[NE]);
   Be_data.SetSize(Be_offsets[NE]); Be_data = 0.;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   if (c_bfi_p)
   {
      AllocEG();
      if (IsNonlinear())
      {
         AllocH();
      }
   }
}

void DarcyHybridization::AssembleFluxMassMatrix(int el, const DenseMatrix &A)
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   int Af_el_idx = Af_offsets[el];
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   int Ae_el_idx = Ae_offsets[el];
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   for (int j = 0; j < s; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         for (int i = 0; i < s; i++)
         {
            Ae_data[Ae_el_idx++] = A(i, j);
         }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < s; i++)
      {
         if (hat_dofs_marker[o + i] == 1) { continue; }
         Af_data[Af_el_idx++] = A(i, j);
      }
   }
   MFEM_ASSERT(Af_el_idx == Af_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   MFEM_ASSERT(Ae_el_idx == Ae_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   A_empty = false;
}

void DarcyHybridization::AssemblePotMassMatrix(int el, const DenseMatrix &D)
{
   const int s = Df_f_offsets[el+1] - Df_f_offsets[el];
   DenseMatrix D_i(&Df_data[Df_offsets[el]], s, s);
   MFEM_ASSERT(D.Size() == s, "Incompatible sizes");

   D_i += D;

   D_empty = false;
}

void DarcyHybridization::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int o = hat_offsets[el];
   const int w = hat_offsets[el+1] - o;
   const int h = Df_f_offsets[el+1] - Df_f_offsets[el];
   int Bf_el_idx = Bf_offsets[el];
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   int Be_el_idx = Be_offsets[el];
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS

   for (int j = 0; j < w; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         for (int i = 0; i < h; i++)
         {
            Be_data[Be_el_idx++] += B(i, j);
         }
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < h; i++)
      {
         Bf_data[Bf_el_idx++] += B(i, j);
      }
   }
   MFEM_ASSERT(Bf_el_idx == Bf_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   MFEM_ASSERT(Be_el_idx == Be_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
}

void DarcyHybridization::ComputeAndAssemblePotFaceMatrix(
   int face, DenseMatrix &elmat1, DenseMatrix &elmat2, Array<int> &vdofs1,
   Array<int> &vdofs2, int skip_zeros)
{
   Mesh *mesh = fes_p.GetMesh();
   const int num_faces = mesh->GetNumFaces();
   const FiniteElement *tr_fe, *fe1, *fe2;
   DenseMatrix elmat;
   int ndof1, ndof2;
   bool save2 = false;

   tr_fe = c_fes.GetFaceElement(face);
   const int c_dof = tr_fe->GetDof() * c_fes.GetVDim();

   int el1, el2;
   mesh->GetFaceElements(face, &el1, &el2);
   fes_p.GetElementVDofs(el1, vdofs1);
   fe1 = fes_p.GetFE(el1);
   ndof1 = vdofs1.Size();
   FaceElementTransformations *ftr = NULL;

   int inf1, inf2, nc;
   if (mesh->Nonconforming())
   {
      mesh->GetFaceInfos(face, &inf1, &inf2, &nc);
      MFEM_ASSERT(nc < 0 || el2 >= 0 ||
                  inf2 >= 0, "Master face should not be integrated directly!");
   }
   else
   {
      inf1 = inf2 = nc = -1;
   }

   if (el2 >= 0)
   {
      ftr = mesh->GetFaceElementTransformations(face);
      save2 = true;
   }
#ifdef MFEM_USE_MPI
   else if (ParallelC())
   {
      ParMesh *pmesh = c_pfes->GetParMesh();
      if (pmesh->FaceIsTrueInterior(face))
      {
         ftr = pmesh->GetSharedFaceTransformationsByLocalIndex(face);
      }
   }
#endif

   if (save2)
   {
      fes_p.GetElementVDofs(ftr->Elem2No, vdofs2);
      fe2 = fes_p.GetFE(ftr->Elem2No);
      ndof2 = vdofs2.Size();
      c_bfi_p->AssembleHDGFaceMatrix(*tr_fe, *fe1, *fe2, *ftr, elmat);
   }
   else
   {
      if (!ftr) { ftr = mesh->GetFaceElementTransformations(face); }
      vdofs2.SetSize(0);
      ndof2 = 0;
      c_bfi_p->AssembleHDGFaceMatrix(0, *tr_fe, *fe1, *ftr, elmat);
   }

   MFEM_ASSERT(elmat.Width() == ndof1+ndof2+c_dof &&
               elmat.Height() == ndof1+ndof2+c_dof,
               "Size mismatch");

   // assemble D element matrices
   elmat1.CopyMN(elmat, ndof1, ndof1, 0, 0);
   AssemblePotMassMatrix(ftr->Elem1No, elmat1);
   if (save2)
   {
      elmat2.CopyMN(elmat, ndof2, ndof2, ndof1, ndof1);
      AssemblePotMassMatrix(ftr->Elem2No, elmat2);
   }

   // assemble E and G constraints
   if (nc >= 0 && face >= num_faces)
   {
      DenseMatrix E_f_1, G_f_1;
      E_f_1.CopyMN(elmat, ndof1, c_dof, 0, ndof1+ndof2);
      G_f_1.CopyMN(elmat, c_dof, ndof1, ndof1+ndof2, 0);
      AssembleNCSlaveEGFaceMatrix(face, E_f_1, G_f_1);
   }
   else
   {
      DenseMatrix E_f_1(&E_data[E_offsets[face]], ndof1, c_dof);
      DenseMatrix G_f_1(&G_data[G_offsets[face]], c_dof, ndof1);
      E_f_1.CopyMN(elmat, ndof1, c_dof, 0, ndof1+ndof2);
      G_f_1.CopyMN(elmat, c_dof, ndof1, ndof1+ndof2, 0);
   }
   if (save2)
   {
      if (nc >= 0)
      {
         // interior slave face
         DenseMatrix E_f_2, G_f_2;
         E_f_2.CopyMN(elmat, ndof2, c_dof, ndof1, ndof1+ndof2);
         G_f_2.CopyMN(elmat, c_dof, ndof2, ndof1+ndof2, ndof1);
         AssembleNCSlaveEGFaceMatrix(face, E_f_2, G_f_2);
      }
      else
      {
         DenseMatrix E_f_2(&E_data[E_offsets[face] + c_dof*ndof1], ndof2, c_dof);
         DenseMatrix G_f_2(&G_data[G_offsets[face] + c_dof*ndof1], c_dof, ndof2);
         E_f_2.CopyMN(elmat, ndof2, c_dof, ndof1, ndof1+ndof2);
         G_f_2.CopyMN(elmat, c_dof, ndof2, ndof1+ndof2, ndof1);
      }
   }

   // assemble H matrix
   if (IsNonlinear())
   {
      if (face < num_faces)
      {
         DenseMatrix H_f(&H_data[H_offsets[face]], c_dof, c_dof);
         H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
      }
      else
      {
         DenseMatrix H_f;
         H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
         AssembleNCSlaveHFaceMatrix(face, H_f);
      }
   }
   else if (face < num_faces)
   {
      Array<int> c_dofs;
      c_fes.GetFaceVDofs(face, c_dofs);

      if (!H) { H.reset(new SparseMatrix(c_fes.GetVSize())); }
      DenseMatrix H_f;
      H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
      H->AddSubMatrix(c_dofs, c_dofs, H_f, skip_zeros);
   }
   else
   {
      int face_master;
      DenseMatrix H_f, ItHI_f;
      H_f.CopyMN(elmat, c_dof, c_dof, ndof1+ndof2, ndof1+ndof2);
      face_getter fx([this, &ItHI_f, &face_master](int f, DenseMatrix &m)
      {
         const int c_dof = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
         ItHI_f.SetSize(c_dof);
         m.Reset(ItHI_f.GetData(), c_dof, c_dof);
         face_master = f;
      });
      AssembleNCSlaveFaceMatrix(face, face_getter(), NULL, face_getter(), NULL,
                                fx, &H_f);

      Array<int> c_dofs;
      c_fes.GetFaceVDofs(face_master, c_dofs);
      H->AddSubMatrix(c_dofs, c_dofs, ItHI_f, skip_zeros);
   }
}

void DarcyHybridization::ComputeAndAssemblePotBdrFaceMatrix(
   int bface, DenseMatrix &elmat1, Array<int> &vdofs, int skip_zeros)
{
   Mesh *mesh = fes_p.GetMesh();
   const FiniteElement *tr_fe, *fe;
   DenseMatrix elmat, elmat_aux, h_elmat;
   Array<int> c_dofs;

   const int face = mesh->GetBdrElementFaceIndex(bface);
   tr_fe = c_fes.GetFaceElement(face);
   c_fes.GetFaceVDofs(face, c_dofs);
   const int c_dof = c_dofs.Size();

   FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(face);
   fes_p.GetElementVDofs(ftr->Elem1No, vdofs);
   fe = fes_p.GetFE(ftr->Elem1No);
   const int ndof = fe->GetDof() * fes_p.GetVDim();

   MFEM_ASSERT(boundary_constraint_pot_integs.size() > 0,
               "No boundary constraint integrators");

   const int bdr_attr = mesh->GetBdrAttribute(bface);
   for (size_t i = 0; i < boundary_constraint_pot_integs.size(); i++)
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
   DenseMatrix E_f_1(&E_data[E_offsets[face]], ndof, c_dof);
   E_f_1.CopyMN(elmat, ndof, c_dof, 0, ndof);

   // assemble G constraint
   DenseMatrix G_f(&G_data[G_offsets[face]], c_dof, ndof);
   G_f.CopyMN(elmat, c_dof, ndof, ndof, 0);

   // assemble H matrix
   if (IsNonlinear())
   {
      DenseMatrix H_f(&H_data[H_offsets[face]], c_dof, c_dof);
      H_f.CopyMN(elmat, c_dof, c_dof, ndof, ndof);
   }
   else
   {
      if (!H) { H.reset(new SparseMatrix(c_fes.GetVSize())); }
      h_elmat.CopyMN(elmat, c_dof, c_dof, ndof, ndof);
      H->AddSubMatrix(c_dofs, c_dofs, h_elmat, skip_zeros);
   }
}

void DarcyHybridization::GetFDofs(int el, Array<int> &fdofs) const
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   Array<int> vdofs;
   fes.GetElementVDofs(el, vdofs);
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
   fes.GetElementVDofs(el, vdofs);
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

FaceElementTransformations *DarcyHybridization::GetFaceTransformation(
   int f) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   FaceElementTransformations *FTr;
   if (el2 >= 0)
   {
      FTr = fes.GetMesh()->GetFaceElementTransformations(f);
   }
#ifdef MFEM_USE_MPI
   else if (ParallelC() && c_pfes->GetParMesh()->FaceIsTrueInterior(f))
   {
      FTr = c_pfes->GetParMesh()->GetSharedFaceTransformationsByLocalIndex(f);
   }
#endif
   else
   {
      FTr = fes.GetMesh()->GetFaceElementTransformations(f, 21);
   }

   return FTr;
}

void DarcyHybridization::AssembleCtFaceMatrix(int face,
                                              const DenseMatrix &elmat)
{
   const Mesh *mesh = fes.GetMesh();
   const int num_faces = mesh->GetNumFaces();
   int el1, el2;
   mesh->GetFaceElements(face, &el1, &el2);

   const int hat_size_1 = hat_offsets[el1+1] - hat_offsets[el1];
   const int f_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
   const int c_size = c_fes.GetFaceElement(face)->GetDof() * c_fes.GetVDim();

   int inf1, inf2, nc;
   if (mesh->Nonconforming())
   {
      mesh->GetFaceInfos(face, &inf1, &inf2, &nc);
      MFEM_ASSERT(nc < 0 || el2 >= 0 ||
                  inf2 >= 0, "Master face should not be integrated directly!");
   }
   else
   {
      inf1 = inf2 = nc = -1;
   }

   //el1
   if (nc >= 0 && face >= num_faces)
   {
      // ghost slave
      DenseMatrix Ct_face(f_size_1, c_size);
      AssembleCtSubMatrix(el1, elmat, Ct_face);
      AssembleNCSlaveCtFaceMatrix(face, Ct_face);
      return;
   }
   DenseMatrix Ct_face_1(&Ct_data[Ct_offsets[face]], f_size_1, c_size);
   AssembleCtSubMatrix(el1, elmat, Ct_face_1);

   //el2
   if (el2 >= 0)
   {
      //const int hat_size_2 = hat_offsets[el2+1] - hat_offsets[el2];
      const int f_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];

      if (nc >= 0)
      {
         //interior slave
         DenseMatrix Ct_face(f_size_2, c_size);
         AssembleCtSubMatrix(el2, elmat, Ct_face, hat_size_1);
         AssembleNCSlaveCtFaceMatrix(face, Ct_face);
         return;
      }
      DenseMatrix Ct_face_2(&Ct_data[Ct_offsets[face] + f_size_1*c_size],
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

void DarcyHybridization::AssembleNCSlaveFaceMatrix(int f,
                                                   face_getter fx_Ct, const DenseMatrix *Ct,
                                                   face_getter fx_C, const DenseMatrix *C,
                                                   face_getter fx_H, const DenseMatrix *H)
{
   const Mesh *mesh = fes.GetMesh();
#ifdef MFEM_DEBUG
   int el1, el2, inf1, inf2, nc;
   mesh->GetFaceElements(f, &el1, &el2);
   mesh->GetFaceInfos(f, &inf1, &inf2, &nc);
   MFEM_ASSERT(nc >= 0 && (el2 >= 0 || inf2 >= 0), "Not a slave face");
#endif

   const int dim = mesh->Dimension();
   const int num_faces = mesh->GetNumFaces();
   auto &nclist = mesh->ncmesh->GetNCList(dim-1);
   const FiniteElementCollection *c_fec = c_fes.FEColl();

   auto find = nclist.GetMeshIdAndType(f);
   MFEM_ASSERT(find.type == NCMesh::NCList::MeshIdType::SLAVE, "Not a slave face");
   const NCMesh::Slave &slave = static_cast<const NCMesh::Slave&>(*find.id);

   if (slave.master >= num_faces) { return; }

#ifdef MFEM_DEBUG
   mesh->GetFaceElements(slave.master, &el1, &el2);
   mesh->GetFaceInfos(slave.master, &inf1, &inf2, &nc);
   MFEM_ASSERT(nc >= 0 && el2 < 0, "Not a master face");
#endif
   Geometry::Type geom_m = mesh->GetFaceGeometry(slave.master);

   IsoparametricTransformation T;
   DenseMatrix Ct_m, C_m, H_m, I, Io;
   Array<int> edges_m, edges_s, oris_m, oris_s;
   const int *ord_m, *ord_s;

   if (dim == 2)
   {
      // get edge/face ordering
      mesh->GetFaceEdges(slave.master, edges_m, oris_m);
      ord_m = c_fec->DofOrderForOrientation(geom_m, oris_m[0]);
   }

   // compound the master matrix from the slave ones
   if (fx_Ct)
   {
      fx_Ct(slave.master, Ct_m);
   }
   if (fx_C)
   {
      fx_C(slave.master, C_m);
   }
   if (fx_H)
   {
      fx_H(slave.master, H_m);
   }

   const FiniteElement *fe_m = c_fes.GetFaceElement(slave.master);
   switch (geom_m)
   {
      case Geometry::SQUARE:   T.SetFE(&QuadrilateralFE); break;
      case Geometry::TRIANGLE: T.SetFE(&TriangleFE); break;
      case Geometry::SEGMENT:  T.SetFE(&SegmentFE); break;
      default: MFEM_ABORT("unsupported geometry");
   }

   nclist.OrientedPointMatrix(slave, T.GetPointMat());
   const FiniteElement *fe_s = c_fes.GetFaceElement(slave.index);
   fe_s->GetTransferMatrix(*fe_m, T, I);

   if (dim == 2)
   {
      //get edge/face ordering
      if (slave.index < num_faces)
      {
         mesh->GetFaceEdges(slave.index, edges_s, oris_s);
      }
      else
      {
         int verts[4], edges[4], oris[4];
         mesh->ncmesh->GetFaceVerticesEdges(slave, verts, edges, oris);
         oris_s.SetSize(1);
         oris_s[0] = oris[0];
         // check for inverted orientation
         int inf1, inf2;
         mesh->GetFaceInfos(slave.index, &inf1, &inf2);
         if (Mesh::DecodeFaceInfoOrientation(inf2)) { oris_s[0] *= -1; }
      }

      ord_s = c_fec->DofOrderForOrientation(slave.Geom(), oris_s[0]);

      //reorder the interpolation matrix edge->face
      Io.SetSize(I.Height(), I.Width());
      for (int j = 0; j < I.Width(); j++)
         for (int i = 0; i < I.Height(); i++)
         {
            Io(ord_s[i], ord_m[j]) = I(i,j);
         }
   }
   else
   {
      Io.Reset(I.GetData(), I.Height(), I.Width());
   }

   if (fx_Ct)
   {
      mfem::AddMult(*Ct, Io, Ct_m);
   }
   if (fx_C)
   {
      mfem::AddMultAtB(Io, *C, C_m);
   }
   if (fx_H)
   {
      DenseMatrix H_ma(H_m.Height(), H_m.Width());
      RAP(*H, Io, H_ma);
      H_m += H_ma;
   }
}

void DarcyHybridization::AssembleNCSlaveCtFaceMatrix(int f,
                                                     const DenseMatrix &Ct)
{
   AssembleNCSlaveFaceMatrix(f,
   [this](int f, DenseMatrix &m) { GetCtFaceMatrix(f, 0, m); }, &Ct);
}

void DarcyHybridization::AssembleNCSlaveEGFaceMatrix(int f,
                                                     const DenseMatrix &E, const DenseMatrix &G)
{
   AssembleNCSlaveFaceMatrix(f,
   [this](int f, DenseMatrix &m) { GetEFaceMatrix(f, 0, m); }, &E,
   [this](int f, DenseMatrix &m) { GetGFaceMatrix(f, 0, m); }, &G);
}

void DarcyHybridization::AssembleNCSlaveHFaceMatrix(int f, const DenseMatrix &H)
{
   AssembleNCSlaveFaceMatrix(f,
                             face_getter(), NULL,
                             face_getter(), NULL,
   [this](int f, DenseMatrix &m) { GetHFaceMatrix(f, m); }, &H);
}

void DarcyHybridization::ConstructC()
{
#ifndef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   Hybridization::ConstructC();
   return;
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY

   Mesh *mesh = fes.GetMesh();
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
      int el1, el2, inf1, inf2, nc = -1;
      mesh->GetFaceElements(f, &el1, &el2);
      if (mesh->Nonconforming())
      {
         mesh->GetFaceInfos(f, &inf1, &inf2, &nc);
      }

      int f_size = Af_f_offsets[el1+1] - Af_f_offsets[el1];
      if (el2 >= 0 && nc < 0)
      {
         f_size += Af_f_offsets[el2+1] - Af_f_offsets[el2];
      }
      const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
      Ct_offsets[f+1] = Ct_offsets[f] + c_size * f_size;
   }

   Ct_data.SetSize(Ct_offsets[num_faces]); Ct_data = 0.;

   // Assemble the constraint element matrices
   if (c_bfi)
   {
      DenseMatrix elmat;

      for (int f = 0; f < num_faces; f++)
      {
         FaceElementTransformations *FTr = mesh->GetInteriorFaceTransformations(f);
         if (!FTr) { continue; }

         const FiniteElement *fe1 = fes.GetFE(FTr->Elem1No);
         const FiniteElement *fe2 = fes.GetFE(FTr->Elem2No);

         c_bfi->AssembleFaceMatrix(*c_fes.GetFaceElement(f),
                                   *fe1, *fe2, *FTr, elmat);
         // zero-out small elements in elmat
         elmat.Threshold(mtol * elmat.MaxMaxNorm());

         // assemble the matrix
         AssembleCtFaceMatrix(f, elmat);
      }

#ifdef MFEM_USE_MPI
      if (ParallelU()) { pfes->ExchangeFaceNbrData(); }
      ParMesh *pmesh = NULL;
      if (pfes) { pmesh = pfes->GetParMesh(); }
      else if (c_pfes) { pmesh = c_pfes->GetParMesh(); }
      const int NE = mesh->GetNE();

      if (pmesh)
      {
         const int num_shared_faces = pmesh->GetNSharedFaces();
         for (int sf = 0; sf < num_shared_faces; sf++)
         {
            const int f = pmesh->GetSharedFace(sf);
            FaceElementTransformations *FTr = pmesh->GetSharedFaceTransformations(sf);

            const FiniteElement *fe1 = fes.GetFE(FTr->Elem1No);
            const FiniteElement *fe2 =
               (pfes)?(pfes->GetFaceNbrFE(FTr->Elem2No - NE)):(fe1);

            c_bfi->AssembleFaceMatrix(*c_fes.GetFaceElement(f),
                                      *fe1, *fe2, *FTr, elmat);
            // zero-out small elements in elmat
            elmat.Threshold(mtol * elmat.MaxMaxNorm());

            // assemble the matrix
            AssembleCtFaceMatrix(f, elmat);
         }
      }
#endif

      if (boundary_constraint_integs.size())
      {
         const FiniteElement *fe1, *fe2;
         const FiniteElement *face_el;

         // Which boundary attributes need to be processed?
         Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
         bdr_attr_marker = 0;
         for (size_t k = 0; k < boundary_constraint_integs.size(); k++)
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

         for (int i = 0; i < fes.GetNBE(); i++)
         {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            FaceElementTransformations *FTr = mesh->GetBdrFaceTransformations(i);
            if (!FTr) { continue; }

            int iface = mesh->GetBdrElementFaceIndex(i);
            face_el = c_fes.GetFaceElement(iface);
            fe1 = fes.GetFE (FTr -> Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (size_t k = 0; k < boundary_constraint_integs.size(); k++)
            {
               if (boundary_constraint_integs_marker[k] &&
                   (*boundary_constraint_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               boundary_constraint_integs[k]->AssembleFaceMatrix(*face_el, *fe1, *fe2, *FTr,
                                                                 elmat);
               // zero-out small elements in elmat
               elmat.Threshold(mtol * elmat.MaxMaxNorm());

               // assemble the matrix
               AssembleCtFaceMatrix(iface, elmat);
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
   Df_data.SetSize(Df_offsets.Last()); Df_data = 0.;
   Df_ipiv.SetSize(Df_f_offsets.Last());
}

void DarcyHybridization::AllocEG() const
{
   Mesh *mesh = fes.GetMesh();
   const int num_faces = mesh->GetNumFaces();

   // Define E_offsets and allocate E_data and G_data
   E_offsets.SetSize(num_faces+1);
   E_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      int el1, el2, inf1, inf2, nc = -1;
      mesh->GetFaceElements(f, &el1, &el2);
      if (mesh->Nonconforming())
      {
         mesh->GetFaceInfos(f, &inf1, &inf2, &nc);
      }

      int d_size = Df_f_offsets[el1+1] - Df_f_offsets[el1];
      if (el2 >= 0 && nc < 0)
      {
         d_size += Df_f_offsets[el2+1] - Df_f_offsets[el2];
      }
      const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
      E_offsets[f+1] = E_offsets[f] + c_size * d_size;
   }

   E_data.SetSize(E_offsets.Last()); E_data = 0.;
   G_data.SetSize(G_offsets.Last()); G_data = 0.;
}

void DarcyHybridization::AllocH() const
{
   Mesh *mesh = fes.GetMesh();
   int num_faces = mesh->GetNumFaces();

   // Define E_offsets and allocate E_data and G_data
   H_offsets.SetSize(num_faces+1);
   H_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
      H_offsets[f+1] = H_offsets[f] + c_size * c_size;
   }

   H_data.SetSize(H_offsets[num_faces]); H_data = 0.;
}

void DarcyHybridization::InvertA()
{
   const int NE = fes.GetNE();

   for (int el = 0; el < NE; el++)
   {
      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];

      // Decompose A

      LUFactors LU_A(&Af_data[Af_offsets[el]], Af_ipiv + Af_f_offsets[el]);

      LU_A.Factor(a_dofs_size);
   }
}

void DarcyHybridization::InvertD()
{
   const int NE = fes.GetNE();

   for (int el = 0; el < NE; el++)
   {
      int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

      // Decompose D

#ifdef MFEM_DEBUG
      DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
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

      LUFactors LU_D(&Df_data[Df_offsets[el]], &Df_ipiv[Df_f_offsets[el]]);

      LU_D.Factor(d_dofs_size);
   }
}

void DarcyHybridization::GetElementFaces(int el, Array<int> &faces) const
{
   const Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();

#ifdef MFEM_THREAD_SAFE
   Array<int> oris;
#else
   static Array<int> oris;
#endif

   switch (dim)
   {
      case 1:
         mesh->GetElementVertices(el, faces);
         break;
      case 2:
         mesh->GetElementEdges(el, faces, oris);
         break;
      case 3:
         mesh->GetElementFaces(el, faces, oris);
         break;
   }
}

void DarcyHybridization::ComputeH(ComputeHMode mode,
                                  std::unique_ptr<SparseMatrix> &H) const
{
   MFEM_ASSERT(mode != ComputeHMode::Linear || !IsNonlinear(),
               "Cannot assemble H matrix in the non-linear regime");

   const int skip_zeros = 1;
   const int NE = fes.GetNE();
   DenseMatrix S;
   Array<int> S_ipiv;
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   DenseMatrix AiBt, AiCt, BAiCt, CAiBt, H_l;
   Array<int> c_dofs_1, c_dofs_2;
   Array<int> faces;
   if (!H) { H.reset(new SparseMatrix(c_fes.GetVSize())); }
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
      LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);
      if (mode == ComputeHMode::Linear || lop_type != LocalOpType::PotNL)
      {
         LU_A.Factor(a_dofs_size);
      }

      // Construct Schur complement
      const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                          d_dofs_size, a_dofs_size);
      DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
      AiBt.SetSize(a_dofs_size, d_dofs_size);

      AiBt.Transpose(B);
      if (!bsym) { AiBt.Neg(); }
      LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());

      LUFactors LU_S;
      if (mode == ComputeHMode::Linear || lop_type != LocalOpType::FluxNL)
      {
         mfem::AddMult(B, AiBt, D);

         // Decompose Schur complement
         LU_S.data = D.GetData();
         LU_S.ipiv = &Df_ipiv[Df_f_offsets[el]];
         LU_S.Factor(d_dofs_size);
      }
      else
      {
         const DenseMatrix D_lin(&Df_lin_data[Df_offsets[el]],
                                 d_dofs_size, d_dofs_size);
         S = D_lin;
         mfem::AddMult(B, AiBt, S);

         // Decompose Schur complement
         LU_S.data = S.GetData();
         S_ipiv.SetSize(d_dofs_size);
         LU_S.ipiv = S_ipiv.GetData();
         LU_S.Factor(d_dofs_size);
      }

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      GetElementFaces(el, faces);

      // Mult C^T
      for (int f1 = 0; f1 < faces.Size(); f1++)
      {
         c_fes.GetFaceVDofs(faces[f1], c_dofs_1);

         int el1_1, el1_2;
         fes.GetMesh()->GetFaceElements(faces[f1], &el1_1, &el1_2);
         DenseMatrix Ct1;
         GetCtFaceMatrix(faces[f1], el1_1 != el, Ct1);

         //A^-1 C^T
         AiCt.SetSize(Ct1.Height(), Ct1.Width());
         AiCt = Ct1;
         LU_A.Solve(Ct1.Height(), Ct1.Width(), AiCt.GetData());

         //S^-1 (B A^-1 C^T - E)
         BAiCt.SetSize(B.Height(), Ct1.Width());
         mfem::Mult(B, AiCt, BAiCt);

         if (c_bfi_p || mode == ComputeHMode::Gradient)
         {
            DenseMatrix E;
            GetEFaceMatrix(faces[f1], el1_1 != el, E);

            BAiCt -= E;
         }

         LU_S.Solve(BAiCt.Height(), BAiCt.Width(), BAiCt.GetData());

         for (int f2 = 0; f2 < faces.Size(); f2++)
         {
            int el2_1, el2_2;
            fes.GetMesh()->GetFaceElements(faces[f2], &el2_1, &el2_2);
            DenseMatrix Ct2;
            GetCtFaceMatrix(faces[f2], el2_1 != el, Ct2);

            //- C A^-1 C^T
            H_l.SetSize(Ct2.Width(), Ct1.Width());
            mfem::MultAtB(Ct2, AiCt, H_l);
            H_l.Neg();

            //(C A^-1 B^T + G) S^-1 (B A^-1 C^T - E)
            CAiBt.SetSize(Ct2.Width(), B.Height());
            mfem::MultAtB(Ct2, AiBt, CAiBt);

            if (c_bfi_p || mode == ComputeHMode::Gradient)
            {
               DenseMatrix G;
               GetGFaceMatrix(faces[f2], el2_1 != el, G);

               CAiBt += G;
            }

            mfem::AddMult(CAiBt, BAiCt, H_l);

            if (f1 == f2)
            {
               //integrate the face contrbution only on one (first) side
               if (mode == ComputeHMode::Gradient && el2_1 == el)
               {
                  DenseMatrix H_f;
                  GetHFaceMatrix(faces[f1], H_f);
                  H_l += H_f;
               }
               H->AddSubMatrix(c_dofs_1, c_dofs_1, H_l, skip_zeros);
            }
            else
            {
               c_fes.GetFaceVDofs(faces[f2], c_dofs_2);
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

   if (diag_policy == DIAG_ONE || diag_policy == DIAG_ZERO)
   {
      // put zeroes on the diagonal
      for (int i = 0; i < H->Height(); i++)
      {
         H->SetColPtr(i);
         H->SearchRow(i);
      }
      H->Finalize(0);
   }
   else
   {
      H->Finalize(skip_zeros);
   }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Hb->Finalize(skip_zeros);
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

   if (!ParallelC())
   {
      const SparseMatrix *cP = c_fes.GetConformingProlongation();
      if (cP)
      {
         if (H->Height() != cP->Width())
         {
            SparseMatrix *cH = mfem::RAP(*cP, *H, *cP);
            H.reset(cH);
         }
      }

      // ensure diagonal is non-zero
      if (diag_policy == DIAG_ONE)
      {
         H->SetDiagIdentity();
      }
   }
}

#ifdef MFEM_USE_MPI
void DarcyHybridization::ComputeParH(ComputeHMode mode,
                                     std::unique_ptr<SparseMatrix> &H, OperatorHandle &pH) const
{
   ComputeH(mode, H);

   if (!ParallelC())
   {
      pH.Reset(H.get(), false);
   }
   else // parallel
   {
      OperatorHandle dH(pH.Type()), pP(pH.Type());
      dH.MakeSquareBlockDiag(c_pfes->GetComm(), c_pfes->GlobalVSize(),
                             c_pfes->GetDofOffsets(), H.get());
      // TODO - construct Dof_TrueDof_Matrix directly in the pS format
      pP.ConvertFrom(c_pfes->Dof_TrueDof_Matrix());
      pH.MakePtAP(dH, pP);
      dH.Clear();
      H.reset();
   }
}
#endif //MFEM_USE_MPI

void DarcyHybridization::GetCtFaceMatrix(
   int f, int side, DenseMatrix &Ct) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
   const int f_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];

   if (side == 0)
   {
      Ct.Reset(const_cast<real_t*>(&Ct_data[Ct_offsets[f]]), f_size_1, c_size);
   }
   else
   {
      MFEM_ASSERT(el2 >= 0, "Invalid element");
      const int f_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];
      Ct.Reset(const_cast<real_t*>(&Ct_data[Ct_offsets[f] + f_size_1*c_size]),
               f_size_2, c_size);
   }
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
   Array<int> c_dofs;
   c_fes.GetFaceVDofs(f, c_dofs);
   if (side == 0)
   {
      GetCtSubMatrix(el1, c_dofs, Ct);
   }
   else
   {
      MFEM_ASSERT(el2 >= 0, "Invalid element");
      GetCtSubMatrix(el2, c_dofs, Ct);
   }
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK_ASSEMBLY
}

void DarcyHybridization::GetEFaceMatrix(
   int f, int side, DenseMatrix &E) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
   const int d_size_1 = Df_f_offsets[el1+1] - Df_f_offsets[el1];

   if (side == 0)
   {
      E.Reset(&E_data[E_offsets[f]], d_size_1, c_size);
   }
   else
   {
      MFEM_ASSERT(el2 >= 0, "Invalid element");
      const int d_size_2 = Df_f_offsets[el2+1] - Df_f_offsets[el2];
      E.Reset(&E_data[E_offsets[f] + d_size_1*c_size], d_size_2, c_size);
   }
}

void DarcyHybridization::GetGFaceMatrix(
   int f, int side, DenseMatrix &G) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
   const int d_size_1 = Df_f_offsets[el1+1] - Df_f_offsets[el1];

   if (side == 0)
   {
      G.Reset(G_data + G_offsets[f], c_size, d_size_1);
   }
   else
   {
      MFEM_ASSERT(el2 >= 0, "Invalid element");
      const int d_size_2 = Df_f_offsets[el2+1] - Df_f_offsets[el2];
      G.Reset(G_data + G_offsets[f] + d_size_1*c_size, c_size, d_size_2);
   }
}

void DarcyHybridization::GetHFaceMatrix(int f, DenseMatrix &H) const
{
   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();

   H.Reset(&H_data[H_offsets[f]], c_size, c_size);
}

void DarcyHybridization::GetCtSubMatrix(int el, const Array<int> &c_dofs,
                                        DenseMatrix &Ct_l) const
{
   const int hat_offset = hat_offsets[el  ];
   const int hat_size = hat_offsets[el+1] - hat_offset;
   const int f_size = Af_f_offsets[el+1] - Af_f_offsets[el];

   Array<int> vdofs;
   fes.GetElementVDofs(el, vdofs);

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

   if (!Df_data.Size()) { AllocD(); }// D is resetted in ConstructGrad()
   if (!E_data.Size() || !G_data.Size()) { AllocEG(); }// E and G are rewritten
   if (!H_data.Size()) { AllocH(); }
   else if (c_nlfi_p || c_nlfi)
   {
      // H is resetted here for additive double side integration
      H_data = 0.;
   }

   Vector y;//dummy
   MultNL(MultNlMode::Grad, darcy_rhs, x, y);

#ifdef MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
   //assemble gradient matrix
   Grad.reset();
   ComputeH(ComputeHMode::Gradient, Grad);
   return *Grad;
#else
   //construct gradient operator
   pGrad.Reset(new Gradient(*this));
   return *pGrad;
#endif
}

void DarcyHybridization::MultNL(MultNlMode mode, const Vector &bu,
                                const Vector &bp, const Vector &x, Vector &y) const
{
   const int NE = fes.GetNE();
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   const int dim = fes.GetMesh()->Dimension();
   DenseMatrix H;
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
      f_2_b = fes.GetMesh()->GetFaceToBdrElMap();
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

         fes_p.GetElementVDofs(el, p_dofs);
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
            fes.GetMesh()->GetElementVertices(el, faces);
            break;
         case 2:
            fes.GetMesh()->GetElementEdges(el, faces, oris);
            break;
         case 3:
            fes.GetMesh()->GetElementFaces(el, faces, oris);
            break;
      }

      c_offsets.SetSize(faces.Size()+1);
      c_offsets[0] = 0;
      for (int f = 0; f < faces.Size(); f++)
      {
         const int c_size = c_fes.GetFaceElement(faces[f])->GetDof() * c_fes.GetVDim();
         c_offsets[f+1] = c_offsets[f] + c_size;
      }

      x_l.Update(c_offsets);
      for (int f = 0; f < faces.Size(); f++)
      {
         c_fes.GetFaceVDofs(faces[f], c_dofs);
         x.GetSubVector(c_dofs, x_l.GetBlock(f));
      }

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      // bu - C^T x
      for (int f = 0; f < faces.Size(); f++)
      {
         int el1, el2;
         fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
         DenseMatrix Ct;
         GetCtFaceMatrix(faces[f], el1 != el, Ct);

         const Vector &x_f = x_l.GetBlock(f);

         Ct.AddMult_a(-1., x_f, bu_l);

         //bp - E x
         if (c_bfi_p || mode == MultNlMode::GradMult)
         {
            DenseMatrix E;
            GetEFaceMatrix(faces[f], el1 != el, E);

            E.AddMult_a(-1., x_f, bp_l);
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
         int el1, el2;
         fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
         DenseMatrix Ct;
         GetCtFaceMatrix(faces[f], el1 != el, Ct);

         const Vector &x_f = x_l.GetBlock(f);

         y_l.SetSize(x_f.Size());
         Ct.MultTranspose(u_l, y_l);

         //G p_l + H x_l
         if (c_bfi_p || mode == MultNlMode::GradMult)
         {
            //linear
            DenseMatrix G;
            GetGFaceMatrix(faces[f], el1 != el, G);

            G.AddMult(p_l, y_l);

            //integrate the face contrbution only on one (first) side
            if (el1 == el)
            {
               GetHFaceMatrix(faces[f], H);
               H.AddMult(x_f, y_l);
            }
         }
         else
         {
            //nonlinear
            if (c_nlfi_p)
            {
               Vector GpHx_l;
               int type = NonlinearFormIntegrator::HDGFaceType::CONSTR
                          | NonlinearFormIntegrator::HDGFaceType::FACE;

               FaceElementTransformations *FTr = GetFaceTransformation(faces[f]);

               if (FTr->Elem2No >= 0)
               {
                  //interior
                  if (FTr->Elem1No != el) { type |= 1; }

                  c_nlfi_p->AssembleHDGFaceVector(type,
                                                  *c_fes.GetFaceElement(faces[f]),
                                                  *fes_p.GetFE(el),
                                                  *FTr,
                                                  x_f, p_l, GpHx_l);

                  y_l += GpHx_l;
               }
               else
               {
                  //boundary
                  const int bdr_attr = fes.GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

                  for (size_t i = 0; i < boundary_constraint_pot_nonlin_integs.size(); i++)
                  {
                     if (boundary_constraint_pot_nonlin_integs_marker[i]
                         && (*boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

                     boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                                     *c_fes.GetFaceElement(faces[f]),
                                                                                     *fes_p.GetFE(el),
                                                                                     *FTr,
                                                                                     x_f, p_l, GpHx_l);

                     y_l += GpHx_l;
                  }
               }
            }

            if (c_nlfi)
            {
               Vector GpHx_l;
               const FiniteElement *fe_u = fes.GetFE(el);
               const FiniteElement *fe_p = fes_p.GetFE(el);
               Array<const FiniteElement*> fe_arr({fe_u, fe_p});
               Array<const Vector*> x_arr({&u_l, &p_l});
               Array<Vector*> y_arr((Vector*[]) {NULL, NULL, &GpHx_l});

               int type = BlockNonlinearFormIntegrator::HDGFaceType::CONSTR
                          | BlockNonlinearFormIntegrator::HDGFaceType::FACE;

               FaceElementTransformations *FTr = GetFaceTransformation(faces[f]);

               if (FTr->Elem2No >= 0)
               {
                  //interior
                  if (FTr->Elem1No != el) { type |= 1; }

                  c_nlfi->AssembleHDGFaceVector(type,
                                                *c_fes.GetFaceElement(faces[f]),
                                                fe_arr,
                                                *FTr,
                                                x_f, x_arr, y_arr);

                  if (GpHx_l.Size() > 0) { y_l += GpHx_l; }
               }
               else
               {
                  //boundary
                  const int bdr_attr = fes.GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

                  for (size_t i = 0; i < boundary_constraint_nonlin_integs.size(); i++)
                  {
                     if (boundary_constraint_nonlin_integs_marker[i]
                         && (*boundary_constraint_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

                     boundary_constraint_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                                 *c_fes.GetFaceElement(faces[f]),
                                                                                 fe_arr,
                                                                                 *FTr,
                                                                                 x_f, x_arr, y_arr);

                     if (GpHx_l.Size() > 0) { y_l += GpHx_l; }
                  }
               }
            }
         }

         c_fes.GetFaceVDofs(faces[f], c_dofs);
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

void DarcyHybridization::ParMultNL(MultNlMode mode, const BlockVector &b_t,
                                   const Vector &x_t, Vector &y_t) const
{
   Vector x;
   const Operator *tr_cP;

   if (!ParallelC())
   {
      tr_cP = c_fes.GetConformingProlongation();
      if (!tr_cP)
      {
         x.MakeRef(const_cast<Vector&>(x_t), 0, x_t.Size());
      }
      else
      {
         x.SetSize(c_fes.GetVSize());
         tr_cP->Mult(x_t, x);
      }
   }
   else
   {
      x.SetSize(c_fes.GetVSize());
      c_fes.GetProlongationMatrix()->Mult(x_t, x);
   }

   Vector bu;
   const Operator *cR;

   if (!ParallelU())
   {
      if (!(cR = fes.GetConformingRestriction()))
      {
         bu.MakeRef(const_cast<Vector&>(b_t.GetBlock(0)), 0, fes.GetVSize());
      }
      else
      {
         bu.SetSize(fes.GetVSize());
         cR->MultTranspose(b_t.GetBlock(0), bu);
      }
   }
   else
   {
      bu.SetSize(fes.GetVSize());
      fes.GetRestrictionOperator()->MultTranspose(b_t.GetBlock(0), bu);
   }

   const Vector &bp = b_t.GetBlock(1);
   Vector y;

   if (mode == MultNlMode::Sol)
   {
      if (!ParallelU() && !cR)
      {
         y.MakeRef(y_t, 0, darcy_offsets.Last());
      }
      else
      {
         y.SetSize(darcy_offsets.Last());
      }
   }
   else
   {
      const Operator *tr_cR;
      if (!ParallelC() && !(tr_cR = c_fes.GetRestrictionOperator()))
      {
         y.MakeRef(y_t, 0, c_fes.GetVSize());
      }
      else
      {
         y.SetSize(c_fes.GetVSize());
      }
   }

   MultNL(mode, bu, bp, x, y);

   if (mode == MultNlMode::Sol)
   {
      if (ParallelU() || cR)
      {
         BlockVector yb(y, darcy_offsets);
         BlockVector yb_t(y_t, darcy_toffsets);

         if (!ParallelU())
         {
            cR->Mult(yb.GetBlock(0), yb_t.GetBlock(0));
         }
         else
         {
            fes.GetRestrictionOperator()->Mult(yb.GetBlock(0), yb_t.GetBlock(0));
         }

         yb_t.GetBlock(1) = yb.GetBlock(1);
      }
   }
   else if (mode != MultNlMode::Grad)
   {
      if (!ParallelC())
      {
         if (tr_cP)
         {
            tr_cP->MultTranspose(y, y_t);
         }
      }
      else
      {
         c_fes.GetProlongationMatrix()->MultTranspose(y, y_t);
      }
   }
}

void DarcyHybridization::Finalize()
{
   if (bfin) { return; }

   if (!IsNonlinear())
   {
#ifndef MFEM_USE_MPI
      ComputeH(ComputeHMode::Linear, H);
#else //MFEM_USE_MPI      
      ComputeParH(ComputeHMode::Linear, H, pH);
      pOp = pH;
#endif //MFEM_USE_MPI
   }
   else
   {
      if (!m_nlfi_u && !m_nlfi && !c_nlfi)
      {
         lop_type = LocalOpType::PotNL;
         // backup the data for gradient construction
         Af_lin_data = Af_data;
         InvertA();
      }
      else if (!m_nlfi_p && !c_nlfi_p && !D_empty && !m_nlfi && !c_nlfi)
      {
         lop_type = LocalOpType::FluxNL;
         // backup the data for gradient construction
         Df_lin_data = Df_data;
         InvertD();
      }
      else
      {
         lop_type = LocalOpType::FullNL;
         if (!A_empty)
         {
            Swap(Af_data, Af_lin_data);
            Af_data.SetSize(Af_offsets.Last());
         }

         if (!D_empty)
         {
            Swap(Df_data, Df_lin_data);
            if (!Df_data.Size())
            {
               Df_data.SetSize(Df_offsets.Last());
               Df_data = 0.;
            }
         }
      }

#ifdef MFEM_USE_MPI
      pOp.Reset(new ParOperator(*this));
#endif //MFEM_USE_MPI
   }

   bfin = true;
}

void DarcyHybridization::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                             const BlockVector &x, BlockVector &b)
{
   if (IsNonlinear())
   {
      MFEM_ASSERT(!ParallelU() && !ParallelP(),
                  "In parallel, use ParallelEliminateTDofsInRHS() instead!");

      //save the rhs for initial guess in the iterative local solve
      darcy_u = x.GetBlock(0);
      darcy_p = x.GetBlock(1);
   }

   MFEM_ASSERT(x.Size() == fes.GetVSize() + fes_p.GetVSize(),
               "Wrong size of the solution vector!");
   MFEM_ASSERT(b.Size() == fes.GetVSize() + fes_p.GetVSize(),
               "Wrong size of the rhs vector!");

   const int NE = fes.GetNE();
   Vector u_e, bu_e, bp_e;
   Array<int> u_vdofs, p_dofs, edofs;

   const Vector &xu = x.GetBlock(0);
   Vector &bu = b.GetBlock(0);
   Vector &bp = b.GetBlock(1);

   for (int el = 0; el < NE; el++)
   {
      GetEDofs(el, edofs);
      if (edofs.Size() == 0) { continue; }

      xu.GetSubVector(edofs, u_e);
      u_e.Neg();

      //bu -= A_e u_e
      const int a_size = hat_offsets[el+1] - hat_offsets[el];
      const DenseMatrix Ae(&Ae_data[Ae_offsets[el]], a_size, edofs.Size());

      bu_e.SetSize(a_size);
      Ae.Mult(u_e, bu_e);

      fes.GetElementVDofs(el, u_vdofs);
      bu.AddElementVector(u_vdofs, bu_e);

      //bp -= B_e u_e
      const int d_size = Df_f_offsets[el+1] - Df_f_offsets[el];
      const DenseMatrix Be(&Be_data[Be_offsets[el]], d_size, edofs.Size());

      bp_e.SetSize(d_size);
      Be.Mult(u_e, bp_e);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         bp_e.Neg();
      }

      fes_p.GetElementVDofs(el, p_dofs);
      bp.AddElementVector(p_dofs, bp_e);
   }

   for (int vdof : vdofs_flux)
   {
      bu(vdof) = xu(vdof);//<--can be arbitrary as it is ignored
   }
}

void DarcyHybridization::EliminateTrueDofsInRHS(
   const Array<int> &tdofs_flux, const BlockVector &x_t, BlockVector &b_t)
{
   Vector xu, bu;

   if (!ParallelU())
   {
      const Operator *cP = fes.GetConformingProlongation();
      if (!cP)
      {
         xu.MakeRef(const_cast<Vector&>(x_t.GetBlock(0)), 0, fes.GetVSize());
      }
      else
      {
         xu.SetSize(cP->Height());
         cP->Mult(x_t.GetBlock(0), xu);
      }

      const Operator *cR = fes.GetConformingRestriction();
      if (!cR)
      {
         bu.MakeRef(b_t.GetBlock(0), 0, fes.GetVSize());
      }
      else
      {
         bu.SetSize(cR->Width());
         cR->MultTranspose(b_t.GetBlock(0), bu);
      }
   }
   else
   {
      xu.SetSize(fes.GetVSize());
      fes.GetProlongationMatrix()->Mult(x_t.GetBlock(0), xu);
      bu.SetSize(xu.Size());
      fes.GetRestrictionOperator()->MultTranspose(b_t.GetBlock(0), bu);
   }

   if (IsNonlinear())
   {
      //save the rhs for initial guess in the iterative local solve
      darcy_u = xu;
      darcy_p = x_t.GetBlock(1);
   }

   Vector &bp = b_t.GetBlock(1);

   const int NE = fes.GetNE();
   Vector u_e, bu_e, bp_e;
   Array<int> u_vdofs, p_dofs, edofs;

   for (int el = 0; el < NE; el++)
   {
      GetEDofs(el, edofs);
      if (edofs.Size() == 0) { continue; }

      xu.GetSubVector(edofs, u_e);
      u_e.Neg();

      //bu -= A_e u_e
      const int a_size = hat_offsets[el+1] - hat_offsets[el];
      const DenseMatrix Ae(&Ae_data[Ae_offsets[el]], a_size, edofs.Size());

      bu_e.SetSize(a_size);
      Ae.Mult(u_e, bu_e);

      fes.GetElementVDofs(el, u_vdofs);
      bu.AddElementVector(u_vdofs, bu_e);

      //bp -= B_e u_e
      const int d_size = Df_f_offsets[el+1] - Df_f_offsets[el];
      const DenseMatrix Be(&Be_data[Be_offsets[el]], d_size, edofs.Size());

      bp_e.SetSize(d_size);
      Be.Mult(u_e, bp_e);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         bp_e.Neg();
      }

      fes_p.GetElementVDofs(el, p_dofs);
      bp.AddElementVector(p_dofs, bp_e);
   }

   if (!ParallelU())
   {
      const Operator *cP = fes.GetConformingProlongation();
      if (cP)
      {
         cP->MultTranspose(bu, b_t.GetBlock(0));
      }
   }
   else
   {
      fes.GetProlongationMatrix()->MultTranspose(bu, b_t.GetBlock(0));
   }

   for (int tdof : tdofs_flux)
   {
      b_t(tdof) = x_t(tdof);//<--can be arbitrary as it is ignored
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
   const int dim = fes.GetMesh()->Dimension();
   switch (dim)
   {
      case 1:
         fes.GetMesh()->GetElementVertices(el, faces);
         break;
      case 2:
         fes.GetMesh()->GetElementEdges(el, faces, oris);
         break;
      case 3:
         fes.GetMesh()->GetElementFaces(el, faces, oris);
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

   Solver *prec = NULL;
   if (use_prec)
   {
      IterativeSolver *iter_prec = NULL;
      switch (lsolve.prec.type)
      {
         case LPrecType::GMRES:
            prec = iter_prec = new GMRESSolver();
            break;
         case LPrecType::LU:
            prec = new DenseMatrixLUSolver();
            break;
         default:
            MFEM_ABORT("Unknown local preconditioner");
      }

      if (iter_prec)
      {
         iter_prec->SetMaxIter(lsolve.prec.iters);
         iter_prec->SetRelTol((lsolve.prec.rtol >= 0)?
                              (lsolve.prec.rtol):(lsolve.rtol));
         iter_prec->SetAbsTol((lsolve.prec.atol >= 0)?
                              (lsolve.prec.atol):(lsolve.atol));
      }
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
         mfem::out << "el: " << el
                   << " iters: " << lsolver->GetNumIterations()
                   << " rel. norm: " << lsolver->GetFinalRelNorm()
                   << std::endl;
   }
   else
   {
      mfem::out << "el: " << el
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

   LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);
   LUFactors LU_S(&Df_data[Df_offsets[el]], &Df_ipiv[Df_f_offsets[el]]);

   // Load B

   const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                       d_dofs_size, a_dofs_size);

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
   const FiniteElement *fe_u = fes.GetFE(el);
   const FiniteElement *fe_p = fes_p.GetFE(el);
   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   ElementTransformation *Tr = fes.GetElementTransformation(el);

   DenseMatrix A(&Af_data[Af_offsets[el]], a_dofs_size, a_dofs_size);
   DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
   LUFactors LU_A(A.GetData(), &Af_ipiv[Af_f_offsets[el]]);

   if (m_nlfi)
   {
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array2D<DenseMatrix*> grad_arr(2,2);
      DenseMatrix grad_A, grad_D;
      grad_arr(0,0) = &grad_A;
      grad_arr(1,0) = NULL;
      grad_arr(0,1) = NULL;
      grad_arr(1,1) = &grad_D;
      m_nlfi->AssembleElementGrad(fe_arr, *Tr, x_arr, grad_arr);
      if (grad_A.Height() != 0) { A = grad_A; }
      else { A = 0.; }
      if (grad_D.Height() != 0) { D = grad_D; }
      else { D = 0.; }
   }
   else
   {
      // if only linear data are present, A is already factored
      if (lop_type != LocalOpType::PotNL)
      {
         A = 0.;
      }
      // if only linear data are present, D is already factored
      if (lop_type != LocalOpType::FluxNL)
      {
         D = 0.;
      }
   }

   if (m_nlfi_u)
   {
      DenseMatrix grad_A;
      m_nlfi_u->AssembleElementGrad(*fe_u, *Tr, u_l, grad_A);
      A += grad_A;
   }
   else if (!A_empty && lop_type != LocalOpType::PotNL)
   {
      DenseMatrix A_lin(const_cast<real_t*>(&Af_lin_data[Af_offsets[el]]),
                        a_dofs_size, a_dofs_size);
      A += A_lin;
   }

   if (m_nlfi_p)
   {
      DenseMatrix grad_D;
      m_nlfi_p->AssembleElementGrad(*fe_p, *Tr, p_l, grad_D);
      D += grad_D;
   }
   else if (!D_empty && lop_type != LocalOpType::FluxNL)
   {
      DenseMatrix D_lin(&Df_lin_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
      D += D_lin;
   }

   if (c_nlfi_p)
   {
      //bp += E x
      for (int f = 0; f < faces.Size(); f++)
      {
         const Vector &x_f = x_l.GetBlock(f);

         FaceElementTransformations *FTr = GetFaceTransformation(faces[f]);

         if (FTr->Elem2No >= 0)
         {
            //interior
            AssembleHDGGrad(el, FTr, *c_nlfi_p, x_f, p_l);
         }
         else
         {
            //boundary
            const int bdr_attr = fes.GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

            for (size_t i = 0; i < boundary_constraint_pot_nonlin_integs.size(); i++)
            {
               if (boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               AssembleHDGGrad(el, FTr, *boundary_constraint_pot_nonlin_integs[i], x_f,
                               p_l);
            }
         }
      }
   }

   if (c_nlfi)
   {
      //bp += E x
      for (int f = 0; f < faces.Size(); f++)
      {
         const Vector &x_f = x_l.GetBlock(f);

         FaceElementTransformations *FTr = GetFaceTransformation(faces[f]);

         if (FTr->Elem2No >= 0)
         {
            //interior
            AssembleHDGGrad(el, FTr, *c_nlfi, x_f, u_l, p_l);
         }
         else
         {
            //boundary
            const int bdr_attr = fes.GetMesh()->GetBdrAttribute(f_2_b[faces[f]]);

            for (size_t i = 0; i < boundary_constraint_nonlin_integs.size(); i++)
            {
               if (boundary_constraint_nonlin_integs_marker[i]
                   && (*boundary_constraint_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               AssembleHDGGrad(el, FTr, *boundary_constraint_nonlin_integs[i], x_f,
                               u_l, p_l);
            }
         }
      }
   }

#ifndef MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
   if (lop_type != LocalOpType::PotNL)
   {
      // Decompose A
      LU_A.Factor(a_dofs_size);
   }

   if (lop_type != LocalOpType::FluxNL)
   {
      // Construct Schur complement
      const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                          d_dofs_size, a_dofs_size);
      DenseMatrix AiBt(a_dofs_size, d_dofs_size);

      AiBt.Transpose(B);
      if (!bsym) { AiBt.Neg(); }
      LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());
      mfem::AddMult(B, AiBt, D);

      // Decompose Schur complement
      LUFactors LU_S(D.GetData(), &Df_ipiv[Df_f_offsets[el]]);

      LU_S.Factor(d_dofs_size);
   }
   else
   {
      MFEM_ABORT("Not implemented");
   }
#endif //MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
}

void DarcyHybridization::AssembleHDGGrad(
   int el, FaceElementTransformations *FTr, NonlinearFormIntegrator &nlfi,
   const Vector &x_f, const Vector &p_l) const
{
   const int f = FTr->Face->ElementNo;
   const FiniteElement *fe_c = c_fes.GetFaceElement(f);
   const FiniteElement *fe_p = fes_p.GetFE(el);
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   const int c_dofs_size = x_f.Size();

   int type = NonlinearFormIntegrator::HDGFaceType::ELEM
              | NonlinearFormIntegrator::HDGFaceType::TRACE
              | NonlinearFormIntegrator::HDGFaceType::CONSTR
              | NonlinearFormIntegrator::HDGFaceType::FACE;

   if (FTr->Elem1No != el) { type |= 1; }

   DenseMatrix elmat;

   nlfi.AssembleHDGFaceGrad(type, *fe_c, *fe_p, *FTr, x_f, p_l, elmat);

   // assemble D element matrices
   DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
   DenseMatrix elmat_D;
   elmat_D.CopyMN(elmat, d_dofs_size, d_dofs_size, 0, 0);
   D += elmat_D;

   // assemble E constraint
   const int E_off = (FTr->Elem1No == el)?(0):(c_dofs_size*d_dofs_size);
   DenseMatrix E_f(&E_data[E_offsets[f] + E_off], d_dofs_size, c_dofs_size);
   E_f.CopyMN(elmat, d_dofs_size, c_dofs_size, 0, d_dofs_size);

   // assemble G constraint
   const int G_off = E_off;
   DenseMatrix G_f(&G_data[G_offsets[f] + G_off], c_dofs_size, d_dofs_size);
   G_f.CopyMN(elmat, c_dofs_size, d_dofs_size, d_dofs_size, 0);

   // assemble H matrix
   DenseMatrix H_f(&H_data[H_offsets[f]], c_dofs_size, c_dofs_size);
   DenseMatrix elmat_H;
   elmat_H.CopyMN(elmat, c_dofs_size, c_dofs_size, d_dofs_size, d_dofs_size);
   H_f += elmat_H;
}

void DarcyHybridization::AssembleHDGGrad(
   int el, FaceElementTransformations *FTr, BlockNonlinearFormIntegrator &nlfi,
   const Vector &x_f, const Vector &u_l, const Vector &p_l) const
{
   const int f = FTr->Face->ElementNo;
   const FiniteElement *fe_c = c_fes.GetFaceElement(f);
   const FiniteElement *fe_u = fes.GetFE(el);
   const FiniteElement *fe_p = fes_p.GetFE(el);
   const Array<const FiniteElement*> el_arr({fe_u, fe_p});
   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];
   const int c_dofs_size = x_f.Size();
   const Array<const Vector*> x_arr({&u_l, &p_l});

   int type = NonlinearFormIntegrator::HDGFaceType::ELEM
              | NonlinearFormIntegrator::HDGFaceType::TRACE
              | NonlinearFormIntegrator::HDGFaceType::CONSTR
              | NonlinearFormIntegrator::HDGFaceType::FACE;

   if (FTr->Elem1No != el) { type |= 1; }

   Array2D<DenseMatrix*> elmats;
   DenseMatrix elmat_A, elmat_D, elmat_E, elmat_G, elmat_H;
   elmats = NULL;
   elmats(0,0) = &elmat_A;
   elmats(1,1) = &elmat_D;
   elmats(1,2) = &elmat_E;
   elmats(2,1) = &elmat_G;
   elmats(2,2) = &elmat_H;

   nlfi.AssembleHDGFaceGrad(type, *fe_c, el_arr, *FTr, x_f, x_arr, elmats);

   // assemble A element matrices
   DenseMatrix A(&Af_data[Af_offsets[el]], a_dofs_size, a_dofs_size);
   if (elmat_A.Height() != 0) { A += elmat_A; }

   // assemble D element matrices
   DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
   if (elmat_D.Height() != 0) { D += elmat_D; }

   // assemble E constraint
   const int E_off = (FTr->Elem1No == el)?(0):(c_dofs_size*d_dofs_size);
   DenseMatrix E_f(&E_data[E_offsets[f] + E_off], d_dofs_size, c_dofs_size);
   if (elmat_E.Height() != 0) { E_f += elmat_E; }

   // assemble G constraint
   const int G_off = E_off;
   DenseMatrix G_f(&G_data[G_offsets[f] + G_off], c_dofs_size, d_dofs_size);
   if (elmat_G.Height() != 0) { G_f += elmat_G; }

   // assemble H matrix
   DenseMatrix H_f(&H_data[H_offsets[f]], c_dofs_size, c_dofs_size);
   if (elmat_H.Height() != 0) { H_f += elmat_H; }
}

void DarcyHybridization::ReduceRHS(const BlockVector &b_t, Vector &b_tr) const
{
   const Operator *tr_cP;

   if (IsNonlinear())
   {
      //store RHS for Mult
      if (!darcy_offsets.Size())
      {
         darcy_offsets.SetSize(3);
         darcy_offsets[0] = 0;
         darcy_offsets[1] = fes.GetVSize();
         darcy_offsets[2] = fes_p.GetVSize();
         darcy_offsets.PartialSum();
      }
      if (!darcy_toffsets.Size())
      {
         darcy_toffsets.SetSize(3);
         darcy_toffsets[0] = 0;
         darcy_toffsets[1] = fes.GetTrueVSize();
         darcy_toffsets[2] = fes_p.GetTrueVSize();
         darcy_toffsets.PartialSum();

         darcy_rhs.Update(darcy_toffsets);
      }
      darcy_rhs = b_t;

      //initialize reduced rhs
      if (ParallelC())
      {
         const Operator *tr_P = c_fes.GetProlongationMatrix();
         if (b_tr.Size() != tr_P->Width())
         {
            b_tr.SetSize(tr_P->Width());
            b_tr = 0.;
         }
      }
      else if ((tr_cP = c_fes.GetConformingProlongation()))
      {
         if (b_tr.Size() != tr_cP->Width())
         {
            b_tr.SetSize(tr_cP->Width());
            b_tr = 0.;
         }
      }
      else
      {
         if (b_tr.Size() != c_fes.GetVSize())
         {
            b_tr.SetSize(c_fes.GetVSize());
            b_tr = 0.;
         }
      }
      return;
   }

   Vector bu;

   if (!ParallelU())
   {
      const Operator *cR = fes.GetConformingRestriction();
      if (cR)
      {
         bu.SetSize(cR->Width());
         cR->MultTranspose(b_t.GetBlock(0), bu);
      }
      else
      {
         bu.MakeRef(const_cast<Vector&>(b_t.GetBlock(0)), 0, fes.GetVSize());
      }
   }
   else
   {
      const Operator *R = fes.GetRestrictionOperator();
      bu.SetSize(R->Width());
      R->MultTranspose(b_t.GetBlock(0), bu);
   }

   const Vector &bp = b_t.GetBlock(1);

   Vector b_r;

   if (!ParallelC() && !(tr_cP = c_fes.GetConformingProlongation()))
   {
      if (b_tr.Size() != c_fes.GetVSize())
      {
         b_tr.SetSize(c_fes.GetVSize());
         b_tr = 0.;
      }
      b_r.MakeRef(b_tr, 0, b_tr.Size());
   }
   else
   {
      b_r.SetSize(c_fes.GetVSize());
      b_r = 0.;
   }

   const int NE = fes.GetNE();
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Vector b_rl;
   Array<int> c_dofs;
   Array<int> faces;
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   MFEM_ASSERT(!c_bfi_p,
               "Potential constraint is not supported in non-block assembly!");
   Vector hat_u(hat_offsets.Last());
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Vector bu_l, bp_l, u_l, p_l;
   Array<int> u_vdofs, p_dofs;

   for (int el = 0; el < NE; el++)
   {
      // Load RHS

      GetFDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      fes_p.GetElementVDofs(el, p_dofs);
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
      GetElementFaces(el, faces);

      // Mult C u + G p
      for (int f = 0; f < faces.Size(); f++)
      {
         int el1, el2;
         fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
         DenseMatrix Ct;
         GetCtFaceMatrix(faces[f], el1 != el, Ct);

         b_rl.SetSize(Ct.Width());
         Ct.MultTranspose(u_l, b_rl);

         if (c_bfi_p)
         {
            DenseMatrix G;
            GetGFaceMatrix(faces[f], el1 != el, G);

            G.AddMult(p_l, b_rl);
         }

         c_fes.GetFaceVDofs(faces[f], c_dofs);
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

   if (!ParallelC())
   {
      if (tr_cP)
      {
         if (b_tr.Size() != tr_cP->Width())
         {
            b_tr.SetSize(tr_cP->Width());
            tr_cP->MultTranspose(b_r, b_tr);
         }
         else
         {
            tr_cP->AddMultTranspose(b_r, b_tr);
         }
      }
   }
   else
   {
      const Operator *tr_P = c_fes.GetProlongationMatrix();

      if (b_tr.Size() != tr_P->Width())
      {
         b_tr.SetSize(tr_P->Width());
         tr_P->MultTranspose(b_r, b_tr);
      }
      else
      {
         tr_P->AddMultTranspose(b_r, b_tr);
      }
   }
}

void DarcyHybridization::ProjectSolution(const BlockVector &sol,
                                         Vector &sol_r) const
{
   Mesh *mesh = c_fes.GetMesh();
   const int nfaces = mesh->GetNumFaces();

   const GridFunction p(&fes_p, const_cast<Vector&>(sol.GetBlock(1)), 0);

   DenseMatrix val_tr;
   Vector val1, val2;
   Array<int> c_vdofs;

   for (int f = 0; f < nfaces; f++)
   {
      FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(f);
      const FiniteElement *c_fe = c_fes.GetFaceElement(f);
      const IntegrationRule &nodes = c_fe->GetNodes();
      c_fes.GetFaceVDofs(f, c_vdofs);
      val_tr.SetSize(nodes.Size(), fes_p.GetVDim());
      MFEM_ASSERT(c_vdofs.Size() == nodes.Size() * fes_p.GetVDim(), "Internal error");

      for (int n = 0; n < nodes.Size(); n++)
      {
         const IntegrationPoint &ip = nodes[n];
         ftr->SetIntPoint(&ip);
         p.GetVectorValue(*ftr->Elem1, ftr->GetElement1IntPoint(), val1);
         if (ftr->Elem2No >= 0)
         {
            p.GetVectorValue(*ftr->Elem2, ftr->GetElement2IntPoint(), val2);
            val1 += val2;
            val1 *= 0.5;
         }
         val_tr.SetRow(n, val1);
      }

      sol_r.SetSubVector(c_vdofs, val_tr.GetData());
   }
}

void DarcyHybridization::ComputeSolution(const BlockVector &b_t,
                                         const Vector &sol_tr, BlockVector &sol_t) const
{
   if (IsNonlinear())
   {
      ParMultNL(MultNlMode::Sol, b_t, sol_tr, sol_t);
      return;
   }

   Vector sol_r;
   if (!ParallelC())
   {
      const SparseMatrix *tr_cP = c_fes.GetConformingProlongation();
      if (!tr_cP)
      {
         sol_r.SetDataAndSize(sol_tr.GetData(), sol_tr.Size());
      }
      else
      {
         sol_r.SetSize(c_fes.GetVSize());
         tr_cP->Mult(sol_tr, sol_r);
      }
   }
   else
   {
      sol_r.SetSize(c_fes.GetVSize());
      c_fes.GetProlongationMatrix()->Mult(sol_tr, sol_r);
   }

   Vector bu, u;

   if (!ParallelU())
   {
      const Operator *cR = fes.GetConformingRestriction();
      if (!cR)
      {
         bu.MakeRef(const_cast<Vector&>(b_t.GetBlock(0)), 0, fes.GetVSize());
         u.MakeRef(sol_t.GetBlock(0), 0, fes.GetVSize());
      }
      else
      {
         bu.SetSize(fes.GetVSize());
         cR->MultTranspose(b_t.GetBlock(0), bu);
         u.SetSize(bu.Size());
         cR->MultTranspose(sol_t.GetBlock(0), u);
      }
   }
   else
   {
      bu.SetSize(fes.GetVSize());
      fes.GetRestrictionOperator()->MultTranspose(b_t.GetBlock(0), bu);
      u.SetSize(bu.Size());
      fes.GetRestrictionOperator()->MultTranspose(sol_t.GetBlock(0), u);
   }

   const Vector &bp = b_t.GetBlock(1);
   Vector &p = sol_t.GetBlock(1);

   const int NE = fes.GetNE();
#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Vector sol_rl;
   Array<int> c_dofs;
   Array<int> faces;
#else //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   MFEM_ASSERT(!c_bfi_p,
               "Potential constraint is not supported in non-block assembly!");
   Vector hat_bu(hat_offsets.Last());
#endif //MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Vector bu_l, bp_l, u_l, p_l;
   Array<int> u_vdofs, p_dofs;

#ifndef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
   Ct->Mult(sol_r, hat_bu);
#endif //!MFEM_DARCY_HYBRIDIZATION_CT_BLOCK

   for (int el = 0; el < NE; el++)
   {
      //Load RHS

      GetFDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      fes_p.GetElementVDofs(el, p_dofs);
      bp.GetSubVector(p_dofs, bp_l);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         bp_l.Neg();
      }

#ifdef MFEM_DARCY_HYBRIDIZATION_CT_BLOCK
      GetElementFaces(el, faces);

      // bu - C^T sol
      for (int f = 0; f < faces.Size(); f++)
      {
         int el1, el2;
         fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
         DenseMatrix Ct;
         GetCtFaceMatrix(faces[f], el1 != el, Ct);

         c_fes.GetFaceVDofs(faces[f], c_dofs);
         sol_r.GetSubVector(c_dofs, sol_rl);

         Ct.AddMult_a(-1., sol_rl, bu_l);

         //bp - E sol
         if (c_bfi_p)
         {
            DenseMatrix E;
            GetEFaceMatrix(faces[f], el1 != el, E);

            E.AddMult_a(-1., sol_rl, bp_l);
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

   if (!ParallelU())
   {
      const Operator *cR = fes.GetConformingRestriction();
      if (cR)
      {
         cR->Mult(u, sol_t.GetBlock(0));
      }
   }
   else
   {
      fes.GetRestrictionOperator()->Mult(u, sol_t.GetBlock(0));
   }
}

void DarcyHybridization::ReconstructTotalFlux(
   const BlockVector &sol, const Vector &x, total_flux_fun ut_fx,
   GridFunction &ut) const
{
   const Vector &u = sol.GetBlock(0);
   const Vector &p = sol.GetBlock(1);

   const FiniteElementSpace &fes_ut = *ut.FESpace();

   MFEM_ASSERT(fes.GetMesh() == fes_ut.GetMesh(),
               "Different meshes are not supported!");

   Mesh *mesh = fes_ut.GetMesh();
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = (c_pfes)?(c_pfes->GetParMesh()):(NULL);
   const int NE = mesh->GetNE();
   ParGridFunction pu, pp;
   if (pfes && pfes_p && pmesh)
   {
      pu.MakeRef(const_cast<ParFiniteElementSpace*>(pfes), const_cast<Vector&>(u), 0);
      pu.ExchangeFaceNbrData();
      pp.MakeRef(const_cast<ParFiniteElementSpace*>(pfes_p), const_cast<Vector&>(p),
                 0);
      pp.ExchangeFaceNbrData();
   }
   else if (pfes || pfes_p || pmesh)
   {
      MFEM_ABORT("All, flux, potential and constraint parallel spaces are needed");
   }
#endif

   //element faces

   const int nfaces = mesh->GetNumFaces();
   Array<int> f_2_b = mesh->GetFaceToBdrElMap();
   Array<int> vdofs_ut, vdofs_xf, vdofs1, vdofs2, dofs1, dofs2;
   DenseMatrix Ct, Ct1, Ct2, Mf;
   Vector u1, u2, p1, p2, xf, bf, bf1, bf2, ut_f;
   MassIntegrator fbfi;
   DenseMatrixInverse Mfi;

   for (int f = 0; f < nfaces; f++)
   {
      fes_ut.GetFaceVDofs(f, vdofs_ut);
      MFEM_ASSERT(vdofs_ut.Size() == c_fes.GetFaceElement(f)->GetDof() *
                  c_fes.GetVDim(), "Incompatible constraint and total flux spaces");
      bf.SetSize(vdofs_ut.Size());
      ut_f.SetSize(vdofs_ut.Size());

      FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(f);
      const FiniteElement *fe_c = c_fes.GetFaceElement(f);

#ifdef MFEM_USE_MPI
      if (pmesh && pmesh->FaceIsTrueInterior(f) && ftr->Elem2No < 0)
      {
         // we do not store face neighbor constraint matrices so we must
         // integrate here over the face
         const FiniteElement *fe1 = fes.GetFE(ftr->Elem1No);
         const int nbr_el = -1 - ftr->Elem2No;
         const FiniteElement *fe2 = pfes->GetFaceNbrFE(nbr_el);
         ftr = pmesh->GetSharedFaceTransformationsByLocalIndex(f);
         c_bfi->AssembleFaceMatrix(*fe_c, *fe1, *fe2, *ftr, Ct);

         //side 1
         fes.GetElementVDofs(ftr->Elem1No, vdofs1);
         u.GetSubVector(vdofs1, u1);
         Ct1.CopyMN(Ct, vdofs1.Size(), vdofs_ut.Size(), 0, 0);
         Ct1.MultTranspose(u1, bf);

         //side 2
         pfes->GetFaceNbrElementVDofs(nbr_el, vdofs2);
         pu.FaceNbrData().GetSubVector(vdofs2, u2);
         Ct2.CopyMN(Ct, vdofs2.Size(), vdofs_ut.Size(), vdofs1.Size(), 0);
         // here we use the constraint integrator as well, but flip the sign
         // corresponding to the opposite normal for the total flux
         Ct2.AddMultTranspose(u2, bf, -1.);
      }
      else
#endif
      {
         //flux constraint

         //side 1
         if (ftr->Elem2No >= 0)
         {
            GetCtFaceMatrix(f, 0, Ct1);
         }
         else
         {
            // we do not rely on the boundary constraint integrators, which
            // might or might not be present, and apply the constraint
            // integrator at the boundaries as well
            const FiniteElement *fe1 = fes.GetFE(ftr->Elem1No);
            c_bfi->AssembleFaceMatrix(*fe_c, *fe1, *fe1, *ftr, Ct1);
         }

         fes.GetElementVDofs(ftr->Elem1No, vdofs1);
         u.GetSubVector(vdofs1, u1);
         Ct1.MultTranspose(u1, bf);

         //side 2
         if (ftr->Elem2No >= 0)
         {
            fes.GetElementVDofs(ftr->Elem2No, vdofs2);
            u.GetSubVector(vdofs2, u2);
            GetCtFaceMatrix(f, 1, Ct2);
            // here we use the constraint integrator as well, but flip the sign
            // corresponding to the opposite normal for the total flux
            Ct2.AddMultTranspose(u2, bf, -1.);
         }
      }

      //potential constraint

      if ((c_bfi_p || c_nlfi_p) && ftr->Elem2No >= 0)
      {
         // first side
         fes_p.GetElementVDofs(ftr->Elem1No, dofs1);
         p.GetSubVector(dofs1, p1);
         c_fes.GetFaceVDofs(f, vdofs_xf);
         x.GetSubVector(vdofs_xf, xf);

         const FiniteElement *fe1_p = fes_p.GetFE(ftr->Elem1No);
         const FiniteElement *face_fe = c_fes.GetFaceElement(f);

         int type = NonlinearFormIntegrator::HDGFaceType::CONSTR
                    | NonlinearFormIntegrator::HDGFaceType::FACE;

         if (c_bfi_p)
         {
            c_bfi_p->AssembleHDGFaceVector(type, *face_fe, *fe1_p, *ftr, xf, p1, bf1);
         }
         else
         {
            c_nlfi_p->AssembleHDGFaceVector(type, *face_fe, *fe1_p, *ftr, xf, p1, bf1);
         }
         bf += bf1;

         // second side
         const FiniteElement *fe2_p;
#ifdef MFEM_USE_MPI
         if (ftr->Elem2No >= NE)
         {
            const int nbr_el = ftr->Elem2No - NE;
            pfes_p->GetFaceNbrElementVDofs(nbr_el, dofs2);
            pp.FaceNbrData().GetSubVector(dofs2, p2);
            fe2_p = pfes_p->GetFaceNbrFE(nbr_el);
         }
         else
#endif
         {
            fes_p.GetElementVDofs(ftr->Elem2No, dofs2);
            p.GetSubVector(dofs2, p2);
            fe2_p = fes_p.GetFE(ftr->Elem2No);
         }

         type |= 1;
         if (c_bfi_p)
         {
            c_bfi_p->AssembleHDGFaceVector(type, *face_fe, *fe2_p, *ftr, xf, p2, bf2);
         }
         else
         {
            c_nlfi_p->AssembleHDGFaceVector(type, *face_fe, *fe2_p, *ftr, xf, p2, bf2);
         }
         bf -= bf2;
      }

      // boundary potential constraint
      if (ftr->Elem2No < 0 && (!boundary_constraint_pot_integs.empty() ||
                               !boundary_constraint_pot_nonlin_integs.empty()))
      {
         constexpr int type = NonlinearFormIntegrator::HDGFaceType::CONSTR
                              | NonlinearFormIntegrator::HDGFaceType::FACE;

         const FiniteElement *fe_p = fes_p.GetFE(ftr->Elem1No);
         const FiniteElement *face_fe = c_fes.GetFaceElement(f);

         fes_p.GetElementVDofs(ftr->Elem1No, dofs1);
         p.GetSubVector(dofs1, p1);
         c_fes.GetFaceVDofs(f, vdofs_xf);
         x.GetSubVector(vdofs_xf, xf);

         const int bdr_attr = mesh->GetBdrAttribute(f_2_b[f]);

         // linear
         for (size_t i = 0; i < boundary_constraint_pot_integs.size(); i++)
         {
            if (boundary_constraint_pot_integs_marker[i]
                && (*boundary_constraint_pot_integs_marker[i])[bdr_attr-1] == 0) { continue; }

            boundary_constraint_pot_integs[i]->AssembleHDGFaceVector(type, *face_fe, *fe_p,
                                                                     *ftr, xf, p1, bf1);

            bf += bf1;
         }

         // nonlinear
         for (size_t i = 0; i < boundary_constraint_pot_nonlin_integs.size(); i++)
         {
            if (boundary_constraint_pot_nonlin_integs_marker[i]
                && (*boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

            boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceVector(type, *face_fe,
                                                                            *fe_p, *ftr, xf, p1, bf1);

            bf += bf1;
         }
      }

      //face
      fbfi.AssembleElementMatrix2(*fes_ut.GetFaceElement(f), *fe_c,
                                  *ftr, Mf);

      Mfi.Factor(Mf);
      Mfi.Mult(bf, ut_f);
      if (ftr->Elem2No >= 0)
      {
         // the face term should be double integrated to account for both sides
         // so divide the values by two after inversion
         ut_f *= .5;
      }

      ut.SetSubVector(vdofs_ut, ut_f);
   }

   if (fes_ut.FEColl()->GetOrder() <= 1) { return; }

   //element interior

   const int dim = mesh->Dimension();
   VectorFEMassIntegrator Mut;
   Array<int> vdofs, dofs, vdofs_ut_b, vdofs_ut_i;
   DenseMatrix Mut_z, Mut_zi;
   DenseMatrix vshape_u, vshape_ut;
   Vector shape_u, shape_ut, shape_p, u_q(dim), ut_q(dim);
   Vector u_z, p_z, b_z, b_zi, ut_zb, ut_zi;
   DenseMatrixInverse Muti_zi;

   for (int z = 0; z < fes.GetNE(); z++)
   {
      const FiniteElement *fe_ut = fes_ut.GetFE(z);
      const FiniteElement *fe_u = fes.GetFE(z);
      const FiniteElement *fe_p = fes_p.GetFE(z);

      ElementTransformation *Tr = mesh->GetElementTransformation(z);

      fes.GetElementVDofs(z, vdofs);
      u.GetSubVector(vdofs, u_z);

      fes_p.GetElementVDofs(z, dofs);
      p.GetSubVector(dofs, p_z);

      fes_ut.GetElementVDofs(z, vdofs_ut);
      const int nvdofs = vdofs_ut.Size();

      //integrate rhs

      if (fe_u->GetRangeType() == FiniteElement::VECTOR)
      {
         vshape_u.SetSize(vdofs.Size(), dim);
      }
      else
      {
         shape_u.SetSize(fe_u->GetDof());
      }
      shape_p.SetSize(dofs.Size());
      vshape_ut.SetSize(nvdofs, dim);
      shape_ut.SetSize(nvdofs);

      b_z.SetSize(nvdofs);
      b_z = 0.;

      const int order = Tr->OrderW()
                        + std::max(fe_u->GetOrder(), fe_p->GetOrder())
                        + fe_ut->GetOrder();
      const IntegrationRule *ir = &IntRules.Get(fe_ut->GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Tr->SetIntPoint(&ip);

         if (fe_u->GetRangeType() == FiniteElement::VECTOR)
         {
            fe_u->CalcVShape(*Tr, vshape_u);
            vshape_u.MultTranspose(u_z, u_q);
         }
         else
         {
            fe_u->CalcPhysShape(*Tr, shape_u);
            DenseMatrix u_zm(u_z.GetData(), shape_u.Size(), dim);
            u_zm.MultTranspose(shape_u, u_q);
         }

         fe_p->CalcShape(ip, shape_p);
         const real_t p_q = p_z * shape_p;

         ut_fx(*Tr, u_q, p_q, ut_q);

         fe_ut->CalcVShape(*Tr, vshape_ut);
         vshape_ut.Mult(ut_q, shape_ut);

         const real_t w = ip.weight * Tr->Weight();
         b_z.Add(w, shape_ut);
      }

      //assemble mass matrix

      Mut.AssembleElementMatrix(*fe_ut, *Tr, Mut_z);

      //eliminate boundary rows

      const int nidofs = fes_ut.GetNumElementInteriorDofs(z);
      const int nbdofs = nvdofs - nidofs;
      vdofs_ut_b.MakeRef(vdofs_ut.GetData(), nbdofs);
      ut.GetSubVector(vdofs_ut_b, ut_zb);

      for (int j = 0; j < nbdofs; j++)
      {
         for (int i = 0; i < nidofs; i++)
         {
            b_z(i+nbdofs) -= Mut_z(i+nbdofs,j) * ut_zb(j);
         }
      }

      Mut_zi.CopyMN(Mut_z, nidofs, nidofs, nbdofs, nbdofs);

      //solve for the interior dofs

      Muti_zi.Factor(Mut_zi);
      ut_zi.SetSize(Mut_zi.Width());
      b_zi.MakeRef(b_z, nbdofs, nidofs);
      Muti_zi.Mult(b_zi, ut_zi);

      vdofs_ut_i.MakeRef(vdofs_ut.GetData()+nbdofs, nidofs);
      ut.SetSubVector(vdofs_ut_i, ut_zi);
   }
}

void DarcyHybridization::Reset()
{
   Hybridization::Reset();
   bfin = false;

   A_empty = true;
   Bf_data = 0.;
   if (Df_data.Size())
   {
      Df_data = 0.;
      D_empty = true;
   }
#ifdef MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
   Be_data = 0.;
#endif //MFEM_DARCY_HYBRIDIZATION_ELIM_BCS
}

#ifndef MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
void DarcyHybridization::Gradient::Mult(const Vector &x, Vector &y) const
{
   //note that rhs is not used, it is only a dummy
   dh.MultNL(MultNlMode::GradMult, dh.darcy_rhs, x, y);
}
#endif //MFEM_DARCY_HYBRIDIZATION_GRAD_MAT

#ifdef MFEM_USE_MPI
void DarcyHybridization::ParOperator::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(dh.bfin, "DarcyHybridization must be finalized");

   if (dh.pH.Ptr())
   {
      dh.pH->Mult(x, y);
      return;
   }

   dh.ParMultNL(MultNlMode::Mult, dh.darcy_rhs, x, y);
}
Operator &DarcyHybridization::ParOperator::GetGradient(const Vector &x) const
{
   MFEM_VERIFY(dh.bfin, "DarcyHybridization must be finalized");

   if (dh.pH.Ptr()) { return *dh.pH.Ptr(); }

   if (!dh.Df_data.Size()) { dh.AllocD(); }// D is resetted in ConstructGrad()
   if (!dh.E_data.Size() || !dh.G_data.Size()) { dh.AllocEG(); }// E and G are rewritten
   if (!dh.H_data.Size()) { dh.AllocH(); }
   else if (dh.c_nlfi_p || dh.c_nlfi)
   {
      // H is resetted here for additive double side integration
      dh.H_data = 0.;
   }

   Vector y;//dummy
   dh.ParMultNL(MultNlMode::Grad, dh.darcy_rhs, x, y);

#ifdef MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
   //assemble gradient matrix
   dh.Grad.reset();
   pGrad.SetType(dh.pH.Type());
   dh.ComputeParH(ComputeHMode::Gradient, dh.Grad, pGrad);
#else
   //construct gradient operator
   pGrad.Reset(new ParGradient(dh));
#endif
   return *pGrad;
}

#ifndef MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
void DarcyHybridization::ParGradient::Mult(const Vector &x, Vector &y) const
{
   //note that rhs is not used, it is only a dummy
   dh.ParMultNL(MultNlMode::GradMult, dh.darcy_rhs, x, y);
}
#endif //MFEM_DARCY_HYBRIDIZATION_GRAD_MAT
#endif // MFEM_USE_MPI

DarcyHybridization::LocalNLOperator::LocalNLOperator(
   const DarcyHybridization &dh_, int el_, const BlockVector &trps_,
   const Array<int> &faces_)
   : dh(dh_), el(el_), trps(trps_), faces(faces_),
     a_dofs_size(dh.Af_f_offsets[el+1] - dh.Af_f_offsets[el]),
     d_dofs_size(dh.Df_f_offsets[el+1] - dh.Df_f_offsets[el]),
     B(const_cast<real_t*>(&dh.Bf_data[dh.Bf_offsets[el]]),
       d_dofs_size, a_dofs_size),
     Bt(B), offsets({0, a_dofs_size, a_dofs_size+d_dofs_size}), grad(offsets)
{
   width = height = a_dofs_size + d_dofs_size;

   fe_u = dh.fes.GetFE(el);
   fe_p = dh.fes_p.GetFE(el);

   const Mesh *mesh = dh.fes.GetMesh();

   // element transformation
   Tr = new IsoparametricTransformation();
   if (faces.Size() <= 0)
   {
      mesh->GetElementTransformation(el, Tr);
   }

   // face transformations
   FTrs.resize(faces.Size());
   NbrTrs.resize(faces.Size());
   for (int f = 0; f < faces.Size(); f++)
   {
      FaceElementTransformations *&FTr = FTrs[f];
      FTr = new FaceElementTransformations();
      mesh->GetFaceElementTransformations(faces[f], *FTr, *Tr, *Tr, 0);
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

         mesh->GetFaceElementTransformations(faces[f], *FTr, *Tr1, *Tr2);
      }
#ifdef MFEM_USE_MPI
      else if (dh.ParallelC() &&
               dh.c_pfes->GetParMesh()->FaceIsTrueInterior(faces[f]))
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

         dh.c_pfes->GetParMesh()->GetSharedFaceTransformationsByLocalIndex(faces[f],
                                                                           *FTr, *Tr1, *Tr2);
      }
#endif
      else
      {
         NbrTrs[f] = NULL;
         Tr1 = Tr2 = Tr;

         mesh->GetFaceElementTransformations(faces[f], *FTr, *Tr1, *Tr2, 21);
      }
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

void DarcyHybridization::LocalNLOperator::AddMultBlock(const Vector &u_l,
                                                       const Vector &p_l, Vector &bu, Vector &bp) const
{
   if (dh.m_nlfi)
   {
      //element contribution
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array<Vector*> y_arr({&Au, &Dp});

      dh.m_nlfi->AssembleElementVector(fe_arr, *Tr, x_arr, y_arr);
      if (Au.Size() != 0) { bu += Au; }
      if (Dp.Size() != 0) { bp += Dp; }
   }

   if (dh.c_nlfi)
   {
      //face contribution
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array<Vector*> y_arr({&Au, &Dp, (Vector*)NULL});

      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = FTrs[f];

         int type = BlockNonlinearFormIntegrator::HDGFaceType::ELEM
                    | BlockNonlinearFormIntegrator::HDGFaceType::TRACE;

         const Vector &trp_f = trps.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior
            if (FTr->Elem1No != el) { type |= 1; }

            dh.c_nlfi->AssembleHDGFaceVector(type, *dh.c_fes.GetFaceElement(faces[f]),
                                             fe_arr, *FTr, trp_f, x_arr, y_arr);

            if (Au.Size() != 0) { bu += Au; }
            if (Dp.Size() != 0) { bp += Dp; }
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes.GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (size_t i = 0; i < dh.boundary_constraint_nonlin_integs.size(); i++)
            {
               if (dh.boundary_constraint_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               dh.boundary_constraint_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                              *dh.c_fes.GetFaceElement(faces[f]),
                                                                              fe_arr,
                                                                              *FTr,
                                                                              trp_f, x_arr, y_arr);

               if (Au.Size() != 0) { bu += Au; }
               if (Dp.Size() != 0) { bp += Dp; }
            }
         }
      }
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
      const DenseMatrix A(const_cast<real_t*>(&dh.Af_lin_data[dh.Af_offsets[el]]),
                          a_dofs_size, a_dofs_size);
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
      const DenseMatrix D(&dh.Df_lin_data[dh.Df_offsets[el]],
                          d_dofs_size, d_dofs_size);
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

            dh.c_nlfi_p->AssembleHDGFaceVector(type, *dh.c_fes.GetFaceElement(faces[f]),
                                               *fe_p, *FTr, trp_f, p_l, DpEx);

            bp += DpEx;
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes.GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (size_t i = 0; i < dh.boundary_constraint_pot_nonlin_integs.size(); i++)
            {
               if (dh.boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               dh.boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceVector(type,
                                                                                  *dh.c_fes.GetFaceElement(faces[f]),
                                                                                  *fe_p, *FTr, trp_f, p_l, DpEx);

               bp += DpEx;
            }
         }
      }
   }
}

void DarcyHybridization::LocalNLOperator::AddGradBlock(const Vector &u_l,
                                                       const Vector &p_l, DenseMatrix &gA, DenseMatrix &gD) const
{
   if (dh.m_nlfi)
   {
      //element contribution
      DenseMatrix gA, gD;
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array2D<DenseMatrix*> grad_arr(2,2);
      grad_arr = NULL;
      grad_arr(0,0) = &gA;
      grad_arr(1,1) = &gD;
      dh.m_nlfi->AssembleElementGrad(fe_arr, *Tr, x_arr, grad_arr);
      if (gA.Height() != 0) { grad_A += gA; }
      if (gD.Height() != 0) { grad_D += gD; }
   }

   if (dh.c_nlfi)
   {
      //face contribution
      DenseMatrix gA, gD;
      Array<const FiniteElement*> fe_arr({fe_u, fe_p});
      Array<const Vector*> x_arr({&u_l, &p_l});
      Array2D<DenseMatrix*> grad_arr(3,3);
      grad_arr = NULL;
      grad_arr(0,0) = &gA;
      grad_arr(1,1) = &gD;

      for (int f = 0; f < faces.Size(); f++)
      {
         FaceElementTransformations *FTr = FTrs[f];

         int type = BlockNonlinearFormIntegrator::HDGFaceType::ELEM;

         const Vector &trp_f = trps.GetBlock(f);

         if (FTr->Elem2No >= 0)
         {
            //interior
            if (FTr->Elem1No != el) { type |= 1; }

            dh.c_nlfi->AssembleHDGFaceGrad(type, *dh.c_fes.GetFaceElement(faces[f]),
                                           fe_arr, *FTr, trp_f, x_arr, grad_arr);

            if (gA.Height() != 0) { grad_A += gA; }
            if (gD.Height() != 0) { grad_D += gD; }
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes.GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (size_t i = 0; i < dh.boundary_constraint_nonlin_integs.size(); i++)
            {
               if (dh.boundary_constraint_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               dh.boundary_constraint_nonlin_integs[i]->AssembleHDGFaceGrad(type,
                                                                            *dh.c_fes.GetFaceElement(faces[f]),
                                                                            fe_arr,
                                                                            *FTr,
                                                                            trp_f, x_arr, grad_arr);

               if (gA.Height() != 0) { grad_A += gA; }
               if (gD.Height() != 0) { grad_D += gD; }
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
      DenseMatrix A(const_cast<real_t*>(&dh.Af_lin_data[dh.Af_offsets[el]]),
                    a_dofs_size, a_dofs_size);
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
      DenseMatrix D(&dh.Df_lin_data[dh.Df_offsets[el]], d_dofs_size, d_dofs_size);
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

            dh.c_nlfi_p->AssembleHDGFaceGrad(type, *dh.c_fes.GetFaceElement(faces[f]),
                                             *fe_p, *FTr, trp_f, p_l, grad_Df);

            grad += grad_Df;
         }
         else
         {
            //boundary
            const int bdr_attr = dh.fes.GetMesh()->GetBdrAttribute(dh.f_2_b[faces[f]]);

            for (size_t i = 0; i < dh.boundary_constraint_pot_nonlin_integs.size(); i++)
            {
               if (dh.boundary_constraint_pot_nonlin_integs_marker[i]
                   && (*dh.boundary_constraint_pot_nonlin_integs_marker[i])[bdr_attr-1] == 0) { continue; }

               dh.boundary_constraint_pot_nonlin_integs[i]->AssembleHDGFaceGrad(type,
                                                                                *dh.c_fes.GetFaceElement(faces[f]),
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
   //bp += B u_l + D p_l + E x_f
   AddMultBlock(u_l, p_l, bu, bp);
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
   grad_A = 0.;
   grad_D = 0.;

   //block
   AddGradBlock(u_l, p_l, grad_A, grad_D);

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
     LU_D(&dh.Df_data[dh.Df_offsets[el]], &dh.Df_ipiv[dh.Df_f_offsets[el]])
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
     LU_A(&dh.Af_data[dh.Af_offsets[el]], &dh.Af_ipiv[dh.Af_f_offsets[el]])
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
