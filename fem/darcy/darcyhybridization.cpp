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

#include "darcyhybridization.hpp"

#include "../../mesh/segment.hpp"
#include "../../mesh/triangle.hpp"
#include "../../mesh/quadrilateral.hpp"

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
}

DarcyHybridization::~DarcyHybridization()
{
   if (!extern_bdr_constr_pot_integs)
   {
      for (size_t k=0; k < boundary_constraint_pot_integs.size(); k++)
      { delete boundary_constraint_pot_integs[k]; }
   }
}

void DarcyHybridization::SetConstraintIntegrators(
   BilinearFormIntegrator *c_flux_integ, BilinearFormIntegrator *c_pot_integ)
{
   c_bfi.reset(c_flux_integ);
   c_bfi_p.reset(c_pot_integ);
}

void DarcyHybridization::Init(const Array<int> &ess_flux_tdof_list)
{
   const int NE = fes.GetNE();

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

   // Define Bf_offsets, Df_offsets and Df_f_offsets
   Bf_offsets.SetSize(NE+1);
   Bf_offsets[0] = 0;
   Df_offsets.SetSize(NE+1);
   Df_offsets[0] = 0;
   Df_f_offsets.SetSize(NE+1);
   Df_f_offsets[0] = 0;
   Ae_offsets.SetSize(NE+1);
   Ae_offsets[0] = 0;
   Be_offsets.SetSize(NE+1);
   Be_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      int f_size = Af_f_offsets[i+1] - Af_f_offsets[i];
      int d_size = fes_p.GetFE(i)->GetDof() * fes_p.GetVDim();
      Bf_offsets[i+1] = Bf_offsets[i] + f_size*d_size;
      Df_offsets[i+1] = Df_offsets[i] + d_size*d_size;
      Df_f_offsets[i+1] = Df_f_offsets[i] + d_size;
      int a_size = hat_offsets[i+1] - hat_offsets[i];
      int e_size = a_size - f_size;
      Ae_offsets[i+1] = Ae_offsets[i] + e_size*a_size;
      Be_offsets[i+1] = Be_offsets[i] + e_size*d_size;
   }

   Bf_data.SetSize(Bf_offsets[NE]); Bf_data = 0.;
   AllocD();
   Ae_data.SetSize(Ae_offsets[NE]);
   Be_data.SetSize(Be_offsets[NE]); Be_data = 0.;

   if (c_bfi_p)
   {
      AllocEG();
   }
}

void DarcyHybridization::SetEssentialBC(const Array<int> &bdr_attr_is_ess)
{
   c_fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
}

void DarcyHybridization::SetEssentialVDofs(const Array<int> &ess_vdofs_list)
{
   if (c_fes.Conforming() && !ParallelC())
   {
      ess_vdofs_list.Copy(ess_tdof_list); // ess_vdofs_list --> ess_tdof_list
   }
   else
   {
      Array<int> ess_vdof_marker, ess_tdof_marker;
      FiniteElementSpace::ListToMarker(ess_vdofs_list, c_fes.GetVSize(),
                                       ess_vdof_marker);
      if (!ParallelC())
      {
         c_fes.ConvertToConformingVDofs(ess_vdof_marker, ess_tdof_marker);
      }
      else
      {
#ifdef MFEM_USE_MPI
         ess_tdof_marker.SetSize(c_pfes->GetTrueVSize());
         c_pfes->Dof_TrueDof_Matrix()->BooleanMultTranspose(1, ess_vdof_marker,
                                                            0, ess_tdof_marker);
#else
         MFEM_ABORT("internal MFEM error");
#endif
      }
      FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdof_list);
   }
}

void DarcyHybridization::AssembleFluxMassMatrix(int el, const DenseMatrix &A)
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   int Af_el_idx = Af_offsets[el];
   int Ae_el_idx = Ae_offsets[el];

   for (int j = 0; j < s; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
         for (int i = 0; i < s; i++)
         {
            Ae_data[Ae_el_idx++] = A(i, j);
         }
         continue;
      }
      for (int i = 0; i < s; i++)
      {
         if (hat_dofs_marker[o + i] == 1) { continue; }
         Af_data[Af_el_idx++] = A(i, j);
      }
   }
   MFEM_ASSERT(Af_el_idx == Af_offsets[el+1], "Internal error");
   MFEM_ASSERT(Ae_el_idx == Ae_offsets[el+1], "Internal error");

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
   int Be_el_idx = Be_offsets[el];

   for (int j = 0; j < w; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
         for (int i = 0; i < h; i++)
         {
            Be_data[Be_el_idx++] += B(i, j);
         }
         continue;
      }
      for (int i = 0; i < h; i++)
      {
         Bf_data[Bf_el_idx++] += B(i, j);
      }
   }
   MFEM_ASSERT(Bf_el_idx == Bf_offsets[el+1], "Internal error");
   MFEM_ASSERT(Be_el_idx == Be_offsets[el+1], "Internal error");
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
   if (face < num_faces)
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
         const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
         ItHI_f.SetSize(c_size);
         m.Reset(ItHI_f.GetData(), c_size, c_size);
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
   if (!H) { H.reset(new SparseMatrix(c_fes.GetVSize())); }
   h_elmat.CopyMN(elmat, c_dof, c_dof, ndof, ndof);
   H->AddSubMatrix(c_dofs, c_dofs, h_elmat, skip_zeros);
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
                                             DenseMatrix &Ct_, int ioff)
{
   const int hat_offset = hat_offsets[el];
   const int hat_size = hat_offsets[el+1] - hat_offset;

   int row = 0;
   for (int i = 0; i < hat_size; i++)
   {
      if (hat_dofs_marker[hat_offset + i] == 1) { continue; }
      bool bzero = true;
      for (int j = 0; j < Ct_.Width(); j++)
      {
         const real_t val = elmat(i + ioff, j);
         if (val == 0.) { continue; }
         Ct_(row, j) = val;
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
                                                   face_getter fx_Ct, const DenseMatrix *Ct_,
                                                   face_getter fx_C, const DenseMatrix *C_,
                                                   face_getter fx_H, const DenseMatrix *H_)
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
         int sinf1, sinf2;
         mesh->GetFaceInfos(slave.index, &sinf1, &sinf2);
         if (Mesh::DecodeFaceInfoOrientation(sinf2)) { oris_s[0] *= -1; }
      }

      ord_s = c_fec->DofOrderForOrientation(slave.Geom(), oris_s[0]);

      //reorder the interpolation matrix edge->face
      Io.SetSize(I.Height(), I.Width());
      if (c_fec->GetContType() == FiniteElementCollection::CONTINUOUS)
      {
         for (int j = 0; j < I.Width(); j++)
         {
            int ord_mj;
            if (j <= 1)
            {
               // vertices
               ord_mj = (oris_m[0] > 0)?(j):(1-j);
            }
            else
            {
               // internal DOFs
               ord_mj = ord_m[j-2]+2;
            }
            for (int i = 0; i < I.Height(); i++)
            {
               int ord_si;
               if (i <= 1)
               {
                  // vertices
                  ord_si = (oris_s[0] > 0)?(i):(1-i);
               }
               else
               {
                  // internal DOFs
                  ord_si = ord_s[i-2]+2;
               }
               Io(ord_si, ord_mj) = I(i,j);
            }
         }
      }
      else
      {
         // reorder DOFs
         for (int j = 0; j < I.Width(); j++)
            for (int i = 0; i < I.Height(); i++)
            {
               Io(ord_s[i], ord_m[j]) = I(i,j);
            }
      }
   }
   else
   {
      Io.Reset(I.GetData(), I.Height(), I.Width());
   }

   if (c_fes.GetVDim() > 0)
   {
      const int vdim = c_fes.GetVDim();
      const int dofs_in =  Io.Height();
      const int dofs_out = Io.Width();
      if (fx_Ct)
      {
         const int dofs_el = Ct_->Height();
         DenseMatrix Ct_d(dofs_el, dofs_in);
         DenseMatrix Ct_md(dofs_el, dofs_out);
         for (int d = 0; d < vdim; d++)
         {
            Ct_d.CopyMN(*Ct_, dofs_el, dofs_in, 0, d*dofs_in);
            mfem::Mult(Ct_d, Io, Ct_md);
            Ct_m.AddMatrix(Ct_md, 0, d*dofs_out);
         }
      }
      if (fx_C)
      {
         const int dofs_el = C_->Width();
         DenseMatrix C_d(dofs_in, dofs_el);
         DenseMatrix C_md(dofs_out, dofs_el);
         for (int d = 0; d < vdim; d++)
         {
            C_d.CopyMN(*C_, dofs_in, dofs_el, d*dofs_in, 0);
            mfem::MultAtB(Io, C_d, C_md);
            C_m.AddMatrix(C_md, d*dofs_out, 0);
         }
      }
      if (fx_H)
      {
         DenseMatrix H_d(dofs_in);
         DenseMatrix H_md(dofs_out);
         for (int di = 0; di < vdim; di++)
            for (int dj = 0; dj < vdim; dj++)
            {
               H_d.CopyMN(*H_, dofs_in, dofs_in, di*dofs_in, dj*dofs_in);
               RAP(H_d, Io, H_md);
               H_m.AddMatrix(H_md, di*dofs_out, dj*dofs_out);
            }
      }
   }
   else
   {
      if (fx_Ct)
      {
         mfem::AddMult(*Ct_, Io, Ct_m);
      }
      if (fx_C)
      {
         mfem::AddMultAtB(Io, *C_, C_m);
      }
      if (fx_H)
      {
         DenseMatrix H_ma(H_m.Height(), H_m.Width());
         RAP(*H_, Io, H_ma);
         H_m += H_ma;
      }
   }
}

void DarcyHybridization::AssembleNCSlaveCtFaceMatrix(int face,
                                                     const DenseMatrix &Ct_)
{
   AssembleNCSlaveFaceMatrix(face,
   [this](int f, DenseMatrix &m) { GetCtFaceMatrix(f, 0, m); }, &Ct_);
}

void DarcyHybridization::AssembleNCSlaveEGFaceMatrix(int face,
                                                     const DenseMatrix &E, const DenseMatrix &G)
{
   AssembleNCSlaveFaceMatrix(face,
   [this](int f, DenseMatrix &m) { GetEFaceMatrix(f, 0, m); }, &E,
   [this](int f, DenseMatrix &m) { GetGFaceMatrix(f, 0, m); }, &G);
}

void DarcyHybridization::ConstructC()
{
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

void DarcyHybridization::InvertA()
{
   const int NE = fes.GetNE();

   for (int el = 0; el < NE; el++)
   {
      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];

      // Decompose A

      LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);

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
      const real_t norm = D.MaxMaxNorm();
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

void DarcyHybridization::ComputeH(std::unique_ptr<SparseMatrix> &H_) const
{
   const int skip_zeros = 1;
   const int NE = fes.GetNE();
   DenseMatrix S;
   Array<int> S_ipiv;
   DenseMatrix AiBt, AiCt, BAiCt, CAiBt, H_l;
   Array<int> c_dofs_1, c_dofs_2;
   Array<int> faces;
   if (!H_) { H_.reset(new SparseMatrix(c_fes.GetVSize())); }

   for (int el = 0; el < NE; el++)
   {
      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

      // Decompose A
      LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);
      LU_A.Factor(a_dofs_size);

      // Construct Schur complement
      const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                          d_dofs_size, a_dofs_size);
      DenseMatrix D(&Df_data[Df_offsets[el]], d_dofs_size, d_dofs_size);
      AiBt.SetSize(a_dofs_size, d_dofs_size);

      AiBt.Transpose(B);
      if (!bsym) { AiBt.Neg(); }
      LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());

      LUFactors LU_S;
      mfem::AddMult(B, AiBt, D);

      // Decompose Schur complement
      LU_S.data = D.GetData();
      LU_S.ipiv = &Df_ipiv[Df_f_offsets[el]];
      LU_S.Factor(d_dofs_size);

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

         if (c_bfi_p)
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

            if (c_bfi_p)
            {
               DenseMatrix G;
               GetGFaceMatrix(faces[f2], el2_1 != el, G);

               CAiBt += G;
            }

            mfem::AddMult(CAiBt, BAiCt, H_l);

            if (f1 == f2)
            {
               H_->AddSubMatrix(c_dofs_1, c_dofs_1, H_l, skip_zeros);
            }
            else
            {
               c_fes.GetFaceVDofs(faces[f2], c_dofs_2);
               H_->AddSubMatrix(c_dofs_2, c_dofs_1, H_l, skip_zeros);
            }
         }

      }
   }

   if (diag_policy == DIAG_ONE || diag_policy == DIAG_ZERO)
   {
      // put zeroes on the diagonal
      for (int i = 0; i < H_->Height(); i++)
      {
         H_->SetColPtr(i);
         H_->SearchRow(i);
      }
      H_->Finalize(0);
   }
   else
   {
      H_->Finalize(skip_zeros);
   }

   if (!ParallelC())
   {
      const SparseMatrix *cP = c_fes.GetConformingProlongation();
      if (cP)
      {
         if (H_->Height() != cP->Width())
         {
            SparseMatrix *cH = mfem::RAP(*cP, *H_, *cP);
            H_.reset(cH);
         }
      }

      // ensure diagonal is non-zero
      if (diag_policy == DIAG_ONE)
      {
         H_->SetDiagIdentity();
      }
   }
}

#ifdef MFEM_USE_MPI
void DarcyHybridization::ComputeParH(std::unique_ptr<SparseMatrix> &H_,
                                     OperatorHandle &pH_) const
{
   ComputeH(H_);

   if (!ParallelC())
   {
      pH_.Reset(H_.get(), false);
   }
   else // parallel
   {
      OperatorHandle dH(pH_.Type()), pP(pH_.Type());
      dH.MakeSquareBlockDiag(c_pfes->GetComm(), c_pfes->GlobalVSize(),
                             c_pfes->GetDofOffsets(), H_.get());
      // TODO - construct Dof_TrueDof_Matrix directly in the pS format
      pP.ConvertFrom(c_pfes->Dof_TrueDof_Matrix());
      pH_.MakePtAP(dH, pP);
      dH.Clear();
      pP.Clear();
      H_.reset();

      if (diag_policy == DIAG_ONE)
      {
         MFEM_ASSERT(pH_.Type() == Operator::Hypre_ParCSR,
                     "Fix of the diagonal is implemented only for HypreParMatrix");
         pH_.As<HypreParMatrix>()->EliminateZeroRows();
      }
   }
}
#endif //MFEM_USE_MPI

void DarcyHybridization::GetCtFaceMatrix(
   int f, int side, DenseMatrix &Ct_) const
{
   int el1, el2;
   fes.GetMesh()->GetFaceElements(f, &el1, &el2);

   const int c_size = c_fes.GetFaceElement(f)->GetDof() * c_fes.GetVDim();
   const int f_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];

   if (side == 0)
   {
      Ct_.Reset(const_cast<real_t*>(&Ct_data[Ct_offsets[f]]), f_size_1, c_size);
   }
   else
   {
      MFEM_ASSERT(el2 >= 0, "Invalid element");
      const int f_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];
      Ct_.Reset(const_cast<real_t*>(&Ct_data[Ct_offsets[f] + f_size_1*c_size]),
                f_size_2, c_size);
   }
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

void DarcyHybridization::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(bfin, "DarcyHybridization must be finalized");

   H->Mult(x, y);
}

Operator &DarcyHybridization::GetGradient(const Vector &x) const
{
   MFEM_VERIFY(bfin, "DarcyHybridization must be finalized");

   return *H;
}

void DarcyHybridization::Finalize()
{
   if (bfin) { return; }

#ifndef MFEM_USE_MPI
   ComputeH(H);
#else //MFEM_USE_MPI      
   ComputeParH(H, pH);
#endif //MFEM_USE_MPI
   EliminateTraceTrueDofs(diag_policy);

   bfin = true;
}

void DarcyHybridization::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                             const BlockVector &x, BlockVector &b)
{
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

void DarcyHybridization::EliminateTraceTrueDofs(const Array<int> &tdofs,
                                                DiagonalPolicy dpolicy)
{
   if (!ParallelC())
   {
      He.reset(new SparseMatrix(H->Height()));

      if (tdofs.Size() == 0) { return; }

      for (int vdof : tdofs)
      {
         H->EliminateRowCol(vdof, *He, diag_policy);
      }

      He->Finalize();
   }
   else
   {
#ifdef MFEM_USE_MPI
      MFEM_ASSERT(pH.Type() == Operator::Hypre_ParCSR,
                  "Implemented for HypreParMatrix only!");
      pHe.Reset(pH.As<HypreParMatrix>()->EliminateRowsCols(tdofs));
#endif //MFEM_USE_MPI
   }
}

void DarcyHybridization::EliminateTraceTrueDofs(DiagonalPolicy dpolicy)
{
   EliminateTraceTrueDofs(GetEssentialTrueDofs(), dpolicy);
}

void DarcyHybridization::EliminateTraceTrueDofsInRHS(const Array<int> &tdofs_,
                                                     const Vector &x, Vector &b)
{
   if (!ParallelC())
   {
      MFEM_VERIFY(H && He, "The hybridization matrix is not assembled!");
      He->AddMult(x, b, -1.);
      H->PartMult(tdofs_, x, b);
   }
   else
   {
#ifdef MFEM_USE_MPI
      MFEM_VERIFY(pH.Ptr() && pHe.Ptr(),
                  "The hybridization matrix is not assembled!");
      pH.As<HypreParMatrix>()->EliminateBC(*pHe.As<HypreParMatrix>(), tdofs_, x, b);
#endif //MFEM_USE_MPI
   }
}

void DarcyHybridization::EliminateTraceTrueDofsInRHS(const Vector &x, Vector &b)
{
   EliminateTraceTrueDofsInRHS(GetEssentialTrueDofs(), x, b);
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

void DarcyHybridization::ReduceRHS(const BlockVector &b_t, Vector &b_tr) const
{
   const Operator *tr_cP;
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
   Vector b_rl;
   Array<int> c_dofs;
   Array<int> faces;
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

      GetElementFaces(el, faces);

      // Mult C u + G p
      for (int f = 0; f < faces.Size(); f++)
      {
         int el1, el2;
         fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
         DenseMatrix Ct_l;
         GetCtFaceMatrix(faces[f], el1 != el, Ct_l);

         b_rl.SetSize(Ct_l.Width());
         Ct_l.MultTranspose(u_l, b_rl);

         if (c_bfi_p)
         {
            DenseMatrix G;
            GetGFaceMatrix(faces[f], el1 != el, G);

            G.AddMult(p_l, b_rl);
         }

         c_fes.GetFaceVDofs(faces[f], c_dofs);
         b_r.AddElementVector(c_dofs, b_rl);
      }
   }

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
   MFEM_VERIFY(c_fes.FEColl()->GetContType() !=
               FiniteElementCollection::CONTINUOUS,
               "Continuous trace collections are not supported in projection!");

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
   Vector sol_rl;
   Array<int> c_dofs;
   Array<int> faces;
   Vector bu_l, bp_l, u_l, p_l;
   Array<int> u_vdofs, p_dofs;

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

      GetElementFaces(el, faces);

      // bu - C^T sol
      for (int f = 0; f < faces.Size(); f++)
      {
         int el1, el2;
         fes.GetMesh()->GetFaceElements(faces[f], &el1, &el2);
         DenseMatrix Ct_l;
         GetCtFaceMatrix(faces[f], el1 != el, Ct_l);

         c_fes.GetFaceVDofs(faces[f], c_dofs);
         sol_r.GetSubVector(c_dofs, sol_rl);

         Ct_l.AddMult_a(-1., sol_rl, bu_l);

         //bp - E sol
         if (c_bfi_p)
         {
            DenseMatrix E;
            GetEFaceMatrix(faces[f], el1 != el, E);

            E.AddMult_a(-1., sol_rl, bp_l);
         }
      }

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
   const Vector &sol_u = sol.GetBlock(0);
   const Vector &sol_p = sol.GetBlock(1);

   const FiniteElementSpace &fes_ut = *ut.FESpace();

   MFEM_ASSERT(fes.GetMesh() == fes_ut.GetMesh(),
               "Different meshes are not supported!");

   MFEM_ASSERT(fes.GetMesh()->Conforming(),
               "Non-conforming meshes are not supported!");

   Mesh *mesh = fes_ut.GetMesh();
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = (c_pfes)?(c_pfes->GetParMesh()):(NULL);
   const int NE = mesh->GetNE();
   ParGridFunction pu, pp;
   if (pfes && pfes_p && pmesh)
   {
      pu.MakeRef(const_cast<ParFiniteElementSpace*>(pfes), const_cast<Vector&>(sol_u),
                 0);
      pu.ExchangeFaceNbrData();
      pp.MakeRef(const_cast<ParFiniteElementSpace*>(pfes_p),
                 const_cast<Vector&>(sol_p),
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
   DenseMatrix Ct_l, Ct1, Ct2, Mf, fmat1, fmat2;
   Vector u1, u2, p1, p2, xf, bf, ut_f;
   MassIntegrator fbfi;
   DenseMatrixInverse Mfi;

   auto add_mult_face = [](const DenseMatrix &fmat, const Vector &p,
                           const Vector &tr, Vector &b, real_t s = 1.)
   {
      const int d_size = p.Size();
      const int tr_size = tr.Size();

      for (int i = 0; i < d_size; i++)
         for (int j = 0; j < tr_size; j++)
         {
            b(j) += s * fmat(d_size+j, i) * p(i);
         }

      for (int i = 0; i < d_size; i++)
         for (int j = 0; j < tr_size; j++)
         {
            b(j) += s * fmat(d_size+j, d_size+i) * tr(i);
         }
   };

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
         c_bfi->AssembleFaceMatrix(*fe_c, *fe1, *fe2, *ftr, Ct_l);

         //side 1
         fes.GetElementVDofs(ftr->Elem1No, vdofs1);
         sol_u.GetSubVector(vdofs1, u1);
         Ct1.CopyMN(Ct_l, vdofs1.Size(), vdofs_ut.Size(), 0, 0);
         Ct1.MultTranspose(u1, bf);

         //side 2
         pfes->GetFaceNbrElementVDofs(nbr_el, vdofs2);
         pu.FaceNbrData().GetSubVector(vdofs2, u2);
         Ct2.CopyMN(Ct_l, vdofs2.Size(), vdofs_ut.Size(), vdofs1.Size(), 0);
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
         sol_u.GetSubVector(vdofs1, u1);
         Ct1.MultTranspose(u1, bf);

         //side 2
         if (ftr->Elem2No >= 0)
         {
            fes.GetElementVDofs(ftr->Elem2No, vdofs2);
            sol_u.GetSubVector(vdofs2, u2);
            GetCtFaceMatrix(f, 1, Ct2);
            // here we use the constraint integrator as well, but flip the sign
            // corresponding to the opposite normal for the total flux
            Ct2.AddMultTranspose(u2, bf, -1.);
         }
      }

      //potential constraint

      if (c_bfi_p && ftr->Elem2No >= 0)
      {
         // first side
         fes_p.GetElementVDofs(ftr->Elem1No, dofs1);
         sol_p.GetSubVector(dofs1, p1);
         c_fes.GetFaceVDofs(f, vdofs_xf);
         x.GetSubVector(vdofs_xf, xf);

         const FiniteElement *fe1_p = fes_p.GetFE(ftr->Elem1No);
         const FiniteElement *face_fe = c_fes.GetFaceElement(f);

         if (c_bfi_p)
         {
            c_bfi_p->AssembleHDGFaceMatrix(0, *face_fe, *fe1_p, *ftr, fmat1);
            add_mult_face(fmat1, p1, xf, bf);
         }

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
            sol_p.GetSubVector(dofs2, p2);
            fe2_p = fes_p.GetFE(ftr->Elem2No);
         }

         if (c_bfi_p)
         {
            c_bfi_p->AssembleHDGFaceMatrix(1, *face_fe, *fe2_p, *ftr, fmat2);
            add_mult_face(fmat2, p2, xf, bf, -1.);
         }
      }

      // boundary potential constraint
      if (ftr->Elem2No < 0 && !boundary_constraint_pot_integs.empty())
      {
         const FiniteElement *fe_p = fes_p.GetFE(ftr->Elem1No);
         const FiniteElement *face_fe = c_fes.GetFaceElement(f);

         fes_p.GetElementVDofs(ftr->Elem1No, dofs1);
         sol_p.GetSubVector(dofs1, p1);
         c_fes.GetFaceVDofs(f, vdofs_xf);
         x.GetSubVector(vdofs_xf, xf);

         const int bdr_attr = mesh->GetBdrAttribute(f_2_b[f]);

         // linear
         for (size_t i = 0; i < boundary_constraint_pot_integs.size(); i++)
         {
            if (boundary_constraint_pot_integs_marker[i]
                && (*boundary_constraint_pot_integs_marker[i])[bdr_attr-1] == 0) { continue; }

            boundary_constraint_pot_integs[i]->AssembleHDGFaceMatrix(0, *face_fe, *fe_p,
                                                                     *ftr, fmat1);
            add_mult_face(fmat1, p1, xf, bf);
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
      sol_u.GetSubVector(vdofs, u_z);

      fes_p.GetElementVDofs(z, dofs);
      sol_p.GetSubVector(dofs, p_z);

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
   He.reset();
   pHe.Clear();

   A_empty = true;
   Bf_data = 0.;
   if (Df_data.Size())
   {
      Df_data = 0.;
      D_empty = true;
   }
   Be_data = 0.;
}
}
