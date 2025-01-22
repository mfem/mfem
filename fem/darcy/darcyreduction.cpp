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
{ }

DarcyReduction::~DarcyReduction()
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   if (own_m_nlfi_p) { delete m_nlfi_p; }

   delete[] Af_data;
   delete[] Bf_data;
   delete[] Bf_face_data;
   delete[] D_data;
   delete[] D_face_data;

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

void DarcyReduction::InitA()
{
   const int NE = fes_u->GetNE();

   // Define Af_offsets and Af_f_offsets
   Af_offsets.SetSize(NE+1);
   Af_offsets[0] = 0;
   Af_f_offsets.SetSize(NE+1);
   Af_f_offsets[0] = 0;

   for (int i = 0; i < NE; i++)
   {
      int f_size = fes_u->GetFE(i)->GetDof() * fes_u->GetVDim();
      Af_offsets[i+1] = Af_offsets[i] + f_size*f_size;
      Af_f_offsets[i+1] = Af_f_offsets[i] + f_size;
   }

   if (!m_nlfi_u)
   {
      Af_data = new real_t[Af_offsets[NE]];
   }
}

void DarcyReduction::InitBD()
{
   const int NE = fes_u->GetNE();

   // Define Bf_offsets, D_offsets and D_f_offsets
   Bf_offsets.SetSize(NE+1);
   Bf_offsets[0] = 0;
   D_offsets.SetSize(NE+1);
   D_offsets[0] = 0;
   D_f_offsets.SetSize(NE+1);
   D_f_offsets[0] = 0;

   for (int i = 0; i < NE; i++)
   {
      int f_size = Af_f_offsets[i+1] - Af_f_offsets[i];
      int d_size = fes_p->GetFE(i)->GetDof();
      Bf_offsets[i+1] = Bf_offsets[i] + f_size*d_size;
      D_offsets[i+1] = D_offsets[i] + d_size*d_size;
      D_f_offsets[i+1] = D_f_offsets[i] + d_size;
   }

   Bf_data = new real_t[Bf_offsets[NE]]();//init by zeros
   if (!m_nlfi_p)
   {
      D_data = new real_t[D_offsets[NE]]();//init by zeros
   }
}

void DarcyReduction::InitBFaces()
{
   Mesh *mesh = fes_u->GetMesh();
   const int num_faces = mesh->GetNumFaces();

   // Define Bf_face_offsets and allocate Bf_face_data
   Bf_face_offsets.SetSize(num_faces+1);
   Bf_face_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      int el1, el2;
      mesh->GetFaceElements(f, &el1, &el2);
      if (el2 < 0)
      {
         Bf_face_offsets[f+1] = Bf_face_offsets[f];
         continue;
      }

      int a_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
      int a_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];
      int d_size_1 = D_f_offsets[el1+1] - D_f_offsets[el1];
      int d_size_2 = D_f_offsets[el2+1] - D_f_offsets[el2];

      Bf_face_offsets[f+1] = Bf_face_offsets[f] + a_size_1 * d_size_2 + a_size_2 *
                             d_size_1;
   }

   Bf_face_data = new real_t[Bf_face_offsets[num_faces]]();//init by zeros
}

void DarcyReduction::InitDFaces()
{
   Mesh *mesh = fes_u->GetMesh();
   int num_faces = mesh->GetNumFaces();

   // Define D_face_offsets and allocate D_face_data
   D_face_offsets.SetSize(num_faces+1);
   D_face_offsets[0] = 0;
   for (int f = 0; f < num_faces; f++)
   {
      int el1, el2;
      mesh->GetFaceElements(f, &el1, &el2);
      if (el2 < 0)
      {
         D_face_offsets[f+1] = D_face_offsets[f];
         continue;
      }

      int d_size_1 = D_f_offsets[el1+1] - D_f_offsets[el1];
      int d_size_2 = D_f_offsets[el2+1] - D_f_offsets[el2];

      D_face_offsets[f+1] = D_face_offsets[f] + d_size_1 * d_size_2 * 2;
   }

   D_face_data = new real_t[D_face_offsets[num_faces]]();//init by zeros
}

void DarcyReduction::Init(const Array<int> &)
{
   InitA();
   InitBD();
}

void DarcyReduction::AssembleFluxMassMatrix(int el, const DenseMatrix &A)
{
   const int s = Af_f_offsets[el+1] - Af_f_offsets[el];
   DenseMatrix A_i(Af_data + Af_offsets[el], s, s);
   MFEM_ASSERT(A.Size() == s, "Incompatible sizes");

   A_i = A;
}

void DarcyReduction::AssemblePotMassMatrix(int el, const DenseMatrix &D)
{
   const int s = D_f_offsets[el+1] - D_f_offsets[el];
   DenseMatrix D_i(D_data + D_offsets[el], s, s);
   MFEM_ASSERT(D.Size() == s, "Incompatible sizes");

   D_i += D;
}

void DarcyReduction::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int w = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int h = D_f_offsets[el+1] - D_f_offsets[el];
   DenseMatrix B_i(Bf_data + Bf_offsets[el], h, w);
   MFEM_ASSERT(B.Width() == w && B.Height() == h, "Incompatible sizes");

   B_i += B;
}

void DarcyReduction::AssembleDivFaceMatrix(int face, const DenseMatrix &elmat)
{
   if (!Bf_face_data) { InitBFaces(); }

   int el1, el2;
   fes_p->GetMesh()->GetFaceElements(face, &el1, &el2);

   const int a_dofs_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
   const int d_dofs_size_1 = D_f_offsets[el1+1] - D_f_offsets[el1];

   //B_11
   DenseMatrix B_1(d_dofs_size_1, a_dofs_size_1);
   B_1.CopyMN(elmat, d_dofs_size_1, a_dofs_size_1, 0, 0);
   AssembleDivMatrix(el1, B_1);

   if (el2 >= 0)
   {
      const int a_dofs_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];
      const int d_dofs_size_2 = D_f_offsets[el2+1] - D_f_offsets[el2];

      //B_22
      DenseMatrix B_2(d_dofs_size_2, a_dofs_size_2);
      B_2.CopyMN(elmat, d_dofs_size_2, a_dofs_size_2, d_dofs_size_1, a_dofs_size_1);
      AssembleDivMatrix(el2, B_2);

      //B_12
      DenseMatrix B_12(d_dofs_size_1, a_dofs_size_2);
      B_12.CopyMN(elmat, d_dofs_size_1, a_dofs_size_2, 0, a_dofs_size_1);
      DenseMatrix Bf_f_12(Bf_face_data + Bf_face_offsets[face],
                          d_dofs_size_1, a_dofs_size_2);
      Bf_f_12 += B_12;

      //B_21
      DenseMatrix B_21(d_dofs_size_2, a_dofs_size_1);
      B_21.CopyMN(elmat, d_dofs_size_2, a_dofs_size_1, d_dofs_size_1, 0);
      DenseMatrix Bf_f_21(Bf_face_data + Bf_face_offsets[face] +
                          d_dofs_size_1 * a_dofs_size_2, d_dofs_size_2, a_dofs_size_1);
      Bf_f_21 += B_21;
   }
}

void DarcyReduction::AssemblePotFaceMatrix(int face, const DenseMatrix &elmat)
{
   if (!D_face_data) { InitDFaces(); }

   int el1, el2;
   fes_p->GetMesh()->GetFaceElements(face, &el1, &el2);

   const int ndof1 = fes_p->GetFE(el1)->GetDof();

   //D_11
   DenseMatrix D_1(ndof1);
   D_1.CopyMN(elmat, ndof1, ndof1, 0, 0);
   AssemblePotMassMatrix(el1, D_1);

   if (el2 >= 0)
   {
      const int ndof2 = fes_p->GetFE(el2)->GetDof();

      //D_22
      DenseMatrix D_2(ndof2);
      D_2.CopyMN(elmat, ndof2, ndof2, ndof1, ndof1);
      AssemblePotMassMatrix(el2, D_2);

      //D_12
      DenseMatrix D_12(ndof1, ndof2);
      D_12.CopyMN(elmat, ndof1, ndof2, 0, ndof1);
      DenseMatrix D_f_12(D_face_data + D_face_offsets[face], ndof1, ndof2);
      D_f_12 += D_12;

      //D_21
      DenseMatrix D_21(ndof2, ndof1);
      D_21.CopyMN(elmat, ndof2, ndof1, ndof1, 0);
      DenseMatrix D_f_21(D_face_data + D_face_offsets[face] + ndof1*ndof2,
                         ndof2, ndof1);
      D_f_21 += D_21;
   }
}

void DarcyReduction::Mult(const Vector &x, Vector &y) const
{
   S->Mult(x, y);
}

void DarcyReduction::Finalize()
{
   if (!S) { ComputeS(); }
}

void DarcyReduction::Reset()
{
   delete S;
   S = NULL;

   memset(Bf_data, 0, Bf_offsets.Last() * sizeof(real_t));
   if (Bf_face_data)
   {
      memset(Bf_face_data, 0, Bf_face_offsets.Last() * sizeof(real_t));
   }
   if (D_data)
   {
      memset(D_data, 0, D_offsets.Last() * sizeof(real_t));
   }
   if (D_face_data)
   {
      memset(D_face_data, 0, D_face_offsets.Last() * sizeof(real_t));
   }
}

DarcyFluxReduction::DarcyFluxReduction(FiniteElementSpace *fes_u,
                                       FiniteElementSpace *fes_p, bool bsym)
   : DarcyReduction(fes_u, fes_p, bsym)
{
   width = height = fes_p->GetVSize();
}

DarcyFluxReduction::~DarcyFluxReduction()
{
   delete[] Af_ipiv;
}

void DarcyFluxReduction::Init(const Array<int> &ess_flux_tdof_list)
{
   MFEM_ASSERT(ess_flux_tdof_list.Size() == 0,
               "Essential VDOFs are not supported");

   DarcyReduction::Init(ess_flux_tdof_list);

   const int NE = fes_u->GetNE();
   Af_ipiv = new int[Af_f_offsets[NE]];
}

void DarcyFluxReduction::ComputeS()
{
   MFEM_ASSERT(!m_nlfi_u && !m_nlfi_p,
               "Cannot assemble S matrix in the non-linear regime");

   const int skip_zeros = 1;
   Mesh *mesh = fes_u->GetMesh();
   const int NE = mesh->GetNE();
   const int dim = mesh->Dimension();

   if (!S) { S = new SparseMatrix(fes_p->GetVSize()); }

   DenseMatrix AiBt, AiBt1, S1, S1t, S12;
   Array<int> p_dofs, p_dofs_1, p_dofs_2;
   Array<int> faces, oris;

   for (int el = 0; el < NE; el++)
   {
      const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      const int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];

      DenseMatrix D(D_data + D_offsets[el], d_dofs_size, d_dofs_size);
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);

      // Decompose A
      LUFactors LU_A(Af_data + Af_offsets[el], Af_ipiv + Af_f_offsets[el]);

      LU_A.Factor(a_dofs_size);

      // Schur complement
      AiBt.Transpose(B);
      if (!bsym) { AiBt.Neg(); }
      LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());
      mfem::AddMult(B, AiBt, D);

      fes_p->GetElementDofs(el, p_dofs);

      S->AddSubMatrix(p_dofs, p_dofs, D, skip_zeros);

      // Face contributions

      if (Bf_face_data)
      {
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

         for (int f1 : faces)
         {
            int el1_1, el1_2;
            mesh->GetFaceElements(f1, &el1_1, &el1_2);
            if (el1_2 < 0) { continue; }

            const int el1 = (el1_1 == el)?(el1_2):(el1_1);
            fes_p->GetElementDofs(el1, p_dofs_1);
            const int d_dofs_size_1 = p_dofs_1.Size();

            const int a_dofs_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
            const int B1_off = (el1_1 == el)?(a_dofs_size_1 * d_dofs_size):(0);
            DenseMatrix B1(Bf_face_data + Bf_face_offsets[f1] + B1_off,
                           d_dofs_size_1, a_dofs_size);

            AiBt1.Transpose(B1);
            if (!bsym) { AiBt1.Neg(); }
            LU_A.Solve(AiBt1.Height(), AiBt1.Width(), AiBt1.GetData());

            //B A^-1 B_1^T
            S1t.SetSize(d_dofs_size, d_dofs_size_1);
            mfem::Mult(B, AiBt1, S1t);

            S->AddSubMatrix(p_dofs, p_dofs_1, S1t, skip_zeros);

            //B_1 A^-1 B^T
            S1.SetSize(d_dofs_size_1, d_dofs_size);
            mfem::Mult(B1, AiBt, S1);

            S->AddSubMatrix(p_dofs_1, p_dofs, S1, skip_zeros);

            //B_1 A^-1 B_1^T
            S12.SetSize(d_dofs_size_1, d_dofs_size_1);
            mfem::Mult(B1, AiBt1, S12);

            S->AddSubMatrix(p_dofs_1, p_dofs_1, S12, skip_zeros);

            for (int f2 : faces)
            {
               if (f1 == f2) { continue; }

               int el2_1, el2_2;
               mesh->GetFaceElements(f2, &el2_1, &el2_2);
               if (el2_2 < 0) { continue; }

               const int el2 = (el2_1 == el)?(el2_2):(el2_1);
               fes_p->GetElementDofs(el2, p_dofs_2);
               const int d_dofs_size_2 = p_dofs_2.Size();

               const int a_dofs_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];
               const int B2_off = (el2_1 == el)?(a_dofs_size_2 * d_dofs_size):(0);
               DenseMatrix B2(Bf_face_data + Bf_face_offsets[f2] + B2_off,
                              d_dofs_size_2, a_dofs_size);

               //B_2 A^-1 B_1^T
               S12.SetSize(d_dofs_size_2, d_dofs_size_1);
               mfem::Mult(B2, AiBt1, S12);

               S->AddSubMatrix(p_dofs_2, p_dofs_1, S12, skip_zeros);
            }
         }
      }
   }

   // D face contributions

   if (D_face_data)
   {
      const int nfaces = mesh->GetNumFaces();
      Array<int> p_dofs_1, p_dofs_2;

      for (int f = 0; f < nfaces; f++)
      {
         int el1, el2;
         mesh->GetFaceElements(f, &el1, &el2);
         if (el2 < 0) { continue; }

         fes_p->GetElementDofs(el1, p_dofs_1);
         fes_p->GetElementDofs(el2, p_dofs_2);
         const int d_dofs_size_1 = p_dofs_1.Size();
         const int d_dofs_size_2 = p_dofs_2.Size();

         DenseMatrix D_12(D_face_data + D_face_offsets[f], d_dofs_size_1, d_dofs_size_2);
         S->AddSubMatrix(p_dofs_1, p_dofs_2, D_12, skip_zeros);

         DenseMatrix D_21(D_face_data + D_face_offsets[f] +
                          d_dofs_size_1 * d_dofs_size_2, d_dofs_size_2, d_dofs_size_1);
         S->AddSubMatrix(p_dofs_2, p_dofs_1, D_21, skip_zeros);
      }
   }

   S->Finalize();
}

void DarcyFluxReduction::ReduceRHS(const BlockVector &b, Vector &b_r) const
{
   Mesh *mesh = fes_u->GetMesh();
   const int NE = mesh->GetNE();
   const int dim = mesh->Dimension();
   Vector bu_l, bp_l, bp_f;
   Array<int> u_vdofs, p_dofs, p_dofs_f;
   Array<int> faces, oris;

   const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);

   if (bsym)
   {
      b_r.SetSize(bp.Size());
      b_r.Set(-1., bp);
   }
   else
   {
      b_r = bp;
   }

   for (int el = 0; el < NE; el++)
   {
      // Load RHS

      fes_u->GetElementVDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      fes_p->GetElementDofs(el, p_dofs);

      // -B A^-1 bu

      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);
      LUFactors LU_A(Af_data + Af_offsets[el], Af_ipiv + Af_f_offsets[el]);

      LU_A.Solve(a_dofs_size, 1, bu_l.GetData());
      if (bsym) { bu_l.Neg(); }

      bp_l.SetSize(d_dofs_size);
      B.Mult(bu_l, bp_l);

      b_r.AddElementVector(p_dofs, bp_l);

      // Face contributions

      if (Bf_face_data)
      {
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

         for (int f : faces)
         {
            int el1, el2;
            mesh->GetFaceElements(f, &el1, &el2);
            if (el2 < 0) { continue; }

            const int el_f = (el1 == el)?(el2):(el1);
            fes_p->GetElementDofs(el_f, p_dofs_f);
            const int d_dofs_size_f = p_dofs_f.Size();

            const int a_dofs_size_f = Af_f_offsets[el_f+1] - Af_f_offsets[el_f];
            const int B_off = (el1 == el)?(a_dofs_size_f * d_dofs_size):(0);
            DenseMatrix B_f(Bf_face_data + Bf_face_offsets[f] + B_off,
                            d_dofs_size_f, a_dofs_size);

            // -B_f A^-1 bu

            bp_f.SetSize(d_dofs_size_f);
            B_f.Mult(bu_l, bp_f);

            b_r.AddElementVector(p_dofs_f, bp_f);
         }
      }
   }
}

void DarcyFluxReduction::ComputeSolution(const BlockVector &b,
                                         const Vector &sol_r,
                                         BlockVector &sol) const
{
   Mesh *mesh = fes_u->GetMesh();
   const int NE = mesh->GetNE();
   const int dim = mesh->Dimension();
   Vector bu_l, p_l, p_f;
   Array<int> u_vdofs, p_dofs, p_dofs_f;
   Array<int> faces, oris;

   const Vector &bu = b.GetBlock(0);
   //const Vector &bp = b.GetBlock(1);
   Vector &u = sol.GetBlock(0);
   Vector &p = sol.GetBlock(1);

   p = sol_r;

   for (int el = 0; el < NE; el++)
   {
      //Load RHS

      fes_u->GetElementVDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      fes_p->GetElementDofs(el, p_dofs);
      p.GetSubVector(p_dofs, p_l);

      // bu = R - B^T p

      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);
      LUFactors LU_A(Af_data + Af_offsets[el], Af_ipiv + Af_f_offsets[el]);

      B.AddMultTranspose(p_l, bu_l, (bsym)?(+1.):(-1.));

      // Face contributions

      if (Bf_face_data)
      {
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

         for (int f : faces)
         {
            int el1, el2;
            mesh->GetFaceElements(f, &el1, &el2);
            if (el2 < 0) { continue; }

            const int el_f = (el1 == el)?(el2):(el1);
            fes_p->GetElementDofs(el_f, p_dofs_f);
            const int d_dofs_size_f = p_dofs_f.Size();

            const int a_dofs_size_f = Af_f_offsets[el_f+1] - Af_f_offsets[el_f];
            const int B_off = (el1 == el)?(a_dofs_size_f * d_dofs_size):(0);
            DenseMatrix B_f(Bf_face_data + Bf_face_offsets[f] + B_off,
                            d_dofs_size_f, a_dofs_size);

            // bu -= B_f^T p

            p.GetSubVector(p_dofs_f, p_f);
            B_f.AddMultTranspose(p_f, bu_l, (bsym)?(+1.):(-1.));
         }
      }

      // u = A^-1 bu

      LU_A.Solve(a_dofs_size, 1, bu_l.GetData());
      u.SetSubVector(u_vdofs, bu_l);
   }
}

DarcyPotentialReduction::DarcyPotentialReduction(FiniteElementSpace *fes_u,
                                                 FiniteElementSpace *fes_p, bool bsym)
   : DarcyReduction(fes_u, fes_p, bsym)
{
   width = height = fes_u->GetVSize();
}

DarcyPotentialReduction::~DarcyPotentialReduction()
{
   delete[] Ae_data;
   delete[] Be_data;
   delete[] D_ipiv;
}

void DarcyPotentialReduction::Init(const Array<int> &ess_flux_tdof_list)
{
   const int NE = fes_p->GetNE();

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
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   Ae_offsets.SetSize(NE+1);
   Ae_offsets[0] = 0;
   Be_offsets.SetSize(NE+1);
   Be_offsets[0] = 0;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   for (int i = 0; i < NE; i++)
   {
      int f_size = 0; // count the "free" hat_dofs in element i
      for (int j = hat_offsets[i]; j < hat_offsets[i+1]; j++)
      {
         if (hat_dofs_marker[j] != 1) { f_size++; }
      }
      Af_offsets[i+1] = Af_offsets[i] + f_size*f_size;
      Af_f_offsets[i+1] = Af_f_offsets[i] + f_size;
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
      int a_size = hat_offsets[i+1] - hat_offsets[i];
      int e_size = a_size - f_size;
      int d_size = fes_p->GetFE(i)->GetDof();
      Ae_offsets[i+1] = Ae_offsets[i] + e_size*a_size;
      Be_offsets[i+1] = Be_offsets[i] + e_size*d_size;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   }

   if (!m_nlfi_u)
   {
      Af_data = new real_t[Af_offsets[NE]];
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
      Ae_data = new real_t[Ae_offsets[NE]];
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   }

   InitBD();

#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   Be_data = new real_t[Be_offsets[NE]]();//init by zeros
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   D_ipiv = new int[D_f_offsets[NE]];
}

void DarcyPotentialReduction::GetFDofs(int el, Array<int> &fdofs) const
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

void DarcyPotentialReduction::GetEDofs(int el, Array<int> &edofs) const
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
      int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];

      DenseMatrix A(Af_data + Af_offsets[el], a_dofs_size, a_dofs_size);
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);

      // Decompose D
      LUFactors LU_D(D_data + D_offsets[el], D_ipiv + D_f_offsets[el]);

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

void DarcyPotentialReduction::AssembleFluxMassMatrix(int el,
                                                     const DenseMatrix &A)
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

void DarcyPotentialReduction::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int o = hat_offsets[el];
   const int w = hat_offsets[el+1] - o;
   const int h = D_f_offsets[el+1] - D_f_offsets[el];
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

void DarcyPotentialReduction::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
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
      const int d_size = D_f_offsets[el+1] - D_f_offsets[el];
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
      int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);
      LUFactors LU_D(D_data + D_offsets[el], D_ipiv + D_f_offsets[el]);

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
      int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];
      DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);
      LUFactors LU_D(D_data + D_offsets[el], D_ipiv + D_f_offsets[el]);

      B.AddMult(u_l, bp_l, -1.);

      LU_D.Solve(d_dofs_size, 1, bp_l.GetData());

      p.SetSubVector(p_dofs, bp_l);
   }
}

void DarcyPotentialReduction::Reset()
{
   DarcyReduction::Reset();

#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   const int NE = fes_p->GetNE();
   memset(Be_data, 0, Be_offsets[NE] * sizeof(real_t));
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
}

}
