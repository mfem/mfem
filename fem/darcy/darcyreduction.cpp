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
#include "../pgridfunc.hpp"

namespace mfem
{

DarcyReduction::DarcyReduction(FiniteElementSpace *fes_u_,
                               FiniteElementSpace *fes_p_, bool bsym_)
   : fes_u(*fes_u_), fes_p(*fes_p_), bsym(bsym_)
{
#ifdef MFEM_USE_MPI
   pfes_u = dynamic_cast<ParFiniteElementSpace*>(&fes_u);
   pfes_p = dynamic_cast<ParFiniteElementSpace*>(&fes_p);

   pS.SetType(Operator::Hypre_ParCSR);
#endif //MFEM_USE_MPI
}

DarcyReduction::~DarcyReduction()
{
   if (own_m_nlfi_u) { delete m_nlfi_u; }
   if (own_m_nlfi_p) { delete m_nlfi_p; }
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
   const int NE = fes_u.GetNE();

   // Define Af_offsets and Af_f_offsets
   Af_offsets.SetSize(NE+1);
   Af_offsets[0] = 0;
   Af_f_offsets.SetSize(NE+1);
   Af_f_offsets[0] = 0;

   for (int i = 0; i < NE; i++)
   {
      int f_size = fes_u.GetFE(i)->GetDof() * fes_u.GetVDim();
      Af_offsets[i+1] = Af_offsets[i] + f_size*f_size;
      Af_f_offsets[i+1] = Af_f_offsets[i] + f_size;
   }

   if (!m_nlfi_u)
   {
      Af_data.SetSize(Af_offsets[NE]);
   }
}

void DarcyReduction::InitBD()
{
   const int NE = fes_u.GetNE();

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
      int d_size = fes_p.GetFE(i)->GetDof() * fes_p.GetVDim();
      Bf_offsets[i+1] = Bf_offsets[i] + f_size*d_size;
      D_offsets[i+1] = D_offsets[i] + d_size*d_size;
      D_f_offsets[i+1] = D_f_offsets[i] + d_size;
   }

   Bf_data.SetSize(Bf_offsets[NE]); Bf_data = 0.;
   if (!m_nlfi_p)
   {
      D_data.SetSize(D_offsets[NE]); D_data = 0.;
   }
}

void DarcyReduction::InitBFaces()
{
   Mesh *mesh = fes_u.GetMesh();
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
         Bf_face_offsets[f+1] = 0;
         continue;
      }

      int a_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
      int a_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];
      int d_size_1 = D_f_offsets[el1+1] - D_f_offsets[el1];
      int d_size_2 = D_f_offsets[el2+1] - D_f_offsets[el2];

      Bf_face_offsets[f+1] = a_size_1 * d_size_2 + a_size_2 * d_size_1;
   }

#ifdef MFEM_USE_MPI
   CountBSharedFaces(Bf_face_offsets);
#endif //MFEM_USE_MPI

   Bf_face_offsets.PartialSum();

   Bf_face_data.SetSize(Bf_face_offsets[num_faces]); Bf_face_data = 0.;
}

void DarcyReduction::InitDFaces()
{
   Mesh *mesh = fes_u.GetMesh();
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
         D_face_offsets[f+1] = 0;
         continue;
      }

      int d_size_1 = D_f_offsets[el1+1] - D_f_offsets[el1];
      int d_size_2 = D_f_offsets[el2+1] - D_f_offsets[el2];

      D_face_offsets[f+1] = d_size_1 * d_size_2 * 2;
   }

#ifdef MFEM_USE_MPI
   CountDSharedFaces(D_face_offsets);
#endif //MFEM_USE_MPI

   D_face_offsets.PartialSum();

   D_face_data.SetSize(D_face_offsets[num_faces]); D_face_data = 0.;
}

void DarcyReduction::Init(const Array<int> &)
{
   InitA();
   InitBD();
}

void DarcyReduction::AssembleFluxMassMatrix(int el, const DenseMatrix &A)
{
   const int s = Af_f_offsets[el+1] - Af_f_offsets[el];
   DenseMatrix A_i(&Af_data[Af_offsets[el]], s, s);
   MFEM_ASSERT(A.Size() == s, "Incompatible sizes");

   A_i = A;
}

void DarcyReduction::AssemblePotMassMatrix(int el, const DenseMatrix &D)
{
   const int s = D_f_offsets[el+1] - D_f_offsets[el];
   DenseMatrix D_i(&D_data[D_offsets[el]], s, s);
   MFEM_ASSERT(D.Size() == s, "Incompatible sizes");

   D_i += D;
}

void DarcyReduction::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int w = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int h = D_f_offsets[el+1] - D_f_offsets[el];
   DenseMatrix B_i(&Bf_data[Bf_offsets[el]], h, w);
   MFEM_ASSERT(B.Width() == w && B.Height() == h, "Incompatible sizes");

   B_i += B;
}

void DarcyReduction::AssembleDivFaceMatrix(int face, const DenseMatrix &elmat)
{
   if (!Bf_face_data.Size()) { InitBFaces(); }

   int el1, el2;
   fes_p.GetMesh()->GetFaceElements(face, &el1, &el2);

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
      DenseMatrix Bf_f_12(&Bf_face_data[Bf_face_offsets[face]],
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
   if (!D_face_data.Size()) { InitDFaces(); }

   int el1, el2;
   fes_p.GetMesh()->GetFaceElements(face, &el1, &el2);

   const int ndof1 = fes_p.GetFE(el1)->GetDof() * fes_p.GetVDim();

   //D_11
   DenseMatrix D_1(ndof1);
   D_1.CopyMN(elmat, ndof1, ndof1, 0, 0);
   AssemblePotMassMatrix(el1, D_1);

   if (el2 >= 0)
   {
      const int ndof2 = fes_p.GetFE(el2)->GetDof() * fes_p.GetVDim();

      //D_22
      DenseMatrix D_2(ndof2);
      D_2.CopyMN(elmat, ndof2, ndof2, ndof1, ndof1);
      AssemblePotMassMatrix(el2, D_2);

      //D_12
      DenseMatrix D_12(ndof1, ndof2);
      D_12.CopyMN(elmat, ndof1, ndof2, 0, ndof1);
      DenseMatrix D_f_12(&D_face_data[D_face_offsets[face]], ndof1, ndof2);
      D_f_12 += D_12;

      //D_21
      DenseMatrix D_21(ndof2, ndof1);
      D_21.CopyMN(elmat, ndof2, ndof1, ndof1, 0);
      DenseMatrix D_f_21(&D_face_data[D_face_offsets[face] + ndof1*ndof2],
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
#ifndef MFEM_USE_MPI
   if (!S) { ComputeS(); }
#else
   if (!S && !pS.Ptr()) { ComputeS(); }
#endif
}

void DarcyReduction::Reset()
{
   S.reset();
#ifdef MFEM_USE_MPI
   pS.Clear();
#endif

   Bf_data = 0.;
   Bf_face_data = 0.;
   D_data = 0.;
   D_face_data = 0.;
}

#ifdef MFEM_USE_MPI
HypreParMatrix *DarcyReduction::ConstructParMatrix(SparseMatrix *spmat,
                                                   ParFiniteElementSpace *pfes_tr, ParFiniteElementSpace *pfes_te)
{
   HYPRE_BigInt num_rows = spmat->Height();
   const int num_face_dofs = pfes_tr->GetVSize();
   Array<HYPRE_BigInt> rows;
   if (pfes_te)
   {
      rows.MakeRef(pfes_te->GetDofOffsets(), pfes_te->GetNRanks()+1);
   }
   else
   {
      Array<HYPRE_BigInt> *offsets[1] = { &rows };
      const ParMesh *pmesh = pfes_tr->GetParMesh();
      pmesh->GenerateOffsets(1, &num_rows, offsets);
   }
   HYPRE_BigInt ldof_offset = pfes_tr->GetMyDofOffset();
   const HYPRE_BigInt *face_nbr_glob_ldof = pfes_tr->GetFaceNbrGlobalDofMap();
   Array<HYPRE_BigInt> hJ(spmat->NumNonZeroElems());
   int *J = spmat->GetJ();
   for (int i = 0; i < hJ.Size(); i++)
   {
      hJ[i] = J[i] < num_face_dofs ?
              J[i] + ldof_offset :
              face_nbr_glob_ldof[J[i] - num_face_dofs];
   }

   return new HypreParMatrix(pfes_tr->GetComm(), spmat->Height(),
                             rows.Last(), pfes_tr->GlobalVSize(),
                             spmat->GetI(), hJ.GetData(), spmat->GetData(),
                             rows, pfes_tr->GetDofOffsets());
}
#endif //MFEM_USE_MPI

DarcyFluxReduction::DarcyFluxReduction(FiniteElementSpace *fes_u_,
                                       FiniteElementSpace *fes_p_, bool bsym)
   : DarcyReduction(fes_u_, fes_p_, bsym)
{
   width = height = fes_p.GetVSize();
}

DarcyFluxReduction::~DarcyFluxReduction()
{
}

void DarcyFluxReduction::Init(const Array<int> &ess_flux_tdof_list)
{
   MFEM_ASSERT(ess_flux_tdof_list.Size() == 0,
               "Essential VDOFs are not supported");

   DarcyReduction::Init(ess_flux_tdof_list);
#ifdef MFEM_USE_MPI
   if (Parallel()) { pfes_p->ExchangeFaceNbrData(); }
   InitDNbr();
#endif

   const int NE = fes_u.GetNE();
   Af_ipiv.SetSize(Af_f_offsets[NE]);
}

#ifdef MFEM_USE_MPI
void DarcyFluxReduction::InitDNbr()
{
   if (!Parallel()) { return; }

   const ParMesh *pmesh = pfes_p->GetParMesh();
   const int NE = pmesh->GetNE();
   const int NENbr = pmesh->GetNFaceNeighborElements();

   // Define Df_f_offsets for neighbors
   D_f_offsets.SetSize(NE + NENbr + 1);

   for (int i = 0; i < NENbr; i++)
   {
      int d_size = pfes_p->GetFaceNbrFE(i)->GetDof() * fes_p.GetVDim();
      D_f_offsets[NE+i+1] = D_f_offsets[NE+i] + d_size;
   }
}

void DarcyFluxReduction::CountBSharedFaces(Array<int> &face_offs) const
{
   if (!Parallel()) { return; }

   const ParMesh *pmesh = pfes_p->GetParMesh();
   const int NE = pfes_p->GetNE();

   const int num_shared_faces = pmesh->GetNSharedFaces();
   for (int sf = 0; sf < num_shared_faces; sf++)
   {
      int e1, e2;
      const int f = pmesh->GetSharedFace(sf);
      pmesh->GetFaceElements(f, &e1, &e2);
      MFEM_ASSERT(e2 < 0, "");
      const int e2nbr = -e2 - 1 + NE;

      const int a_size_1 = Af_f_offsets[e1+1] - Af_f_offsets[e1];
      const int d_size_2 = D_f_offsets[e2nbr+1] - D_f_offsets[e2nbr];

      face_offs[f+1] = a_size_1 * d_size_2;
   }
}

void DarcyFluxReduction::CountDSharedFaces(Array<int> &face_offs) const
{
   if (!Parallel()) { return; }

   const ParMesh *pmesh = pfes_p->GetParMesh();
   const int NE = pfes_p->GetNE();

   const int num_shared_faces = pmesh->GetNSharedFaces();
   for (int sf = 0; sf < num_shared_faces; sf++)
   {
      int e1, e2;
      const int f = pmesh->GetSharedFace(sf);
      pmesh->GetFaceElements(f, &e1, &e2);
      MFEM_ASSERT(e2 < 0, "");
      const int e2nbr = -e2 - 1 + NE;

      const int d_size_1 = D_f_offsets[e1+1] - D_f_offsets[e1];
      const int d_size_2 = D_f_offsets[e2nbr+1] - D_f_offsets[e2nbr];

      face_offs[f+1] = d_size_1 * d_size_2;
   }
}

void DarcyFluxReduction::AssembleDivSharedFaceMatrix(int sface,
                                                     const DenseMatrix &elmat)
{
   if (!Bf_face_data.Size()) { InitBFaces(); }

   const int face = pfes_p->GetParMesh()->GetSharedFace(sface);

   int el1, el2;
   fes_p.GetMesh()->GetFaceElements(face, &el1, &el2);
   MFEM_ASSERT(el2 < 0, "");
   const int NE = fes_p.GetMesh()->GetNE();
   const int el2nbr = -el2 - 1 + NE;

   const int a_dofs_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
   const int d_dofs_size_1 = D_f_offsets[el1+1] - D_f_offsets[el1];
   const int d_dofs_size_2 = D_f_offsets[el2nbr+1] - D_f_offsets[el2nbr];

   //B_11
   DenseMatrix B_1(d_dofs_size_1, a_dofs_size_1);
   B_1.CopyMN(elmat, d_dofs_size_1, a_dofs_size_1, 0, 0);
   AssembleDivMatrix(el1, B_1);

   //B_21
   DenseMatrix B_21(d_dofs_size_2, a_dofs_size_1);
   B_21.CopyMN(elmat, d_dofs_size_2, a_dofs_size_1, d_dofs_size_1, 0);
   DenseMatrix Bf_f_21(&Bf_face_data[Bf_face_offsets[face]], d_dofs_size_2,
                       a_dofs_size_1);
   Bf_f_21 += B_21;
}

void DarcyFluxReduction::AssemblePotSharedFaceMatrix(int sface,
                                                     const DenseMatrix &elmat)
{
   if (!D_face_data.Size()) { InitDFaces(); }

   const int face = pfes_p->GetParMesh()->GetSharedFace(sface);

   int el1, el2;
   fes_p.GetMesh()->GetFaceElements(face, &el1, &el2);
   MFEM_ASSERT(el2 < 0, "");
   const int NE = fes_p.GetMesh()->GetNE();
   const int el2nbr = -el2 - 1 + NE;

   const int ndof1 = fes_p.GetFE(el1)->GetDof() * fes_p.GetVDim();
   const int ndof2 = pfes_p->GetFaceNbrFE(el2nbr - NE)->GetDof() * fes_p.GetVDim();

   //D_11
   DenseMatrix D_1(ndof1);
   D_1.CopyMN(elmat, ndof1, ndof1, 0, 0);
   AssemblePotMassMatrix(el1, D_1);

   //D_12
   DenseMatrix D_12(ndof1, ndof2);
   D_12.CopyMN(elmat, ndof1, ndof2, 0, ndof1);
   DenseMatrix D_f_12(&D_face_data[D_face_offsets[face]], ndof1, ndof2);
   D_f_12 += D_12;
}

int DarcyFluxReduction::GetFaceNbrVDofs(int el, Array<int> &vdofs,
                                        bool adjust_vdofs) const
{
   MFEM_ASSERT(el < 0, "Not a face neighbor");
   const int NE = pfes_p->GetNE();
   pfes_p->GetFaceNbrElementVDofs(-1 - el, vdofs);
   if (adjust_vdofs)
   {
      const int nvdofs = pfes_p->GetVSize();
      for (int &vdof : vdofs)
      {
         if (vdof >= 0)
         {
            vdof += nvdofs;
         }
         else
         {
            vdof -= nvdofs;
         }
      }
   }
   return -1 - el + NE;
}
#endif // MFEM_USE_MPI

void DarcyFluxReduction::ComputeS()
{
   MFEM_ASSERT(!m_nlfi_u && !m_nlfi_p,
               "Cannot assemble S matrix in the non-linear regime");

   const int skip_zeros = 1;
   Mesh *mesh = fes_u.GetMesh();
   const int NE = mesh->GetNE();

   const int num_face_dofs = fes_p.GetVSize();
#ifdef MFEM_USE_MPI
   const int num_nbr_face_dofs = (Parallel() && (Bf_face_data.Size() ||
                                                 D_face_data.Size()))?
                                 (pfes_p->GetFaceNbrVSize()):(0);
#else
   const int num_nbr_face_dofs = 0;
#endif
   if (Bf_face_data.Size())
   {
      sBt.reset(new SparseMatrix(fes_u.GetVSize(),
                                 num_face_dofs + num_nbr_face_dofs));
      sAiBt.reset(new SparseMatrix(fes_u.GetVSize(),
                                   num_face_dofs + num_nbr_face_dofs));
   }
   if (!S) { S.reset(new SparseMatrix(num_face_dofs, num_face_dofs + num_nbr_face_dofs)); }

   DenseMatrix AiBt;
   Array<int> p_dofs, p_dofs_1, p_dofs_2, u_vdofs, u_vdofs_1, u_vdofs_2;

   for (int el = 0; el < NE; el++)
   {
      const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      const int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];

      DenseMatrix D(&D_data[D_offsets[el]], d_dofs_size, d_dofs_size);
      const DenseMatrix B(&Bf_data[Bf_offsets[el]], d_dofs_size, a_dofs_size);

      fes_p.GetElementVDofs(el, p_dofs);

      // Decompose A
      LUFactors LU_A(&Af_data[Af_offsets[el]], &Af_ipiv[Af_f_offsets[el]]);

      LU_A.Factor(a_dofs_size);

      // Schur complement

      //B^T
      AiBt.Transpose(B);
      if (sBt)
      {
         fes_u.GetElementVDofs(el, u_vdofs);
         sBt->AddSubMatrix(u_vdofs, p_dofs, AiBt, skip_zeros);
      }

      //A^-1 B^T
      if (!bsym) { AiBt.Neg(); }
      LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());

      if (sAiBt)
      {
         //S = D
         sAiBt->AddSubMatrix(u_vdofs, p_dofs, AiBt, skip_zeros);
      }
      else
      {
         //S = D + B A^-1 B^T
         mfem::AddMult(B, AiBt, D);
      }
      S->AddSubMatrix(p_dofs, p_dofs, D, skip_zeros);
   }

   // Face contributions

   if (Bf_face_data.Size() || D_face_data.Size())
   {
      const int nfaces = mesh->GetNumFaces();
      Array<int> p_dofs_1, p_dofs_2;

      for (int f = 0; f < nfaces; f++)
      {
         int el1, el2, inf1, inf2;
         mesh->GetFaceElements(f, &el1, &el2);
         mesh->GetFaceInfos(f, &inf1, &inf2);
         if (Parallel())
         {
#ifdef MFEM_USE_MPI
            if (el2 < 0)
            {
               if (inf2 < 0) { continue; }
               GetFaceNbrVDofs(el2, p_dofs_2);
            }
#endif
         }
         else if (el2 < 0) { continue; }

         fes_p.GetElementVDofs(el1, p_dofs_1);
         if (el2 >= 0)
         {
            fes_p.GetElementVDofs(el2, p_dofs_2);
         }

         const int d_dofs_size_1 = p_dofs_1.Size();
         const int d_dofs_size_2 = p_dofs_2.Size();

         if (Bf_face_data.Size())
         {
            fes_u.GetElementVDofs(el1, u_vdofs_1);

            int B_off = 0;
            if (el2 >= 0)
            {
               fes_u.GetElementVDofs(el2, u_vdofs_2);

               const int a_dofs_size_2 = Af_f_offsets[el2+1] - Af_f_offsets[el2];
               B_off = d_dofs_size_1 * a_dofs_size_2;
               const DenseMatrix B1(&Bf_face_data[Bf_face_offsets[f]],
                                    d_dofs_size_1, a_dofs_size_2);

               LUFactors LU_A2(&Af_data[Af_offsets[el2]], &Af_ipiv[Af_f_offsets[el2]]);

               //B_1^T
               AiBt.Transpose(B1);
               sBt->AddSubMatrix(u_vdofs_2, p_dofs_1, AiBt, skip_zeros);

               //A_2^-1 B_1^T
               if (!bsym) { AiBt.Neg(); }
               LU_A2.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());
               sAiBt->AddSubMatrix(u_vdofs_2, p_dofs_1, AiBt, skip_zeros);
            }

            const int a_dofs_size_1 = Af_f_offsets[el1+1] - Af_f_offsets[el1];
            const DenseMatrix B2(&Bf_face_data[Bf_face_offsets[f] + B_off],
                                 d_dofs_size_2, a_dofs_size_1);

            LUFactors LU_A1(&Af_data[Af_offsets[el1]], &Af_ipiv[Af_f_offsets[el1]]);

            //B_2^T
            AiBt.Transpose(B2);
            sBt->AddSubMatrix(u_vdofs_1, p_dofs_2, AiBt, skip_zeros);

            //A_1^-1 B_2^T
            if (!bsym) { AiBt.Neg(); }
            LU_A1.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());
            sAiBt->AddSubMatrix(u_vdofs_1, p_dofs_2, AiBt, skip_zeros);
         }

         if (D_face_data.Size())
         {
            const DenseMatrix D_12(&D_face_data[D_face_offsets[f]],
                                   d_dofs_size_1, d_dofs_size_2);
            S->AddSubMatrix(p_dofs_1, p_dofs_2, D_12, skip_zeros);

            if (el2 >= 0)
            {
               const DenseMatrix D_21(&D_face_data[D_face_offsets[f]
                                                   + d_dofs_size_1 * d_dofs_size_2],
                                      d_dofs_size_2, d_dofs_size_1);
               S->AddSubMatrix(p_dofs_2, p_dofs_1, D_21, skip_zeros);
            }
         }
      }
   }

   S->Finalize(skip_zeros);

   if (!Parallel())
   {
      if (Bf_face_data.Size())
      {
         sBt->Finalize(skip_zeros);
         sAiBt->Finalize(skip_zeros);
         SparseMatrix *sBAiBt = TransposeMult(*sBt, *sAiBt);
         sAiBt.reset();// not needed anymore
         S.reset(Add(*S, *sBAiBt));
         delete sBAiBt;
      }

      const SparseMatrix *cP = fes_p.GetConformingProlongation();
      if (cP)
      {
         S.reset(mfem::RAP(*cP, *S, *cP));
      }
   }
   else // parallel
   {
#ifdef MFEM_USE_MPI
      OperatorHandle dS(pS.Type()), pP(pS.Type());
      std::unique_ptr<HypreParMatrix> hS;
      if (D_face_data.Size())
      {
         hS.reset(ConstructParMatrix(S.get(), pfes_p, pfes_p));
         dS.ConvertFrom(hS.get());
      }
      else
      {
         dS.MakeSquareBlockDiag(pfes_p->GetComm(), pfes_p->GlobalVSize(),
                                pfes_p->GetDofOffsets(), S.get());
      }

      if (Bf_face_data.Size())
      {
         // Convert Bt to parallel
         sBt->Finalize(skip_zeros);
         std::unique_ptr<HypreParMatrix> hBt(ConstructParMatrix(sBt.get(), pfes_p,
                                                                pfes_u));
         sBt.reset();

         // B
         hB.reset(hBt->Transpose());
         hBt.reset();

         // Convert AiBt to parallel
         sAiBt->Finalize(skip_zeros);
         std::unique_ptr<HypreParMatrix> hAiBt(ConstructParMatrix(sAiBt.get(), pfes_p,
                                                                  pfes_u));
         sAiBt.reset();

         // B A^-1 B^T
         std::unique_ptr<HypreParMatrix> hBAiBt(ParMult(hB.get(), hAiBt.get()));
         hAiBt.reset();

         // D + B A^-1 B^T
         if (!hS)
         {
            // Only works for HyperParMatrix for now
            hS.reset(dS.Is<HypreParMatrix>());
            dS.SetOperatorOwner(false);
         }
         dS.Reset(ParAdd(hS.get(), hBAiBt.get()));
         hS.reset();
         hBAiBt.reset();
      }

      // TODO - construct Dof_TrueDof_Matrix directly in the pS format
      pP.ConvertFrom(pfes_p->Dof_TrueDof_Matrix());
      pS.MakePtAP(dS, pP);
      dS.Clear();
      S.reset();
#endif
   }
}

void DarcyFluxReduction::ReduceRHS(const BlockVector &b, Vector &b_tr) const
{
   Mesh *mesh = fes_u.GetMesh();
   const int NE = mesh->GetNE();
   Vector bu_l, bp_l;
   Array<int> u_vdofs, p_dofs;

   const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);

   Vector b_r;
   const Operator *tr_cP;

   if (!Parallel() && !(tr_cP = fes_p.GetConformingProlongation()))
   {
      b_tr.SetSize(fes_p.GetVSize());
      b_r.SetDataAndSize(b_tr.GetData(), b_tr.Size());
   }
   else
   {
      b_r.SetSize(fes_p.GetVSize());
   }

   if (bsym)
   {
      b_r.Set(-1., bp);
   }
   else
   {
      b_r = bp;
   }

   Vector Aibu;
   if (Bf_face_data.Size())
   {
      Aibu.SetSize(fes_u.GetVSize());
   }

   for (int el = 0; el < NE; el++)
   {
      // Load RHS

      fes_u.GetElementVDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      fes_p.GetElementVDofs(el, p_dofs);

      // -A^-1 bu

      const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      const LUFactors LU_A(const_cast<real_t*>(&Af_data[Af_offsets[el]]),
                           const_cast<int*>(&Af_ipiv[Af_f_offsets[el]]));

      LU_A.Solve(a_dofs_size, 1, bu_l.GetData());
      if (bsym) { bu_l.Neg(); }

      if (Aibu.Size() > 0)
      {
         Aibu.SetSubVector(u_vdofs, bu_l);
         continue;
      }

      // -B A^-1 bu

      const int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];
      const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                          d_dofs_size, a_dofs_size);

      bp_l.SetSize(d_dofs_size);
      B.Mult(bu_l, bp_l);

      b_r.AddElementVector(p_dofs, bp_l);
   }

   if (!Parallel())
   {
      if (Bf_face_data.Size())
      {
         MFEM_ASSERT(sBt, "Global B matrix is not assembled");
         sBt->AddMultTranspose(Aibu, b_r);
      }

      if (tr_cP)
      {
         b_tr.SetSize(tr_cP->Width());
         tr_cP->MultTranspose(b_r, b_tr);
      }
   }
   else
   {
#ifdef MFEM_USE_MPI
      MFEM_ASSERT(hB, "Parallel B matrix is not assembled");
      if (hB)
      {
         hB->AddMult(Aibu, b_r);
      }
#endif
      const Operator *tr_P = fes_p.GetProlongationMatrix();
      b_tr.SetSize(tr_P->Width());
      tr_P->MultTranspose(b_r, b_tr);
   }
}

void DarcyFluxReduction::ComputeSolution(const BlockVector &b,
                                         const Vector &sol_tr,
                                         BlockVector &sol) const
{
   Mesh *mesh = fes_u.GetMesh();
   const int NE = mesh->GetNE();
   Vector bu_l, p_l, p_f;
   Array<int> u_vdofs, p_dofs;

   //const Vector &bp = b.GetBlock(1);
   Vector &u = sol.GetBlock(0);
   Vector &p = sol.GetBlock(1);

   Vector sol_r;
   if (!Parallel())
   {
      const SparseMatrix *tr_cP = fes_p.GetConformingProlongation();
      if (!tr_cP)
      {
         sol_r.SetDataAndSize(sol_tr.GetData(), sol_tr.Size());
      }
      else
      {
         sol_r.SetSize(fes_p.GetVSize());
         tr_cP->Mult(sol_tr, sol_r);
      }
   }
   else
   {
      sol_r.SetSize(fes_p.GetVSize());
      fes_p.GetProlongationMatrix()->Mult(sol_tr, sol_r);
   }

   p = sol_r;

   Vector bu;
   if (Bf_face_data.Size())
   {
      bu = b.GetBlock(0);
      if (Parallel())
      {
#ifdef MFEM_USE_MPI
         MFEM_ASSERT(hB, "Parallel B matrix is not assembled");
         const Operator *R = pfes_p->GetRestrictionOperator();
         Vector tp(R->Height());
         R->Mult(p, tp);
         hB->AddMultTranspose(tp, bu, (bsym)?(+1.):(-1.));
#endif
      }
      else
      {
         MFEM_ASSERT(sBt, "Global B matrix is not assembled");
         sBt->AddMult(p, bu, (bsym)?(+1.):(-1.));
      }
   }
   else
   {
      bu.MakeRef(const_cast<Vector&>(b.GetBlock(0)), 0, fes_u.GetVSize());
   }

   for (int el = 0; el < NE; el++)
   {
      //Load RHS

      fes_u.GetElementVDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      if (!Bf_face_data.Size())
      {
         fes_p.GetElementVDofs(el, p_dofs);
         p.GetSubVector(p_dofs, p_l);

         // bu = R - B^T p

         int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
         int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];
         const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]), d_dofs_size,
                             a_dofs_size);

         B.AddMultTranspose(p_l, bu_l, (bsym)?(+1.):(-1.));
      }

      // u = A^-1 bu

      const LUFactors LU_A(const_cast<real_t*>(&Af_data[Af_offsets[el]]),
                           const_cast<int*>(&Af_ipiv[Af_f_offsets[el]]));

      LU_A.Solve(u_vdofs.Size(), 1, bu_l.GetData());
      u.SetSubVector(u_vdofs, bu_l);
   }
}

DarcyPotentialReduction::DarcyPotentialReduction(FiniteElementSpace *fes_u_,
                                                 FiniteElementSpace *fes_p_, bool bsym)
   : DarcyReduction(fes_u_, fes_p_, bsym)
{
   width = height = fes_u.GetVSize();
}

DarcyPotentialReduction::~DarcyPotentialReduction()
{
}

void DarcyPotentialReduction::Init(const Array<int> &ess_flux_tdof_list)
{
   const int NE = fes_p.GetNE();

   // count the number of dofs in the discontinuous version of fes:
   Array<int> vdofs;
   int num_hat_dofs = 0;
   hat_offsets.SetSize(NE+1);
   hat_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      fes_u.GetElementVDofs(i, vdofs);
      num_hat_dofs += vdofs.Size();
      hat_offsets[i+1] = num_hat_dofs;
   }

   // Define the "free" (0) and "essential" (1) hat_dofs.
   // The "essential" hat_dofs are those that depend only on essential cdofs;
   // all other hat_dofs are "free".
   hat_dofs_marker.SetSize(num_hat_dofs);
   Array<int> free_tdof_marker;
   if (Parallel())
   {
#ifdef MFEM_USE_MPI
      free_tdof_marker.SetSize(pfes_u->TrueVSize());
#endif
   }
   else
   {
      free_tdof_marker.SetSize(fes_u.GetConformingVSize());
   }

   free_tdof_marker = 1;
   for (int i = 0; i < ess_flux_tdof_list.Size(); i++)
   {
      free_tdof_marker[ess_flux_tdof_list[i]] = 0;
   }

   Array<int> free_vdofs_marker;
   if (Parallel())
   {
#ifdef MFEM_USE_MPI
      HypreParMatrix *P = pfes_u->Dof_TrueDof_Matrix();
      free_vdofs_marker.SetSize(fes_u.GetVSize());
      P->BooleanMult(1, free_tdof_marker, 0, free_vdofs_marker);
#endif
   }
   else
   {
      const SparseMatrix *cP = fes_u.GetConformingProlongation();
      if (!cP)
      {
         free_vdofs_marker.MakeRef(free_tdof_marker);
      }
      else
      {
         free_vdofs_marker.SetSize(fes_u.GetVSize());
         cP->BooleanMult(free_tdof_marker, free_vdofs_marker);
      }
   }

   for (int i = 0; i < NE; i++)
   {
      fes_u.GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         hat_dofs_marker[hat_offsets[i]+j] = ! free_vdofs_marker[vdofs[j]];
      }
   }

   free_vdofs_marker.DeleteAll();

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
      int d_size = fes_p.GetFE(i)->GetDof() * fes_p.GetVDim();
      Ae_offsets[i+1] = Ae_offsets[i] + e_size*a_size;
      Be_offsets[i+1] = Be_offsets[i] + e_size*d_size;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   }

   if (!m_nlfi_u)
   {
      Af_data.SetSize(Af_offsets[NE]);
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
      Ae_data.SetSize(Ae_offsets[NE]);
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   }

   InitBD();

#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   Be_data.SetSize(Be_offsets[NE]); Be_data = 0.;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
   D_ipiv.SetSize(D_f_offsets[NE]);
}

void DarcyPotentialReduction::GetFDofs(int el, Array<int> &fdofs) const
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   Array<int> vdofs;
   fes_u.GetElementVDofs(el, vdofs);
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
   fes_u.GetElementVDofs(el, vdofs);
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
   const int NE = fes_u.GetNE();

   if (!S) { S.reset(new SparseMatrix(fes_u.GetVSize())); }

   DenseMatrix DiB;
   Array<int> a_dofs;

   for (int el = 0; el < NE; el++)
   {
      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];

      DenseMatrix A(&Af_data[Af_offsets[el]], a_dofs_size, a_dofs_size);
      const DenseMatrix B(&Bf_data[Bf_offsets[el]], d_dofs_size, a_dofs_size);

      // Decompose D
      LUFactors LU_D(&D_data[D_offsets[el]], &D_ipiv[D_f_offsets[el]]);

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

   S->Finalize(skip_zeros);

   if (!Parallel())
   {
      const SparseMatrix *cP = fes_u.GetConformingProlongation();
      if (cP)
      {
         if (S->Height() != cP->Width())
         {
            S.reset(mfem::RAP(*cP, *S, *cP));
         }
      }
   }
   else // parallel
   {
#ifdef MFEM_USE_MPI
      OperatorHandle dS(pS.Type()), pP(pS.Type());
      dS.MakeSquareBlockDiag(pfes_u->GetComm(), pfes_u->GlobalVSize(),
                             pfes_u->GetDofOffsets(), S.get());

      // TODO - construct Dof_TrueDof_Matrix directly in the pS format
      pP.ConvertFrom(pfes_u->Dof_TrueDof_Matrix());
      pS.MakePtAP(dS, pP);
      dS.Clear();
      S.reset();
#endif
   }
}

void DarcyPotentialReduction::AssembleFluxMassMatrix(int el,
                                                     const DenseMatrix &A)
{
   const int o = hat_offsets[el];
   const int s = hat_offsets[el+1] - o;
   int Af_el_idx = Af_offsets[el];
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   int Ae_el_idx = Ae_offsets[el];
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   for (int j = 0; j < s; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
         for (int i = 0; i < s; i++)
         {
            Ae_data[Ae_el_idx++] = A(i, j);
         }
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < s; i++)
      {
         if (hat_dofs_marker[o + i] == 1) { continue; }
         Af_data[Af_el_idx++] = A(i, j);
      }
   }
   MFEM_ASSERT(Af_el_idx == Af_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   MFEM_ASSERT(Ae_el_idx == Ae_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
}

void DarcyPotentialReduction::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int o = hat_offsets[el];
   const int w = hat_offsets[el+1] - o;
   const int h = D_f_offsets[el+1] - D_f_offsets[el];
   int Bf_el_idx = Bf_offsets[el];
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   int Be_el_idx = Be_offsets[el];
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS

   for (int j = 0; j < w; j++)
   {
      if (hat_dofs_marker[o + j] == 1)
      {
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
         for (int i = 0; i < h; i++)
         {
            Be_data[Be_el_idx++] += B(i, j);
         }
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
         continue;
      }
      for (int i = 0; i < h; i++)
      {
         Bf_data[Bf_el_idx++] += B(i, j);
      }
   }
   MFEM_ASSERT(Bf_el_idx == Bf_offsets[el+1], "Internal error");
#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   MFEM_ASSERT(Be_el_idx == Be_offsets[el+1], "Internal error");
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
}

void DarcyPotentialReduction::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                                  const BlockVector &x, BlockVector &b)
{
   MFEM_ASSERT(x.Size() == fes_u.GetVSize() + fes_p.GetVSize(),
               "Wrong size of the solution vector!");
   MFEM_ASSERT(b.Size() == fes_u.GetVSize() + fes_p.GetVSize(),
               "Wrong size of the rhs vector!");

   const int NE = fes_u.GetNE();
   Vector u_e, bu_e, bp_e;
   Array<int> u_vdofs, p_dofs, edofs;

   const Vector &xu = x.GetBlock(0);
   Vector &bu = b.GetBlock(0);
   Vector &bp = b.GetBlock(1);

   for (int el = 0; el < NE; el++)
   {
      GetEDofs(el, edofs);
      if (edofs.Size() <= 0) { continue; }
      xu.GetSubVector(edofs, u_e);
      u_e.Neg();

      //bu -= A_e u_e
      const int a_size = hat_offsets[el+1] - hat_offsets[el];
      const DenseMatrix Ae(&Ae_data[Ae_offsets[el]], a_size, edofs.Size());

      bu_e.SetSize(a_size);
      Ae.Mult(u_e, bu_e);

      fes_u.GetElementVDofs(el, u_vdofs);
      bu.AddElementVector(u_vdofs, bu_e);

      //bp -= B_e u_e
      const int d_size = D_f_offsets[el+1] - D_f_offsets[el];
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

void DarcyPotentialReduction::EliminateTrueDofsInRHS(
   const Array<int> &tdofs_flux, const BlockVector &x, BlockVector &b)
{
   const int NE = fes_u.GetNE();
   Vector u_e, bu_e, bp_e;
   Array<int> u_vdofs, p_dofs, edofs;

   const Vector &xu_t = x.GetBlock(0);
   Vector &bu_t = b.GetBlock(0);
   Vector &bp = b.GetBlock(1);

   Vector xu, bu;
   if (!Parallel())
   {
      const SparseMatrix *tr_cP = fes_u.GetConformingProlongation();
      if (!tr_cP)
      {
         xu.SetDataAndSize(xu_t.GetData(), xu_t.Size());
         bu.SetDataAndSize(bu.GetData(), bu.Size());
      }
      else
      {
         xu.SetSize(fes_u.GetVSize());
         tr_cP->Mult(xu_t, xu);
         bu.SetSize(fes_u.GetVSize());
         bu = 0.;
      }
   }
   else
   {
      xu.SetSize(fes_u.GetVSize());
      fes_u.GetProlongationMatrix()->Mult(xu_t, xu);
      bu.SetSize(fes_u.GetVSize());
      bu = 0.;
   }

   for (int el = 0; el < NE; el++)
   {
      GetEDofs(el, edofs);
      xu.GetSubVector(edofs, u_e);
      u_e.Neg();

      //bu -= A_e u_e
      const int a_size = hat_offsets[el+1] - hat_offsets[el];
      const DenseMatrix Ae(&Ae_data[Ae_offsets[el]], a_size, edofs.Size());

      bu_e.SetSize(a_size);
      Ae.Mult(u_e, bu_e);

      fes_u.GetElementVDofs(el, u_vdofs);
      bu.AddElementVector(u_vdofs, bu_e);

      //bp -= B_e u_e
      const int d_size = D_f_offsets[el+1] - D_f_offsets[el];
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

   if (!Parallel())
   {
      const SparseMatrix *tr_cP = fes_u.GetConformingProlongation();
      if (tr_cP)
      {
         tr_cP->AddMultTranspose(bu, bu_t);
      }
   }
   else
   {
      fes_u.GetProlongationMatrix()->AddMultTranspose(bu, bu_t);
   }

   for (int tdof : tdofs_flux)
   {
      bu_t(tdof) = xu_t(tdof);//<--can be arbitrary as it is ignored
   }
}

void DarcyPotentialReduction::ReduceRHS(const BlockVector &b,
                                        Vector &b_tr) const
{
   const int NE = fes_u.GetNE();
   Vector bu_l, bp_l;
   Array<int> u_vdofs, p_dofs;

   const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);

   Vector b_r;
   const Operator *tr_cP;

   if (!Parallel() && !(tr_cP = fes_u.GetConformingProlongation()))
   {
      b_tr.SetSize(fes_u.GetVSize());
      b_r.SetDataAndSize(b_tr.GetData(), b_tr.Size());
      b_r = bu;
   }
   else
   {
      b_r.SetSize(fes_u.GetVSize());
      if (!Parallel())
      {
         tr_cP->Mult(bu, b_r);
      }
      else
      {
         fes_u.GetProlongationMatrix()->Mult(bu, b_r);
      }
   }

   for (int el = 0; el < NE; el++)
   {
      // Load RHS

      GetFDofs(el, u_vdofs);
      bu_l.SetSize(u_vdofs.Size());

      fes_p.GetElementVDofs(el, p_dofs);
      bp.GetSubVector(p_dofs, bp_l);

      // -B^T D^-1 bp

      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];
      const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                          d_dofs_size, a_dofs_size);
      const LUFactors LU_D(const_cast<real_t*>(&D_data[D_offsets[el]]),
                           const_cast<int*>(&D_ipiv[D_f_offsets[el]]));

      LU_D.Solve(d_dofs_size, 1, bp_l.GetData());
      bp_l.Neg();
      B.MultTranspose(bp_l, bu_l);

      b_r.AddElementVector(u_vdofs, bu_l);
   }

   if (!Parallel())
   {
      if (tr_cP)
      {
         b_tr.SetSize(tr_cP->Width());
         tr_cP->MultTranspose(b_r, b_tr);
      }
   }
   else
   {
      const Operator *tr_P = fes_u.GetProlongationMatrix();
      b_tr.SetSize(tr_P->Width());
      tr_P->MultTranspose(b_r, b_tr);
   }
}

void DarcyPotentialReduction::ComputeSolution(const BlockVector &b,
                                              const Vector &sol_tr,
                                              BlockVector &sol) const
{
   const int NE = fes_u.GetNE();
   Vector bp_l, u_l;
   Array<int> u_vdofs, p_dofs;

   //const Vector &bu = b.GetBlock(0);
   const Vector &bp = b.GetBlock(1);
   Vector &u = sol.GetBlock(0);
   Vector &p = sol.GetBlock(1);

   Vector sol_r;
   if (!Parallel())
   {
      const SparseMatrix *tr_cP = fes_u.GetConformingProlongation();
      if (!tr_cP)
      {
         sol_r.SetDataAndSize(sol_tr.GetData(), sol_tr.Size());
      }
      else
      {
         sol_r.SetSize(fes_u.GetVSize());
         tr_cP->Mult(sol_tr, sol_r);
      }
   }
   else
   {
      sol_r.SetSize(fes_u.GetVSize());
      fes_u.GetProlongationMatrix()->Mult(sol_tr, sol_r);
   }

   u = sol_tr;

   for (int el = 0; el < NE; el++)
   {
      //Load RHS

      GetFDofs(el, u_vdofs);
      sol_r.GetSubVector(u_vdofs, u_l);

      fes_p.GetElementVDofs(el, p_dofs);
      bp.GetSubVector(p_dofs, bp_l);
      if (bsym)
      {
         //In the case of the symmetrized system, the sign is oppposite!
         bp_l.Neg();
      }

      // D^-1 (F - B u)

      int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
      int d_dofs_size = D_f_offsets[el+1] - D_f_offsets[el];
      const DenseMatrix B(const_cast<real_t*>(&Bf_data[Bf_offsets[el]]),
                          d_dofs_size, a_dofs_size);
      const LUFactors LU_D(const_cast<real_t*>(&D_data[D_offsets[el]]),
                           const_cast<int*>(&D_ipiv[D_f_offsets[el]]));

      B.AddMult(u_l, bp_l, -1.);

      LU_D.Solve(d_dofs_size, 1, bp_l.GetData());

      p.SetSubVector(p_dofs, bp_l);
   }
}

void DarcyPotentialReduction::Reset()
{
   DarcyReduction::Reset();

#ifdef MFEM_DARCY_REDUCTION_ELIM_BCS
   Be_data = 0.;
#endif //MFEM_DARCY_REDUCTION_ELIM_BCS
}

}
