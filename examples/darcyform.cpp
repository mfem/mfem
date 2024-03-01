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

// Implementation of class LinearForm

#include "darcyform.hpp"

namespace mfem
{

DarcyForm::DarcyForm(FiniteElementSpace *fes_u_, FiniteElementSpace *fes_p_)
   : fes_u(fes_u_), fes_p(fes_p_)
{
   offsets.SetSize(3);
   offsets[0] = 0;
   offsets[1] = fes_u->GetVSize();
   offsets[2] = fes_p->GetVSize();
   offsets.PartialSum();

   width = height = offsets.Last();

   M_u = NULL;
   M_p = NULL;
   B = NULL;

   block_op = new BlockOperator(offsets);

   hybridization = NULL;
}

BilinearForm* DarcyForm::GetFluxMassForm()
{
   if (!M_u) { M_u = new BilinearForm(fes_u); }
   return M_u;
}

const BilinearForm* DarcyForm::GetFluxMassForm() const
{
   MFEM_ASSERT(M_u, "Flux mass form not allocated!");
   return M_u;
}

BilinearForm* DarcyForm::GetPotentialMassForm()
{
   if (!M_p) { M_p = new BilinearForm(fes_p); }
   return M_p;
}

const BilinearForm* DarcyForm::GetPotentialMassForm() const
{
   MFEM_ASSERT(M_p, "Potential mass form not allocated!");
   return M_p;
}

MixedBilinearForm* DarcyForm::GetFluxDivForm()
{
   if (!B) { B = new MixedBilinearForm(fes_u, fes_p); }
   return B;
}

const MixedBilinearForm* DarcyForm::GetFluxDivForm() const
{
   MFEM_ASSERT(B, "Flux div form not allocated!");
   return B;
}

void DarcyForm::SetAssemblyLevel(AssemblyLevel assembly_level)
{
   assembly = assembly_level;

   if (M_u) { M_u->SetAssemblyLevel(assembly); }
   if (M_p) { M_p->SetAssemblyLevel(assembly); }
   if (B) { B->SetAssemblyLevel(assembly); }
}

void DarcyForm::EnableHybridization(FiniteElementSpace *constr_space,
                                    BilinearFormIntegrator *constr_flux_integ,
                                    const Array<int> &ess_flux_tdof_list)
{
   MFEM_ASSERT(M_u, "Mass form for the fluxes must be set prior to this call!");
   delete hybridization;
   if (assembly != AssemblyLevel::LEGACY)
   {
      delete constr_flux_integ;
      hybridization = NULL;
      MFEM_WARNING("Hybridization not supported for this assembly level");
      return;
   }
   hybridization = new DarcyHybridization(fes_u, fes_p, constr_space);
   BilinearFormIntegrator *constr_pot_integ = NULL;
   if (M_p)
   {
      auto fbfi = M_p->GetFBFI();
      if (fbfi->Size())
      {
         if (fbfi->Size() > 1)
         {
            MFEM_WARNING("Only one face integrator is considered for hybridization");
         }
         constr_pot_integ = (*fbfi)[0];
         fbfi->DeleteFirst(constr_pot_integ);
      }
   }
   hybridization->SetConstraintIntegrators(constr_flux_integ, constr_pot_integ);
   hybridization->Init(ess_flux_tdof_list);
}

void DarcyForm::Assemble(int skip_zeros)
{
   if (M_u)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_u -> GetNE(); i++)
         {
            M_u->ComputeElementMatrix(i, elmat);
            M_u->AssembleElementMatrix(i, elmat, skip_zeros);
            hybridization->AssembleFluxMassMatrix(i, elmat);
         }
      }
      else
      {
         M_u->Assemble(skip_zeros);
      }
   }

   if (B)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_u -> GetNE(); i++)
         {
            B->ComputeElementMatrix(i, elmat);
            B->AssembleElementMatrix(i, elmat, skip_zeros);
            hybridization->AssembleDivMatrix(i, elmat);
         }
      }
      else
      {
         B->Assemble(skip_zeros);
      }
   }

   if (M_p)
   {
      if (hybridization)
      {
         DenseMatrix elmat;

         // Element-wise integration
         for (int i = 0; i < fes_p -> GetNE(); i++)
         {
            M_p->ComputeElementMatrix(i, elmat);
            M_p->AssembleElementMatrix(i, elmat, skip_zeros);
            hybridization->AssembleFluxMassMatrix(i, elmat);
         }

         AssembleHDGFaces(skip_zeros);
      }
      else
      {
         M_p->Assemble(skip_zeros);
      }
   }
}

void DarcyForm::Finalize(int skip_zeros)
{
   if (M_u)
   {
      M_u->Finalize(skip_zeros);
      block_op->SetDiagonalBlock(0, M_u);
   }

   if (M_p)
   {
      M_p->Finalize(skip_zeros);
      block_op->SetDiagonalBlock(1, M_p);
   }

   if (B)
   {
      B->Finalize(skip_zeros);

      if (!pBt.Ptr()) { ConstructBT(B); }

      block_op->SetBlock(0, 1, pBt.Ptr(), -1.);
      block_op->SetBlock(1, 0, B, -1.);
   }

   if (hybridization)
   {
      hybridization->Finalize();
   }
}

void DarcyForm::FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                                 BlockVector &x, BlockVector &b, OperatorHandle &A, Vector &X_, Vector &B_,
                                 int copy_interior)
{
   FormSystemMatrix(ess_flux_tdof_list, A);

   //conforming

   if (hybridization)
   {
      // Reduction to the Lagrange multipliers system
      EliminateVDofsInRHS(ess_flux_tdof_list, x, b);
      hybridization->ReduceRHS(b, B_);
      X_.SetSize(B_.Size());
      X_ = 0.0;
   }
   else
   {
      // A, X and B point to the same data as mat, x and b
      EliminateVDofsInRHS(ess_flux_tdof_list, x, b);
      X_.MakeRef(x, 0, x.Size());
      B_.MakeRef(b, 0, b.Size());
      if (!copy_interior)
      {
         x.GetBlock(0).SetSubVectorComplement(ess_flux_tdof_list, 0.0);
         x.GetBlock(1) = 0.;
      }
   }
}

void DarcyForm::FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                                 OperatorHandle &A)
{
   Array<int> ess_pot_tdof_list;//empty for discontinuous potentials

   if (M_u)
   {
      M_u->FormSystemMatrix(ess_flux_tdof_list, pM_u);
      block_op->SetDiagonalBlock(0, pM_u.Ptr());
   }

   if (M_p)
   {
      M_p->FormSystemMatrix(ess_pot_tdof_list, pM_p);
      block_op->SetDiagonalBlock(1, pM_p.Ptr());
   }

   if (B)
   {
      B->FormRectangularSystemMatrix(ess_flux_tdof_list, ess_pot_tdof_list, pB);

      ConstructBT(pB.Ptr());

      block_op->SetBlock(0, 1, pBt.Ptr(), -1.);
      block_op->SetBlock(1, 0, pB.Ptr(), -1.);
   }

   if (hybridization)
   {
      hybridization->Finalize();
      A.Reset(&hybridization->GetMatrix(), false);
   }
   else
   {
      A.Reset(block_op, false);
   }
}

void DarcyForm::RecoverFEMSolution(const Vector &X, const BlockVector &b,
                                   BlockVector &x)
{
   if (hybridization)
   {
      //conforming
      hybridization->ComputeSolution(b, X, x);
   }
   else
   {
      BlockVector X_b(const_cast<Vector&>(X), offsets);
      if (M_u)
      {
         M_u->RecoverFEMSolution(X_b.GetBlock(0), b.GetBlock(0), x.GetBlock(0));
      }
      if (M_p)
      {
         M_p->RecoverFEMSolution(X_b.GetBlock(1), b.GetBlock(1), x.GetBlock(1));
      }
   }
}

void DarcyForm::EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                    const BlockVector &x, BlockVector &b)
{
   if (B)
   {
      if (assembly != AssemblyLevel::LEGACY && assembly != AssemblyLevel::FULL)
      {
         //TODO
         MFEM_ABORT("");
      }
      else
      {
         Array<int> vdofs_flux_marker;
         FiniteElementSpace::ListToMarker(vdofs_flux, fes_u->GetTrueVSize(),
                                          vdofs_flux_marker);
         B->EliminateEssentialBCFromTrialDofs(vdofs_flux_marker, x.GetBlock(0),
                                              b.GetBlock(1));
      }
   }
   if (M_u)
   {
      M_u->EliminateVDofsInRHS(vdofs_flux, x.GetBlock(0), b.GetBlock(0));
   }
}

DarcyForm::~DarcyForm()
{
   if (M_u) { delete M_u; }
   if (M_p) { delete M_p; }
   if (B) { delete B; }

   delete block_op;

   delete hybridization;
}

void DarcyForm::AssembleHDGFaces(int skip_zeros)
{
   Mesh *mesh = fes_p->GetMesh();
   DenseMatrix elemmat;
   Array<int> vdofs;

   auto &interior_face_integs = *M_p->GetFBFI();

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *tr;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr != NULL)
         {
            hybridization->ComputeAndAssembleFaceMatrix(i, elemmat, vdofs);
            M_p->SpMat().AddSubMatrix(vdofs, vdofs, elemmat, skip_zeros);
         }
      }
   }

   auto &boundary_face_integs = *M_p->GetBFBFI();
   auto &boundary_face_integs_marker = *M_p->GetBFBFI_Marker();

   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs.Size(); k++)
      {
         if (boundary_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < fes_p -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh -> GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            if (boundary_face_integs_marker[0] &&
                (*boundary_face_integs_marker[0])[bdr_attr-1] == 0)
            { continue; }

            int faceno = mesh->GetBdrElementFaceIndex(i);
            hybridization->ComputeAndAssembleFaceMatrix(faceno, elemmat, vdofs);
            M_p->SpMat().AddSubMatrix(vdofs, vdofs, elemmat, skip_zeros);
         }
      }
   }
}

const Operator *DarcyForm::ConstructBT(const MixedBilinearForm *B)
{
   pBt.Reset(Transpose(B->SpMat()));
   return pBt.Ptr();
}

const Operator* DarcyForm::ConstructBT(const Operator *opB)
{
   pBt.Reset(new TransposeOperator(opB));
   return pBt.Ptr();
}

DarcyHybridization::DarcyHybridization(FiniteElementSpace *fes_u_,
                                       FiniteElementSpace *fes_p_,
                                       FiniteElementSpace *fes_c_)
   : Hybridization(fes_u_, fes_c_), fes_p(fes_p_)
{
   c_bfi_p = NULL;

   Bf_data = NULL;
   Df_data = NULL;
   Df_ipiv = NULL;
}

DarcyHybridization::~DarcyHybridization()
{
   delete c_bfi_p;

   delete Bf_data;
   delete Df_data;
   delete Df_ipiv;
}

void DarcyHybridization::SetConstraintIntegrators(BilinearFormIntegrator
                                                  *c_flux_integ, BilinearFormIntegrator *c_pot_integ)
{
   delete c_bfi;
   c_bfi = c_flux_integ;
   delete c_bfi_p;
   c_bfi_p = c_pot_integ;
}

void DarcyHybridization::Init(const Array<int> &ess_flux_tdof_list)
{
   if (Ct) { return; }

   Hybridization::Init(ess_flux_tdof_list);

   const int NE = fes->GetNE();

   // Define Bf_offsets, Df_offsets and Df_f_offsets
   Bf_offsets.SetSize(NE+1);
   Bf_offsets[0] = 0;
   Df_offsets.SetSize(NE+1);
   Df_offsets[0] = 0;
   Df_f_offsets.SetSize(NE+1);
   Df_f_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      int f_size = Af_f_offsets[i+1] - Af_f_offsets[i];
      int d_size = fes_p->GetFE(i)->GetDof();
      Bf_offsets[i+1] = Bf_offsets[i] + f_size*d_size;
      Df_offsets[i+1] = Df_offsets[i] + d_size*d_size;
      Df_f_offsets[i+1] = Df_f_offsets[i] + d_size;
   }

   Bf_data = new double[Bf_offsets[NE]]();//init by zeros
   Df_data = new double[Df_offsets[NE]]();//init by zeros
   Df_ipiv = new int[Df_f_offsets[NE]];
}

void DarcyHybridization::AssembleFluxMassMatrix(int el, const DenseMatrix &A)
{
   const int s = Af_f_offsets[el+1] - Af_f_offsets[el];
   DenseMatrix A_i(Af_data + Af_offsets[el], s, s);

   A_i = A;
}

void DarcyHybridization::AssemblePotMassMatrix(int el, const DenseMatrix &D)
{
   const int s = Df_f_offsets[el+1] - Df_f_offsets[el];
   DenseMatrix D_i(Df_data + Df_offsets[el], s, s);

   D_i = D;
}

void DarcyHybridization::AssembleDivMatrix(int el, const DenseMatrix &B)
{
   const int w = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int h = Df_f_offsets[el+1] - Df_f_offsets[el];
   DenseMatrix B_i(Bf_data + Bf_offsets[el], h, w);

   B_i = B;
}

void DarcyHybridization::ComputeAndAssembleFaceMatrix(int face,
                                                      DenseMatrix &elmat, Array<int> &vdofs)
{
   Mesh *mesh = fes_p->GetMesh();
   const FiniteElement *fe1, *fe2;

   FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(face);
   fes_p->GetElementVDofs(ftr->Elem2No, vdofs);
   fe1 = fes_p->GetFE(ftr->Elem1No);

   if (ftr->Elem2No >= 0)
   {
      Array<int> vdofs2;
      fes_p->GetElementVDofs(ftr->Elem2No, vdofs2);
      vdofs.Append(vdofs2);
      fe2 = fes_p->GetFE(ftr->Elem2No);
   }
   else
   {
      fe2 = fe1;
   }

   c_bfi_p->AssembleFaceMatrix(*fe1, *fe2, *ftr, elmat);
}

void DarcyHybridization::ComputeH()
{
   const int skip_zeros = 1;
   const int NE = fes->GetNE();
   DenseMatrix AiBt, BAi, Ct_l, AiCt, BAiCt, CAiBt, H_l;
   Array<int> c_dofs;

   H = new SparseMatrix(Ct->Width());

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
      LU_A.Solve(AiBt.Height(), AiBt.Width(), AiBt.GetData());
      mfem::AddMult(B, AiBt, D);

      // Decompose Schur complement
      LUFactors LU_S(D.GetData(), Df_ipiv + Df_f_offsets[el]);

      LU_S.Factor(d_dofs_size);

      // Get C^T
      GetCt(el, Ct_l, c_dofs);

      //-C A^-1 C^T
      AiCt.SetSize(Ct_l.Height(), Ct_l.Width());
      AiCt = Ct_l;
      LU_A.Solve(Ct_l.Height(), Ct_l.Width(), AiCt.GetData());

      H_l.SetSize(Ct_l.Width());
      mfem::MultAtB(Ct_l, AiCt, H_l);
      H_l.Neg();

      //C A^-1 B^T S^-1 B A^-1 C^T
      BAiCt.SetSize(B.Height(), Ct_l.Width());
      mfem::Mult(B, AiCt, BAiCt);

      CAiBt.SetSize(Ct_l.Width(), B.Height());
      mfem::MultAtB(Ct_l, AiBt, CAiBt);

      LU_S.Solve(BAiCt.Height(), BAiCt.Width(), BAiCt.GetData());

      mfem::AddMult(CAiBt, BAiCt, H_l);

      H->AddSubMatrix(c_dofs, c_dofs, H_l, skip_zeros);
   }

   H->Finalize();
}

void DarcyHybridization::GetCt(int el, DenseMatrix &Ct_l,
                               Array<int> &c_dofs) const
{
   const int hat_o = hat_offsets[el  ];
   const int hat_s = hat_offsets[el+1] - hat_o;
   MFEM_ASSERT(hat_s == Af_f_offsets[el+1] - Af_f_offsets[el], "");
   c_fes->GetElementDofs(el, c_dofs);
   Ct_l.SetSize(hat_s, c_dofs.Size());
   Ct_l = 0.;
   for (int i = 0; i < hat_s; i++)
   {
      const int row = hat_o + i;
      int col = 0;
      const int ncols = Ct->RowSize(row);
      const int *cols = Ct->GetRowColumns(row);
      const double *vals = Ct->GetRowEntries(row);
      for (int j = 0; j < c_dofs.Size() && col < ncols; j++)
      {
         if (cols[col] != c_dofs[j]) { continue; }
         Ct_l(i,j) = vals[col++];
      }
   }
}

void DarcyHybridization::Finalize()
{
   if (!H) { ComputeH(); }
}

void DarcyHybridization::MultInv(int el, const Vector &bu, const Vector &bp,
                                 Vector &u, Vector &p) const
{
   Vector BAibu, BtSiBAibu;

   const int a_dofs_size = Af_f_offsets[el+1] - Af_f_offsets[el];
   const int d_dofs_size = Df_f_offsets[el+1] - Df_f_offsets[el];

   // Load LU decomposition of A and Schur complement

   LUFactors LU_A(Af_data + Af_offsets[el], Af_ipiv + Af_f_offsets[el]);
   LUFactors LU_S(Df_data + Df_offsets[el], Df_ipiv + Df_f_offsets[el]);

   // Load B

   DenseMatrix B(Bf_data + Bf_offsets[el], d_dofs_size, a_dofs_size);

   //-A^-1 bu
   u.SetSize(bu.Size());
   u = bu;
   u.Neg();
   LU_A.Solve(u.Size(), 1, u.GetData());

   //A^-1 B^T S^-1 B A^-1 bu
   BAibu.SetSize(B.Height());
   B.Mult(u, BAibu);

   LU_S.Solve(BAibu.Size(), 1, BAibu.GetData());

   BtSiBAibu.SetSize(B.Width());
   B.MultTranspose(BAibu, BtSiBAibu);

   LU_A.Solve(BtSiBAibu.Size(), 1, BtSiBAibu.GetData());

   u -= BtSiBAibu;
}

void DarcyHybridization::ReduceRHS(const BlockVector &b, Vector &b_r) const
{
   const int NE = fes->GetNE();
   DenseMatrix Ct_l;
   Vector bu_l, bp_l, b_rl, u_l, p_l;
   Array<int> u_vdofs, c_dofs;

   if (b_r.Size() != H->Height())
   {
      b_r.SetSize(H->Height());
      b_r = 0.;
   }

   const Vector &bu = b.GetBlock(0);

   for (int el = 0; el < NE; el++)
   {
      // Load RHS

      fes->GetElementVDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      // Get C^T
      GetCt(el, Ct_l, c_dofs);

      //C (-A^-1 bu + A^-1 B^T S^-1 B A^-1 bu)
      MultInv(el, bu_l, bp_l, u_l, p_l);

      b_rl.SetSize(c_dofs.Size());
      Ct_l.MultTranspose(u_l, b_rl);

      b_r.AddElementVector(c_dofs, b_rl);
   }
}

void DarcyHybridization::ComputeSolution(const BlockVector &b,
                                         const Vector &sol_r, BlockVector &sol) const
{
   const int NE = fes->GetNE();
   DenseMatrix Ct_l;
   Vector bu_l, bp_l, b_rl, sol_rl, u_l, p_l;
   Array<int> u_vdofs, c_dofs;

   const Vector &bu = b.GetBlock(0);
   Vector &u = sol.GetBlock(0);

   for (int el = 0; el < NE; el++)
   {
      //Load RHS

      fes->GetElementVDofs(el, u_vdofs);
      bu.GetSubVector(u_vdofs, bu_l);

      // Get C^T
      GetCt(el, Ct_l, c_dofs);

      // bu - C^T sol

      sol_r.GetSubVector(c_dofs, sol_rl);
      Ct_l.AddMult_a(-1., sol_rl, bu_l);

      //(-A^-1 + A^-1 B^T S^-1 B A^-1) (bu - C^T sol)
      MultInv(el, bu_l, bp_l, u_l, p_l);

      u.SetSubVector(u_vdofs, u_l);
   }
}

}
