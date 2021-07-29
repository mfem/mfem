// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/forall.hpp"

namespace mfem
{

ParNonlinearForm::ParNonlinearForm(ParFiniteElementSpace *pf)
   : NonlinearForm(pf), pGrad(Operator::Hypre_ParCSR)
{
   X.MakeRef(pf, NULL);
   Y.MakeRef(pf, NULL);
   MFEM_VERIFY(!Serial(), "internal MFEM error");
}

double ParNonlinearForm::GetParGridFunctionEnergy(const Vector &x) const
{
   double loc_energy, glob_energy;

   loc_energy = GetGridFunctionEnergy(x);

   if (fnfi.Size())
   {
      MFEM_ABORT("TODO: add energy contribution from shared faces");
   }

   MPI_Allreduce(&loc_energy, &glob_energy, 1, MPI_DOUBLE, MPI_SUM,
                 ParFESpace()->GetComm());

   return glob_energy;
}

void ParNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   NonlinearForm::Mult(x, y); // x --(P)--> aux1 --(A_local)--> aux2

   if (fnfi.Size())
   {
      MFEM_VERIFY(!NonlinearForm::ext, "Not implemented (extensions + faces");
      // Terms over shared interior faces in parallel.
      ParFiniteElementSpace *pfes = ParFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2;
      Vector el_x, el_y;

      aux1.HostReadWrite();
      X.MakeRef(aux1, 0); // aux1 contains P.x
      X.ExchangeFaceNbrData();
      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);
         int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

         fe1 = pfes->GetFE(tr->Elem1No);
         fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
         pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
         X.GetSubVector(vdofs1, el_x.GetData());
         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

         for (int k = 0; k < fnfi.Size(); k++)
         {
            fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
            aux2.AddElementVector(vdofs1, el_y.GetData());
         }
      }
   }

   P->MultTranspose(aux2, y);

   const int N = ess_tdof_list.Size();
   const auto idx = ess_tdof_list.Read();
   auto Y = y.ReadWrite();
   MFEM_FORALL(i, N, Y[idx[i]] = 0.0; );
}

void ParNonlinearForm::Mult(const Vector &x, Vector &y, ParMesh* pmesh,
                            ParGridFunction* end_crds, ParGridFunction* beg_crds,
                            const Vector &v) const
{
   //We're going to pretty much take everything from in NonlinearForm::Mult(x, y) into here
   //with a few changes made to it

   Array<int> vdofs;
   Vector el_x, el_y, el_v;
   const FiniteElement *fe;
   IsoparametricTransformation beg_T;
   IsoparametricTransformation end_T;
   Mesh *mesh = fes->GetMesh();
   const Vector &ptemp = Prolongate(x);
   const Vector px(ptemp);
   const Vector &ptemp1 = Prolongate(v);
   const Vector pv(ptemp1);
   Vector &py = P ? aux2.SetSize(P->Height()), aux2 : y;

   py = 0.0;
   GridFunction *nodes;

   if (dnfi.Size())
   {
      for (int i = 0; i < fes->GetNE(); i++)
      {
         //Here we're pretty much forced to do things a little different
         //from before since we have to explicitly call the pmesh GetElementTransformation
         //if we don't then we end up with the exact same ElementTransformations
         //for beg_T and end_T
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         pmesh->GetElementTransformation(i, &beg_T);
         //These lines are updating
         int own_nodes = 0;
         nodes = end_crds;
         pmesh->SwapNodes(nodes, own_nodes);
         fe = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         pmesh->GetElementTransformation(i, &end_T);
         px.GetSubVector(vdofs, el_x);
         pv.GetSubVector(vdofs, el_v);
         for (int k = 0; k < dnfi.Size(); k++)
         {
            //This is the part that we're changing by adding more transformation terms
            //as well
            dnfi[k]->AssembleElementVector(*fe, beg_T, end_T, el_x, el_y, el_v);
            py.AddElementVector(vdofs, el_y);
         }
         nodes = beg_crds;
         pmesh->SwapNodes(nodes, own_nodes);
      }
   }
   nodes = NULL;
   //fix_me: Do we need to worry about the face transformations down here as well
   //and what their element transformation is based upon? I'm feeling like the answer
   //might be yes for certain applications. However, I'm not seeing the need for something
   //like UMATs for the below at least currently.
   if (fnfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            fes->GetElementVDofs(tr->Elem2No, vdofs2);
            vdofs.Append (vdofs2);

            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            fe2 = fes->GetFE(tr->Elem2No);

            for (int k = 0; k < fnfi.Size(); k++)
            {
               fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               py.AddElementVector(vdofs, el_y);
            }
         }
      }
   }
   //fix_me: Same as the previous fix_me in this function.
   if (bfnfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfnfi.Size(); k++)
      {
         if (bfnfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfnfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < bfnfi.Size(); k++)
            {
               if (bfnfi_marker[k] &&
                   (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

               bfnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               py.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   //The below is directly from ParNonlinearForm::Mult(x,y)

   Y.SetData(aux2.GetData()); // aux2 contains A_local.P.x
   //fix_me: same as the other previous fix_mes in this function
   if (fnfi.Size())
   {
      // Terms over shared interior faces in parallel.
      ParFiniteElementSpace *pfes = ParFESpace();
      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs1, vdofs2;
      Vector el_x, el_y;

      X.SetData(aux1.GetData()); // aux1 contains P.x
      X.ExchangeFaceNbrData();
      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);

         fe1 = pfes->GetFE(tr->Elem1No);
         fe2 = pfes->GetFaceNbrFE(tr->Elem2No);

         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
         pfes->GetFaceNbrElementVDofs(tr->Elem2No, vdofs2);

         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
         X.GetSubVector(vdofs1, el_x.GetData());
         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

         for (int k = 0; k < fnfi.Size(); k++)
         {
            fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
            Y.AddElementVector(vdofs1, el_y.GetData());
         }
      }
   }

   P->MultTranspose(Y, y);

   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      y(ess_tdof_list[i]) = 0.0;
   }
}


const SparseMatrix &ParNonlinearForm::GetLocalGradient(const Vector &x) const
{
   MFEM_VERIFY(NonlinearForm::ext == nullptr,
               "this method is not supported yet with partial assembly");

   NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

   return *Grad;
}

Operator &ParNonlinearForm::GetLocalGradient2(const Vector &x) const
{
   ParFiniteElementSpace *pfes = ParFESpace();

   pGrad.Clear();

   NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

   OperatorHandle dA(pGrad.Type()), Ph(pGrad.Type());

   if (fnfi.Size() == 0)
   {
      dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                             pfes->GetDofOffsets(), Grad);
   }
   else
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   // TODO - construct Dof_TrueDof_Matrix directly in the pGrad format
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   pGrad.MakePtAP(dA, Ph);

   return *pGrad.Ptr();
}

Operator &ParNonlinearForm::GetGradient(const Vector &x) const
{
   if (NonlinearForm::ext) { return NonlinearForm::GetGradient(x); }

   ParFiniteElementSpace *pfes = ParFESpace();

   pGrad.Clear();

   NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

   OperatorHandle dA(pGrad.Type()), Ph(pGrad.Type());

   if (fnfi.Size() == 0)
   {
      dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                             pfes->GetDofOffsets(), Grad);
   }
   else
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   // RAP the local gradient dA.
   // TODO - construct Dof_TrueDof_Matrix directly in the pGrad format
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   pGrad.MakePtAP(dA, Ph);

   // Impose b.c. on pGrad
   OperatorHandle pGrad_e;
   pGrad_e.EliminateRowsCols(pGrad, ess_tdof_list);

   return *pGrad.Ptr();
}

void ParNonlinearForm::Update()
{
   Y.MakeRef(ParFESpace(), NULL);
   X.MakeRef(ParFESpace(), NULL);
   pGrad.Clear();
   NonlinearForm::Update();
}


ParBlockNonlinearForm::ParBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf)
   : BlockNonlinearForm()
{
   pBlockGrad = NULL;
   SetParSpaces(pf);
}

void ParBlockNonlinearForm::SetParSpaces(Array<ParFiniteElementSpace *> &pf)
{
   delete pBlockGrad;
   pBlockGrad = NULL;

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         delete phBlockGrad(s1,s2);
      }
   }

   Array<FiniteElementSpace *> serialSpaces(pf.Size());

   for (int s=0; s<pf.Size(); s++)
   {
      serialSpaces[s] = (FiniteElementSpace *) pf[s];
   }

   SetSpaces(serialSpaces);

   phBlockGrad.SetSize(fes.Size(), fes.Size());

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         phBlockGrad(s1,s2) = new OperatorHandle(Operator::Hypre_ParCSR);
      }
   }
}

ParFiniteElementSpace * ParBlockNonlinearForm::ParFESpace(int k)
{
   return (ParFiniteElementSpace *)fes[k];
}

const ParFiniteElementSpace *ParBlockNonlinearForm::ParFESpace(int k) const
{
   return (const ParFiniteElementSpace *)fes[k];
}

// Here, rhs is a true dof vector
void ParBlockNonlinearForm::SetEssentialBC(const
                                           Array<Array<int> *>&bdr_attr_is_ess,
                                           Array<Vector *> &rhs)
{
   Array<Vector *> nullarray(fes.Size());
   nullarray = NULL;

   BlockNonlinearForm::SetEssentialBC(bdr_attr_is_ess, nullarray);

   for (int s = 0; s < fes.Size(); ++s)
   {
      if (rhs[s])
      {
         rhs[s]->SetSubVector(*ess_tdofs[s], 0.0);
      }
   }
}

double ParBlockNonlinearForm::GetEnergy(const Vector &x) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   xs.Update(block_offsets);

   for (int s = 0; s < fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(xs_true.GetBlock(s), xs.GetBlock(s));
   }

   double enloc = BlockNonlinearForm::GetEnergyBlocked(xs);
   double englo = 0.0;

   MPI_Allreduce(&enloc, &englo, 1, MPI_DOUBLE, MPI_SUM,
                 ParFESpace(0)->GetComm());

   return englo;
}

void ParBlockNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   ys_true.Update(y, block_trueOffsets);
   xs.Update(block_offsets);
   ys.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   BlockNonlinearForm::MultBlocked(xs, ys);

   if (fnfi.Size() > 0)
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->MultTranspose(
         ys.GetBlock(s), ys_true.GetBlock(s));

      ys_true.GetBlock(s).SetSubVector(*ess_tdofs[s], 0.0);
   }

   ys_true.SyncFromBlocks();
   y.SyncMemory(ys_true);
}

/// Return the local gradient matrix for the given true-dof vector x
const BlockOperator & ParBlockNonlinearForm::GetLocalGradient(
   const Vector &x) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   xs.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   // (re)assemble Grad without b.c. into 'Grads'
   BlockNonlinearForm::ComputeGradientBlocked(xs);

   delete BlockGrad;
   BlockGrad = new BlockOperator(block_offsets);

   for (int i = 0; i < fes.Size(); ++i)
   {
      for (int j = 0; j < fes.Size(); ++j)
      {
         BlockGrad->SetBlock(i, j, Grads(i, j));
      }
   }
   return *BlockGrad;
}

// Set the operator type id for the parallel gradient matrix/operator.
void ParBlockNonlinearForm::SetGradientType(Operator::Type tid)
{
   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         phBlockGrad(s1,s2)->SetType(tid);
      }
   }
}

BlockOperator & ParBlockNonlinearForm::GetGradient(const Vector &x) const
{
   if (pBlockGrad == NULL)
   {
      pBlockGrad = new BlockOperator(block_trueOffsets);
   }

   Array<const ParFiniteElementSpace *> pfes(fes.Size());

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      pfes[s1] = ParFESpace(s1);

      for (int s2=0; s2<fes.Size(); ++s2)
      {
         phBlockGrad(s1,s2)->Clear();
      }
   }

   GetLocalGradient(x); // gradients are stored in 'Grads'

   if (fnfi.Size() > 0)
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         OperatorHandle dA(phBlockGrad(s1,s2)->Type()),
                        Ph(phBlockGrad(s1,s2)->Type()),
                        Rh(phBlockGrad(s1,s2)->Type());

         if (s1 == s2)
         {
            dA.MakeSquareBlockDiag(pfes[s1]->GetComm(), pfes[s1]->GlobalVSize(),
                                   pfes[s1]->GetDofOffsets(), Grads(s1,s1));
            Ph.ConvertFrom(pfes[s1]->Dof_TrueDof_Matrix());
            phBlockGrad(s1,s1)->MakePtAP(dA, Ph);

            OperatorHandle Ae;
            Ae.EliminateRowsCols(*phBlockGrad(s1,s1), *ess_tdofs[s1]);
         }
         else
         {
            dA.MakeRectangularBlockDiag(pfes[s1]->GetComm(),
                                        pfes[s1]->GlobalVSize(),
                                        pfes[s2]->GlobalVSize(),
                                        pfes[s1]->GetDofOffsets(),
                                        pfes[s2]->GetDofOffsets(),
                                        Grads(s1,s2));
            Rh.ConvertFrom(pfes[s1]->Dof_TrueDof_Matrix());
            Ph.ConvertFrom(pfes[s2]->Dof_TrueDof_Matrix());

            phBlockGrad(s1,s2)->MakeRAP(Rh, dA, Ph);

            phBlockGrad(s1,s2)->EliminateRows(*ess_tdofs[s1]);
            phBlockGrad(s1,s2)->EliminateCols(*ess_tdofs[s2]);
         }

         pBlockGrad->SetBlock(s1, s2, phBlockGrad(s1,s2)->Ptr());
      }
   }

   return *pBlockGrad;
}

ParBlockNonlinearForm::~ParBlockNonlinearForm()
{
   delete pBlockGrad;
   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         delete phBlockGrad(s1,s2);
      }
   }
}

}

#endif
