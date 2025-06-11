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

real_t ParNonlinearForm::GetParGridFunctionEnergy(const Vector &x) const
{
   real_t loc_energy, glob_energy;

   loc_energy = GetGridFunctionEnergy(x);

   if (fnfi.Size())
   {
      MFEM_ABORT("TODO: add energy contribution from shared faces");
   }

   MPI_Allreduce(&loc_energy, &glob_energy, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
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
   auto Y_RW = y.ReadWrite();
   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { Y_RW[idx[i]] = 0.0; });
}

const SparseMatrix &ParNonlinearForm::GetLocalGradient(const Vector &x) const
{
   MFEM_VERIFY(NonlinearForm::ext == nullptr,
               "this method is not supported yet with partial assembly");

   NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

   return *Grad;
}

void ParNonlinearForm::GradientSharedFaces(const Vector &x,
                                           int skip_zeros) const
{
   ParFiniteElementSpace *pfes = ParFESpace();
   ParMesh *pmesh = pfes->GetParMesh();
   FaceElementTransformations *T;
   Array<int> vdofs1, vdofs2, vdofs_all;
   DenseMatrix elemmat;
   Vector el_x, nbr_x, face_x;
   const Vector &px = Prolongate(x);

   ParGridFunction pgf(pfes, const_cast<Vector&>(px), 0);
   pgf.ExchangeFaceNbrData();

   int nfaces = pmesh->GetNSharedFaces();
   for (int i = 0; i < nfaces; i++)
   {
      T = pmesh->GetSharedFaceTransformations(i);
      int Elem2NbrNo = T->Elem2No - pmesh->GetNE();

      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
      face_x.SetSize(vdofs1.Size() + vdofs2.Size());

      el_x.MakeRef(face_x, 0, vdofs1.Size());
      pgf.GetSubVector(vdofs1, el_x);

      nbr_x.MakeRef(face_x, vdofs1.Size(), vdofs2.Size());
      pgf.FaceNbrData().GetSubVector(vdofs2, nbr_x);

      vdofs1.Copy(vdofs_all);
      for (int j = 0; j < vdofs2.Size(); j++)
      {
         if (vdofs2[j] >= 0)
         {
            vdofs2[j] += height;
         }
         else
         {
            vdofs2[j] -= height;
         }
      }
      vdofs_all.Append(vdofs2);
      for (int k = 0; k < fnfi.Size(); k++)
      {
         fnfi[k]->AssembleFaceGrad(*pfes->GetFE(T->Elem1No),
                                   *pfes->GetFaceNbrFE(Elem2NbrNo),
                                   *T, face_x, elemmat);
         Grad->AddSubMatrix(vdofs1, vdofs_all, elemmat, skip_zeros);
      }
   }
}

Operator &ParNonlinearForm::GetGradient(const Vector &x) const
{
   if (NonlinearForm::ext) { return NonlinearForm::GetGradient(x); }

   ParFiniteElementSpace *pfes = ParFESpace();

   pGrad.Clear();
   OperatorHandle dA(pGrad.Type()), Ph(pGrad.Type()), hdA;

   if (fnfi.Size())
   {
      const int skip_zeros = 0;

      pfes->ExchangeFaceNbrData();
      if (Grad == NULL)
      {
         int nbr_size = pfes->GetFaceNbrVSize();
         Grad = new SparseMatrix(pfes->GetVSize(), pfes->GetVSize() + nbr_size);
      }

      NonlinearForm::GetGradient(x, false); // (re)assemble Grad, no b.c.

      GradientSharedFaces(x, skip_zeros);

      Grad->Finalize(skip_zeros);

      // handle the case when 'a' contains off-diagonal
      int lvsize = pfes->GetVSize();
      const HYPRE_BigInt *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      HYPRE_BigInt ldof_offset = pfes->GetMyDofOffset();

      Array<HYPRE_BigInt> glob_J(Grad->NumNonZeroElems());
      int *J = Grad->GetJ();
      for (int i = 0; i < glob_J.Size(); i++)
      {
         if (J[i] < lvsize)
         {
            glob_J[i] = J[i] + ldof_offset;
         }
         else
         {
            glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];
         }
      }

      // TODO - construct dA directly in the A format
      hdA.Reset(
         new HypreParMatrix(pfes->GetComm(), lvsize, pfes->GlobalVSize(),
                            pfes->GlobalVSize(), Grad->GetI(), glob_J,
                            Grad->GetData(), pfes->GetDofOffsets(),
                            pfes->GetDofOffsets()));
      // - hdA owns the new HypreParMatrix
      // - the above constructor copies all input arrays
      glob_J.DeleteAll();
      dA.ConvertFrom(hdA);
   }
   else
   {
      NonlinearForm::GetGradient(x); // (re)assemble Grad, no b.c.

      dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                             pfes->GetDofOffsets(), Grad);
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

real_t ParBlockNonlinearForm::GetEnergy(const Vector &x) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   xs.Update(block_offsets);

   for (int s = 0; s < fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(xs_true.GetBlock(s), xs.GetBlock(s));
   }

   real_t enloc = BlockNonlinearForm::GetEnergyBlocked(xs);
   real_t englo = 0.0;

   MPI_Allreduce(&enloc, &englo, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
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
      // Terms over shared interior faces in parallel.
      ParMesh *pmesh = ParFESpace(0)->GetParMesh();
      FaceElementTransformations *tr;

      Array<Array<int> *>vdofs(fes.Size());
      Array<Array<int> *>vdofs2(fes.Size());
      Array<Vector *> el_x(fes.Size());
      Array<const Vector *> el_x_const(fes.Size());
      Array<Vector *> el_y(fes.Size());
      Array<const FiniteElement *> fe(fes.Size());
      Array<const FiniteElement *> fe2(fes.Size());
      Array<ParGridFunction *> pgfs(fes.Size());
      for (int s=0; s<fes.Size(); ++s)
      {
         el_x_const[s] = el_x[s] = new Vector();
         el_y[s] = new Vector();
         vdofs[s] = new Array<int>;
         vdofs2[s] = new Array<int>;
         pgfs[s] = new ParGridFunction(const_cast<ParFiniteElementSpace*>(ParFESpace(s)),
                                       xs.GetBlock(s));
         pgfs[s]->ExchangeFaceNbrData();
      }

      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);
         int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

         for (int s=0; s<fes.Size(); ++s)
         {
            const ParFiniteElementSpace *pfes = ParFESpace(s);
            fe[s] = pfes->GetFE(tr->Elem1No);
            fe2[s] = pfes->GetFaceNbrFE(Elem2NbrNo);

            pfes->GetElementVDofs(tr->Elem1No, *(vdofs[s]));
            pfes->GetFaceNbrElementVDofs(Elem2NbrNo, *(vdofs2[s]));

            el_x[s]->SetSize(vdofs[s]->Size() + vdofs2[s]->Size());
            xs.GetBlock(s).GetSubVector(*(vdofs[s]), el_x[s]->GetData());
            pgfs[s]->FaceNbrData().GetSubVector(*(vdofs2[s]),
                                                el_x[s]->GetData() + vdofs[s]->Size());
         }

         for (int k = 0; k < fnfi.Size(); ++k)
         {
            fnfi[k]->AssembleFaceVector(fe, fe2, *tr, el_x_const, el_y);

            for (int s=0; s<fes.Size(); ++s)
            {
               if (el_y[s]->Size() == 0) { continue; }
               ys.GetBlock(s).AddElementVector(*(vdofs[s]), *el_y[s]);
            }
         }
      }

      for (int s=0; s<fes.Size(); ++s)
      {
         delete pgfs[s];
         delete vdofs2[s];
         delete vdofs[s];
         delete el_y[s];
         delete el_x[s];
      }
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

void ParBlockNonlinearForm::GradientSharedFaces(const BlockVector &xs,
                                                int skip_zeros) const
{
   // Terms over shared interior faces in parallel.
   ParMesh *pmesh = ParFESpace(0)->GetParMesh();
   FaceElementTransformations *tr;

   Array<Array<int> *>vdofs(fes.Size());
   Array<Array<int> *>vdofs2(fes.Size());
   Array<Array<int> *>vdofs_all(fes.Size());
   Array<Vector *> el_x(fes.Size());
   Array<const Vector *> el_x_const(fes.Size());
   Array2D<DenseMatrix *> elmats(fes.Size(), fes.Size());
   Array<const FiniteElement *> fe(fes.Size());
   Array<const FiniteElement *> fe2(fes.Size());
   Array<ParGridFunction *> pgfs(fes.Size());

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      el_x_const[s1] = el_x[s1] = new Vector();
      vdofs[s1] = new Array<int>;
      vdofs2[s1] = new Array<int>;
      vdofs_all[s1] = new Array<int>;
      pgfs[s1] = new ParGridFunction(
         const_cast<ParFiniteElementSpace*>(ParFESpace(s1)),
         const_cast<Vector&>(xs.GetBlock(s1)));
      pgfs[s1]->ExchangeFaceNbrData();
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         elmats(s1,s2) = new DenseMatrix();
      }
   }

   const int n_shared_faces = pmesh->GetNSharedFaces();
   for (int i = 0; i < n_shared_faces; i++)
   {
      tr = pmesh->GetSharedFaceTransformations(i, true);
      int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

      for (int s=0; s<fes.Size(); ++s)
      {
         const ParFiniteElementSpace *pfes = ParFESpace(s);
         fe[s] = pfes->GetFE(tr->Elem1No);
         fe2[s] = pfes->GetFaceNbrFE(Elem2NbrNo);

         pfes->GetElementVDofs(tr->Elem1No, *(vdofs[s]));
         pfes->GetFaceNbrElementVDofs(Elem2NbrNo, *(vdofs2[s]));

         el_x[s]->SetSize(vdofs[s]->Size() + vdofs2[s]->Size());
         xs.GetBlock(s).GetSubVector(*(vdofs[s]), el_x[s]->GetData());
         pgfs[s]->FaceNbrData().GetSubVector(*(vdofs2[s]),
                                             el_x[s]->GetData() + vdofs[s]->Size());

         vdofs[s]->Copy(*vdofs_all[s]);

         const int lvsize = pfes->GetVSize();
         for (int j = 0; j < vdofs2[s]->Size(); j++)
         {
            if ((*vdofs2[s])[j] >= 0)
            {
               (*vdofs2[s])[j] += lvsize;
            }
            else
            {
               (*vdofs2[s])[j] -= lvsize;
            }
         }
         vdofs_all[s]->Append(*(vdofs2[s]));
      }

      for (int k = 0; k < fnfi.Size(); ++k)
      {
         fnfi[k]->AssembleFaceGrad(fe, fe2, *tr, el_x_const, elmats);

         for (int s1=0; s1<fes.Size(); ++s1)
         {
            for (int s2=0; s2<fes.Size(); ++s2)
            {
               if (elmats(s1,s2)->Height() == 0) { continue; }
               Grads(s1,s2)->AddSubMatrix(*vdofs[s1], *vdofs_all[s2],
                                          *elmats(s1,s2), skip_zeros);
            }
         }
      }
   }

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      delete pgfs[s1];
      delete vdofs_all[s1];
      delete vdofs2[s1];
      delete vdofs[s1];
      delete el_x[s1];
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         delete elmats(s1,s2);
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

   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   xs.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   if (fnfi.Size() > 0)
   {
      const int skip_zeros = 0;

      for (int s=0; s<fes.Size(); ++s)
      {
         const_cast<ParFiniteElementSpace*>(pfes[s])->ExchangeFaceNbrData();
      }

      for (int s1=0; s1<fes.Size(); ++s1)
      {
         for (int s2=0; s2<fes.Size(); ++s2)
         {
            if (Grads(s1,s2) == NULL)
            {
               int nbr_size = pfes[s2]->GetFaceNbrVSize();
               Grads(s1,s2) = new SparseMatrix(pfes[s1]->GetVSize(),
                                               pfes[s2]->GetVSize() + nbr_size);
            }
         }
      }

      // (re)assemble Grad without b.c. into 'Grads'
      BlockNonlinearForm::ComputeGradientBlocked(xs, false);

      GradientSharedFaces(xs, skip_zeros);

      // finalize the gradients
      for (int s1=0; s1<fes.Size(); ++s1)
         for (int s2=0; s2<fes.Size(); ++s2)
         {
            Grads(s1,s2)->Finalize(skip_zeros);
         }

      for (int s1=0; s1<fes.Size(); ++s1)
      {
         for (int s2=0; s2<fes.Size(); ++s2)
         {
            OperatorHandle hdA;
            OperatorHandle dA(phBlockGrad(s1,s2)->Type()),
                           Ph(phBlockGrad(s1,s2)->Type()),
                           Rh(phBlockGrad(s1,s2)->Type());

            // handle the case when 'a' contains off-diagonal
            int lvsize = pfes[s2]->GetVSize();
            const HYPRE_BigInt *face_nbr_glob_ldof =
               const_cast<ParFiniteElementSpace*>(pfes[s2])->GetFaceNbrGlobalDofMap();
            HYPRE_BigInt ldof_offset = pfes[s2]->GetMyDofOffset();

            Array<HYPRE_BigInt> glob_J(Grads(s1,s2)->NumNonZeroElems());
            int *J = Grads(s1,s2)->GetJ();
            for (int i = 0; i < glob_J.Size(); i++)
            {
               if (J[i] < lvsize)
               {
                  glob_J[i] = J[i] + ldof_offset;
               }
               else
               {
                  glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];
               }
            }

            // TODO - construct dA directly in the A format
            hdA.Reset(
               new HypreParMatrix(pfes[s2]->GetComm(), pfes[s1]->GetVSize(),
                                  pfes[s1]->GlobalVSize(), pfes[s2]->GlobalVSize(),
                                  Grads(s1,s2)->GetI(), glob_J, Grads(s1,s2)->GetData(),
                                  pfes[s1]->GetDofOffsets(), pfes[s2]->GetDofOffsets()));
            // - hdA owns the new HypreParMatrix
            // - the above constructor copies all input arrays
            glob_J.DeleteAll();
            dA.ConvertFrom(hdA);

            if (s1 == s2)
            {
               Ph.ConvertFrom(pfes[s1]->Dof_TrueDof_Matrix());
               phBlockGrad(s1,s1)->MakePtAP(dA, Ph);

               OperatorHandle Ae;
               Ae.EliminateRowsCols(*phBlockGrad(s1,s1), *ess_tdofs[s1]);
            }
            else
            {
               Rh.ConvertFrom(pfes[s1]->Dof_TrueDof_Matrix());
               Ph.ConvertFrom(pfes[s2]->Dof_TrueDof_Matrix());

               phBlockGrad(s1,s2)->MakeRAP(Rh, dA, Ph);

               phBlockGrad(s1,s2)->EliminateRows(*ess_tdofs[s1]);
               phBlockGrad(s1,s2)->EliminateCols(*ess_tdofs[s2]);
            }

            pBlockGrad->SetBlock(s1, s2, phBlockGrad(s1,s2)->Ptr());
         }
      }
   }
   else
   {
      // (re)assemble Grad without b.c. into 'Grads'
      BlockNonlinearForm::ComputeGradientBlocked(xs);

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
