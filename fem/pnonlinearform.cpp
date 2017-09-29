// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"

namespace mfem
{

void ParNonlinearForm::SetEssentialBC(const Array<int> &bdr_attr_is_ess,
                                      Vector *rhs)
{
   ParFiniteElementSpace *pfes = ParFESpace();

   NonlinearForm::SetEssentialBC(bdr_attr_is_ess);

   // ess_vdofs is a list of local vdofs
   if (rhs)
   {
      for (int i = 0; i < ess_vdofs.Size(); i++)
      {
         int tdof = pfes->GetLocalTDofNumber(ess_vdofs[i]);
         if (tdof >= 0)
         {
            (*rhs)(tdof) = 0.0;
         }
      }
   }
}

double ParNonlinearForm::GetEnergy(const ParGridFunction &x) const
{
   double loc_energy, glob_energy;

   loc_energy = NonlinearForm::GetEnergy(x);

   MPI_Allreduce(&loc_energy, &glob_energy, 1, MPI_DOUBLE, MPI_SUM,
                 ParFESpace()->GetComm());

   return glob_energy;
}

double ParNonlinearForm::GetEnergy(const Vector &x) const
{
   X.Distribute(&x);
   return GetEnergy(X);
}

void ParNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   const Operator *P = ParFESpace()->GetProlongationMatrix();

   P->Mult(x, X);

   NonlinearForm::Mult(X, Y);

   P->MultTranspose(Y, y);
}

const SparseMatrix &ParNonlinearForm::GetLocalGradient(const Vector &x) const
{
   X.Distribute(&x);

   NonlinearForm::GetGradient(X); // (re)assemble Grad with b.c.

   return *Grad;
}

Operator &ParNonlinearForm::GetGradient(const Vector &x) const
{
   ParFiniteElementSpace *pfes = ParFESpace();

   pGrad.Clear();

   X.Distribute(&x);

   NonlinearForm::GetGradient(X); // (re)assemble Grad with b.c.

   OperatorHandle dA(pGrad.Type()), Ph(pGrad.Type());
   dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                          pfes->GetDofOffsets(), Grad);
   // TODO - construct Dof_TrueDof_Matrix directly in the pGrad format
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   pGrad.MakePtAP(dA, Ph);

   return *pGrad.Ptr();
}

ParBlockNonlinearForm::ParBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf) :
   BlockNonlinearForm()
{
   height = 0;
   width = 0;
   pBlockGrad = NULL;
   
   Array<FiniteElementSpace *> serialSpaces(pf.Size());

   for (int s=0; s<pf.Size(); s++) {
      serialSpaces[s] = (FiniteElementSpace *) pf[s];
   }

   SetSpaces(serialSpaces);
   
   X.SetSize(fes.Size());
   Y.SetSize(fes.Size());
   phBlockGrad.SetSize(fes.Size(), fes.Size());

   for (int s=0; s<fes.Size(); s++) {
      X[s] = new ParGridFunction(pf[s]);
      Y[s] = new ParGridFunction(pf[s]);
   }
   
   for (int s1=0; s1<fes.Size(); s1++) {
      for (int s2=0; s2<fes.Size(); s2++) {
         phBlockGrad(s1,s2) = new OperatorHandle(Operator::HYPRE_PARCSR);
      }
   }

   for (int i=0; i<fes.Size(); i++) {
      height += fes[i]->GetTrueVSize();
      width += fes[i]->GetTrueVSize();
   }
}

ParFiniteElementSpace * ParBlockNonlinearForm::ParFESpace(int block) const
{
   return (ParFiniteElementSpace *)fes[block];
}

   // Here, rhs is a true dof vector
void ParBlockNonlinearForm::SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                                           Array<Vector *> &rhs)
{
   
   Array<Vector *> nullarray(fes.Size());
   nullarray = NULL;

   BlockNonlinearForm::SetEssentialBC(bdr_attr_is_ess, nullarray);

   for (int s=0; s<fes.Size(); s++) {
      if (rhs[s]) {
         ParFiniteElementSpace *pfes = ParFESpace(s);
         for (int i=0; i < ess_vdofs[s]->Size(); i++) {
            int tdof = pfes->GetLocalTDofNumber((*(ess_vdofs[s]))[i]);
            if (tdof >= 0) {
               (*rhs[s])(tdof) = 0.0;
            }
         }
      }
   }
}
   
void ParBlockNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   Vector xs_true, ys_true;
   BlockVector bxs(block_offsets), bys(block_offsets);


   for (int s=0; s<fes.Size(); s++) {
      xs_true.SetDataAndSize(x.GetData() + block_trueOffsets[s], fes[s]->GetTrueVSize());

      X[s]->Distribute(xs_true);
      
      bxs.SetVector(*X[s], block_offsets[s]);
   }

   BlockNonlinearForm::Mult(bxs,bys);

   for (int s=0; s<fes.Size(); s++) {
      ParFESpace(s)->GroupComm().Reduce<double>(bys, GroupCommunicator::Sum);
      *Y[s] = bys.GetBlock(s);
      ys_true.SetDataAndSize(y.GetData() + block_trueOffsets[s], fes[s]->GetTrueVSize());
      Y[s]->GetTrueDofs(ys_true);
   }
}

   /// Return the local gradient matrix for the given true-dof vector x
const BlockOperator & ParBlockNonlinearForm::GetLocalGradient(const Vector &x) const
{
   Vector xs_true;
   BlockVector bxs(block_offsets);

   for (int s=0; s<fes.Size(); s++) {
      xs_true.SetDataAndSize(x.GetData() + block_offsets[s], fes[s]->GetTrueVSize());
      X[s]->Distribute(xs_true);

      bxs.SetVector(*X[s], block_offsets[s]);
   }

   BlockNonlinearForm::GetGradient(bxs); // (re)assemble Grad with b.c.

   return *BlockGrad;

}

/// Set the operator type id for the parallel gradient matrix/operator.
void ParBlockNonlinearForm::SetGradientType(Operator::Type tid) 
{ 
   for (int s1=0; s1<fes.Size(); s1++) {
      for (int s2=0; s2<fes.Size(); s2++) {
         phBlockGrad(s1,s2)->SetType(tid); 
      }
   }
}


BlockOperator & ParBlockNonlinearForm::GetGradient(const Vector &x) const
{
   if (pBlockGrad == NULL) {
      pBlockGrad = new BlockOperator(block_trueOffsets);
   }

   Array<ParFiniteElementSpace *> pfes(fes.Size());

   Vector xs_true;
   BlockVector bxs(block_offsets);

   for (int s1=0; s1<fes.Size(); s1++) {
      xs_true.SetDataAndSize(x.GetData() + block_trueOffsets[s1], fes[s1]->GetTrueVSize());
      X[s1]->Distribute(xs_true);
      bxs.SetVector(*X[s1], block_offsets[s1]);
      pfes[s1] = ParFESpace(s1);

      for (int s2=0; s2<fes.Size(); s2++) {
         phBlockGrad(s1,s2)->Clear();
      }
   }
   
   BlockNonlinearForm::GetGradient(bxs); // (re)assemble Grad with b.c.

   for (int s1=0; s1<fes.Size(); s1++) {
      for (int s2=0; s2<fes.Size(); s2++) {
         OperatorHandle dA(phBlockGrad(s1,s2)->Type()), Ph(phBlockGrad(s1,s2)->Type()), Rh(phBlockGrad(s1,s2)->Type());
         
         if (s1 == s2) {
            dA.MakeSquareBlockDiag(pfes[s1]->GetComm(), pfes[s1]->GlobalVSize(),
                                   pfes[s1]->GetDofOffsets(), Grads(s1,s1));
            Ph.ConvertFrom(pfes[s1]->Dof_TrueDof_Matrix());
            phBlockGrad(s1,s1)->MakePtAP(dA, Ph);
         }
         else {
            dA.MakeRectangularBlockDiag(pfes[s1]->GetComm(), 
                                        pfes[s1]->GlobalVSize(),
                                        pfes[s2]->GlobalVSize(), 
                                        pfes[s1]->GetDofOffsets(),
                                        pfes[s2]->GetDofOffsets(), 
                                        Grads(s1,s2));
            Rh.ConvertFrom(pfes[s1]->Dof_TrueDof_Matrix());
            Ph.ConvertFrom(pfes[s2]->Dof_TrueDof_Matrix());
  
            phBlockGrad(s1,s2)->MakeRAP(Rh, dA, Ph);
         }

         pBlockGrad->SetBlock(s1, s2, phBlockGrad(s1,s2)->Ptr());
      }
   }

   return *pBlockGrad;

}

ParBlockNonlinearForm::~ParBlockNonlinearForm() 
{
   for (int s1=0; s1<fes.Size(); s1++) {
      delete X[s1];
      delete Y[s1];
   }
}

}

#endif
