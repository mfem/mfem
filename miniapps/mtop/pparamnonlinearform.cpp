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

#include "mfem.hpp"
#include "pparamnonlinearform.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

ParParametricBNLForm::ParParametricBNLForm(Array<ParFiniteElementSpace *>
                                           &statef,
                                           Array<ParFiniteElementSpace *> &paramf)
   :ParametricBNLForm()
{
   pBlockGrad = nullptr;
   SetParSpaces(statef,paramf);
}

void ParParametricBNLForm::SetParSpaces(Array<ParFiniteElementSpace *> &statef,
                                        Array<ParFiniteElementSpace *> &paramf)
{
   delete pBlockGrad;
   pBlockGrad = nullptr;

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         delete phBlockGrad(s1,s2);
      }
   }

   Array<FiniteElementSpace *> serialSpaces(statef.Size());
   Array<FiniteElementSpace *> prmserialSpaces(paramf.Size());
   for (int s=0; s<statef.Size(); s++)
   {
      serialSpaces[s] = (FiniteElementSpace *) statef[s];
   }
   for (int s=0; s<paramf.Size(); s++)
   {
      prmserialSpaces[s] = (FiniteElementSpace *) paramf[s];
   }

   SetSpaces(serialSpaces,prmserialSpaces);

   phBlockGrad.SetSize(fes.Size(), fes.Size());

   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         phBlockGrad(s1,s2) = new OperatorHandle(Operator::Hypre_ParCSR);
      }
   }
}

ParFiniteElementSpace * ParParametricBNLForm::ParFESpace(int k)
{
   return (ParFiniteElementSpace *)fes[k];
}

const ParFiniteElementSpace *ParParametricBNLForm::ParFESpace(int k) const
{
   return (const ParFiniteElementSpace *)fes[k];
}


ParFiniteElementSpace * ParParametricBNLForm::ParParamFESpace(int k)
{
   return (ParFiniteElementSpace *)paramfes[k];
}

const ParFiniteElementSpace *ParParametricBNLForm::ParParamFESpace(int k) const
{
   return (const ParFiniteElementSpace *)paramfes[k];
}

// Here, rhs is a true dof vector
void ParParametricBNLForm::SetEssentialBC(const
                                          Array<Array<int> *>&bdr_attr_is_ess,
                                          Array<Vector *> &rhs)
{
   Array<Vector *> nullarray(fes.Size());
   nullarray = NULL;

   ParametricBNLForm::SetEssentialBC(bdr_attr_is_ess, nullarray);

   for (int s = 0; s < fes.Size(); ++s)
   {
      if (rhs[s])
      {
         rhs[s]->SetSubVector(*ess_tdofs[s], 0.0);
      }
   }
}

void ParParametricBNLForm::SetParamEssentialBC(const
                                               Array<Array<int> *>&bdr_attr_is_ess,
                                               Array<Vector *> &rhs)
{
   Array<Vector *> nullarray(fes.Size());
   nullarray = NULL;

   ParametricBNLForm::SetParamEssentialBC(bdr_attr_is_ess, nullarray);

   for (int s = 0; s < paramfes.Size(); ++s)
   {
      if (rhs[s])
      {
         rhs[s]->SetSubVector(*paramess_tdofs[s], 0.0);
      }
   }
}

double ParParametricBNLForm::GetEnergy(const Vector &x) const
{
   xs_true.Update(x.GetData(), block_trueOffsets);
   xs.Update(block_offsets);

   for (int s = 0; s < fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(xs_true.GetBlock(s), xs.GetBlock(s));
   }

   double enloc = ParametricBNLForm::GetEnergyBlocked(xs,xdv);
   double englo = 0.0;

   MPI_Allreduce(&enloc, &englo, 1, MPI_DOUBLE, MPI_SUM,
                 ParFESpace(0)->GetComm());

   return englo;
}

void ParParametricBNLForm::Mult(const Vector &x, Vector &y) const
{
   xs_true.Update(x.GetData(), block_trueOffsets);
   ys_true.Update(y.GetData(), block_trueOffsets);
   xs.Update(block_offsets);
   ys.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   ParametricBNLForm::MultBlocked(xs, xdv, ys);

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
}

/// Block T-Vector to Block T-Vector
void ParParametricBNLForm::ParamMult(const Vector &x, Vector &y) const
{
   xs_true.Update(x.GetData(), paramblock_trueOffsets);
   ys_true.Update(y.GetData(), paramblock_trueOffsets);
   prmxs.Update(paramblock_offsets);
   prmys.Update(paramblock_offsets);

   for (int s=0; s<paramfes.Size(); ++s)
   {
      paramfes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), prmxs.GetBlock(s));
   }

   ParametricBNLForm::MultParamBlocked(xsv,adv,xdv,prmys);

   if (fnfi.Size() > 0)
   {
      MFEM_ABORT("TODO: assemble contributions from shared face terms");
   }

   for (int s=0; s<paramfes.Size(); ++s)
   {
      paramfes[s]->GetProlongationMatrix()->MultTranspose(
         prmys.GetBlock(s), ys_true.GetBlock(s));

      ys_true.GetBlock(s).SetSubVector(*paramess_tdofs[s], 0.0);
   }

}

/// Return the local gradient matrix for the given true-dof vector x
const BlockOperator & ParParametricBNLForm::GetLocalGradient(
   const Vector &x) const
{
   xs_true.Update(x.GetData(), block_trueOffsets);
   xs.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   ParametricBNLForm::ComputeGradientBlocked(xs,
                                             xdv); // (re)assemble Grad with b.c.

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
void ParParametricBNLForm::SetGradientType(Operator::Type tid)
{
   for (int s1=0; s1<fes.Size(); ++s1)
   {
      for (int s2=0; s2<fes.Size(); ++s2)
      {
         phBlockGrad(s1,s2)->SetType(tid);
      }
   }
}

BlockOperator & ParParametricBNLForm::GetGradient(const Vector &x) const
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

ParParametricBNLForm::~ParParametricBNLForm()
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


void ParParametricBNLForm::SetStateFields(const Vector &xv) const
{
   xs_true.Update(xv.GetData(), block_trueOffsets);
   xsv.Update(block_offsets);
   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xsv.GetBlock(s));
   }
}


void ParParametricBNLForm::SetAdjointFields(const Vector &av) const
{
   xs_true.Update(av.GetData(), block_trueOffsets);
   adv.Update(block_offsets);
   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), adv.GetBlock(s));
   }
}

void ParParametricBNLForm::SetParamFields(const Vector &dv) const
{
   xs_true.Update(dv.GetData(),paramblock_trueOffsets);
   xdv.Update(paramblock_offsets);
   for (int s=0; s<paramfes.Size(); ++s)
   {
      paramfes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xdv.GetBlock(s));
   }
}


}

#endif
