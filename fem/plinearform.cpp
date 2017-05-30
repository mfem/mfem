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

void ParLinearForm::Update(ParFiniteElementSpace *pf)
{
   if (pf) { pfes = pf; }

   LinearForm::Update(pfes);
}

void ParLinearForm::Update(ParFiniteElementSpace *pf, Vector &v, int v_offset)
{
   pfes = pf;
   LinearForm::Update(pf,v,v_offset);
}

void ParLinearForm::ParallelAssemble(Vector &tv)
{
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, tv);
}

HypreParVector *ParLinearForm::ParallelAssemble()
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, *tv);
   return tv;
}

void ParLinearForm::AssembleSharedFaces()
{
   int myid; 
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   ParMesh *pmesh = pfes->GetParMesh();
   FaceElementTransformations *T;
   Array<int> vdofs1, vdofs2, vdofs_all;
   Vector elemvect, this_vect;
   
   int nfaces = pmesh->GetNSharedFaces();
   for (int i = 0; i < nfaces; i++)
   {
      T = pmesh->GetSharedFaceTransformations(i);
      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      pfes->GetFaceNbrElementVDofs(T->Elem2No, vdofs2);

      Array<int> offset; 
      offset.SetSize(vdofs1.Size());
      for (int k = 0; k < vdofs1.Size(); k++) offset[k] = k; 

      for (int k = 0; k < ilfi.Size(); k++)
      {
          ilfi[k] -> AssembleRHSElementVect (*pfes->GetFE(T -> Elem1No),
                                             *pfes->GetFaceNbrFE(T -> Elem2No),
                                              *T, elemvect);

          //Assemble Shared Faces is called from each processor for the same faces
          //Therefore only the local vector on this processor needs to be updated 
          //Each processor will then update its own vector 
          elemvect.GetSubVector(offset, this_vect);
          AddElementVector (vdofs1, this_vect);
       }
    }
}


void ParLinearForm::Assemble()
{
   LinearForm::Assemble();

   if (ilfi.Size() > 0)
   {
      AssembleSharedFaces();
   }
}


}

#endif
