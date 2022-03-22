// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"

namespace mfem
{

BlockLinearForm::BlockLinearForm(Array<FiniteElementSpace * > & fespaces_) :
   Vector(0), fespaces(fespaces_)
{
   int s = 0;
   int nblocks = fespaces.Size();
   for (int i =0; i<nblocks; i++)
   {
      s += fespaces[i]->GetVSize();
   }
   // mfem::out << "size = " << size << std::endl;

   SetSize(s);

}


void BlockLinearForm::AddDomainIntegrator(BlockLinearFormIntegrator *lfi)
{
   domain_integs.Append(lfi);
}

void BlockLinearForm::Assemble()
{
   ElementTransformation *eltrans;
   DofTransformation *doftrans;
   Mesh *mesh = fespaces[0] -> GetMesh();
   Vector subvect,elvect, *elvect_p;

   int nblocks = fespaces.Size();
   Array<const FiniteElement *> fe(nblocks);
   Array<int> offsetvdofs;
   Array<int> elementblockoffsets(nblocks+1);
   elementblockoffsets[0] = 0;
   Array<int> blockoffsets(nblocks+1);
   blockoffsets[0] = 0;
   for (int i =0; i<nblocks; i++)
   {
      blockoffsets[i+1] = fespaces[i]->GetVSize();
   }
   blockoffsets.PartialSum();

   Vector::operator=(0.0);

   if (domain_integs.Size())
   {
      // loop through elements
      for (int i = 0; i < mesh -> GetNE(); i++)
      {
         elvect.SetSize(0);
         for (int k = 0; k < domain_integs.Size(); k++)
         {
            for (int j = 0; j<nblocks; j++)
            {
               fe[j] = fespaces[j]->GetFE(i);
               elementblockoffsets[j+1] = fe[j]->GetDof();
            }
            elementblockoffsets.PartialSum();
            eltrans = mesh->GetElementTransformation(i);

            domain_integs[k]->AssembleRHSElementVect(fe, *eltrans, elemvect);
            if (elvect.Size() == 0)
            {
               elvect = elemvect;
            }
            else
            {
               elvect += elemvect;
            }
         }
         if (elvect.Size() == 0)
         {
            continue;
         }
         else
         {
            elvect_p = &elvect;
         }

         double *data = elvect_p->GetData();

         for (int j = 0; j<nblocks; j++)
         {
            doftrans = fespaces[j]->GetElementVDofs(i, vdofs);
            int offset = blockoffsets[j];
            offsetvdofs.SetSize(vdofs.Size());
            for (int l = 0; l<vdofs.Size(); l++)
            {
               offsetvdofs[l] = vdofs[l]<0 ? -offset + vdofs[l]
                                :  offset + vdofs[l];
            }
            int jbeg = elementblockoffsets[j];
            int jend = elementblockoffsets[j+1]-1;
            subvect.SetSize(jend-jbeg+1);
            subvect.SetData(&data[jbeg]);

            if (doftrans)
            {
               doftrans->TransformDual(subvect);
            }
            AddElementVector(offsetvdofs,subvect);
         }
      }
   }
}


} // name space mfem
