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

void BlockBilinearFormIntegrator::AssembleElementMatrix(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   mfem_error ("BlockBilinearFormIntegrator::AssembleElementMatrix\n"
               "   is not implemented for this class.");
}

void BlockLinearFormIntegrator::AssembleRHSElementVect(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Trans,
   Vector &elvect)
{
   mfem_error ("BlockLinearFormIntegrator::AssembleElementVector\n"
               "   is not implemented for this class.");
}

/** Given a particular Finite Element computes the element vector */
void TestBlockBilinearFormIntegrator::AssembleElementMatrix
(const Array<const FiniteElement *> &el,
 ElementTransformation &Trans,
 DenseMatrix &elmat)
{
   int nd = 0;
   int nblocks = el.Size();
   Array<int> offsets(nblocks+1);
   offsets[0] = 0;
   for (int i = 0; i<nblocks; i++)
   {
      nd += el[i]->GetDof();
      offsets[i+1] = el[i]->GetDof();
   }
   offsets.PartialSum();
   elmat.SetSize(nd);
   elmat = 0.0;
   DenseMatrix dmat;

   if (blfis.NumRows())
   {
      // Get the matrices directly from the existing BilinearFormIntegrators
      for (int i = 0; i<nblocks; i++)
      {
         // mfem::out << "i = " << i << std::endl;
         int offset_i = offsets[i];
         const FiniteElement * fe_i = el[i];
         for (int j = 0; j<nblocks; j++)
         {
            // mfem::out << "j = " << j << std::endl;
            BilinearFormIntegrator * blfi = blfis(i,j);
            if (!blfi) { continue; }
            if (j == i)
            {
               blfi->AssembleElementMatrix(*fe_i,Trans,dmat);
               // mfem::out << "j 1 = " << j << std::endl;
               elmat.SetSubMatrix(offset_i,dmat);
            }
            else
            {
               const FiniteElement * fe_j = el[j];
               blfi->AssembleElementMatrix2(*fe_j,*fe_i,Trans,dmat);
               // mfem::out << "j 2 = " << j << std::endl;
               int offset_j = offsets[j];
               elmat.SetSubMatrix(offset_i,offset_j,dmat);
            }
         }
      }
      return;
   }

   // else compute the matrices
   elmat = 25.0;
   // TODO

}

/** Given a particular Finite Element computes the element vector */
void TestBlockLinearFormIntegrator::AssembleRHSElementVect
(const Array<const FiniteElement *> &el,
 ElementTransformation &Trans,
 Vector &elvector)
{
   int nd = 0;
   int nblocks = el.Size();
   Array<int> offsets(nblocks+1);
   offsets[0] = 0;
   for (int i = 0; i<nblocks; i++)
   {
      nd += el[i]->GetDof();
      offsets[i+1] = el[i]->GetDof();
   }
   offsets.PartialSum();
   elvector.SetSize(nd);
   elvector = 0.0;
   Vector subvector;

   if (lfis.Size())
   {
      // Get the matrices directly from the existing BilinearFormIntegrators
      for (int i = 0; i<nblocks; i++)
      {
         int offset = offsets[i];
         const FiniteElement * fe_i = el[i];
         LinearFormIntegrator * lfi = lfis[i];
         if (!lfi)
         {
            continue;
         }
         lfi->AssembleRHSElementVect(*fe_i,Trans,subvector);
         elvector.SetVector(subvector,offset);
      }
      return;
   }

   // else, compute the block linear form integrator
   // elvector = 1.0;
   // TODO
}

} // namespace mfem
