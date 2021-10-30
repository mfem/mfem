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
   for (int i = 0; i<nblocks; i++)
   {
      nd += el[i]->GetDof();
   }

   elmat.SetSize(nd);

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
   for (int i = 0; i<nblocks; i++)
   {
      nd += el[i]->GetDof();
   }

   elvector.SetSize(nd);

   elvector = 1.0;
   // TODO

}

} // namespace mfem
