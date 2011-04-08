// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_PBILINEARFORM
#define MFEM_PBILINEARFORM

/// Class for parallel bilinear form
class ParBilinearForm : public BilinearForm
{
protected:
   ParFiniteElementSpace *pfes;

public:
   ParBilinearForm(ParFiniteElementSpace *pf)
      : BilinearForm(pf) { pfes = pf; }

   ParBilinearForm(ParFiniteElementSpace *pf, ParBilinearForm *bf)
      : BilinearForm(pf, bf) { pfes = pf; }

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   HypreParMatrix *ParallelAssemble();

   virtual ~ParBilinearForm() { }
};

#endif
