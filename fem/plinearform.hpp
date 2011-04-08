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

#ifndef MFEM_PLINEARFORM
#define MFEM_PLINEARFORM

/// Class for parallel linear form
class ParLinearForm : public LinearForm
{
protected:
   ParFiniteElementSpace *pfes;

public:
   ParLinearForm(ParFiniteElementSpace *pf) : LinearForm(pf) { pfes = pf; }

   void Update(ParFiniteElementSpace *pf = NULL);

   /// Returns the vector assembled on the true dofs, i.e. P^t v.
   HypreParVector *ParallelAssemble();
};

#endif
