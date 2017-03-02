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

double ParLinearForm::operator()(const ParGridFunction &gf) const
{
   double loc_prod = this->Vector::operator*(gf);
   double glb_prod;
   MPI_Allreduce(&loc_prod, &glb_prod, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());
   return glb_prod;
}

}

#endif
