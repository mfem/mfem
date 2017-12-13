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
using namespace std;

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
   pfes->GetProlongationMatrix()->MultTranspose(*this, tv);
}

HypreParVector *ParLinearForm::ParallelAssemble()
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   pfes->GetProlongationMatrix()->MultTranspose(*this, *tv);
   return tv;
}


ParComplexLinearForm::ParComplexLinearForm(ParFiniteElementSpace *pf)
   : Vector(2*(pf->GetVSize()))
{
   plfr_ = new ParLinearForm(pf, &data[0]);
   plfi_ = new ParLinearForm(pf, &data[pf->GetVSize()]);

   HYPRE_Int * tdof_offsets = pf->GetTrueDofOffsets();

   int n = (HYPRE_AssumedPartitionCheck()) ? 2 : pf->GetNRanks();
   tdof_offsets_ = new HYPRE_Int[n+1];

   for (int i=0; i<=n; i++)
   {
      tdof_offsets_[i] = 2 * tdof_offsets[i];
   }
}

ParComplexLinearForm::~ParComplexLinearForm()
{
   delete plfr_;
   delete plfi_;
   delete [] tdof_offsets_;
}

void
ParComplexLinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                                          LinearFormIntegrator *lfi_imag)
{
   if ( lfi_real ) { plfr_->AddDomainIntegrator(lfi_real); }
   if ( lfi_imag ) { plfi_->AddDomainIntegrator(lfi_imag); }
}

void
ParComplexLinearForm::Update(ParFiniteElementSpace *pf)
{
   plfr_->Update(pf);
   plfi_->Update(pf);
}

void
ParComplexLinearForm::Assemble()
{
   plfr_->Assemble();
   plfi_->Assemble();
}

void
ParComplexLinearForm::ParallelAssemble(Vector &tv)
{
   HYPRE_Int size = plfr_->ParFESpace()->GetTrueVSize();

   double * tvd = tv.GetData();
   Vector tvr(tvd, size);
   Vector tvi(&tvd[size], size);

   plfr_->ParallelAssemble(tvr);
   plfi_->ParallelAssemble(tvi);
}

HypreParVector *
ParComplexLinearForm::ParallelAssemble()
{
   const ParFiniteElementSpace * pfes = plfr_->ParFESpace();

   HypreParVector * tv = new HypreParVector(pfes->GetComm(),
                                            2*(pfes->GlobalTrueVSize()),
                                            tdof_offsets_);

   HYPRE_Int size = pfes->GetTrueVSize();

   double * tvd = tv->GetData();
   Vector tvr(tvd, size);
   Vector tvi(&tvd[size], size);

   plfr_->ParallelAssemble(tvr);
   plfi_->ParallelAssemble(tvi);

   return tv;
}

complex<double>
ParComplexLinearForm::operator()(const ParComplexGridFunction &gf) const
{
   // return InnerProduct(plfr_->ParFESpace()->GetComm(), *this, gf);
   return 0.0;
}

}

#endif
