// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fefunctional.hpp"

namespace mfem
{

namespace optimization
{

BilinearFunctional::BilinearFunctional(BilinearForm *blf_, LinearForm *lf_,
                                       bool own_blf_, bool own_lf_)
   : QuadraticFunctional(blf_, lf_),
     blf(blf_), own_blf(own_blf_), lf(lf_), own_lf(own_lf_)
{
   // A and b are set by Assemble().
   A = nullptr;
   b = nullptr;
   // blf->Width() is vdof size, not tdof size.
   field_sizes[0] = blf->FESpace()->GetTrueVSize();
   if (lf_ != nullptr)
   {
      MFEM_VERIFY(lf_->FESpace() == blf_->FESpace(),
                  "BilinearFunctional: lf and blf must have the same FESpace.");
   }
#ifdef MFEM_USE_MPI
   if ((par_blf = dynamic_cast<ParBilinearForm*>(blf_)))
   {
      SetComm(par_blf->ParFESpace()->GetComm());
      par_lf = dynamic_cast<ParLinearForm*>(lf_);
      MFEM_VERIFY(!(lf != nullptr && par_lf == nullptr),
                  "BilinearFunctional: lf must be ParLinearForm if blf is ParBilinearForm.");
   }
#endif
}

BilinearFunctional::~BilinearFunctional()
{
   if (own_blf) { delete blf; }
   if (own_lf) { delete lf; }
}

void BilinearFunctional::Assemble(int skip_zeros)
{
   // if assembled, then refresh
   if (A != nullptr)
   {
      blf->Update();
      if (lf) { lf->Update(); }
   }
   field_sizes[0] = blf->FESpace()->GetTrueVSize();
#ifdef MFEM_USE_MPI
   if (par_blf)
   {
      par_blf->Assemble(skip_zeros); // Assemble is non-virtual
      par_blf->Finalize();
      if (par_lf)
      {
         par_lf->Assemble();
         B.SetSize(par_blf->ParFESpace()->GetTrueVSize());
         par_lf->ParallelAssemble(B);
         b = &B;
      }
   }
   else
#endif
   {
      blf->Assemble(skip_zeros);
      blf->Finalize();
      if (lf)
      {
         lf->Assemble();
         if (auto *P = lf->FESpace()->GetProlongationMatrix())
         { B.SetSize(P->Width()); P->MultTranspose(*lf, B); }
         else { B.MakeRef(*lf, 0, lf->Size()); }
         b = &B;
      }
   }
   blf->FormSystemMatrix(Array<int>(), matrix);
   A = matrix.Ptr();
}

} // namespace optimization

} // namespace mfem
