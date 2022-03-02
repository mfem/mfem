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

#ifndef MFEM_LOR_RESTRICTION
#define MFEM_LOR_RESTRICTION

#include "../bilinearform.hpp"

namespace mfem
{

/// Create a low-order refined version of a Restriction.
/// Only used here for the FillI and FillJAndZeroData methods.
class LORRestriction
{
   const FiniteElementSpace &fes_ho;
   FiniteElementCollection *fec_lo;
   const Geometry::Type geom;
   const int order;
   const int ne_ref;
   const int ne;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int lo_dof_per_el;

   Array<int> offsets;
   Array<int> indices;
   Array<int> gatherMap;

protected:
   static int GetNRefinedElements(const FiniteElementSpace &fes);
   static FiniteElementCollection *MakeLowOrderFEC(const FiniteElementSpace &fes);

public:
   LORRestriction(const FiniteElementSpace &fes_ho);

   void Setup();

   int FillI(SparseMatrix &mat) const;
   // TODO: Really should make a better version with Fill Data
   void FillJAndZeroData(SparseMatrix &mat) const;

   ~LORRestriction();
};

} // namespace mfem

#endif
