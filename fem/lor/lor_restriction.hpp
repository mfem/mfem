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
   const int ne_ref;
   const int ne;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int dof;

   Array<int> offsets;
   Array<int> indices;
   Array<int> gatherMap;

   Array<int> dof_glob2loc;
   Array<int> dof_glob2loc_offsets;
   Array<int> el_dof_lex;

protected:
   static int GetNRefinedElements(const FiniteElementSpace &fes);
   static FiniteElementCollection *GetLowOrderFEC(const FiniteElementSpace &fes);

public:
   LORRestriction(const FiniteElementSpace &fes_ho);

   int FillI(SparseMatrix &mat) const;
   void FillJAndZeroData(SparseMatrix &mat) const;

   const Array<int> &GatherMap() const { return el_dof_lex; }
   const Array<int> &Indices() const { return dof_glob2loc; }
   const Array<int> &Offsets() const { return dof_glob2loc_offsets; }

   ~LORRestriction();

   // Device lambda cannot have private or protected access
public:
   void SetupLocalToElement();
   void SetupGlobalToLocal();
};

} // namespace mfem

#endif
