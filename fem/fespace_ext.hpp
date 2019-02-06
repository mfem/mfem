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

#ifndef MFEM_FESPACE_EXT
#define MFEM_FESPACE_EXT

#include "../config/config.hpp"

#include "../general/array.hpp"

namespace mfem
{

// ***************************************************************************
// * FiniteElementSpaceExtension
// ***************************************************************************
class FiniteElementSpaceExtension
{
public:
   const FiniteElementSpace &fes;
   const int ne;
   const int vdim;
   const bool byvdim;
   const int NDofs;
   const int Dof;
   const int neDofs;
   Array<int> offsets;
   Array<int> indices;
public:
   FiniteElementSpaceExtension(const FiniteElementSpace&);
   void L2E(const Vector&, Vector&) const;
   void E2L(const Vector&, Vector&) const;
};

} // namespace mfem

#endif // MFEM_KFESPACE_EXT
