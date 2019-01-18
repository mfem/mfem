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

#ifndef MFEM_KFESPACE_HPP
#define MFEM_KFESPACE_HPP

#include "../config/config.hpp"

#include "../general/karray.hpp"

namespace mfem
{

// ***************************************************************************
// * kFiniteElementSpace
//  **************************************************************************
class kFiniteElementSpace
{
private:
   FiniteElementSpace *fes;
   int globalDofs, localDofs;
   karray<int> offsets;
   karray<int> indices, *reorderIndices;
   karray<int> map;
public:
   kFiniteElementSpace(FiniteElementSpace*);
   ~kFiniteElementSpace();
   void GlobalToLocal(const Vector&, Vector&) const;
   void LocalToGlobal(const Vector&, Vector&) const;
   FiniteElementSpace& GetFes() { return *fes; }
};

} // namespace mfem

#endif // MFEM_KFESPACE_HPP
