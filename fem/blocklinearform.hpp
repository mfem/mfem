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

#ifndef MFEM_BLOCKLINEARFORM
#define MFEM_BLOCKLINEARFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"

namespace mfem
{


class BlockLinearForm : public Vector
{
protected:
   /// FE spaces on which the LinearForm lives. Not owned.
   Array<FiniteElementSpace * > fespaces;

   /// Set of Domain Integrators to be applied.
   Array<BlockLinearFormIntegrator*> domain_integs;

   Vector elemvect;
   Array<int> vdofs;

public:
   BlockLinearForm(Array<FiniteElementSpace * > & fespaces_);

   /// Adds new Domain Integrator. Assumes ownership of @a lfi.
   void AddDomainIntegrator(BlockLinearFormIntegrator *lfi);

   /// Assembles the block linear form i.e. sums over all domain integrators.
   void Assemble();


};

} // namespace mfem


#endif