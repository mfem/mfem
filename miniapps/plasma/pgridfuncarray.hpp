// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PLASMA_PGFA
#define MFEM_PLASMA_PGFA

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

class ParGridFunctionArray : public Array<ParGridFunction*>
{
private:
   bool owns_data;

public:
   ParGridFunctionArray() : owns_data(true) {}
   ParGridFunctionArray(int size)
      : Array<ParGridFunction*>(size), owns_data(true) {}
   ParGridFunctionArray(int size, ParFiniteElementSpace *pf);

   ~ParGridFunctionArray();

   void SetOwner(bool owner) { owns_data = owner; }
   bool GetOwner() const { return owns_data; }

   void ProjectCoefficient(Array<Coefficient*> &coeff);

   void Update();

   void ExchangeFaceNbrData();
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PLASMA_PGFA
