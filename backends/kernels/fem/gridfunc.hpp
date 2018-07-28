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

#ifndef MFEM_BACKENDS_KERNELS_GRID_FUNC_HPP
#define MFEM_BACKENDS_KERNELS_GRID_FUNC_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

class IntegrationRule;
class GridFunction;

namespace kernels
{

class KernelsIntegrator;
class KernelsDofQuadMaps;

class KernelsGridFunction : public Vector
{
protected:
   kFiniteElementSpace &fes;
   long sequence;
   kvector v;
public:

   KernelsGridFunction(kFiniteElementSpace&);

   KernelsGridFunction(const KernelsGridFunction &gf);

   KernelsGridFunction& operator = (double value);
   KernelsGridFunction& operator = (const Vector &v);
   KernelsGridFunction& operator = (const KernelsGridFunction &gf);

   void GetTrueDofs(Vector &v);
   void SetFromTrueDofs(Vector &v);

   mfem::FiniteElementSpace* GetFESpace();
   const mfem::FiniteElementSpace* GetFESpace() const;

   void ToQuad(const mfem::IntegrationRule &ir, Vector &quadValues);

   void Distribute(const Vector &v);
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_GRID_FUNC_HPP
