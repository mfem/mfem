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

#ifndef MFEM_BACKENDS_OCCA_GRID_FUNC_HPP
#define MFEM_BACKENDS_OCCA_GRID_FUNC_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "vector.hpp"
#include "fespace.hpp"

namespace mfem
{

class IntegrationRule;
class GridFunction;

namespace occa
{

class OccaIntegrator;
class OccaDofQuadMaps;

// TODO: make this object part of the backend or the engine.
extern std::map<std::string, ::occa::kernel> gridFunctionKernels;

// TODO: make this a method of the backend or the engine.
::occa::kernel GetGridFunctionKernel(::occa::device device,
                                     FiniteElementSpace &fespace,
                                     const mfem::IntegrationRule &ir);

// DEPRECATED: This used to be used to be an extension of
// mfem::GridFunction, esecially for use in OccaCoefficient, but this
// is now deprecated.
class OccaGridFunction : public Vector
{
protected:
   FiniteElementSpace *ofespace;
   long sequence;

   ::occa::kernel gridFuncToQuad[3];

public:
   // OccaGridFunction();

   OccaGridFunction(FiniteElementSpace *ofespace_);

   // OccaGridFunction(FiniteElementSpace *ofespace_,
   //                  OccaVectorRef ref);

   OccaGridFunction(const OccaGridFunction &gf);

   OccaGridFunction& operator = (double value);
   OccaGridFunction& operator = (const Vector &v);
   // OccaGridFunction& operator = (const OccaVectorRef &v);
   OccaGridFunction& operator = (const OccaGridFunction &gf);

   // void SetGridFunction(mfem::GridFunction &gf);

   void GetTrueDofs(Vector &v);
   void SetFromTrueDofs(Vector &v);

   mfem::FiniteElementSpace* GetFESpace();
   const mfem::FiniteElementSpace* GetFESpace() const;

   void ToQuad(const mfem::IntegrationRule &ir, Vector &quadValues);

   void Distribute(const Vector &v);
};

// ToQuad version without the deprecated class.
void ToQuad(const IntegrationRule &ir, FiniteElementSpace &ofespace, Vector &gf, Vector &quadValues);

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_GRID_FUNC_HPP
