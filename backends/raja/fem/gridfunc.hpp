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

#ifndef MFEM_BACKENDS_RAJA_GRID_FUNC_HPP
#define MFEM_BACKENDS_RAJA_GRID_FUNC_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#include "../linalg/vector.hpp"
#include "../fem/fespace.hpp"

namespace mfem
{

class IntegrationRule;
class GridFunction;

namespace raja
{

class RajaIntegrator;
class RajaDofQuadMaps;

//extern std::map<std::string, raja::kernel> gridFunctionKernels;

/*::raja::kernel GetGridFunctionKernel(::raja::device device,
                                     FiniteElementSpace &fespace,
                                     const mfem::IntegrationRule &ir);
*/
class RajaGridFunction : public Vector
{
protected:
   FiniteElementSpace *ofespace;
   long sequence;

   //::raja::kernel gridFuncToQuad[3];

public:
   // RajaGridFunction();

   RajaGridFunction(FiniteElementSpace *ofespace_);

   // RajaGridFunction(FiniteElementSpace *ofespace_,
   //                  RajaVectorRef ref);

   RajaGridFunction(const RajaGridFunction &gf);

   RajaGridFunction& operator = (double value);
   RajaGridFunction& operator = (const Vector &v);
   // RajaGridFunction& operator = (const RajaVectorRef &v);
   RajaGridFunction& operator = (const RajaGridFunction &gf);

   // void SetGridFunction(mfem::GridFunction &gf);

   void GetTrueDofs(Vector &v);
   void SetFromTrueDofs(Vector &v);

   mfem::FiniteElementSpace* GetFESpace();
   const mfem::FiniteElementSpace* GetFESpace() const;

   void ToQuad(const mfem::IntegrationRule &ir, Vector &quadValues);

   void Distribute(const Vector &v);
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_GRID_FUNC_HPP
