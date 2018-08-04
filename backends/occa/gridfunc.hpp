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

namespace occa
{

// TODO: make this object part of the backend or the engine.
extern std::map<std::string, ::occa::kernel> gridFunctionKernels;

// TODO: make this a method of the backend or the engine.
::occa::kernel GetGridFunctionKernel(::occa::device device,
                                     FiniteElementSpace &fespace,
                                     const mfem::IntegrationRule &ir);

// ToQuad version without the deprecated class.
//
// FIXME: This is the action of a global B matrix, mapping L-vector to Q-vector,
//        so it should be made into an operator that can be constructed by the
//        FE space class. A batched version, where only a subset of the elements
//        are processed should be defined as well.
//
//        The abstract operator construction method in the FE space class is:
//           PFiniteElementSpace::GetInterpolationOperator(...)
void ToQuad(const IntegrationRule &ir, FiniteElementSpace &ofespace, Vector &gf,
            Vector &quadValues);

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_GRID_FUNC_HPP
