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

#ifndef MFEM_FEM_GEOM_HPP
#define MFEM_FEM_GEOM_HPP

#include "../config/config.hpp"
#include "../general/array_ext.hpp"

#include "fespace.hpp"

namespace mfem
{

// ***************************************************************************
// * GeometryExtension
// ***************************************************************************
class GeometryExtension
{
public:
   kernels::Array<int> eMap;
   kernels::Array<double> meshNodes;
   kernels::Array<double> J, invJ, detJ;
   static const int Jacobian    = (1 << 0);
   static const int JacobianInv = (1 << 1);
   static const int JacobianDet = (1 << 2);
   static GeometryExtension* Get(const FiniteElementSpace&,
                                 const IntegrationRule&);
   static GeometryExtension* Get(const FiniteElementSpace&,
                                 const IntegrationRule&,
                                 const Vector&);
   static void ReorderByVDim(const GridFunction*);
   static void ReorderByNodes(const GridFunction*);
};

} // namespace mfem

#endif // MFEM_FEM_KERNEL_GEOM_HPP
