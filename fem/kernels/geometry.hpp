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

#ifndef MFEM_BILIN_GEOM_HPP
#define MFEM_BILIN_GEOM_HPP

#include "../../config/config.hpp"
#include "../../general/karray.hpp"

#include "../fespace.hpp"

namespace mfem
{

// ***************************************************************************
// * kGeometry
// ***************************************************************************
class kGeometry
{
public:
   ~kGeometry();
   karray<int> eMap;
   karray<double> meshNodes;
   karray<double> J, invJ, detJ;
   static const int Jacobian    = (1 << 0);
   static const int JacobianInv = (1 << 1);
   static const int JacobianDet = (1 << 2);
   static kGeometry* Get(const FiniteElementSpace&,
                         const IntegrationRule&);
   static kGeometry* Get(const FiniteElementSpace&,
                         const IntegrationRule&,
                         const Vector&);
   static void ReorderByVDim(GridFunction&);
   static void ReorderByNodes(GridFunction&);
};

} // namespace mfem

#endif // MFEM_BILIN_GEOM_HPP
