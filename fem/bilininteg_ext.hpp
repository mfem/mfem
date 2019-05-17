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

#ifndef MFEM_BILININTEG_EXT
#define MFEM_BILININTEG_EXT

#include "fespace.hpp"

namespace mfem
{

// FIXME: objects of this class should be owned by the Mesh objects?
/// GeometryExtension
class GeometryExtension
{
public:
   Array<int> eMap;
   Array<double> nodes;
   Array<double> X, J, invJ, detJ;
   static GeometryExtension* Get(const FiniteElementSpace&,
                                 const IntegrationRule&);
   static GeometryExtension* Get(const FiniteElementSpace&,
                                 const IntegrationRule&,
                                 const Vector&);
   static void ReorderByVDim(GridFunction*);
   static void ReorderByNodes(GridFunction*);
};

// FIXME: this class should store B and G for one FiniteElement and objects of
// this class should be owned by the respective FiniteElement objects.
/// DofToQuad
class DofToQuad
{
private:
   std::string hash;

public:
   Array<double> W, B, G, Bt, Gt;

   ~DofToQuad();

   static DofToQuad* Get(const FiniteElementSpace&,
                         const IntegrationRule&,
                         const bool = false);
   static DofToQuad* Get(const FiniteElementSpace&,
                         const FiniteElementSpace&,
                         const IntegrationRule&,
                         const bool = false);
   static DofToQuad* Get(const FiniteElement&,
                         const FiniteElement&,
                         const IntegrationRule&,
                         const bool = false);
   static DofToQuad* GetTensorMaps(const FiniteElement&,
                                   const FiniteElement&,
                                   const IntegrationRule&,
                                   const bool = false);
   static DofToQuad* GetD2QTensorMaps(const FiniteElement&,
                                      const IntegrationRule&,
                                      const bool = false);
   static DofToQuad* GetSimplexMaps(const FiniteElement&,
                                    const IntegrationRule&,
                                    const bool = false);
   static DofToQuad* GetSimplexMaps(const FiniteElement&,
                                    const FiniteElement&,
                                    const IntegrationRule&,
                                    const bool = false);
   static DofToQuad* GetD2QSimplexMaps(const FiniteElement&,
                                       const IntegrationRule&,
                                       const bool = false);
};

}

#endif
