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

#ifndef MFEM_MESH_EXT
#define MFEM_MESH_EXT

namespace mfem
{

/// GeometryExtension
class GeometryExtension
{
private:
  long sequence;
public:
   Array<int> eMap;
   Array<double> nodes;
   Array<double> X, J, invJ, detJ;
   GeometryExtension():sequence(-1){}
   long& GetSequence() {return sequence;}
   static GeometryExtension* Get(Mesh*,
                                 const IntegrationRule&,
                                 GeometryExtension*& geom);
   static GeometryExtension* Get(Mesh*,
                                 const IntegrationRule&,
                                 const Vector&,
                                 GeometryExtension*& geom);
   static void ReorderByVDim(const GridFunction*);
   static void ReorderByNodes(const GridFunction*);
};

}

#endif