// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "submesh.hpp"

using namespace mfem;

SubMesh::SubMesh(Mesh &parent, From from,
                 Array<int> attributes) : parent_(parent), from_(from),
   attributes_(attributes)
{
   SetEmpty();
    
   if (From::Domain) {
     Dim = parent.Dimension();
   } else {
     Dim = parent.Dimension() - 1;
   }
   spaceDim = parent.SpaceDimension();

   Array<int> element_ids;
   for (int i = 0; i < parent.GetNE(); i++)
   {
      Element *pel = parent.GetElement(i);
      int attr = pel->GetAttribute();

      bool in_subdomain = false;
      for (int i = 0; i < attributes.Size(); i++)
      {
         if (attr == attributes[i])
         {
            in_subdomain = true;
            break;
         }
      }

      if (!in_subdomain) continue;

      AddElement(NewElement(pel->GetType()));
   }

   SetMeshGen();
   CheckElementOrientation();
   Finalize();
}

SubMesh::~SubMesh() {}

SubMesh SubMesh::CreateFromDomain(Mesh &parent,
                                  Array<int> domain_attributes)
{
   return SubMesh(parent, From::Domain, domain_attributes);
}


SubMesh SubMesh::CreateFromBoundary(Mesh &parent,
                                    Array<int> boundary_attributes)
{
   Array<int> element_ids;
   return SubMesh(parent, From::Boundary, boundary_attributes);
}
