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

#include <unordered_map>
#include "submesh.hpp"

using namespace mfem;

struct UniqueIndexGenerator
{
   int counter = 0;
   std::unordered_map<int,int> idx;

   int Get(int i, bool &new_index)
   {
      auto f = idx.find(i);
      if (f == idx.end())
      {
         idx[i] = counter;
         new_index = true;
         return counter++;
      }
      else
      {
         new_index = false;
         return (*f).second;
      }
   }
};

SubMesh::SubMesh(Mesh &parent, From from,
                 Array<int> attributes) : parent_(parent), from_(from), attributes_(attributes)
{
   SetEmpty();
   NumOfVertices = 0;

   if (From::Domain)
   {
      Dim = parent.Dimension();
   }
   else
   {
      Dim = parent.Dimension() - 1;
   }
   spaceDim = parent.SpaceDimension();

   UniqueIndexGenerator vertex_ids;
   for (int i = 0; i < parent.GetNE(); i++)
   {
      Element *pel = parent.GetElement(i);
      int attr = pel->GetAttribute();

      bool in_subdomain = false;
      for (int a = 0; a < attributes.Size(); a++)
      {
         if (attr == attributes[a])
         {
            in_subdomain = true;
            break;
         }
      }
      if (in_subdomain)
      {
         Array<int> v;
         pel->GetVertices(v);
         Array<int> submesh_v(v.Size());

         for (int iv = 0; iv < v.Size(); iv++)
         {
            bool new_vertex;
            int mesh_vertex_id = v[iv];
            int submesh_vertex_id = vertex_ids.Get(mesh_vertex_id, new_vertex);
            if (new_vertex)
            {
               AddVertex(parent.GetVertex(mesh_vertex_id));
            }
            submesh_v[iv] = submesh_vertex_id;
         }

         Element *parent_el = parent.GetElement(i);
         Element *el = NewElement(parent_el->GetType());
         el->SetVertices(submesh_v);
         AddElement(el);
      }
   }

   SetAttributes();
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
