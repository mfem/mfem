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
   std::unordered_map<int,int> idx_inverse;

   void Add(int i)
   {
      auto f = idx.find(i);
      if (f == idx.end())
      {
         idx[i] = counter;
         counter++;
      }
   }

   int Get(int i)
   {
      auto f = idx.find(i);
      if (f == idx.end())
      {
         return -1;
      }
      return (*f).second;
   }

   int GetInverse(int i)
   {
      auto f = idx_inverse.find(i);
      if (f == idx_inverse.end())
      {
         return -1;
      }
      return (*f).second;
   }

   void BuildInverse()
   {
      for (const auto &p : idx)
      {
         idx_inverse[p.second] = p.first;
      }
   }

   void Reset()
   {
      counter = 0;
      idx.clear();
      idx_inverse.clear();
   }

   int Size()
   {
      return counter;
   }
};

SubMesh::SubMesh(Mesh &parent, From from,
                 Array<int> attributes) : parent_(parent), from_(from), attributes_(attributes)
{
   if (From::Domain)
   {
      Dim = parent.Dimension();
   }
   else
   {
      Dim = parent.Dimension() - 1;
   }
   spaceDim = parent.SpaceDimension();

   UniqueIndexGenerator element_ids;
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
         element_ids.Add(i);

         Array<int> v;
         pel->GetVertices(v);

         for (int i = 0; i < v.Size(); i++)
         {
            vertex_ids.Add(v[i]);
         }
      }
   }

   element_ids.BuildInverse();
   vertex_ids.BuildInverse();

   InitMesh(Dim, spaceDim, element_ids.Size(), vertex_ids.Size(), 0);

   // Add vertices
   for (int i = 0; i < vertex_ids.Size(); i++)
   {
      int parent_vtx_id = vertex_ids.GetInverse(i);
      AddVertex(parent.GetVertex(parent_vtx_id));
   }

   // Add elements
   for (int i = 0; i < element_ids.Size(); i++)
   {
      int parent_element_id = element_ids.GetInverse(i);

      Element *parent_el = parent.GetElement(parent_element_id);
      Element *el = NewElement(parent_el->GetType());

      Array<int> parent_v;
      parent_el->GetVertices(parent_v);
      Array<int> v(parent_v.Size());
      for (int j = 0; j < parent_v.Size(); j++)
      {
         v[j] = vertex_ids.Get(parent_v[j]);
      }

      el->SetVertices(v);
      AddElement(el);
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
