// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of FiniteElementSpace

#include "qspace.hpp"

namespace mfem
{

void QuadratureSpace::Construct()
{
   // protected method
   int offset = 0;
   const int num_elem = mesh->GetNE();
   element_offsets = new int[num_elem + 1];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      int_rule[g] = NULL;
   }
   for (int i = 0; i < num_elem; i++)
   {
      element_offsets[i] = offset;
      int geom = mesh->GetElementBaseGeometry(i);
      if (int_rule[geom] == NULL)
      {
         int_rule[geom] = &IntRules.Get(geom, order);
      }
      offset += int_rule[geom]->GetNPoints();
   }
   element_offsets[num_elem] = size = offset;
}

QuadratureSpace::QuadratureSpace(Mesh *mesh_, std::istream &in)
   : mesh(mesh_)
{
   const char *msg = "invalid input stream";
   std::string ident;

   in >> ident; MFEM_VERIFY(ident == "QuadratureSpace", msg);
   in >> ident; MFEM_VERIFY(ident == "Type:", msg);
   in >> ident;
   if (ident == "default_quadrature")
   {
      in >> ident; MFEM_VERIFY(ident == "Order:", msg);
      in >> order;
   }
   else
   {
      MFEM_ABORT("unknown QuadratureSpace type: " << ident);
      return;
   }

   Construct();
}

void QuadratureSpace::Save(std::ostream &os) const
{
   os << "QuadratureSpace\n"
      << "Type: default_quadrature\n"
      << "Order: " << order << '\n';
}

} // namespace mfem
