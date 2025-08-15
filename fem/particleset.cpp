// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "particleset.hpp"


namespace mfem
{

Particle::Particle(int dim, const Array<int> &field_vdims, int num_tags)
   : coords(dim),
     fields(),
     tags()
{
   coords = 0.0;

   for (int f = 0; f < field_vdims.Size(); f++)
   {
      fields.emplace_back(field_vdims[f]);
      fields.back() = 0.0;
   }

   for (int t = 0; t < num_tags; t++)
   {
      tags.emplace_back(1);
      tags.back()[0] = 0;
   }

}

bool Particle::operator==(const Particle &rhs) const
{
   if (coords.Size() != rhs.coords.Size())
   {
      return false;
   }
   for (int d = 0; d < coords.Size(); d++)
   {
      if (coords[d] != rhs.coords[d])
      {
         return false;
      }
   }

   if (fields.size() != rhs.fields.size())
   {
      return false;
   }
   for (int f = 0; f < fields.size(); f++)
   {
      if (fields[f].Size() != rhs.fields[f].Size())
      {
         return false;
      }
      for (int c = 0; c < fields[f].Size(); c++)
      {
         if (fields[f][c] != rhs.fields[f][c])
         {
            return false;
         }
      }
   }
   if (tags.size() != rhs.tags.size())
   {
      return false;
   }
   for (int t = 0; t < tags.size(); t++)
   {
      if (tags[t][0] != rhs.tags[t][0])
      {
         return false;
      }
   }

   return true;
}

void Particle::Print(std::ostream &out) const
{
   out << "Coords: (";
   for (int d = 0; d < coords.Size(); d++)
   {
      out << coords[d] << ( (d+1 < coords.Size()) ? "," : ")\n");
   }
   for (int f = 0; f < fields.size(); f++)
   {
      out << "Field " << f << ": (";
      for (int c = 0; c < fields[f].Size(); c++)
      {
         out << fields[f][c] << ( (c+1 < fields[f].Size()) ? "," : ")\n");
      }
   }
   for (int t = 0; t < tags.size(); t++)
   {
      out << "Tag " << t << ": " << tags[t][0] << "\n";
   }
}


} // namespace mfem
