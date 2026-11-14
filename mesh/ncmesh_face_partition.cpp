// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "ncmesh_face_partition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mfem
{

NCFaceRelation CompareNCFacePartitions(const NCFacePartition &first,
                                       const NCFacePartition &second)
{
   const auto num_vertices = [](const NCFacePartition::Cell &cell)
   {
      MFEM_VERIFY(cell.geometry == Geometry::TRIANGLE ||
                  cell.geometry == Geometry::SQUARE,
                  "Face partition cell must be a triangle or quadrilateral");
      return cell.geometry == Geometry::TRIANGLE ? 3 : 4;
   };
   const auto contains = [&](const NCFacePartition::Cell &outer,
                             const NCFacePartition::Cell &inner)
   {
      const int outer_size = num_vertices(outer);
      const int inner_size = num_vertices(inner);
      real_t area = 0.0;
      real_t min_x = outer.vertex[0].x, max_x = min_x;
      real_t min_y = outer.vertex[0].y, max_y = min_y;
      for (int i = 0; i < outer_size; i++)
      {
         const auto &a = outer.vertex[i];
         const auto &b = outer.vertex[(i + 1) % outer_size];
         const auto &origin = outer.vertex[0];
         area += (a.x-origin.x)*(b.y-origin.y) -
                 (a.y-origin.y)*(b.x-origin.x);
         min_x = std::min(min_x, a.x);
         max_x = std::max(max_x, a.x);
         min_y = std::min(min_y, a.y);
         max_y = std::max(max_y, a.y);
      }
      const real_t scale = (max_x-min_x)*(max_y-min_y);
      const real_t epsilon = std::numeric_limits<real_t>::epsilon()*scale;
      MFEM_VERIFY(std::abs(area) > 128*epsilon,
                  "Degenerate face partition cell");
      const real_t sign = area > 0.0 ? 1.0 : -1.0;
      for (int j = 0; j < inner_size; j++)
      {
         const auto &p = inner.vertex[j];
         for (int i = 0; i < outer_size; i++)
         {
            const auto &a = outer.vertex[i];
            const auto &b = outer.vertex[(i + 1) % outer_size];
            const real_t cross = (b.x-a.x)*(p.y-a.y) -
                                 (b.y-a.y)*(p.x-a.x);
            if (sign*cross < -16*epsilon) { return false; }
         }
      }
      return true;
   };
   const auto refines = [&](const NCFacePartition &fine,
                            const NCFacePartition &coarse)
   {
      for (int i = 0; i < fine.cells.Size(); i++)
      {
         bool inside = false;
         for (int j = 0; j < coarse.cells.Size(); j++)
         {
            if (contains(coarse.cells[j], fine.cells[i]))
            { inside = true; break; }
         }
         if (!inside) { return false; }
      }
      return true;
   };

   MFEM_VERIFY(first.cells.Size() && second.cells.Size(),
               "Face partitions must not be empty");
   const bool first_refines = refines(first, second);
   const bool second_refines = refines(second, first);
   if (first_refines && second_refines) { return NCFaceRelation::Matching; }
   if (first_refines) { return NCFaceRelation::FirstRefinesSecond; }
   if (second_refines) { return NCFaceRelation::SecondRefinesFirst; }
   return NCFaceRelation::Incompatible;
}

} // namespace mfem
