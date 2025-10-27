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

#ifndef MFEM_BB_GRID_MAP
#define MFEM_BB_GRID_MAP

#include "../config/config.hpp"
#ifdef MFEM_USE_MPI
#include "../fem/pgridfunc.hpp"
#else
#include "../fem/gridfunc.hpp"
#endif

namespace mfem
{

class BoundingBoxTensorGridMap
{
private:
   int dim;
   Array<int> lmap_n;
   Vector lmap_bnd_min, lmap_bnd_max;
   Vector lmap_fac;
   Array<unsigned int> lmap_offset;
   int lmap_nd;

public:
   // Constructor for a given mesh and number of hash divisions
   BoundingBoxTensorGridMap(Mesh &mesh, int nx);

   // Constructor for given element bounds. The user can either specify
   // the max size of map (by_max_size=true) or the
   // number of divisions (by_max_size=false).
   // Assumes elmin, elmax Ordering::byVDim
   BoundingBoxTensorGridMap(Vector &elmin, Vector &elmax,
                            int n, int nel, bool by_max_size=false);

   // Constructor for given element bounds and number of hash divisions in
   // each direction
   // Assumes elmin, elmax Ordering::byVDim
   BoundingBoxTensorGridMap(Vector &elmin, Vector &elmax,
                            Array<int> &nx, int nel);

   /// Map a point to possible overlapping elements.
   Array<int> MapPointToElements(Vector &xyz) const;

   /// Get list of elements corresponding to a hash cell.
   Array<int> HashCellToElements(int i) const;

   // Some getters
   Array<int> GetHashMap() const { return lmap_offset; }
   Vector GetHashFac() const { return lmap_fac; }
   Vector GetHashMin() const { return lmap_bnd_min; }
   Vector GetHashMax() const { return lmap_bnd_max; }
   Array<int> GetHashN() const { return lmap_n; }

private:
   // Setup using the element-wise bounding boxes.
   // When by_max_size = false, nx gives number of cells in each direction.
   // When by_max_size = true, nx[0] gives the max size of the map. i.e.
   // lmap_offset.Size() <= nx[0].
   void Setup(Vector &elmin, Vector &elmax,
              Array<int> &nx, int nel, bool by_max_size);
   /// Get hash cell index for a given point.
   int GetHashCellFromPoint(Vector &xyz) const;
};

} // namespace mfem

#endif // MFEM_BB_GRID_MAP
