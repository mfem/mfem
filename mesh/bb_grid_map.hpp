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

/** \brief Class to map a point in physical space to candidate elements
 *   of a curved mesh.
 *
 *  This class builds a Cartesian-aligned tensor grid that covers the domain
 *  and precomputes, for each grid cell, the set of curved mesh elements whose
 *  axis-aligned bounding boxes (AABBs) intersect that cell. Given a point (xyz)
 *  in physical coordinates, the Cartesian grid cell containing the point is
 *  determined, and the list of candidate element indices whose AABBs are
 *  intersecting that cell is returned. This yields a fast, conservative
 *  point-to-element candidate query.
 *
 *  Construction workflow:
 *  1) Compute an axis-aligned bounding box for each curved element.
 *     Optionally, the user can directly provided the element bounding boxes.
 *  2) Overlay the domain with a tensor-product grid. The user can specify the
 *     number of divisions in each coordinate direction, or specify a maximum
 *     size of the underlying map which implicitly determines the grid
 *     resolution.
 *  3) For each grid cell, record the indices of all elements whose AABBs
 *     intersect that cell.
 */
class BoundingBoxTensorGridMap
{
private:
   int dim;
   Array<int> lmap_n;
   Vector lmap_bnd_min, lmap_bnd_max;
   Vector lmap_fac;
   Array<unsigned int> lgrid_map;
   int lmap_nd;

public:
   // Constructor for a given mesh and resolution of Cartesian grid.
   BoundingBoxTensorGridMap(Mesh &mesh, int nx);

   // Constructor with mesh element bounding boxes and resolution of Cartesian
   // grid in each direction.
   // Assumes elmin, elmax Ordering::byVDim:
   // elmin -> [x_{0,min},y_{0,min},z_{0,min}, x_{1,min},... ,z_{nel-1,min}]
   // elmax -> [x_{0,max},y_{0,max},z_{0,max}, x_{1,max},... ,z_{nel-1,max}]
   BoundingBoxTensorGridMap(Vector &elmin, Vector &elmax,
                            Array<int> &nx, int nel);

   // Constructor for given element bounds. The user can either specify
   // the max size of map (by_max_size=true) or the
   // number of divisions (by_max_size=false).
   // When by_max_size=true, lgrid_map.Size() <= n.
   // Assumes elmin, elmax Ordering::byVDim:
   // elmin -> [x_{0,min},y_{0,min},z_{0,min}, x_{1,min},... ,z_{nel-1,min}]
   // elmax -> [x_{0,max},y_{0,max},z_{0,max}, x_{1,max},... ,z_{nel-1,max}]
   BoundingBoxTensorGridMap(Vector &elmin, Vector &elmax,
                            int n, int nel, bool by_max_size=false);

   /// Map a point to possible overlapping elements.
   Array<int> MapPointToElements(Vector &xyz) const;

   // Some getters
   Array<int> GetHashMap() const { return lgrid_map; }
   Vector GetHashFac() const { return lmap_fac; }
   Vector GetHashMin() const { return lmap_bnd_min; }
   Vector GetHashMax() const { return lmap_bnd_max; }
   Array<int> GetHashN() const { return lmap_n; }
private:
   // Setup using the element-wise bounding boxes.
   // When by_max_size = false, nx gives number of cells in each direction.
   // When by_max_size = true, nx[0] gives the max size of the map. i.e.
   // lgrid_map.Size() <= nx[0].
   void Setup(Vector &elmin, Vector &elmax,
              Array<int> &nx, int nel, bool by_max_size);

              /// Get hash cell index for a given point.
   int GetHashCellFromPoint(Vector &xyz) const;

   /// Get list of elements corresponding to a hash cell.
   Array<int> HashCellToElements(int i) const;
};

} // namespace mfem

#endif // MFEM_BB_GRID_MAP
