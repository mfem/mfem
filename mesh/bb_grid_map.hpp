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

/** \brief Map a point in physical space to candidate elements of a curved mesh.
 *
 *  This class builds a Cartesian-aligned tensor grid that covers the domain
 *  and precomputes, for each grid cell, the set of curved mesh elements whose
 *  axis-aligned bounding boxes (AABBs) intersect that cell. Given a point (xyz)
 *  in physical coordinates, the Cartesian grid cell containing the point is
 *  determined, and the list of candidate element indices whose AABBs are
 *  intersecting that cell is returned. This yields a fast, conservative
 *  point-to-element candidate query.
 *
 *  The mapping procedure is currently setup such that if an element's bounding
 *  box boundary is exactly on a grid cell boundary, the element is assigned
 *  to the grid cell on the left/bottom side of the boundary in each dimension.
 *
 *  The map itself is stored as a single array CSR structure where the offsets
 *  and values are stored in the same array. For a tensor grid with a total of
 *  N cells, the first N+1 entries store the offsets and the remaining entries
 *  store the values.
 *
 *  The "lgrid_map" looks something like this:
 *
 *  Index: 0         1          ...  N             N+1        ...
 *  Value: [start_0] [start_1]  ... [Length(Map)] [elem_A] [elem_B] [elem_C]...
 *              |          |                          ^                 ^
 *              |          |__________________________|_________________|
 *              |_____________________________________|
 *
 *  For grid cell index i, the element indices are stored in
 *  lgrid_map[j], where lgrid_map[i] <= j < lgrid_map[i+1].
 *
 *  If lgrid_map[i] = lgrid_map[i+1], the grid cell i does not intersect any
 *  elements.
 *
 *  See Mittal et al., "General Field Evaluation in High-Order Meshes on GPUs".
 *  (2025). Computers & Fluids. for technical details.
 */
class BBoxTensorGridMap
{
private:
   int dim; // spatial dimension
   Array<int> lmap_nx; // grid resolution in each direction
   Vector lmap_bnd_min, lmap_bnd_max; // min and max extend of grid in x/y/z
   Vector lmap_fac; // number of cells per unit extent
   Array<unsigned int> lgrid_map; // actual map from grid cell to mesh elements.
   unsigned int lmap_nxd; // total number of grid cells

public:
   /// Constructor for a given mesh and resolution of Cartesian grid.
   BBoxTensorGridMap(Mesh &mesh, int nx);

   /** @brief Constructor with mesh element bounding boxes and resolution of
    *  Cartesian grid in each direction.
    *
    *  @details Assumes elmin, elmax Ordering::byNodes:
    *  elmin -> [x_{0,min},x_{1,min},... ,y_{0,min},y_{1,min},..,z_{nel-1,min}]
    *  elmax -> [x_{0,max},x_{1,max},... ,y_{0,max},y_{1,max},..,z_{nel-1,max}]
    *  Note elmin, elmax can be obtained using GridFunction::GetElementBounds()
    */
   BBoxTensorGridMap(Vector &elmin, Vector &elmax,
                     Array<int> &nx, int nel);

   /** @brief Constructor for given element bounds. The user can either specify
    *  the max size of map (by_max_size=true) or the number of divisions
    *  (by_max_size=false).
    *
    *  @details  When by_max_size=true, lgrid_map.Size() <= n.
    *  Assumes elmin, elmax Ordering::byNodes:
    *  elmin -> [x_{0,min},x_{1,min},... ,y_{0,min},y_{1,min},..,z_{nel-1,min}]
    *  elmax -> [x_{0,max},x_{1,max},... ,y_{0,max},y_{1,max},..,z_{nel-1,max}]
    *  Note elmin, elmax can be obtained using GridFunction::GetElementBounds()
    */
   BBoxTensorGridMap(Vector &elmin, Vector &elmax,
                     int n, int nel, bool by_max_size=false);

   /// Map a point to possible overlapping elements.
   Array<int> MapPointToElements(Vector &xyz) const;

   /// Get grid cell index for a given point.
   int GetGridCellFromPoint(Vector &xyz) const;

   /// Get list of elements corresponding to a grid cell.
   Array<int> GridCellToElements(int i) const;

   // Some getters
   Array<int> GetGridMap() const { return lgrid_map; }
   Vector GetGridFac() const { return lmap_fac; }
   Vector GetGridMin() const { return lmap_bnd_min; }
   Vector GetGridMax() const { return lmap_bnd_max; }
   Array<int> GetGridN() const { return lmap_nx; }
private:
   /** @brief Setup using the element-wise bounding boxes.
    *
    *  @details When by_max_size = false, nx gives number of cells in each
    * direction. When by_max_size = true, nx[0] gives the max size of the map.
    * i.e. lgrid_map.Size() <= nx[0]. */
   void Setup(Vector &elmin, Vector &elmax,
              Array<int> &nx, int nel, bool by_max_size);

public:
   /** @brief Get local (1D) indices for cells of tensor grid that intersect
    *  with the given bounding box. */
   static void GetGridRange(const int d, const Array<int> &lh_n,
                            const Vector &lh_fac,
                            const Vector &lh_bnd_min, const Vector &lh_bnd_max,
                            const double &xmin, const double &xmax,
                            int &imin, int &imax);

   /// Set grid fac - number of grid cells per unit grid extent.
   static void SetGridFac(Vector &lh_fac, const Array<int> &nx,
                          const Vector &lh_bnd_min, const Vector &lh_bnd_max);

   /** @brief Get grid count and range - total number of grid cells that
    *  intersect with all elements of the mesh and get corresponding ranges. */
   static int GetGridCountAndRange(const Array<int> &lh_n, const Vector &lh_fac,
                                   const Vector &lh_bnd_min,
                                   const Vector &lh_bnd_max,
                                   const Vector &elmin, const Vector &elmax,
                                   Array<int> &elmin_h, Array<int> &elmax_h);
};

} // namespace mfem

#endif // MFEM_BB_GRID_MAP
