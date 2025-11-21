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

#include "bb_grid_map.hpp"

#include <limits>
#include <cstring>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace mfem
{

using namespace std;

BBoxTensorGridMap::BBoxTensorGridMap(Mesh &mesh, int nx)
{
   GridFunction *nodes = mesh.GetNodes();
   const int nel = mesh.GetNE();
   dim = mesh.SpaceDimension();
   Vector elmin(nel*dim), elmax(nel*dim);
   elmin = numeric_limits<real_t>::max();
   elmax = -numeric_limits<real_t>::max();
   if (!nodes)
   {
      Array<int> verts;
      real_t *coord;
      // create bounding boxes from vertex coordinates
      for (int e = 0; e < nel; e++)
      {
         mesh.GetElementVertices(e, verts);
         Vector center(dim);
         for (int v = 0; v < verts.Size(); v++)
         {
            coord = mesh.GetVertex(verts[v]);
            for (int d = 0; d < dim; d++)
            {
               elmin(d*nel + e) = min(elmin(d*nel + e), coord[d]);
               elmax(d*nel + e) = max(elmax(d*nel + e), coord[d]);
            }
         }
      }
   }
   else
   {
      int nref = 3;
      nodes->GetElementBounds(elmin, elmax, nref);
   }
   Array<int> nx_arr(dim);
   nx_arr = nx;
   Setup(elmin, elmax, nx_arr, nel, false);
}

BBoxTensorGridMap::BBoxTensorGridMap(Vector &elmin,
                                     Vector &elmax,
                                     int n, int nel,
                                     bool by_max_size)
{
   dim = elmin.Size() / nel;
   Array<int> nx_arr(dim);
   nx_arr = n;
   Setup(elmin, elmax, nx_arr, nel, by_max_size);
}

BBoxTensorGridMap::BBoxTensorGridMap(Vector &elmin, Vector &elmax,
                                     Array<int> &nx, int nel)
{
   Setup(elmin, elmax, nx, nel, false);
}

void BBoxTensorGridMap::Setup(Vector &elmin, Vector &elmax,
                              Array<int> &nx, int nel, bool by_max_size)
{
   dim = elmin.Size() / nel;
   lmap_bnd_min.SetSize(dim);
   lmap_bnd_max.SetSize(dim);
   lmap_fac.SetSize(dim);
   lmap_nx.SetSize(dim);
   lmap_nx = nx;

   MFEM_VERIFY(nx.Size() == dim,
               "BBoxTensorGridMap requires nx to have the same size as the number of dimensions.");
   for (int d = 0; d < nx.Size(); d++)
   {
      MFEM_VERIFY(nx[d] > 0,
                  "BBoxTensorGridMap requires positive number of divisions in each dimension.");
   }
   for (int d = 0; d < dim; d++)
   {
      Vector elmind(elmin.GetData() + d*nel, nel);
      Vector elmaxd(elmax.GetData() + d*nel, nel);
      lmap_bnd_min[d] = elmind.Min();
      lmap_bnd_max[d] = elmaxd.Max();
   }

   Array<int> elmin_h, elmax_h;
   unsigned int store_size;
   if (by_max_size)
   {
      int nmax = nx[0];
      int nlow = 1, nhigh = nmax > nel ? ceil(pow(nmax - nel, 1.0 / dim)) : 1;
      int size_low = 2 + nel;
      int size = 0;
      while (nhigh - nlow > 1)
      {
         int nmid = nlow + (nhigh - nlow) / 2;
         int nmd = nmid*nmid;
         if (dim == 3) { nmd *= nmid; }
         lmap_nx = nmid;
         SetGridFac(lmap_fac, lmap_nx, lmap_bnd_min, lmap_bnd_max);
         size = nmd + 1 + GetGridCountAndRange(lmap_nx, lmap_fac,
                                               lmap_bnd_min, lmap_bnd_max,
                                               elmin, elmax,
                                               elmin_h, elmax_h);
         if (size <= nmax) { nlow = nmid; size_low = size; }
         else { nhigh = nmid; }
      }
      lmap_nx = nlow;
      lmap_nxd = nlow*nlow;
      if (dim == 3) { lmap_nxd *= nlow; }
      store_size = size_low;
      SetGridFac(lmap_fac, lmap_nx, lmap_bnd_min, lmap_bnd_max);
      if (size != size_low)
      {
         GetGridCountAndRange(lmap_nx, lmap_fac,
                              lmap_bnd_min, lmap_bnd_max,
                              elmin, elmax,
                              elmin_h, elmax_h);
      }
   }
   else
   {
      SetGridFac(lmap_fac, lmap_nx, lmap_bnd_min, lmap_bnd_max);

      lmap_nxd = lmap_nx[0];
      for (int d = 1; d < dim; d++)
      {
         lmap_nxd *= lmap_nx[d];
      }

      // Grid cell ranges for each element in each direction
      store_size = lmap_nxd + 1 + GetGridCountAndRange(lmap_nx, lmap_fac,
                                                       lmap_bnd_min,
                                                       lmap_bnd_max,
                                                       elmin, elmax,
                                                       elmin_h, elmax_h);
   }

   lgrid_map.SetSize(store_size);
   lgrid_map[0] = lmap_nxd + 1;

   Array<unsigned int> grid_el_count(lmap_nxd);
   grid_el_count = 0;

   for (int e = 0; e < nel; e++)
   {
      int klim = dim < 3 ? 1 : (elmax_h[2*nel+e]-elmin_h[2*nel+e]);
      int jlim = dim < 2 ? 1 : (elmax_h[1*nel+e]-elmin_h[1*nel+e]);
      int ilim = (elmax_h[0*nel+e]-elmin_h[0*nel+e]);
      for (int k = 0; k < klim; k++)
      {
         int koff = dim<3 ? 0 : (elmin_h[2*nel + e] + k)* lmap_nx[0]*lmap_nx[1];
         for (int j = 0; j < jlim; j++)
         {
            int joff = dim < 2 ? 0 : (elmin_h[1*nel + e] + j) * lmap_nx[0];
            for (int i = 0; i < ilim; i++)
            {
               int ioff = elmin_h[e] + i;
               int idx = ioff + joff + koff;
               grid_el_count[idx]++;
            }
         }
      }
   }

   for (unsigned int e = 0; e < lmap_nxd; e++)
   {
      lgrid_map[e + 1] = lgrid_map[e] + grid_el_count[e];
   }

   for (int e = 0; e < nel; e++)
   {
      int klim = dim < 3 ? 1 : (elmax_h[2*nel+e]-elmin_h[2*nel+e]);
      int jlim = dim < 2 ? 1 : (elmax_h[1*nel+e]-elmin_h[1*nel+e]);
      int ilim = (elmax_h[0*nel+e]-elmin_h[0*nel+e]);
      for (int k = 0; k < klim; k++)
      {
         int koff = dim<3 ? 0 : (elmin_h[2*nel+e] + k)* lmap_nx[0] * lmap_nx[1];
         for (int j = 0; j < jlim; j++)
         {
            int joff = dim < 2 ? 0 : (elmin_h[1*nel + e] + j) * lmap_nx[0];
            for (int i = 0; i < ilim; i++)
            {
               int ioff = elmin_h[e] + i;
               int idx = ioff + joff + koff;
               lgrid_map[lgrid_map[idx+1]-grid_el_count[idx]]=e;
               grid_el_count[idx]--;
            }
         }
      }
   }
}

Array<int> BBoxTensorGridMap::GridCellToElements(unsigned int i) const
{
   MFEM_ASSERT(i >= 0 && i < lmap_nxd,
               "Access element " << i << " of local grid with cells = "
               << lmap_nxd);
   int start = lgrid_map[i];
   int end = lgrid_map[i + 1];
   Array<int> elements(end - start);
   for (int j = start; j < end; j++)
   {
      elements[j - start] = lgrid_map[j];
   }
   return elements;
}

int BBoxTensorGridMap::GetGridCellFromPoint(Vector &xyz) const
{
   MFEM_ASSERT(xyz.Size() == dim,
               "Point must have the same dimension as the grid.");
   int sum = 0;
   for (int d = dim-1; d >= 0; --d)
   {
      if (xyz(d) < lmap_bnd_min(d) || xyz(d) > lmap_bnd_max(d))
      {
         return -1; // Point is outside the bounds of the grid
      }
      sum *= lmap_nx[d];
      int i = (int)floor((xyz(d) - lmap_bnd_min(d)) * lmap_fac[d]);
      sum += i < 0 ? 0 : (lmap_nx[d] - 1 < i ? lmap_nx[d] - 1 : i);
   }
   return sum;
}

Array<int> BBoxTensorGridMap::MapPointToElements(Vector &xyz) const
{
   MFEM_ASSERT(xyz.Size() == dim,
               "Point must have the same dimension as the grid.");
   int cell = GetGridCellFromPoint(xyz);
   if (cell < 0)
   {
      return Array<int>(); // Point is outside the bounds of the tensor grid
   }
   return GridCellToElements(cell);
}

void BBoxTensorGridMap::GetGridRange(const int d, const Array<int> &lh_n,
                                     const Vector &lh_fac,
                                     const Vector &lh_bnd_min,
                                     const Vector &lh_bnd_max,
                                     const double &xmin, const double &xmax,
                                     int &imin, int &imax)
{
   // if xmin/xmax is exactly on grid boundary, use the cell to the right
   // if xmin/xmax is within a grid cell, we set i0/i1 to include that cell
   int i0 = floor( (xmin - lh_bnd_min[d]) * lh_fac[d] );
   int i1 = ceil ( (xmax - lh_bnd_min[d]) * lh_fac[d] );
   imin = i0 < 0 ? 0 : i0;
   imax = i1 < lh_n[d] ? i1 : lh_n[d];
   if (imax == imin) { ++imax; }
}

void BBoxTensorGridMap::SetGridFac(Vector &lh_fac, const Array<int> &nx,
                                   const Vector &lh_bnd_min,
                                   const Vector &lh_bnd_max)
{
   int dim = lh_bnd_min.Size();
   for (int d = 0; d < dim; d++)
   {
      lh_fac[d] = nx[d] / (lh_bnd_max[d] - lh_bnd_min[d]);
   }
}

int BBoxTensorGridMap::GetGridCountAndRange(const Array<int> &lh_n,
                                            const Vector &lh_fac,
                                            const Vector &lh_bnd_min,
                                            const Vector &lh_bnd_max,
                                            const Vector &elmin,
                                            const Vector &elmax,
                                            Array<int> &elmin_h,
                                            Array<int> &elmax_h)
{
   int count = 0;
   const int dim = lh_bnd_min.Size();
   const int nel = elmin.Size()/dim;
   elmin_h.SetSize(dim * nel);
   elmax_h.SetSize(dim * nel);
   for (int i = 0; i < nel; i++)
   {
      int count_el = 1;
      for (int d = 0; d < dim; d++)
      {
         GetGridRange(d, lh_n, lh_fac, lh_bnd_min, lh_bnd_max,
                      elmin[d*nel + i], elmax[d*nel + i],
                      elmin_h[d*nel + i], elmax_h[d*nel + i]);
         int imax = elmax_h[d*nel + i];
         int imin = elmin_h[d*nel + i];
         count_el *= (imax - imin);
      }
      count += count_el;
   }
   return count;
}

} // namespace mfem
