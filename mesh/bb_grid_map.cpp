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

// Implementation of bounds

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

// Get Hash Range
static void GetHashRange(const int d, const Array<int> &lh_n,
                         const Vector &lh_fac,
                         const Vector &lh_bnd_min, const Vector &lh_bnd_max,
                         const double &xmin, const double &xmax,
                         int &imin, int &imax)
{
   int i0 = floor( (xmin - lh_bnd_min[d]) * lh_fac[d] );
   int i1 = ceil ( (xmax - lh_bnd_min[d]) * lh_fac[d] );
   imin = i0 < 0 ? 0 : i0;
   imax = i1 < lh_n[d] ? i1 : lh_n[d];
   if (imax == imin) { ++imax; }
}


static void SetHashFac(Vector &lh_fac, const Array<int> &nx,
                       const Vector &lh_bnd_min, const Vector &lh_bnd_max)
{
   int dim = lh_bnd_min.Size();
   for (int d = 0; d < dim; d++)
   {
      lh_fac[d] = nx[d] / (lh_bnd_max[d] - lh_bnd_min[d]);
   }
}


// Get Local Hash Count
static int GetHashCount(const Array<int> &lh_n, const Vector &lh_fac,
                        const Vector &lh_bnd_min, const Vector &lh_bnd_max,
                        const Vector &elmin, const Vector &elmax,
                        Array<int> &elmin_h, Array<int> &elmax_h)
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
         GetHashRange(d, lh_n, lh_fac, lh_bnd_min, lh_bnd_max,
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

BoundingBoxTensorGridMap::BoundingBoxTensorGridMap(Mesh &mesh, int nx)
{
   GridFunction *nodes = mesh.GetNodes();
   MFEM_VERIFY(nodes != nullptr,
               "BoundingBoxTensorGridMap requires a mesh with nodes defined.");
   const int nel = mesh.GetNE();
   Vector elmin, elmax;
   int nref = 3;
   nodes->GetElementBounds(elmin, elmax, nref);
   dim = elmin.Size() / nel;
   Array<int> nx_arr(dim);
   nx_arr = nx;
   Setup(elmin, elmax, nx_arr, nel, false);
}

BoundingBoxTensorGridMap::BoundingBoxTensorGridMap(Vector &elmin,
                                                   Vector &elmax,
                                                   int n, int nel,
                                                   bool by_max_size)
{
   dim = elmin.Size() / nel;
   Array<int> nx_arr(dim);
   nx_arr = n;
   Setup(elmin, elmax, nx_arr, nel, by_max_size);
}

BoundingBoxTensorGridMap::BoundingBoxTensorGridMap(Vector &elmin, Vector &elmax,
                                                   Array<int> &nx, int nel)
{
   Setup(elmin, elmax, nx, nel, false);
}

void BoundingBoxTensorGridMap::Setup(Vector &elmin, Vector &elmax,
                                     Array<int> &nx, int nel, bool by_max_size)
{
   dim = elmin.Size() / nel;
   lmap_bnd_min.SetSize(dim);
   lmap_bnd_max.SetSize(dim);
   lmap_fac.SetSize(dim);
   lmap_n.SetSize(dim);
   lmap_n = nx;

   MFEM_VERIFY(nx.Size() == dim,
               "BoundingBoxTensorGridMap requires nx to have the same size as the number of dimensions.");
   for (int d = 0; d < nx.Size(); d++)
   {
      MFEM_VERIFY(nx[d] > 0,
                  "BoundingBoxTensorGridMap requires positive number of divisions in each dimension.");
   }
   for (int d = 0; d < dim; d++)
   {
      Vector elmind(elmin.GetData() + d*nel, nel);
      Vector elmaxd(elmax.GetData() + d*nel, nel);
      lmap_bnd_min[d] = elmind.Min();
      lmap_bnd_max[d] = elmaxd.Max();
   }

   Array<int> elmin_h, elmax_h;
   int store_size;
   if (by_max_size)
   {
      int nmax = nx[0];
      int nlow = 1, nhigh = ceil(pow(nmax - nel, 1.0 / dim));
      int size_low = 2 + nel;
      int size;
      while (nhigh - nlow > 1)
      {
         int nmid = nlow + (nhigh - nlow) / 2;
         int nmd = nmid*nmid;
         if (dim == 3) { nmd *= nmid; }
         lmap_n = nmid;
         SetHashFac(lmap_fac, lmap_n, lmap_bnd_min, lmap_bnd_max);
         size = nmd + 1 + GetHashCount(lmap_n, lmap_fac,
                                       lmap_bnd_min, lmap_bnd_max,
                                       elmin, elmax,
                                       elmin_h, elmax_h);
         if (size <= nmax) { nlow = nmid; size_low = size; }
         else { nhigh = nmid; }
      }
      lmap_n = nlow;
      lmap_nd = nlow*nlow;
      if (dim == 3) { lmap_nd *= nlow; }
      store_size = size_low;
      SetHashFac(lmap_fac, lmap_n, lmap_bnd_min, lmap_bnd_max);
      if (size != size_low)
      {
         GetHashCount(lmap_n, lmap_fac,
                     lmap_bnd_min, lmap_bnd_max,
                     elmin, elmax,
                     elmin_h, elmax_h);
      }
   }
   else
   {
      SetHashFac(lmap_fac, lmap_n, lmap_bnd_min, lmap_bnd_max);

      int lmap_nd = lmap_n[0];
      for (int d = 1; d < dim; d++)
      {
         lmap_nd *= lmap_n[d];
      }

      // Hash cell ranges for each element in each dimension
      store_size = lmap_nd + 1 + GetHashCount(lmap_n, lmap_fac,
                                              lmap_bnd_min, lmap_bnd_max,
                                              elmin, elmax,
                                              elmin_h, elmax_h);
   }

   lmap_offset.SetSize(store_size);
   lmap_offset[0] = lmap_nd + 1;

   Array<unsigned int> hash_el_count(lmap_nd);
   hash_el_count = 0;

   for (int e = 0; e < nel; e++)
   {
      int klim = dim < 3 ? 1 : (elmax_h[2*nel+e]-elmin_h[2*nel+e]);
      int jlim = dim < 2 ? 1 : (elmax_h[1*nel+e]-elmin_h[1*nel+e]);
      int ilim = (elmax_h[0*nel+e]-elmin_h[0*nel+e]);
      for (int k = 0; k < klim; k++)
      {
         int kidx = dim < 3 ? 0 : elmin_h[2*nel + e] + k;
         int koff = dim < 3 ? 0 : (elmin_h[2*nel + e] + k)* lmap_n[0] * lmap_n[1];
         for (int j = 0; j < jlim; j++)
         {
            int jidx = dim < 2 ? 0 : elmin_h[1*nel + e] + j;
            int joff = dim < 2 ? 0 : (elmin_h[1*nel + e] + j) * lmap_n[0];
            for (int i = 0; i < ilim; i++)
            {
               int ioff = elmin_h[0*nel + e] + i;
               int idx = ioff + joff + koff;
               hash_el_count[idx]++;
            }
         }
      }
   }

   for (int e = 0; e < lmap_nd; e++)
   {
      lmap_offset[e + 1] = lmap_offset[e] + hash_el_count[e];
   }

   for (int e = 0; e < nel; e++)
   {
      int klim = dim < 3 ? 1 : (elmax_h[2*nel+e]-elmin_h[2*nel+e]);
      int jlim = dim < 2 ? 1 : (elmax_h[1*nel+e]-elmin_h[1*nel+e]);
      int ilim = (elmax_h[0*nel+e]-elmin_h[0*nel+e]);
      for (int k = 0; k < klim; k++)
      {
         int kidx = dim < 3 ? 0 : elmin_h[2*nel + e] + k;
         int koff = dim < 3 ? 0 : (elmin_h[2*nel + e] + k)* lmap_n[0] * lmap_n[1];
         for (int j = 0; j < jlim; j++)
         {
            int jidx = dim < 2 ? 0 : elmin_h[1*nel + e] + j;
            int joff = dim < 2 ? 0 : (elmin_h[1*nel + e] + j) * lmap_n[0];
            for (int i = 0; i < ilim; i++)
            {
               int ioff = elmin_h[0*nel + e] + i;
               int idx = ioff + joff + koff;
               lmap_offset[lmap_offset[idx+1]-hash_el_count[idx]]=e;
               hash_el_count[idx]--;
            }
         }
      }
   }
}

Array<int> BoundingBoxTensorGridMap::HashCellToElements(int i) const
{
   MFEM_ASSERT(i >= 0 && i < lmap_nd-1,
               "Access element " << i << " of local hash with cells = "
               << lmap_nd - 1);
   int start = lmap_offset[i];
   int end = lmap_offset[i + 1];
   Array<int> elements(end - start);
   for (int j = start; j < end; j++)
   {
      elements[j - start] = lmap_offset[j];
   }
   return elements;
}

int BoundingBoxTensorGridMap::GetHashCellFromPoint(Vector &xyz) const
{
   MFEM_ASSERT(xyz.Size() == dim,
               "Point must have the same dimension as the hash.");
   int sum = 0;
   for (int d = dim-1; d >= 0; --d)
   {
      if (xyz(d) < lmap_bnd_min(d) || xyz(d) > lmap_bnd_max(d))
      {
         return -1; // Point is outside the bounds of the hash
      }
      sum *= lmap_n[d];
      int i = (int)floor((xyz(d) - lmap_bnd_min(d)) * lmap_fac[d]);
      sum += i < 0 ? 0 : (lmap_n[d] - 1 < i ? lmap_n[d] - 1 : i);
   }
   return sum;
}

Array<int> BoundingBoxTensorGridMap::MapPointToElements(Vector &xyz) const
{
   MFEM_ASSERT(xyz.Size() == dim,
               "Point must have the same dimension as the hash.");
   int cell = GetHashCellFromPoint(xyz);
   if (cell < 0)
   {
      return Array<int>(); // Point is outside the bounds of the hash
   }
   return HashCellToElements(cell);
}

} // namespace mfem
