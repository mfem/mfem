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

#include "solvers-atpmg.hpp"

#include "../interface/ceed.hpp"
#include "../interface/util.hpp"

#ifdef MFEM_USE_CEED

#include <ceed/backend.h>
#include <math.h>
// todo: should probably use Ceed memory wrappers instead of calloc/free?
#include <stdlib.h>

namespace mfem
{

namespace ceed
{

// In one dimension, return corresponding coarse edof index for
// given fine index, with -1 meaning the edof disappears on the
// coarse grid
int coarse_1d_edof(int i, int P1d, int coarse_P1d)
{
   int coarse_i = (i < coarse_P1d - 1) ? i : -1;
   if (i == P1d - 1)
   {
      coarse_i = coarse_P1d - 1;
   }
   return coarse_i;
}

int reverse_coarse_1d_edof(int i, int P1d, int coarse_P1d)
{
   int coarse_i;
   if (i > P1d - coarse_P1d)
   {
      coarse_i = i - (P1d - coarse_P1d);
   }
   else
   {
      coarse_i = -1;
   }
   if (i == 0)
   {
      coarse_i = 0;
   }
   return coarse_i;
}

int min4(int a, int b, int c, int d)
{
   if (a <= b && a <= c && a <= d)
   {
      return a;
   }
   else if (b <= a && b <= c && b <= d)
   {
      return b;
   }
   else if (c <= a && c <= b && c <= d)
   {
      return c;
   }
   else
   {
      return d;
   }
}

int CeedATPMGElemRestriction(int order,
                             int order_reduction,
                             CeedElemRestriction er_in,
                             CeedElemRestriction* er_out,
                             CeedInt *&dof_map)
{
   int ierr;
   Ceed ceed;
   ierr = CeedElemRestrictionGetCeed(er_in, &ceed); PCeedChk(ierr);

   CeedInt numelem, numcomp, elemsize;
   CeedSize numnodes;
   ierr = CeedElemRestrictionGetNumElements(er_in, &numelem); PCeedChk(ierr);
   ierr = CeedElemRestrictionGetLVectorSize(er_in, &numnodes); PCeedChk(ierr);
   ierr = CeedElemRestrictionGetElementSize(er_in, &elemsize); PCeedChk(ierr);
   ierr = CeedElemRestrictionGetNumComponents(er_in, &numcomp); PCeedChk(ierr);
   if (numcomp != 1)
   {
      // todo: multi-component will require more thought
      return CeedError(ceed, 1, "Algebraic element restriction not "
                       "implemented for multiple components.");
   }

   int P1d = order + 1;
   int coarse_P1d = P1d - order_reduction;
   int dim = (log((double) elemsize) / log((double) P1d)) + 1.e-3;

   CeedVector in_lvec, in_evec;
   ierr = CeedElemRestrictionCreateVector(er_in, &in_lvec, &in_evec);
   PCeedChk(ierr);

   // Create the elem_dof array from the given high-order ElemRestriction
   // by using it to map the L-vector indices to an E-vector
   CeedScalar * lvec_data;
   ierr = CeedVectorGetArrayWrite(in_lvec, CEED_MEM_HOST, &lvec_data);
   PCeedChk(ierr);
   for (CeedSize i = 0; i < numnodes; ++i)
   {
      lvec_data[i] = (CeedScalar) i;
   }
   ierr = CeedVectorRestoreArray(in_lvec, &lvec_data); PCeedChk(ierr);
   CeedInt in_layout[3];
#if CEED_VERSION_GE(0, 13, 0)
   ierr = CeedElemRestrictionGetELayout(er_in, in_layout); PCeedChk(ierr);
#else
   ierr = CeedElemRestrictionGetELayout(er_in, &in_layout); PCeedChk(ierr);
#endif
   if (in_layout[0] == 0 && in_layout[1] == 0 && in_layout[2] == 0)
   {
      return CeedError(ceed, 1, "Cannot interpret e-vector ordering of given"
                       "CeedElemRestriction!");
   }
   ierr = CeedElemRestrictionApply(er_in, CEED_NOTRANSPOSE, in_lvec, in_evec,
                                   CEED_REQUEST_IMMEDIATE); PCeedChk(ierr);
   ierr = CeedVectorDestroy(&in_lvec); PCeedChk(ierr);
   const CeedScalar * in_elem_dof;
   ierr = CeedVectorGetArrayRead(in_evec, CEED_MEM_HOST, &in_elem_dof);
   PCeedChk(ierr);

   // Create a map (dof_map) that maps high-order ldof indices to
   // low-order ldof indices, with -1 indicating no correspondence
   // (NOTE: it is the caller's responsibility to free dof_map)
   dof_map = new CeedInt[numnodes];
   for (CeedSize i = 0; i < numnodes; ++i)
   {
      dof_map[i] = -1;
   }
   CeedInt coarse_elemsize = pow(coarse_P1d, dim);
   CeedInt * out_elem_dof = new CeedInt[coarse_elemsize * numelem];
   const double rounding_guard = 1.e-10;
   int running_out_ldof_count = 0;
   if (dim == 2)
   {
      for (int e = 0; e < numelem; ++e)
      {
         // Loop over edofs in element
         for (int i = 0; i < P1d; ++i)
         {
            for (int j = 0; j < P1d; ++j)
            {
               // Determine topology; is this edof on the outside of the element
               // in the i or j direction?
               int in_edof = i*P1d + j;
               const int edof_index = in_edof*in_layout[0] + e*in_layout[2];
               int in_ldof = in_elem_dof[edof_index] + rounding_guard;
               bool i_edge = (i == 0 || i == P1d - 1);
               bool j_edge = (j == 0 || j == P1d - 1);

               // Determine corresponding coarse 1D edof indices
               // We do this systematically, orienting edges and faces based on ldof
               // orientation, so that the choices are consistent when we visit a
               // shared dof multiple times
               int coarse_i, coarse_j;
               if (i_edge == j_edge)  // edof is a vertex or interior
               {
                  // note that interiors could be done with elements in parallel
                  // (you'd have to rethink numbering but it could be done in advance)
                  coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                  coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
               }
               else  // edof is on an edge but not a vertex
               {
                  // Orient coarse_i, coarse_j based on numbering of ldofs on vertices
                  int left_in_edof, left_in_ldof, right_in_edof, right_in_ldof;
                  if (i_edge)
                  {
                     left_in_edof = i*P1d + 0;
                     right_in_edof = i*P1d + (P1d - 1);
                     left_in_ldof = in_elem_dof[left_in_edof*in_layout[0]+e*in_layout[2]]
                                    + rounding_guard;
                     right_in_ldof = in_elem_dof[right_in_edof*in_layout[0]+e*in_layout[2]]
                                     + rounding_guard;
                     coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                     coarse_j = (left_in_ldof < right_in_ldof) ?
                                coarse_1d_edof(j, P1d, coarse_P1d) : reverse_coarse_1d_edof(j, P1d, coarse_P1d);
                  }
                  else
                  {
                     left_in_edof = 0*P1d + j;
                     right_in_edof = (P1d - 1)*P1d + j;
                     left_in_ldof = in_elem_dof[left_in_edof*in_layout[0]+e*in_layout[2]]
                                    + rounding_guard;
                     right_in_ldof = in_elem_dof[right_in_edof*in_layout[0]+e*in_layout[2]]
                                     + rounding_guard;
                     coarse_i = (left_in_ldof < right_in_ldof) ?
                                coarse_1d_edof(i, P1d, coarse_P1d) : reverse_coarse_1d_edof(i, P1d, coarse_P1d);
                     coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                  }
               }

               // Select edof to be on coarse grid and assign numbering and maps
               if (coarse_i >= 0 && coarse_j >= 0)
               {
                  int out_edof = coarse_i*coarse_P1d + coarse_j;
                  if (dof_map[in_ldof] >= 0)
                  {
                     out_elem_dof[e*coarse_elemsize + out_edof] = dof_map[in_ldof];
                  }
                  else
                  {
                     out_elem_dof[e*coarse_elemsize + out_edof] = running_out_ldof_count;
                     dof_map[in_ldof] = running_out_ldof_count;
                     running_out_ldof_count++;
                  }
               }
            }
         }
      }
   }
   else if (dim == 3)
   {
      // The 3D code is perhaps overly complicated and could be optimized
      for (int e = 0; e < numelem; ++e)
      {
         // Loop over edofs in element
         for (int i = 0; i < P1d; ++i)
         {
            for (int j = 0; j < P1d; ++j)
            {
               for (int k = 0; k < P1d; ++k)
               {
                  // Determine topology; is this edof on the outside of the element
                  // in the i, j, or k direction?
                  int in_edof = i*P1d*P1d + j*P1d + k;
                  int in_ldof = in_elem_dof[in_edof*in_layout[0]+e*in_layout[2]]
                                + rounding_guard;
                  bool i_edge = (i == 0 || i == P1d - 1);
                  bool j_edge = (j == 0 || j == P1d - 1);
                  bool k_edge = (k == 0 || k == P1d - 1);
                  int topo = 0;
                  if (i_edge) { topo++; }
                  if (j_edge) { topo++; }
                  if (k_edge) { topo++; }

                  // Determine corresponding coarse 1D edof indices
                  // We do this systematically, orienting edges and faces based on ldof
                  // orientation, so that the choices are consistent when we visit a
                  // shared dof multiple times
                  int coarse_i, coarse_j, coarse_k;
                  if (topo == 0 || topo == 3)
                  {
                     // edof is a vertex or interior
                     coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                     coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                     coarse_k = coarse_1d_edof(k, P1d, coarse_P1d);
                  }
                  else if (topo == 2)
                  {
                     // edof is on an edge, not a vertex
                     // Orient based on ldof numbering of vertices that define edge
                     int left_in_edof, left_in_ldof, right_in_edof, right_in_ldof;
                     if (!i_edge)
                     {
                        left_in_edof = 0*P1d*P1d + j*P1d + k;
                        right_in_edof = (P1d - 1)*P1d*P1d + j*P1d + k;
                        left_in_ldof = in_elem_dof[left_in_edof*in_layout[0]+e*in_layout[2]]
                                       + rounding_guard;
                        right_in_ldof = in_elem_dof[right_in_edof*in_layout[0]+e*in_layout[2]]
                                        + rounding_guard;
                        coarse_i = (left_in_ldof < right_in_ldof) ?
                                   coarse_1d_edof(i, P1d, coarse_P1d) : reverse_coarse_1d_edof(i, P1d, coarse_P1d);
                        coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                        coarse_k = coarse_1d_edof(k, P1d, coarse_P1d);
                     }
                     else if (!j_edge)
                     {
                        left_in_edof = i*P1d*P1d + 0*P1d + k;
                        right_in_edof = i*P1d*P1d + (P1d - 1)*P1d + k;
                        left_in_ldof = in_elem_dof[left_in_edof*in_layout[0]+e*in_layout[2]]
                                       + rounding_guard;
                        right_in_ldof = in_elem_dof[right_in_edof*in_layout[0]+e*in_layout[2]]
                                        + rounding_guard;
                        coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                        coarse_j = (left_in_ldof < right_in_ldof) ?
                                   coarse_1d_edof(j, P1d, coarse_P1d) : reverse_coarse_1d_edof(j, P1d, coarse_P1d);
                        coarse_k = coarse_1d_edof(k, P1d, coarse_P1d);
                     }
                     else
                     {
                        if (k_edge)
                        {
                           return CeedError(ceed, 1,
                                            "Element connectivity does not make sense!");
                        }
                        left_in_edof = i*P1d*P1d + j*P1d + 0;
                        right_in_edof = i*P1d*P1d + j*P1d + (P1d - 1);
                        left_in_ldof = in_elem_dof[left_in_edof*in_layout[0]+e*in_layout[2]]
                                       + rounding_guard;
                        right_in_ldof = in_elem_dof[right_in_edof*in_layout[0]+e*in_layout[2]]
                                        + rounding_guard;
                        coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                        coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                        coarse_k = (left_in_ldof < right_in_ldof) ?
                                   coarse_1d_edof(k, P1d, coarse_P1d) : reverse_coarse_1d_edof(k, P1d, coarse_P1d);
                     }
                  }
                  else
                  {
                     // edof is on a face, not an edge
                     // Orient based on four vertices that define the face
                     if (topo != 1)
                     {
                        return CeedError(ceed, 1,
                                         "Element connectivity does not match topology!");
                     }
                     int bottom_left_edof, bottom_right_edof, top_left_edof, top_right_edof;
                     int bottom_left_ldof, bottom_right_ldof, top_left_ldof, top_right_ldof;
                     if (i_edge)
                     {
                        bottom_left_edof = i*P1d*P1d + 0*P1d + 0;
                        bottom_right_edof = i*P1d*P1d + 0*P1d + (P1d - 1);
                        top_right_edof = i*P1d*P1d + (P1d - 1)*P1d + (P1d - 1);
                        top_left_edof = i*P1d*P1d + (P1d - 1)*P1d + 0;
                        bottom_left_ldof = in_elem_dof[bottom_left_edof*in_layout[0]+e*in_layout[2]]
                                           + rounding_guard;
                        bottom_right_ldof = in_elem_dof[bottom_right_edof*in_layout[0]+e*in_layout[2]]
                                            + rounding_guard;
                        top_right_ldof = in_elem_dof[top_right_edof*in_layout[0]+e*in_layout[2]]
                                         + rounding_guard;
                        top_left_ldof = in_elem_dof[top_left_edof*in_layout[0]+e*in_layout[2]]
                                        + rounding_guard;
                        int m = min4(bottom_left_ldof, bottom_right_ldof, top_right_ldof,
                                     top_left_ldof);
                        coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                        if (m == bottom_left_ldof)
                        {
                           coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                           coarse_k = coarse_1d_edof(k, P1d, coarse_P1d);
                        }
                        else if (m == bottom_right_ldof)     // j=0, k=P1d-1
                        {
                           coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                           coarse_k = reverse_coarse_1d_edof(k, P1d, coarse_P1d);
                        }
                        else if (m == top_right_ldof)
                        {
                           coarse_j = reverse_coarse_1d_edof(j, P1d, coarse_P1d);
                           coarse_k = reverse_coarse_1d_edof(k, P1d, coarse_P1d);
                        }
                        else     // j=P1d-1, k=0
                        {
                           coarse_j = reverse_coarse_1d_edof(j, P1d, coarse_P1d);
                           coarse_k = coarse_1d_edof(k, P1d, coarse_P1d);
                        }
                     }
                     else if (j_edge)
                     {
                        bottom_left_edof = 0*P1d*P1d + j*P1d + 0;
                        bottom_right_edof = 0*P1d*P1d + j*P1d + (P1d - 1);
                        top_right_edof = (P1d - 1)*P1d*P1d + j*P1d + (P1d - 1);
                        top_left_edof = (P1d - 1)*P1d*P1d + j*P1d + 0;
                        bottom_left_ldof = in_elem_dof[bottom_left_edof*in_layout[0]+e*in_layout[2]]
                                           + rounding_guard;
                        bottom_right_ldof = in_elem_dof[bottom_right_edof*in_layout[0]+e*in_layout[2]]
                                            + rounding_guard;
                        top_right_ldof = in_elem_dof[top_right_edof*in_layout[0]+e*in_layout[2]]
                                         + rounding_guard;
                        top_left_ldof = in_elem_dof[top_left_edof*in_layout[0]+e*in_layout[2]]
                                        + rounding_guard;
                        int m = min4(bottom_left_ldof, bottom_right_ldof, top_right_ldof,
                                     top_left_ldof);
                        coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                        if (m == bottom_left_ldof)
                        {
                           coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                           coarse_k = coarse_1d_edof(k, P1d, coarse_P1d);
                        }
                        else if (m == bottom_right_ldof)     // i=0, k=P1d-1
                        {
                           coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                           coarse_k = reverse_coarse_1d_edof(k, P1d, coarse_P1d);
                        }
                        else if (m == top_right_ldof)
                        {
                           coarse_i = reverse_coarse_1d_edof(i, P1d, coarse_P1d);
                           coarse_k = reverse_coarse_1d_edof(k, P1d, coarse_P1d);
                        }
                        else     // i=P1d-1, k=0
                        {
                           coarse_i = reverse_coarse_1d_edof(i, P1d, coarse_P1d);
                           coarse_k = coarse_1d_edof(k, P1d, coarse_P1d);
                        }
                     }
                     else
                     {
                        if (!k_edge)
                        {
                           return CeedError(ceed, 1,
                                            "Element connectivity does not make sense!");
                        }
                        bottom_left_edof = 0*P1d*P1d + 0*P1d + k;
                        bottom_right_edof = 0*P1d*P1d + (P1d - 1)*P1d + k;
                        top_right_edof = (P1d - 1)*P1d*P1d + (P1d - 1)*P1d + k;
                        top_left_edof = (P1d - 1)*P1d*P1d + 0*P1d + k;
                        bottom_left_ldof = in_elem_dof[bottom_left_edof*in_layout[0]+e*in_layout[2]]
                                           + rounding_guard;
                        bottom_right_ldof = in_elem_dof[bottom_right_edof*in_layout[0]+e*in_layout[2]]
                                            + rounding_guard;
                        top_right_ldof = in_elem_dof[top_right_edof*in_layout[0]+e*in_layout[2]]
                                         + rounding_guard;
                        top_left_ldof = in_elem_dof[top_left_edof*in_layout[0]+e*in_layout[2]]
                                        + rounding_guard;
                        int m = min4(bottom_left_ldof, bottom_right_ldof,
                                     top_right_ldof, top_left_ldof);
                        coarse_k = coarse_1d_edof(k, P1d, coarse_P1d);
                        if (m == bottom_left_ldof)
                        {
                           coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                           coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                        }
                        else if (m == bottom_right_ldof)   // i=0, j=P1d-1
                        {
                           coarse_i = coarse_1d_edof(i, P1d, coarse_P1d);
                           coarse_j = reverse_coarse_1d_edof(j, P1d, coarse_P1d);
                        }
                        else if (m == top_right_ldof)
                        {
                           coarse_i = reverse_coarse_1d_edof(i, P1d, coarse_P1d);
                           coarse_j = reverse_coarse_1d_edof(j, P1d, coarse_P1d);
                        }
                        else   // i=P1d-1, j=0
                        {
                           coarse_i = reverse_coarse_1d_edof(i, P1d, coarse_P1d);
                           coarse_j = coarse_1d_edof(j, P1d, coarse_P1d);
                        }
                     }
                  }

                  // Select edof to be on coarse grid and assign numbering and maps
                  if (coarse_i >= 0 && coarse_j >= 0 && coarse_k >= 0)
                  {
                     int out_edof = coarse_i*coarse_P1d*coarse_P1d + coarse_j*coarse_P1d + coarse_k;
                     if (dof_map[in_ldof] >= 0)
                     {
                        out_elem_dof[e*coarse_elemsize + out_edof] = dof_map[in_ldof];
                     }
                     else
                     {
                        out_elem_dof[e*coarse_elemsize + out_edof] = running_out_ldof_count;
                        dof_map[in_ldof] = running_out_ldof_count;
                        running_out_ldof_count++;
                     }
                  }
               }
            }
         }
      }

   }
   else
   {
      return CeedError(ceed, 1,
                       "CeedATPMGElemRestriction does not yet support this dimension.");
   }

   ierr = CeedVectorRestoreArrayRead(in_evec, &in_elem_dof); PCeedChk(ierr);
   ierr = CeedVectorDestroy(&in_evec); PCeedChk(ierr);

   ierr = CeedElemRestrictionCreate(ceed, numelem, coarse_elemsize, numcomp,
                                    0, running_out_ldof_count,
                                    CEED_MEM_HOST, CEED_COPY_VALUES, out_elem_dof,
                                    er_out); PCeedChk(ierr);

   delete [] out_elem_dof;

   return 0;
}


int CeedBasisATPMGCoarseToFine(Ceed ceed, int P1d, int dim, int order_reduction,
                               CeedBasis *basisc2f)
{
   // this assumes Lobatto nodes on fine and coarse again
   // (not so hard to generalize, but we would have to write it ourselves instead of
   // calling the following Ceed function)
   int ierr;
   ierr = CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P1d - order_reduction, P1d,
                                          CEED_GAUSS_LOBATTO, basisc2f); PCeedChk(ierr);
   return 0;
}

int CeedBasisATPMGCoarseToFine(CeedBasis basisin,
                               CeedBasis *basisc2f,
                               int order_reduction)
{
   int ierr;
   Ceed ceed;
   ierr = CeedBasisGetCeed(basisin, &ceed); PCeedChk(ierr);

   CeedInt dim, P1d;
   ierr = CeedBasisGetDimension(basisin, &dim); PCeedChk(ierr);
   ierr = CeedBasisGetNumNodes1D(basisin, &P1d); PCeedChk(ierr);
   ierr = CeedBasisATPMGCoarseToFine(ceed, P1d, dim, order_reduction,
                                     basisc2f); PCeedChk(ierr);
   return 0;
}

int CeedBasisATPMGCoarsen(CeedBasis basisin,
                          CeedBasis basisc2f,
                          CeedBasis* basisout,
                          int order_reduction)
{
   int ierr;
   Ceed ceed;
   ierr = CeedBasisGetCeed(basisin, &ceed); PCeedChk(ierr);

   CeedInt dim, ncomp, P1d, Q1d;
   ierr = CeedBasisGetDimension(basisin, &dim); PCeedChk(ierr);
   ierr = CeedBasisGetNumComponents(basisin, &ncomp); PCeedChk(ierr);
   ierr = CeedBasisGetNumNodes1D(basisin, &P1d); PCeedChk(ierr);
   ierr = CeedBasisGetNumQuadraturePoints1D(basisin, &Q1d); PCeedChk(ierr);

   CeedInt coarse_P1d = P1d - order_reduction;

   const CeedScalar *interp1d;
   ierr = CeedBasisGetInterp1D(basisin, &interp1d); PCeedChk(ierr);
   const CeedScalar * grad1d;
   ierr = CeedBasisGetGrad1D(basisin, &grad1d); PCeedChk(ierr);

   CeedScalar * coarse_interp1d = new CeedScalar[coarse_P1d * Q1d];
   CeedScalar * coarse_grad1d = new CeedScalar[coarse_P1d * Q1d];
   CeedScalar * fine_nodal_points = new CeedScalar[P1d];

   // these things are in [-1, 1], not [0, 1], which matters
   // (todo: how can we determine this or something related, algebraically?)
   /* one way you might be able to tell is to just run this algorithm
      with coarse_P1d = 2 (i.e., linear) and look for symmetry in the coarse
      basis matrix? */
   ierr = CeedLobattoQuadrature(P1d, fine_nodal_points, NULL); PCeedChk(ierr);
   for (int i = 0; i < P1d; ++i)
   {
      fine_nodal_points[i] = 0.5 * fine_nodal_points[i] + 0.5; // cheating
   }

   const CeedScalar *interp_ctof;
   ierr = CeedBasisGetInterp1D(basisc2f, &interp_ctof); PCeedChk(ierr);

   for (int i = 0; i < Q1d; ++i)
   {
      for (int j = 0; j < coarse_P1d; ++j)
      {
         coarse_interp1d[i * coarse_P1d + j] = 0.0;
         coarse_grad1d[i * coarse_P1d + j] = 0.0;
         for (int k = 0; k < P1d; ++k)
         {
            coarse_interp1d[i * coarse_P1d + j] += interp_ctof[k * coarse_P1d + j] *
                                                   interp1d[i * P1d + k];
            coarse_grad1d[i * coarse_P1d + j] += interp_ctof[k * coarse_P1d + j] *
                                                 grad1d[i * P1d + k];
         }
      }
   }

   const CeedScalar * qref1d;
   ierr = CeedBasisGetQRef(basisin, &qref1d); PCeedChk(ierr);
   const CeedScalar * qweight1d;
   ierr = CeedBasisGetQWeights(basisin, &qweight1d); PCeedChk(ierr);
   ierr = CeedBasisCreateTensorH1(ceed, dim, ncomp,
                                  coarse_P1d, Q1d, coarse_interp1d, coarse_grad1d,
                                  qref1d, qweight1d, basisout); PCeedChk(ierr);

   delete [] fine_nodal_points;
   delete [] coarse_interp1d;
   delete [] coarse_grad1d;

   return 0;
}

int CeedATPMGOperator(CeedOperator oper, int order_reduction,
                      CeedElemRestriction coarse_er,
                      CeedBasis coarse_basis_in,
                      CeedBasis basis_ctof_in,
                      CeedOperator* out)
{
   (void)order_reduction;
   (void)basis_ctof_in;

   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(oper, &ceed); PCeedChk(ierr);

   CeedQFunction qf;
   ierr = CeedOperatorGetQFunction(oper, &qf); PCeedChk(ierr);
   CeedInt numinputfields, numoutputfields;
   CeedQFunctionField *inputqfields, *outputqfields;
   ierr = CeedQFunctionGetFields(qf, &numinputfields, &inputqfields,
                                 &numoutputfields, &outputqfields);
   PCeedChk(ierr);
   CeedOperatorField *inputfields, *outputfields;
   ierr = CeedOperatorGetFields(oper, &numinputfields, &inputfields,
                                &numoutputfields, &outputfields);
   PCeedChk(ierr);

   CeedElemRestriction * er_input = new CeedElemRestriction[numinputfields];
   CeedElemRestriction * er_output = new CeedElemRestriction[numoutputfields];
   CeedVector * if_vector = new CeedVector[numinputfields];
   CeedVector * of_vector = new CeedVector[numoutputfields];
   CeedBasis * basis_input = new CeedBasis[numinputfields];
   CeedBasis * basis_output = new CeedBasis[numoutputfields];
   CeedBasis cbasis = coarse_basis_in;

   int active_input_basis = -1;
   for (int i = 0; i < numinputfields; ++i)
   {
      ierr = CeedOperatorFieldGetElemRestriction(inputfields[i],
                                                 &er_input[i]); PCeedChk(ierr);
      ierr = CeedOperatorFieldGetVector(inputfields[i], &if_vector[i]);
      PCeedChk(ierr);
      ierr = CeedOperatorFieldGetBasis(inputfields[i], &basis_input[i]);
      PCeedChk(ierr);
      if (if_vector[i] == CEED_VECTOR_ACTIVE)
      {
         if (active_input_basis < 0)
         {
            active_input_basis = i;
         }
         else if (basis_input[i] != basis_input[active_input_basis])
         {
            return CeedError(ceed, 1, "Two different active input basis!");
         }
      }
   }
   for (int i = 0; i < numoutputfields; ++i)
   {
      ierr = CeedOperatorFieldGetElemRestriction(outputfields[i],
                                                 &er_output[i]); PCeedChk(ierr);
      ierr = CeedOperatorFieldGetVector(outputfields[i], &of_vector[i]);
      PCeedChk(ierr);
      ierr = CeedOperatorFieldGetBasis(outputfields[i], &basis_output[i]);
      PCeedChk(ierr);
      if (of_vector[i] == CEED_VECTOR_ACTIVE)
      {
         // should already be coarsened
         if (basis_output[i] != basis_input[active_input_basis])
         {
            return CeedError(ceed, 1, "Input and output basis do not match!");
         }
         if (er_output[i] != er_input[active_input_basis])
         {
            return CeedError(ceed, 1, "Input and output elem-restriction do not match!");
         }
      }
   }

   CeedOperator coper;
   ierr = CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                             &coper); PCeedChk(ierr);

   for (int i = 0; i < numinputfields; ++i)
   {
#if CEED_VERSION_GE(0, 13, 0)
      const char * fieldname;
#else
      char * fieldname;
#endif
      ierr = CeedQFunctionFieldGetName(inputqfields[i], &fieldname); PCeedChk(ierr);
      if (if_vector[i] == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorSetField(coper, fieldname, coarse_er, cbasis,
                                     if_vector[i]); PCeedChk(ierr);
      }
      else
      {
         ierr = CeedOperatorSetField(coper, fieldname, er_input[i], basis_input[i],
                                     if_vector[i]); PCeedChk(ierr);
      }
#if CEED_VERSION_GE(0, 13, 0)
      ierr = CeedVectorDestroy(&if_vector[i]); PCeedChk(ierr);
      ierr = CeedElemRestrictionDestroy(&er_input[i]); PCeedChk(ierr);
      ierr = CeedBasisDestroy(&basis_input[i]); PCeedChk(ierr);
#endif
   }
   for (int i = 0; i < numoutputfields; ++i)
   {
#if CEED_VERSION_GE(0, 13, 0)
      const char * fieldname;
#else
      char * fieldname;
#endif
      ierr = CeedQFunctionFieldGetName(outputqfields[i], &fieldname); PCeedChk(ierr);
      if (of_vector[i] == CEED_VECTOR_ACTIVE)
      {
         ierr = CeedOperatorSetField(coper, fieldname, coarse_er, cbasis,
                                     of_vector[i]); PCeedChk(ierr);
      }
      else
      {
         ierr = CeedOperatorSetField(coper, fieldname, er_output[i], basis_output[i],
                                     of_vector[i]); PCeedChk(ierr);
      }
#if CEED_VERSION_GE(0, 13, 0)
      ierr = CeedVectorDestroy(&of_vector[i]); PCeedChk(ierr);
      ierr = CeedElemRestrictionDestroy(&er_output[i]); PCeedChk(ierr);
      ierr = CeedBasisDestroy(&basis_output[i]); PCeedChk(ierr);
#endif
   }
   delete [] er_input;
   delete [] er_output;
   delete [] if_vector;
   delete [] of_vector;
   delete [] basis_input;
   delete [] basis_output;

   *out = coper;
   return 0;
}

int CeedATPMGOperator(CeedOperator oper, int order_reduction,
                      CeedElemRestriction coarse_er,
                      CeedBasis *coarse_basis_out,
                      CeedBasis *basis_ctof_out,
                      CeedOperator *out)
{
   int ierr;

   CeedQFunction qf;
   ierr = CeedOperatorGetQFunction(oper, &qf); PCeedChk(ierr);
   CeedInt numinputfields, numoutputfields;
   CeedOperatorField *inputfields;
   ierr = CeedOperatorGetFields(oper, &numinputfields, &inputfields,
                                &numoutputfields, NULL);
   PCeedChk(ierr);

   CeedBasis basis;
   ierr = CeedOperatorGetActiveBasis(oper, &basis); PCeedChk(ierr);
   ierr = CeedBasisATPMGCoarseToFine(basis, basis_ctof_out, order_reduction);
   PCeedChk(ierr);
   ierr = CeedBasisATPMGCoarsen(basis, *basis_ctof_out, coarse_basis_out,
                                order_reduction); PCeedChk(ierr);
   ierr = CeedATPMGOperator(oper, order_reduction, coarse_er, *coarse_basis_out,
                            *basis_ctof_out, out); PCeedChk(ierr);
   return 0;
}

int CeedOperatorGetOrder(CeedOperator oper, CeedInt * order)
{
   int ierr;

   CeedOperatorField active_field;
   ierr = CeedOperatorGetActiveField(oper, &active_field); PCeedChk(ierr);
   CeedBasis basis;
   ierr = CeedOperatorFieldGetBasis(active_field, &basis); PCeedChk(ierr);
   int P1d;
   ierr = CeedBasisGetNumNodes1D(basis, &P1d); PCeedChk(ierr);
   *order = P1d - 1;
#if CEED_VERSION_GE(0, 13, 0)
   ierr = CeedBasisDestroy(&basis); PCeedChk(ierr);
#endif
   return 0;
}

int CeedATPMGBundle(CeedOperator oper, int order_reduction,
                    CeedBasis* coarse_basis_out,
                    CeedBasis* basis_ctof_out,
                    CeedElemRestriction* er_out,
                    CeedOperator* coarse_oper,
                    CeedInt *&dof_map)
{
   int ierr;
   CeedInt order;
   ierr = CeedOperatorGetOrder(oper, &order); PCeedChk(ierr);
   CeedElemRestriction ho_er;
   ierr = CeedOperatorGetActiveElemRestriction(oper, &ho_er); PCeedChk(ierr);
   ierr = CeedATPMGElemRestriction(order, order_reduction, ho_er, er_out, dof_map);
   PCeedChk(ierr);
   ierr = CeedATPMGOperator(oper, order_reduction, *er_out, coarse_basis_out,
                            basis_ctof_out, coarse_oper); PCeedChk(ierr);
   return 0;
}

} // namespace ceed

} // namespace mfem

#endif // MFEM_USE_CEED
