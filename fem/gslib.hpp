// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_GSLIB
#define MFEM_GSLIB

#include "gridfunc.hpp"
#include "pgridfunc.hpp"

#ifdef MFEM_USE_GSLIB

#include "gslib.h"

namespace mfem
{

class findpts_gslib
{
private:
   IntegrationRule ir;
   Vector gllmesh;
   struct findpts_data_2 *fda;
   struct findpts_data_3 *fdb;
   int dim, nel, qo, msz;

   struct comm cc;

   void GetNodeValues(const GridFunction &gf_in, Vector &node_vals);

public:
   findpts_gslib();

#ifdef MFEM_USE_MPI
   findpts_gslib(MPI_Comm _comm);
#endif

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of @a mesh.
       Note: not tested with periodic (DG meshes).
       Note: the given @a mesh must have Nodes set.

       @param[in] bb_t      Relative size of bounding box around each element.
       @param[in] newt_tol  Newton tolerance for the gslib search methods.
       @param[in] npt_max   Number of points for simultaneous iteration. This
                            alters performance and memory footprint. */
   void gslib_findpts_setup(Mesh &mesh, double bb_t,
                            double newt_tol, int npt_max);

   /** Searches positions given in physical space by @a point_pos. All output
       Arrays and Vectors are expected to have the correct size.

       @param[in]  point_pos  Positions to be found. Must by ordered by nodes
                              (XXX...,YYY...,ZZZ).
       @param[out] codes      Return codes for each point: inside element (0),
                              element boundary (1), not found (2).
       @param[out] proc_ids   MPI proc ids where the points were found.
       @param[out] elem_ids   Element ids where the points were found.
       @param[out] ref_pos    Reference coordinates of the found point. Ordered
                              by vdim (XYZ,XYZ,XYZ...).
                              Note: the gslib reference frame is [-1,1].
       @param[out] dist       Distance between the seeked and the found point
                              in physical space. */
   void gslib_findpts(Vector &point_pos, Array<uint> &codes,
                      Array<uint> &proc_ids, Array<uint> &elem_ids,
                      Vector &ref_pos, Vector &dist);

   /** Interpolation of field values at prescribed reference space positions.

       @param[in] codes       Return codes for each point: inside element (0),
                              element boundary (1), not found (2).
       @param[in] proc_ids    MPI proc ids where the points were found.
       @param[in] elem_ids    Element ids where the points were found.
       @param[in] ref_pos     Reference coordinates of the found point. Ordered
                              by vdim (XYZ,XYZ,XYZ...).
                              Note: the gslib reference frame is [-1,1].
       @param[in] field_in    Function values that will be interpolated on the
                              reference positions. Note: it is assumed that
                              @a field_in is associated with the mesh given to
                              gslib_findpts_setup.
       @param[out] field_out  Interpolated values. */
   void gslib_findpts_eval(Array<uint> &codes, Array<uint> &proc_ids,
                           Array<uint> &elem_ids, Vector &ref_pos,
                           const GridFunction &field_in, Vector &field_out);

//    clears up memory
      void gslib_findpts_free();

      ~findpts_gslib();
};

} // namespace mfem

#endif //MFEM_USE_GSLIB
#endif //MFEM_GSLIB guard
