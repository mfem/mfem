// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_GSLIB
#define MFEM_GSLIB

#include "../config/config.hpp"
#include "gridfunc.hpp"

#ifdef MFEM_USE_GSLIB

struct comm;
struct findpts_data_2;
struct findpts_data_3;

namespace mfem
{

class FindPointsGSLIB
{
protected:
   Mesh *mesh;
   Vector gsl_mesh;
   struct findpts_data_2 *fdata2D;
   struct findpts_data_3 *fdata3D;
   int dim;

   struct comm *gsl_comm;

   void GetNodeValues(const GridFunction &gf_in, Vector &node_vals);

public:
   FindPointsGSLIB();

#ifdef MFEM_USE_MPI
   FindPointsGSLIB(MPI_Comm _comm);
#endif

   ~FindPointsGSLIB();

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of the input Mesh object @a m.
       Note: not tested with periodic (DG meshes).
       Note: the input mesh @a m must have Nodes set.

       @param[in] m         Input mesh.
       @param[in] bb_t      Relative size of bounding box around each element.
       @param[in] newt_tol  Newton tolerance for the gslib search methods.
       @param[in] npt_max   Number of points for simultaneous iteration. This
                            alters performance and memory footprint. */
   void Setup(Mesh &m, double bb_t, double newt_tol, int npt_max);

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
   void FindPoints(Vector &point_pos, Array<unsigned int> &codes,
                   Array<unsigned int> &proc_ids, Array<unsigned int> &elem_ids,
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
                              @a field_in is in H1 and in the same space as the
                              mesh that was given to Setup().
       @param[out] field_out  Interpolated values. */
   void Interpolate(Array<unsigned int> &codes, Array<unsigned int> &proc_ids,
                    Array<unsigned int> &elem_ids, Vector &ref_pos,
                    const GridFunction &field_in, Vector &field_out);

   /** Cleans up memory allocated internally by gslib.
       Note that in parallel, this must be called before MPI_Finalize(), as
       it calls MPI_Comm_free() for internal gslib communicators. */
   void FreeData();
};

} // namespace mfem

#endif //MFEM_USE_GSLIB

#endif //MFEM_GSLIB guard
