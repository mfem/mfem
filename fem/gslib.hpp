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

//    Interpolates fieldin for given r,s,t,e,p and puts it in fieldout
      void gslib_findpts_eval (Vector *fieldout,Array<uint> *pcode,Array<uint> *pproc,
           Array<uint> *pel,Vector *pr,Vector *fieldin, int nxyz);

//    Interpolates ParGrudFunction for given r,s,t,e,p and puts it in fieldout
#ifdef MFEM_USE_MPI
      void gslib_findpts_eval (Vector *fieldout,Array<uint> *pcode,Array<uint> *pproc,
           Array<uint> *pel,Vector *pr,ParGridFunction *fieldin, int nxyz);
#else
      void gslib_findpts_eval (Vector *fieldout,Array<uint> *pcode,Array<uint> *pproc,
           Array<uint> *pel,Vector *pr,GridFunction *fieldin, int nxyz);
#endif

//    Convert gridfunction to double
#ifdef MFEM_USE_MPI
      void gf2vec(ParGridFunction *fieldin, Vector *fieldout);
#else
      void gf2vec(GridFunction *fieldin, Vector *fieldout);
#endif

//    Get 
      inline int GetFptMeshSize() const { return msz; }
      inline int GetQorder() const { return qo; }

//    clears up memory
      void gslib_findpts_free();

      ~findpts_gslib();
};

} // namespace mfem

#endif //MFEM_USE_GSLIB
#endif //MFEM_GSLIB guard
