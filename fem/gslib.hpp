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

//    finds r,s,t,e,p for given x,y,z
      void gslib_findpts(Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,
           Vector *pr,Vector *pd,Vector *xp, Vector *yp, Vector *zp, int nxyz);
//    xyz is a single Vector
      void gslib_findpts(Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,
           Vector *pr,Vector *pd,Vector *xyzp, int nxyz);

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
