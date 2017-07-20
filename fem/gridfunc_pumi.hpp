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

#ifndef MFEM_GRIDFUNC_PUMI
#define MFEM_GRIDFUNC_PUMI

#include "../config/config.hpp"

#ifdef MFEM_USE_SCOREC
#ifdef MFEM_USE_MPI

#include "fespace.hpp"
#include "gridfunc.hpp"
#include "coefficient.hpp"
#include "bilininteg.hpp"
#include <limits>
#include <ostream>
#include <string>

#include <pumi.h>
#include <apf.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include <apfDynamicVector.h>

namespace mfem
{

/// Class for grid function - Vector with associated FE space.
class GridFunctionPumi : public GridFunction 
{
public:

   ///Construct a GridFunction from PUMI mesh
   GridFunctionPumi(Mesh* m, apf::Mesh2* PumiM, apf::Numbering* v_num_loc, 
           const int mesh_order);
  
   /// Destroys grid function.
   virtual ~GridFunctionPumi() { }
};


} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SCOREC

#endif
