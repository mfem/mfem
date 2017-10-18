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

#ifndef MFEM_PMESH_PUMI
#define MFEM_PMESH_PUMI

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_SCOREC

#include "pmesh.hpp"
#include "mesh_pumi.hpp"

namespace mfem
{

/// Class for parallel meshes
class ParPumiMesh : public ParMesh
{
protected:
   Element *ReadElement(apf::MeshEntity* Ent, const int geom, apf::Downward Verts,
                        const int Attr, apf::Numbering* vert_num);
public:
   ///Build a parallel MFEM mesh from a parallel PUMI mesh
   ParPumiMesh(MPI_Comm comm, apf::Mesh2* apf_mesh);

   virtual ~ParPumiMesh() {};
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SCOREC

#endif
