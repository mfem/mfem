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

#ifndef MFEM_MESH_PUMI
#define MFEM_MESH_PUMI

#ifndef MFEM_PMESH_PUMI
#define MFEM_PMESH_PUMI

#ifndef MFEM_GRIDFUNC_PUMI
#define MFEM_GRIDFUNC_PUMI

#include "../config/config.hpp"

#ifdef MFEM_USE_PUMI
#ifdef MFEM_USE_MPI

//#include "../fem/gridfunc_pumi.hpp"
#include "../fem/fespace.hpp"
#include "../fem/gridfunc.hpp"
#include "../fem/coefficient.hpp"
#include "../fem/bilininteg.hpp"

#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include "mesh.hpp"
#include "pmesh.hpp"

#include <pumi.h>
#include <apf.h>
#include <apfMesh2.h>
#include <apfShape.h>
#include <apfNumbering.h>
#include <apfDynamicVector.h>


namespace mfem
{

class PumiMesh : public Mesh
{

protected:

   Element *ReadElement(apf::MeshEntity* Ent, const int geom, apf::Downward Verts,
                        const int Attr, apf::Numbering* vert_num);
   void CountBoundaryEntity(apf::Mesh2* apf_mesh, const int BcDim, int &NumBC);

   // Readers for PUMI mesh formats, used in the Load() method.
   void ReadSCORECMesh(apf::Mesh2* apf_mesh, apf::Numbering* v_num_loc,
                       const int curved);

public:

   ///This is to generate a MFEM mesh from a PUMI mesh
   PumiMesh(apf::Mesh2* apf_mesh, int generate_edges = 0, int refine = 1,
            bool fix_orientation = true);

   /** This is to load a PUMI mesh, it is written following the
      steps in MFEM load function*/
   void Load(apf::Mesh2* apf_mesh, int generate_edges = 0, int refine = 1,
             bool fix_orientation = true);


   /// Destroys Mesh.
   virtual ~PumiMesh() { }
};

//Paralel mesh class 
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


//Grid function class
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

}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_PUMI

#endif
#endif
#endif
