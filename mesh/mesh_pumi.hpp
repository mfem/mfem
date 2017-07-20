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

#include "../config/config.hpp"

#ifdef MFEM_USE_SCOREC
#ifdef MFEM_USE_MPI

#include "../fem/gridfunc_pumi.hpp"
#include <iostream>
#include "mesh.hpp"

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

}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SCOREC

#endif
