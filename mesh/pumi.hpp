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

#ifndef MFEM_PUMI
#define MFEM_PUMI

#include "../config/config.hpp"

#ifdef MFEM_USE_PUMI
#ifdef MFEM_USE_MPI

#include "../fem/fespace.hpp"
#include "../fem/gridfunc.hpp"
#include "../fem/pgridfunc.hpp"
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

/// Base class for PUMI meshes
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
   /// Generate an MFEM mesh from a PUMI mesh.
   PumiMesh(apf::Mesh2* apf_mesh, int generate_edges = 0, int refine = 1,
            bool fix_orientation = true);

   using Mesh::Load;

   /// Load a PUMI mesh (following the steps in the MFEM Load function).
   void Load(apf::Mesh2* apf_mesh, int generate_edges = 0, int refine = 1,
             bool fix_orientation = true);

   /// Destroys Mesh.
   virtual ~PumiMesh() { }
};


/// Class for PUMI parallel meshes
class ParPumiMesh : public ParMesh
{
private:
   apf::Numbering* v_num_loc;

protected:
   Element *ReadElement(apf::MeshEntity* Ent, const int geom, apf::Downward Verts,
                        const int Attr, apf::Numbering* vert_num);

public:
   /// Build a parallel MFEM mesh from a parallel PUMI mesh.
   ParPumiMesh(MPI_Comm comm, apf::Mesh2* apf_mesh);

   /// Transfer field from MFEM mesh to PUMI mesh [Mixed].
   void FieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                        ParGridFunction* Vel,
                        ParGridFunction* Pr,
                        apf::Field* VelField,
                        apf::Field* PrField,
                        apf::Field* VelMagField);

   /// Transfer field from MFEM mesh to PUMI mesh [Scalar].
   void FieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                        ParGridFunction* Pr,
                        apf::Field* PrField,
                        apf::Field* PrMagField);

   /// Transfer field from MFEM mesh to PUMI mesh [Vector].
   void VectorFieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                              ParGridFunction* Vel,
                              apf::Field* VelField,
                              apf::Field* VelMagField);

   /// Update the mesh after adaptation.
   void UpdateMesh(const ParMesh* AdaptedpMesh);

   /// Transfer a field from PUMI to MFEM after mesh adapt [Scalar].
   void FieldPUMItoMFEM(apf::Mesh2* apf_mesh,
                        apf::Field* ScalarField,
                        ParGridFunction* Pr);

   virtual ~ParPumiMesh() { }
};


/// Class for PUMI grid functions
class GridFunctionPumi : public GridFunction
{
public:
   /// Construct a GridFunction from a PUMI mesh.
   GridFunctionPumi(Mesh* m, apf::Mesh2* PumiM, apf::Numbering* v_num_loc,
                    const int mesh_order);

   /// Destroy the grid function.
   virtual ~GridFunctionPumi() { }
};

} // namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_PUMI

#endif
