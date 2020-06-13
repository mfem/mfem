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

#include <apf.h>
#include <apfMesh2.h>
#include <apfShape.h>
#include <apfField.h>
#include <apfNumbering.h>
#include <apfDynamicVector.h>
#include <maMesh.h>

namespace mfem
{

/// Base class for PUMI meshes
class PumiMesh : public Mesh
{
protected:
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
   // This has to persist during an adaptive simulation, and therefore
   // needs to be updated each time the mesh changes.
   apf::Numbering* v_num_loc;

public:
   /// Build a parallel MFEM mesh from a parallel PUMI mesh.
   ParPumiMesh(MPI_Comm comm, apf::Mesh2* apf_mesh,
               int refine = 1, bool fix_orientation = true);


   /// Returns the PUMI-to-MFEM permutation (aka rotation, aka orientation)
   /** This represents the change in tet-to-vertex connectivity between
       the PUMI and MFEM meshes. E.g.,
       PUMI_tet{v0,v1,v2,v3}  --->  MFEM_tet{v1,v0,v3,v2}
       * Note that change in the orientation can be caused by
         a) fixing wrong boundary element orientations
         b) a call to ReorientTetMesh() which is required for Nedelec */
   int RotationPUMItoMFEM(apf::Mesh2* apf_mesh,
                          apf::MeshEntity* tet,
                          int elemId);
   /// Convert the parent coordinate from PUMI to MFEM
   /** By default this functions assumes that there is always
       change in the orientations of some of the elements. In case it
       is known for sure that there is NO change in the orientation,
       call the functions with last argument = false */
   IntegrationRule ParentXisPUMItoMFEM(apf::Mesh2* apf_mesh,
                                       apf::MeshEntity* tet,
                                       int elemId,
                                       apf::NewArray<apf::Vector3>& pumi_xi,
                                       bool checkOrientation = true);
   /// Convert the parent coordinate from MFEM to PUMI
   /** This is the inverse of ParentXisPUMItoMFEM.
       By default this functions assumes that there is always
       change in the orientations of some of the elements. In case it
       is known for sure that there is NO change in the orientation,
       call the functions with last argument = false */
   void ParentXisMFEMtoPUMI(apf::Mesh2* apf_mesh,
                            int elemId,
                            apf::MeshEntity* tet,
                            const IntegrationRule& mfem_xi,
                            apf::NewArray<apf::Vector3>& pumi_xi,
                            bool checkOrientation = true);
   /// Transfer field from MFEM mesh to PUMI mesh [Mixed].
   void FieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                        ParGridFunction* grid_vel,
                        ParGridFunction* grid_pr,
                        apf::Field* vel_field,
                        apf::Field* pr_field,
                        apf::Field* vel_mag_field);

   /// Transfer field from MFEM mesh to PUMI mesh [Scalar].
   void FieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                        ParGridFunction* grid_pr,
                        apf::Field* pr_field,
                        apf::Field* pr_mag_field);

   /// Transfer field from MFEM mesh to PUMI mesh [Vector].
   void VectorFieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                              ParGridFunction* grid_vel,
                              apf::Field* vel_field,
                              apf::Field* vel_mag_field);

   /// Transfer Nedelec field from MFEM mesh to PUMI mesh [Vector].
   void NedelecFieldMFEMtoPUMI(apf::Mesh2* apf_mesh,
                               ParGridFunction* gf,
                               apf::Field* nedelec_field);

   /// Update the mesh after adaptation.
   void UpdateMesh(const ParMesh* AdaptedpMesh);

   /// Transfer a field from PUMI to MFEM after mesh adapt [Scalar and Vector].
   void FieldPUMItoMFEM(apf::Mesh2* apf_mesh,
                        apf::Field* field,
                        ParGridFunction* grid);

   virtual ~ParPumiMesh() {}
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
