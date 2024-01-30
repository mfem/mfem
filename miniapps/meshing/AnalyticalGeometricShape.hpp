// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_ANALYTICAL_GEOMETRIC_SHAPE
#define MFEM_ANALYTICAL_GEOMETRIC_SHAPE

#include "mfem.hpp"

namespace mfem{

class AnalyticalGeometricShape{
protected:
   ParFiniteElementSpace &pfes_mesh;
   ParGridFunction &distance_gf;
   const ParMesh &pmesh;
   Array<int> ess_vdofs;
   const ParGridFunction &coord;
public:
   /// Element type related to shifted boundaries (not interfaces).
   /// For more than 1 level-set, we set the marker to CUT+level_set_index
   /// to discern between different level-sets.

   AnalyticalGeometricShape(ParFiniteElementSpace &pfes_mesh, ParGridFunction &distance_gf, const ParMesh & pmesh, const ParGridFunction & coord, Array<int> &ess_vdofs);
   virtual void GetTFromX(Vector &coordsT, const Vector &coordsX, const int& j_x, const Vector &dist) = 0;
   virtual void GetXFromT(Vector &coordsX, const Vector &coordsT, const int& j_x, const Vector &dist) = 0;
   virtual void ComputeDistances(const ParGridFunction &coord, const ParMesh & pmesh, const ParFiniteElementSpace &pfes_mesh) = 0;
   virtual void SetScaleMatrix(const Vector &elfun, const Array<int> & vdofs, int i, int j, DenseMatrix &Pmat_scale) = 0;
   virtual void SetScaleMatrixFourthOrder(const Vector &elfun, const Array<int> & vdofs, DenseMatrix &Pmat_scale) = 0;
   virtual void SetHessianScaleMatrix(const Vector &elfun, const Array<int> & vdofs, int i, int idim, int j, int jdim, DenseMatrix &Pmat_hessian) = 0; 
   virtual void convertToPhysical(const Array<int> & vdofs,const Vector &elfun, Vector &convertedX) = 0;
   virtual  ~AnalyticalGeometricShape();
};
}
#endif // MFEM_LAGHOS
