// Copyright (c) 2017, Lawrence LivermoreA National Security, LLC. Produced at
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

#include "AnalyticalSurface.hpp"

namespace mfem
{

  AnalyticalSurface::AnalyticalSurface(int geometryType, ParFiniteElementSpace &h1_fes):
    geometryType(geometryType),
    H1(h1_fes),
    pmesh(h1_fes.GetParMesh()),
    b_ir(IntRules.Get((pmesh->GetBdrFaceTransformations(0))->GetGeometryType(), 4*H1.GetOrder(0) + 4*(pmesh->GetBdrFaceTransformations(0))->OrderW() )),
    elementalStatus(h1_fes.GetNE()+pmesh->GetNSharedFaces()),
    faceTags(h1_fes.GetNF()+pmesh->GetNSharedFaces()),
    initialBoundaryFaceTags(h1_fes.GetNBE()),
    initialElementTags(h1_fes.GetNE()),
    quadratureDistance((h1_fes.GetNF()+pmesh->GetNSharedFaces()) * b_ir.GetNPoints(),pmesh->Dimension()),
    quadratureTrueNormal((h1_fes.GetNF()+pmesh->GetNSharedFaces()) * b_ir.GetNPoints(),pmesh->Dimension()),
    maxBoundaryTag(0),
    ess_edofs(h1_fes.GetVSize()),
    geometry(NULL)
  {
    ess_edofs = -1;
    elementalStatus = AnalyticalGeometricShape::SBElementType::INSIDE;
    quadratureDistance = 0.0;
    quadratureTrueNormal = 0.0;
    std::cout << " size of owned face " << h1_fes.GetNF() << " shared " << pmesh->GetNSharedFaces() << std::endl;
    int localMaxBoundaryTag = 0;
    for (int i = 0; i < H1.GetNBE(); i++)
      {    
	FaceElementTransformations *eltrans_bound = pmesh->GetBdrFaceTransformations(i);
	const int faceElemNo = eltrans_bound->ElementNo;
	initialBoundaryFaceTags[i] = pmesh->GetBdrAttribute(faceElemNo);
	if (localMaxBoundaryTag < initialBoundaryFaceTags[i]){
	  localMaxBoundaryTag = initialBoundaryFaceTags[i];
	}
      }

    MPI_Allreduce(&localMaxBoundaryTag, &maxBoundaryTag, 1, MPI_INT, MPI_MAX, pmesh->GetComm());
    faceTags = maxBoundaryTag;
    for (int i = 0; i < H1.GetNE(); i++)
      {    
	ElementTransformation *eltrans = pmesh->GetElementTransformation(i);
	const int ElemNo = eltrans->ElementNo;
	initialElementTags[i] = pmesh->GetAttribute(ElemNo);
      }
    switch (geometryType)
      {
      case 1: geometry = new Circle(H1); break;
      default:
	out << "Unknown geometry type: " << geometryType << '\n';
	break;
      }
  }

  AnalyticalSurface::~AnalyticalSurface(){
    delete geometry;
  }
  
  void AnalyticalSurface::SetupElementStatus(){
    geometry->SetupElementStatus(elementalStatus,ess_edofs);
  }

  void AnalyticalSurface::SetupFaceTags(){
    geometry->SetupFaceTags(elementalStatus, faceTags, ess_edofs, initialBoundaryFaceTags, maxBoundaryTag);
  }

  void AnalyticalSurface::ComputeDistanceAndNormalAtQuadraturePoints(){
    geometry->ComputeDistanceAndNormalAtQuadraturePoints(b_ir, elementalStatus, faceTags, quadratureDistance, quadratureTrueNormal);
  }
  void AnalyticalSurface::ComputeDistanceAndNormalAtCoordinates(const Vector &x, Vector &D, Vector &tN){
    geometry->ComputeDistanceAndNormalAtCoordinates(x, D, tN);
  }
 
  void AnalyticalSurface::ResetData(){
    quadratureDistance = 0.0;
    quadratureTrueNormal = 0.0;
    ess_edofs = -1;
    elementalStatus = AnalyticalGeometricShape::SBElementType::INSIDE;
    faceTags = maxBoundaryTag;
    for (int i = 0; i < H1.GetNE(); i++)
      {    
	ElementTransformation *eltrans = H1.GetElementTransformation(i);
	const int ElemNo = eltrans->ElementNo;
	pmesh->SetAttribute(i,initialElementTags[i]);
      }
    pmesh->SetAttributes();
  }

  Array<int>& AnalyticalSurface::GetEss_Vdofs(){
    return ess_edofs;
  }
  Array<int>& AnalyticalSurface::GetFace_Tags(){
    return faceTags;
  }
  Array<int>& AnalyticalSurface::GetElement_Status(){
    return elementalStatus;
  }
  const DenseMatrix& AnalyticalSurface::GetQuadratureDistance(){
    return quadratureDistance;
  }
  const DenseMatrix& AnalyticalSurface::GetQuadratureTrueNormal(){
    return quadratureTrueNormal;
  }
}
