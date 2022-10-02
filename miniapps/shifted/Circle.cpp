// Copyright (c) 2017, Lawrence Livermore National Security, OALLC. Produced at
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


#include "Circle.hpp"

namespace mfem{

  Circle::Circle(ParFiniteElementSpace &h1_fes, ParFiniteElementSpace &Ph1_fes, bool includeCut): AnalyticalGeometricShape(h1_fes,Ph1_fes, includeCut), radius(0.2), center(2)
 {
   center(0) = 0.5;
   center(1) = 0.5;
  }

  Circle::~Circle(){}
  
  void Circle::SetupElementStatus(Array<int> &elemStatus, Array<int> &ess_inactive, Array<int> &ess_inactive_p){
 
    const int max_elem_attr = (pmesh->attributes).Max();
    int activeCount = 0;
    int inactiveCount = 0;
    int cutCount = 0;
    ess_inactive = -1;
    ess_inactive_p = -1;

  // Check elements on the current MPI rank
  for (int i = 0; i < H1.GetNE(); i++)
    {
      const FiniteElement *FElem = H1.GetFE(i);
      const IntegrationRule &ir = FElem->GetNodes();
      ElementTransformation &T = *H1.GetElementTransformation(i);
      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
	const IntegrationPoint &ip = ir.IntPoint(j);
	Vector x(3);
	T.Transform(ip,x);
	double radiusOfPt = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
	if ( radiusOfPt >= radius){
	  count++;
	}
      }
      // std::cout << " count " << count << std::endl;
      if (count == ir.GetNPoints()){
	elemStatus[i] = SBElementType::INSIDE;
	Array<int> dofs;
	H1.GetElementVDofs(i, dofs);
	activeCount++;
	for (int k = 0; k < dofs.Size(); k++)
	  {
	    ess_inactive[dofs[k]] = 0;	       
	  }
	Array<int> dofs_p;
	PH1.GetElementVDofs(i, dofs_p);
	for (int k = 0; k < dofs_p.Size(); k++)
	  {
	    ess_inactive_p[dofs_p[k]] = 0;	       
	  }
      }
      else if ( (count > 0) && (count < ir.GetNPoints())){
	elemStatus[i] = SBElementType::CUT;
	cutCount++;
	if (include_cut){
	  Array<int> dofs;
	  H1.GetElementVDofs(i, dofs);
	  for (int k = 0; k < dofs.Size(); k++)
	    {
	      ess_inactive[dofs[k]] = 0;
	    }
	  Array<int> dofs_p;
	  PH1.GetElementVDofs(i, dofs_p);
	  for (int k = 0; k < dofs_p.Size(); k++)
	    {
	      ess_inactive_p[dofs_p[k]] = 0;	       
	    }
	}
	else{
	  pmesh->SetAttribute(i, max_elem_attr+1);
	}
      }
      else if (count == 0){
	elemStatus[i] = SBElementType::OUTSIDE;
	inactiveCount++;
	pmesh->SetAttribute(i, max_elem_attr+1);
      }
    }

 
  pmesh->ExchangeFaceNbrNodes();
  for (int i = H1.GetNE(); i < (H1.GetNE() + pmesh->GetNSharedFaces()) ; i++){
    FaceElementTransformations *eltrans = pmesh->GetSharedFaceTransformations(i-H1.GetNE());
    if (eltrans != NULL){
      int Elem2No = eltrans->Elem2No;
      int Elem2NbrNo = Elem2No - pmesh->GetNE();
      const FiniteElement *FElem = H1.GetFE(Elem2No);
      const IntegrationRule &ir = FElem->GetNodes();
      ElementTransformation *nbrftr = H1.GetFaceNbrElementTransformation(Elem2NbrNo);
      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
        const IntegrationPoint &ip = ir.IntPoint(j);
        Vector x(3);
        nbrftr->Transform(ip,x);
	double radiusOfPt = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
	if ( radiusOfPt >= radius){
	  count++;
	}
      }

      if (count == ir.GetNPoints()){
        elemStatus[i] = SBElementType::INSIDE;
      }
      else if ( (count > 0) && (count < ir.GetNPoints())){
        elemStatus[i] = SBElementType::CUT;
      }
      else if (count == 0){
        elemStatus[i] = SBElementType::OUTSIDE;
      }
    }
  }

  std::cout << " active elemSta " << activeCount << " cut " << cutCount << " inacive " << inactiveCount <<  std::endl;
  //  elemStatus.Print(std::cout,1);
     // Synchronize
  for (int i = 0; i < ess_inactive.Size() ; i++) { ess_inactive[i] += 1; }
  H1.Synchronize(ess_inactive);
  for (int i = 0; i < ess_inactive.Size() ; i++) { ess_inactive[i] -= 1; }
  for (int i = 0; i < ess_inactive_p.Size() ; i++) { ess_inactive_p[i] += 1; }
  PH1.Synchronize(ess_inactive_p);
  for (int i = 0; i < ess_inactive_p.Size() ; i++) { ess_inactive_p[i] -= 1; }
  pmesh->SetAttributes();
  
  // ess_inactive.Print(std::cout,1);
  // H1.Synchronize(ess_inactive);
  /*std::cout << " eleSt " << std::endl;
  elemStatus.Print();
  std::cout << " ess inac " << std::endl;
  ess_inactive.Print();*/
  }

  void Circle::ComputeDistanceAndNormalAtCoordinates(const Vector &x, Vector &D, Vector &tN){
    double r = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
    double distX = ((x(0)-center(0))/r)*(radius-r);
    double distY = ((x(1)-center(1))/r)*(radius-r);
    D(0) = distX;
    D(1) = distY;
    double normD = sqrt(distX * distX + distY * distY);
    tN(0) = distX /  normD;
    tN(1) = distY /  normD;
  }	      
}
