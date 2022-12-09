// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "ghost_penalty.hpp"
#include <unordered_map>

namespace mfem
{
  
  void GhostStressPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
							    const FiniteElement &fe2,
							    FaceElementTransformations &Tr,
							    DenseMatrix &elmat)
  {
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();

    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    int NEproc = pmesh->GetNE();
    int elem1 = Tr.Elem1No;
    int elem2 = Tr.Elem2No;

    int elemStatus1 = elemStatus[elem1];
    int elemStatus2;
    if (Tr.Elem2No >= NEproc)
      {
        elemStatus2 = elemStatus[NEproc+par_shared_face_count];
	par_shared_face_count++;
      }
    else
      {
        elemStatus2 = elemStatus[elem2];
      }

    const int e = Tr.ElementNo;
    bool elem1_inside = (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE);
    bool elem1_cut = (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT);
    bool elem1_outside = (elemStatus1 == AnalyticalGeometricShape::SBElementType::OUTSIDE);
    
    bool elem2_inside = (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE);
    bool elem2_cut = (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT);
    bool elem2_outside = (elemStatus2 == AnalyticalGeometricShape::SBElementType::OUTSIDE);
    if ( (elem1_inside && elem2_cut) || (elem1_cut && elem2_inside) ||  (elem1_cut && elem2_cut) ) {
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim);
      Vector shape_el1(h1dofs_cnt), shape_el2(h1dofs_cnt);
      DenseMatrix gradUResDotShape_el1(h1dofs_cnt,dim), gradUResDotShape_el2(h1dofs_cnt,dim);
      shape_el1 = 0.0;
      gradUResDotShape_el1 = 0.0;
      shape_el2 = 0.0;
      gradUResDotShape_el2 = 0.0;

      nor = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 10 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      DenseMatrix nodalGrad_el1;
      DenseMatrix nodalGrad_el2;
      fe.ProjectGrad(fe,Trans_el1,nodalGrad_el1);
      fe2.ProjectGrad(fe2,Trans_el2,nodalGrad_el2);

      for (int q = 0; q < nqp_face; q++)
	{
      	  
	  nor = 0.0;
	  shape_el1 = 0.0;
	  gradUResDotShape_el1 = 0.0;
	  shape_el2 = 0.0;
	  gradUResDotShape_el2 = 0.0;
      
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);

	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);
	  double Kappa = kappa->Eval(*Tr.Elem1, eip_el1);

	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);

	  double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	  weighted_h = pow(weighted_h,2*nTerms-1);

	  // element 1
	  fe.CalcShape(eip_el1, shape_el1);
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int j = 0; j < dim; j++){
	      for (int k = 0; k < h1dofs_cnt; k++){
		gradUResDotShape_el1(s,j) += nodalGrad_el1(k + j * h1dofs_cnt, s) * shape_el1(k);
	      }
	    }
	  }
	
	  // element 2
	  fe2.CalcShape(eip_el2, shape_el2);
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int j = 0; j < dim; j++){
	      for (int k = 0; k < h1dofs_cnt; k++){
		gradUResDotShape_el2(s,j) += nodalGrad_el2(k + j * h1dofs_cnt, s) * shape_el2(k);
	      }
	    }
	  }
	  double standardFactor =  nor_norm * ip_f.weight * 2 * (1.0/std::max(3 * Kappa, 2 * Mu)) * penaltyParameter;
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int sd = 0; sd < dim; sd++)
			{
			  elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,sd) * gradUResDotShape_el1(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * standardFactor;
			  elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,sd) * gradUResDotShape_el1(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * standardFactor;
			  elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,sd) * gradUResDotShape_el2(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * standardFactor;
			  elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,sd) * gradUResDotShape_el2(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * standardFactor;
			  for (int od = 0; od < dim; od++)
			    {
			      /////////////////////
			      elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt) += gradUResDotShape_el1(j,od) * gradUResDotShape_el1(i,sd) * weighted_h * Mu * Mu * (nor(sd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,od) * gradUResDotShape_el1(i,sd) * weighted_h * Mu * Mu * (nor(sd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt) -= gradUResDotShape_el1(j,od) * gradUResDotShape_el2(i,sd) * weighted_h * Mu * Mu * (nor(sd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,od) * gradUResDotShape_el2(i,sd) * weighted_h * Mu * Mu * (nor(sd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      ////

			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,vd) * gradUResDotShape_el1(i,od) * weighted_h * Mu * Mu * (nor(od) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt+ dim * h1dofs_cnt) -= gradUResDotShape_el2(j,vd) * gradUResDotShape_el1(i,od) * weighted_h * Mu * Mu * (nor(od) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,vd) * gradUResDotShape_el2(i,od) * weighted_h * Mu * Mu * (nor(od) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,vd) * gradUResDotShape_el2(i,od) * weighted_h * Mu * Mu * (nor(od) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      ////
			      
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,sd) * gradUResDotShape_el1(i,od) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(vd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,sd) * gradUResDotShape_el1(i,od) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(vd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,sd) * gradUResDotShape_el2(i,od) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(vd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,sd) * gradUResDotShape_el2(i,od) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(vd) / nor_norm) * standardFactor;

			      //////////////////

			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,od) * gradUResDotShape_el1(i,sd) * weighted_h * Mu * Mu * (nor(vd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,od) * gradUResDotShape_el1(i,sd) * weighted_h * Mu * Mu * (nor(vd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,od) * gradUResDotShape_el2(i,sd) * weighted_h * Mu * Mu * (nor(vd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,od) * gradUResDotShape_el2(i,sd) * weighted_h * Mu * Mu * (nor(vd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      

			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,od) * gradUResDotShape_el1(i,od) * weighted_h * Mu * Mu * (nor(vd) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,od) * gradUResDotShape_el1(i,od) * weighted_h * Mu * Mu * (nor(vd) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,od) * gradUResDotShape_el2(i,od) * weighted_h * Mu * Mu * (nor(vd) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,od) * gradUResDotShape_el2(i,od) * weighted_h * Mu * Mu * (nor(vd) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      
			      
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,sd) * gradUResDotShape_el1(i,od) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(vd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,sd) * gradUResDotShape_el1(i,od) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(vd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,sd) * gradUResDotShape_el2(i,od) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(vd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,sd) * gradUResDotShape_el2(i,od) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(vd) / nor_norm) * standardFactor;
			      ///////////////////

			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,od) * gradUResDotShape_el1(i,vd) * weighted_h * Mu * (Kappa - (2.0/3.0) * Mu) * (nor(sd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,od) * gradUResDotShape_el1(i,vd) * weighted_h * Mu * (Kappa - (2.0/3.0) * Mu) * (nor(sd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,od) * gradUResDotShape_el2(i,vd) * weighted_h * Mu * (Kappa - (2.0/3.0) * Mu) * (nor(sd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,od) * gradUResDotShape_el2(i,vd) * weighted_h * Mu * (Kappa - (2.0/3.0) * Mu) * (nor(sd) / nor_norm) * (nor(od) / nor_norm) * standardFactor;

			      
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,od) * gradUResDotShape_el1(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,od) * gradUResDotShape_el1(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,od) * gradUResDotShape_el2(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,od) * gradUResDotShape_el2(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * (nor(od) / nor_norm) * (nor(sd) / nor_norm) * standardFactor;			      
			      ///////////////////
			      
			    }
			}
		    }
		}
	    }	
	}
    }
    else{
      const int dim = fe.GetDim();
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
    }
  }

  void GhostStressFullGradPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
							    const FiniteElement &fe2,
							    FaceElementTransformations &Tr,
							    DenseMatrix &elmat)
  {
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();

    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    int NEproc = pmesh->GetNE();
    int elem1 = Tr.Elem1No;
    int elem2 = Tr.Elem2No;

    int elemStatus1 = elemStatus[elem1];
    int elemStatus2;
    if (Tr.Elem2No >= NEproc)
      {
        elemStatus2 = elemStatus[NEproc+par_shared_face_count];
	par_shared_face_count++;
      }
    else
      {
        elemStatus2 = elemStatus[elem2];
      }

    const int e = Tr.ElementNo;
    bool elem1_inside = (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE);
    bool elem1_cut = (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT);
    bool elem1_outside = (elemStatus1 == AnalyticalGeometricShape::SBElementType::OUTSIDE);
    
    bool elem2_inside = (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE);
    bool elem2_cut = (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT);
    bool elem2_outside = (elemStatus2 == AnalyticalGeometricShape::SBElementType::OUTSIDE);
    if ( (elem1_inside && elem2_cut) || (elem1_cut && elem2_inside) ||  (elem1_cut && elem2_cut) ) {
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim);
      Vector shape_el1(h1dofs_cnt), shape_el2(h1dofs_cnt);
      Vector gradUResDotShape_el1(h1dofs_cnt), gradUResDotShape_el2(h1dofs_cnt);
      Vector gradUResDotShape_TrialTest_el1el1(h1dofs_cnt*h1dofs_cnt), gradUResDotShape_TrialTest_el1el2(h1dofs_cnt*h1dofs_cnt);
      Vector gradUResDotShape_TrialTest_el2el1(h1dofs_cnt*h1dofs_cnt), gradUResDotShape_TrialTest_el2el2(h1dofs_cnt*h1dofs_cnt);
      DenseMatrix TrialTestContract_el1el1(h1dofs_cnt * h1dofs_cnt), TrialTestContract_el1el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix TrialTestContract_el2el1(h1dofs_cnt * h1dofs_cnt), TrialTestContract_el2el2(h1dofs_cnt * h1dofs_cnt);

      shape_el1 = 0.0;
      shape_el2 = 0.0;
    
      gradUResDotShape_el1 = 0.0;
      gradUResDotShape_el2 = 0.0;

      TrialTestContract_el1el1 = 0.0;
      TrialTestContract_el1el2 = 0.0;
      TrialTestContract_el2el1 = 0.0;
      TrialTestContract_el2el2 = 0.0;

      gradUResDotShape_TrialTest_el1el1 = 0.0;
      gradUResDotShape_TrialTest_el1el2 = 0.0;
      gradUResDotShape_TrialTest_el2el1 = 0.0;
      gradUResDotShape_TrialTest_el2el2 = 0.0;

      nor = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 10 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      DenseMatrix nodalGrad_el1;
      DenseMatrix nodalGrad_el2;
      fe.ProjectGrad(fe,Trans_el1,nodalGrad_el1);
      fe2.ProjectGrad(fe2,Trans_el2,nodalGrad_el2);

      for (int q = 0; q < nqp_face; q++)
	{
	  shape_el1 = 0.0;
	  shape_el2 = 0.0;
	  
	  gradUResDotShape_el1 = 0.0;
	  gradUResDotShape_el2 = 0.0;
	  
	  TrialTestContract_el1el1 = 0.0;
	  TrialTestContract_el1el2 = 0.0;
	  TrialTestContract_el2el1 = 0.0;
	  TrialTestContract_el2el2 = 0.0;
	  
	  gradUResDotShape_TrialTest_el1el1 = 0.0;
	  gradUResDotShape_TrialTest_el1el2 = 0.0;
	  gradUResDotShape_TrialTest_el2el1 = 0.0;
	  gradUResDotShape_TrialTest_el2el2 = 0.0;
	  
	  nor = 0.0;
      
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);

	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);
	  double Kappa = kappa->Eval(*Tr.Elem1, eip_el1);

	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);

	  Vector tN(dim);
	  tN = 0.0;
	  for (int s = 0; s < dim; s++){
	    tN(s) = nor(s) / nor_norm;
	  }

	  double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	  weighted_h = pow(weighted_h,2*nTerms-1);

	  // element 1
	  fe.CalcShape(eip_el1, shape_el1);
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_el1(s) += nodalGrad_el1(k + j * h1dofs_cnt, s) * shape_el1(k) * tN(j);
	      }
	    }
	  }
	  
	  // element 2
	  fe2.CalcShape(eip_el2, shape_el2);
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_el2(s) += nodalGrad_el2(k + j * h1dofs_cnt, s) * shape_el2(k) * tN(j);
	      }
	    }
	  }

	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int k = 0; k < h1dofs_cnt; k++){
	      gradUResDotShape_TrialTest_el1el1(s + k * h1dofs_cnt) += gradUResDotShape_el1(s) * gradUResDotShape_el1(k);
	      gradUResDotShape_TrialTest_el1el2(s + k * h1dofs_cnt) += gradUResDotShape_el1(s) * gradUResDotShape_el2(k);
	      gradUResDotShape_TrialTest_el2el1(s + k * h1dofs_cnt) += gradUResDotShape_el2(s) * gradUResDotShape_el1(k);
	      gradUResDotShape_TrialTest_el2el2(s + k * h1dofs_cnt) += gradUResDotShape_el2(s) * gradUResDotShape_el2(k); 
	    }
	  }

	  double standardFactor =  nor_norm * ip_f.weight * 2 * (1.0/std::max(3 * Kappa, 2 * Mu)) * penaltyParameter;

	  // u_{i,j}*u_{i,j} + u_{j,i}*u_{j,i}
	  DenseMatrix tmp_el1el1(h1dofs_cnt * h1dofs_cnt), tmp_el1el2(h1dofs_cnt * h1dofs_cnt);
	  DenseMatrix tmp_el2el1(h1dofs_cnt * h1dofs_cnt), tmp_el2el2(h1dofs_cnt * h1dofs_cnt);
	  tmp_el1el1 = 0.0;
	  tmp_el1el2 = 0.0;
	  tmp_el2el1 = 0.0;
	  tmp_el2el2 = 0.0;
	  for (int a = 0; a < h1dofs_cnt; a++){
	    for (int o = 0; o < h1dofs_cnt; o++){
	      for (int b = 0; b < h1dofs_cnt; b++){
		for (int r = 0; r < h1dofs_cnt; r++){
		  for (int j = 0; j < dim; j++){		
		    tmp_el1el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) += nodalGrad_el1(o + j * h1dofs_cnt, a) * nodalGrad_el1(r + j * h1dofs_cnt, b) ;
		    tmp_el1el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) += nodalGrad_el1(o + j * h1dofs_cnt, a) * nodalGrad_el2(r + j * h1dofs_cnt, b) ;
		    tmp_el2el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) += nodalGrad_el2(o + j * h1dofs_cnt, a) * nodalGrad_el1(r + j * h1dofs_cnt, b) ;
		    tmp_el2el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) += nodalGrad_el2(o + j * h1dofs_cnt, a) * nodalGrad_el2(r + j * h1dofs_cnt, b) ;
		  }
		}
	      }
	    }
	  }
	  TrialTestContract_el1el1 = tmp_el1el1;
	  TrialTestContract_el1el2 = tmp_el1el2;
	  TrialTestContract_el2el1 = tmp_el2el1;
	  TrialTestContract_el2el2 = tmp_el2el2;
	  DenseMatrix base_el1el1(h1dofs_cnt * h1dofs_cnt), base_el1el2(h1dofs_cnt * h1dofs_cnt);
	  DenseMatrix base_el2el1(h1dofs_cnt * h1dofs_cnt), base_el2el2(h1dofs_cnt * h1dofs_cnt);
	  base_el1el1 = tmp_el1el1;
	  base_el1el2 = tmp_el1el2;
	  base_el2el1 = tmp_el2el1;
	  base_el2el2 = tmp_el2el2;
	  for (int i = 2; i < nTerms ; i++){
	    
	    TrialTestContract_el1el1 = 0.0;
	    TrialTestContract_el1el2 = 0.0;  
	    TrialTestContract_el2el1 = 0.0;  
	    TrialTestContract_el2el2 = 0.0;  
	    
	    for (int a = 0; a < h1dofs_cnt; a++){
	      for (int p = 0; p < h1dofs_cnt; p++){
		for (int b = 0; b < h1dofs_cnt; b++){
		  for (int q = 0; q < h1dofs_cnt; q++){
		    for (int o = 0; o < h1dofs_cnt; o++){
		      for (int r = 0; r < h1dofs_cnt; r++){
			TrialTestContract_el1el1(a + p * h1dofs_cnt, b + q * h1dofs_cnt) += base_el1el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * tmp_el1el1(o + p * h1dofs_cnt, r + q * h1dofs_cnt);
			TrialTestContract_el1el2(a + p * h1dofs_cnt, b + q * h1dofs_cnt) += base_el1el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * tmp_el1el2(o + p * h1dofs_cnt, r + q * h1dofs_cnt);
			TrialTestContract_el2el1(a + p * h1dofs_cnt, b + q * h1dofs_cnt) += base_el2el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * tmp_el2el1(o + p * h1dofs_cnt, r + q * h1dofs_cnt);
			TrialTestContract_el2el2(a + p * h1dofs_cnt, b + q * h1dofs_cnt) += base_el2el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * tmp_el2el2(o + p * h1dofs_cnt, r + q * h1dofs_cnt);  
		      }
		    }
		  }
		}
	      }
	    }
	    tmp_el1el1 = TrialTestContract_el1el1;
	    tmp_el1el2 = TrialTestContract_el1el2;
	    tmp_el2el1 = TrialTestContract_el2el1;
	    tmp_el2el2 = TrialTestContract_el2el2;
	  }
	  
	  for (int a = 0; a < h1dofs_cnt; a++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int b = 0; b < h1dofs_cnt; b++)
		    {
		      for (int o = 0; o < h1dofs_cnt; o++)
			{
			  for (int r = 0; r < h1dofs_cnt; r++)
			    {
			      elmat(a + vd * h1dofs_cnt, b + vd * h1dofs_cnt) += 2.0 * Mu * Mu * weighted_h * TrialTestContract_el1el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el1el1(o + r * h1dofs_cnt) * standardFactor;
			      elmat(a + vd * h1dofs_cnt, b + vd * h1dofs_cnt + dim * h1dofs_cnt) -= 2.0 * Mu * Mu * weighted_h * TrialTestContract_el1el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el1el2(o + r * h1dofs_cnt) * standardFactor;
			      elmat(a + vd * h1dofs_cnt + dim * h1dofs_cnt, b + vd * h1dofs_cnt) -= 2.0 * Mu * Mu * weighted_h * TrialTestContract_el2el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el2el1(o + r * h1dofs_cnt) * standardFactor;
			      elmat(a + vd * h1dofs_cnt + dim * h1dofs_cnt, b + vd * h1dofs_cnt + dim * h1dofs_cnt) += 2.0 * Mu * Mu * weighted_h * TrialTestContract_el2el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el2el2(o + r * h1dofs_cnt) * standardFactor;
			    }
			}
		    }	 
		}
	    }
	  // end u_{i,j}*u_{i,j} + u_{j,i}*u_{j,i}

	  // begin u_{i,j} * u_{j,i}
	  tmp_el1el1 = 0.0;
	  tmp_el1el2 = 0.0;
	  tmp_el2el1 = 0.0;
	  tmp_el2el2 = 0.0;
	  for (int a = 0; a < h1dofs_cnt; a++){
	    for (int o = 0; o < h1dofs_cnt; o++){
	      tmp_el1el1(a + a * h1dofs_cnt, o + o * h1dofs_cnt) = 1.0;
	      tmp_el1el2(a + a * h1dofs_cnt, o + o * h1dofs_cnt) = 1.0;
	      tmp_el2el1(a + a * h1dofs_cnt, o + o * h1dofs_cnt) = 1.0;
	      tmp_el2el2(a + a * h1dofs_cnt, o + o * h1dofs_cnt) = 1.0;
	    }
	  }

	  TrialTestContract_el1el1 = tmp_el1el1;
	  TrialTestContract_el1el2 = tmp_el1el2;
	  TrialTestContract_el2el1 = tmp_el2el1;
	  TrialTestContract_el2el2 = tmp_el2el2;
	  if (nTerms > 2){
	    TrialTestContract_el1el1 = 0.0;
	    TrialTestContract_el1el2 = 0.0;  
	    TrialTestContract_el2el1 = 0.0;  
	    TrialTestContract_el2el2 = 0.0;  
	    tmp_el1el1 = 0.0;
	    tmp_el1el2 = 0.0;
	    tmp_el2el1 = 0.0;
	    tmp_el2el2 = 0.0;
	   
	    for (int a = 0; a < h1dofs_cnt; a++){
	      for (int o = 0; o < h1dofs_cnt; o++){
		for (int b = 0; b < h1dofs_cnt; b++){
		  for (int r = 0; r < h1dofs_cnt; r++){
		    for (int j = 0; j < dim; j++){		
		      tmp_el1el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) += nodalGrad_el1(o + j * h1dofs_cnt, a) * nodalGrad_el1(r + j * h1dofs_cnt, b) ;
		      tmp_el1el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) += nodalGrad_el1(o + j * h1dofs_cnt, a) * nodalGrad_el2(r + j * h1dofs_cnt, b) ;
		      tmp_el2el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) += nodalGrad_el2(o + j * h1dofs_cnt, a) * nodalGrad_el1(r + j * h1dofs_cnt, b) ;
		      tmp_el2el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) += nodalGrad_el2(o + j * h1dofs_cnt, a) * nodalGrad_el2(r + j * h1dofs_cnt, b) ;
		    }
		  }
		}
	      }
	    }
	    TrialTestContract_el1el1 = tmp_el1el1;
	    TrialTestContract_el1el2 = tmp_el1el2;
	    TrialTestContract_el2el1 = tmp_el2el1;
	    TrialTestContract_el2el2 = tmp_el2el2;
	    DenseMatrix base_el1el1(h1dofs_cnt * h1dofs_cnt), base_el1el2(h1dofs_cnt * h1dofs_cnt);
	    DenseMatrix base_el2el1(h1dofs_cnt * h1dofs_cnt), base_el2el2(h1dofs_cnt * h1dofs_cnt);
	    base_el1el1 = tmp_el1el1;
	    base_el1el2 = tmp_el1el2;
	    base_el2el1 = tmp_el2el1;
	    base_el2el2 = tmp_el2el2;
	    
	    for (int i = 3; i < nTerms ; i++){  
	      TrialTestContract_el1el1 = 0.0;
	      TrialTestContract_el1el2 = 0.0;  
	      TrialTestContract_el2el1 = 0.0;  
	      TrialTestContract_el2el2 = 0.0;  
	      for (int a = 0; a < h1dofs_cnt; a++){
		for (int p = 0; p < h1dofs_cnt; p++){
		  for (int b = 0; b < h1dofs_cnt; b++){
		    for (int q = 0; q < h1dofs_cnt; q++){
		      for (int o = 0; o < h1dofs_cnt; o++){
			for (int r = 0; r < h1dofs_cnt; r++){
			  TrialTestContract_el1el1(a + p * h1dofs_cnt, b + q * h1dofs_cnt) += base_el1el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * tmp_el1el1(o + p * h1dofs_cnt, r + q * h1dofs_cnt);
			  TrialTestContract_el1el2(a + p * h1dofs_cnt, b + q * h1dofs_cnt) += base_el1el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * tmp_el1el2(o + p * h1dofs_cnt, r + q * h1dofs_cnt);
			  TrialTestContract_el2el1(a + p * h1dofs_cnt, b + q * h1dofs_cnt) += base_el2el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * tmp_el2el1(o + p * h1dofs_cnt, r + q * h1dofs_cnt);
			  TrialTestContract_el2el2(a + p * h1dofs_cnt, b + q * h1dofs_cnt) += base_el2el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * tmp_el2el2(o + p * h1dofs_cnt, r + q * h1dofs_cnt);  
			}
		      }
		    }
		  }
		}
	      }
	      tmp_el1el1 = TrialTestContract_el1el1;
	      tmp_el1el2 = TrialTestContract_el1el2;
	      tmp_el2el1 = TrialTestContract_el2el1;
	      tmp_el2el2 = TrialTestContract_el2el2;
	    }
	  }
	  for (int q = 0; q < h1dofs_cnt; q++)
	    {
	      for (int vd = 0; vd < dim; vd++)
		{
		  for (int z = 0; z < h1dofs_cnt; z++)
		    {
		      for (int md = 0; md < dim; md++)
			{
			  for (int a = 0; a < h1dofs_cnt; a++)
			    {
			      for (int b = 0; b < h1dofs_cnt; b++)
				{
				  for (int o = 0; o < h1dofs_cnt; o++)
				    {
				      for (int r = 0; r < h1dofs_cnt; r++)
					{
					  elmat(q + vd * h1dofs_cnt, z + md * h1dofs_cnt) += 2.0 * Mu * Mu * weighted_h * TrialTestContract_el1el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el1el1(o + r * h1dofs_cnt) * nodalGrad_el1(a + md * h1dofs_cnt, q) * nodalGrad_el1(b + vd * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) -= 2.0 * Mu * Mu * weighted_h * TrialTestContract_el1el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el1el2(o + r * h1dofs_cnt) * nodalGrad_el1(a + md * h1dofs_cnt, q) * nodalGrad_el2(b + vd * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt) -= 2.0 * Mu * Mu * weighted_h * TrialTestContract_el2el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el2el1(o + r * h1dofs_cnt) * nodalGrad_el2(a + md * h1dofs_cnt, q) * nodalGrad_el1(b + vd * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) += 2.0 * Mu * Mu * weighted_h * TrialTestContract_el2el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el2el2(o + r * h1dofs_cnt) * nodalGrad_el2(a + md * h1dofs_cnt, q) * nodalGrad_el2(b + vd * h1dofs_cnt, z) * standardFactor;

					  elmat(q + vd * h1dofs_cnt, z + md * h1dofs_cnt) += 4.0 * (Kappa - (2.0/3.0) * Mu) * Mu * weighted_h * TrialTestContract_el1el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el1el1(o + r * h1dofs_cnt) * nodalGrad_el1(a + vd * h1dofs_cnt, q) * nodalGrad_el1(b + md * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) -= 4.0 * (Kappa - (2.0/3.0) * Mu) * Mu * weighted_h * TrialTestContract_el1el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el1el2(o + r * h1dofs_cnt) * nodalGrad_el1(a + vd * h1dofs_cnt, q) * nodalGrad_el2(b + md * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt) -= 4.0 * (Kappa - (2.0/3.0) * Mu) * Mu * weighted_h * TrialTestContract_el2el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el2el1(o + r * h1dofs_cnt) * nodalGrad_el2(a + vd * h1dofs_cnt, q) * nodalGrad_el1(b + md * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) += 4.0 * (Kappa - (2.0/3.0) * Mu) * Mu * weighted_h * TrialTestContract_el2el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el2el2(o + r * h1dofs_cnt) * nodalGrad_el2(a + vd * h1dofs_cnt, q) * nodalGrad_el2(b + md * h1dofs_cnt, z) * standardFactor;

					  elmat(q + vd * h1dofs_cnt, z + md * h1dofs_cnt) += dim * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * weighted_h * TrialTestContract_el1el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el1el1(o + r * h1dofs_cnt) * nodalGrad_el1(a + vd * h1dofs_cnt, q) * nodalGrad_el1(b + md * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) -= dim * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * weighted_h * TrialTestContract_el1el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el1el2(o + r * h1dofs_cnt) * nodalGrad_el1(a + vd * h1dofs_cnt, q) * nodalGrad_el2(b + md * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt) -= dim * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * weighted_h * TrialTestContract_el2el1(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el2el1(o + r * h1dofs_cnt) * nodalGrad_el2(a + vd * h1dofs_cnt, q) * nodalGrad_el1(b + md * h1dofs_cnt, z) * standardFactor;
					  elmat(q + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) += dim * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * weighted_h * TrialTestContract_el2el2(a + o * h1dofs_cnt, b + r * h1dofs_cnt) * gradUResDotShape_TrialTest_el2el2(o + r * h1dofs_cnt) * nodalGrad_el2(a + vd * h1dofs_cnt, q) * nodalGrad_el2(b + md * h1dofs_cnt, z) * standardFactor; 				  	  
					}
				    }
				}	 
			    }
			}
		    }
		}
	    }
	  // end u_{i,j} * u_{j,i}

	  
	}
    }
    else{
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
    }
  }

}
