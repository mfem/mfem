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
  
  void GhostStrainPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
							    const FiniteElement &fe2,
							    FaceElementTransformations &Tr,
							    DenseMatrix &elmat)
  {
    double penaltyParameter_scaled = std::pow(penaltyParameter,1.0);

    Array<int> &elemStatus = analyticalSurface->GetElement_Status();
    const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
    const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
    
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
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
      Vector nor(dim);
      DenseMatrix dshape_el1(dofs_cnt,dim), dshape_ps_el1(dofs_cnt,dim), adjJ_el1(dim), dshape_el2(dofs_cnt,dim), dshape_ps_el2(dofs_cnt,dim), adjJ_el2(dim);
      Vector shape_el1(dofs_cnt), gradURes_el1(dofs_cnt), shape_el2(dofs_cnt), gradURes_el2(dofs_cnt);
 
      nor = 0.0;

      shape_el1 = 0.0;
      gradURes_el1 = 0.0;
      dshape_el1 = 0.0;
      dshape_ps_el1 = 0.0;
      adjJ_el1 = 0.0;

      shape_el2 = 0.0;
      gradURes_el2 = 0.0;
      dshape_el2 = 0.0;
      dshape_ps_el2 = 0.0;
      adjJ_el2 = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 5 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      
      for (int q = 0; q < nqp_face; q++)
	{
	  nor = 0.0;
	  shape_el1 = 0.0;
	  gradURes_el1 = 0.0;
	  dshape_el1 = 0.0;
	  dshape_ps_el1 = 0.0;
	  adjJ_el1 = 0.0;
	  
	  shape_el2 = 0.0;
	  gradURes_el2 = 0.0;
	  dshape_el2 = 0.0;
	  dshape_ps_el2 = 0.0;
	  adjJ_el2 = 0.0;
    	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);

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

	  //
	  fe.CalcShape(eip_el1, shape_el1);
	  fe.CalcDShape(eip_el1, dshape_el1);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ_el1);
	  Mult(dshape_el1, adjJ_el1, dshape_ps_el1);
	  dshape_ps_el1.Mult(tN,gradURes_el1);
	  
	  fe2.CalcShape(eip_el1, shape_el2);
	  fe2.CalcDShape(eip_el2, dshape_el2);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ_el2);
	  Mult(dshape_el2, adjJ_el2, dshape_ps_el2);
	  dshape_ps_el2.Mult(tN,gradURes_el2);
	  //
	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double diff_volFrac = std::abs(volumeFraction_el1 - volumeFraction_el2);
	  
	  double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	  weighted_h = pow(weighted_h,1.0);
	  //  std::cout << " penPar " << penaltyParameter << std::endl;
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      elmat(i + vd * dofs_cnt, j + vd * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * gradURes_el1(j) * (1.0/2.0) * gradURes_el1(i) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem1->Weight());
		      elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + vd * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * gradURes_el1(j) * (1.0/2.0) * gradURes_el2(i) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem2->Weight());
		      elmat(i + vd * dofs_cnt, j + vd * dofs_cnt + dofs_cnt * dim) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * gradURes_el2(j) * (1.0/2.0) * gradURes_el1(i) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem2->Weight());
		      elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + vd * dofs_cnt + dim * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * gradURes_el2(j) * (1.0/2.0) * gradURes_el2(i) * ip_f.weight * nor_norm / (Tr.Elem2->Weight() * Tr.Elem2->Weight());
	      
		      for (int md = 0; md < dim; md++){
			elmat(i + vd * dofs_cnt, j + md * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * gradURes_el1(i) * (1.0/2.0) * dshape_ps_el1(j,vd) * tN(md) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem1->Weight());
			elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * gradURes_el2(i) * (1.0/2.0) * dshape_ps_el1(j,vd) * tN(md) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem2->Weight());
			elmat(i + vd * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * gradURes_el1(i) * (1.0/2.0) * dshape_ps_el2(j,vd) * tN(md) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem2->Weight());
			elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * gradURes_el2(i) * (1.0/2.0) * dshape_ps_el2(j,vd) * tN(md) * ip_f.weight * nor_norm / (Tr.Elem2->Weight() * Tr.Elem2->Weight());
		
			elmat(i + vd * dofs_cnt, j + md * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * dshape_ps_el1(i,md) * (1.0/2.0) * gradURes_el1(j) * tN(vd) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem1->Weight());
			elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * dshape_ps_el2(i,md) * (1.0/2.0) * gradURes_el1(j) * tN(vd) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem2->Weight());
			elmat(i + vd * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * dshape_ps_el1(i,md) * (1.0/2.0) * gradURes_el2(j) * tN(vd) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem2->Weight());
			elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * dshape_ps_el2(i,md) * (1.0/2.0) * gradURes_el2(j) * tN(vd) * ip_f.weight * nor_norm / (Tr.Elem2->Weight() * Tr.Elem2->Weight());
			
			for (int r = 0; r < dim; r++){
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * dshape_ps_el1(j,r) * tN(md) * (1.0/2.0) * dshape_ps_el1(i,r) * tN(vd) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem1->Weight());
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * dshape_ps_el1(j,r) * tN(md) * (1.0/2.0) * dshape_ps_el2(i,r) * tN(vd) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem2->Weight());
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * dshape_ps_el2(j,r) * tN(md) * (1.0/2.0) * dshape_ps_el1(i,r) * tN(vd) * ip_f.weight * nor_norm / (Tr.Elem1->Weight() * Tr.Elem2->Weight());
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * dshape_ps_el2(j,r) * tN(md) * (1.0/2.0) * dshape_ps_el2(i,r) * tN(vd) * ip_f.weight * nor_norm / (Tr.Elem2->Weight() * Tr.Elem2->Weight());
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

  void GhostDivStrainGradQPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
								const FiniteElement &trial_fe2,
								const FiniteElement &test_fe1,
								const FiniteElement &test_fe2,
								FaceElementTransformations &Tr,
								DenseMatrix &elmat)
  {
    double penaltyParameter_scaled = std::pow(penaltyParameter,1.0);
    
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();
    const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
    const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
    
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
      const int dim = trial_fe1.GetDim();
      const int trial_dofs_cnt = trial_fe1.GetDof();
      const int test_dofs_cnt = test_fe1.GetDof();
     
      elmat.SetSize(2*test_dofs_cnt, 2*trial_dofs_cnt*dim);
      elmat = 0.0;
      Vector nor(dim);
      DenseMatrix dshape_el1(test_dofs_cnt,dim), dshape_ps_el1(test_dofs_cnt,dim), adjJ_el1(dim), dshape_el2(test_dofs_cnt,dim), dshape_ps_el2(test_dofs_cnt,dim), adjJ_el2(dim);
      Vector gradURes_el1(test_dofs_cnt), gradURes_el2(test_dofs_cnt);
      Vector shape_el1(trial_dofs_cnt), shape_el2(trial_dofs_cnt);
      DenseMatrix gradUResDotShape_el1(trial_dofs_cnt,dim), gradUResDotShape_el2(trial_dofs_cnt,dim), hessian_el1(trial_dofs_cnt,dim * dim), hessian_el2(trial_dofs_cnt,dim * dim);
      
      nor = 0.0;
      shape_el1 = 0.0;
      gradURes_el1 = 0.0;
      dshape_el1 = 0.0;
      dshape_ps_el1 = 0.0;
      adjJ_el1 = 0.0;
      gradUResDotShape_el1 = 0.0;
      hessian_el1 = 0.0;
      
      shape_el2 = 0.0;
      gradURes_el2 = 0.0;
      dshape_el2 = 0.0;
      dshape_ps_el2 = 0.0;
      adjJ_el2 = 0.0;
      gradUResDotShape_el2 = 0.0;
      hessian_el2 = 0.0;
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 5 * max(trial_fe1.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      
      DenseMatrix nodalGrad_el1;
      DenseMatrix nodalGrad_el2;
      trial_fe1.ProjectGrad(trial_fe1,Trans_el1,nodalGrad_el1);
      trial_fe2.ProjectGrad(trial_fe2,Trans_el2,nodalGrad_el2);

      for (int q = 0; q < nqp_face; q++)
	{
	   nor = 0.0;
	   shape_el1 = 0.0;
	   gradURes_el1 = 0.0;
	   dshape_el1 = 0.0;
	   dshape_ps_el1 = 0.0;
	   adjJ_el1 = 0.0;  
	   gradUResDotShape_el1 = 0.0;
	   hessian_el1 = 0.0;
      
	   shape_el2 = 0.0;
	   gradURes_el2 = 0.0;
	   dshape_el2 = 0.0;
	   dshape_ps_el2 = 0.0;
	   adjJ_el2 = 0.0;
	   gradUResDotShape_el2 = 0.0;
	   hessian_el2 = 0.0;
      
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);

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

	  test_fe1.CalcDShape(eip_el1, dshape_el1);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ_el1);
	  Mult(dshape_el1, adjJ_el1, dshape_ps_el1);
	  dshape_ps_el1.Mult(nor,gradURes_el1);
	  
	  test_fe2.CalcDShape(eip_el2, dshape_el2);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ_el2);
	  Mult(dshape_el2, adjJ_el2, dshape_ps_el2);
	  dshape_ps_el2.Mult(nor,gradURes_el2);

	  // element 1
	  trial_fe1.CalcShape(eip_el1, shape_el1);
	  for (int s = 0; s < trial_dofs_cnt; s++){
	    for (int j = 0; j < dim; j++){
	      for (int k = 0; k < trial_dofs_cnt; k++){
		gradUResDotShape_el1(s,j) += nodalGrad_el1(k + j * trial_dofs_cnt, s) * shape_el1(k);
	      }
	    }
	  }
	  
	  for (int a = 0; a < trial_dofs_cnt; a++){
	    for (int j = 0; j < dim; j++){
	      for (int k = 0; k < dim; k++){
		for (int b = 0; b < trial_dofs_cnt; b++){
		  hessian_el1(a, j + dim * k ) += nodalGrad_el1(b + j * trial_dofs_cnt,a) * gradUResDotShape_el1(b,k); 
		}
	      }
	    }
	  }
	  //
	  // element 2
	  trial_fe2.CalcShape(eip_el2, shape_el2);
	  for (int s = 0; s < trial_dofs_cnt; s++){
	    for (int j = 0; j < dim; j++){
	      for (int k = 0; k < trial_dofs_cnt; k++){
		gradUResDotShape_el2(s,j) += nodalGrad_el2(k + j * trial_dofs_cnt, s) * shape_el2(k);
	      }
	    }
	  }
	  
	  for (int a = 0; a < trial_dofs_cnt; a++){
	    for (int j = 0; j < dim; j++){
	      for (int k = 0; k < dim; k++){
		for (int b = 0; b < trial_dofs_cnt; b++){
		  hessian_el2(a, j + dim * k ) += nodalGrad_el2(b + j * trial_dofs_cnt,a) * gradUResDotShape_el2(b,k); 
		}
	      }
	    }
	  }
	  // 

	  double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	  weighted_h = pow(weighted_h,2.0*nTerms+1.0);
	  //  std::cout << " penPar " << penaltyParameter << std::endl;
	  for (int i = 0; i < test_dofs_cnt; i++)
	    {
	      for (int j = 0; j < trial_dofs_cnt; j++)
		{
		  for (int vd = 0; vd < dim; vd++)
		    {
		      for (int md = 0; md < dim; md++)
			{ 
			  elmat(i, j + vd * trial_dofs_cnt) -= 0.5 * penaltyParameter_scaled * weighted_h * gradURes_el1(i) * hessian_el1(j,md + md * dim) * ip_f.weight * tN(vd) / Tr.Elem1->Weight();
			  elmat(i, j + vd * trial_dofs_cnt) -= 0.5 * penaltyParameter_scaled * weighted_h * gradURes_el1(i) * hessian_el1(j,md + vd * dim) * ip_f.weight * tN(md) / Tr.Elem1->Weight();
			  
			  elmat(i, j + vd * trial_dofs_cnt + dim * trial_dofs_cnt) += 0.5 * penaltyParameter_scaled * weighted_h * gradURes_el1(i) * hessian_el2(j,md + md * dim) * ip_f.weight * tN(vd) / Tr.Elem1->Weight();
			  elmat(i, j + vd * trial_dofs_cnt + dim * trial_dofs_cnt) += 0.5 * penaltyParameter_scaled * weighted_h * gradURes_el1(i) * hessian_el2(j,md + vd * dim) * ip_f.weight * tN(md) / Tr.Elem1->Weight();

			  elmat(i + test_dofs_cnt, j + vd * trial_dofs_cnt) += 0.5 * penaltyParameter_scaled * weighted_h * gradURes_el2(i) * hessian_el1(j,md + md * dim) * ip_f.weight * tN(vd) / Tr.Elem2->Weight();
			  elmat(i + test_dofs_cnt, j + vd * trial_dofs_cnt) += 0.5 * penaltyParameter_scaled * weighted_h * gradURes_el2(i) * hessian_el1(j,md + vd * dim) * ip_f.weight * tN(md) / Tr.Elem2->Weight();

			  elmat(i + test_dofs_cnt, j + vd * trial_dofs_cnt + dim * trial_dofs_cnt) -= 0.5 * penaltyParameter_scaled * weighted_h * gradURes_el2(i) * hessian_el2(j,md + md * dim) * ip_f.weight * tN(vd) / Tr.Elem2->Weight();
			  elmat(i + test_dofs_cnt, j + vd * trial_dofs_cnt + dim * trial_dofs_cnt) -= 0.5 * penaltyParameter_scaled * weighted_h * gradURes_el2(i) * hessian_el2(j,md + vd * dim) * ip_f.weight * tN(md) / Tr.Elem2->Weight();
			}
		    }
		}
	    }
	}
    }
    else{
      const int dim = trial_fe1.GetDim();
      const int trial_dofs_cnt = trial_fe1.GetDof();
      const int test_dofs_cnt = test_fe1.GetDof();
      elmat.SetSize(2*test_dofs_cnt, 2*trial_dofs_cnt*dim);
      elmat = 0.0;
    }
  }

  void GhostPenaltyFullGradIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
							  const FiniteElement &fe2,
							  FaceElementTransformations &Tr,
							  DenseMatrix &elmat)
  {
    double penaltyParameter_scaled = std::pow(penaltyParameter,1.0);
    
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();
    const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
    const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
    
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
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt);
      elmat = 0.0;
      Vector nor(dim);
      Vector shape_el1(dofs_cnt), shape_el2(dofs_cnt), gradUResDotShape_el1(dofs_cnt), gradUResDotShape_el2(dofs_cnt);
      DenseMatrix TrialTestContract_el1el1(dofs_cnt * dofs_cnt), TrialTestContract_el1el2(dofs_cnt * dofs_cnt);
      DenseMatrix TrialTestContract_el2el1(dofs_cnt * dofs_cnt), TrialTestContract_el2el2(dofs_cnt * dofs_cnt);
      Vector gradUResDotShape_TrialTest_el1el1(dofs_cnt*dofs_cnt), gradUResDotShape_TrialTest_el1el2(dofs_cnt*dofs_cnt);
      Vector gradUResDotShape_TrialTest_el2el1(dofs_cnt*dofs_cnt), gradUResDotShape_TrialTest_el2el2(dofs_cnt*dofs_cnt);
      
      nor = 0.0;
      shape_el1 = 0.0;
      gradUResDotShape_el1 = 0.0;
      TrialTestContract_el1el1 = 0.0;
      TrialTestContract_el1el2 = 0.0;
      TrialTestContract_el2el1 = 0.0;
      TrialTestContract_el2el2 = 0.0;
      
      shape_el2 = 0.0;
      gradUResDotShape_el2 = 0.0;
      gradUResDotShape_TrialTest_el1el1 = 0.0;
      gradUResDotShape_TrialTest_el1el2 = 0.0;
      gradUResDotShape_TrialTest_el2el1 = 0.0;
      gradUResDotShape_TrialTest_el2el2 = 0.0;
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 5 * max(fe.GetOrder(), 1);
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
	  TrialTestContract_el1el1 = 0.0;
	  TrialTestContract_el1el2 = 0.0;
	  TrialTestContract_el2el1 = 0.0;
	  TrialTestContract_el2el2 = 0.0;
	  
	  shape_el2 = 0.0;
	  gradUResDotShape_el2 = 0.0;
	  gradUResDotShape_TrialTest_el1el1 = 0.0;
	  gradUResDotShape_TrialTest_el1el2 = 0.0;
	  gradUResDotShape_TrialTest_el2el1 = 0.0;
	  gradUResDotShape_TrialTest_el2el2 = 0.0;
    	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);

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
	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double diff_volFrac = std::abs(volumeFraction_el1 - volumeFraction_el2);
	  
	  // element 1
	  fe.CalcShape(eip_el1, shape_el1);
	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_el1(s) += nodalGrad_el1(k + j * dofs_cnt, s) * shape_el1(k) * tN(j);
	      }
	    }
	  }
	  
	  // element 2
	  fe2.CalcShape(eip_el2, shape_el2);
	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_el2(s) += nodalGrad_el2(k + j * dofs_cnt, s) * shape_el2(k) * tN(j);
	      }
	    }
	  }

	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      gradUResDotShape_TrialTest_el1el1(s + k * dofs_cnt) += gradUResDotShape_el1(s) * gradUResDotShape_el1(k);
	      gradUResDotShape_TrialTest_el1el2(s + k * dofs_cnt) += gradUResDotShape_el1(s) * gradUResDotShape_el2(k);
	      gradUResDotShape_TrialTest_el2el1(s + k * dofs_cnt) += gradUResDotShape_el2(s) * gradUResDotShape_el1(k);
	      gradUResDotShape_TrialTest_el2el2(s + k * dofs_cnt) += gradUResDotShape_el2(s) * gradUResDotShape_el2(k); 
	    }
	  }

	  DenseMatrix tmp_el1el1(dofs_cnt * dofs_cnt), tmp_el1el2(dofs_cnt * dofs_cnt);
	  DenseMatrix tmp_el2el1(dofs_cnt * dofs_cnt), tmp_el2el2(dofs_cnt * dofs_cnt);
	  tmp_el1el1 = 0.0;
	  tmp_el1el2 = 0.0;
	  tmp_el2el1 = 0.0;
	  tmp_el2el2 = 0.0;
	  for (int a = 0; a < dofs_cnt; a++){
	    for (int o = 0; o < dofs_cnt; o++){
	      tmp_el1el1(a + a * dofs_cnt, o + o * dofs_cnt) = 1.0;
	      tmp_el1el2(a + a * dofs_cnt, o + o * dofs_cnt) = 1.0;
	      tmp_el2el1(a + a * dofs_cnt, o + o * dofs_cnt) = 1.0;
	      tmp_el2el2(a + a * dofs_cnt, o + o * dofs_cnt) = 1.0;
	    }
	  }

	  TrialTestContract_el1el1 = tmp_el1el1;
	  TrialTestContract_el1el2 = tmp_el1el2;
	  TrialTestContract_el2el1 = tmp_el2el1;
	  TrialTestContract_el2el2 = tmp_el2el2;
	  if (nTerms > 1){
	    tmp_el1el1 = 0.0;
	    tmp_el1el2 = 0.0;
	    tmp_el2el1 = 0.0;
	    tmp_el2el2 = 0.0;
	
	    for (int a = 0; a < dofs_cnt; a++){
	      for (int o = 0; o < dofs_cnt; o++){
		for (int b = 0; b < dofs_cnt; b++){
		  for (int r = 0; r < dofs_cnt; r++){
		    for (int j = 0; j < dim; j++){		
		      tmp_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el1(o + j * dofs_cnt, a) * nodalGrad_el1(r + j * dofs_cnt, b) ;
		      tmp_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el1(o + j * dofs_cnt, a) * nodalGrad_el2(r + j * dofs_cnt, b) ;
		      tmp_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el2(o + j * dofs_cnt, a) * nodalGrad_el1(r + j * dofs_cnt, b) ;
		      tmp_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el2(o + j * dofs_cnt, a) * nodalGrad_el2(r + j * dofs_cnt, b) ;
		    }
		  }
		}
	      }
	    }
	    TrialTestContract_el1el1 = tmp_el1el1;
	    TrialTestContract_el1el2 = tmp_el1el2;
	    TrialTestContract_el2el1 = tmp_el2el1;
	    TrialTestContract_el2el2 = tmp_el2el2;
	    DenseMatrix base_el1el1(dofs_cnt * dofs_cnt), base_el1el2(dofs_cnt * dofs_cnt);
	    DenseMatrix base_el2el1(dofs_cnt * dofs_cnt), base_el2el2(dofs_cnt * dofs_cnt);
	    base_el1el1 = tmp_el1el1;
	    base_el1el2 = tmp_el1el2;
	    base_el2el1 = tmp_el2el1;
	    base_el2el2 = tmp_el2el2;
	    for (int i = 2; i < nTerms ; i++){	     
	      TrialTestContract_el1el1 = 0.0;
	      TrialTestContract_el1el2 = 0.0;  
	      TrialTestContract_el2el1 = 0.0;  
	      TrialTestContract_el2el2 = 0.0;  
	      for (int a = 0; a < dofs_cnt; a++){
		for (int p = 0; p < dofs_cnt; p++){
		  for (int b = 0; b < dofs_cnt; b++){
		    for (int q = 0; q < dofs_cnt; q++){
		      for (int o = 0; o < dofs_cnt; o++){
			for (int r = 0; r < dofs_cnt; r++){
			  TrialTestContract_el1el1(a + p * dofs_cnt, b + q * dofs_cnt) += base_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el1(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el1el2(a + p * dofs_cnt, b + q * dofs_cnt) += base_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el2(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el2el1(a + p * dofs_cnt, b + q * dofs_cnt) += base_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el1(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el2el2(a + p * dofs_cnt, b + q * dofs_cnt) += base_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el2(o + p * dofs_cnt, r + q * dofs_cnt);  
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
	  
	  double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	  weighted_h = pow(weighted_h,(2.0*nTerms+1.0)*(1.0));
	  for (int a = 0; a < dofs_cnt; a++)
	    {
	      for (int b = 0; b < dofs_cnt; b++)
		{
		  for (int o = 0; o < dofs_cnt; o++)
		    {
		      for (int r = 0; r < dofs_cnt; r++)
			{
			  elmat(a, b) += 2.0 * penaltyParameter_scaled * weighted_h * TrialTestContract_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el1el1(o + r * dofs_cnt) * ip_f.weight * nor_norm * (1.0/(2.0*Mu));
			  elmat(a, b + dofs_cnt) -= 2.0 * penaltyParameter_scaled * weighted_h * TrialTestContract_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el1el2(o + r * dofs_cnt) * ip_f.weight * nor_norm * (1.0/(2.0*Mu));
			  elmat(a + dofs_cnt, b) -= 2.0 * penaltyParameter_scaled * weighted_h * TrialTestContract_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el2el1(o + r * dofs_cnt) * ip_f.weight * nor_norm * (1.0/(2.0*Mu));
			  elmat(a + dofs_cnt, b + dofs_cnt) += 2.0 * penaltyParameter_scaled * weighted_h * TrialTestContract_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el2el2(o + r * dofs_cnt) * ip_f.weight * nor_norm * (1.0/(2.0*Mu));
			}
		    }
		}
	    }
	}
    }
    else{
      const int dim = fe.GetDim();
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt);
      elmat = 0.0;
    }
  }

  void GhostStrainFullGradPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
								const FiniteElement &fe2,
								FaceElementTransformations &Tr,
								DenseMatrix &elmat)
  {
    double penaltyParameter_scaled = std::pow(penaltyParameter,1.0);
    
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();
    const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
    const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
    
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
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;

      Vector nor(dim);
      Vector shape_el1(dofs_cnt), shape_el2(dofs_cnt), gradUResDotShape_el1(dofs_cnt), gradUResDotShape_el2(dofs_cnt);
      DenseMatrix TrialTestContract_el1el1(dofs_cnt * dofs_cnt), TrialTestContract_el1el2(dofs_cnt * dofs_cnt);
      DenseMatrix TrialTestContract_el2el1(dofs_cnt * dofs_cnt), TrialTestContract_el2el2(dofs_cnt * dofs_cnt);
      Vector gradUResDotShape_TrialTest_el1el1(dofs_cnt*dofs_cnt), gradUResDotShape_TrialTest_el1el2(dofs_cnt*dofs_cnt);
      Vector gradUResDotShape_TrialTest_el2el1(dofs_cnt*dofs_cnt), gradUResDotShape_TrialTest_el2el2(dofs_cnt*dofs_cnt);
      
      nor = 0.0;
      shape_el1 = 0.0;
      gradUResDotShape_el1 = 0.0;
      TrialTestContract_el1el1 = 0.0;
      TrialTestContract_el1el2 = 0.0;
      TrialTestContract_el2el1 = 0.0;
      TrialTestContract_el2el2 = 0.0;
      
      shape_el2 = 0.0;
      gradUResDotShape_el2 = 0.0;
      gradUResDotShape_TrialTest_el1el1 = 0.0;
      gradUResDotShape_TrialTest_el1el2 = 0.0;
      gradUResDotShape_TrialTest_el2el1 = 0.0;
      gradUResDotShape_TrialTest_el2el2 = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 5 * max(fe.GetOrder(), 1);
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
	  TrialTestContract_el1el1 = 0.0;
	  TrialTestContract_el1el2 = 0.0;
	  TrialTestContract_el2el1 = 0.0;
	  TrialTestContract_el2el2 = 0.0;
	  
	  shape_el2 = 0.0;
	  gradUResDotShape_el2 = 0.0;
	  gradUResDotShape_TrialTest_el1el1 = 0.0;
	  gradUResDotShape_TrialTest_el1el2 = 0.0;
	  gradUResDotShape_TrialTest_el2el1 = 0.0;
	  gradUResDotShape_TrialTest_el2el2 = 0.0;
    	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);

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
	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double diff_volFrac = std::abs(volumeFraction_el1 - volumeFraction_el2);
	  
	  double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	  weighted_h = pow(weighted_h,(2*(nTerms-1)+1)*(1.0));

	  // element 1
	  fe.CalcShape(eip_el1, shape_el1);
	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_el1(s) += nodalGrad_el1(k + j * dofs_cnt, s) * shape_el1(k) * tN(j);
	      }
	    }
	  }
	  
	  // element 2
	  fe2.CalcShape(eip_el2, shape_el2);
	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_el2(s) += nodalGrad_el2(k + j * dofs_cnt, s) * shape_el2(k) * tN(j);
	      }
	    }
	  }

	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      gradUResDotShape_TrialTest_el1el1(s + k * dofs_cnt) += gradUResDotShape_el1(s) * gradUResDotShape_el1(k);
	      gradUResDotShape_TrialTest_el1el2(s + k * dofs_cnt) += gradUResDotShape_el1(s) * gradUResDotShape_el2(k);
	      gradUResDotShape_TrialTest_el2el1(s + k * dofs_cnt) += gradUResDotShape_el2(s) * gradUResDotShape_el1(k);
	      gradUResDotShape_TrialTest_el2el2(s + k * dofs_cnt) += gradUResDotShape_el2(s) * gradUResDotShape_el2(k); 
	    }
	  }

	  // u_{i,j}*u_{i,j} + u_{j,i}*u_{j,i}
	  DenseMatrix tmp_el1el1(dofs_cnt * dofs_cnt), tmp_el1el2(dofs_cnt * dofs_cnt);
	  DenseMatrix tmp_el2el1(dofs_cnt * dofs_cnt), tmp_el2el2(dofs_cnt * dofs_cnt);
	  tmp_el1el1 = 0.0;
	  tmp_el1el2 = 0.0;
	  tmp_el2el1 = 0.0;
	  tmp_el2el2 = 0.0;
	  for (int a = 0; a < dofs_cnt; a++){
	    for (int o = 0; o < dofs_cnt; o++){
	      for (int b = 0; b < dofs_cnt; b++){
		for (int r = 0; r < dofs_cnt; r++){
		  for (int j = 0; j < dim; j++){		
		    tmp_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el1(o + j * dofs_cnt, a) * nodalGrad_el1(r + j * dofs_cnt, b) ;
		    tmp_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el1(o + j * dofs_cnt, a) * nodalGrad_el2(r + j * dofs_cnt, b) ;
		    tmp_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el2(o + j * dofs_cnt, a) * nodalGrad_el1(r + j * dofs_cnt, b) ;
		    tmp_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el2(o + j * dofs_cnt, a) * nodalGrad_el2(r + j * dofs_cnt, b) ;
		  }
		}
	      }
	    }
	  }
	  TrialTestContract_el1el1 = tmp_el1el1;
	  TrialTestContract_el1el2 = tmp_el1el2;
	  TrialTestContract_el2el1 = tmp_el2el1;
	  TrialTestContract_el2el2 = tmp_el2el2;
	  DenseMatrix base_el1el1(dofs_cnt * dofs_cnt), base_el1el2(dofs_cnt * dofs_cnt);
	  DenseMatrix base_el2el1(dofs_cnt * dofs_cnt), base_el2el2(dofs_cnt * dofs_cnt);
	  base_el1el1 = tmp_el1el1;
	  base_el1el2 = tmp_el1el2;
	  base_el2el1 = tmp_el2el1;
	  base_el2el2 = tmp_el2el2;
	  for (int i = 2; i < nTerms ; i++){
	    
	    TrialTestContract_el1el1 = 0.0;
	    TrialTestContract_el1el2 = 0.0;  
	    TrialTestContract_el2el1 = 0.0;  
	    TrialTestContract_el2el2 = 0.0;  
	    
	    for (int a = 0; a < dofs_cnt; a++){
	      for (int p = 0; p < dofs_cnt; p++){
		for (int b = 0; b < dofs_cnt; b++){
		  for (int q = 0; q < dofs_cnt; q++){
		    for (int o = 0; o < dofs_cnt; o++){
		      for (int r = 0; r < dofs_cnt; r++){
			TrialTestContract_el1el1(a + p * dofs_cnt, b + q * dofs_cnt) += base_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el1(o + p * dofs_cnt, r + q * dofs_cnt);
			TrialTestContract_el1el2(a + p * dofs_cnt, b + q * dofs_cnt) += base_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el2(o + p * dofs_cnt, r + q * dofs_cnt);
			TrialTestContract_el2el1(a + p * dofs_cnt, b + q * dofs_cnt) += base_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el1(o + p * dofs_cnt, r + q * dofs_cnt);
			TrialTestContract_el2el2(a + p * dofs_cnt, b + q * dofs_cnt) += base_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el2(o + p * dofs_cnt, r + q * dofs_cnt);  
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
	  // TrialTestContract_el1el1.Print(std::cout,1);
	  for (int a = 0; a < dofs_cnt; a++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int b = 0; b < dofs_cnt; b++)
		    {
		      for (int o = 0; o < dofs_cnt; o++)
			{
			  for (int r = 0; r < dofs_cnt; r++)
			    {
			      elmat(a + vd * dofs_cnt, b + vd * dofs_cnt) += 4.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * TrialTestContract_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el1el1(o + r * dofs_cnt) * (1.0/2.0) * ip_f.weight * nor_norm;
			      elmat(a + vd * dofs_cnt, b + vd * dofs_cnt + dim * dofs_cnt) -= 4.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * TrialTestContract_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el1el2(o + r * dofs_cnt) * (1.0/2.0) * ip_f.weight * nor_norm;
			      elmat(a + vd * dofs_cnt + dim * dofs_cnt, b + vd * dofs_cnt) -= 4.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * TrialTestContract_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el2el1(o + r * dofs_cnt) * (1.0/2.0) * ip_f.weight * nor_norm;
			      elmat(a + vd * dofs_cnt + dim * dofs_cnt, b + vd * dofs_cnt + dim * dofs_cnt) += 4.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * TrialTestContract_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el2el2(o + r * dofs_cnt) * (1.0/2.0) * ip_f.weight * nor_norm;
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
	  for (int a = 0; a < dofs_cnt; a++){
	    for (int o = 0; o < dofs_cnt; o++){
	      tmp_el1el1(a + a * dofs_cnt, o + o * dofs_cnt) = 1.0;
	      tmp_el1el2(a + a * dofs_cnt, o + o * dofs_cnt) = 1.0;
	      tmp_el2el1(a + a * dofs_cnt, o + o * dofs_cnt) = 1.0;
	      tmp_el2el2(a + a * dofs_cnt, o + o * dofs_cnt) = 1.0;
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
	   
	    for (int a = 0; a < dofs_cnt; a++){
	      for (int o = 0; o < dofs_cnt; o++){
		for (int b = 0; b < dofs_cnt; b++){
		  for (int r = 0; r < dofs_cnt; r++){
		    for (int j = 0; j < dim; j++){		
		      tmp_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el1(o + j * dofs_cnt, a) * nodalGrad_el1(r + j * dofs_cnt, b) ;
		      tmp_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el1(o + j * dofs_cnt, a) * nodalGrad_el2(r + j * dofs_cnt, b) ;
		      tmp_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el2(o + j * dofs_cnt, a) * nodalGrad_el1(r + j * dofs_cnt, b) ;
		      tmp_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el2(o + j * dofs_cnt, a) * nodalGrad_el2(r + j * dofs_cnt, b) ;
		    }
		  }
		}
	      }
	    }
	    TrialTestContract_el1el1 = tmp_el1el1;
	    TrialTestContract_el1el2 = tmp_el1el2;
	    TrialTestContract_el2el1 = tmp_el2el1;
	    TrialTestContract_el2el2 = tmp_el2el2;
	    DenseMatrix base_el1el1(dofs_cnt * dofs_cnt), base_el1el2(dofs_cnt * dofs_cnt);
	    DenseMatrix base_el2el1(dofs_cnt * dofs_cnt), base_el2el2(dofs_cnt * dofs_cnt);
	    base_el1el1 = tmp_el1el1;
	    base_el1el2 = tmp_el1el2;
	    base_el2el1 = tmp_el2el1;
	    base_el2el2 = tmp_el2el2;
	    
	    for (int i = 3; i < nTerms ; i++){  
	      TrialTestContract_el1el1 = 0.0;
	      TrialTestContract_el1el2 = 0.0;  
	      TrialTestContract_el2el1 = 0.0;  
	      TrialTestContract_el2el2 = 0.0;  
	      for (int a = 0; a < dofs_cnt; a++){
		for (int p = 0; p < dofs_cnt; p++){
		  for (int b = 0; b < dofs_cnt; b++){
		    for (int q = 0; q < dofs_cnt; q++){
		      for (int o = 0; o < dofs_cnt; o++){
			for (int r = 0; r < dofs_cnt; r++){
			  TrialTestContract_el1el1(a + p * dofs_cnt, b + q * dofs_cnt) += base_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el1(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el1el2(a + p * dofs_cnt, b + q * dofs_cnt) += base_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el2(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el2el1(a + p * dofs_cnt, b + q * dofs_cnt) += base_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el1(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el2el2(a + p * dofs_cnt, b + q * dofs_cnt) += base_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el2(o + p * dofs_cnt, r + q * dofs_cnt);  
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
	  for (int q = 0; q < dofs_cnt; q++)
	    {
	      for (int vd = 0; vd < dim; vd++)
		{
		  for (int z = 0; z < dofs_cnt; z++)
		    {
		      for (int md = 0; md < dim; md++)
			{
			  for (int a = 0; a < dofs_cnt; a++)
			    {
			      for (int b = 0; b < dofs_cnt; b++)
				{
				  for (int o = 0; o < dofs_cnt; o++)
				    {
				      for (int r = 0; r < dofs_cnt; r++)
					{
					  elmat(q + vd * dofs_cnt, z + md * dofs_cnt) += 4.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * (1.0/2.0) * ip_f.weight * nor_norm * TrialTestContract_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el1el1(o + r * dofs_cnt) * nodalGrad_el1(a + md * dofs_cnt, q) * nodalGrad_el1(b + vd * dofs_cnt, z);
					  elmat(q + vd * dofs_cnt, z + md * dofs_cnt + dim * dofs_cnt) -= 4.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * (1.0/2.0) * ip_f.weight * nor_norm * TrialTestContract_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el1el2(o + r * dofs_cnt) * nodalGrad_el1(a + md * dofs_cnt, q) * nodalGrad_el2(b + vd * dofs_cnt, z);
					  elmat(q + vd * dofs_cnt + dim * dofs_cnt, z + md * dofs_cnt) -= 4.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * (1.0/2.0) * ip_f.weight * nor_norm * TrialTestContract_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el2el1(o + r * dofs_cnt) * nodalGrad_el2(a + md * dofs_cnt, q) * nodalGrad_el1(b + vd * dofs_cnt, z);
					  elmat(q + vd * dofs_cnt + dim * dofs_cnt, z + md * dofs_cnt + dim * dofs_cnt) += 4.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * (1.0/2.0) * (1.0/2.0) * ip_f.weight * nor_norm * TrialTestContract_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el2el2(o + r * dofs_cnt) * nodalGrad_el2(a + md * dofs_cnt, q) * nodalGrad_el2(b + vd * dofs_cnt, z); 					  
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
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
    }
  }

    void GhostDivStrainFullGradQPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
								const FiniteElement &trial_fe2,
								const FiniteElement &test_fe1,
								const FiniteElement &test_fe2,
								FaceElementTransformations &Tr,
								DenseMatrix &elmat)
  {
    double penaltyParameter_scaled = std::pow(penaltyParameter,1.0);
    
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();
    const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
    const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
    
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
      const int dim = trial_fe1.GetDim();
      const int trial_dofs_cnt = trial_fe1.GetDof();
      const int test_dofs_cnt = test_fe1.GetDof();
     
      elmat.SetSize(2*test_dofs_cnt, 2*trial_dofs_cnt*dim);
      elmat = 0.0;
      Vector nor(dim);
      Vector shape_trial_el1(trial_dofs_cnt), shape_trial_el2(trial_dofs_cnt), shape_test_el1(test_dofs_cnt), shape_test_el2(test_dofs_cnt), gradUResDotShape_trial_el1(trial_dofs_cnt), gradUResDotShape_trial_el2(trial_dofs_cnt), gradUResDotShape_test_el1(test_dofs_cnt), gradUResDotShape_test_el2(test_dofs_cnt);

      Vector gradUResDotShape_TrialTest_el1el1(test_dofs_cnt*trial_dofs_cnt), gradUResDotShape_TrialTest_el1el2(test_dofs_cnt*trial_dofs_cnt);
      Vector gradUResDotShape_TrialTest_el2el1(test_dofs_cnt*trial_dofs_cnt), gradUResDotShape_TrialTest_el2el2(test_dofs_cnt*trial_dofs_cnt);
      DenseMatrix TrialTestContract_el1el1(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt), TrialTestContract_el1el2(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt);
      DenseMatrix TrialTestContract_el2el1(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt), TrialTestContract_el2el2(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt);
    
      nor = 0.0;
      shape_trial_el1 = 0.0;
      shape_trial_el2 = 0.0;
      shape_test_el1 = 0.0;
      shape_test_el2 = 0.0;
      TrialTestContract_el1el1 = 0.0;
      TrialTestContract_el1el2 = 0.0;
      TrialTestContract_el2el1 = 0.0;
      TrialTestContract_el2el2 = 0.0;

      gradUResDotShape_trial_el1 = 0.0;
      gradUResDotShape_trial_el2 = 0.0;
      gradUResDotShape_test_el1 = 0.0;
      gradUResDotShape_test_el2 = 0.0;

      gradUResDotShape_TrialTest_el1el1 = 0.0;
      gradUResDotShape_TrialTest_el1el2 = 0.0;
      gradUResDotShape_TrialTest_el2el1 = 0.0;
      gradUResDotShape_TrialTest_el2el2 = 0.0;
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 5 * max(trial_fe1.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      
      DenseMatrix nodalGrad_trial_el1;
      DenseMatrix nodalGrad_trial_el2;
      DenseMatrix nodalGrad_test_el1;
      DenseMatrix nodalGrad_test_el2;
      trial_fe1.ProjectGrad(trial_fe1,Trans_el1,nodalGrad_trial_el1);
      trial_fe2.ProjectGrad(trial_fe2,Trans_el2,nodalGrad_trial_el2);
      test_fe1.ProjectGrad(test_fe1,Trans_el1,nodalGrad_test_el1);
      test_fe2.ProjectGrad(test_fe2,Trans_el2,nodalGrad_test_el2);
      for (int q = 0; q < nqp_face; q++)
	{
	  nor = 0.0;
	  shape_trial_el1 = 0.0;
	  shape_trial_el2 = 0.0;
	  shape_test_el1 = 0.0;
	  shape_test_el2 = 0.0;
	  TrialTestContract_el1el1 = 0.0;
	  TrialTestContract_el1el2 = 0.0;
	  TrialTestContract_el2el1 = 0.0;
	  TrialTestContract_el2el2 = 0.0;
	  
	  gradUResDotShape_trial_el1 = 0.0;
	  gradUResDotShape_trial_el2 = 0.0;
	  gradUResDotShape_test_el1 = 0.0;
	  gradUResDotShape_test_el2 = 0.0;
	  gradUResDotShape_TrialTest_el1el1 = 0.0;
	  gradUResDotShape_TrialTest_el1el2 = 0.0;
	  gradUResDotShape_TrialTest_el2el1 = 0.0;
	  gradUResDotShape_TrialTest_el2el2 = 0.0;
	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);

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
	  weighted_h = pow(weighted_h,2.0*nTerms+1.0);

	  // element 1
	  trial_fe1.CalcShape(eip_el1, shape_trial_el1);
	  for (int s = 0; s < trial_dofs_cnt; s++){
	    for (int k = 0; k < trial_dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_trial_el1(s) += nodalGrad_trial_el1(k + j * trial_dofs_cnt, s) * shape_trial_el1(k) * tN(j);
	      }
	    }
	  }

	  test_fe1.CalcShape(eip_el1, shape_test_el1);
	  for (int s = 0; s < test_dofs_cnt; s++){
	    for (int k = 0; k < test_dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_test_el1(s) += nodalGrad_test_el1(k + j * test_dofs_cnt, s) * shape_test_el1(k) * tN(j);
	      }
	    }
	  }
	  ///////
	  // element 2
	  trial_fe2.CalcShape(eip_el2, shape_trial_el2);
	  for (int s = 0; s < trial_dofs_cnt; s++){
	    for (int k = 0; k < trial_dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_trial_el2(s) += nodalGrad_trial_el2(k + j * trial_dofs_cnt, s) * shape_trial_el2(k) * tN(j);
	      }
	    }
	  }

	  test_fe2.CalcShape(eip_el2, shape_test_el2);
	  for (int s = 0; s < test_dofs_cnt; s++){
	    for (int k = 0; k < test_dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_test_el2(s) += nodalGrad_test_el2(k + j * test_dofs_cnt, s) * shape_test_el2(k) * tN(j);
	      }
	    }
	  }
	  //////
	  for (int s = 0; s < test_dofs_cnt; s++){
	    for (int k = 0; k < trial_dofs_cnt; k++){
	      gradUResDotShape_TrialTest_el1el1(s + k * test_dofs_cnt) += gradUResDotShape_test_el1(s) * gradUResDotShape_trial_el1(k);
	      gradUResDotShape_TrialTest_el1el2(s + k * test_dofs_cnt) += gradUResDotShape_test_el1(s) * gradUResDotShape_trial_el2(k);
	      gradUResDotShape_TrialTest_el2el1(s + k * test_dofs_cnt) += gradUResDotShape_test_el2(s) * gradUResDotShape_trial_el1(k);
	      gradUResDotShape_TrialTest_el2el2(s + k * test_dofs_cnt) += gradUResDotShape_test_el2(s) * gradUResDotShape_trial_el2(k); 
	    }
	  }
	  ////

	  // ui,jj n_i, q_s n_s	  
	  // base
	  DenseMatrix base_el1el1(trial_dofs_cnt);
	  DenseMatrix base_el2el2(trial_dofs_cnt);
	  base_el1el1 = 0.0;
	  base_el2el2 = 0.0;
	  for (int a = 0; a < trial_dofs_cnt; a++){
	    for (int o = 0; o < trial_dofs_cnt; o++){
	      for (int b = 0; b < trial_dofs_cnt; b++){
		for (int j = 0; j < dim; j++){		
		  base_el1el1(a, o) += nodalGrad_trial_el1(b + j * trial_dofs_cnt, a) * nodalGrad_trial_el1(o + j * trial_dofs_cnt, b) ;
		  base_el2el2(a, o) += nodalGrad_trial_el2(b + j * trial_dofs_cnt, a) * nodalGrad_trial_el2(o + j * trial_dofs_cnt, b) ;
		}
	      }
	    }
	  }
	  // uj,ij n_i, q_s n_s
	  // base
	  DenseMatrix base_p2_el1el1(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt), base_p2_el1el2(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt);
	  DenseMatrix base_p2_el2el1(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt), base_p2_el2el2(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt);
	  base_p2_el1el1 = 0.0;
	  base_p2_el1el2 = 0.0;
	  base_p2_el2el1 = 0.0;
	  base_p2_el2el2 = 0.0;
	  for (int b = 0; b < test_dofs_cnt; b++){
	    for (int r = 0; r < test_dofs_cnt; r++){
	      for (int a = 0; a < trial_dofs_cnt; a++){
		for (int z = 0; z < trial_dofs_cnt; z++){
		  for (int j = 0; j < dim; j++){		
		    base_p2_el1el1(b + r * test_dofs_cnt, a + z * trial_dofs_cnt) += nodalGrad_test_el1(r + j * test_dofs_cnt, b) * nodalGrad_trial_el1(z + j * trial_dofs_cnt, a);
		    base_p2_el1el2(b + r * test_dofs_cnt, a + z * trial_dofs_cnt) += nodalGrad_test_el1(r + j * test_dofs_cnt, b) * nodalGrad_trial_el2(z + j * trial_dofs_cnt, a);
		    base_p2_el2el1(b + r * test_dofs_cnt, a + z * trial_dofs_cnt) += nodalGrad_test_el2(r + j * test_dofs_cnt, b) * nodalGrad_trial_el1(z + j * trial_dofs_cnt, a);
		    base_p2_el2el2(b + r * test_dofs_cnt, a + z * trial_dofs_cnt) += nodalGrad_test_el2(r + j * test_dofs_cnt, b) * nodalGrad_trial_el2(z + j * trial_dofs_cnt, a);
		  }
		}
	      }
	    }
	  }  
	  
	  if (nTerms == 2){	   
	    for (int b = 0; b < test_dofs_cnt; b++)
	      {
		for (int a = 0; a < trial_dofs_cnt; a++)
		  {
		    for (int s = 0; s < dim; s++)
		      {
			for (int r = 0; r < trial_dofs_cnt; r++)
			  {
			    for (int w = 0; w < test_dofs_cnt; w++)
			      {		
				elmat(b, a + s * trial_dofs_cnt) -= penaltyParameter_scaled * weighted_h * nodalGrad_test_el1(w + s * test_dofs_cnt, b) * gradUResDotShape_TrialTest_el1el1(w + r * test_dofs_cnt) * base_el1el1(a, r) * ip_f.weight * nor_norm * 0.5; 
				
				elmat(b, a + s * trial_dofs_cnt + dim * trial_dofs_cnt) += penaltyParameter_scaled * weighted_h * nodalGrad_test_el1(w + s * test_dofs_cnt, b) * gradUResDotShape_TrialTest_el1el2(w + r * test_dofs_cnt) * base_el2el2(a, r) * ip_f.weight * nor_norm * 0.5;

				elmat(b + test_dofs_cnt, a + s * trial_dofs_cnt) += penaltyParameter_scaled * weighted_h * nodalGrad_test_el2(w + s * test_dofs_cnt, b) * gradUResDotShape_TrialTest_el2el1(w + r * test_dofs_cnt) * base_el1el1(a, r) * ip_f.weight * nor_norm * 0.5; 
				
				elmat(b + test_dofs_cnt, a + s * trial_dofs_cnt + dim * trial_dofs_cnt) -= penaltyParameter_scaled * weighted_h * nodalGrad_test_el2(w + s * test_dofs_cnt, b) * gradUResDotShape_TrialTest_el2el2(w + r * test_dofs_cnt) * base_el2el2(a, r) * ip_f.weight * nor_norm * 0.5;
				for (int z = 0; z < trial_dofs_cnt; z++)
				  {
				    elmat(b, a + s * trial_dofs_cnt) -= penaltyParameter_scaled * weighted_h * nodalGrad_trial_el1(r + s * test_dofs_cnt, z) * gradUResDotShape_TrialTest_el1el1(w + r * test_dofs_cnt) * base_p2_el1el1(b + w * test_dofs_cnt, a + z * trial_dofs_cnt) * ip_f.weight * nor_norm * 0.5; 

				    elmat(b, a + s * trial_dofs_cnt + dim * trial_dofs_cnt) += penaltyParameter_scaled * weighted_h * nodalGrad_trial_el2(r + s * test_dofs_cnt, z) * gradUResDotShape_TrialTest_el1el2(w + r * test_dofs_cnt) * base_p2_el1el2(b + w * test_dofs_cnt, a + z * trial_dofs_cnt) * ip_f.weight * nor_norm * 0.5; 
				    elmat(b + test_dofs_cnt, a + s * trial_dofs_cnt) += penaltyParameter_scaled * weighted_h * nodalGrad_trial_el1(r + s * test_dofs_cnt, z) * gradUResDotShape_TrialTest_el2el1(w + r * test_dofs_cnt) * base_p2_el2el1(b + w * test_dofs_cnt, a + z * trial_dofs_cnt) * ip_f.weight * nor_norm * 0.5; 
				    elmat(b + test_dofs_cnt, a + s * trial_dofs_cnt + dim * trial_dofs_cnt) -= penaltyParameter_scaled * weighted_h * nodalGrad_trial_el2(r + s * test_dofs_cnt, z) * gradUResDotShape_TrialTest_el2el2(w + r * test_dofs_cnt) * base_p2_el2el2(b + w * test_dofs_cnt, a + z * trial_dofs_cnt) * ip_f.weight * nor_norm * 0.5; 
				  }
			      }
			  }
		      }
		  }
	      }
	  }
	  else{
	    /*  DenseMatrix tmp_el1el1(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt), tmp_el1el2(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt);
	    DenseMatrix tmp_el2el1(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt), tmp_el2el2(test_dofs_cnt * test_dofs_cnt, trial_dofs_cnt * trial_dofs_cnt);
	    tmp_el1el1 = 0.0;
	    tmp_el1el2 = 0.0;
	    tmp_el2el1 = 0.0;
	    tmp_el2el2 = 0.0;
	    
	    for (int r = 0; r < test_dofs_cnt; r++){
	      for (int m = 0; m < test_dofs_cnt; m++){
		for (int t = 0; t < trial_dofs_cnt; t++){
		  for (int w = 0; w < trial_dofs_cnt; w++){
		    for (int j = 0; j < dim; j++){		
		      tmp_el1el1(r + m * test_dofs_cnt, t + w * trial_dofs_cnt) += nodalGrad_test_el1(m + j * test_dofs_cnt, r) * nodalGrad_trial_el1(w + j * trial_dofs_cnt, t);
		      tmp_el1el2(r + m * test_dofs_cnt, t + w * trial_dofs_cnt) += nodalGrad_test_el1(m + j * test_dofs_cnt, r) * nodalGrad_trial_el2(w + j * trial_dofs_cnt, t);
		      tmp_el2el1(r + m * test_dofs_cnt, t + w * trial_dofs_cnt) += nodalGrad_test_el2(m + j * test_dofs_cnt, r) * nodalGrad_trial_el1(w + j * trial_dofs_cnt, t);
		      tmp_el2el2(r + m * test_dofs_cnt, t + w * trial_dofs_cnt) += nodalGrad_test_el2(m + j * test_dofs_cnt, r) * nodalGrad_trial_el2(w + j * trial_dofs_cnt, t);
		    }
		  }
		}
	      }
	    }
	    TrialTestContract_el1el1 = tmp_el1el1;
	    TrialTestContract_el1el2 = tmp_el1el2;
	    TrialTestContract_el2el1 = tmp_el2el1;
	    TrialTestContract_el2el2 = tmp_el2el2;
	    
	    for (int i = 3; i < nTerms ; i++){
	      TrialTestContract_el1el1 = 0.0;
	      TrialTestContract_el1el2 = 0.0;  
	      TrialTestContract_el2el1 = 0.0;  
	      TrialTestContract_el2el2 = 0.0;  
	      
	      for (int a = 0; a < dofs_cnt; a++){
		for (int p = 0; p < dofs_cnt; p++){
		  for (int b = 0; b < dofs_cnt; b++){
		    for (int q = 0; q < dofs_cnt; q++){
		      for (int o = 0; o < dofs_cnt; o++){
			for (int r = 0; r < dofs_cnt; r++){
			  TrialTestContract_el1el1(a + p * dofs_cnt, b + q * dofs_cnt) += tmp_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el1(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el1el2(a + p * dofs_cnt, b + q * dofs_cnt) += tmp_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el2(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el2el1(a + p * dofs_cnt, b + q * dofs_cnt) += tmp_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el1(o + p * dofs_cnt, r + q * dofs_cnt);
			  TrialTestContract_el2el2(a + p * dofs_cnt, b + q * dofs_cnt) += tmp_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el2(o + p * dofs_cnt, r + q * dofs_cnt);  
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
	    
	    for (int b = 0; b < test_dofs_cnt; b++)
	      {
		for (int a = 0; a < trial_dofs_cnt; a++)
		  {
		    for (int s = 0; s < dim; s++)
		      {
			for (int r = 0; r < test_dofs_cnt; r++)
			  {
			    for (int m = 0; m < test_dofs_cnt; m++)
			      {
				for (int w = 0; w < trial_dofs_cnt; w++)
				  {		
				    for (int t = 0; t < trial_dofs_cnt; t++)
				      {
					elmat(b, a + s * trial_dofs_cnt) += penaltyParameter_scaled * weighted_h * nodalGrad_test_el1(r + s * test_dofs_cnt, b) * gradUResDotShape_TrialTest_el1el1(m + t * test_dofs_cnt) * base_el1el1(a, w) * TrialTestContract_el1el1(r + m * test_dofs_cnt , w + t * trial_dofs_cnt);
					elmat(b, a + s * trial_dofs_cnt + dim * trial_dofs_cnt ) += penaltyParameter_scaled * weighted_h * nodalGrad_test_el1(r + s * test_dofs_cnt, b) * gradUResDotShape_TrialTest_el1el2(m + t * test_dofs_cnt) * base_el2el2(a, w) * TrialTestContract_el1el2(r + m * test_dofs_cnt , w + t * trial_dofs_cnt);
					elmat(b + test_dofs_cnt, a + s * trial_dofs_cnt) += penaltyParameter_scaled * weighted_h * nodalGrad_test_el2(r + s * test_dofs_cnt, b) * gradUResDotShape_TrialTest_el2el1(m + t * test_dofs_cnt) * base_el1el1(a, w) * TrialTestContract_el2el1(r + m * test_dofs_cnt , w + t * trial_dofs_cnt);
					elmat(b + test_dofs_cnt, a + s * trial_dofs_cnt + dim * trial_dofs_cnt) += penaltyParameter_scaled * weighted_h * nodalGrad_test_el2(r + s * test_dofs_cnt, b) * gradUResDotShape_TrialTest_el2el2(m + t * test_dofs_cnt) * base_el2el2(a, w) * TrialTestContract_el2el2(r + m * test_dofs_cnt , w + t * trial_dofs_cnt);
				      }
				  }
			      }
			  }
		      }
		  }
	      }*/
	  }
	}
    }
    else{
      const int dim = trial_fe1.GetDim();
      const int trial_dofs_cnt = trial_fe1.GetDof();
      const int test_dofs_cnt = test_fe1.GetDof();
      elmat.SetSize(2*test_dofs_cnt, 2*trial_dofs_cnt*dim);
      elmat = 0.0;
    }
  }

  void GhostFullGradVelocityPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
								const FiniteElement &fe2,
								FaceElementTransformations &Tr,
								DenseMatrix &elmat)
  {
    double penaltyParameter_scaled = std::pow(penaltyParameter,1.0);
    
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();
    const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
    const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
    
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
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;

      Vector nor(dim);
      Vector shape_el1(dofs_cnt), shape_el2(dofs_cnt), gradUResDotShape_el1(dofs_cnt), gradUResDotShape_el2(dofs_cnt);
      DenseMatrix TrialTestContract_el1el1(dofs_cnt * dofs_cnt), TrialTestContract_el1el2(dofs_cnt * dofs_cnt);
      DenseMatrix TrialTestContract_el2el1(dofs_cnt * dofs_cnt), TrialTestContract_el2el2(dofs_cnt * dofs_cnt);
      Vector gradUResDotShape_TrialTest_el1el1(dofs_cnt*dofs_cnt), gradUResDotShape_TrialTest_el1el2(dofs_cnt*dofs_cnt);
      Vector gradUResDotShape_TrialTest_el2el1(dofs_cnt*dofs_cnt), gradUResDotShape_TrialTest_el2el2(dofs_cnt*dofs_cnt);
      
      nor = 0.0;
      shape_el1 = 0.0;
      gradUResDotShape_el1 = 0.0;
      TrialTestContract_el1el1 = 0.0;
      TrialTestContract_el1el2 = 0.0;
      TrialTestContract_el2el1 = 0.0;
      TrialTestContract_el2el2 = 0.0;
      
      shape_el2 = 0.0;
      gradUResDotShape_el2 = 0.0;
      gradUResDotShape_TrialTest_el1el1 = 0.0;
      gradUResDotShape_TrialTest_el1el2 = 0.0;
      gradUResDotShape_TrialTest_el2el1 = 0.0;
      gradUResDotShape_TrialTest_el2el2 = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 5 * max(fe.GetOrder(), 1);
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
	  TrialTestContract_el1el1 = 0.0;
	  TrialTestContract_el1el2 = 0.0;
	  TrialTestContract_el2el1 = 0.0;
	  TrialTestContract_el2el2 = 0.0;
	  
	  shape_el2 = 0.0;
	  gradUResDotShape_el2 = 0.0;
	  gradUResDotShape_TrialTest_el1el1 = 0.0;
	  gradUResDotShape_TrialTest_el1el2 = 0.0;
	  gradUResDotShape_TrialTest_el2el1 = 0.0;
	  gradUResDotShape_TrialTest_el2el2 = 0.0;
    	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);

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
	  weighted_h = pow(weighted_h,2*(nTerms-1)+1);

	  // element 1
	  fe.CalcShape(eip_el1, shape_el1);
	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_el1(s) += nodalGrad_el1(k + j * dofs_cnt, s) * shape_el1(k) * tN(j);
	      }
	    }
	  }
	  
	  // element 2
	  fe2.CalcShape(eip_el2, shape_el2);
	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){
		gradUResDotShape_el2(s) += nodalGrad_el2(k + j * dofs_cnt, s) * shape_el2(k) * tN(j);
	      }
	    }
	  }

	  for (int s = 0; s < dofs_cnt; s++){
	    for (int k = 0; k < dofs_cnt; k++){
	      gradUResDotShape_TrialTest_el1el1(s + k * dofs_cnt) += gradUResDotShape_el1(s) * gradUResDotShape_el1(k);
	      gradUResDotShape_TrialTest_el1el2(s + k * dofs_cnt) += gradUResDotShape_el1(s) * gradUResDotShape_el2(k);
	      gradUResDotShape_TrialTest_el2el1(s + k * dofs_cnt) += gradUResDotShape_el2(s) * gradUResDotShape_el1(k);
	      gradUResDotShape_TrialTest_el2el2(s + k * dofs_cnt) += gradUResDotShape_el2(s) * gradUResDotShape_el2(k); 
	    }
	  }

	  // u_{i,j}*u_{i,j} + u_{j,i}*u_{j,i}
	  DenseMatrix tmp_el1el1(dofs_cnt * dofs_cnt), tmp_el1el2(dofs_cnt * dofs_cnt);
	  DenseMatrix tmp_el2el1(dofs_cnt * dofs_cnt), tmp_el2el2(dofs_cnt * dofs_cnt);
	  tmp_el1el1 = 0.0;
	  tmp_el1el2 = 0.0;
	  tmp_el2el1 = 0.0;
	  tmp_el2el2 = 0.0;
	  for (int a = 0; a < dofs_cnt; a++){
	    for (int o = 0; o < dofs_cnt; o++){
	      for (int b = 0; b < dofs_cnt; b++){
		for (int r = 0; r < dofs_cnt; r++){
		  for (int j = 0; j < dim; j++){		
		    tmp_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el1(o + j * dofs_cnt, a) * nodalGrad_el1(r + j * dofs_cnt, b) ;
		    tmp_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el1(o + j * dofs_cnt, a) * nodalGrad_el2(r + j * dofs_cnt, b) ;
		    tmp_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el2(o + j * dofs_cnt, a) * nodalGrad_el1(r + j * dofs_cnt, b) ;
		    tmp_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) += nodalGrad_el2(o + j * dofs_cnt, a) * nodalGrad_el2(r + j * dofs_cnt, b) ;
		  }
		}
	      }
	    }
	  }
	  TrialTestContract_el1el1 = tmp_el1el1;
	  TrialTestContract_el1el2 = tmp_el1el2;
	  TrialTestContract_el2el1 = tmp_el2el1;
	  TrialTestContract_el2el2 = tmp_el2el2;
	  for (int i = 2; i < nTerms ; i++){
	    TrialTestContract_el1el1 = 0.0;
	    TrialTestContract_el1el2 = 0.0;  
	    TrialTestContract_el2el1 = 0.0;  
	    TrialTestContract_el2el2 = 0.0;  
	    
	    for (int a = 0; a < dofs_cnt; a++){
	      for (int p = 0; p < dofs_cnt; p++){
		for (int b = 0; b < dofs_cnt; b++){
		  for (int q = 0; q < dofs_cnt; q++){
		    for (int o = 0; o < dofs_cnt; o++){
		      for (int r = 0; r < dofs_cnt; r++){
			TrialTestContract_el1el1(a + p * dofs_cnt, b + q * dofs_cnt) += tmp_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el1(o + p * dofs_cnt, r + q * dofs_cnt);
			TrialTestContract_el1el2(a + p * dofs_cnt, b + q * dofs_cnt) += tmp_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el1el2(o + p * dofs_cnt, r + q * dofs_cnt);
			TrialTestContract_el2el1(a + p * dofs_cnt, b + q * dofs_cnt) += tmp_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el1(o + p * dofs_cnt, r + q * dofs_cnt);
			TrialTestContract_el2el2(a + p * dofs_cnt, b + q * dofs_cnt) += tmp_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * tmp_el2el2(o + p * dofs_cnt, r + q * dofs_cnt);  
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
	  for (int a = 0; a < dofs_cnt; a++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int b = 0; b < dofs_cnt; b++)
		    {
		      for (int o = 0; o < dofs_cnt; o++)
			{
			  for (int r = 0; r < dofs_cnt; r++)
			    {
			      elmat(a + vd * dofs_cnt, b + vd * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * TrialTestContract_el1el1(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el1el1(o + r * dofs_cnt) * ip_f.weight * nor_norm;
			      elmat(a + vd * dofs_cnt, b + vd * dofs_cnt + dim * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * TrialTestContract_el1el2(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el1el2(o + r * dofs_cnt) * ip_f.weight * nor_norm;
			      elmat(a + vd * dofs_cnt + dim * dofs_cnt, b + vd * dofs_cnt) -= 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * TrialTestContract_el2el1(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el2el1(o + r * dofs_cnt) * ip_f.weight * nor_norm;
			      elmat(a + vd * dofs_cnt + dim * dofs_cnt, b + vd * dofs_cnt + dim * dofs_cnt) += 2.0 * penaltyParameter_scaled * 2 * Mu * weighted_h * TrialTestContract_el2el2(a + o * dofs_cnt, b + r * dofs_cnt) * gradUResDotShape_TrialTest_el2el2(o + r * dofs_cnt) * ip_f.weight * nor_norm;
			    }
			}
		    }	 
		}
	    }
	  // end u_{i,j}*u_{i,j} + u_{j,i}*u_{j,i}
	}
    }
    else{
      const int dim = fe.GetDim();
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
    }
  }

}
