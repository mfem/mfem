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

  /* void GhostStressFullGradPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
								const FiniteElement &fe2,
								FaceElementTransformations &Tr,
								DenseMatrix &elmat)
  {
    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    int NEproc = pmesh->GetNE();
    if (Tr.Attribute == 77){   
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim), tN(dim);
      Vector shape_el1(h1dofs_cnt), shape_el2(h1dofs_cnt), normalGradU_el1(h1dofs_cnt), normalGradU_el2(h1dofs_cnt);
      Vector gradUResDotShape_TrialTest_el1el1(h1dofs_cnt*h1dofs_cnt), gradUResDotShape_TrialTest_el1el2(h1dofs_cnt*h1dofs_cnt);
      Vector gradUResDotShape_TrialTest_el2el1(h1dofs_cnt*h1dofs_cnt), gradUResDotShape_TrialTest_el2el2(h1dofs_cnt*h1dofs_cnt);
      DenseMatrix gradUResDotShape_el1(h1dofs_cnt,dim), gradUResDotShape_el2(h1dofs_cnt,dim), TrialTestContract_el1el1(h1dofs_cnt * h1dofs_cnt), TrialTestContract_el1el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix TrialTestContract_el2el1(h1dofs_cnt * h1dofs_cnt), TrialTestContract_el2el2(h1dofs_cnt * h1dofs_cnt), gradUTrialDotGradUTest_el1el1(h1dofs_cnt), gradUTrialDotGradUTest_el2el1(h1dofs_cnt), gradUTrialDotGradUTest_el1el2(h1dofs_cnt), gradUTrialDotGradUTest_el2el2(h1dofs_cnt);
      DenseMatrix base_el1el1(h1dofs_cnt * h1dofs_cnt), base_el1el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix base_el2el1(h1dofs_cnt * h1dofs_cnt), base_el2el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix tmp_el1el1(h1dofs_cnt * h1dofs_cnt), tmp_el1el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix tmp_el2el1(h1dofs_cnt * h1dofs_cnt), tmp_el2el2(h1dofs_cnt * h1dofs_cnt);
      Vector lumped_el1el1(h1dofs_cnt * h1dofs_cnt), lumped_el1el2(h1dofs_cnt * h1dofs_cnt);
      Vector lumped_el2el1(h1dofs_cnt * h1dofs_cnt), lumped_el2el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix nodalGradContraction_el1el1(h1dofs_cnt * h1dofs_cnt * dim * dim, h1dofs_cnt * h1dofs_cnt), nodalGradContraction_el1el2(h1dofs_cnt * h1dofs_cnt * dim * dim, h1dofs_cnt * h1dofs_cnt);
      DenseMatrix nodalGradContraction_el2el1(h1dofs_cnt * h1dofs_cnt * dim * dim, h1dofs_cnt * h1dofs_cnt), nodalGradContraction_el2el2(h1dofs_cnt * h1dofs_cnt * dim * dim, h1dofs_cnt * h1dofs_cnt);
      Vector assembledContraction_el1el1(h1dofs_cnt * h1dofs_cnt * dim * dim), assembledContraction_el1el2(h1dofs_cnt * h1dofs_cnt * dim * dim);
      Vector assembledContraction_el2el1(h1dofs_cnt * h1dofs_cnt * dim * dim), assembledContraction_el2el2(h1dofs_cnt * h1dofs_cnt * dim * dim);
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 2 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);	  
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      DenseMatrix nodalGrad_el1;
      DenseMatrix nodalGrad_el2;
      fe.ProjectGrad(fe,Trans_el1,nodalGrad_el1);
      fe2.ProjectGrad(fe2,Trans_el2,nodalGrad_el2);
      tmp_el1el1 = 0.0;
      tmp_el1el2 = 0.0;
      tmp_el2el1 = 0.0;
      tmp_el2el2 = 0.0;
      
      nodalGradContraction_el1el1 = 0.0;
      nodalGradContraction_el1el2 = 0.0;
      nodalGradContraction_el2el1 = 0.0;
      nodalGradContraction_el2el2 = 0.0;
      
      for (int a = 0; a < h1dofs_cnt; a++){
	for (int o = 0; o < h1dofs_cnt; o++){
	  for (int b = 0; b < h1dofs_cnt; b++){
	    for (int r = 0; r < h1dofs_cnt; r++){
	      for (int j = 0; j < dim; j++){		
		tmp_el1el1(a + b * h1dofs_cnt, o + r * h1dofs_cnt) += nodalGrad_el1(o + j * h1dofs_cnt, a) * nodalGrad_el1(r + j * h1dofs_cnt, b) ;
		tmp_el1el2(a + b * h1dofs_cnt, o + r * h1dofs_cnt) += nodalGrad_el1(o + j * h1dofs_cnt, a) * nodalGrad_el2(r + j * h1dofs_cnt, b) ;
		tmp_el2el1(a + b * h1dofs_cnt, o + r * h1dofs_cnt) += nodalGrad_el2(o + j * h1dofs_cnt, a) * nodalGrad_el1(r + j * h1dofs_cnt, b) ;
		tmp_el2el2(a + b * h1dofs_cnt, o + r * h1dofs_cnt) += nodalGrad_el2(o + j * h1dofs_cnt, a) * nodalGrad_el2(r + j * h1dofs_cnt, b) ;
		for (int k = 0; k < dim; k++){		
		  nodalGradContraction_el1el1(a + j * h1dofs_cnt + o * dim * h1dofs_cnt + k * h1dofs_cnt * h1dofs_cnt * dim, b + r * h1dofs_cnt) = nodalGrad_el1(b + j * h1dofs_cnt, a) * nodalGrad_el1(r + k * h1dofs_cnt, o);
		  nodalGradContraction_el1el2(a + j * h1dofs_cnt + o * dim * h1dofs_cnt + k * h1dofs_cnt * h1dofs_cnt * dim, b + r * h1dofs_cnt) = nodalGrad_el1(b + j * h1dofs_cnt, a) * nodalGrad_el2(r + k * h1dofs_cnt, o);
		  nodalGradContraction_el2el1(a + j * h1dofs_cnt + o * dim * h1dofs_cnt + k * h1dofs_cnt * h1dofs_cnt * dim, b + r * h1dofs_cnt) = nodalGrad_el2(b + j * h1dofs_cnt, a) * nodalGrad_el1(r + k * h1dofs_cnt, o);
		  nodalGradContraction_el2el2(a + j * h1dofs_cnt + o * dim * h1dofs_cnt + k * h1dofs_cnt * h1dofs_cnt * dim, b + r * h1dofs_cnt) = nodalGrad_el2(b + j * h1dofs_cnt, a) * nodalGrad_el2(r + k * h1dofs_cnt, o);
		}
	      }
	    }
	  }
	}
      }
      
      base_el1el1 = tmp_el1el1;
      base_el1el2 = tmp_el1el2;
      base_el2el1 = tmp_el2el1;
      base_el2el2 = tmp_el2el2;
      
      for (int q = 0; q < nqp_face; q++)
	{
	  penaltyParameter = dupPenaltyParameter;
	  shape_el1 = 0.0;
	  shape_el2 = 0.0;
	  
	  gradUResDotShape_el1 = 0.0;
	  gradUResDotShape_el2 = 0.0;
	  
	  nor = 0.0;
	  tN = 0.0;

	  TrialTestContract_el1el1 = 0.0;
	  TrialTestContract_el1el2 = 0.0;  
	  TrialTestContract_el2el1 = 0.0;  
	  TrialTestContract_el2el2 = 0.0;  

	  normalGradU_el1 = 0.0;
	  normalGradU_el2 = 0.0;
    	  
	  gradUResDotShape_TrialTest_el1el1 = 0.0;
	  gradUResDotShape_TrialTest_el1el2 = 0.0;
	  gradUResDotShape_TrialTest_el2el1 = 0.0;
	  gradUResDotShape_TrialTest_el2el2 = 0.0;	    
	 
	  gradUTrialDotGradUTest_el1el1 = 0.0;
	  gradUTrialDotGradUTest_el2el1 = 0.0;
	  gradUTrialDotGradUTest_el1el2 = 0.0;
	  gradUTrialDotGradUTest_el2el2 = 0.0;

	  	  
	  tmp_el1el1 = base_el1el1;
	  tmp_el1el2 = base_el1el2;
	  tmp_el2el1 = base_el2el1;
	  tmp_el2el2 = base_el2el2;

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
	    tN(s) = nor(s);	
	  }
	  nor_norm = sqrt(nor_norm);
	  tN /= nor_norm;

	  // element 1
	  fe.CalcShape(eip_el1, shape_el1);
	  // element 2
	  fe2.CalcShape(eip_el2, shape_el2);	  
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int j = 0; j < dim; j++){
	      for (int k = 0; k < h1dofs_cnt; k++){
		gradUResDotShape_el1(s,j) += nodalGrad_el1(k + j * h1dofs_cnt, s) * shape_el1(k);
		gradUResDotShape_el2(s,j) += nodalGrad_el2(k + j * h1dofs_cnt, s) * shape_el2(k);		
	      }
	    }
	  }	  	  	      

	  MultABt(gradUResDotShape_el1, gradUResDotShape_el1, gradUTrialDotGradUTest_el1el1);
	  MultABt(gradUResDotShape_el2, gradUResDotShape_el1, gradUTrialDotGradUTest_el2el1);
 	  MultABt(gradUResDotShape_el1, gradUResDotShape_el2, gradUTrialDotGradUTest_el1el2);
 	  MultABt(gradUResDotShape_el2, gradUResDotShape_el2, gradUTrialDotGradUTest_el2el2);
  
	  gradUResDotShape_el1.Mult(tN,normalGradU_el1);
	  gradUResDotShape_el2.Mult(tN,normalGradU_el2);
	  	    
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int k = 0; k < h1dofs_cnt; k++){
	      gradUResDotShape_TrialTest_el1el1(s + k * h1dofs_cnt) += normalGradU_el1(s) * normalGradU_el1(k);
	      gradUResDotShape_TrialTest_el1el2(s + k * h1dofs_cnt) += normalGradU_el1(s) * normalGradU_el2(k);
	      gradUResDotShape_TrialTest_el2el1(s + k * h1dofs_cnt) += normalGradU_el2(s) * normalGradU_el1(k);
	      gradUResDotShape_TrialTest_el2el2(s + k * h1dofs_cnt) += normalGradU_el2(s) * normalGradU_el2(k); 
	    }
	  }
	  
	  for (int nT = 1; nT <= nTerms; nT++){
	    penaltyParameter /= (double)nT;
	    double standardFactor =  nor_norm * ip_f.weight * 2 * (1.0/std::max(3 * Kappa, 2 * Mu)) * penaltyParameter;	    	   
	    double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	    weighted_h = pow(weighted_h,2*nT-1);	    

	    if (nT == 1){
	      for (int i = 0; i < h1dofs_cnt; i++)
		{
		  for (int vd = 0; vd < dim; vd++) // Velocity components.
		    {
		      for (int j = 0; j < h1dofs_cnt; j++)
			{
			  ////
			  elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt) += normalGradU_el1(i) * normalGradU_el1(j) * weighted_h * Mu * Mu * standardFactor;
			  elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) -= normalGradU_el1(i) * normalGradU_el2(j) * weighted_h * Mu * Mu * standardFactor;
			  elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt) -= normalGradU_el2(i) * normalGradU_el1(j) * weighted_h * Mu * Mu * standardFactor;
			  elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) += normalGradU_el2(i) * normalGradU_el2(j) * weighted_h * Mu * Mu * standardFactor;
			  ////
			  for (int sd = 0; sd < dim; sd++)
			    {
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUResDotShape_el1(j,sd) * gradUResDotShape_el1(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUResDotShape_el2(j,sd) * gradUResDotShape_el1(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUResDotShape_el1(j,sd) * gradUResDotShape_el2(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUResDotShape_el2(j,sd) * gradUResDotShape_el2(i,vd) * weighted_h * (Kappa - (2.0/3.0) * Mu) * (Kappa - (2.0/3.0) * Mu) * standardFactor;
			      /////////////////////		  
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += weighted_h * Mu * Mu * standardFactor * (gradUResDotShape_el1(j,vd) * normalGradU_el1(i) * tN(sd) + gradUResDotShape_el1(i,sd) * normalGradU_el1(j) * tN(vd) );
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= weighted_h * Mu * Mu * standardFactor * (gradUResDotShape_el2(j,vd) * normalGradU_el1(i) * tN(sd) + gradUResDotShape_el1(i,sd) * normalGradU_el2(j) * tN(vd) );
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= weighted_h * Mu * Mu * standardFactor * (gradUResDotShape_el1(j,vd) * normalGradU_el2(i) * tN(sd) + gradUResDotShape_el2(i,sd) * normalGradU_el1(j) * tN(vd) );
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += weighted_h * Mu * Mu * standardFactor * (gradUResDotShape_el2(j,vd) * normalGradU_el2(i) * tN(sd) + gradUResDotShape_el2(i,sd) * normalGradU_el2(j) * tN(vd) );
			      ////			  
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += 2.0 * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * standardFactor * ( tN(vd) * gradUResDotShape_el1(j,sd) * normalGradU_el1(i) + tN(sd) * gradUResDotShape_el1(i,vd) * normalGradU_el1(j) );
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= 2.0 * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * standardFactor * ( tN(vd) * gradUResDotShape_el2(j,sd) * normalGradU_el1(i) + tN(sd) * gradUResDotShape_el1(i,vd) * normalGradU_el2(j) );
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= 2.0 * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * standardFactor * ( tN(vd) * gradUResDotShape_el1(j,sd) * normalGradU_el2(i) + tN(sd) * gradUResDotShape_el2(i,vd) * normalGradU_el1(j) );
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt ) += 2.0 * weighted_h * (Kappa - (2.0/3.0) * Mu) * Mu * standardFactor * ( tN(vd) * gradUResDotShape_el2(j,sd) * normalGradU_el2(i) + tN(sd) * gradUResDotShape_el2(i,vd) * normalGradU_el2(j) );
			      ////////			  
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt) += gradUTrialDotGradUTest_el1el1(j,i) * weighted_h * Mu * Mu * tN(vd) * tN(sd) * standardFactor;
			      elmat(i + vd * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) -= gradUTrialDotGradUTest_el2el1(j,i) * weighted_h * Mu * Mu * tN(vd) * tN(sd) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt) -= gradUTrialDotGradUTest_el1el2(j,i) * weighted_h * Mu * Mu * tN(vd) * tN(sd) * standardFactor;
			      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + sd * h1dofs_cnt + dim * h1dofs_cnt) += gradUTrialDotGradUTest_el2el2(j,i) * weighted_h * Mu * Mu * tN(vd) * tN(sd) * standardFactor;
			      /////////
			    }
			}
		    }
		}	    
	    } 
	    else{      	      
	      lumped_el1el1 = 0.0;
	      lumped_el1el2 = 0.0;
	      lumped_el2el1 = 0.0;
	      lumped_el2el2 = 0.0;    

	      if (nT == 2){
		lumped_el1el1 = gradUResDotShape_TrialTest_el1el1;
		lumped_el1el2 = gradUResDotShape_TrialTest_el1el2;
		lumped_el2el1 = gradUResDotShape_TrialTest_el2el1;
		lumped_el2el2 = gradUResDotShape_TrialTest_el2el2;
	      }
	      else if (nT > 2){
		if (nT == 3){
		  TrialTestContract_el1el1 = tmp_el1el1;
		  TrialTestContract_el1el2 = tmp_el1el2;
		  TrialTestContract_el2el1 = tmp_el2el1;
		  TrialTestContract_el2el2 = tmp_el2el2;	      	  
		}
		else {  
		  Mult(base_el1el1,tmp_el1el1, TrialTestContract_el1el1);
		  Mult(base_el1el2,tmp_el1el2, TrialTestContract_el1el2);
		  Mult(base_el2el1,tmp_el2el1, TrialTestContract_el2el1);
		  Mult(base_el2el2,tmp_el2el2, TrialTestContract_el2el2);
		  
		  tmp_el1el1 = TrialTestContract_el1el1;
		  tmp_el1el2 = TrialTestContract_el1el2;
		  tmp_el2el1 = TrialTestContract_el2el1;
		  tmp_el2el2 = TrialTestContract_el2el2;
		}	  
	
		TrialTestContract_el1el1.Mult(gradUResDotShape_TrialTest_el1el1, lumped_el1el1);
		TrialTestContract_el2el1.Mult(gradUResDotShape_TrialTest_el2el1, lumped_el2el1);
		TrialTestContract_el1el2.Mult(gradUResDotShape_TrialTest_el1el2, lumped_el1el2);
		TrialTestContract_el2el2.Mult(gradUResDotShape_TrialTest_el2el2, lumped_el2el2);
	      }		

	      TrialTestContract_el1el1 = 0.0;
	      TrialTestContract_el1el2 = 0.0;  
	      TrialTestContract_el2el1 = 0.0;  
	      TrialTestContract_el2el2 = 0.0;  	

	      nodalGradContraction_el1el1.Mult(lumped_el1el1, assembledContraction_el1el1);
	      nodalGradContraction_el2el1.Mult(lumped_el2el1, assembledContraction_el2el1);
	      nodalGradContraction_el1el2.Mult(lumped_el1el2, assembledContraction_el1el2);
	      nodalGradContraction_el2el2.Mult(lumped_el2el2, assembledContraction_el2el2);
	      
	      for (int qt = 0; qt < h1dofs_cnt; qt++)
		{
		  for (int vd = 0; vd < dim; vd++)
		    {
		      for (int z = 0; z < h1dofs_cnt; z++)
			{
			  for (int md = 0; md < dim; md++)
			    {
			      elmat(qt + vd * h1dofs_cnt, z + vd * h1dofs_cnt) += 2.0 * Mu * Mu * weighted_h * assembledContraction_el1el1(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt, z + vd * h1dofs_cnt + dim * h1dofs_cnt) -= 2.0 * Mu * Mu * weighted_h * assembledContraction_el1el2(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + vd * h1dofs_cnt) -= 2.0 * Mu * Mu * weighted_h * assembledContraction_el2el1(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + vd * h1dofs_cnt + dim * h1dofs_cnt) += 2.0 * Mu * Mu * weighted_h * assembledContraction_el2el2(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      
			      elmat(qt + vd * h1dofs_cnt, z + md * h1dofs_cnt) += 2.0 * Mu * Mu * weighted_h * assembledContraction_el1el1(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + vd * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) -= 2.0 * Mu * Mu * weighted_h * assembledContraction_el1el2(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + vd * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt) -= 2.0 * Mu * Mu * weighted_h * assembledContraction_el2el1(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + vd * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) += 2.0 * Mu * Mu * weighted_h * assembledContraction_el2el2(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + vd * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      
			      elmat(qt + vd * h1dofs_cnt, z + md * h1dofs_cnt) += (Kappa - (2.0/3.0) * Mu) * ( (Kappa - (2.0/3.0) * Mu) * dim + 4.0 * Mu ) * weighted_h * assembledContraction_el1el1(qt + vd * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) -=  (Kappa - (2.0/3.0) * Mu) * ( (Kappa - (2.0/3.0) * Mu) * dim + 4.0 * Mu ) * weighted_h * assembledContraction_el1el2(qt + vd * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt) -=  (Kappa - (2.0/3.0) * Mu) * ( (Kappa - (2.0/3.0) * Mu) * dim + 4.0 * Mu ) * weighted_h * assembledContraction_el2el1(qt + vd * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + md * h1dofs_cnt + dim * h1dofs_cnt) +=  (Kappa - (2.0/3.0) * Mu) * ( (Kappa - (2.0/3.0) * Mu) * dim + 4.0 * Mu ) * weighted_h * assembledContraction_el2el2(qt + vd * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			    }
			}
		    }
		}	      
	    }
	  }
	}
    }
    else{
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
    }
  }*/

  /*  void GhostStressFullGradPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
								const FiniteElement &fe2,
								FaceElementTransformations &Tr,
								DenseMatrix &elmat)
  {
    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    int NEproc = pmesh->GetNE();
   if (Tr.Attribute == 77){    
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim), tN(dim);
      Vector shape_el1(h1dofs_cnt), shape_el2(h1dofs_cnt), normalGradU_el1(h1dofs_cnt), normalGradU_el2(h1dofs_cnt);
      Vector gradUResDotShape_TrialTest_el1el1(h1dofs_cnt*h1dofs_cnt), gradUResDotShape_TrialTest_el1el2(h1dofs_cnt*h1dofs_cnt);
      Vector gradUResDotShape_TrialTest_el2el1(h1dofs_cnt*h1dofs_cnt), gradUResDotShape_TrialTest_el2el2(h1dofs_cnt*h1dofs_cnt);
      DenseMatrix gradUResDotShape_el1(h1dofs_cnt,dim), gradUResDotShape_el2(h1dofs_cnt,dim), TrialTestContract_el1el1(h1dofs_cnt * h1dofs_cnt), TrialTestContract_el1el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix TrialTestContract_el2el1(h1dofs_cnt * h1dofs_cnt), TrialTestContract_el2el2(h1dofs_cnt * h1dofs_cnt), gradUTrialDotGradUTest_el1el1(h1dofs_cnt), gradUTrialDotGradUTest_el2el1(h1dofs_cnt), gradUTrialDotGradUTest_el1el2(h1dofs_cnt), gradUTrialDotGradUTest_el2el2(h1dofs_cnt);
      DenseMatrix base_el1el1(h1dofs_cnt * h1dofs_cnt), base_el1el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix base_el2el1(h1dofs_cnt * h1dofs_cnt), base_el2el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix tmp_el1el1(h1dofs_cnt * h1dofs_cnt), tmp_el1el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix tmp_el2el1(h1dofs_cnt * h1dofs_cnt), tmp_el2el2(h1dofs_cnt * h1dofs_cnt);
      Vector lumped_el1el1(h1dofs_cnt * h1dofs_cnt), lumped_el1el2(h1dofs_cnt * h1dofs_cnt);
      Vector lumped_el2el1(h1dofs_cnt * h1dofs_cnt), lumped_el2el2(h1dofs_cnt * h1dofs_cnt);
      DenseMatrix nodalGradContraction_el1el1(h1dofs_cnt * h1dofs_cnt * dim * dim, h1dofs_cnt * h1dofs_cnt), nodalGradContraction_el1el2(h1dofs_cnt * h1dofs_cnt * dim * dim, h1dofs_cnt * h1dofs_cnt);
      DenseMatrix nodalGradContraction_el2el1(h1dofs_cnt * h1dofs_cnt * dim * dim, h1dofs_cnt * h1dofs_cnt), nodalGradContraction_el2el2(h1dofs_cnt * h1dofs_cnt * dim * dim, h1dofs_cnt * h1dofs_cnt);
      Vector assembledContraction_el1el1(h1dofs_cnt * h1dofs_cnt * dim * dim), assembledContraction_el1el2(h1dofs_cnt * h1dofs_cnt * dim * dim);
      Vector assembledContraction_el2el1(h1dofs_cnt * h1dofs_cnt * dim * dim), assembledContraction_el2el2(h1dofs_cnt * h1dofs_cnt * dim * dim);
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  //	  const int order = 5 * max(fe.GetOrder(), 1);
	  const int order = 25;
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);	  
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      DenseMatrix nodalGrad_el1;
      DenseMatrix nodalGrad_el2;
      fe.ProjectGrad(fe,Trans_el1,nodalGrad_el1);
      fe2.ProjectGrad(fe2,Trans_el2,nodalGrad_el2);
      tmp_el1el1 = 0.0;
      tmp_el1el2 = 0.0;
      tmp_el2el1 = 0.0;
      tmp_el2el2 = 0.0;
      
      nodalGradContraction_el1el1 = 0.0;
      nodalGradContraction_el1el2 = 0.0;
      nodalGradContraction_el2el1 = 0.0;
      nodalGradContraction_el2el2 = 0.0;
      
      for (int a = 0; a < h1dofs_cnt; a++){
	for (int o = 0; o < h1dofs_cnt; o++){
	  for (int b = 0; b < h1dofs_cnt; b++){
	    for (int r = 0; r < h1dofs_cnt; r++){
	      for (int j = 0; j < dim; j++){		
		tmp_el1el1(a + b * h1dofs_cnt, o + r * h1dofs_cnt) += nodalGrad_el1(o + j * h1dofs_cnt, a) * nodalGrad_el1(r + j * h1dofs_cnt, b) ;
		tmp_el1el2(a + b * h1dofs_cnt, o + r * h1dofs_cnt) += nodalGrad_el1(o + j * h1dofs_cnt, a) * nodalGrad_el2(r + j * h1dofs_cnt, b) ;
		tmp_el2el1(a + b * h1dofs_cnt, o + r * h1dofs_cnt) += nodalGrad_el2(o + j * h1dofs_cnt, a) * nodalGrad_el1(r + j * h1dofs_cnt, b) ;
		tmp_el2el2(a + b * h1dofs_cnt, o + r * h1dofs_cnt) += nodalGrad_el2(o + j * h1dofs_cnt, a) * nodalGrad_el2(r + j * h1dofs_cnt, b) ;
		for (int k = 0; k < dim; k++){		
		  nodalGradContraction_el1el1(a + j * h1dofs_cnt + o * dim * h1dofs_cnt + k * h1dofs_cnt * h1dofs_cnt * dim, b + r * h1dofs_cnt) = nodalGrad_el1(b + j * h1dofs_cnt, a) * nodalGrad_el1(r + k * h1dofs_cnt, o);
		  nodalGradContraction_el1el2(a + j * h1dofs_cnt + o * dim * h1dofs_cnt + k * h1dofs_cnt * h1dofs_cnt * dim, b + r * h1dofs_cnt) = nodalGrad_el1(b + j * h1dofs_cnt, a) * nodalGrad_el2(r + k * h1dofs_cnt, o);
		  nodalGradContraction_el2el1(a + j * h1dofs_cnt + o * dim * h1dofs_cnt + k * h1dofs_cnt * h1dofs_cnt * dim, b + r * h1dofs_cnt) = nodalGrad_el2(b + j * h1dofs_cnt, a) * nodalGrad_el1(r + k * h1dofs_cnt, o);
		  nodalGradContraction_el2el2(a + j * h1dofs_cnt + o * dim * h1dofs_cnt + k * h1dofs_cnt * h1dofs_cnt * dim, b + r * h1dofs_cnt) = nodalGrad_el2(b + j * h1dofs_cnt, a) * nodalGrad_el2(r + k * h1dofs_cnt, o);
		}
	      }
	    }
	  }
	}
      }
      
      base_el1el1 = tmp_el1el1;
      base_el1el2 = tmp_el1el2;
      base_el2el1 = tmp_el2el1;
      base_el2el2 = tmp_el2el2;
      
      for (int q = 0; q < nqp_face; q++)
	{
	  penaltyParameter = dupPenaltyParameter;
	  shape_el1 = 0.0;
	  shape_el2 = 0.0;
	  
	  gradUResDotShape_el1 = 0.0;
	  gradUResDotShape_el2 = 0.0;
	  
	  nor = 0.0;
	  tN = 0.0;

	  TrialTestContract_el1el1 = 0.0;
	  TrialTestContract_el1el2 = 0.0;  
	  TrialTestContract_el2el1 = 0.0;  
	  TrialTestContract_el2el2 = 0.0;  

	  normalGradU_el1 = 0.0;
	  normalGradU_el2 = 0.0;
    	  
	  gradUResDotShape_TrialTest_el1el1 = 0.0;
	  gradUResDotShape_TrialTest_el1el2 = 0.0;
	  gradUResDotShape_TrialTest_el2el1 = 0.0;
	  gradUResDotShape_TrialTest_el2el2 = 0.0;	    
	 
	  gradUTrialDotGradUTest_el1el1 = 0.0;
	  gradUTrialDotGradUTest_el2el1 = 0.0;
	  gradUTrialDotGradUTest_el1el2 = 0.0;
	  gradUTrialDotGradUTest_el2el2 = 0.0;

	  	  
	  tmp_el1el1 = base_el1el1;
	  tmp_el1el2 = base_el1el2;
	  tmp_el2el1 = base_el2el1;
	  tmp_el2el2 = base_el2el2;

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
	    tN(s) = nor(s);	
	  }
	  nor_norm = sqrt(nor_norm);
	  tN /= nor_norm;

	  // element 1
	  fe.CalcShape(eip_el1, shape_el1);
	  // element 2
	  fe2.CalcShape(eip_el2, shape_el2);	  
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int j = 0; j < dim; j++){
	      for (int k = 0; k < h1dofs_cnt; k++){
		gradUResDotShape_el1(s,j) += nodalGrad_el1(k + j * h1dofs_cnt, s) * shape_el1(k);
		gradUResDotShape_el2(s,j) += nodalGrad_el2(k + j * h1dofs_cnt, s) * shape_el2(k);		
	      }
	    }
	  }	  	  	      

	  MultABt(gradUResDotShape_el1, gradUResDotShape_el1, gradUTrialDotGradUTest_el1el1);
	  MultABt(gradUResDotShape_el2, gradUResDotShape_el1, gradUTrialDotGradUTest_el2el1);
 	  MultABt(gradUResDotShape_el1, gradUResDotShape_el2, gradUTrialDotGradUTest_el1el2);
 	  MultABt(gradUResDotShape_el2, gradUResDotShape_el2, gradUTrialDotGradUTest_el2el2);
  
	  gradUResDotShape_el1.Mult(tN,normalGradU_el1);
	  gradUResDotShape_el2.Mult(tN,normalGradU_el2);
	  	    
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int k = 0; k < h1dofs_cnt; k++){
	      gradUResDotShape_TrialTest_el1el1(s + k * h1dofs_cnt) += normalGradU_el1(s) * normalGradU_el1(k);
	      gradUResDotShape_TrialTest_el1el2(s + k * h1dofs_cnt) += normalGradU_el1(s) * normalGradU_el2(k);
	      gradUResDotShape_TrialTest_el2el1(s + k * h1dofs_cnt) += normalGradU_el2(s) * normalGradU_el1(k);
	      gradUResDotShape_TrialTest_el2el2(s + k * h1dofs_cnt) += normalGradU_el2(s) * normalGradU_el2(k); 
	    }
	  }
	  
	  for (int nT = 1; nT <= nTerms; nT++){
	    penaltyParameter /= (double)nT;
	    double standardFactor =  nor_norm * ip_f.weight * 2 * std::max(3 * Kappa, 2 * Mu) * penaltyParameter;	    	   
	    double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	    weighted_h = pow(weighted_h,2*nT-1);	    

	    if (nT == 1){
	      for (int i = 0; i < h1dofs_cnt; i++)
		{
		  for (int vd = 0; vd < dim; vd++) // Velocity components.
		    {
		      for (int j = 0; j < h1dofs_cnt; j++)
			{
			  ////
			  elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt) += normalGradU_el1(i) * normalGradU_el1(j) * weighted_h * standardFactor;
			  elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) -= normalGradU_el1(i) * normalGradU_el2(j) * weighted_h * standardFactor;
			  elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt) -= normalGradU_el2(i) * normalGradU_el1(j) * weighted_h * standardFactor;
			  elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) += normalGradU_el2(i) * normalGradU_el2(j) * weighted_h * standardFactor;
			  ////
			}
		    }
		}	    
	    } 
	    else{      	      
	      lumped_el1el1 = 0.0;
	      lumped_el1el2 = 0.0;
	      lumped_el2el1 = 0.0;
	      lumped_el2el2 = 0.0;    

	      if (nT == 2){
		lumped_el1el1 = gradUResDotShape_TrialTest_el1el1;
		lumped_el1el2 = gradUResDotShape_TrialTest_el1el2;
		lumped_el2el1 = gradUResDotShape_TrialTest_el2el1;
		lumped_el2el2 = gradUResDotShape_TrialTest_el2el2;
	      }
	      else if (nT > 2){
		if (nT == 3){
		  TrialTestContract_el1el1 = tmp_el1el1;
		  TrialTestContract_el1el2 = tmp_el1el2;
		  TrialTestContract_el2el1 = tmp_el2el1;
		  TrialTestContract_el2el2 = tmp_el2el2;	      	  
		}
		else {  
		  Mult(base_el1el1,tmp_el1el1, TrialTestContract_el1el1);
		  Mult(base_el1el2,tmp_el1el2, TrialTestContract_el1el2);
		  Mult(base_el2el1,tmp_el2el1, TrialTestContract_el2el1);
		  Mult(base_el2el2,tmp_el2el2, TrialTestContract_el2el2);
		  
		  tmp_el1el1 = TrialTestContract_el1el1;
		  tmp_el1el2 = TrialTestContract_el1el2;
		  tmp_el2el1 = TrialTestContract_el2el1;
		  tmp_el2el2 = TrialTestContract_el2el2;
		}	  
	
		TrialTestContract_el1el1.Mult(gradUResDotShape_TrialTest_el1el1, lumped_el1el1);
		TrialTestContract_el2el1.Mult(gradUResDotShape_TrialTest_el2el1, lumped_el2el1);
		TrialTestContract_el1el2.Mult(gradUResDotShape_TrialTest_el1el2, lumped_el1el2);
		TrialTestContract_el2el2.Mult(gradUResDotShape_TrialTest_el2el2, lumped_el2el2);
	      }		

	      TrialTestContract_el1el1 = 0.0;
	      TrialTestContract_el1el2 = 0.0;  
	      TrialTestContract_el2el1 = 0.0;  
	      TrialTestContract_el2el2 = 0.0;  	

	      nodalGradContraction_el1el1.Mult(lumped_el1el1, assembledContraction_el1el1);
	      nodalGradContraction_el2el1.Mult(lumped_el2el1, assembledContraction_el2el1);
	      nodalGradContraction_el1el2.Mult(lumped_el1el2, assembledContraction_el1el2);
	      nodalGradContraction_el2el2.Mult(lumped_el2el2, assembledContraction_el2el2);
	      
	      for (int qt = 0; qt < h1dofs_cnt; qt++)
		{
		  for (int vd = 0; vd < dim; vd++)
		    {
		      for (int z = 0; z < h1dofs_cnt; z++)
			{
			  for (int md = 0; md < dim; md++)
			    {
			      elmat(qt + vd * h1dofs_cnt, z + vd * h1dofs_cnt) += weighted_h * assembledContraction_el1el1(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt, z + vd * h1dofs_cnt + dim * h1dofs_cnt) -= weighted_h * assembledContraction_el1el2(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + vd * h1dofs_cnt) -= weighted_h * assembledContraction_el2el1(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			      elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + vd * h1dofs_cnt + dim * h1dofs_cnt) += weighted_h * assembledContraction_el2el2(qt + md * h1dofs_cnt + z * dim * h1dofs_cnt + md * h1dofs_cnt * h1dofs_cnt * dim) * standardFactor;
			    }
			}
		    }
		}	      
	    }
	  }
	}
    }
    else{
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
    }
    }*/

  void AddOneToBinaryArray(Array<int> & binary, int size, int dim){
    if (dim == 3){
      if ( (binary[size-1] == 0) ||  (binary[size-1] == 1) ){
	binary[size-1] += 1;
      }
      else{
	binary[size-1] = 0;
	AddOneToBinaryArray(binary,size-1,dim);
      }
    }
    else{
      if (binary[size-1] == 0) {
	binary[size-1] += 1;
      }
      else{
	binary[size-1] = 0;
	AddOneToBinaryArray(binary,size-1,dim);
      }
    }
  }
  
  void GhostStressFullGradPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
								const FiniteElement &fe2,
								FaceElementTransformations &Tr,
								DenseMatrix &elmat)
  {
    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    int NEproc = pmesh->GetNE();
    if (Tr.Attribute == 77){
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
      Vector nor(dim), tN(dim), tang1(dim), tang1_Unit(dim), tang2(dim), tang2_Unit(dim);
      Vector shape_el1(h1dofs_cnt), shape_el2(h1dofs_cnt); 
      DenseMatrix normalGradU_el1(h1dofs_cnt), normalGradU_el2(h1dofs_cnt), tangent1GradU_el1(h1dofs_cnt), tangent1GradU_el2(h1dofs_cnt), tangent2GradU_el1(h1dofs_cnt), tangent2GradU_el2(h1dofs_cnt), UnitVectorProjectionsAssembled_el1(h1dofs_cnt), UnitVectorProjectionsAssembled_el2(h1dofs_cnt), UnitVectorProjectionsAssembled_el1_tmp(h1dofs_cnt), UnitVectorProjectionsAssembled_el2_tmp(h1dofs_cnt);
      Vector base_el1(h1dofs_cnt), assembledTrial_el1(h1dofs_cnt);
      Vector base_el2(h1dofs_cnt), assembledTrial_el2(h1dofs_cnt);     
      Array<DenseMatrix *> UnitVectorProjections_el1;
      Array<DenseMatrix *> UnitVectorProjections_el2;      
      UnitVectorProjections_el1.SetSize(dim);
      UnitVectorProjections_el2.SetSize(dim);
      normalGradU_el1 = 0.0;
      normalGradU_el2 = 0.0;
      tangent1GradU_el1 = 0.0;
      tangent1GradU_el2 = 0.0;
      tangent2GradU_el1 = 0.0;
      tangent2GradU_el2 = 0.0;
	  
      if (dim == 2){
	UnitVectorProjections_el1[0] = &normalGradU_el1;
	UnitVectorProjections_el1[1] = &tangent1GradU_el1;
	UnitVectorProjections_el2[0] = &normalGradU_el2;
	UnitVectorProjections_el2[1] = &tangent1GradU_el2;	    	 
      }
      else{
	UnitVectorProjections_el1[0] = &normalGradU_el1;
	UnitVectorProjections_el1[1] = &tangent1GradU_el1;
	UnitVectorProjections_el1[2] = &tangent2GradU_el1;	   
	UnitVectorProjections_el2[0] = &normalGradU_el2;
	UnitVectorProjections_el2[1] = &tangent1GradU_el2;
	UnitVectorProjections_el2[2] = &tangent2GradU_el2;	    	 	
      }
	  
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 5 * max(fe.GetOrder(), 1);
	  //	  const int order = 25;
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
	  penaltyParameter = dupPenaltyParameter;
	  shape_el1 = 0.0;
	  shape_el2 = 0.0;
	  base_el1 = 0.0;
	  base_el2 = 0.0;
	  
	  nor = 0.0;
	  tN = 0.0;
	  tang1 = 0.0;
	  tang2 = 0.0;
	  tang1_Unit = 0.0;
	  tang2_Unit = 0.0;
	  
	  normalGradU_el1 = 0.0;
	  normalGradU_el2 = 0.0;
	  tangent1GradU_el1 = 0.0;
	  tangent1GradU_el2 = 0.0;
	  tangent2GradU_el1 = 0.0;
	  tangent2GradU_el2 = 0.0;
	  
	  assembledTrial_el1 = 0.0;
	  assembledTrial_el2 = 0.0;

	  UnitVectorProjectionsAssembled_el1 = 0.0;
	  UnitVectorProjectionsAssembled_el2 = 0.0;
	  UnitVectorProjectionsAssembled_el1_tmp = 0.0;
	  UnitVectorProjectionsAssembled_el2_tmp = 0.0;
	 
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);
	  double Kappa = kappa->Eval(*Tr.Elem1, eip_el1);
	  const double *d = (Tr.Jacobian()).Data();
	  if ((Tr.Jacobian()).Height() == 2)
	    {
	      tang1(0) = d[0];
	      tang1(1) = d[1];
	      tang2 = 0.0;
	      tang1_Unit = tang1;
	      tang2_Unit = tang2;
	    }
	  else
	    {
	      tang1(0) = d[0];
	      tang1(1) = d[1];
	      tang1(2) = d[2];
	      tang2(0) = d[3];
	      tang2(1) = d[4];
	      tang2(2) = d[5];
	      tang1_Unit = tang1;
	      tang2_Unit = tang2;	  
	    }
	  tN = nor;
	  double nor_norm = 0.0;
	  double tang1_norm = 0.0;
	  double tang2_norm = 0.0;	
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	    tang1_norm += tang1(s) * tang1(s);
	    tang2_norm += tang2(s) * tang2(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  tN /= nor_norm;
	  tang1_norm = sqrt(tang1_norm);	 
	  tang1_Unit /= tang1_norm;
	  if (dim == 3){
	    tang2_norm = sqrt(tang2_norm);	 	
	    tang2_Unit /= tang2_norm; 
	  }
	  
	  // element 1
	  fe.CalcShape(eip_el1, shape_el1);
	  // element 2
	  fe2.CalcShape(eip_el2, shape_el2);
	  
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int j = 0; j < dim; j++){	  
		normalGradU_el1(s,k) += nodalGrad_el1(k + j * h1dofs_cnt, s) * tN(j);
		normalGradU_el2(s,k) += nodalGrad_el2(k + j * h1dofs_cnt, s) * tN(j);
		tangent1GradU_el1(s,k) += nodalGrad_el1(k + j * h1dofs_cnt, s) * tang1_Unit(j);
		tangent1GradU_el2(s,k) += nodalGrad_el2(k + j * h1dofs_cnt, s) * tang1_Unit(j);
		tangent2GradU_el1(s,k) += nodalGrad_el1(k + j * h1dofs_cnt, s) * tang2_Unit(j);
		tangent2GradU_el2(s,k) += nodalGrad_el2(k + j * h1dofs_cnt, s) * tang2_Unit(j);			  
	      }
	    }
	  }	  	  	      
	  
	  normalGradU_el1.Mult(shape_el1,base_el1);
	  normalGradU_el2.Mult(shape_el2,base_el2);

	  for (int nT = 1; nT <= nTerms; nT++){
	    penaltyParameter /= (double)nT;
	    double standardFactor =  nor_norm * ip_f.weight * 2 * std::max(3 * Kappa, 2 * Mu) * penaltyParameter;	
	    double weighted_h = ((Tr.Elem1->Weight()/nor_norm) * (Tr.Elem2->Weight() / nor_norm) )/ ( (Tr.Elem1->Weight()/nor_norm) + (Tr.Elem2->Weight() / nor_norm));
	    weighted_h = pow(weighted_h,2*nT-1);	    

	    if (nT == 1){
	      for (int i = 0; i < h1dofs_cnt; i++)
		{
		  for (int vd = 0; vd < dim; vd++) // Velocity components.
		    {
		      for (int j = 0; j < h1dofs_cnt; j++)
			{
			  ////
			  elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt) += base_el1(i) * base_el1(j) * weighted_h * standardFactor;
			  elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) -= base_el1(i) * base_el2(j) * weighted_h * standardFactor;
			  elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt) -= base_el2(i) * base_el1(j) * weighted_h * standardFactor;
			  elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) += base_el2(i) * base_el2(j) * weighted_h * standardFactor;
			  ////
			}
		    }
		}	    
	    } 
	    else{
	      Array<int> binaries;
	      binaries.SetSize(nT-1);
	      binaries = 0;
	      //	      std::cout << " q " << q << std::endl;
	      for (int i = 0; i < std::pow(dim,nT-1); i++){
		assembledTrial_el1 = 0.0;
		assembledTrial_el2 = 0.0;
		UnitVectorProjectionsAssembled_el1_tmp = *(UnitVectorProjections_el1[binaries[0]]);
		UnitVectorProjectionsAssembled_el2_tmp = *(UnitVectorProjections_el2[binaries[0]]);
		UnitVectorProjectionsAssembled_el1 = UnitVectorProjectionsAssembled_el1_tmp;
		UnitVectorProjectionsAssembled_el2 = UnitVectorProjectionsAssembled_el2_tmp;
		//	binaries.Print(std::cout, nT-1);
		for (int j = 0; j < (binaries.Size()-1); j++){		    
		  Mult(UnitVectorProjectionsAssembled_el1_tmp, *(UnitVectorProjections_el1[binaries[j+1]]), UnitVectorProjectionsAssembled_el1);
		  Mult(UnitVectorProjectionsAssembled_el2_tmp, *(UnitVectorProjections_el2[binaries[j+1]]), UnitVectorProjectionsAssembled_el2);
		  UnitVectorProjectionsAssembled_el1_tmp = UnitVectorProjectionsAssembled_el1;
		  UnitVectorProjectionsAssembled_el2_tmp = UnitVectorProjectionsAssembled_el2;
		}
		UnitVectorProjectionsAssembled_el1.Mult(base_el1,assembledTrial_el1);
		UnitVectorProjectionsAssembled_el2.Mult(base_el2,assembledTrial_el2);
		for (int qt = 0; qt < h1dofs_cnt; qt++)
		  {
		    for (int vd = 0; vd < dim; vd++)
		      {
			for (int z = 0; z < h1dofs_cnt; z++)
			  {
			    elmat(qt + vd * h1dofs_cnt, z + vd * h1dofs_cnt) += weighted_h * assembledTrial_el1(qt) * assembledTrial_el1(z) * standardFactor;
			    elmat(qt + vd * h1dofs_cnt, z + vd * h1dofs_cnt + dim * h1dofs_cnt) -= weighted_h * assembledTrial_el1(qt) * assembledTrial_el2(z) * standardFactor;
			    elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + vd * h1dofs_cnt) -= weighted_h * assembledTrial_el2(qt) * assembledTrial_el1(z) * standardFactor;
			    elmat(qt + vd * h1dofs_cnt + dim * h1dofs_cnt, z + vd * h1dofs_cnt + dim * h1dofs_cnt) += weighted_h * assembledTrial_el2(qt) * assembledTrial_el2(z) * standardFactor;
			  }
		      }
		  }
		if (i < (std::pow(dim,nT-1)-1) ){
		  AddOneToBinaryArray(binaries, binaries.Size(), dim);
		}
	      }		
	    }
	  }	     
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
