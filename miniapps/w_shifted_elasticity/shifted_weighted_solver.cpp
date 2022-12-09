// Copyright (c) 20A17, Lawrence Livermore NatAional Security, LLC. Produced at
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

#include "shifted_weighted_solver.hpp"
#include <unordered_map>

namespace mfem
{
  double factorial(int nTerms){
    double factorial = 1.0;	
    for (int s = 1; s <= nTerms; s++){
      factorial = factorial*s;
    }
    return factorial;
  }

  void ShiftedVectorFunctionCoefficient::Eval(Vector &V,
					      ElementTransformation & T,
					      const IntegrationPoint & ip,
					      const Vector &D)
  {
    Vector transip;
    T.Transform(ip, transip);
    for (int i = 0; i < D.Size(); i++)
      {
	transip(i) += D(i);
      }
    
    Function(transip, V);
  }

  void WeightedShiftedStressBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
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
    
    if ( (elem1_inside && elem2_cut) || (elem1_cut && elem2_inside) ||  (elem1_cut && elem2_cut) || (elem1_cut && elem2_outside) ||  (elem1_outside && elem2_cut) ) {
      const int dim = fe.GetDim();
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim);
      DenseMatrix taylorExp_el1(dofs_cnt, dim), taylorExp_el2(dofs_cnt,dim);
      DenseMatrix gradUResDirD_el1(dofs_cnt), dshape_el2(dofs_cnt,dim), gradUResDirD_el2(dofs_cnt), nodalGrad_taylor_trial_el1(dofs_cnt,dim), nodalGrad_taylor_trial_el2(dofs_cnt,dim);
      Vector shape_el1(dofs_cnt), shape_el2(dofs_cnt), gradUResD_el1(dofs_cnt), gradUResD_el2(dofs_cnt), shape_el1_test(dofs_cnt), shape_el2_test(dofs_cnt);

      shape_el1 = 0.0;
      shape_el1_test = 0.0;
      taylorExp_el1 = 0.0;
      gradUResDirD_el1 = 0.0;
      gradUResD_el1 = 0.0;
      nodalGrad_taylor_trial_el1 = 0.0;
      nodalGrad_taylor_trial_el2 = 0.0;
      
      shape_el2 = 0.0;
      shape_el2_test = 0.0;
      taylorExp_el2 = 0.0;
      gradUResDirD_el2 = 0.0;
      gradUResD_el2 = 0.0;

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

      /*  const IntegrationRule &nodes = fe.GetNodes();
      for (int j = 0; j < nodes.GetNPoints(); j++)
	{
	  const IntegrationPoint &ip_nodes = nodes.IntPoint(j);
	  Vector D_el1(dim);
	  D_el1 = 0.0;
	  //	  Vector x(3);
	  //  Trans_el1.Transform(ip_nodes,x);
	  vD->Eval(D_el1, Trans_el1, ip_nodes);
	  for (int s = 0; s < dim; s++){
	    std::cout << " D " << D_el1(s) << std::endl;
	  }
	}
      */
      for (int q = 0; q < nqp_face; q++)
	{
	  shape_el1 = 0.0;
	  shape_el2_test = 0.0;
	  taylorExp_el1 = 0.0;
	  gradUResDirD_el1 = 0.0;
	  gradUResD_el1 = 0.0;
	  
	  shape_el2 = 0.0;
	  shape_el2_test = 0.0;
	  taylorExp_el2 = 0.0;
	  gradUResDirD_el2 = 0.0;
	  gradUResD_el2 = 0.0;
	  
	  nor = 0.0;
	  nodalGrad_taylor_trial_el1 = 0.0;
	  nodalGrad_taylor_trial_el2 = 0.0;

	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);

	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);
	  double Kappa = kappa->Eval(*Tr.Elem1, eip_el1);

	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;

	  /////
	  Vector D_el1(dim);
	  Vector tN_el1(dim);
	  vD->Eval(D_el1, Trans_el1, eip_el1);
	  vN->Eval(tN_el1, Trans_el1, eip_el1);

	  /////
	  double nTildaDotN = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  for (int s = 0; s < dim; s++){
	    nTildaDotN += (nor(s) / nor_norm) * tN_el1(s);
	    // std::cout << " D " << D_el1(s) << std::endl;
	  }
	  
	  fe.CalcShape(eip_el1, shape_el1);
	  fe.CalcShape(eip_el1, shape_el1_test);
	      
	  for (int k = 0; k < dofs_cnt; k++){
	    for (int s = 0; s < dofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el1(s,k) += nodalGrad_el1(k + j * dofs_cnt, s) * D_el1(j);
	      }
	    }
	  }

	  if (nTerms >= 2){
	    DenseMatrix tmp_el1(dofs_cnt);
	    DenseMatrix dummy_tmp_el1(dofs_cnt);
	    tmp_el1 = gradUResDirD_el1;
	    taylorExp_el1 = gradUResDirD_el1;
	    dummy_tmp_el1 = 0.0;
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int s = 0; s < dofs_cnt; s++){
		gradUResD_el1(k) += taylorExp_el1(k,s) * shape_el1(s);  
	      }
	    }
	    for (int p = 2; p < nTerms; p++){
	      dummy_tmp_el1 = 0.0;
	      taylorExp_el1 = 0.0;
	      for (int k = 0; k < dofs_cnt; k++){
		for (int s = 0; s < dofs_cnt; s++){
		  for (int r = 0; r < dofs_cnt; r++){
		    taylorExp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s) * (1.0/factorial(p));
		    dummy_tmp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s);
		  }
		}
	      }
	      tmp_el1 = dummy_tmp_el1;
	      for (int k = 0; k < dofs_cnt; k++){
		for (int s = 0; s < dofs_cnt; s++){
		  gradUResD_el1(k) += taylorExp_el1(k,s) * shape_el1(s);  
		}
	      }
	    }
	  }
	  
	  ////
	  shape_el1 += gradUResD_el1;
	  //
	
	  /////
	  Vector D_el2(dim);
	  Vector tN_el2(dim);
	  vD->Eval(D_el2, Trans_el2, eip_el2);
	  vN->Eval(tN_el2, Trans_el2, eip_el2);
	  /////
	  
	  fe2.CalcShape(eip_el2, shape_el2);
	  fe2.CalcShape(eip_el2, shape_el2_test);
	       
	  for (int k = 0; k < dofs_cnt; k++){
	    for (int s = 0; s < dofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el2(s,k) += nodalGrad_el2(k + j * dofs_cnt, s) * D_el2(j);
	      }
	    }
	  }
	  
	  if (nTerms >= 2){
	    DenseMatrix tmp_el2(dofs_cnt);
	    DenseMatrix dummy_tmp_el2(dofs_cnt);
	    tmp_el2 = gradUResDirD_el2;
	    taylorExp_el2 = gradUResDirD_el2;
	    dummy_tmp_el2 = 0.0;
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int s = 0; s < dofs_cnt; s++){
		gradUResD_el2(k) += taylorExp_el2(k,s) * shape_el2(s);  
	      }
	    }
	    for ( int p = 2; p < nTerms; p++){
	      dummy_tmp_el2 = 0.0;
	      taylorExp_el2 = 0.0;
	      for (int k = 0; k < dofs_cnt; k++){
		for (int s = 0; s < dofs_cnt; s++){
		  for (int r = 0; r < dofs_cnt; r++){
		    taylorExp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s) * (1.0/factorial(p));
		    dummy_tmp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s);
		  }
		}
	      }
	      tmp_el2 = dummy_tmp_el2;
	      for (int k = 0; k < dofs_cnt; k++){
		for (int s = 0; s < dofs_cnt; s++){
		  gradUResD_el2(k) += taylorExp_el2(k,s) * shape_el2(s);  
		}
	      }
	    }
	  }
	  
	  ////
	  shape_el2 += gradUResD_el2;
	  //
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      nodalGrad_taylor_trial_el1(i,vd) += nodalGrad_el1(j + vd * dofs_cnt, i) * shape_el1(j);
		      nodalGrad_taylor_trial_el2(i,vd) += nodalGrad_el2(j + vd * dofs_cnt, i) * shape_el2(j);
		    }
		}
	    }

	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{			      
			  elmat(i + vd * dofs_cnt, j + vd * dofs_cnt) += nodalGrad_taylor_trial_el1(j,md) * volumeFraction_el1 * gamma_1 * shape_el1_test(i) * Mu * ip_f.weight * nTildaDotN * nor_norm * tN_el1(md);
			  elmat(i + vd * dofs_cnt, j + vd * dofs_cnt + dim * dofs_cnt) += nodalGrad_taylor_trial_el2(j,md) * volumeFraction_el1 * gamma_2 * shape_el1_test(i) * Mu * ip_f.weight * nTildaDotN * nor_norm * tN_el1(md);
			  
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + vd * dofs_cnt) -= nodalGrad_taylor_trial_el1(j,md) * volumeFraction_el2 * gamma_1 * shape_el2_test(i) * Mu * ip_f.weight * nTildaDotN * nor_norm * tN_el1(md);
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + vd * dofs_cnt + dim * dofs_cnt) -= nodalGrad_taylor_trial_el2(j,md) * volumeFraction_el2 * gamma_2 * shape_el2_test(i) * Mu * ip_f.weight * nTildaDotN * nor_norm * tN_el1(md);
			  
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt) += nodalGrad_taylor_trial_el1(j,vd) * volumeFraction_el1 * gamma_1 * shape_el1_test(i) * Mu * ip_f.weight * nTildaDotN * nor_norm * tN_el1(md);
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) += nodalGrad_taylor_trial_el2(j,vd) * volumeFraction_el1 * gamma_2 * shape_el1_test(i) * Mu * ip_f.weight * nTildaDotN * nor_norm * tN_el1(md);
			  
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt) -= nodalGrad_taylor_trial_el1(j,vd) * volumeFraction_el2 * gamma_1 * shape_el2_test(i) * Mu * ip_f.weight * nTildaDotN * nor_norm * tN_el1(md);
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= nodalGrad_taylor_trial_el2(j,vd) * volumeFraction_el2 * gamma_2 * shape_el2_test(i) * Mu * ip_f.weight * nTildaDotN * nor_norm * tN_el1(md);
			  
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt) += nodalGrad_taylor_trial_el1(j,md) * volumeFraction_el1 * gamma_1 * shape_el1_test(i) * (Kappa - (2.0/3.0) * Mu) * ip_f.weight * nTildaDotN * nor_norm * tN_el1(vd);
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) += nodalGrad_taylor_trial_el2(j,md) * volumeFraction_el1 * gamma_2 * shape_el1_test(i) * (Kappa - (2.0/3.0) * Mu) * ip_f.weight * nTildaDotN * nor_norm * tN_el1(vd);
			  
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt) -= nodalGrad_taylor_trial_el1(j,md) * volumeFraction_el2 * gamma_1 * shape_el2_test(i) * (Kappa - (2.0/3.0) * Mu) * ip_f.weight * nTildaDotN * nor_norm * tN_el1(vd);
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= nodalGrad_taylor_trial_el2(j,md) * volumeFraction_el2 * gamma_2 * shape_el2_test(i) * (Kappa - (2.0/3.0) * Mu) * ip_f.weight * nTildaDotN * nor_norm * tN_el1(vd);
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
  
  void WeightedShiftedStressBoundaryForceTransposeIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
									 const FiniteElement &fe2,
									 FaceElementTransformations &Tr,
									 DenseMatrix &elmat)
  {
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();

    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    int NEproc = pmesh->GetNE();

    //  std::cout << " I AM IN ASSEMBLE FACE " << std::endl;
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
    
    if ( (elem1_inside && elem2_cut) || (elem1_cut && elem2_inside) ||  (elem1_cut && elem2_cut) || (elem1_cut && elem2_outside) ||  (elem1_outside && elem2_cut) ) {
      const int dim = fe.GetDim();
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim);
      DenseMatrix dshape_el1(dofs_cnt,dim), dshape_ps_el1(dofs_cnt,dim), adjJ_el1(dim), dshape_el2(dofs_cnt,dim), dshape_ps_el2(dofs_cnt,dim), adjJ_el2(dim);
      Vector shape_el1(dofs_cnt), gradURes_el1(dofs_cnt), shape_el2(dofs_cnt), gradURes_el2(dofs_cnt);

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
	  const int order = 10 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      
      for (int q = 0; q < nqp_face; q++)
	{
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
	  double Kappa = kappa->Eval(*Tr.Elem1, eip_el1);
		  
	  fe.CalcShape(eip_el1, shape_el1);
	  fe.CalcDShape(eip_el1, dshape_el1);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ_el1);
	  Mult(dshape_el1, adjJ_el1, dshape_ps_el1);
	  dshape_ps_el1.Mult(nor,gradURes_el1);
	  
	  fe2.CalcShape(eip_el2, shape_el2);
	  fe2.CalcDShape(eip_el2, dshape_el2);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ_el2);
	  Mult(dshape_el2, adjJ_el2, dshape_ps_el2);
	  dshape_ps_el2.Mult(nor,gradURes_el2);

	  //
	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;
	  //

	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      elmat(i + vd * dofs_cnt, j + vd * dofs_cnt) -= volumeFraction_el1 * shape_el1(i) * gamma_1 * gradURes_el1(j) * Mu * ip_f.weight / Tr.Elem1->Weight();
		      elmat(i + vd * dofs_cnt, j + vd * dofs_cnt + dim * dofs_cnt) -= volumeFraction_el1 * shape_el1(i) * gamma_2 * gradURes_el2(j) * Mu * ip_f.weight / Tr.Elem2->Weight();
		      
		      elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + vd * dofs_cnt) += volumeFraction_el2 * shape_el2(i) * gamma_1 * gradURes_el1(j) * Mu * ip_f.weight / Tr.Elem1->Weight();
		      elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + vd * dofs_cnt + dim * dofs_cnt) += volumeFraction_el2 * shape_el2(i) * gamma_2 * gradURes_el2(j) * Mu * ip_f.weight / Tr.Elem2->Weight();
		      
		      for (int md = 0; md < dim; md++) // Velocity components.
			{
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt) -= volumeFraction_el1 * shape_el1(i) * gamma_1 * nor(md) * dshape_ps_el1(j,vd) * Mu * ip_f.weight / Tr.Elem1->Weight();
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= volumeFraction_el1 * shape_el1(i) * gamma_2 * nor(md) * dshape_ps_el2(j,vd) * Mu * ip_f.weight / Tr.Elem2->Weight();

			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt) += volumeFraction_el2 * shape_el2(i) * gamma_1 * nor(md) * dshape_ps_el1(j,vd) * Mu * ip_f.weight / Tr.Elem1->Weight();
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) += volumeFraction_el2 * shape_el2(i) * gamma_2 * nor(md) * dshape_ps_el2(j,vd) * Mu * ip_f.weight / Tr.Elem2->Weight();

			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt) -= volumeFraction_el1 * shape_el1(i) * gamma_1 * nor(vd) * dshape_ps_el1(j,md) * (Kappa - (2.0/3.0) * Mu) * ip_f.weight / Tr.Elem1->Weight();
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= volumeFraction_el1 * shape_el1(i) * gamma_2 * nor(vd) * dshape_ps_el2(j,md) * (Kappa - (2.0/3.0) * Mu) * ip_f.weight / Tr.Elem2->Weight();

			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt) += volumeFraction_el2 * shape_el2(i) * gamma_1 * nor(vd) * dshape_ps_el1(j,md) * (Kappa - (2.0/3.0) * Mu) * ip_f.weight / Tr.Elem1->Weight();
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) += volumeFraction_el2 * shape_el2(i) * gamma_2 * nor(vd) * dshape_ps_el2(j,md) * (Kappa - (2.0/3.0) * Mu) * ip_f.weight / Tr.Elem2->Weight();
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

  void WeightedShiftedStressNitscheBCForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
								     const FiniteElement &el2,
								     FaceElementTransformations &Tr,
								     Vector &elvect)
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
    if ( (elem1_inside && elem2_cut) || (elem1_cut && elem2_inside) ||  (elem1_cut && elem2_cut) || (elem1_cut && elem2_outside) ||  (elem1_outside && elem2_cut) ) {
      const int dim = el.GetDim();
      const int dofs_cnt = el.GetDof();
      elvect.SetSize(2*dofs_cnt*dim);
      elvect = 0.0;
	
      Vector nor(dim), bcEval(dim);
      Vector shape_el1(dofs_cnt), shape_el2(dofs_cnt);

      shape_el1 = 0.0;

      shape_el2 = 0.0;
      nor = 0.0;
      bcEval = 0.0;
	
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 10 * max(el.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();	
      for (int q = 0; q  < nqp_face; q++)
	{
	  shape_el1 = 0.0;
	  
	  shape_el2 = 0.0;
	  nor = 0.0;
	  bcEval = 0.0;
	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);

	  ///
	  Vector D_el1(dim);
	  Vector tN_el1(dim);
	  D_el1 = 0.0;
	  tN_el1 = 0.0; 
	  vD->Eval(D_el1, Trans_el1, eip_el1);
	  vN->Eval(tN_el1, Trans_el1, eip_el1);

	  uD->Eval(bcEval, Trans_el1, eip_el1, D_el1);
	  ///
	    
	  el.CalcShape(eip_el1, shape_el1);
	  
	  el2.CalcShape(eip_el2, shape_el2);

	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);

	  double nTildaDotN = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  for (int s = 0; s < dim; s++){
	    nTildaDotN += (nor(s) / nor_norm) * tN_el1(s);
	  }
	
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{	      
		  elvect(i + vd * dofs_cnt) += bcEval(vd) * shape_el1(i) * volumeFraction_el1 * ip_f.weight * nTildaDotN * nor_norm;
		  elvect(i + vd * dofs_cnt + dim * dofs_cnt) -= bcEval(vd) * shape_el2(i) * volumeFraction_el2 * ip_f.weight * nTildaDotN * nor_norm;
		}
	    }
	}
    }
    else{
      const int dim = el.GetDim();
      const int dofs_cnt = el.GetDof();
      elvect.SetSize(2*dofs_cnt*dim);
      elvect = 0.0;
    }
  }

}
