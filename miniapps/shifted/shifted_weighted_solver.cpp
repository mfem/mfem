// Copyright (c) 20A17, Lawrence Livermore National Security, LLC. Produced at
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

  void WeightedShiftedStrainBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
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
      DenseMatrix dshape_el1(dofs_cnt,dim), dshape_ps_el1(dofs_cnt,dim), adjJ_el1(dim), gradUResDirD_el1(dofs_cnt), taylorExp_el1(dofs_cnt), dshape_el2(dofs_cnt,dim), dshape_ps_el2(dofs_cnt,dim), adjJ_el2(dim), gradUResDirD_el2(dofs_cnt), taylorExp_el2(dofs_cnt);
      Vector shape_el1(dofs_cnt), gradURes_el1(dofs_cnt), gradUResD_el1(dofs_cnt), shape_el2(dofs_cnt), gradURes_el2(dofs_cnt), gradUResD_el2(dofs_cnt);
      shape_el1 = 0.0;
      gradURes_el1 = 0.0;
      gradUResDirD_el1 = 0.0;
      dshape_el1 = 0.0;
      dshape_ps_el1 = 0.0;
      adjJ_el1 = 0.0;
      gradUResD_el1 = 0.0;
      taylorExp_el1 = 0.0;

      shape_el2 = 0.0;
      gradURes_el2 = 0.0;
      gradUResDirD_el2 = 0.0;
      dshape_el2 = 0.0;
      dshape_ps_el2 = 0.0;
      adjJ_el2 = 0.0;
      gradUResD_el2 = 0.0;
      taylorExp_el2 = 0.0;

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
	  gradURes_el1 = 0.0;
	  gradUResDirD_el1 = 0.0;
	  dshape_el1 = 0.0;
	  dshape_ps_el1 = 0.0;
	  adjJ_el1 = 0.0;
	  gradUResD_el1 = 0.0;
	  taylorExp_el1 = 0.0;
	  
	  shape_el2 = 0.0;
	  gradURes_el2 = 0.0;
	  gradUResDirD_el2 = 0.0;
	  dshape_el2 = 0.0;
	  dshape_ps_el2 = 0.0;
	  adjJ_el2 = 0.0;
	  gradUResD_el2 = 0.0;
	  taylorExp_el2 = 0.0;

	  nor = 0.0;

	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);

	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);

	  /////
	  Vector D_el1(dim);
	  Vector tN_el1(dim);
	  D_el1 = 0.0;
	  tN_el1 = 0.0;
	  Vector x_eip_el1(3);
	  x_eip_el1 = 0.0;
	  Trans_el1.Transform(eip_el1,x_eip_el1);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el1,D_el1,tN_el1);
	  /////
	  
	  fe.CalcShape(eip_el1, shape_el1);
	  fe.CalcDShape(eip_el1, dshape_el1);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ_el1);
	  Mult(dshape_el1, adjJ_el1, dshape_ps_el1);
	  dshape_ps_el1.Mult(nor,gradURes_el1);
	    
	  for (int k = 0; k < dofs_cnt; k++){
	    for (int s = 0; s < dofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el1(s,k) += nodalGrad_el1(k + j * dofs_cnt, s) * D_el1(j);
	      }
	    }
	  }

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
	  for ( int p = 1; p < nTerms; p++){
	    dummy_tmp_el1 = 0.0;
	    taylorExp_el1 = 0.0;
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int s = 0; s < dofs_cnt; s++){
		for (int r = 0; r < dofs_cnt; r++){
		  taylorExp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s) * (1.0/factorial(p+1));
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
	  
	  ////
	  shape_el1 += gradUResD_el1;
	  //

	  /////
	  Vector D_el2(dim);
	  Vector tN_el2(dim);
	  D_el2 = 0.0;
	  tN_el2 = 0.0;
	  Vector x_eip_el2(3);
	  x_eip_el2 = 0.0;
	  Trans_el2.Transform(eip_el2,x_eip_el2);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el2,D_el2,tN_el2);
	  /////
	  
	  fe2.CalcShape(eip_el2, shape_el2);
	  fe2.CalcDShape(eip_el2, dshape_el2);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ_el2);
	  Mult(dshape_el2, adjJ_el2, dshape_ps_el2);
	  dshape_ps_el2.Mult(nor,gradURes_el2);
	    
	  for (int k = 0; k < dofs_cnt; k++){
	    for (int s = 0; s < dofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el2(s,k) += nodalGrad_el2(k + j * dofs_cnt, s) * D_el2(j);
	      }
	    }
	  }

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
	  for ( int p = 1; p < nTerms; p++){
	    dummy_tmp_el2 = 0.0;
	    taylorExp_el2 = 0.0;
	    for (int k = 0; k < dofs_cnt; k++){
	      for (int s = 0; s < dofs_cnt; s++){
		for (int r = 0; r < dofs_cnt; r++){
		  taylorExp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s) * (1.0/factorial(p+1));
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
	  
	  ////
	  shape_el2 += gradUResD_el2;
	  //

	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;

	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      elmat(i + vd * dofs_cnt, j + vd * dofs_cnt) -= gamma_1 * (volumeFraction_el1 - volumeFraction_el2) * shape_el1(j) * gamma_1 * gradURes_el1(i) * Mu * ip_f.weight / Tr.Elem1->Weight();
		      elmat(i + vd * dofs_cnt, j + vd * dofs_cnt + dim * dofs_cnt) -= gamma_2 * (volumeFraction_el1 - volumeFraction_el2) * shape_el2(j) * gamma_1 * gradURes_el1(i) * Mu * ip_f.weight / Tr.Elem1->Weight();
		      
		      elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + vd * dofs_cnt) -= gamma_1 * (volumeFraction_el1 - volumeFraction_el2) * shape_el1(j) * gamma_2 * gradURes_el2(i) * Mu * ip_f.weight / Tr.Elem2->Weight();
		      elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + vd * dofs_cnt + dim * dofs_cnt) -= gamma_2 * (volumeFraction_el1 - volumeFraction_el2) * shape_el2(j) * gamma_2 * gradURes_el2(i) * Mu * ip_f.weight / Tr.Elem2->Weight();
		      for (int md = 0; md < dim; md++) // Velocity components.
			{
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt) -= gamma_1 * (volumeFraction_el1 - volumeFraction_el2) * shape_el1(j) * gamma_1 * nor(vd) * dshape_ps_el1(i,md) * Mu * ip_f.weight / Tr.Elem1->Weight();
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= gamma_2 * (volumeFraction_el1 - volumeFraction_el2) * shape_el2(j) * gamma_1 * nor(vd) * dshape_ps_el1(i,md) * Mu * ip_f.weight / Tr.Elem1->Weight();

			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt) -= gamma_1 * (volumeFraction_el1 - volumeFraction_el2) * shape_el1(j) * gamma_2 * nor(vd) * dshape_ps_el2(i,md) * Mu * ip_f.weight / Tr.Elem2->Weight();
			  elmat(i + vd * dofs_cnt + dim * dofs_cnt, j + md * dofs_cnt + dim * dofs_cnt) -= gamma_2 * (volumeFraction_el1 - volumeFraction_el2) * shape_el2(j) * gamma_2 * nor(vd) * dshape_ps_el2(i,md) * Mu * ip_f.weight / Tr.Elem2->Weight();
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
  
  void WeightedShiftedStrainBoundaryForceTransposeIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
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
  
  void WeightedShiftedPressureBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1 ,
								  const FiniteElement &trial_fe2,
								  const FiniteElement &test_fe1,
								  const FiniteElement &test_fe2,
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
      const int dim = test_fe1.GetDim();
      const int testdofs_cnt = test_fe1.GetDof();
      const int trialdofs_cnt = trial_fe1.GetDof();
      
      elmat.SetSize(2 * testdofs_cnt, 2 * trialdofs_cnt * dim);
      elmat = 0.0;
      
      Vector nor(dim);
      Vector te_shape_el1(testdofs_cnt),tr_shape_el1(trialdofs_cnt), gradUResD_el1(trialdofs_cnt), te_shape_el2(testdofs_cnt),tr_shape_el2(trialdofs_cnt), gradUResD_el2(trialdofs_cnt);
      DenseMatrix dshape_el1(trialdofs_cnt,dim), dshape_ps_el1(trialdofs_cnt,dim), adjJ_el1(dim), gradUResDirD_el1(trialdofs_cnt), taylorExp_el1(trialdofs_cnt), dshape_el2(trialdofs_cnt,dim), dshape_ps_el2(trialdofs_cnt,dim), adjJ_el2(dim), gradUResDirD_el2(trialdofs_cnt), taylorExp_el2(trialdofs_cnt);

      nor = 0.0;
      
      te_shape_el1 = 0.0;
      tr_shape_el1 = 0.0;
      gradUResD_el1 = 0.0;
      gradUResDirD_el1 = 0.0;
      dshape_el1 = 0.0;
      dshape_ps_el1 = 0.0;
      adjJ_el1 = 0.0;
      taylorExp_el1 = 0.0;

      te_shape_el2 = 0.0;
      tr_shape_el2 = 0.0;
      gradUResD_el2 = 0.0;
      gradUResDirD_el2 = 0.0;
      dshape_el2 = 0.0;
      dshape_ps_el2 = 0.0;
      adjJ_el2 = 0.0;
      taylorExp_el2 = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 10 * max(test_fe1.GetOrder(), 1);
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
	  te_shape_el1 = 0.0;
	  tr_shape_el1 = 0.0;
	  gradUResD_el1 = 0.0;
	  gradUResDirD_el1 = 0.0;
	  dshape_el1 = 0.0;
	  dshape_ps_el1 = 0.0;
	  adjJ_el1 = 0.0;
	  taylorExp_el1 = 0.0;
	  
	  te_shape_el2 = 0.0;
	  tr_shape_el2 = 0.0;
	  gradUResD_el2 = 0.0;
	  gradUResDirD_el2 = 0.0;
	  dshape_el2 = 0.0;
	  dshape_ps_el2 = 0.0;
	  adjJ_el2 = 0.0;
	  taylorExp_el2 = 0.0;
	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  
	  test_fe1.CalcShape(eip_el1, te_shape_el1);
	  trial_fe1.CalcShape(eip_el1, tr_shape_el1);

	  /////
	  Vector D_el1(dim);
	  Vector tN_el1(dim);
	  D_el1 = 0.0;
	  tN_el1 = 0.0;
	  Vector x_eip_el1(3);
	  x_eip_el1 = 0.0;
	  Trans_el1.Transform(eip_el1,x_eip_el1);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el1,D_el1,tN_el1);
	  /////

	  for (int k = 0; k < trialdofs_cnt; k++){
	    for (int s = 0; s < trialdofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el1(s,k) += nodalGrad_el1(k + j * trialdofs_cnt, s) * D_el1(j);
	      }
	    }
	  }

	  DenseMatrix tmp_el1(trialdofs_cnt);
	  DenseMatrix dummy_tmp_el1(trialdofs_cnt);
	  tmp_el1 = gradUResDirD_el1;
	  taylorExp_el1 = gradUResDirD_el1;
	  dummy_tmp_el1 = 0.0;
	  for (int k = 0; k < trialdofs_cnt; k++){
	    for (int s = 0; s < trialdofs_cnt; s++){
	      gradUResD_el1(k) += taylorExp_el1(k,s) * tr_shape_el1(s);  
	    }
	  }
	  for ( int p = 1; p < nTerms; p++){
	    dummy_tmp_el1 = 0.0;
	    taylorExp_el1 = 0.0;
	    for (int k = 0; k < trialdofs_cnt; k++){
	      for (int s = 0; s < trialdofs_cnt; s++){
		for (int r = 0; r < trialdofs_cnt; r++){
		  taylorExp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s) * (1.0/factorial(p+1));
		  dummy_tmp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s);
		}
	      }
	    }
	    tmp_el1 = dummy_tmp_el1;
	    for (int k = 0; k < trialdofs_cnt; k++){
	      for (int s = 0; s < trialdofs_cnt; s++){
		gradUResD_el1(k) += taylorExp_el1(k,s) * tr_shape_el1(s);  
	      }
	    }
	  }
	  
	  
	  tr_shape_el1 += gradUResD_el1;
	  //
	  
	  test_fe2.CalcShape(eip_el2, te_shape_el2);
	  trial_fe2.CalcShape(eip_el2, tr_shape_el2);

	  /////
	  Vector D_el2(dim);
	  Vector tN_el2(dim);
	  D_el2 = 0.0;
	  tN_el2 = 0.0;
	  Vector x_eip_el2(3);
	  x_eip_el2 = 0.0;
	  Trans_el2.Transform(eip_el2,x_eip_el2);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el2,D_el2,tN_el2);
	  /////

	  for (int k = 0; k < trialdofs_cnt; k++){
	    for (int s = 0; s < trialdofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el2(s,k) += nodalGrad_el2(k + j * trialdofs_cnt, s) * D_el2(j);
	      }
	    }
	  }

	  DenseMatrix tmp_el2(trialdofs_cnt);
	  DenseMatrix dummy_tmp_el2(trialdofs_cnt);
	  tmp_el2 = gradUResDirD_el2;
	  taylorExp_el2 = gradUResDirD_el2;
	  dummy_tmp_el2 = 0.0;
	  for (int k = 0; k < trialdofs_cnt; k++){
	    for (int s = 0; s < trialdofs_cnt; s++){
	      gradUResD_el2(k) += taylorExp_el2(k,s) * tr_shape_el2(s);  
	    }
	  }
	  for ( int p = 1; p < nTerms; p++){
	    dummy_tmp_el2 = 0.0;
	    taylorExp_el2 = 0.0;
	    for (int k = 0; k < trialdofs_cnt; k++){
	      for (int s = 0; s < trialdofs_cnt; s++){
		for (int r = 0; r < trialdofs_cnt; r++){
		  taylorExp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s) * (1.0/factorial(p+1));
		  dummy_tmp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s);
		}
	      }
	    }
	    tmp_el2 = dummy_tmp_el2;
	    for (int k = 0; k < trialdofs_cnt; k++){
	      for (int s = 0; s < trialdofs_cnt; s++){
		gradUResD_el2(k) += taylorExp_el2(k,s) * tr_shape_el2(s);  
	      }
	    }
	  }
	  
	  //
	  tr_shape_el2 += gradUResD_el2;
	  //

	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;

	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int j = 0; j < trialdofs_cnt; j++)
		{
		  for (int vd = 0; vd < dim; vd++) // Velocity components.
		    {
		      elmat(i, j + vd * trialdofs_cnt) -= gamma_1 * tr_shape_el1(j) *  (volumeFraction_el1 - volumeFraction_el2) * nor(vd) * gamma_1 * te_shape_el1(i) * ip_f.weight;
		      elmat(i, j + vd * trialdofs_cnt + dim * trialdofs_cnt) -= gamma_2 * tr_shape_el2(j) *  (volumeFraction_el1 - volumeFraction_el2) * nor(vd) * gamma_1 * te_shape_el1(i) * ip_f.weight;
		      elmat(i + testdofs_cnt, j + vd * trialdofs_cnt) -= gamma_1 * tr_shape_el1(j) *  (volumeFraction_el1 - volumeFraction_el2) * nor(vd) * gamma_2 * te_shape_el2(i) * ip_f.weight;
		      elmat(i + testdofs_cnt, j + vd * trialdofs_cnt + dim * trialdofs_cnt) -= gamma_2 * tr_shape_el2(j) *  (volumeFraction_el1 - volumeFraction_el2) * nor(vd) * gamma_2 * te_shape_el2(i) * ip_f.weight;
		    }
		}
	    }
	}
    }
    else{
      const int dim = test_fe1.GetDim();
      const int testdofs_cnt = test_fe1.GetDof();
      const int trialdofs_cnt = trial_fe1.GetDof();
      const int testdofs2_cnt = test_fe2.GetDof();
      const int trialdofs2_cnt = trial_fe2.GetDof();

      elmat.SetSize(2 * testdofs_cnt, 2 * trialdofs_cnt * dim);
      elmat = 0.0;
    }
 
  }

  void WeightedShiftedPressureBoundaryForceTransposeIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
									   const FiniteElement &trial_fe2,
									   const FiniteElement &test_fe1,
									   const FiniteElement &test_fe2,
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
  
      const int dim = test_fe1.GetDim();
      const int testdofs_cnt = test_fe1.GetDof();
      const int trialdofs_cnt = trial_fe1.GetDof();
      const int testdofs2_cnt = test_fe2.GetDof();
      const int trialdofs2_cnt = trial_fe2.GetDof();
      
      elmat.SetSize((testdofs_cnt+testdofs2_cnt)*dim, trialdofs_cnt+trialdofs2_cnt);
      elmat = 0.0;
      
      Vector nor(dim);
      Vector te_shape_el1(testdofs_cnt), tr_shape_el1(trialdofs_cnt), te_shape_el2(testdofs_cnt), tr_shape_el2(trialdofs_cnt);
      
      te_shape_el1 = 0.0;
      tr_shape_el1 = 0.0;

      te_shape_el2 = 0.0;
      tr_shape_el2 = 0.0;

      nor = 0.0;
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 10 * max(test_fe1.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
      
      for (int q = 0; q < nqp_face; q++)
	{
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);

	  te_shape_el1 = 0.0;
	  tr_shape_el1 = 0.0;
	  
	  te_shape_el2 = 0.0;
	  tr_shape_el2 = 0.0;

	  trial_fe1.CalcShape(eip_el1, tr_shape_el1);
	  trial_fe2.CalcShape(eip_el2, tr_shape_el2);
	  test_fe1.CalcShape(eip_el1, te_shape_el1);
	  test_fe2.CalcShape(eip_el2, te_shape_el2);

	  //
	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;
	  //
	  
	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < trialdofs_cnt; j++)
		    {
		      elmat(i + vd * testdofs_cnt, j) += gamma_1 * tr_shape_el1(j) * nor(vd) * volumeFraction_el1 * te_shape_el1(i) * ip_f.weight;
		      elmat(i + vd * testdofs_cnt, j + trialdofs_cnt) += gamma_2 * tr_shape_el2(j) * nor(vd) * volumeFraction_el1 * te_shape_el1(i) * ip_f.weight;
		      elmat(i + vd * testdofs_cnt + dim * testdofs_cnt, j) -= gamma_1 * tr_shape_el1(j) * nor(vd) * volumeFraction_el2 * te_shape_el2(i) * ip_f.weight;
		      elmat(i + vd * testdofs_cnt + dim * testdofs_cnt, j + trialdofs_cnt) -= gamma_2 * tr_shape_el2(j) * nor(vd) * volumeFraction_el2 * te_shape_el2(i) * ip_f.weight;
		    }
		}
	    }
	}
    }
    else{
      const int dim = test_fe1.GetDim();
      const int testdofs_cnt = test_fe1.GetDof();
      const int trialdofs_cnt = trial_fe1.GetDof();
      const int testdofs2_cnt = test_fe2.GetDof();
      const int trialdofs2_cnt = trial_fe2.GetDof();
      
      elmat.SetSize((testdofs_cnt+testdofs2_cnt)*dim, trialdofs_cnt+trialdofs2_cnt);
      elmat = 0.0;
    }
  }

  void WeightedShiftedVelocityPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
							    const FiniteElement &fe2,
							    FaceElementTransformations &Tr,
							    DenseMatrix &elmat)
  {
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
    if ( (elem1_inside && elem2_cut) || (elem1_cut && elem2_inside) ||  (elem1_cut && elem2_cut) || (elem1_cut && elem2_outside) ||  (elem1_outside && elem2_cut) ) {
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
      DenseMatrix dshape_el1(h1dofs_cnt,dim), dshape_ps_el1(h1dofs_cnt,dim), adjJ_el1(dim), gradUResDirD_el1(h1dofs_cnt), taylorExp_el1(h1dofs_cnt), dshape_el2(h1dofs_cnt,dim), dshape_ps_el2(h1dofs_cnt,dim), adjJ_el2(dim), gradUResDirD_el2(h1dofs_cnt), taylorExp_el2(h1dofs_cnt);
      Vector shape_el1(h1dofs_cnt), shape_test_el1(h1dofs_cnt), nor(dim), gradURes_el1(h1dofs_cnt), gradUResD_el1(h1dofs_cnt), shape_el2(h1dofs_cnt), shape_test_el2(h1dofs_cnt),  gradURes_el2(h1dofs_cnt), gradUResD_el2(h1dofs_cnt);
      
      nor = 0.0;
 
      shape_el1 = 0.0;
      shape_test_el1 = 0.0;
      dshape_el1 = 0.0;
      dshape_ps_el1 = 0.0;
      adjJ_el1 = 0.0;
      gradURes_el1 = 0.0;
      gradUResDirD_el1 = 0.0; 
      gradUResD_el1 = 0.0;
      taylorExp_el1 = 0.0;

      shape_el2 = 0.0;
      shape_test_el2 = 0.0;
      dshape_el2 = 0.0;
      dshape_ps_el2 = 0.0;
      adjJ_el2 = 0.0;
      gradURes_el2 = 0.0;
      gradUResDirD_el2 = 0.0; 
      gradUResD_el2 = 0.0;
      taylorExp_el2 = 0.0;

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
	  shape_test_el1 = 0.0;
	  dshape_el1 = 0.0;
	  dshape_ps_el1 = 0.0;
	  adjJ_el1 = 0.0;
	  gradURes_el1 = 0.0;
	  gradUResDirD_el1 = 0.0; 
	  gradUResD_el1 = 0.0;
	  taylorExp_el1 = 0.0;
	  
	  shape_el2 = 0.0;
	  shape_test_el2 = 0.0;
	  dshape_el2 = 0.0;
	  dshape_ps_el2 = 0.0;
	  adjJ_el2 = 0.0;
	  gradURes_el2 = 0.0;
	  gradUResDirD_el2 = 0.0; 
	  gradUResD_el2 = 0.0;
	  taylorExp_el2 = 0.0;

	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  
	  fe.CalcShape(eip_el1, shape_el1);
	  fe.CalcShape(eip_el1, shape_test_el1);

	  /////
	  Vector D_el1(dim);
	  Vector tN_el1(dim);
	  D_el1 = 0.0;
	  tN_el1 = 0.0;
	  Vector x_eip_el1(3);
	  x_eip_el1 = 0.0;
	  Trans_el1.Transform(eip_el1,x_eip_el1);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el1,D_el1,tN_el1);
	  /////

	  for (int k = 0; k < h1dofs_cnt; k++){
	    for (int s = 0; s < h1dofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el1(s,k) += nodalGrad_el1(k + j * h1dofs_cnt, s) * D_el1(j);
	      }
	    }
	  }

	  DenseMatrix tmp_el1(h1dofs_cnt);
	  DenseMatrix dummy_tmp_el1(h1dofs_cnt);
	  tmp_el1 = gradUResDirD_el1;
	  taylorExp_el1 = gradUResDirD_el1;
	  dummy_tmp_el1 = 0.0;
	  for (int k = 0; k < h1dofs_cnt; k++){
	    for (int s = 0; s < h1dofs_cnt; s++){
	      gradUResD_el1(k) += taylorExp_el1(k,s) * shape_el1(s);  
	    }
	  }
	  Vector test_gradUResD_el1(h1dofs_cnt);
	  test_gradUResD_el1 = gradUResD_el1;
	  
	  for ( int p = 1; p < nTerms; p++){
	    dummy_tmp_el1 = 0.0;
	    taylorExp_el1 = 0.0;
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int s = 0; s < h1dofs_cnt; s++){
		for (int r = 0; r < h1dofs_cnt; r++){
		  taylorExp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s) * (1.0/factorial(p+1));
		  dummy_tmp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s);
		}
	      }
	    }
	    tmp_el1 = dummy_tmp_el1;
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int s = 0; s < h1dofs_cnt; s++){
		gradUResD_el1(k) += taylorExp_el1(k,s) * shape_el1(s);  
	      }
	    }
	  }
	  
	  ////
	  shape_el1 += gradUResD_el1;
	  //
	  if (fullPenalty){
	    shape_test_el1 += gradUResD_el1;
	  }
	  else{
	    shape_test_el1 += test_gradUResD_el1;
	  }

	  fe2.CalcShape(eip_el2, shape_el2);
	  fe2.CalcShape(eip_el2, shape_test_el2);

	  /////
	  Vector D_el2(dim);
	  Vector tN_el2(dim);
	  D_el2 = 0.0;
	  tN_el2 = 0.0;
	  Vector x_eip_el2(3);
	  x_eip_el2 = 0.0;
	  Trans_el2.Transform(eip_el2,x_eip_el2);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el2,D_el2,tN_el2);
	  /////

	  for (int k = 0; k < h1dofs_cnt; k++){
	    for (int s = 0; s < h1dofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el2(s,k) += nodalGrad_el2(k + j * h1dofs_cnt, s) * D_el2(j);
	      }
	    }
	  }

	  DenseMatrix tmp_el2(h1dofs_cnt);
	  DenseMatrix dummy_tmp_el2(h1dofs_cnt);
	  tmp_el2 = gradUResDirD_el2;
	  taylorExp_el2 = gradUResDirD_el2;
	  dummy_tmp_el2 = 0.0;
	  for (int k = 0; k < h1dofs_cnt; k++){
	    for (int s = 0; s < h1dofs_cnt; s++){
	      gradUResD_el2(k) += taylorExp_el2(k,s) * shape_el2(s);  
	    }
	  }
	  Vector test_gradUResD_el2(h1dofs_cnt);
	  test_gradUResD_el2 = gradUResD_el2;
	  
	  for ( int p = 1; p < nTerms; p++){
	    dummy_tmp_el2 = 0.0;
	    taylorExp_el2 = 0.0;
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int s = 0; s < h1dofs_cnt; s++){
		for (int r = 0; r < h1dofs_cnt; r++){
		  taylorExp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s) * (1.0/factorial(p+1));
		  dummy_tmp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s);
		}
	      }
	    }
	    tmp_el2 = dummy_tmp_el2;
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int s = 0; s < h1dofs_cnt; s++){
		gradUResD_el2(k) += taylorExp_el2(k,s) * shape_el2(s);  
	      }
	    }
	  }
	  
	  ////
	  shape_el2 += gradUResD_el2;
	  //
	  if (fullPenalty){
	    shape_test_el2 += gradUResD_el2;
	  }
	  else{
	    shape_test_el2 += test_gradUResD_el2;
	  }

	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);

	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;
	  double weighted_h = nor_norm * ( gamma_1 * (2.0 * Mu / Tr.Elem1->Weight()) + gamma_2 * (2.0 * Mu / Tr.Elem2->Weight()) ) * penaltyParameter * std::abs(volumeFraction_el1-volumeFraction_el2);

	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt) += 2.0 * shape_test_el1(i) * shape_el1(j) * ip_f.weight * nor_norm * weighted_h * gamma_1 * gamma_1;
		      elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) += 2.0 * shape_test_el1(i) * shape_el2(j) * ip_f.weight * nor_norm * weighted_h * gamma_1 * gamma_2;
		      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt) += 2.0 * shape_test_el2(i) * shape_el1(j) * ip_f.weight * nor_norm * weighted_h * gamma_2 * gamma_1;
		      elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + vd * h1dofs_cnt + dim * h1dofs_cnt) += 2.0 * shape_test_el2(i) * shape_el2(j) * ip_f.weight * nor_norm * weighted_h * gamma_2 * gamma_2;
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

  void WeightedShiftedStrainNitscheBCForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
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
      
      DenseMatrix dshape_el1(dofs_cnt,dim), dshape_ps_el1(dofs_cnt,dim), adjJ_el1(dim),  dshape_el2(dofs_cnt,dim), dshape_ps_el2(dofs_cnt,dim), adjJ_el2(dim);
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
	  gradURes_el1 = 0.0;
	  dshape_el1 = 0.0;
	  dshape_ps_el1 = 0.0;
	  adjJ_el1 = 0.0;
	  
	  shape_el2 = 0.0;
	  gradURes_el2 = 0.0;
	  dshape_el2 = 0.0;
	  dshape_ps_el2 = 0.0;
	  adjJ_el2 = 0.0;
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
	  Vector x_eip_el1(3);
	  Trans_el1.Transform(eip_el1,x_eip_el1);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el1,D_el1,tN_el1);
	  uD->Eval(bcEval, Trans_el1, eip_el1, D_el1);
	  //  uD->Eval(bcEval, Trans_el1, eip);
	  ///
	    
	  el.CalcShape(eip_el1, shape_el1);
	  el.CalcDShape(eip_el1, dshape_el1);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ_el1);
	  Mult(dshape_el1, adjJ_el1, dshape_ps_el1);
	  dshape_ps_el1.Mult(nor,gradURes_el1);
	  
	  el2.CalcShape(eip_el2, shape_el2);
	  el2.CalcDShape(eip_el2, dshape_el2);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ_el2);
	  Mult(dshape_el2, adjJ_el2, dshape_ps_el2);
	  dshape_ps_el2.Mult(nor,gradURes_el2);

	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);

	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;

	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{	      
		  elvect(i + vd * dofs_cnt) -= bcEval(vd) * gradURes_el1(i) * Mu * gamma_1 * (volumeFraction_el1 - volumeFraction_el2) * ip_f.weight / Tr.Elem1->Weight();
		  elvect(i + vd * dofs_cnt + dim * dofs_cnt) -= bcEval(vd) * gradURes_el2(i) * Mu * gamma_2 * (volumeFraction_el1 - volumeFraction_el2) * ip_f.weight / Tr.Elem2->Weight();
		  for (int md = 0; md < dim; md++) // Velocity components.
		    {	      		  
		      elvect(i + vd * dofs_cnt) -= bcEval(md) * nor(vd) * dshape_ps_el1(i,md) * Mu * gamma_1 * (volumeFraction_el1 - volumeFraction_el2) * ip_f.weight / Tr.Elem1->Weight();
		      elvect(i + vd * dofs_cnt + dim * dofs_cnt) -= bcEval(md) * nor(vd) * dshape_ps_el2(i,md) * Mu * gamma_2 * (volumeFraction_el1 - volumeFraction_el2) * ip_f.weight / Tr.Elem2->Weight();
		    }
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

  void WeightedShiftedPressureNitscheBCForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
								       const FiniteElement &el2,
								       FaceElementTransformations &Tr,
								       Vector &elvect)
  {
    Array<int> &elemStatus = analyticalSurface->GetElement_Status();

    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    //  std::cout << " I AM IN ASSEMBLE FACE " << std::endl;
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
      const int testdofs_cnt = el.GetDof();
	
      elvect.SetSize(2*testdofs_cnt);
      elvect = 0.0;
	
      Vector nor(dim), bcEval(dim);
      Vector te_shape_el1(testdofs_cnt), te_shape_el2(testdofs_cnt);
	
      te_shape_el1 = 0.0;
      te_shape_el2 = 0.0;
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
  	
      for (int q = 0; q < nqp_face; q++)
	{
	  	
	  te_shape_el1 = 0.0;
	  te_shape_el2 = 0.0;
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
	  Vector x_eip_el1(3);
	  Trans_el1.Transform(eip_el1,x_eip_el1);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el1,D_el1,tN_el1);
	  uD->Eval(bcEval, Trans_el1, eip_el1, D_el1);
	  //  uD->Eval(bcEval, Trans_el1, eip);
	  ///
	    
	  el.CalcShape(eip_el1, te_shape_el1);
	  el2.CalcShape(eip_el2, te_shape_el2);

	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;

	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i) -= nor(vd) * bcEval(vd) * te_shape_el1(i) * (volumeFraction_el1 - volumeFraction_el2) * ip_f.weight * gamma_1;
		  elvect(i+testdofs_cnt) -= nor(vd) * bcEval(vd) * te_shape_el2(i) * (volumeFraction_el1 - volumeFraction_el2) * ip_f.weight * gamma_2;
		}
	    }
	}
    }
    else{
      const int dim = el.GetDim();
      const int dofs_cnt = el.GetDof();
      elvect.SetSize(2*dofs_cnt);
      elvect = 0.0;
    }
 
  }

  void WeightedShiftedVelocityBCPenaltyIntegrator::AssembleRHSElementVect(const FiniteElement &el,
								  const FiniteElement &el2,
								  FaceElementTransformations &Tr,
								  Vector &elvect)
  {

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
    if ( (elem1_inside && elem2_cut) || (elem1_cut && elem2_inside) ||  (elem1_cut && elem2_cut) || (elem1_cut && elem2_outside) ||  (elem1_outside && elem2_cut) ) {
      const int dim = el.GetDim();
      const int h1dofs_cnt = el.GetDof();
      elvect.SetSize(2*h1dofs_cnt*dim);
      elvect = 0.0;
      DenseMatrix dshape_el1(h1dofs_cnt,dim), dshape_ps_el1(h1dofs_cnt,dim), adjJ_el1(dim), gradUResDirD_el1(h1dofs_cnt), taylorExp_el1(h1dofs_cnt), dshape_el2(h1dofs_cnt,dim), dshape_ps_el2(h1dofs_cnt,dim), adjJ_el2(dim), gradUResDirD_el2(h1dofs_cnt), taylorExp_el2(h1dofs_cnt);
      Vector shape_el1(h1dofs_cnt), nor(dim), gradURes_el1(h1dofs_cnt), gradUResD_el1(h1dofs_cnt), shape_el2(h1dofs_cnt),  gradURes_el2(h1dofs_cnt), gradUResD_el2(h1dofs_cnt),  bcEval(dim);

      nor = 0.0;
      bcEval = 0.0;
      
      shape_el1 = 0.0;
      dshape_el1 = 0.0;
      dshape_ps_el1 = 0.0;
      adjJ_el1 = 0.0;
      gradURes_el1 = 0.0;
      gradUResDirD_el1 = 0.0; 
      gradUResD_el1 = 0.0;
      taylorExp_el1 = 0.0;

      shape_el2 = 0.0;
      dshape_el2 = 0.0;
      dshape_ps_el2 = 0.0;
      adjJ_el2 = 0.0;
      gradURes_el2 = 0.0;
      gradUResDirD_el2 = 0.0; 
      gradUResD_el2 = 0.0;
      taylorExp_el2 = 0.0;

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
      DenseMatrix nodalGrad_el1;
      DenseMatrix nodalGrad_el2;
      el.ProjectGrad(el,Trans_el1,nodalGrad_el1);
      el2.ProjectGrad(el2,Trans_el2,nodalGrad_el2);

      for (int q = 0; q < nqp_face; q++)
	{
	  nor = 0.0;
	  bcEval = 0.0;
	  
	  shape_el1 = 0.0;
	  dshape_el1 = 0.0;
	  dshape_ps_el1 = 0.0;
	  adjJ_el1 = 0.0;
	  gradURes_el1 = 0.0;
	  gradUResDirD_el1 = 0.0; 
	  gradUResD_el1 = 0.0;
	  taylorExp_el1 = 0.0;
	  
	  shape_el2 = 0.0;
	  dshape_el2 = 0.0;
	  dshape_ps_el2 = 0.0;
	  adjJ_el2 = 0.0;
	  gradURes_el2 = 0.0;
	  gradUResDirD_el2 = 0.0; 
	  gradUResD_el2 = 0.0;
	  taylorExp_el2 = 0.0;

	  const IntegrationPoint &ip_f = ir->IntPoint(q);  
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	  const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	    
	  el.CalcShape(eip_el1, shape_el1);
	  el2.CalcShape(eip_el2, shape_el2);

	  /////
	  Vector D_el1(dim);
	  Vector tN_el1(dim);
	  D_el1 = 0.0;
	  tN_el1 = 0.0;
	  Vector x_eip_el1(3);
	  x_eip_el1 = 0.0;
	  Trans_el1.Transform(eip_el1,x_eip_el1);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el1,D_el1,tN_el1);
	  uD->Eval(bcEval, Trans_el1, eip_el1, D_el1);
	  /////
	  

	  for (int k = 0; k < h1dofs_cnt; k++){
	    for (int s = 0; s < h1dofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el1(s,k) += nodalGrad_el1(k + j * h1dofs_cnt, s) * D_el1(j);
	      }
	    }
	  }


	  DenseMatrix tmp_el1(h1dofs_cnt);
	  DenseMatrix dummy_tmp_el1(h1dofs_cnt);
	  tmp_el1 = gradUResDirD_el1;
	  taylorExp_el1 = gradUResDirD_el1;
	  dummy_tmp_el1 = 0.0;
	  for (int k = 0; k < h1dofs_cnt; k++){
	    for (int s = 0; s < h1dofs_cnt; s++){
	      gradUResD_el1(k) += taylorExp_el1(k,s) * shape_el1(s);  
	    }
	  }
	  if (fullPenalty){
	    for ( int p = 1; p < nTerms; p++){
	      dummy_tmp_el1 = 0.0;
	      taylorExp_el1 = 0.0;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  for (int r = 0; r < h1dofs_cnt; r++){
		    taylorExp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s) * (1.0/factorial(p+1));
		    dummy_tmp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s);
		  }
		}
	      }
	      tmp_el1 = dummy_tmp_el1;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
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
	  D_el2 = 0.0;
	  tN_el2 = 0.0;
	  Vector x_eip_el2(3);
	  x_eip_el2 = 0.0;
	  Trans_el2.Transform(eip_el2,x_eip_el2);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip_el2,D_el2,tN_el2);
	  /////
	  

	  for (int k = 0; k < h1dofs_cnt; k++){
	    for (int s = 0; s < h1dofs_cnt; s++){
	      for (int j = 0; j < dim; j++){
		gradUResDirD_el2(s,k) += nodalGrad_el2(k + j * h1dofs_cnt, s) * D_el2(j);
	      }
	    }
	  }


	  DenseMatrix tmp_el2(h1dofs_cnt);
	  DenseMatrix dummy_tmp_el2(h1dofs_cnt);
	  tmp_el2 = gradUResDirD_el2;
	  taylorExp_el2 = gradUResDirD_el2;
	  dummy_tmp_el2 = 0.0;
	  for (int k = 0; k < h1dofs_cnt; k++){
	    for (int s = 0; s < h1dofs_cnt; s++){
	      gradUResD_el2(k) += taylorExp_el2(k,s) * shape_el2(s);  
	    }
	  }
	  if (fullPenalty){
	    for ( int p = 1; p < nTerms; p++){
	      dummy_tmp_el2 = 0.0;
	      taylorExp_el2 = 0.0;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  for (int r = 0; r < h1dofs_cnt; r++){
		    taylorExp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s) * (1.0/factorial(p+1));
		    dummy_tmp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s);
		  }
		}
	      }
	      tmp_el2 = dummy_tmp_el2;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  gradUResD_el2(k) += taylorExp_el2(k,s) * shape_el2(s);  
		}
	      }
	    }
	  }
	  
	  ////
	  shape_el2 += gradUResD_el2;
	  //
	  double Mu = mu->Eval(*Tr.Elem1, eip_el1);
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);

	  double volumeFraction_el1 = alpha->GetValue(Trans_el1, eip_el1);
	  double volumeFraction_el2 = alpha->GetValue(Trans_el2, eip_el2);
	  double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	  double gamma_1 =  volumeFraction_el1/sum_volFrac;
	  double gamma_2 =  volumeFraction_el2/sum_volFrac;
	  double weighted_h = nor_norm * ( gamma_1 * (2.0 * Mu / Tr.Elem1->Weight()) + gamma_2 * (2.0 * Mu / Tr.Elem2->Weight()) ) * penaltyParameter * std::abs(volumeFraction_el1-volumeFraction_el2);

	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i + vd * h1dofs_cnt) += 2.0 * weighted_h * shape_el1(i) * bcEval(vd) * ip_f.weight * nor_norm * gamma_1;
		  elvect(i + vd * h1dofs_cnt + dim * h1dofs_cnt) += 2.0 * weighted_h * shape_el2(i) * bcEval(vd) * ip_f.weight * nor_norm * gamma_2;
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
