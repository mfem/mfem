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

#include "shifted_solver.hpp"
#include <unordered_map>

namespace mfem
{
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

  void ShiftedStrainBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
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

    if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT) ){
      const int dim = fe.GetDim();
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim), ni(dim);
      DenseMatrix dshape(dofs_cnt,dim), dshape_ps(dofs_cnt,dim), adjJ(dim);
      Vector shape(dofs_cnt), gradURes(dofs_cnt), gradUResD(dofs_cnt), q_hess_dot_d(dofs_cnt);
      double w = 0.0;
      
      shape = 0.0;
      gradURes = 0.0;
      gradUResD = 0.0;
      dshape = 0.0;
      dshape_ps = 0.0;
      adjJ = 0.0;
      nor = 0.0;
      ni = 0.0;
      q_hess_dot_d = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();	
      for (int q = 0; q < nqp_face; q++)
	{
	  shape = 0.0;
	  gradURes = 0.0;
	  gradUResD = 0.0;
	  dshape = 0.0;
	  dshape_ps = 0.0;
	  adjJ = 0.0;
	  nor = 0.0;
	  ni = 0.0;
	  q_hess_dot_d = 0.0;
	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  
	  fe.CalcShape(eip, shape);
	  fe.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  w = ip_f.weight/Tr.Elem1->Weight();
	  dshape_ps.Mult(nor,gradURes);
	  
	  ////
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  x_eip = 0.0;
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  dshape_ps.Mult(D,gradUResD);
	  gradUResD /= Tr.Elem1->Weight();
	  // FIRST ORDER TAYLOR
	  shape += gradUResD;
	  //
	  int size = (dim*(dim+1))/2;
          DenseMatrix physical_hess(dofs_cnt,size);
          physical_hess = 0.0;
          fe.CalcPhysHessian2(Trans_el1,physical_hess);
          DenseMatrix adjusted_physical_hess(dofs_cnt,dim*dim);
          adjusted_physical_hess = 0.0;
          for (int s = 0; s < dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g <= l; g++){
                adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
                adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
              }
            }
          }
	  for (int s = 0; s < dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g < dim; g++){
                q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
              }
            }
          }
          q_hess_dot_d *= 1.0/2.0;
	  // SECOND ORDER TAYLOR
	  shape += q_hess_dot_d;
	  ///

	  double Mu = mu->Eval(*Tr.Elem1, eip);
	  ni.Set(w, nor);
	    
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      elmat(i + vd * dofs_cnt, j + vd * dofs_cnt) -= shape(j) * gradURes(i) * Mu * w;
		      for (int md = 0; md < dim; md++) // Velocity components.
			{
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt) -= shape(j) * ni(vd) * dshape_ps(i,md) * Mu;
			}
		    }
		}
	    }
	}
    }
    else if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE) ) {
      const int dim = fe2.GetDim();
      const int dofs_cnt = fe2.GetDof();
      
      int h1dofs_offset = fe.GetDof()*dim;
      
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim), ni(dim);
      DenseMatrix dshape(dofs_cnt,dim), dshape_ps(dofs_cnt,dim), adjJ(dim);
      Vector shape(dofs_cnt), gradURes(dofs_cnt), gradUResD(dofs_cnt), q_hess_dot_d(dofs_cnt);
      double w = 0.0;
      
      shape = 0.0;
      gradURes = 0.0;
      dshape = 0.0;
      dshape_ps = 0.0;
      adjJ = 0.0;
      nor = 0.0;
      ni = 0.0;
      gradUResD = 0.0;
      q_hess_dot_d = 0.0;
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(fe2.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement2Transformation();		
      for (int q = 0; q < nqp_face; q++)
	{
	  shape = 0.0;
	  gradURes = 0.0;
	  dshape = 0.0;
	  dshape_ps = 0.0;
	  adjJ = 0.0;
	  nor = 0.0;
	  ni = 0.0;
	  gradUResD = 0.0;
	  q_hess_dot_d = 0.0;
	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  nor *= -1.0;
	  
	  fe2.CalcShape(eip, shape);
	  fe2.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  w = ip_f.weight/Tr.Elem2->Weight();
	  dshape_ps.Mult(nor,gradURes);
	  
	  ////
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  x_eip = 0.0;
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  dshape_ps.Mult(D,gradUResD);
	  gradUResD /= Tr.Elem2->Weight();
	  // FIRST ORDER TAYLOR
	  shape += gradUResD;
	  //
	  int size = (dim*(dim+1))/2;
          DenseMatrix physical_hess(dofs_cnt,size);
          physical_hess = 0.0;
          fe2.CalcPhysHessian2(Trans_el1,physical_hess);
          DenseMatrix adjusted_physical_hess(dofs_cnt,dim*dim);
          adjusted_physical_hess = 0.0;
          for (int s = 0; s < dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g <= l; g++){
                adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
                adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
              }
            }
          }
	  for (int s = 0; s < dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g < dim; g++){
                q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
              }
            }
          }
          q_hess_dot_d *= 1.0/2.0;
	  // SECOND ORDER TAYLOR
	  shape += q_hess_dot_d;
	  ///
	  
	  double Mu = mu->Eval(*Tr.Elem2, eip);
	  ni.Set(w, nor);
	  
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      elmat(i + vd * dofs_cnt + h1dofs_offset, j + vd * dofs_cnt + h1dofs_offset) -= shape(j) * gradURes(i) * Mu * w;
		      for (int md = 0; md < dim; md++) // Velocity components.
			{
			  elmat(i + vd * dofs_cnt + h1dofs_offset, j + md * dofs_cnt + h1dofs_offset) -= shape(j) * ni(vd) * dshape_ps(i,md) * Mu;
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
  
  void ShiftedStrainBoundaryForceTransposeIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
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
    
    if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT) ){
      const int dim = fe.GetDim();
      const int dofs_cnt = fe.GetDof();
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
      
      Vector nor(dim), ni(dim);
      DenseMatrix dshape(dofs_cnt,dim), dshape_ps(dofs_cnt,dim), adjJ(dim);
      Vector shape(dofs_cnt), gradURes(dofs_cnt);
      double w = 0.0;
      
      shape = 0.0;
      gradURes = 0.0;
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      
      for (int q = 0; q < nqp_face; q++)
	{
	  gradURes = 0.0;
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  
	  fe.CalcShape(eip, shape);
	  fe.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  w = ip_f.weight/Tr.Elem1->Weight();
	  dshape_ps.Mult(nor,gradURes);
	  
	  double Mu = mu->Eval(*Tr.Elem1, eip);
	  ni.Set(w, nor);
	  
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      elmat(i + vd * dofs_cnt, j + vd * dofs_cnt) -= shape(i) * gradURes(j) * Mu * w;
		      for (int md = 0; md < dim; md++) // Velocity components.
			{
			  elmat(i + vd * dofs_cnt, j + md * dofs_cnt) -= shape(i) * ni(md) * dshape_ps(j,vd) * Mu;
			}
		    }
		}
	    }
	}
    }
    else if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE) ) {
      const int dim = fe2.GetDim();
      const int dofs_cnt = fe2.GetDof();
	
      int h1dofs_offset = dofs_cnt*dim;
	
      elmat.SetSize(2*dofs_cnt*dim);
      elmat = 0.0;
	
      Vector nor(dim), ni(dim);
      DenseMatrix dshape(dofs_cnt,dim), dshape_ps(dofs_cnt,dim), adjJ(dim);
      Vector shape(dofs_cnt), gradURes(dofs_cnt);
      double w = 0.0;
	
      shape = 0.0;
      gradURes = 0.0;
	
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(fe2.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
	
      for (int q = 0; q < nqp_face; q++)
	{
	  gradURes = 0.0;
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  nor *= -1.0;
	    
	  fe2.CalcShape(eip, shape);
	  fe2.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  w = ip_f.weight/Tr.Elem2->Weight();
	  dshape_ps.Mult(nor,gradURes);
	    
	  double Mu = mu->Eval(*Tr.Elem2, eip);
	  ni.Set(w, nor);
	    
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < dofs_cnt; j++)
		    {
		      elmat(i + vd * dofs_cnt + h1dofs_offset, j + vd * dofs_cnt + h1dofs_offset) -= shape(i) * gradURes(j) * Mu * w;
		      for (int md = 0; md < dim; md++) // Velocity components.
			{
			  elmat(i + vd * dofs_cnt + h1dofs_offset, j + md * dofs_cnt + h1dofs_offset) -= shape(i) * ni(md) * dshape_ps(j,vd) * Mu;
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
  
  void ShiftedPressureBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
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

    if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT) ){
      const int dim = test_fe1.GetDim();
      const int testdofs_cnt = test_fe1.GetDof();
      const int trialdofs_cnt = trial_fe1.GetDof();
      const int testdofs2_cnt = test_fe2.GetDof();
      const int trialdofs2_cnt = trial_fe2.GetDof();	
      
      elmat.SetSize(testdofs_cnt+testdofs2_cnt, (trialdofs_cnt+trialdofs2_cnt)*dim);
      elmat = 0.0;
      
      Vector nor(dim);
      Vector te_shape(testdofs_cnt),tr_shape(trialdofs_cnt), gradUResD(trialdofs_cnt), q_hess_dot_d(trialdofs_cnt);
      DenseMatrix dshape(trialdofs_cnt,dim), dshape_ps(trialdofs_cnt,dim), adjJ(dim);
      
      te_shape = 0.0;
      tr_shape = 0.0;
      gradUResD = 0.0;
      dshape = 0.0;
      dshape_ps = 0.0;
      adjJ = 0.0;
      q_hess_dot_d = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(test_fe1.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();	
      
      for (int q = 0; q < nqp_face; q++)
	{
	  gradUResD = 0.0;
	  tr_shape = 0.0;  
	  te_shape = 0.0;
	  gradUResD = 0.0;
	  dshape = 0.0;
	  dshape_ps = 0.0;
	  adjJ = 0.0;
	  q_hess_dot_d = 0.0;
	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  
	  test_fe1.CalcShape(eip, te_shape);
	  trial_fe1.CalcShape(eip, tr_shape);
	  
	  ////
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  x_eip = 0.0;
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  trial_fe1.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  dshape_ps.Mult(D,gradUResD);
	  gradUResD /= Tr.Elem1->Weight();
	  // FIRST ORDER TAYLOR
	  tr_shape += gradUResD;
	  //
	  int size = (dim*(dim+1))/2;
          DenseMatrix physical_hess(trialdofs_cnt,size);
          physical_hess = 0.0;
          trial_fe1.CalcPhysHessian2(Trans_el1,physical_hess);
          DenseMatrix adjusted_physical_hess(trialdofs_cnt,dim*dim);
          adjusted_physical_hess = 0.0;
          for (int s = 0; s < trialdofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g <= l; g++){
                adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
                adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
              }
            }
          }
	  for (int s = 0; s < trialdofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g < dim; g++){
                q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
              }
            }
          }
          q_hess_dot_d *= 1.0/2.0;
	  // SECOND ORDER TAYLOR
	  tr_shape += q_hess_dot_d;
	  ///
	  
	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int j = 0; j < trialdofs_cnt; j++)
		{
		  for (int vd = 0; vd < dim; vd++) // Velocity components.
		    {
		      elmat(i, j + vd * trialdofs_cnt) -= tr_shape(j) * nor(vd) * te_shape(i) * ip_f.weight;
		    }
		}
	    }
	}
    }
    else if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE) ) {
      const int dim = test_fe2.GetDim();
      
      int testdofs_offset = test_fe1.GetDof();
      int trialdofs_offset = trial_fe1.GetDof()*dim;
      
      const int testdofs_cnt = test_fe2.GetDof();
      const int trialdofs_cnt = trial_fe2.GetDof();
      
      elmat.SetSize(testdofs_cnt+testdofs_offset, trialdofs_cnt*dim+trialdofs_offset);
      elmat = 0.0;
      
      Vector nor(dim);
      Vector te_shape(testdofs_cnt),tr_shape(trialdofs_cnt), gradUResD(trialdofs_cnt), q_hess_dot_d(trialdofs_cnt);
      DenseMatrix dshape(trialdofs_cnt,dim), dshape_ps(trialdofs_cnt,dim), adjJ(dim);
      
      te_shape = 0.0;
      tr_shape = 0.0;
      gradUResD = 0.0;
      dshape = 0.0;
      dshape_ps = 0.0;
      adjJ = 0.0;
      q_hess_dot_d = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(test_fe2.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement2Transformation();	
      for (int q = 0; q < nqp_face; q++)
	{
	  gradUResD = 0.0;
	  tr_shape = 0.0;  
	  te_shape = 0.0;
	  gradUResD = 0.0;
	  dshape = 0.0;
	  dshape_ps = 0.0;
	  adjJ = 0.0;
	  q_hess_dot_d = 0.0;

	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  nor *= -1.0;
	  
	  test_fe2.CalcShape(eip, te_shape);
	  trial_fe2.CalcShape(eip, tr_shape);
	  
	  ////
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  x_eip = 0.0;
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  trial_fe2.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);	    
	  dshape_ps.Mult(D,gradUResD);
	  gradUResD /= Tr.Elem2->Weight();
	  // FIRST ORDER TAYLOR
	  tr_shape += gradUResD;
	  //
	  int size = (dim*(dim+1))/2;
          DenseMatrix physical_hess(trialdofs_cnt,size);
          physical_hess = 0.0;
          trial_fe2.CalcPhysHessian2(Trans_el1,physical_hess);
          DenseMatrix adjusted_physical_hess(trialdofs_cnt,dim*dim);
          adjusted_physical_hess = 0.0;
          for (int s = 0; s < trialdofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g <= l; g++){
                adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
                adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
              }
            }
          }
	  for (int s = 0; s < trialdofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g < dim; g++){
                q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
              }
            }
          }
          q_hess_dot_d *= 1.0/2.0;
	  // SECOND ORDER TAYLOR
	  tr_shape += q_hess_dot_d;
	  ///

	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int j = 0; j < trialdofs_cnt; j++)
		{
		  for (int vd = 0; vd < dim; vd++) // Velocity components.
		    {
		      elmat(i+testdofs_offset, j + vd * trialdofs_cnt + trialdofs_offset) -= tr_shape(j) * nor(vd) * te_shape(i) * ip_f.weight;
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
      
      elmat.SetSize((testdofs_cnt+testdofs2_cnt), (trialdofs_cnt+trialdofs2_cnt)*dim);
      elmat = 0.0;
    }
 
  }

  void ShiftedPressureBoundaryForceTransposeIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
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

    if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT) ){	
      const int dim = test_fe1.GetDim();
      const int testdofs_cnt = test_fe1.GetDof();
      const int trialdofs_cnt = trial_fe1.GetDof();
      const int testdofs2_cnt = test_fe2.GetDof();
      const int trialdofs2_cnt = trial_fe2.GetDof();
      
      elmat.SetSize((testdofs_cnt+testdofs2_cnt)*dim, trialdofs_cnt+trialdofs2_cnt);
      elmat = 0.0;
      
      Vector nor(dim);
      Vector te_shape(testdofs_cnt),tr_shape(trialdofs_cnt);
      
      te_shape = 0.0;
      tr_shape = 0.0;
      nor = 0.0;
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(test_fe1.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      
      for (int q = 0; q < nqp_face; q++)
	{
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  
	  test_fe1.CalcShape(eip, te_shape);
	  trial_fe1.CalcShape(eip, tr_shape);
	  
	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < trialdofs_cnt; j++)
		    {
		      elmat(i + vd * testdofs_cnt, j) += tr_shape(j) * nor(vd) * te_shape(i) * ip_f.weight;
		    }
		}
	    }
	}
    }
    else if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE) ) {
      const int dim = test_fe2.GetDim();
      
      int testdofs_offset = test_fe1.GetDof()*dim;
      int trialdofs_offset = trial_fe1.GetDof();
      
      const int testdofs_cnt = test_fe2.GetDof();
      const int trialdofs_cnt = trial_fe2.GetDof();
      
      elmat.SetSize(testdofs_cnt*dim+testdofs_offset, trialdofs_cnt+trialdofs_offset);
      elmat = 0.0;
      
      Vector nor(dim);
      Vector te_shape(testdofs_cnt),tr_shape(trialdofs_cnt);
      
      te_shape = 0.0;
      tr_shape = 0.0;
      nor = 0.0;
      
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(test_fe2.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      
      for (int q = 0; q < nqp_face; q++)
	{
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	  CalcOrtho(Tr.Jacobian(), nor);
	  nor *= -1.0;
	  
	  test_fe2.CalcShape(eip, te_shape);
	  trial_fe2.CalcShape(eip, tr_shape);
	  
	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < trialdofs_cnt; j++)
		    {
		      elmat(i + vd * testdofs_cnt + testdofs_offset, j+trialdofs_offset) += tr_shape(j) * nor(vd) * te_shape(i) * ip_f.weight;
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

  void ShiftedVelocityPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
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

    if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT) ){      
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
      Vector shape(h1dofs_cnt), nor(dim), gradURes(h1dofs_cnt), q_hess_dot_d(h1dofs_cnt);
      DenseMatrix dshape(h1dofs_cnt,dim), dshape_ps(h1dofs_cnt,dim), adjJ(dim);
      shape = 0.0;
      nor = 0.0;
      dshape = 0.0;
      dshape_ps = 0.0;
      adjJ = 0.0;
      gradURes = 0.0;
      q_hess_dot_d = 0.0;
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
      
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      
      for (int q = 0; q < nqp_face; q++)
	{
	  shape = 0.0;
	  gradURes = 0.0;
	  q_hess_dot_d = 0.0;
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  CalcOrtho(Tr.Jacobian(), nor);
	  
	  fe.CalcShape(eip, shape);
	  
	  ///
	  Vector x_eip(3);
	  x_eip = 0.0;
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  fe.CalcDShape(eip, dshape);	    
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  dshape_ps.Mult(D,gradURes);
	  gradURes /= Tr.Elem1->Weight();
	  // FIRST ORDER TAYLOR
	  shape += gradURes;
	  //
	  int size = (dim*(dim+1))/2;
          DenseMatrix physical_hess(h1dofs_cnt,size);
          physical_hess = 0.0;
          fe.CalcPhysHessian2(Trans_el1,physical_hess);
          DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
          adjusted_physical_hess = 0.0;
          for (int s = 0; s < h1dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g <= l; g++){
                adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
                adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
              }
            }
          }
	  for (int s = 0; s < h1dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g < dim; g++){
                q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
              }
            }
          }
          q_hess_dot_d *= 1.0/2.0;
	  // SECOND ORDER TAYLOR
	  shape += q_hess_dot_d;
	  ///
	  
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  double Mu = mu->Eval(*Tr.Elem1, eip);
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      elmat(i + vd * h1dofs_cnt, j + vd * h1dofs_cnt) += 4.0 * shape(i) * shape(j) * (nor_norm / Tr.Elem1->Weight()) * ip_f.weight * nor_norm * alpha * Mu;
		    }
		}
	    }
	}
    }
    else if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE) ) {
      const int dim = fe2.GetDim();
      const int dofs_cnt = fe.GetDof();
	
      int h1dofs_offset = dofs_cnt*dim;

      const int h1dofs_cnt = fe2.GetDof();
      elmat.SetSize(2*h1dofs_cnt*dim);
      elmat = 0.0;
      Vector shape(h1dofs_cnt), nor(dim), gradURes(h1dofs_cnt), q_hess_dot_d(h1dofs_cnt);
      DenseMatrix dshape(h1dofs_cnt,dim), dshape_ps(h1dofs_cnt,dim), adjJ(dim);
      shape = 0.0;
      nor = 0.0;
      dshape = 0.0;
      dshape_ps = 0.0;
      adjJ = 0.0;
      gradURes = 0.0;
      q_hess_dot_d = 0.0;
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 3 * max(fe.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement2Transformation();

      for (int q = 0; q < nqp_face; q++)
	{
	  shape = 0.0;
	  gradURes = 0.0;
	  q_hess_dot_d = 0.0;
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;

	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	    
	  const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  CalcOrtho(Tr.Jacobian(), nor);
	  nor *= -1.0;
	  fe2.CalcShape(eip, shape);

	  ///
	  Vector x_eip(3);
	  x_eip = 0.0;	    
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  fe2.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  dshape_ps.Mult(D,gradURes);
	  gradURes /= Tr.Elem2->Weight();
	  // FIRST ORDER TAYLOR
	  shape += gradURes;
	  //
	  int size = (dim*(dim+1))/2;
          DenseMatrix physical_hess(h1dofs_cnt,size);
          physical_hess = 0.0;
          fe2.CalcPhysHessian2(Trans_el1,physical_hess);
          DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
          adjusted_physical_hess = 0.0;
          for (int s = 0; s < h1dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g <= l; g++){
                adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
                adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
              }
            }
          }
	  for (int s = 0; s < h1dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g < dim; g++){
                q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
              }
            }
          }
          q_hess_dot_d *= 1.0/2.0;
	  // SECOND ORDER TAYLOR
	  shape += q_hess_dot_d;
	  ///
	      
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  double Mu = mu->Eval(*Tr.Elem2, eip);
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      elmat(i + vd * h1dofs_cnt + h1dofs_offset, j + vd * h1dofs_cnt + h1dofs_offset) += 4.0 * shape(i) * shape(j) * (nor_norm / Tr.Elem2->Weight()) * ip_f.weight * nor_norm * alpha * Mu;
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


  void ShiftedStrainNitscheBCForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
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

    if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT) ){	      
      const int dim = el.GetDim();
      const int dofs_cnt = el.GetDof();
      elvect.SetSize(2*dofs_cnt*dim);
      elvect = 0.0;
	
      Vector nor(dim), ni(dim), bcEval(dim);
      DenseMatrix dshape(dofs_cnt,dim), dshape_ps(dofs_cnt,dim), adjJ(dim);
      Vector shape(dofs_cnt), gradURes(dofs_cnt);
      double w = 0.0;
	
      shape = 0.0;
      gradURes = 0.0;
	
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 4 * max(el.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	
      for (int q = 0; q  < nqp_face; q++)
	{
	  gradURes = 0.0;
	  shape = 0.0;

	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	    
	  CalcOrtho(Tr.Jacobian(), nor);
	    
	  ///
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  uD->Eval(bcEval, Trans_el1, eip, D);
	  ///
	    
	  el.CalcShape(eip, shape);
	  el.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  w = ip_f.weight/Tr.Elem1->Weight();
	  dshape_ps.Mult(nor,gradURes);

	  double Mu = mu->Eval(*Tr.Elem1, eip);
	  ni.Set(w, nor);
	    
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{	      
		  elvect(i + vd * dofs_cnt) -= bcEval(vd) * gradURes(i) * Mu * w;
		  for (int md = 0; md < dim; md++) // Velocity components.
		    {	      		  
		      elvect(i + vd * dofs_cnt) -= bcEval(md) * ni(vd) * dshape_ps(i,md) * Mu;
		    }
		}
	    }
	}
    }
    else if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE) ) {

      const int dim = el2.GetDim();
      const int dofs_cnt = el2.GetDof();
	
      int h1dofs_offset = el.GetDof()*dim;
      elvect.SetSize(2*dofs_cnt*dim);
      elvect = 0.0;
	
      Vector nor(dim), ni(dim), bcEval(dim);
      DenseMatrix dshape(dofs_cnt,dim), dshape_ps(dofs_cnt,dim), adjJ(dim);
      Vector shape(dofs_cnt), gradURes(dofs_cnt);
      double w = 0.0;
	
      shape = 0.0;
      gradURes = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 4 * max(el2.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement2Transformation();
	
      for (int q = 0; q  < nqp_face; q++)
	{
	  gradURes = 0.0;

	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	    
	  CalcOrtho(Tr.Jacobian(), nor);
	  nor *= -1.0;
	    
	  ///
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  uD->Eval(bcEval, Trans_el1, eip, D);
	  ///
	  
	  el2.CalcShape(eip, shape);
	  el2.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  w = ip_f.weight/Tr.Elem2->Weight();
	  dshape_ps.Mult(nor,gradURes);
	    
	  double Mu = mu->Eval(*Tr.Elem2, eip);
	  ni.Set(w, nor);
	    
	  for (int i = 0; i < dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{	      
		  elvect(i + vd * dofs_cnt + h1dofs_offset) -= bcEval(vd) * gradURes(i) * Mu * w;
		  for (int md = 0; md < dim; md++) // Velocity components.
		    {	      		  
		      elvect(i + vd * dofs_cnt + h1dofs_offset) -= bcEval(md) * ni(vd) * dshape_ps(i,md) * Mu;
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

  void ShiftedPressureNitscheBCForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
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

    if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT) ){
      const int dim = el.GetDim();
      const int testdofs_cnt = el.GetDof();
	
      elvect.SetSize(2*testdofs_cnt);
      elvect = 0.0;
	
      Vector nor(dim), bcEval(dim);
      Vector te_shape(testdofs_cnt);
	
      te_shape = 0.0;
	
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 4 * max(el.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	
      for (int q = 0; q < nqp_face; q++)
	{
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	    
	  CalcOrtho(Tr.Jacobian(), nor);

	  ///
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  uD->Eval(bcEval, Trans_el1, eip, D);
	  ///
	    
	  el.CalcShape(eip, te_shape);
	    
	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i) -= nor(vd) * bcEval(vd) * te_shape(i) * ip_f.weight;
		}
	    }
	}
    }
    else if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE) ) {
      const int dim = el2.GetDim();
      const int testdofs_cnt = el2.GetDof();
	
      elvect.SetSize(2*testdofs_cnt);
      elvect = 0.0;

      int testdofs_offset = el.GetDof();
	
      Vector nor(dim), bcEval(dim);
      Vector te_shape(testdofs_cnt);
	
      te_shape = 0.0;
	
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 4 * max(el2.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement2Transformation();
	
      for (int q = 0; q < nqp_face; q++)
	{
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	    
	  CalcOrtho(Tr.Jacobian(), nor);
	  nor *= -1.0;

	  ///
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  uD->Eval(bcEval, Trans_el1, eip, D);
	  ///
	    
	  el2.CalcShape(eip, te_shape);
	    
	  for (int i = 0; i < testdofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i+testdofs_offset) -= nor(vd) * bcEval(vd) * te_shape(i) * ip_f.weight;
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

  void ShiftedVelocityBCPenaltyIntegrator::AssembleRHSElementVect(const FiniteElement &el,
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

    if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::CUT) ){
      const int dim = el.GetDim();
      const int h1dofs_cnt = el.GetDof();
      elvect.SetSize(2*h1dofs_cnt*dim);
      elvect = 0.0;
      Vector shape(h1dofs_cnt), nor(dim), gradURes(h1dofs_cnt), bcEval(dim), q_hess_dot_d(h1dofs_cnt);
      DenseMatrix dshape(h1dofs_cnt,dim), dshape_ps(h1dofs_cnt,dim), adjJ(dim);
      shape = 0.0;
      nor = 0.0;
      dshape = 0.0;
      dshape_ps = 0.0;
      adjJ = 0.0;
      gradURes = 0.0;
      q_hess_dot_d = 0.0;	
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 4 * max(el.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      for (int q = 0; q < nqp_face; q++)
	{
	  shape = 0.0;
	  gradURes = 0.0;
	  q_hess_dot_d = 0.0;
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	    
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	    
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	    
	  //   Trans_el1.SetIntPoint(&eip);
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Tr.Jacobian(), nor);
	    }

	  el.CalcShape(eip, shape);

	  ///
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  el.CalcDShape(eip, dshape); 
	  CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  dshape_ps.Mult(D,gradURes);
	  gradURes /= Tr.Elem1->Weight();
	  uD->Eval(bcEval, Trans_el1, eip, D);
	  // FIRST ORDER TAYLOR
	  shape += gradURes;
	  //
	  int size = (dim*(dim+1))/2;
          DenseMatrix physical_hess(h1dofs_cnt,size);
          physical_hess = 0.0;
          el.CalcPhysHessian2(Trans_el1,physical_hess);
          DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
          adjusted_physical_hess = 0.0;
          for (int s = 0; s < h1dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g <= l; g++){
                adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
                adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
              }
            }
          }
	  for (int s = 0; s < h1dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g < dim; g++){
                q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
              }
            }
          }
          q_hess_dot_d *= 1.0/2.0;
	  // SECOND ORDER TAYLOR
	  shape += q_hess_dot_d;
	  ///
	    
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	    
	  double Mu = mu->Eval(*Tr.Elem1, eip);
	    
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i + vd * h1dofs_cnt) += 4.0 * shape(i) * bcEval(vd) * alpha * (nor_norm / Tr.Elem1->Weight()) * ip_f.weight  * nor_norm * Mu ;
		}
	    }
	}
    }
    else if ( (elemStatus1 == AnalyticalGeometricShape::SBElementType::CUT) &&  (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE) ) {
      const int dim = el2.GetDim();
      const int h1dofs_cnt = el2.GetDof();
      int h1dofs_offset = el.GetDof()*dim;
	
      elvect.SetSize(2*h1dofs_cnt*dim);
      elvect = 0.0;
      Vector shape(h1dofs_cnt), nor(dim), gradURes(h1dofs_cnt), bcEval(dim), q_hess_dot_d(h1dofs_cnt);
      DenseMatrix dshape(h1dofs_cnt,dim), dshape_ps(h1dofs_cnt,dim), adjJ(dim);
      shape = 0.0;
      nor = 0.0;
      dshape = 0.0;
      dshape_ps = 0.0;
      adjJ = 0.0;
      gradURes = 0.0;
      q_hess_dot_d = 0.0;
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  const int order = 4 * max(el2.GetOrder(), 1);
	  ir = &IntRules.Get(Tr.GetGeometryType(), order);
	}
	
      const int nqp_face = ir->GetNPoints();
      ElementTransformation &Trans_el1 = Tr.GetElement2Transformation();
      for (int q = 0; q < nqp_face; q++)
	{
	  shape = 0.0;
	  gradURes = 0.0;
	  q_hess_dot_d = 0.0;
	  const IntegrationPoint &ip_f = ir->IntPoint(q);
	    
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	    
	  const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	    
	  //   Trans_el1.SetIntPoint(&eip);
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  CalcOrtho(Tr.Jacobian(), nor);
	  nor *= -1.0;
 
	  el2.CalcShape(eip, shape);

	  ///
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  Vector x_eip(3);
	  Trans_el1.Transform(eip,x_eip);
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  el2.CalcDShape(eip, dshape);
	  CalcAdjugate(Tr.Elem2->Jacobian(), adjJ);
	  Mult(dshape, adjJ, dshape_ps);
	  dshape_ps.Mult(D,gradURes);
	  gradURes /= Tr.Elem2->Weight();
	  uD->Eval(bcEval, Trans_el1, eip, D);
	  ///
	  // FIRST ORDER TAYLOR
	  shape += gradURes;
	  //
	  int size = (dim*(dim+1))/2;
          DenseMatrix physical_hess(h1dofs_cnt,size);
          physical_hess = 0.0;
          el2.CalcPhysHessian2(Trans_el1,physical_hess);
          DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
          adjusted_physical_hess = 0.0;
          for (int s = 0; s < h1dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g <= l; g++){
                adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
                adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
              }
            }
          }
	  for (int s = 0; s < h1dofs_cnt; s++){
            for (int l = 0; l < dim; l++){
              for (int g = 0; g < dim; g++){
                q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
              }
            }
          }
          q_hess_dot_d *= 1.0/2.0;
	  // SECOND ORDER TAYLOR
	  shape += q_hess_dot_d;
	  ///
	    
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	    
	  double Mu = mu->Eval(*Tr.Elem2, eip);
	    
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i + vd * h1dofs_cnt+h1dofs_offset) += 4.0 * shape(i) * bcEval(vd) * alpha * (nor_norm / Tr.Elem2->Weight()) * ip_f.weight  * nor_norm * Mu ;
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
