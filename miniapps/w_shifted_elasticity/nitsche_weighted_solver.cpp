// Copyright A(c) 2017, Lawrence Livermore National Security, LLC. Produced at
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

#include "nitsche_weighted_solver.hpp"
#include <unordered_map>

namespace mfem
{

  void WeightedStressBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
							 const FiniteElement &fe2,
							 FaceElementTransformations &Tr,
							 DenseMatrix &elmat)
  {
    const int dim = fe.GetDim();
    const int dofs_cnt = fe.GetDof();
    elmat.SetSize(dofs_cnt*dim);
    elmat = 0.0;

    Vector nor(dim), ni(dim);
    DenseMatrix dshape(dofs_cnt,dim), dshape_ps(dofs_cnt,dim), adjJ(dim);
    Vector shape(dofs_cnt), gradURes(dofs_cnt);
    double w = 0.0;
  
    shape = 0.0;
    gradURes = 0.0;
    dshape = 0.0;
    dshape_ps = 0.0;
    adjJ = 0.0;
    nor = 0.0;
    ni = 0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
      {
	// a simple choice for the integration order; is this OK?
	const int order = 5 * max(fe.GetOrder(), 1);
	ir = &IntRules.Get(Tr.GetGeometryType(), order);
      }

    const int nqp_face = ir->GetNPoints();
    ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
  
    for (int q = 0; q < nqp_face; q++)
      {
	gradURes = 0.0;
	const IntegrationPoint &ip_f = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Tr.SetAllIntPoints(&ip_f);
	const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	CalcOrtho(Tr.Jacobian(), nor);
    	double volumeFraction = alpha->GetValue(Trans_el1, eip);

	fe.CalcShape(eip, shape);
	fe.CalcDShape(eip, dshape);
	CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	Mult(dshape, adjJ, dshape_ps);
	w = ip_f.weight/Tr.Elem1->Weight();
	dshape_ps.Mult(nor,gradURes);

	double Mu = mu->Eval(*Tr.Elem1, eip);
	double Kappa = kappa->Eval(*Tr.Elem1, eip);
	ni.Set(w, nor);

	for (int i = 0; i < dofs_cnt; i++)
	  {
	    for (int vd = 0; vd < dim; vd++) // Velocity components.
	      {
		for (int j = 0; j < dofs_cnt; j++)
		  {
		    elmat(i + vd * dofs_cnt, j + vd * dofs_cnt) += shape(j) * gradURes(i) * Mu * w * volumeFraction;
		    for (int md = 0; md < dim; md++) // Velocity components.
		      {
			elmat(i + vd * dofs_cnt, j + md * dofs_cnt) += shape(j) * ni(vd) * dshape_ps(i,md) * Mu * volumeFraction;
			elmat(i + vd * dofs_cnt, j + md * dofs_cnt) += shape(j) * ni(md) * dshape_ps(i,vd) * (Kappa - (2.0/3.0) * Mu) * volumeFraction;
		      }
		  }
	      }
	  }
      }
  }

  void WeightedStressBoundaryForceTransposeIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
								  const FiniteElement &fe2,
								  FaceElementTransformations &Tr,
								  DenseMatrix &elmat)
  {
    const int dim = fe.GetDim();
    const int dofs_cnt = fe.GetDof();
    elmat.SetSize(dofs_cnt*dim);
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
	const int order = 5 * max(fe.GetOrder(), 1);
	ir = &IntRules.Get(Tr.GetGeometryType(), order);
      }

    const int nqp_face = ir->GetNPoints();
    ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
  
    for (int q = 0; q < nqp_face; q++)
      {
	gradURes = 0.0;
	const IntegrationPoint &ip_f = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Tr.SetAllIntPoints(&ip_f);
	const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	CalcOrtho(Tr.Jacobian(), nor);
    	double volumeFraction = alpha->GetValue(Trans_el1, eip);
    
	fe.CalcShape(eip, shape);
	fe.CalcDShape(eip, dshape);
	CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	Mult(dshape, adjJ, dshape_ps);
	w = ip_f.weight/Tr.Elem1->Weight();
	dshape_ps.Mult(nor,gradURes);

	double Mu = mu->Eval(*Tr.Elem1, eip);
	double Kappa = kappa->Eval(*Tr.Elem1, eip);
	ni.Set(w, nor);
	for (int i = 0; i < dofs_cnt; i++)
	  {
	    for (int vd = 0; vd < dim; vd++) // Velocity components.
	      {
		for (int j = 0; j < dofs_cnt; j++)
		  {
		    elmat(i + vd * dofs_cnt, j + vd * dofs_cnt) -= shape(i) * gradURes(j) * Mu * w * volumeFraction;
		    for (int md = 0; md < dim; md++) // Velocity components.
		      {
			elmat(i + vd * dofs_cnt, j + md * dofs_cnt) -= shape(i) * ni(md) * dshape_ps(j,vd) * Mu * volumeFraction;
			elmat(i + vd * dofs_cnt, j + md * dofs_cnt) -= shape(i) * ni(vd) * dshape_ps(j,md) * (Kappa - (2.0/3.0) * Mu) * volumeFraction;
		      }
		  }
	      }
	  }
      }
  }


  void WeightedNormalDisplacementPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
						     const FiniteElement &fe2,
						     FaceElementTransformations &Tr,
						     DenseMatrix &elmat)
  {
    const int dim = fe.GetDim();
    const int h1dofs_cnt = fe.GetDof();
    elmat.SetSize(h1dofs_cnt*dim);
    elmat = 0.0;
    Vector shape(h1dofs_cnt), nor(dim);
    shape = 0.0;
    nor = 0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
      {
	// a simple choice for the integration order; is this OK?
	const int order = 5 * max(fe.GetOrder(), 1);
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
	Vector nor;
	nor.SetSize(dim);
	nor = 0.0;
	CalcOrtho(Tr.Jacobian(), nor);
	fe.CalcShape(eip, shape);
	double volumeFraction = alpha->GetValue(Trans_el1, eip);
    	
	double nor_norm = 0.0;
	for (int s = 0; s < dim; s++){
	  nor_norm += nor(s) * nor(s);
	}
	nor_norm = sqrt(nor_norm);
	double Kappa = kappa->Eval(*Tr.Elem1, eip);
	for (int i = 0; i < h1dofs_cnt; i++)
	  {
	    for (int vd = 0; vd < dim; vd++) // Velocity components.
	      {
		for (int j = 0; j < h1dofs_cnt; j++)
		  {
		    for (int md = 0; md < dim; md++) // Velocity components.
		      {
			elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += 2.0 * shape(i) * shape(j) * (nor_norm / Tr.Elem1->Weight()) * ip_f.weight * nor_norm * penaltyParameter * 3 * Kappa * volumeFraction * nor(vd) * nor(md) / (nor_norm * nor_norm);
		      }
		  }
	      }
	  }
      }
  }

  void WeightedTangentialDisplacementPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
						     const FiniteElement &fe2,
						     FaceElementTransformations &Tr,
						     DenseMatrix &elmat)
  {
    const int dim = fe.GetDim();
    const int h1dofs_cnt = fe.GetDof();
    elmat.SetSize(h1dofs_cnt*dim);
    elmat = 0.0;
    Vector shape(h1dofs_cnt), nor(dim);
    shape = 0.0;
    nor = 0.0;
  
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
      {
	// a simple choice for the integration order; is this OK?
	const int order = 5 * max(fe.GetOrder(), 1);
	ir = &IntRules.Get(Tr.GetGeometryType(), order);
      }

    const int nqp_face = ir->GetNPoints();
    ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
    DenseMatrix identity(dim);
    identity = 0.0;
    for (int s = 0; s < dim; s++){
      identity(s,s) = 1.0;
    }
    
    for (int q = 0; q < nqp_face; q++)
      {
	const IntegrationPoint &ip_f = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Tr.SetAllIntPoints(&ip_f);
    
	const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	Vector nor;
	nor.SetSize(dim);
	nor = 0.0;
	CalcOrtho(Tr.Jacobian(), nor);
	fe.CalcShape(eip, shape);
	double volumeFraction = alpha->GetValue(Trans_el1, eip);
   
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
		    for (int md = 0; md < dim; md++) // Velocity components.
		      {
			elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += 2.0 * shape(i) * shape(j) * (nor_norm / Tr.Elem1->Weight()) * ip_f.weight * nor_norm * penaltyParameter * 2 * Mu * volumeFraction * (identity(vd,md) - nor(vd) * nor(md) / (nor_norm * nor_norm));
		      }
		  }
	      }
	  }
      }
  }

  void WeightedStressNitscheBCForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
							      FaceElementTransformations &Tr,
							      Vector &elvect)
  {
    const int dim = el.GetDim();
    const int dofs_cnt = el.GetDof();
    elvect.SetSize(dofs_cnt*dim);
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
	const int order = 5 * max(el.GetOrder(), 1);
	ir = &IntRules.Get(Tr.GetGeometryType(), order);
      }

    const int nqp_face = ir->GetNPoints();
    ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
     
    for (int q = 0; q  < nqp_face; q++)
      {
	gradURes = 0.0;
	const IntegrationPoint &ip_f = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Tr.SetAllIntPoints(&ip_f);
	const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	double volumeFraction = alpha->GetValue(Trans_el1, eip);
    
	CalcOrtho(Tr.Jacobian(), nor);
	
	uD->Eval(bcEval, Trans_el1, eip);
	
	el.CalcShape(eip, shape);
	el.CalcDShape(eip, dshape);
	CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
	Mult(dshape, adjJ, dshape_ps);
	w = ip_f.weight/Tr.Elem1->Weight();
	dshape_ps.Mult(nor,gradURes);
	double Mu = mu->Eval(*Tr.Elem1, eip);
	double Kappa = kappa->Eval(*Tr.Elem1, eip);

	ni.Set(w, nor);
	
	for (int i = 0; i < dofs_cnt; i++)
	  {
	    for (int vd = 0; vd < dim; vd++) // Velocity components.
	      {	      
		elvect(i + vd * dofs_cnt) += bcEval(vd) * gradURes(i) * Mu * w * volumeFraction;
		for (int md = 0; md < dim; md++) // Velocity components.
		  {	      		  
		    elvect(i + vd * dofs_cnt) += bcEval(md) * ni(vd) * dshape_ps(i,md) * Mu * volumeFraction;
		    elvect(i + vd * dofs_cnt) += bcEval(md) * ni(md) * dshape_ps(i,vd) * (Kappa - (2.0/3.0) * Mu) * volumeFraction;
		  }
	      }
	  }
      }
  }
  void WeightedStressNitscheBCForceIntegrator::AssembleRHSElementVect(
							      const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
  {
    mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
  }


  void WeightedNormalDisplacementBCPenaltyIntegrator::AssembleRHSElementVect(const FiniteElement &el,
							   FaceElementTransformations &Tr,
							   Vector &elvect)
  {
    const int dim = el.GetDim();
    const int h1dofs_cnt = el.GetDof();
    elvect.SetSize(h1dofs_cnt*dim);
    elvect = 0.0;
    Vector shape(h1dofs_cnt), nor(dim), bcEval(dim);
    shape = 0.0;
    nor = 0.0;
  
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
      {
	// a simple choice for the integration order; is this OK?
	const int order = 5 * max(el.GetOrder(), 1);
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
	double volumeFraction = alpha->GetValue(Trans_el1, eip);
        
	//   Trans_el1.SetIntPoint(&eip);
	Vector nor;
	nor.SetSize(dim);
	nor = 0.0;
	CalcOrtho(Tr.Jacobian(), nor);
	el.CalcShape(eip, shape);
	double nor_norm = 0.0;
	for (int s = 0; s < dim; s++){
	  nor_norm += nor(s) * nor(s);
	}
	nor_norm = sqrt(nor_norm);
	uD->Eval(bcEval, Trans_el1, eip);

	double Kappa = kappa->Eval(*Tr.Elem1, eip);
    
	for (int i = 0; i < h1dofs_cnt; i++)
	  {
	    for (int vd = 0; vd < dim; vd++) // Velocity components.
	      {
		for (int md = 0; md < dim; md++) // Velocity components.
		  {	
		    elvect(i + vd * h1dofs_cnt) += 2.0 * shape(i) * bcEval(md) * penaltyParameter * (nor_norm / Tr.Elem1->Weight()) * ip_f.weight * nor_norm * 3 * Kappa * volumeFraction * (nor(md) / nor_norm) * (nor(vd) / nor_norm);
		  }
	      }
	  }
      }
  }

  void WeightedNormalDisplacementBCPenaltyIntegrator::AssembleRHSElementVect(
							   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
  {
    mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
  }

    void WeightedTangentialDisplacementBCPenaltyIntegrator::AssembleRHSElementVect(const FiniteElement &el,
							   FaceElementTransformations &Tr,
							   Vector &elvect)
  {
    const int dim = el.GetDim();
    const int h1dofs_cnt = el.GetDof();
    elvect.SetSize(h1dofs_cnt*dim);
    elvect = 0.0;
    Vector shape(h1dofs_cnt), nor(dim), bcEval(dim);
    shape = 0.0;
    nor = 0.0;
  
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
      {
	// a simple choice for the integration order; is this OK?
	const int order = 5 * max(el.GetOrder(), 1);
	ir = &IntRules.Get(Tr.GetGeometryType(), order);
      }

    DenseMatrix identity(dim);
    identity = 0.0;
    for (int s = 0; s < dim; s++){
      identity(s,s) = 1.0;
    }
    
    const int nqp_face = ir->GetNPoints();
    ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
    for (int q = 0; q < nqp_face; q++)
      {
	const IntegrationPoint &ip_f = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Tr.SetAllIntPoints(&ip_f);
	const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	double volumeFraction = alpha->GetValue(Trans_el1, eip);
        
	//   Trans_el1.SetIntPoint(&eip);
	Vector nor;
	nor.SetSize(dim);
	nor = 0.0;
	CalcOrtho(Tr.Jacobian(), nor);
	el.CalcShape(eip, shape);
	double nor_norm = 0.0;
	for (int s = 0; s < dim; s++){
	  nor_norm += nor(s) * nor(s);
	}
	nor_norm = sqrt(nor_norm);
	uD->Eval(bcEval, Trans_el1, eip);

	double weight = ip_f.weight/Tr.Elem1->Weight();
	double Mu = mu->Eval(*Tr.Elem1, eip);
    
	for (int i = 0; i < h1dofs_cnt; i++)
	  {
	    for (int vd = 0; vd < dim; vd++) // Velocity components.
	      {
		for (int md = 0; md < dim; md++) // Velocity components.
		  {	
		    elvect(i + vd * h1dofs_cnt) += 2.0 * shape(i) * bcEval(md) * penaltyParameter * (nor_norm / Tr.Elem1->Weight()) * ip_f.weight * nor_norm * 2 * Mu * volumeFraction * (identity(vd,md) - nor(vd) * nor(md) / (nor_norm * nor_norm));
		  }
	      }
	  }
      }
  }
  
  void WeightedTangentialDisplacementBCPenaltyIntegrator::AssembleRHSElementVect(
							   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
  {
    mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
  }

  void WeightedTractionBCIntegrator::AssembleRHSElementVect(const FiniteElement &el,
							   FaceElementTransformations &Tr,
							   Vector &elvect)
  {
    const int dim = el.GetDim();
    const int h1dofs_cnt = el.GetDof();
    elvect.SetSize(h1dofs_cnt*dim);
    elvect = 0.0;
    Vector shape(h1dofs_cnt), nor(dim), bcEval(dim);
    shape = 0.0;
    nor = 0.0;
 
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
      {
	// a simple choice for the integration order; is this OK?
	const int order = 5 * max(el.GetOrder(), 1);
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
	double volumeFraction = alpha->GetValue(Trans_el1, eip);
        
	Vector nor;
	nor.SetSize(dim);
	nor = 0.0;
	CalcOrtho(Tr.Jacobian(), nor);
	el.CalcShape(eip, shape);
	double nor_norm = 0.0;
	for (int s = 0; s < dim; s++){
	  nor_norm += nor(s) * nor(s);
	}
	nor_norm = sqrt(nor_norm);
	tN->Eval(bcEval, Trans_el1, eip);

	for (int i = 0; i < h1dofs_cnt; i++)
	  {
	    for (int vd = 0; vd < dim; vd++) // Velocity components.
	      {
		elvect(i + vd * h1dofs_cnt) += shape(i) * bcEval(vd) * ip_f.weight * nor_norm * volumeFraction;
	      }
	  }
      }
  }

  void WeightedTractionBCIntegrator::AssembleRHSElementVect(
							   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
  {
    mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");

  }
}
