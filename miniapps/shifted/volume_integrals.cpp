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

#include "volume_integrals.hpp"
#include <unordered_map>

namespace mfem
{
  
  void WeightedStrainStrainForceIntegrator::AssembleElementMatrix(const FiniteElement &el,
								  ElementTransformation &Trans,
								  DenseMatrix &elmat)
  {
    const int dim = el.GetDim();
    int dof = el.GetDof();
    
    elmat.SetSize (dof*dim);
    elmat = 0.0;
    
    DenseMatrix dshape(dof,dim), dshape_ps(dof,dim), adjJ(dim);
    dshape = 0.0;
    dshape_ps = 0.0;
    adjJ = 0.0;
    
    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el,
							     Trans);
  
    for (int q = 0; q < ir->GetNPoints(); q++)
      {
	const IntegrationPoint &ip = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Trans.SetIntPoint(&ip);

	el.CalcDShape (ip, dshape);
	CalcAdjugate(Trans.Jacobian(), adjJ);
	Mult(dshape, adjJ, dshape_ps);
	double Mu = mu->Eval(Trans, ip);

	double volumeFraction = alpha->GetValue(Trans, ip);
	double w = (ip.weight/Trans.Weight()) * Mu;
	
	for (int i = 0; i < dof; i++)
	  {
	    for (int k = 0; k < dim; k++)
	      {
		for (int j = 0; j < dof; j++)
		  {
		    for (int s = 0; s < dim; s++)
		      {		  
			elmat(i + k * dof, j + k * dof) += dshape_ps(i,s) * dshape_ps(j,s) * w * volumeFraction;
			elmat(i + k * dof, j + s * dof) += dshape_ps(i,s) * dshape_ps(j,k) * w * volumeFraction; 		
		      }
		  }
	      }
	  }
      }
  }

  const IntegrationRule &WeightedStrainStrainForceIntegrator::GetRule(
						       const FiniteElement &trial_fe,
						       const FiniteElement &test_fe,
						       ElementTransformation &Trans)
  {
    int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
    return IntRules.Get(trial_fe.GetGeomType(), 2*order);
  }

  void WeightedPDivWForceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
						    const FiniteElement &test_fe,
						    ElementTransformation &Trans,
						    DenseMatrix &elmat)
  {
    const int dim = trial_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();

    elmat.SetSize (test_dof*dim, trial_dof);
    elmat = 0.0;
    
    DenseMatrix dshape(test_dof,dim), dshape_ps(test_dof,dim), adjJ(dim);
    Vector shape(trial_dof),divshape(dim*test_dof);
    shape = 0.0;
    dshape = 0.0;
    dshape_ps = 0.0;
    adjJ = 0.0;
    divshape = 0.0;
    
    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
							     Trans);
  
    for (int q = 0; q < ir->GetNPoints(); q++)
      {
	const IntegrationPoint &ip = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Trans.SetIntPoint(&ip);
    	double volumeFraction = alpha->GetValue(Trans, ip);

	test_fe.CalcDShape (ip, dshape);
	trial_fe.CalcShape (ip, shape);

	CalcAdjugate(Trans.Jacobian(), adjJ);

	Mult(dshape, adjJ, dshape_ps);

	dshape_ps.GradToDiv(divshape);

	shape *= -ip.weight * volumeFraction;

	AddMultVWt (divshape, shape, elmat);
      }
  }

  const IntegrationRule &WeightedPDivWForceIntegrator::GetRule(
						       const FiniteElement &trial_fe,
						       const FiniteElement &test_fe,
						       ElementTransformation &Trans)
  {
    int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
    return IntRules.Get(trial_fe.GetGeomType(), 2*order);
  }

  void WeightedQDivUForceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
						    const FiniteElement &test_fe,
						    ElementTransformation &Trans,
						    DenseMatrix &elmat)
  {
    const int dim = trial_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();

    elmat.SetSize (test_dof, trial_dof*dim);
    elmat = 0.0;
    
    DenseMatrix dshape(trial_dof,dim), dshape_ps(trial_dof,dim), adjJ(dim);
    Vector shape(trial_dof);
    shape = 0.0;
    dshape = 0.0;
    dshape_ps = 0.0;
    adjJ = 0.0;
    
    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
							     Trans);
  
    for (int q = 0; q < ir->GetNPoints(); q++)
      {
	const IntegrationPoint &ip = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Trans.SetIntPoint(&ip);
    	double volumeFraction = alpha->GetValue(Trans, ip);
    
	trial_fe.CalcDShape (ip, dshape);
	test_fe.CalcShape (ip, shape);

	CalcAdjugate(Trans.Jacobian(), adjJ);

	Mult(dshape, adjJ, dshape_ps);

	for (int i = 0; i < test_dof; i++)
	  {
	    for (int j = 0; j < trial_dof; j++)
	      {
		for (int md = 0; md < dim; md++)
		  {
		    elmat(i, j + md * trial_dof) += dshape_ps(j,md) * shape(i) * ip.weight * volumeFraction;
		  }
	      }
	  }
      }
  }

  const IntegrationRule &WeightedQDivUForceIntegrator::GetRule(
						       const FiniteElement &trial_fe,
						       const FiniteElement &test_fe,
						       ElementTransformation &Trans)
  {
    int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
    return IntRules.Get(trial_fe.GetGeomType(), 2*order);
  }

  void WeightedVectorForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
						      ElementTransformation &Tr,
						     Vector &elvect)
  {
    const int dim = el.GetDim();
    int dof = el.GetDof();
    Vector forceEval(dim), shape(dof);
    elvect.SetSize (dof*dim);
    elvect = 0.0;
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
      {
	ir = &IntRules.Get(el.GetGeomType(), 5 * el.GetOrder());
      }

    for (int q = 0; q < ir->GetNPoints(); q++)
      {
	const IntegrationPoint &ip = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Tr.SetIntPoint(&ip);
	Q->Eval(forceEval, Tr, ip);
	el.CalcShape (ip, shape);
    	double volumeFraction = alpha->GetValue(Tr, ip);

	for (int i = 0; i < dof; i++)
	  {
	    for (int md = 0; md < dim; md++)
	      {	  
		elvect(i + md * dof) += shape(i) * ip.weight * forceEval(md) * Tr.Weight() * volumeFraction;
	      }
	  }
      }
  }

  void DivergenceStrainStabilizedIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
								    const FiniteElement &test_fe,
								    ElementTransformation &Trans,
								    DenseMatrix &elmat)
  {
    const int dim = trial_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    
    elmat.SetSize (test_dof, trial_dof * dim);
    elmat = 0.0;
    
    DenseMatrix dshape_tr(trial_dof,dim), dshape_te(test_dof,dim), adjJ(dim), dshape_te_ps(test_dof,dim);
    Vector shape_tr(trial_dof);
    shape_tr = 0.0; 
    dshape_tr = 0.0;
    dshape_te = 0.0;
    dshape_te_ps = 0.0;
    adjJ = 0.0;
    
    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
							     Trans);
    
    double w = 0.0;

    DenseMatrix nodalGrad;
    trial_fe.ProjectGrad(trial_fe,Trans,nodalGrad);
    
    for (int q = 0; q < ir->GetNPoints(); q++)
      {
	shape_tr = 0.0;
	dshape_tr = 0.0;
	dshape_te = 0.0;
	dshape_te_ps = 0.0;
	adjJ = 0.0;
	
	const IntegrationPoint &ip = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Trans.SetIntPoint(&ip);
	double volumeFraction = alpha->GetValue(Trans, ip);
	
	// grad q (pressure test)
	test_fe.CalcDShape(ip, dshape_te);
	CalcAdjugate(Trans.Jacobian(), adjJ);
	Mult(dshape_te, adjJ, dshape_te_ps);
	//

	double Mu = mu->Eval(Trans, ip);

	double c = ip.weight * coeff.Eval(Trans, ip) * 2 * Mu;
	
	// divGrad (velocity trial)
	trial_fe.CalcShape(ip, shape_tr);
	DenseMatrix divGrad;
	divGrad.SetSize(trial_dof, dim * dim);
	for (int a = 0; a < trial_dof; a++){
	  for (int j = 0; j < dim; j++){
	    for (int k = 0; k < dim; k++){
	      for (int b = 0; b < trial_dof; b++){
		for (int o = 0; o < trial_dof; o++){      
		  divGrad(a, j + k * dim) += nodalGrad(b + j * trial_dof, a) * nodalGrad(o + k * trial_dof, b) * shape_tr(o); 
		}
	      }
	    }
	  }
	}

	for (int a = 0; a < test_dof; a++){
	  for (int b = 0; b < trial_dof; b++){
	    for (int k = 0; k < dim; k++){
	      for (int i = 0; i < dim; i++){
		elmat(a, b + k * trial_dof) -= 0.5 * c * dshape_te_ps(a,i) * divGrad(b, i + k * dim) * volumeFraction;
		if (i == k){
		  for (int j = 0; j < dim; j++){
		    elmat(a, b + k * trial_dof) -= 0.5 * c * dshape_te_ps(a,i) * divGrad(b, j + j * dim) * volumeFraction;
		  }
		}
	      }
	    }
	  }
	}
      }
  }
  
  const IntegrationRule &DivergenceStrainStabilizedIntegrator::GetRule(
						       const FiniteElement &trial_fe,
						       const FiniteElement &test_fe,
						       ElementTransformation &Trans)
  {
    int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
    return IntRules.Get(trial_fe.GetGeomType(), 5*order);
  }

  void StabilizedDomainLFGradIntegrator::AssembleRHSElementVect(const FiniteElement &el,
						      ElementTransformation &Tr,
						      Vector &elvect)
  {
    const int dim = el.GetDim();
    int dof = el.GetDof();

    elvect.SetSize (dof);
    elvect = 0.0;
    Vector Qvec;
    DenseMatrix dshape(dof,dim), dshape_ps(dof,dim), adjJ(dim);
    dshape = 0.0;
    dshape_ps = 0.0;
    adjJ = 0.0;
    
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
      {
	ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder());
      }

    for (int q = 0; q < ir->GetNPoints(); q++)
      {
	const IntegrationPoint &ip = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Tr.SetIntPoint(&ip);
	double volumeFraction = alpha->GetValue(Tr, ip);
	
	el.CalcDShape (ip, dshape);
	CalcAdjugate(Tr.Jacobian(), adjJ);
	Mult(dshape, adjJ, dshape_ps);
	double c = ip.weight * coeff.Eval(Tr, ip);
	Q.Eval(Qvec, Tr, ip);

	for (int i = 0; i < dof; i++)
	  {
	    for (int md = 0; md < dim; md++)
	      {	  
		elvect(i) += dshape_ps(i,md) * c * Qvec(md) * volumeFraction;
	      }
	  }
      }
  }
  void WeightedDiffusionIntegrator::AssembleElementMatrix(const FiniteElement &el,
							   ElementTransformation &Trans,
							   DenseMatrix &elmat)
  {
    const int dim = el.GetDim();
    int dof = el.GetDof();

    elmat.SetSize(dof);
    elmat = 0.0;
    
    DenseMatrix dshape(dof,dim), dshape_ps(dof,dim), adjJ(dim);
    dshape = 0.0;
    dshape_ps = 0.0;
 
    adjJ = 0.0;
    
    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);
    
    for (int q = 0; q < ir->GetNPoints(); q++)
      {
	const IntegrationPoint &ip = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Trans.SetIntPoint(&ip);

	CalcAdjugate(Trans.Jacobian(), adjJ);

	el.CalcDShape(ip, dshape);

	Mult(dshape, adjJ, dshape_ps);
	double w = coeff.Eval(Trans, ip) * ip.weight/Trans.Weight();
	double volumeFraction = alpha->GetValue(Trans, ip);

	for (int i = 0; i < dof; i++)
	  {
	    for (int j = 0; j < dof; j++)
	      {
		for (int k = 0; k < dim; k++)
		  {
		    elmat(i, j) += dshape_ps(i,k) * dshape_ps(j,k) * w * volumeFraction;
		  }    
	      }
	  }
      }
  }
  const IntegrationRule &WeightedDiffusionIntegrator::GetRule(
						       const FiniteElement &trial_fe,
						       const FiniteElement &test_fe,
						       ElementTransformation &Trans)
  {
    int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
    return IntRules.Get(trial_fe.GetGeomType(), 2*order);
  }

}
