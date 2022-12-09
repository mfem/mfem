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

#include "volume_weighted_integrals.hpp"
#include <unordered_map>

namespace mfem
{
  
  void WeightedStressForceIntegrator::AssembleElementMatrix(const FiniteElement &el,
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
	double Kappa = kappa->Eval(Trans, ip);

	double volumeFraction = alpha->GetValue(Trans, ip);
	double w = (ip.weight/Trans.Weight());
	for (int i = 0; i < dof; i++)
	  {
	    for (int k = 0; k < dim; k++)
	      {
		for (int j = 0; j < dof; j++)
		  {
		    for (int s = 0; s < dim; s++)
		      {		  
			elmat(i + k * dof, j + k * dof) += dshape_ps(i,s) * dshape_ps(j,s) * w * Mu * volumeFraction;
			elmat(i + k * dof, j + s * dof) += dshape_ps(i,s) * dshape_ps(j,k) * w * Mu * volumeFraction;
			elmat(i + k * dof, j + s * dof) += dshape_ps(i,k) * dshape_ps(j,s) * w * (Kappa - (2.0/3.0) * Mu) * volumeFraction;
		      }
		  }
	      }
	  }
      }
  }

  const IntegrationRule &WeightedStressForceIntegrator::GetRule(
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
}
