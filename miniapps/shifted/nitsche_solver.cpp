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

#include "nitsche_solver.hpp"
#include <unordered_map>

namespace mfem
{

  void StrainBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
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

void StrainBoundaryForceTransposeIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
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

void PressureBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe1,
					     FaceElementTransformations &Tr,
                                             DenseMatrix &elmat)
{
  const int dim = test_fe1.GetDim();
  const int testdofs_cnt = test_fe1.GetDof();
  const int trialdofs_cnt = trial_fe.GetDof();

  elmat.SetSize(testdofs_cnt, trialdofs_cnt*dim);
  elmat = 0.0;

  Vector nor(dim);
  Vector te_shape(testdofs_cnt),tr_shape(trialdofs_cnt);

  te_shape = 0.0;
  tr_shape = 0.0;

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
      trial_fe.CalcShape(eip, tr_shape);

      for (int i = 0; i < testdofs_cnt; i++)
	{
	  for (int j = 0; j < trialdofs_cnt; j++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elmat(i, j + vd * trialdofs_cnt) += tr_shape(j) * nor(vd) * te_shape(i) * ip_f.weight;
		}
	    }
	}
    }
}

  void PressureBoundaryForceTransposeIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe1,
					     FaceElementTransformations &Tr,
                                             DenseMatrix &elmat)
{
  const int dim = test_fe1.GetDim();
  const int testdofs_cnt = test_fe1.GetDof();
  const int trialdofs_cnt = trial_fe.GetDof();

  elmat.SetSize(testdofs_cnt*dim, trialdofs_cnt);
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
      trial_fe.CalcShape(eip, tr_shape);

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

void VelocityPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
						   const FiniteElement &fe2,
						   FaceElementTransformations &Tr,
						   DenseMatrix &elmat)
{
  const int dim = fe.GetDim();
  const int h1dofs_cnt = fe.GetDof();
  elmat.SetSize(h1dofs_cnt*dim);
  elmat = 0.0;
  Vector shape(h1dofs_cnt), nor(dim), ni(dim);
  shape = 0.0;
  nor = 0.0;
  ni = 0.0;

  
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
      const IntegrationPoint &ip_f = ir->IntPoint(q);
      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip_f);
    
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
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
      fe.CalcShape(eip, shape);
      double weight = ip_f.weight/Tr.Elem1->Weight();
      ni.Set(weight, nor);
      double wq = ni * nor;
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


void StrainNitscheBCForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
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
      const int order = 3 * max(el.GetOrder(), 1);
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

	CalcOrtho(Tr.Jacobian(), nor);
	
	uD->Eval(bcEval, Trans_el1, eip);
	
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

  void StrainNitscheBCForceIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
}

 void PressureNitscheBCForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
							       FaceElementTransformations &Tr,
							       Vector &elvect)
 {
  const int dim = el.GetDim();
  const int testdofs_cnt = el.GetDof();

  elvect.SetSize(testdofs_cnt);
  elvect = 0.0;

  Vector nor(dim), bcEval(dim);
  Vector te_shape(testdofs_cnt);

  te_shape = 0.0;

  const IntegrationRule *ir = IntRule;
  if (ir == NULL)
    {
      // a simple choice for the integration order; is this OK?
      const int order = 3 * max(el.GetOrder(), 1);
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
    
      uD->Eval(bcEval, Trans_el1, eip);

      el.CalcShape(eip, te_shape);

      for (int i = 0; i < testdofs_cnt; i++)
	{
	  for (int vd = 0; vd < dim; vd++) // Velocity components.
	    {
	      elvect(i) += nor(vd) * bcEval(vd) * te_shape(i) * ip_f.weight;
	    }
	}
    }
}
void PressureNitscheBCForceIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
}

  void VelocityBCPenaltyIntegrator::AssembleRHSElementVect(const FiniteElement &el,
						       FaceElementTransformations &Tr,
						       Vector &elvect)
{
  const int dim = el.GetDim();
  const int h1dofs_cnt = el.GetDof();
  elvect.SetSize(h1dofs_cnt*dim);
  elvect = 0.0;
  Vector shape(h1dofs_cnt), nor(dim), ni(dim), bcEval(dim);
  shape = 0.0;
  nor = 0.0;
  ni = 0.0;
  
  
  const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 3 * max(el.GetOrder(), 1);
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
      double nor_norm = 0.0;
      for (int s = 0; s < dim; s++){
	nor_norm += nor(s) * nor(s);
      }
      nor_norm = sqrt(nor_norm);
      uD->Eval(bcEval, Trans_el1, eip);

      double weight = ip_f.weight/Tr.Elem1->Weight();
      ni.Set(weight, nor);
      double wq = ni * nor;
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

void VelocityBCPenaltyIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
}


}
