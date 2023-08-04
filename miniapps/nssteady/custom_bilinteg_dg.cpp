// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "custom_bilinteg_dg.hpp"

namespace mfem
{
void TensorMassIntegrator::AssembleElementMatrix(
const FiniteElement &el,
ElementTransformation &Trans,
DenseMatrix &elmat
)
{
	// dim & dof
	ndof = el.GetDof();
	dim = el.GetDim();
	tdim=  dim*dim; 	// tensor dimension


	// initialization
	shape.SetSize(ndof);
	partelmat.SetSize(ndof);

	elmat.SetSize(tdim*ndof, tdim*ndof);
	elmat = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  int order = 2 * el.GetOrder() + Trans.OrderW();
	  ir = &IntRules.Get(el.GetGeomType(), order);
	}

	// sum up all quadrature nodes along with weights
   for (int p = 0; p < ir -> GetNPoints(); p++)
   {
	   partelmat = 0.0;
	   const IntegrationPoint &ip = ir->IntPoint(p);
	   Trans.SetIntPoint (&ip);

	   el.CalcShape (ip, shape);
	   weight = ip.weight * Trans.Weight()*Re;

	   MultVVt(shape, partelmat);
	   partelmat *= weight;
	   for (int i = 0; i < tdim; i++)
	   {
		   elmat.AddMatrix(partelmat, ndof*i, ndof*i);
	   }

   }
}

void MixedVectorDivTensorIntegrator::AssembleElementMatrix2(
const FiniteElement &trial_fe,
const FiniteElement &test_fe,
ElementTransformation &Trans,
DenseMatrix &elmat)
{
    // dim & dof
	trial_ndof = trial_fe.GetDof();
	test_ndof  = test_fe.GetDof();
	dim = trial_fe.GetDim();
	vdim = dim;
	tdim = dim*dim;

	// initialization
	shape.SetSize(trial_ndof);
	shape = 0.0;

	dshape.SetSize(test_ndof,vdim);
	gshape.SetSize(test_ndof,vdim);
	Jadj.SetSize(vdim);

	dshape = 0.0;
	gshape = 0.0;
	Jadj = 0.0;

	partelmat.SetSize(vdim*test_ndof, trial_ndof);
	elmat.SetSize(tdim*test_ndof, vdim*trial_ndof);
	partelmat = 0.0;
	elmat = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  int order = 2 * std::max(test_fe.GetOrder(),trial_fe.GetOrder()) + Trans.OrderW();
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}

	// sum up all quadrature nodes along with weights
	for (int p = 0; p < ir -> GetNPoints(); p++)
	{
	   const IntegrationPoint &ip = ir->IntPoint(p);
	   trial_fe.CalcShape (ip, shape);
	   test_fe.CalcDShape (ip, dshape);

	   // calculate the adjugate of the Jacobian
	   Trans.SetIntPoint (&ip);
	   CalcAdjugate(Trans.Jacobian(), Jadj);

	   // calculate the gradient of the function of the physical element
	   Mult (dshape, Jadj, gshape);

	   weight = ip.weight;
	   shape *= weight;

	   for(int tr_d=0; tr_d<vdim; tr_d++)
	   		   for(int te_d=0; te_d<vdim; te_d++)
	   			   for (int i=0; i<test_ndof; i++)
	   				   for(int j=0; j<trial_ndof; j++)
	   					   elmat(i + te_d*test_ndof + tr_d*(test_ndof*vdim), j + tr_d*trial_ndof) += shape(j) * gshape(i,te_d);
	}

}

void VectorGradVectorIntegrator::AssembleElementMatrix
(const FiniteElement &el,
ElementTransformation &Trans,
DenseMatrix &elmat)
{
	// dim & dof
	ndof = el.GetDof();
	dim  = el.GetDim();
	vdim = dim;

	// initialization
	evalQ.SetSize(vdim);
	evalQ = 0.0;
	shape.SetSize(ndof);
	shape = 0.0;
	gshape_Q.SetSize(ndof);
	gshape_Q = 0.0;

	dshape.SetSize(ndof,vdim);
	gshape.SetSize(ndof,vdim);
	Jadj.SetSize(vdim);

	dshape = 0.0;
	gshape = 0.0;
	Jadj = 0.0;

	elmat.SetSize(vdim*ndof, vdim*ndof);
	elmat = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  int order = 3 * el.GetOrder() + Trans.OrderW();
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}

	// sum up all quadrature nodes along with weights
	for (int p = 0; p < ir -> GetNPoints(); p++)
	{
	  const IntegrationPoint &ip = ir->IntPoint(p);
	  el.CalcShape (ip, shape);
	  el.CalcDShape (ip, dshape);

	  // calculate the adjugate of the Jacobian
	  Trans.SetIntPoint (&ip);
	  CalcAdjugate(Trans.Jacobian(), Jadj);

	  // Calculate the gradient of the function of the physical element
	  Mult (dshape, Jadj, gshape);

	  Q->Eval(evalQ, Trans, ip);
	  gshape.Mult(evalQ, gshape_Q);

	  weight = ip.weight;
	  for (int d =0; d<vdim; d++)
		  for (int i=0; i<ndof; i++)
			  for (int j=0; j<ndof; j++)
				  elmat(i +d*ndof, j + d*ndof) += gshape_Q(i)*shape(j)*weight;
	}
	elmat *= lambda;
}


void DGVectorAvgNormalJumpIntegration::AssembleFaceMatrix(
const FiniteElement &tr_fe1,
const FiniteElement &tr_fe2,
const FiniteElement &te_fe1,
const FiniteElement &te_fe2,
FaceElementTransformations &Trans,
DenseMatrix &elmat)
{
	// dim & dof
	dim  = tr_fe1.GetDim();
	vdim = dim;
	tdim = dim*dim;

	tr_ndof1 = tr_fe1.GetDof();
	te_ndof1 = te_fe1.GetDof();

   if (Trans.Elem2No >= 0)
   {
	  // interior faces
	  tr_ndof2 = tr_fe2.GetDof();
	  te_ndof2 = te_fe2.GetDof();
   }
   else
   {
	  // boundary faces
	  tr_ndof2 = 0;
	  te_ndof2 = 0;
   }

   tr_ndofs = tr_ndof1 + tr_ndof2;
   te_ndofs = te_ndof1 + te_ndof2;

   // initialization
   elmat.SetSize(te_ndofs*tdim, tr_ndofs*vdim);
   elmat = 0.0;

   nor.SetSize(dim);  // unit normal vector
   tr_s1.SetSize(tr_ndof1); // trial shape function on element 1
   tr_s2.SetSize(tr_ndof2); // trial shape function on element 2 (neighbor)
   te_s1.SetSize(te_ndof1); // test shape function on element 1
   te_s2.SetSize(te_ndof2); // test shape function on element 2 (neighbor)
   nor = 0.0;
   tr_s1 = 0.0;
   tr_s2 = 0.0;
   te_s2 = 0.0;
   te_s1 = 0.0;

   // elmat = [ A11   A12 ]
   //         [ A21   A22 ]
   // where the blocks corresponds to the terms in the face integral < {u},[tau]*nor > from
   // the different elements and trial/test space, i.e.
   // A11 : terms from element 1 test and element 1 trial space
   // A21 : terms from element 2 test and element 1 trial space
   A11.SetSize(te_ndof1*tdim, tr_ndof1*vdim);
   A12.SetSize(te_ndof1*tdim, tr_ndof2*vdim);
   A21.SetSize(te_ndof2*tdim, tr_ndof1*vdim);
   A22.SetSize(te_ndof2*tdim, tr_ndof2*vdim);
   A11 = 0.0;
   A12 = 0.0;
   A21 = 0.0;
   A22 = 0.0;

   // quadrature rule
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      if (tr_ndof2) // interior faces
      {
         order = 2*(std::max(tr_fe1.GetOrder(), tr_fe2.GetOrder()) + std::max(te_fe1.GetOrder(),
                                                                    te_fe2.GetOrder())) + 2;
      }
      else // boundary faces
      {
         order = 2*(tr_fe1.GetOrder() + te_fe1.GetOrder()) + 2;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }


   // sum up all quadrature nodes along with weights
   for (int p=0; p<ir->GetNPoints(); p++)
   {
		const IntegrationPoint &ip = ir->IntPoint(p);
		Trans.SetAllIntPoints(&ip);
		const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
		const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

		// normal
		if (dim == 1)
		{
			nor(0) = 2*eip1.x - 1.0;
		}
		else
		{
			CalcOrtho(Trans.Jacobian(), nor);
		}

		// normalize nor
		nor /= nor.Norml2();

		if (Trans.Elem2No >= 0)
		{
			weight = ip.weight;
		}
		else
		{
			weight = ip.weight*2;
		}
		weight *= Trans.Weight();

		// calculate shape functions in element 1 at current integration point
		tr_fe1.CalcShape(eip1, tr_s1);
		te_fe1.CalcShape(eip1, te_s1);

		// form A11
		for (int dcol = 0; dcol<vdim; dcol++) // for u_{1},...u_{dim}
			for (int drow = 0; drow<vdim; drow++)
				for (int i = 0; i < te_ndof1; i++)
					for (int j = 0; j < tr_ndof1; j++)
						A11(i + te_ndof1*drow + (te_ndof1*vdim)*dcol, j + tr_ndof1*dcol) += 0.5*tr_s1(j)*(te_s1(i)*nor(drow))*weight;

		// if element 2 exists, form the rest of the blocks
		if (tr_ndof2)
		{
			// calc shape functions in element 2 at current integration point
			tr_fe2.CalcShape(eip2, tr_s2);
			te_fe2.CalcShape(eip2, te_s2);

			// form A12
			for (int dcol = 0; dcol<vdim; dcol++) // for u_{1},...u_{dim}
				for (int drow = 0; drow<vdim; drow++)
					for (int i = 0; i < te_ndof1; i++)
						for (int j = 0; j < tr_ndof2; j++)
							A12(i + te_ndof1*drow + (te_ndof1*vdim)*dcol, j + tr_ndof2*dcol) += 0.5*tr_s2(j)*(te_s1(i)*nor(drow))*weight;

			// form A21
			for (int dcol = 0; dcol<vdim; dcol++) // for u_{1},...u_{dim}
				for (int drow = 0; drow<vdim; drow++)
					for (int i = 0; i < te_ndof2; i++)
						for (int j = 0; j < tr_ndof1; j++)
							A21(i + te_ndof2*drow + (te_ndof2*vdim)*dcol, j + tr_ndof1*dcol) -= 0.5*tr_s1(j)*(te_s2(i)*nor(drow))*weight;
			// form A22
			for (int dcol = 0; dcol<vdim; dcol++) // for u_{1},...u_{dim}
				for (int drow = 0; drow<vdim; drow++)
					for (int i = 0; i < te_ndof2; i++)
						for (int j = 0; j < tr_ndof2; j++)
							A22(i + te_ndof2*drow + (te_ndof2*vdim)*dcol, j + tr_ndof2*dcol) -= 0.5*tr_s2(j)*(te_s2(i)*nor(drow))*weight;
		}

	}

   // populate elmat with the blocks computed above
   elmat.AddMatrix(A11,0,0);
   if (tr_ndof2)
   {
      elmat.AddMatrix(A12, 0, vdim*tr_ndof1);
      elmat.AddMatrix(A21, tdim*te_ndof1, 0);
      elmat.AddMatrix(A22, tdim*te_ndof1, vdim*tr_ndof1);
   }
   elmat *=lambda;
}

void DGVectorNormalJumpIntegrator::AssembleFaceMatrix(
const FiniteElement &el1,
const FiniteElement &el2,
FaceElementTransformations &Trans,
DenseMatrix &elmat
)
{
	// dim & dof
	dim = el1.GetDim();
	vdim = dim;
	ndof1 = el1.GetDof();

	// initialization
	shape1.SetSize(ndof1);
	shape1 = 0.0;
	if (Trans.Elem2No >= 0)
	{
		ndof2 = el2.GetDof();
		shape2.SetSize(ndof2);
		shape2 = 0.0;
	}
	else
	{
		ndof2 = 0;
	}

	ndofs = ndof1 + ndof2;
	elmat.SetSize(ndofs*vdim);
	elmat = 0.;

	// elmat = [ A11   A12 ]
	//         [ A21   A22 ]
	// where the blocks corresponds to the terms in the face integral < {p},[v]*nor > from
	// the different elements and trial/test space, i.e.
	// A11 : terms from element 1 test and element 1 trial space
	// A21 : terms from element 2 test and element 1 trial space
	// In this integrator, A21 = A12^T
	A11.SetSize(ndof1*vdim, ndof1*vdim);
	A12.SetSize(ndof1*vdim, ndof2*vdim);
	A21.SetSize(ndof2*vdim, ndof1*vdim);
	A22.SetSize(ndof2*vdim, ndof2*vdim);
	A11 = 0.0;
	A12 = 0.0;
	A21 = 0.0;
	A22 = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  // a simple choice for the integration order
	  int order;
	  if (ndof2)
	  {
		  order = 2 * std::max(el1.GetOrder(), el2.GetOrder());
	  }
	  else
	  {
		  order = 2 * el1.GetOrder();
	  }
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}

	// Computing edge length (or surface area)
	h=0.0;
	for (int p = 0; p < ir->GetNPoints(); p++)
	{
		const IntegrationPoint &ip = ir->IntPoint(p);

		// Set the integration point in the face and the neighboring elements
		Trans.SetAllIntPoints(&ip);
		h = h + ip.weight*Trans.Face->Weight();
	}


	// sum up all quadrature nodes along with weights
	// TODO: assembly can be optimized (refer to github issue #2909)
	for (int p = 0; p < ir->GetNPoints(); p++)
	{
		const IntegrationPoint &ip = ir->IntPoint(p);

		// Set the integration point in the face and the neighboring elements
		Trans.SetAllIntPoints(&ip);

		// Access the neighboring elements' integration points
		// Note: eip2 will only contain valid data if Elem2 exists
		const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
		const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

		el1.CalcShape(eip1, shape1);
		weight = ip.weight * Trans.Weight();	// weight * face Jacobian
		stab_weight = (weight*kappa/h);
	   // A11
		for (int d = 0; d<vdim; d++)
			for (int i = 0; i < ndof1; i++)
				for (int j = 0; j < ndof1; j++)
					A11(i + ndof1*d, j + ndof1*d) += shape1(j)*shape1(i)*stab_weight;

		// if element 2 exists form the rest of the blocks
		if (ndof2)
		{
			el2.CalcShape(eip2, shape2);


			for (int d = 0; d<vdim; d++)
				for (int i = 0; i < ndof1; i++)
					for (int j = 0; j < ndof2; j++)
						A12(i + ndof1*d, j + ndof2*d) -= shape2(j)*shape1(i)*stab_weight;

			A21.Transpose(A12);

			for (int d = 0; d<vdim; d++)
					for (int i = 0; i < ndof2; i++)
						for (int j = 0; j < ndof2; j++)
							A22(i + ndof2*d, j + ndof2*d) += shape2(j)*shape2(i)*stab_weight;
		}
	}

	// populate elmat with the blocks computed above
	elmat.AddMatrix(A11,0,0);
	if (ndof2)
	{
		elmat.AddMatrix(A12, 0, vdim*ndof1);
		elmat.AddMatrix(A21, vdim*ndof1, 0);
		elmat.AddMatrix(A22, vdim*ndof1, vdim*ndof1);
	}
	elmat *= lambda;
}

void DGVectorUpwindJumpIntegrator::AssembleFaceMatrix
(const FiniteElement &el1,
const FiniteElement &el2,
FaceElementTransformations &Trans,
DenseMatrix &elmat)
{

	// dim & dof
	dim = el1.GetDim();
	vdim = dim;
	ndof1 = el1.GetDof();

	// initialization
	shape1.SetSize(ndof1);
	shape1 = 0.0;
	evalQ.SetSize(vdim);
	evalQ = 0.0;
	nor.SetSize(vdim);
	nor = 0.0;
	if (Trans.Elem2No >= 0)
	{
	  ndof2 = el2.GetDof();
	  shape2.SetSize(ndof2);
	  shape2 = 0.0;
	}
	else
	{
	  ndof2 = 0;
	}

	ndofs = ndof1 + ndof2;
	elmat.SetSize(ndofs*vdim);
	elmat = 0.;

	// elmat = [ A11   A12 ]
	//         [ A21   A22 ]
	// where the blocks corresponds to the terms in the face integral < {p},[v]*nor > from
	// the different elements and trial/test space, i.e.
	// A11 : terms from element 1 test and element 1 trial space
	// A21 : terms from element 2 test and element 1 trial space
	// In this integrator, A21 = A12^T
	A11.SetSize(ndof1*vdim, ndof1*vdim);
	A12.SetSize(ndof1*vdim, ndof2*vdim);
	A21.SetSize(ndof2*vdim, ndof1*vdim);
	A22.SetSize(ndof2*vdim, ndof2*vdim);
	A11 = 0.0;
	A12 = 0.0;
	A21 = 0.0;
	A22 = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  int order;
	  if (ndof2)
	  {
		 order = 3 * std::max(el1.GetOrder(), el2.GetOrder());
	  }
	  else
	  {
		 order = 3 * el1.GetOrder();
	  }
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}

	// sum up all quadrature nodes along with weights
	for (int p = 0; p < ir->GetNPoints(); p++)
	{
		const IntegrationPoint &ip = ir->IntPoint(p);

		// Set the integration point in the face and the neighboring elements
		Trans.SetAllIntPoints(&ip);

		// Access the neighboring elements' integration points
		// Note: eip2 will only contain valid data if Elem2 exists
		const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
		const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

		if (dim == 1)
		{
			nor(0) = 2*eip1.x - 1.0;
		}
		else
		{
			CalcOrtho(Trans.Jacobian(), nor);
		}
		// normalize nor.
		nor /= nor.Norml2();

		el1.CalcShape(eip1, shape1);
		weight = ip.weight * Trans.Weight();
		Q->Eval(evalQ, *Trans.Elem1, eip1);
		inner_prod = evalQ * nor;

		// TODO: The upwind flux can also be expressed in the format of
		//       jump-average operator. Here, I simply use if satatement
		//       to explicitly evaluate the upwind state.
		if (ndof2)
		{
			// interior face construction
			el2.CalcShape(eip2, shape2);
			if (inner_prod< 0.0){
				// A12
				for (int d = 0; d<vdim; d++)
					for (int i = 0; i < ndof1; i++)
						for (int j = 0; j < ndof2; j++)
							A12(i + ndof1*d, j + ndof2*d) += shape2(j)*shape1(i)*weight*inner_prod;
				// A22
				for (int d = 0; d<vdim; d++)
					for (int i = 0; i < ndof2; i++)
						for (int j = 0; j < ndof2; j++)
							A22(i + ndof2*d, j + ndof2*d) -= shape2(j)*shape2(i)*weight*inner_prod;
			}else{
				// A11
				for (int d = 0; d<vdim; d++)
					for (int i = 0; i < ndof1; i++)
						for (int j = 0; j < ndof1; j++)
							A11(i + ndof1*d, j + ndof1*d) += shape1(j)*shape1(i)*weight*inner_prod;
				// A21
				for (int d = 0; d<vdim; d++)
						for (int i = 0; i < ndof2; i++)
							for (int j = 0; j < ndof1; j++)
								A21(i + ndof2*d, j + ndof1*d) -= shape1(j)*shape2(i)*weight*inner_prod;

			}
		}else{
			// boundary face construction
			if (inner_prod < 0.0){
				// Neumann boundary at inflow
				// Do nothing and assign zero to A11.
			}else{
				// Neumann boundary at outflow
				// A11
				for (int d = 0; d<vdim; d++)
					for (int i = 0; i < ndof1; i++)
						for (int j = 0; j < ndof1; j++)
							A11(i + ndof1*d, j + ndof1*d) += shape1(j)*shape1(i)*weight*inner_prod;
			}
		}

	}

	// populate elmat with the blocks computed above
	elmat.AddMatrix(A11,0,0);
	if (ndof2)
	{
		elmat.AddMatrix(A12, 0, vdim*ndof1);
		elmat.AddMatrix(A21, vdim*ndof1, 0);
		elmat.AddMatrix(A22, vdim*ndof1, vdim*ndof1);
	}
	elmat *= lambda;
}

void DGAvgNormalJumpIntegrator::AssembleFaceMatrix(const FiniteElement &tr_fe1,
                                                   const FiniteElement &tr_fe2,
                                                   const FiniteElement &te_fe1,
                                                   const FiniteElement &te_fe2,
                                                   FaceElementTransformations &Trans,
                                                   DenseMatrix &elmat)
{
	// dim & dof
    dim = tr_fe1.GetDim();
    vdim = dim;
    tr_ndof1 = tr_fe1.GetDof();
    te_ndof1 = te_fe1.GetDof();

    if (Trans.Elem2No >= 0)
    {
    	tr_ndof2 = tr_fe2.GetDof();
    	te_ndof2 = te_fe2.GetDof();
    }
    else
    {
    	tr_ndof2 = 0;
    	te_ndof2 = 0;
    }

    tr_ndofs = tr_ndof1 + tr_ndof2;
    te_ndofs = te_ndof1 + te_ndof2;

    // initialization
    nor.SetSize(dim);  // unit normal vector
    tr_s1.SetSize(tr_ndof1); // trial shape function on element 1
    tr_s2.SetSize(tr_ndof2); // trial shape function on element 2 (neighbor)
    te_s1.SetSize(te_ndof1); // test shape function on element 1
    te_s2.SetSize(te_ndof2); // test shape function on element 2 (neighbor)
    nor = 0.0;
    tr_s1 = 0.0;
    tr_s2 = 0.0;
    te_s2 = 0.0;
    te_s1 = 0.0;

    elmat.SetSize(te_ndofs*vdim, tr_ndofs);
    elmat = 0.0;

	// elmat = [ A11   A12 ]
	//         [ A21   A22 ]
	// where the blocks corresponds to the terms in the face integral < {p},[v]*nor > from
	// the different elements and trial/test space, i.e.
	// A11 : terms from element 1 test and element 1 trial space
	// A21 : terms from element 2 test and element 1 trial space
	A11.SetSize(te_ndof1*vdim, tr_ndof1);
	A12.SetSize(te_ndof1*vdim, tr_ndof2);
	A21.SetSize(te_ndof2*vdim, tr_ndof1);
	A22.SetSize(te_ndof2*vdim, tr_ndof2);
	A11 = 0.0;
	A12 = 0.0;
	A21 = 0.0;
	A22 = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  int order;
	  if (tr_ndof2)
	  {
		 order = 2*(std::max(tr_fe1.GetOrder(), tr_fe2.GetOrder()) + std::max(te_fe1.GetOrder(),
																	te_fe2.GetOrder())) + 2;
	  }
	  else
	  {
		 order = 2*(tr_fe1.GetOrder() + te_fe1.GetOrder()) + 2;
	  }
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}


	// sum up all quadrature nodes along with weights
	for (int p=0; p<ir->GetNPoints(); p++)
	{
		const IntegrationPoint &ip = ir->IntPoint(p);
		Trans.SetAllIntPoints(&ip);
		const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
		const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

		if (dim == 1)
		{
			nor(0) = 2*eip1.x - 1.0;
		}
		else
		{
			CalcOrtho(Trans.Jacobian(), nor);
		}
		// normalize nor.
		nor /= nor.Norml2();

		// below if statement needed so on the boundary {p} = p (definition of {} operator)
		if (Trans.Elem2No >= 0)
		{
			weight = ip.weight;
		}
		else
		{
			weight = ip.weight*2;
		}
		weight *= Trans.Weight();

		// calc shape functions in element 1 at current integration point
		tr_fe1.CalcShape(eip1, tr_s1);
		te_fe1.CalcShape(eip1, te_s1);

		// form A11
		for (int d = 0; d<vdim; d++)
			for (int i = 0; i < te_ndof1; i++)
				for (int j = 0; j < tr_ndof1; j++)
					A11(i + te_ndof1*d,j) += 0.5*tr_s1(j)*(te_s1(i)*nor(d))*weight;

		// if element 2 exists form the rest of the blocks
		if (tr_ndof2)
		{
			// calc shape functions in element 2 at current integration point
			tr_fe2.CalcShape(eip2, tr_s2);
			te_fe2.CalcShape(eip2, te_s2);

			// form A12
			for (int d = 0; d<vdim; d++)
				for (int i = 0; i < te_ndof1; i++)
					for (int j = 0; j < tr_ndof2; j++)
						A12(i + te_ndof1*d,j) += 0.5*tr_s2(j)*(te_s1(i)*nor(d))*weight;

			// form A21
			for (int d = 0; d<vdim; d++)
				for (int i = 0; i < te_ndof2; i++)
					for (int j = 0; j < tr_ndof1; j++)
						A21(i + te_ndof2*d,j) += -1.0*0.5*tr_s1(j)*(te_s2(i)*nor(d))*weight;

			// form A22
			for (int d = 0; d<vdim; d++)
				for (int i = 0; i < te_ndof2; i++)
					for (int j = 0; j < tr_ndof2; j++)
						A22(i + te_ndof2*d,j) += -1.0*0.5*tr_s2(j)*(te_s2(i)*nor(d))*weight;
		}
	}

	// populate elmat with the blocks computed above
	elmat.AddMatrix(A11,0,0);
	if (tr_ndof2)
	{
	  elmat.AddMatrix(A12, 0, tr_ndof1);
	  elmat.AddMatrix(A21, vdim*te_ndof1, 0);
	  elmat.AddMatrix(A22, vdim*te_ndof1, tr_ndof1);
	}
}

void TensorDGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Trans, Vector &elvect)
{
   mfem_error("TensorDGDirichletLFIntegrator::AssembleRHSElementVect is not implemented.");
}

void TensorDGDirichletLFIntegrator::AssembleRHSElementVect(
const FiniteElement &el,
FaceElementTransformations &Trans,
Vector &elvect
)
{
	// dof & dim
	dim = el.GetDim();
	vdim = dim;
	tdim = vdim*vdim;
	ndof = el.GetDof();

	// initialization
	nor.SetSize(dim);
	evalQ.SetSize(dim);
	shape.SetSize(ndof);
	nor = 0.0;
	evalQ = 0.0;
	shape = 0.0;

	elvect.SetSize(tdim*ndof);
	elvect = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  int order = 3*el.GetOrder() + 2;
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}

	// sum up all quadrature nodes along with weights
	for (int p = 0; p < ir->GetNPoints(); p++)
	{
		const IntegrationPoint &ip = ir->IntPoint(p);

		// Set the integration point in the face and the neighboring element
		Trans.SetAllIntPoints(&ip);

		// Access the neighboring element's integration point
		const IntegrationPoint &eip = Trans.GetElement1IntPoint();

		if (dim == 1)
		{
		 nor(0) = 2*eip.x - 1.0;
		}
		else
		{
		 CalcOrtho(Trans.Jacobian(), nor); // normal in reference space
		}

		// normalize nor.
		nor /= nor.Norml2();

		// eval coefficient Q and shape functions at current integration point
		Q->Eval(evalQ, Trans, ip);
		el.CalcShape(eip, shape);
		weight = Trans.Weight()*ip.weight;

		for (int d1=0; d1<vdim; d1++) // u_{1} ... u_{dim} (Dirichlet data)
			for (int d2=0; d2<vdim; d2++)
				for (int i=0; i<ndof; i++)
					elvect(d1*(vdim*ndof) + d2*ndof + i) += evalQ(d1)*nor(d2)*shape(i)*weight;
	}
	elvect *= lambda;
}

void VectorDGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Trans, Vector &elvect)
{
   mfem_error("VectorDGDirichletLFIntegrator::AssembleRHSElementVect is not implemented.");
}

void VectorDGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
	// dim & dof
	dim  = el.GetDim();
	vdim = dim;
	ndof = el.GetDof();

	// initialization
	shape.SetSize(ndof);
	evaluD.SetSize(vdim);
	evalQ.SetSize(vdim);
	nor.SetSize(vdim);
	shape = 0.0;
	evaluD = 0.0;
	evalQ = 0.0;
	nor = 0.0;

	elvect.SetSize(vdim*ndof);
	elvect = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  int order = 3*el.GetOrder() + 2;
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}

	// Computing edge length (or surface area)
	double h=0.0;
	if (Q==NULL){
		for (int p = 0; p < ir->GetNPoints(); p++)
		{
			const IntegrationPoint &ip = ir->IntPoint(p);

			// Set the integration point in the face and the neighboring elements
			Trans.SetAllIntPoints(&ip);
			h = h + ip.weight*Trans.Face->Weight();
		}
	}

	// sum up all quadrature nodes along with weights
	for (int p = 0; p < ir->GetNPoints(); p++)
	{
		const IntegrationPoint &ip = ir->IntPoint(p);

		// Set the integration point in the face and the neighboring element
		Trans.SetAllIntPoints(&ip);

		// Access the neighboring element's integration point
		const IntegrationPoint &eip = Trans.GetElement1IntPoint();

		if (dim == 1)
		{
			nor(0) = 2*eip.x - 1.0;
		}
		else
		{
			CalcOrtho(Trans.Jacobian(), nor); // normal in reference space
		}
		// normalize nor.
		nor /= nor.Norml2();

		el.CalcShape(eip, shape);

		// compute uD through the face transformation
		uD->Eval(evaluD, Trans, ip);

		if (Q == NULL){
			// do nothing
		}else{
			Q->Eval(evalQ, *Trans.Elem1, eip);
			inner_prod = evalQ*nor;

					// sanity check for the condition of inflow boundaries
		//			if(wn > 1e-14){
		//				cout << "w \cdot n : " << wn <<endl;
		//				mfem_error("w \cdot n is greater than zero. Not a inflow boundary!");
		//			}
		}

		weight = ip.weight * Trans.Weight();	// weight * face Jacobian
		for (int d=0; d<vdim; d++)
			for(int i=0; i<ndof; i++){
				if (Q == NULL){
					elvect(i+d*ndof) += shape(i)*(weight*kappa/h)*evaluD(d);
				}else{
					elvect(i+d*ndof) += shape(i)*weight*evaluD(d)*inner_prod;
				}
			}
	}
	elvect *= lambda;
}

void VectorDGNeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Trans, Vector &elvect)
{
   mfem_error("VectorDGNeumannLFIntegrator::AssembleRHSElementVect is not implemented.");
}

void VectorDGNeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
	// dim & dof
	dim = el.GetDim();
	ndof = el.GetDof();

	int vdim=dim;
	int tdim=dim*dim;

	// initialization
	shape.SetSize(ndof);
	evalQ.SetSize(tdim);
	nor.SetSize(dim);
	evalQ = 0.0;
	shape = 0.0;
	nor = 0.0;

	elvect.SetSize(vdim*ndof);
	elvect = 0.0;

	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  // a simple choice for the integration order; is this OK?
	  int order = 3*el.GetOrder() + 2;
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}


	for (int p = 0; p < ir->GetNPoints(); p++)
	{
		const IntegrationPoint &ip = ir->IntPoint(p);

		// Set the integration point in the face and the neighboring element
		Trans.SetAllIntPoints(&ip);

		// Access the neighboring element's integration point
		const IntegrationPoint &eip = Trans.GetElement1IntPoint();
		if (dim == 1)
		{
		 nor(0) = 2*eip.x - 1.0;
		}
		else
		{
		 CalcOrtho(Trans.Jacobian(), nor); // normal in reference space
		}
		// normalize nor.
		nor /= nor.Norml2();

		el.CalcShape(eip, shape);
		// compute uD through the face transformation
		Q->Eval(evalQ, Trans,ip);

		weight = ip.weight * Trans.Weight();	// weight * face Jacobian
		for (int te_d=0; te_d<vdim; te_d++)
			for (int tr_d=0; tr_d<vdim; tr_d++)
				for(int i=0; i<ndof; i++)
						elvect(i+te_d*ndof) += evalQ(te_d*vdim + tr_d)*nor(tr_d)*shape(i)*weight;
	}
	elvect *= lambda;
}

void BoundaryNormalLFIntegrator_mod::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Trans, Vector &elvect)
{
	mfem_error("BoundaryNormalLFIntegrator_mod::AssembleRHSElementVect is not implemented.");
}

void BoundaryNormalLFIntegrator_mod::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
	// dim & dof
	dim = el.GetDim();
	ndof = el.GetDof();

	// initialization
	nor.SetSize(dim);
	evalQ.SetSize(dim);
	nor = 0.0;
	evalQ = 0.0;

	shape.SetSize(ndof);
	elvect.SetSize(ndof);
	shape = 0.0;
	elvect = 0.0;

	// quadrature rule
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
	  // a simple choice for the integration order;
	  int order = 3*el.GetOrder() + 2;
	  ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}

	// sum up all quadrature nodes along with weights
	for (int p = 0; p < ir->GetNPoints(); p++)
	{
		const IntegrationPoint &ip = ir->IntPoint(p);

		// Set the integration point in the face and the neighboring element
		Trans.SetAllIntPoints(&ip);

		// Access the neighboring element's integration point
		const IntegrationPoint &eip = Trans.GetElement1IntPoint();
		if (dim == 1)
		{
		 nor(0) = 2*eip.x - 1.0;
		}
		else
		{
		 CalcOrtho(Trans.Jacobian(), nor); // normal in reference space
		}
		// normalize nor.
		nor /= nor.Norml2();

		Q.Eval(evalQ, Trans, ip);
		el.CalcShape(eip, shape);
		weight = Trans.Weight()*ip.weight;

		elvect.Add(weight*(evalQ*nor), shape);
	}
	elvect *= lambda;
}

} // end of name space "mfem"
