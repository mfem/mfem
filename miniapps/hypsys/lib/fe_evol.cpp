#include "fe_evol.hpp"

FE_Evolution::FE_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_, 
									DofInfo &dofs_, EvolutionScheme scheme_,
									const Vector &LumpedMassMat_)
									: TimeDependentOperator(fes_->GetVSize()),
									  fes(fes_), hyp(hyp_), dofs(dofs_),
									  scheme(scheme_), LumpedMassMat(LumpedMassMat_),
									  z(fes_->GetVSize())
									  
{
	const char* fecol = fes->FEColl()->Name();
	if (strncmp(fecol, "L2", 2))
	{
		MFEM_ABORT("FiniteElementSpace must be L2 conforming (DG).");
	}
	if (strncmp(fecol, "L2_T2", 5))
	{
		MFEM_ABORT("Shape functions must be represented in Bernstein basis.");
	}
	
	// Initialize member variables.
	IntRuleElem = GetElementIntegrationRule(fes);
	IntRuleFace = GetFaceIntegrationRule(fes);
	
	Mesh *mesh = fes->GetMesh();
	const FiniteElement *el = fes->GetFE(0);
	
	dim = mesh->Dimension();
	nd = el->GetDof();
	ne = mesh->GetNE();
	nqe = IntRuleElem->GetNPoints();
	nqf = IntRuleFace->GetNPoints();
	QuadWeightFace.SetSize(nqf);
	
	ShapeEval.SetSize(nd,nqe);
	DShapeEval.SetSize(nd,dim,nqe);
	ShapeEvalFace.SetSize(dofs.NumBdrs, dofs.NumFaceDofs, nqf);
	
	ElemInt.SetSize(dim, dim, ne*nqe);
	BdrInt.SetSize(dim, dofs.NumBdrs, ne*nqf);
	
	MassMat = new MassMatrixDG(fes);
	InvMassMat = new InverseMassMatrixDG(MassMat);
	
	uElem.SetSize(nd);
	uEval.SetSize(1); // TODO Vector valued soultion.
	uNbr.SetSize(1);  // TODO Vector valued soultion.
	vec1.SetSize(dim);
	vec2.SetSize(nd);
	vec3.SetSize(nd);

	// Precompute data that is constant for the whole run.
	Array <int> bdrs, orientation;
	Vector shape(nd);
	DenseMatrix dshape(nd,dim);
	DenseMatrix adjJ(dim);
	Array<IntegrationPoint> eip(nqf*dofs.NumBdrs);
	
	// Fill eip, to be used for evaluation of shape functions on element faces.
	if (dim==1)      { mesh->GetElementVertices(0, bdrs); }
   else if (dim==2) { mesh->GetElementEdges(0, bdrs, orientation); }
   else if (dim==3) { mesh->GetElementFaces(0, bdrs, orientation); }
	
	for (int i = 0; i < dofs.NumBdrs; i++)
	{
		FaceElementTransformations *help
			= mesh->GetFaceElementTransformations(bdrs[i]);
			
		if (help->Elem1No != 0)
		{
			// NOTE: If this error ever occurs, use neighbor element to
			// obtain the correct quadrature points and weight.
			MFEM_ABORT("First element has inward pointing normal.");
		}
		for (int k = 0; k < nqf; k++)
		{
			const IntegrationPoint &ip = IntRuleFace->IntPoint(k);
			help->Loc1.Transform(ip, eip[i*nqf + k]);
		}
	}
	
	// Precompute evaluations of shape functions on elements.
	for (int k = 0; k < nqe; k++)
	{
		const IntegrationPoint &ip = IntRuleElem->IntPoint(k);
		el->CalcShape(ip, shape);
		el->CalcDShape(ip, dshape);
		ShapeEval.SetCol(k, shape);
		DShapeEval(k) = dshape;
	}
	
	// Precompute evaluations of shape functions on element faces.
	for (int k = 0; k < nqf; k++)
	{
		const IntegrationPoint &ip = IntRuleFace->IntPoint(k);
		QuadWeightFace(k) = ip.weight;
		
		if (dim==1)      { mesh->GetElementVertices(0, bdrs); }
		else if (dim==2) { mesh->GetElementEdges(0, bdrs, orientation); }
		else if (dim==3) { mesh->GetElementFaces(0, bdrs, orientation); }
		
		for (int i = 0; i < dofs.NumBdrs; i++)
		{
			FaceElementTransformations *facetrans = 
			mesh->GetFaceElementTransformations(bdrs[i]);
			
			IntegrationPoint eip;
			facetrans->Face->SetIntPoint(&ip);
			facetrans->Loc1.Transform(ip, eip);
			el->CalcShape(eip, shape);
			
			for (int j = 0; j < dofs.NumFaceDofs; j++)
			{
				ShapeEvalFace(i,j,k) = shape(dofs.BdrDofs(j,i));
			}
		}
	}
	
	// Compute element and boundary contributions (without shape functions).
	for (int e = 0; e < ne; e++)
	{
		const FiniteElement *el = fes->GetFE(e);
		ElementTransformation *eltrans = fes->GetElementTransformation(e);
		
		for (int k = 0; k < nqe; k++)
		{
			const IntegrationPoint &ip = IntRuleElem->IntPoint(k);
			eltrans->SetIntPoint(&ip);
			CalcAdjugate(eltrans->Jacobian(), adjJ);
			adjJ *= ip.weight;
			ElemInt(nqe*e+k) = adjJ;
		}
		
		if (dim==1)      { mesh->GetElementVertices(e, bdrs); }
      else if (dim==2) { mesh->GetElementEdges(e, bdrs, orientation); }
      else if (dim==3) { mesh->GetElementFaces(e, bdrs, orientation); }
		
		for (int i = 0; i < dofs.NumBdrs; i++)
		{
			Vector vval, nor(dim);
			FaceElementTransformations *facetrans
				= mesh->GetFaceElementTransformations(bdrs[i]);
			
			for (int k = 0; k < nqf; k++)
			{
				const IntegrationPoint &ip = IntRuleFace->IntPoint(k);
				facetrans->Face->SetIntPoint(&ip);
				
				if (dim == 1)
				{
					IntegrationPoint aux;
					facetrans->Loc1.Transform(ip, aux);
					nor(0) = 2.*aux.x - 1.0;
				}
				else
				{
					CalcOrtho(facetrans->Face->Jacobian(), nor);
				}
				
				if (facetrans->Elem1No != e)
				{
					nor *= -1.;
				}
				
				nor /= nor.Norml2();
				for (int l = 0; l < dim; l++)
				{
					BdrInt(l,i,e*nqf+k) = facetrans->Face->Weight() * nor(l);
				}
			}
		}
	}
}

FE_Evolution::~FE_Evolution()
{
	delete MassMat;
	delete InvMassMat;
}


void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
	switch (scheme)
	{
		case 0: // Standard Finite Element Approximation.
		{
			EvolveStandard(x, y);
			break;
		}
		case 1: // Monolithic Convex Limiting.
		{
			EvolveMCL(x, y);
			break;
		}
		default:
		{
			MFEM_ABORT("Unknown Evolution Scheme.");
		}
	}
}

double FE_Evolution::ConvergenceCheck(double dt, double tol, const Vector &u) const
{
	z = u;
	z -= uOld;
	
	double res = 0.;
	if (scheme == 0) // Standard, i.e. use consistent mass matrix.
	{
		MassMat->Mult(z, uOld);
		res = uOld.Norml2() / dt;
	}
	else // use lumped mass matrix.
	{
		for (int i = 0; i < u.Size(); i++)
		{
			res += pow(LumpedMassMat(i) * z(i), 2.);
		}
		res = sqrt(res) / dt;
	}
	
	uOld = u;
	return res;
}

void FE_Evolution::EvolveStandard(const Vector &x, Vector &y) const
{
	z = 0.;

	for (int e = 0; e < fes->GetNE(); e++)
	{
		fes->GetElementVDofs(e, vdofs);
		x.GetSubVector(vdofs, uElem);
		vec3 = 0.;
		DenseMatrix vel = hyp->VelElem(e);
		
		for (int k = 0; k < nqe; k++)
		{
			ShapeEval.GetColumn(k, vec2);
			uEval(0) = uElem * vec2;
			
			ElemInt(nqe*e+k).Mult(vel.GetColumn(k), vec1);
			DShapeEval(k).Mult(vec1, vec2);
			vec2 *= uEval(0);
			vec3 += vec2;
		}
		
		z.AddElementVector(vdofs, vec3); // TODO Vector valued soultion.

		// Here, the use of nodal basis functions is essential, i.e. shape
		// functions must vanish on faces that their node is not associated with.
		for (int i = 0; i < dofs.NumBdrs; i++)
		{
			for (int k = 0; k < nqf; k++)
			{
				double tmp = 0.;
				for (int l = 0; l < dim; l++)
				{
					tmp += BdrInt(l,i,e*nqf+k) * hyp->VelFace(l,i,e*nqf+k);
				}
				
				uEval = 0.;
				if (tmp >= 0)
				{
					for (int j = 0; j < dofs.NumFaceDofs; j++)
					{
						uEval(0) += uElem(dofs.BdrDofs(j,i)) * ShapeEvalFace(i,j,k);
					}
				}
				else
				{
					for (int j = 0; j < dofs.NumFaceDofs; j++)
					{
						DofInd = e*nd+dofs.BdrDofs(j,i);
						nbr = dofs.NbrDofs(i,j,e);
                  if (nbr < 0)
						{
							uNbr(0) = hyp->inflow(DofInd);
						}
                  else
                  {
                     uNbr(0) = x(nbr);
                  }
						uEval(0) += uNbr(0) * ShapeEvalFace(i,j,k);
					}
				}
				
				tmp *= QuadWeightFace(k) * uEval(0);
				
				for (int j = 0; j < dofs.NumFaceDofs; j++)
				{
					z(vdofs[dofs.BdrDofs(j,i)]) -= ShapeEvalFace(i,j,k) * tmp;
				}
			}
		}
	}

	InvMassMat->Mult(z, y);
}

void FE_Evolution::EvolveMCL(const Vector &x, Vector &y) const
{
	MFEM_ABORT("TODO.");
}
