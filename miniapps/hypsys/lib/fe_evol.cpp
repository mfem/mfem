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
   BdrInt.SetSize(dofs.NumBdrs, nqf, ne);
	ElemNor.SetSize(dim, dofs.NumBdrs, ne); // Generalizeable to curved faces.

   MassMat = new MassMatrixDG(fes);
   InvMassMat = new InverseMassMatrixDG(MassMat);

   uElem.SetSize(nd);
   uEval.SetSize(hyp->NumEq);
   uNbr.SetSize(hyp->NumEq);
   uNbrEval.SetSize(hyp->NumEq);
   vec1.SetSize(dim);
   vec2.SetSize(hyp->NumEq*nd);
	Flux.SetSize(hyp->NumEq, dim);
	FluxNbr.SetSize(hyp->NumEq, dim);
	mat1.SetSize(dim, hyp->NumEq);
	mat2.SetSize(nd, hyp->NumEq);

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
				BdrInt(i,k,e) = facetrans->Face->Weight();
				
            for (int l = 0; l < dim; l++)
            {
					ElemNor(l,i,e) = nor(l);
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

void FE_Evolution::EvaluateSolution(const Vector &x, Vector &y, int k) const
{
	y = 0.;
	for (int n = 0; n < hyp->NumEq; n++)
	{
		for (int j = 0; j < nd; j++)
		{
			y(n) += x(n*nd+j) * ShapeEval(j,k);
		}
	}
}

void FE_Evolution::EvaluateSolution(const Vector &x, Vector &y1, Vector &y2,
												int e, int i, int k) const
{
	y1 = y2 = 0.;
	for (int n = 0; n < hyp->NumEq; n++)
	{
		for (int j = 0; j < dofs.NumFaceDofs; j++)
		{
			nbr = dofs.NbrDofs(i,j,e);
			DofInd = e*nd+dofs.BdrDofs(j,i);
			if (nbr < 0)
			{
				uNbr(0) = hyp->inflow(DofInd); // TODO vector valued
			}
			else
			{
				uNbr(0) = x(nbr); // TODO vector valued
			}
			
			uEval(n) += x(DofInd) * ShapeEvalFace(i,j,k);
         uNbrEval(n) += uNbr(n) * ShapeEvalFace(i,j,k);
		}
	}
}

void FE_Evolution::LaxFriedrichs(const Vector &x1, const Vector &x2,
											const Vector &normal, Vector &y) const
{
	hyp->EvaluateFlux(x1, Flux);
	hyp->EvaluateFlux(x2, FluxNbr);
	Flux += FluxNbr;
	double ws = max( hyp->GetWaveSpeed(x1, normal), 
						  hyp->GetWaveSpeed(x2, normal) );
	Flux.Mult(normal, y);
	
	Vector x(y.Size());
	subtract(ws, x1, x2, x);
	y += x;
	y *= 0.5;
}

double FE_Evolution::ConvergenceCheck(double dt, double tol,
                                      const Vector &u) const
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
		mat2 = 0.;

      DenseMatrix vel = hyp->VelElem(e);

      for (int k = 0; k < nqe; k++)
      {
			EvaluateSolution(uElem, uEval, k);
			
// 			hyp->EvaluateFlux(uEval, Flux);

			vec1 = vel.GetColumn(k);
			vec1 *= uEval(0);
			Flux.SetRow(0, vec1);

			MultABt(ElemInt(nqe*e+k), Flux, mat1);
			AddMult(DShapeEval(k), mat1, mat2);
      }

      vec2 = mat2.GetData();
      z.AddElementVector(vdofs, vec2);
		
		DenseMatrix ElemNormals = ElemNor(e); // TODO allocate

      // Here, the use of nodal basis functions is essential, i.e. shape
      // functions must vanish on faces that their node is not associated with.
      for (int i = 0; i < dofs.NumBdrs; i++)
      {
			Vector FaceNor(dim); // TODO allocate
			ElemNormals.GetColumn(i, FaceNor);
         for (int k = 0; k < nqf; k++)
         {
            double tmp = 0.;
            for (int l = 0; l < dim; l++)
            {
               tmp += FaceNor(l) * hyp->VelFace(l,i,e*nqf+k);
            }

            EvaluateSolution(x, uEval, uNbrEval, e, i, k);
				
// 				Vector NumFlux(hyp->NumEq); // TODO allocate
// 				LaxFriedrichs(uEval, uNbrEval, FaceNor, NumFlux);

            // Lax-Friedrichs flux (equals full upwinding for Advection).
            tmp = 0.5 * ( tmp * (uEval(0) + uNbrEval(0)) + abs(tmp) 
						* (uEval(0) - uNbrEval(0)) );
// 
				tmp *= BdrInt(i,k,e) * QuadWeightFace(k);
				
				for (int n = 0; n < hyp->NumEq; n++)
				{
					for (int j = 0; j < dofs.NumFaceDofs; j++)
					{
               	z(vdofs[dofs.BdrDofs(j,i)]) -= ShapeEvalFace(i,j,k) * tmp;

// 						z(vdofs[n*nd+dofs.BdrDofs(j,i)]) -= ShapeEvalFace(i,j,k)
// 							* NumFlux(n) * BdrInt(i,k,e) * QuadWeightFace(k);
					}
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
