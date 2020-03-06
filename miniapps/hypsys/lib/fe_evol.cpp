#include "fe_evol.hpp"

FE_Evolution::FE_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                           DofInfo &dofs_, EvolutionScheme scheme_,
                           const Vector &LumpedMassMat_)
   : TimeDependentOperator(fes_->GetVSize()),
     fes(fes_), hyp(hyp_), dofs(dofs_),
     scheme(scheme_), LumpedMassMat(LumpedMassMat_),
     z(fes_->GetVSize()), inflow(fes_)
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

   ShapeEval.SetSize(nd,nqe);
   DShapeEval.SetSize(nd,dim,nqe);
   ShapeEvalFace.SetSize(dofs.NumBdrs, dofs.NumFaceDofs, nqf);

   ElemInt.SetSize(dim, dim, ne*nqe);
   BdrInt.SetSize(dofs.NumBdrs, nqf, ne);
   OuterUnitNormals.SetSize(dim, nqf, ne*dofs.NumBdrs);

   MassMat = new MassMatrixDG(fes);
   InvMassMat = new InverseMassMatrixDG(MassMat);

   uElem.SetSize(nd);
   uEval.SetSize(hyp->NumEq);
   uNbrEval.SetSize(hyp->NumEq);
   normal.SetSize(dim);
   NumFlux.SetSize(hyp->NumEq);

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
         ElemInt(e*nqe+k) = adjJ;
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
            BdrInt(i,k,e) = facetrans->Face->Weight() * ip.weight;

            for (int l = 0; l < dim; l++)
            {
               OuterUnitNormals(l,k,e*dofs.NumBdrs+i) = nor(l);
            }
         }
      }
   }
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

void FE_Evolution::ElemEval(const Vector &uElem, Vector &uEval, int k) const
{
   uEval = 0.;
   for (int n = 0; n < hyp->NumEq; n++)
   {
      for (int j = 0; j < nd; j++)
      {
         uEval(n) += uElem(n*nd+j) * ShapeEval(j,k);
      }
   }
}

void FE_Evolution::FaceEval(const Vector &x, Vector &y1, Vector &y2,
                            int e, int i, int k) const
{
   y1 = y2 = 0.;
   for (int n = 0; n < hyp->NumEq; n++)
   {
      for (int j = 0; j < dofs.NumFaceDofs; j++)
      {
         nbr = dofs.NbrDofs(i,j,e);
         DofInd = n*ne*nd + e*nd+dofs.BdrDofs(j,i);

         if (nbr < 0)
         {
            // TODO more general boundary conditions, Riemann problem
            uNbr = inflow(DofInd);
         }
         else
         {
            uNbr = x(n*ne*nd + nbr);
         }

         y1(n) += x(DofInd) * ShapeEvalFace(i,j,k);
         y2(n) += uNbr * ShapeEvalFace(i,j,k);
      }
   }
}

void FE_Evolution::LaxFriedrichs(const Vector &x1, const Vector &x2,
                                 const Vector &normal, Vector &y,
                                 int e, int k, int i) const
{
   hyp->EvaluateFlux(x1, Flux, e, k, i);
   hyp->EvaluateFlux(x2, FluxNbr, e, k, i);
   Flux += FluxNbr;
   double ws = max( hyp->GetWaveSpeed(x1, normal, e, k, i),
                    hyp->GetWaveSpeed(x2, normal, e, k, i) );

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
   if (hyp->TimeDepBC)
   {
      hyp->BdrCond.SetTime(t);
      if (!hyp->ProjType)
      {
         hyp->L2_Projection(hyp->BdrCond, inflow);
      }
      else
      {
         inflow.ProjectCoefficient(hyp->BdrCond);
      }
   }

   z = 0.;
   for (int e = 0; e < ne; e++)
   {
      fes->GetElementVDofs(e, vdofs);
      x.GetSubVector(vdofs, uElem);
      mat2 = 0.;

      for (int k = 0; k < nqe; k++)
      {
         ElemEval(uElem, uEval, k);
         hyp->EvaluateFlux(uEval, Flux, e, k);
         MultABt(ElemInt(e*nqe+k), Flux, mat1);
         AddMult(DShapeEval(k), mat1, mat2);
      }

      z.AddElementVector(vdofs, mat2.GetData());

      // Here, the use of nodal basis functions is essential, i.e. shape
      // functions must vanish on faces that their node is not associated with.
      for (int i = 0; i < dofs.NumBdrs; i++)
      {
         for (int k = 0; k < nqf; k++)
         {
            OuterUnitNormals(e*dofs.NumBdrs+i).GetColumn(k, normal);
            FaceEval(x, uEval, uNbrEval, e, i, k);
            LaxFriedrichs(uEval, uNbrEval, normal, NumFlux, e, k, i);
            NumFlux *= BdrInt(i,k,e);

            for (int n = 0; n < hyp->NumEq; n++)
            {
               for (int j = 0; j < dofs.NumFaceDofs; j++)
               {
                  z(vdofs[n*nd+dofs.BdrDofs(j,i)]) -= ShapeEvalFace(i,j,k)
                                                      * NumFlux(n);
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
