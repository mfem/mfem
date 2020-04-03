#include "monolithic_convex_limiting.hpp"

MCL_Evolution::MCL_Evolution(FiniteElementSpace *fes_,
                             HyperbolicSystem *hyp_,
                             DofInfo &dofs_)
   : FE_Evolution(fes_, hyp_, dofs_)
{
   Mesh *mesh = fes->GetMesh();
   const FiniteElement *el = fes->GetFE(0);
   IntegrationRule nodes =  el->GetNodes();
   Vector shape(nd), dx_shape(nd), RefMassLumped(nd);
   DenseMatrix dshape(nd,dim), GradOp(nd,nd), RefMass(nd,nd), InvRefMass(nd,nd);
   DenseTensor Grad_aux(nd,nd,dim);

   ShapeEval.SetSize(nd,nd);
   ElemInt.SetSize(dim,dim,ne);
   PrecGrad.SetSize(nd,dim,nd);
   CTilde.SetSize(nd,dim,nd);
   OuterUnitNormals.SetSize(dim, dofs.NumBdrs, ne);
   C_eij(dim);

   nscd = nscd = dofs.SubcellCross.Width();

   RefMassLumped =  0.;
   RefMass = 0.;
   PrecGrad = 0.;
   Grad_aux = 0.;

   Array <int> bdrs, orientation;

   for (int j = 0; j < nd; j++)
   {
      const IntegrationPoint &ip = nodes.IntPoint(j);
      el->CalcShape(ip, shape);
      ShapeEval.SetCol(j, shape);
   }

   for (int k = 0; k < nqe; k++)
   {
      const IntegrationPoint &ip = IntRuleElem->IntPoint(k);
      el->CalcShape(ip, shape);
      el->CalcDShape(ip, dshape);
      RefMassLumped.Add(ip.weight, shape);
      AddMult_a_VVt(ip.weight, shape, RefMass);

      for (int l = 0; l < dim; l++)
      {
         dshape.GetColumn(l, dx_shape);
         AddMult_a_VWt(ip.weight, shape, dx_shape, Grad_aux(l));
      }
   }

   DenseMatrixInverse inv(&RefMass);
   inv.Factor();
   inv.GetInverseMatrix(InvRefMass);

   for (int l = 0; l < dim; l++)
   {
      GradOp = 0.;
      AddMult(InvRefMass, Grad_aux(l), GradOp);
      GradOp.LeftScaling(RefMassLumped);

      for (int i = 0; i < nd; i++)
      {
         for (int j = 0; j < nd; j++)
         {
            PrecGrad(i,l,j) -= GradOp(i,j);
         }
      }
   }

   for (int e = 0; e < ne; e++)
   {
      ElementTransformation *eltrans = fes->GetElementTransformation(e);
      const IntegrationPoint &ip = IntRuleElem->IntPoint(0);
      eltrans->SetIntPoint(&ip);
      DenseMatrix mat_aux(dim,dim);
      CalcAdjugate(eltrans->Jacobian(), mat_aux);
      mat_aux.Transpose();
      ElemInt(e) = mat_aux;

      if (dim==1)      { mesh->GetElementVertices(e, bdrs); }
      else if (dim==2) { mesh->GetElementEdges(e, bdrs, orientation); }
      else if (dim==3) { mesh->GetElementFaces(e, bdrs, orientation); }

      for (int i = 0; i < dofs.NumBdrs; i++)
      {
         Vector nor(dim);
         FaceElementTransformations *facetrans
            = mesh->GetFaceElementTransformations(bdrs[i]);

         const IntegrationPoint &ip = IntRuleFace->IntPoint(0);
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
            OuterUnitNormals(l,i,e) = nor(l);
         }
      }
   }

   // Construct the P1 subcell mass matrix of the reference element.
   MassMatLOR.SetSize(nd,nd);
   MassMatLOR = 0.;
   DenseMatrix RefMat(dofs.numDofsSubcell, dofs.numDofsSubcell);
   Geometry::Type gtype = el->GetGeomType();

   switch (gtype)
   {
      case Geometry::SEGMENT:
      {
         RefMat = 1.;
         RefMat(0,0) = RefMat(1,1) = 2.;
         RefMat *= 1. / 6.;
         break;
      }
      case Geometry::TRIANGLE:
      {
         RefMat = 1.;
         RefMat(0,0) = RefMat(1,1) = RefMat(2,2) = 2.;
         RefMat *= 1. / 24.;
         break;
      }
      case Geometry::SQUARE:
      {
         RefMat = 2.;
         RefMat(0,0) = RefMat(1,1) = RefMat(2,2) = RefMat(3,3) = 4.;
         RefMat(0,3) = RefMat(1,2) = RefMat(2,1) = RefMat(3,0) = 1.;
         RefMat *= 1. / 36.;
         break;
      }
      case Geometry::CUBE:
      {
         RefMat = 2.;
         RefMat(0,0) = RefMat(1,1) = RefMat(2,2) = RefMat(3,3) = 8.;
         RefMat(4,4) = RefMat(5,5) = RefMat(6,6) = RefMat(7,7) = 8.;
         RefMat(0,7) = RefMat(1,6) = RefMat(2,5) = RefMat(3,4) = 1.;
         RefMat(4,3) = RefMat(5,2) = RefMat(6,1) = RefMat(7,0) = 1.;
         RefMat(0,1) = RefMat(0,2) = RefMat(1,0) = RefMat(1,3) = 4.;
         RefMat(2,0) = RefMat(2,3) = RefMat(3,1) = RefMat(3,2) = 4.;
         RefMat(4,5) = RefMat(4,6) = RefMat(5,4) = RefMat(5,7) = 4.;
         RefMat(6,4) = RefMat(6,7) = RefMat(7,5) = RefMat(7,6) = 4.;
         RefMat(0,4) = RefMat(1,5) = RefMat(2,6) = RefMat(3,7) = 4.;
         RefMat(4,0) = RefMat(5,1) = RefMat(6,2) = RefMat(7,3) = 4.;
         RefMat *= 1. / 216.;
         break;
      }
   }

   RefMat *= 1. / ((double) dofs.numSubcells);

   for (int m = 0; m < dofs.numSubcells; m++)
   {
      for (int i = 0; i < dofs.numDofsSubcell; i++)
      {
         int I = dofs.Sub2Ind(m,i);
         for (int j = 0; j < dofs.numDofsSubcell; j++)
         {
            int J = dofs.Sub2Ind(m,j);
            MassMatLOR(I,J) += RefMat(i,j);
         }
      }
   }
}

void MCL_Evolution::Mult(const Vector &x, Vector &y) const
{
   ComputeTimeDerivative(x, y);
}

void MCL_Evolution::ElemEval(const Vector &uElem, Vector &uEval, int k) const
{
   for (int n = 0; n < hyp->NumEq; n++)
   {
      uEval(n) = uElem(n*nd+k);
   }
}

void MCL_Evolution::FaceEval(const Vector &x, Vector &y1, Vector &y2,
                             const Vector &xMPI, const Vector &normal,
                             int e, int i, int j) const
{
   nbr = dofs.NbrDofs(i,j,e);

   for (int n = 0; n < hyp->NumEq; n++)
   {
      DofInd = n * ne * nd + e * nd + dofs.BdrDofs(j, i);

      if (nbr < 0)
      {
         uNbr = inflow(DofInd);
      }
      else
      {
         // nbr in different MPI task?
         uNbr = (nbr < xSizeMPI) ? x(n * ne * nd + nbr) : xMPI(int((nbr - xSizeMPI) / nd) * nd * hyp->NumEq + n * nd + (nbr - xSizeMPI) % nd);
      }

      y1(n) = x(DofInd);
      y2(n) = uNbr;
   }

   if (nbr < 0)
   {
      hyp->SetBdrCond(y1, y2, normal, nbr);
   }
}

void MCL_Evolution::LaxFriedrichs(const Vector &x1, const Vector &x2, const Vector &normal,
                                  Vector &y, int e, int j, int i) const
{
   double c_eij = 0.;
   C_eij = normal;
   for (int k = 0; k < nqf; k++)
   {
      c_eij += BdrInt(i,k,e) * ShapeEvalFace(i,j,k);
   }

   C_eij *= c_eij;

   double ws = max(hyp->GetWaveSpeed(x1, normal, e, 0, i),
                   hyp->GetWaveSpeed(x2, normal, e, 0, i));

   c_eij *= ws;

   subtract(c_eij, x2, x1, y);

   hyp->EvaluateFlux(x1, Flux, e, dofs.BdrDofs(j, i));
   hyp->EvaluateFlux(x2, FluxNbr, e, dofs.BdrDofs(j, i));
   Flux -= FluxNbr;
   Flux.AddMult(C_eij, y);

   y *= 0.5;
}

void MCL_Evolution::ComputeTimeDerivative(const Vector &x, Vector &y,
                                          const Vector &xMPI) const
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

      // Volume terms.
      for (int j = 0; j < nd; j++)
      {
         ElemEval(uElem, uEval, j);
         hyp->EvaluateFlux(uEval, Flux, e, j);
         MultABt(PrecGrad(j), ElemInt(e), CTilde(j));
         AddMultABt(CTilde(j), Flux, mat2);
      }

      z.AddElementVector(vdofs, mat2.GetData());

      // Artificial diffusion.
      for (int m = 0; m < dofs.numSubcells; m++)
      {
         for (int i = 0; i < dofs.numDofsSubcell; i++)
         {
            int I = dofs.Sub2Ind(m,i);
            ElemEval(uElem, uEval, I);

            for (int j = 0; j < nscd; j++)
            {
               int J = dofs.Sub2Ind(m, dofs.SubcellCross(i,j));
               ElemEval(uElem, uNbrEval, J);

               double CTildeNorm1 = 0.;
               double CTildeNorm2 = 0.;

               for (int l = 0; l < dim; l++)
               {
                  CTildeNorm1 += CTilde(I,l,J) * CTilde(I,l,J);
                  CTildeNorm2 += CTilde(J,l,I) * CTilde(J,l,I);
               }

               CTildeNorm1 = sqrt(CTildeNorm1);
               CTildeNorm2 = sqrt(CTildeNorm2);

               // double dij = max( 0., max( -CTilde(I,0,J), -CTilde(J,0,I) ) );

               CTilde(J).GetRow(I, normal);
               normal /= CTildeNorm1;

               double ws1 = max(hyp->GetWaveSpeed(uEval, normal, e, I),
                                hyp->GetWaveSpeed(uNbrEval, normal, e, J));

               CTilde(I).GetRow(J, normal);
               normal /= CTildeNorm2;

               double ws2 = max(hyp->GetWaveSpeed(uEval, normal, e, I),
                                hyp->GetWaveSpeed(uNbrEval, normal, e, J));
               double dij = max(CTildeNorm1 * ws1, CTildeNorm2 * ws2);

               for (int n = 0; n < hyp->NumEq; n++)
               {
                  z(vdofs[n * nd + I]) += dij * (x(vdofs[n * nd + J]) - x(vdofs[n * nd + I]));
               }
            }
         }
      }

      // DG flux terms and boundary conditions.
      for (int i = 0; i < dofs.NumBdrs; i++)
      {
         OuterUnitNormals(e).GetColumn(i, normal);

         for (int j = 0; j < dofs.NumFaceDofs; j++)
         {
            FaceEval(x, uEval, uNbrEval, xMPI, normal, e, i, j);
            LaxFriedrichs(uEval, uNbrEval, normal, NumFlux, e, j, i);

            for (int n = 0; n < hyp->NumEq; n++)
            {
               z(vdofs[n * nd + dofs.BdrDofs(j, i)]) += NumFlux(n);
            }
         }
      }

      for (int n = 0; n < hyp->NumEq; n++)
      {
         for (int j = 0; j < nd; j++)
         {
            DofInd = n*ne*nd + e*nd + j;
            y(DofInd) = z(DofInd) / LumpedMassMat(DofInd);
         }
      }
   }
}
