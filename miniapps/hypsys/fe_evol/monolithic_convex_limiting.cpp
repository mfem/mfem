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
   DenseMatrix dshape(nd,dim), GradAux(nd,nd), RefMass(nd,nd), InvRefMass(nd,nd);
   DenseTensor GradOp(nd,nd,dim);

   // ShapeNodes.SetSize(nd,nd); // TODO unnecessary?
   Adjugates.SetSize(dim,dim,ne);
   PrecGradOp.SetSize(nd,dim,nd);
   GradProd.SetSize(nd,dim,nd);
   CTilde.SetSize(nd,dim,nd);
   CFull.SetSize(nd,dim,nd);
   OuterUnitNormals.SetSize(dim, dofs.NumBdrs, ne);
   C_eij.SetSize(dim); // TODO rename
   eldofs.SetSize(hyp->NumEq*nd);
   DistributionMatrix.SetSize(nd,nd);

   nscd = nscd = dofs.SubcellCross.Width();

   uFace.SetSize(hyp->NumEq, dofs.NumFaceDofs);
   uNbrFace.SetSize(hyp->NumEq, dofs.NumFaceDofs);

   RefMassLumped =  0.;
   RefMass = 0.;
   PrecGradOp = 0.;
   GradOp = 0.;

   Array <int> bdrs, orientation;

   // for (int j = 0; j < nd; j++)
   // {
   //    const IntegrationPoint &ip = nodes.IntPoint(j);
   //    el->CalcShape(ip, shape);
   //    ShapeNodes.SetCol(j, shape);
   // }

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
         AddMult_a_VWt(ip.weight, shape, dx_shape, GradOp(l));
      }
   }

   DenseMatrixInverse inv(&RefMass);
   inv.Factor();
   inv.GetInverseMatrix(InvRefMass);

   for (int l = 0; l < dim; l++)
   {
      GradAux = 0.;
      AddMult(InvRefMass, GradOp(l), GradAux);
      GradAux.LeftScaling(RefMassLumped);

      for (int i = 0; i < nd; i++)
      {
         for (int j = 0; j < nd; j++)
         {
            PrecGradOp(i,l,j) += GradAux(i,j);
            GradProd(i,l,j) = GradOp(i,j,l) + GradOp(j,i,l);
         }
      }
   }

   FaceMat.SetSize(dofs.NumFaceDofs, dofs.NumFaceDofs);
   FaceMat = 0.;

   for (int k = 0; k < nqf; k++)
   {
      for (int i = 0; i < dofs.NumFaceDofs; i++)
      {
         for (int j = 0; j < dofs.NumFaceDofs; j++)
         {
            FaceMat(i,j) += IntRuleFaceWeights(k)
                            * ShapeEvalFace(0,i,k) * ShapeEvalFace(0,j,k);
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
      Adjugates(e) = mat_aux;

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

   ComputeLORMassMatrix(RefMat, gtype, false);

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

   Vector MassMatLORLumped(nd);
   DenseMatrix LowOrderRefinedMat(MassMatLOR);
   LowOrderRefinedMat *= -1;
   LowOrderRefinedMat.GetRowSums(MassMatLORLumped);

   for (int i = 0; i < nd; i++)
   {
      LowOrderRefinedMat(i,i) -= MassMatLORLumped(i);
      LowOrderRefinedMat(0,i) = 1.;
   }

   DenseMatrixInverse InvLOR(&LowOrderRefinedMat);
   InvLOR.Factor();
   InvLOR.GetInverseMatrix(DistributionMatrix);
}

void MCL_Evolution::Mult(const Vector &x, Vector &y) const
{
   ComputeTimeDerivative(x, y);
}

void MCL_Evolution::GetNodeVal(const Vector &uElem, Vector &uEval, int j) const
{
   for (int n = 0; n < hyp->NumEq; n++)
   {
      uEval(n) = uElem(n*nd+j);
   }
}

void MCL_Evolution::FaceTerm(const Vector &x, Vector &uEval, Vector &uNbrEval,
                             const Vector &xMPI, const Vector &normal,
                             int e, int i, int j) const
{
   // nbr = dofs.NbrDofs(i,j,e);

   // for (int n = 0; n < hyp->NumEq; n++)
   // {
   //    DofInd = n * ne * nd + e * nd + dofs.BdrDofs(j, i);

   //    if (nbr < 0)
   //    {
   //       uNbr = inflow(DofInd);
   //    }
   //    else
   //    {
   //       // nbr in different MPI task?
   //       uNbr = (nbr < xSizeMPI) ? x(n * ne * nd + nbr) : xMPI(int((nbr - xSizeMPI) / nd) * nd * hyp->NumEq + n * nd + (nbr - xSizeMPI) % nd);
   //    }

   //    uEval(n) = x(DofInd);
   //    uNbrEval(n) = uNbr;
   // }

   // if (nbr < 0)
   // {
   //    hyp->SetBdrCond(uEval, uNbrEval, normal, nbr);
   // }
}

void MCL_Evolution::LinearFluxLumping(const Vector &x1, const Vector &x2, const Vector &normal,
                                      Vector &y, int e, int j, int i) const
{
   // double c_eij = 0.; // TODO use FaceMat
   // C_eij = normal;
   // for (int k = 0; k < nqf; k++)
   // {
   //    c_eij += BdrInt(i,k,e)  * IntRuleFaceWeights(k) * ShapeEvalFace(i,j,k);
   // }

   // C_eij *= c_eij;

   // double ws = max(hyp->GetWaveSpeed(uEval, normal, e, 0, i),
   //                 hyp->GetWaveSpeed(uNbrEval, normal, e, 0, i));

   // c_eij *= ws;

   // subtract(c_eij, uNbrEval, uEval, NumFlux);

   // hyp->EvaluateFlux(uEval, Flux, e, dofs.BdrDofs(j, i));
   // hyp->EvaluateFlux(uNbrEval, FluxNbr, e, dofs.BdrDofs(j, i));
   // Flux -= FluxNbr;
   // Flux.AddMult(C_eij, NumFlux);

   // NumFlux *= 0.5;
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

   dofs.ComputeBounds(x);

   DenseMatrix mat3(nd, hyp->NumEq);
   DenseMatrix galerkin(nd, hyp->NumEq), ElFlux(nd, hyp->NumEq), BdrFlux(hyp->NumEq*nd, dofs.NumBdrs),
               uDot(nd, hyp->NumEq);
   DenseTensor AntiDiff(nd, nd, hyp->NumEq), wij(nd, nd, hyp->NumEq);

   z = 0.;
   for (int e = 0; e < ne; e++)
   {
      fes->GetElementVDofs(e, vdofs);
      x.GetSubVector(vdofs, uElem);
      mat2 = uDot = 0.;
      AntiDiff = 0.;
      BdrFlux = 0.;

      for (int k = 0; k < nqe; k++)
      {
         ElemEval(uElem, uEval, k);
         hyp->EvaluateFlux(uEval, Flux, e, k);
         MultABt(ElemInt(e * nqe + k), Flux, mat1);
         AddMult(DShapeEval(k), mat1, mat2);
      }

      galerkin = mat2;

      for (int i = 0; i < dofs.NumBdrs; i++)
      {
         OuterUnitNormals(e).GetColumn(i, normal);

         for (int k = 0; k < nqf; k++)
         {
            FaceEval(x, uEval, uNbrEval, xMPI, normal, e, i, k);

            LaxFriedrichs(uEval, uNbrEval, normal, NumFlux, e, k, i);
            NumFlux *= BdrInt(i,k,e) * IntRuleFaceWeights(k);

            for (int n = 0; n < hyp->NumEq; n++)
            {
               for (int j = 0; j < dofs.NumFaceDofs; j++)
               {
                  double tmp = ShapeEvalFace(i,j,k) * NumFlux(n);
                  galerkin(dofs.BdrDofs(j,i), n) -= tmp;
                  BdrFlux(n*nd + dofs.BdrDofs(j,i), i) -= tmp;
               }
            }
         }
      }

      // Using the fact that mass matrices are the same for all components.
      AddMult(InvMassMat->Minv(e), galerkin, uDot);

      // TODO: Optimization possible by just using flux terms
      Add(mat2, galerkin, -1., ElFlux); // ElFlux = int f(u_h) \nabla \phi - M_C uDot

      mat2 = mat3 = 0.;

      // Volume terms.
      for (int j = 0; j < nd; j++)
      {
         GetNodeVal(uElem, uEval, j);
         hyp->EvaluateFlux(uEval, Flux, e, j);
         MultABt(PrecGradOp(j), Adjugates(e), CTilde(j));
         AddMultABt(CTilde(j), Flux, mat2);

         MultABt(GradProd(j), Adjugates(e), CFull(j));
         AddMultABt(CFull(j), Flux, mat3);

         // ElFlux += M_L uDot
         for (int n = 0; n < hyp->NumEq; n++)
         {
            ElFlux(j,n) += LumpedMassMat(n*ne*nd + e*nd + j) * uDot(j,n);
         }
      }

      ElFlux += mat2; // ElFlux_i += sum_j f_j cTilde_ij
      ElFlux -= mat3; // ElFlux_i -= sum_j f_j (c_ij + c_ji) )
      Vector ElTerms(mat2.GetData(), nd*hyp->NumEq);
      z.AddElementVector(vdofs, -1., ElTerms);

      // Artificial diffusion.
      for (int m = 0; m < dofs.numSubcells; m++)
      {
         for (int i = 0; i < dofs.numDofsSubcell; i++)
         {
            int I = dofs.Sub2Ind(m,i);
            GetNodeVal(uElem, uEval, I);

            for (int j = 0; j < nscd; j++)
            {
               int J = dofs.Sub2Ind(m, dofs.SubcellCross(i,j));
               GetNodeVal(uElem, uNbrEval, J);

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
                  double rus = dij * (x(vdofs[n * nd + J]) - x(vdofs[n * nd + I]));
                  z(vdofs[n * nd + I]) += rus;
                  AntiDiff(I,J,n) -= rus;
                  // wij(I,J,n) = dij * (x(vdofs[n * nd + J]) + x(vdofs[n * nd + I]))
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

               uEval(n) = x(DofInd);
               uNbrEval(n) = uNbr;
            }

            if (nbr < 0)
            {
               hyp->SetBdrCond(uEval, uNbrEval, normal, nbr);
            }

            uFace.SetCol(j, uEval);
            uNbrFace.SetCol(j, uNbrEval);
         }

         for (int j = 0; j < dofs.NumFaceDofs; j++)
         {
            double FaceMatLumped = 0.; // TODO optimize?

            uFace.GetColumn(j, uEval);
            hyp->EvaluateFlux(uEval, Flux, e, dofs.BdrDofs(j, i)); // TODO default argument k wrong

            for (int l = 0; l < dofs.NumFaceDofs; l++)
            {
               FaceMatLumped += FaceMat(j,l);

               if (j == l) { continue; }

               uFace.GetColumn(l, uNbrEval);
               hyp->EvaluateFlux(uNbrEval, FluxNbr, e, dofs.BdrDofs(j, i)); // TODO default argument k wrong
               FluxNbr -= Flux;
               Vector contribution(hyp->NumEq);
               FluxNbr.Mult(normal, contribution);
               contribution *= BdrInt(i,0,e) * FaceMat(j,l); // TODO BdrInt

               for (int n = 0; n < hyp->NumEq; n++)
               {
                  ElFlux(dofs.BdrDofs(j,i), n) += contribution(n);
               }
            }

            uNbrFace.GetColumn(j, uNbrEval);

            // LinearFluxLumping(uEval, uNbrEval, normal, NumFlux, e, j, i);

            double ws = max( hyp->GetWaveSpeed(uEval, normal, e, 0, i),
                             hyp->GetWaveSpeed(uNbrEval, normal, e, 0, i) );

            FaceMatLumped *= BdrInt(i,0,e);

            subtract(ws * FaceMatLumped, uNbrEval, uEval, NumFlux);

            hyp->EvaluateFlux(uEval, Flux, e, dofs.BdrDofs(j, i));
            hyp->EvaluateFlux(uNbrEval, FluxNbr, e, dofs.BdrDofs(j, i));
            Vector helper(hyp->NumEq);
            Flux.Mult(normal, helper);
            helper *= FaceMatLumped;

            Flux -= FluxNbr;
            Flux.AddMult_a(FaceMatLumped, normal, NumFlux);

            NumFlux *= 0.5;

            for (int n = 0; n < hyp->NumEq; n++)
            {
               z(vdofs[n*nd + dofs.BdrDofs(j,i)]) += NumFlux(n);
               BdrFlux(n*nd + dofs.BdrDofs(j,i), i) += helper(n) - NumFlux(n);
            }
         }
      }

      for (int n = 0; n < hyp->NumEq; n++)
      {
         double test = 0.;
         for (int j = 0; j < nd; j++)
         {
            test += ElFlux(j,n);
         }
         if (abs(test) > 1.E-12)
         {
            cout << abs(test) << endl;
            MFEM_ABORT("non-zero sum.");
         }
      }

      DenseMatrix AuxiliaryVectors(nd, hyp->NumEq);
      mfem::Mult(DistributionMatrix, ElFlux, AuxiliaryVectors);
      DenseTensor UnlimitedFluxes(nd, nd, hyp->NumEq); // TODO: Try to use nscd

      ElFlux = 0.;
      for (int i = 0; i < nd; i++)
      {
         for (int j = 0; j < nd; j++)
         {
            if (i==j) { continue; }

            for (int n = 0; n < hyp->NumEq; n++)
            {
               UnlimitedFluxes(i,j,n) = AntiDiff(i,j,n) + MassMatLOR(i,j)
                                        * (AuxiliaryVectors(i,n) - AuxiliaryVectors(j,n));

               ElFlux(i,n) += UnlimitedFluxes(i,j,n);
            }
         }
      }

      z.AddElementVector(vdofs, ElFlux.GetData());

      Vector sums(hyp->NumEq*nd);
      BdrFlux.GetRowSums(sums);
      z.AddElementVector(vdofs, sums);

      for (int n = 0; n < hyp->NumEq; n++)
      {
         for (int j = 0; j < nd; j++)
         {
            DofInd = n*ne*nd + e*nd + j;
            y(DofInd) = z(DofInd) / LumpedMassMat(DofInd);
         }
      }

      // z.SetSubVector(vdofs, galerkin.GetData());
   }
   // InvMassMat->Mult(z, y);
}

void MCL_Evolution::ComputeLORMassMatrix(DenseMatrix &RefMat, Geometry::Type gtype, bool UseDiagonalNbrs)
{
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
         if (UseDiagonalNbrs)
         {
            RefMat(0,0) = RefMat(1,1) = RefMat(2,2) = RefMat(3,3) = 4.;
            RefMat(0,3) = RefMat(1,2) = RefMat(2,1) = RefMat(3,0) = 1.;
         }
         else
         {
            RefMat(0,0) = RefMat(1,1) = RefMat(2,2) = RefMat(3,3) = 5.;
            RefMat(0,3) = RefMat(1,2) = RefMat(2,1) = RefMat(3,0) = 0.;
         }
         RefMat *= 1. / 36.;
         break;
      }
      case Geometry::CUBE:
      {
         if (UseDiagonalNbrs)
         {
            RefMat = 2.;
            RefMat(0,0) = RefMat(1,1) = RefMat(2,2) = RefMat(3,3) = 8.;
            RefMat(4,4) = RefMat(5,5) = RefMat(6,6) = RefMat(7,7) = 8.;
            RefMat(0,7) = RefMat(1,6) = RefMat(2,5) = RefMat(3,4) = 1.;
            RefMat(4,3) = RefMat(5,2) = RefMat(6,1) = RefMat(7,0) = 1.;
         }
         else
         {
            RefMat = 0.;
            RefMat(0,0) = RefMat(1,1) = RefMat(2,2) = RefMat(3,3) = 15.;
            RefMat(4,4) = RefMat(5,5) = RefMat(6,6) = RefMat(7,7) = 15.;
         }

         RefMat(0,1) = RefMat(0,2) = RefMat(1,0) = RefMat(1,3) = 4.;
         RefMat(2,0) = RefMat(2,3) = RefMat(3,1) = RefMat(3,2) = 4.;
         RefMat(4,5) = RefMat(4,6) = RefMat(5,4) = RefMat(5,7) = 4.;
         RefMat(6,4) = RefMat(6,7) = RefMat(7,5) = RefMat(7,6) = 4.;
         RefMat(0,4) = RefMat(1,5) = RefMat(2,6) = RefMat(3,7) = 4.;
         RefMat(4,0) = RefMat(5,1) = RefMat(6,2) = RefMat(7,3) = 4.;

         RefMat *= 1. / 216.;
         break;
      }
      default:
         MFEM_ABORT("Unsupported Geometry.");
   }
}