#include "galerkin.hpp"

GalerkinEvolution::GalerkinEvolution(FiniteElementSpace *fes_,
                                     HyperbolicSystem *hyp_, DofInfo &dofs_)
   : FE_Evolution(fes_, hyp_, dofs_)
{
   // TODO
}

void GalerkinEvolution::Mult(const Vector &x, Vector &y) const
{
   ComputeTimeDerivative(x, y);
}

void GalerkinEvolution::ElemEval(const Vector &uElem, Vector &uEval, int k) const
{
   uEval = 0.;
   for (int n = 0; n < hyp->NumEq; n++)
   {
      for (int j = 0; j < nd; j++)
      {
         uEval(n) += uElem(n * nd + j) * ShapeEval(j, k);
      }
   }
}

void GalerkinEvolution::FaceEval(const Vector &x, Vector &y1, Vector &y2,
                                 const Vector &xMPI, const Vector &normal,
                                 int e, int i, int k) const
{
   y1 = y2 = 0.;
   for (int n = 0; n < hyp->NumEq; n++)
   {
      for (int j = 0; j < dofs.NumFaceDofs; j++)
      {
         nbr = dofs.NbrDofs(i, j, e);
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

         y1(n) += x(DofInd) * ShapeEvalFace(i, j, k);
         y2(n) += uNbr * ShapeEvalFace(i, j, k);
      }
   }

   if (nbr < 0) // TODO better distinction
   {
      hyp->SetBdrCond(y1, y2, normal, nbr);
   }
}

void GalerkinEvolution::LaxFriedrichs(const Vector &x1, const Vector &x2,
                                      const Vector &normal, Vector &y,
                                      int e, int k, int i) const
{
   hyp->EvaluateFlux(x1, Flux, e, k, i);
   hyp->EvaluateFlux(x2, FluxNbr, e, k, i);
   Flux += FluxNbr;
   double ws = max(hyp->GetWaveSpeed(x1, normal, e, k, i),
                   hyp->GetWaveSpeed(x2, normal, e, k, i));

   subtract(ws, x1, x2, y);
   Flux.AddMult(normal, y);
   y *= 0.5;
}

void GalerkinEvolution::ComputeTimeDerivative(const Vector &x, Vector &y,
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

      for (int k = 0; k < nqe; k++)
      {
         ElemEval(uElem, uEval, k);
         hyp->EvaluateFlux(uEval, Flux, e, k);
         MultABt(ElemInt(e * nqe + k), Flux, mat1);
         AddMult(DShapeEval(k), mat1, mat2);
      }

      z.AddElementVector(vdofs, mat2.GetData());

      // Here, the use of nodal basis functions is essential, i.e. shape
      // functions must vanish on faces that their node is not associated with.
      for (int i = 0; i < dofs.NumBdrs; i++)
      {
         for (int k = 0; k < nqf; k++)
         {
            OuterUnitNormals(e * dofs.NumBdrs + i).GetColumn(k, normal);
            FaceEval(x, uEval, uNbrEval, xMPI, normal, e, i, k);

            LaxFriedrichs(uEval, uNbrEval, normal, NumFlux, e, k, i);
            NumFlux *= BdrInt(i, k, e);

            for (int n = 0; n < hyp->NumEq; n++)
            {
               for (int j = 0; j < dofs.NumFaceDofs; j++)
               {
                  z(vdofs[n * nd + dofs.BdrDofs(j,i)]) -= ShapeEvalFace(i,j,k)
                                                          * NumFlux(n);
               }
            }
         }
      }
   }

   InvMassMat->Mult(z, y);
}
