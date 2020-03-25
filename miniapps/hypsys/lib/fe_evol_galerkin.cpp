#include "fe_evol_galerkin.hpp"

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
