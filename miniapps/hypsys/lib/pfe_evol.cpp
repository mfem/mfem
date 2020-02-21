#include "pfe_evol.hpp"

ParFE_Evolution::ParFE_Evolution(ParFiniteElementSpace *pfes_,
                                 HyperbolicSystem *hyp_,
                                 DofInfo &dofs_, EvolutionScheme scheme_,
                                 const Vector &LumpedMassMat_)
   : FE_Evolution(pfes_, hyp_, dofs_, scheme_,
                  LumpedMassMat_), pfes(pfes_), x_gf_MPI(pfes_),
     xSizeMPI(dofs.fes->GetTrueVSize()) { }

void ParFE_Evolution::FaceEval(const Vector &x, Vector &y1, Vector &y2,
                               Vector &xMPI, int e, int i, int k) const
{
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   y1 = y2 = 0.;
   for (int n = 0; n < hyp->NumEq; n++)
   {
      for (int j = 0; j < dofs.NumFaceDofs; j++)
      {
         nbr = dofs.NbrDofs(i, j, e);
         DofInd = n * ne * nd + e * nd + dofs.BdrDofs(j, i);
         if (nbr < 0)
         {
            // TODO more general boundary conditions, Riemann problem
            uNbr = x_gf_MPI(DofInd);
         }
         else
         {
            // nbr in different MPI task?
            uNbr = (nbr < xSizeMPI) ? x(n * ne * nd + nbr) : xMPI(int((
                                                                         nbr - xSizeMPI) / nd) * nd * hyp->NumEq + n * nd + (nbr - xSizeMPI) % nd);
         }

         y1(n) += x(DofInd) * ShapeEvalFace(i, j, k);
         y2(n) += uNbr * ShapeEvalFace(i, j, k);
      }
   }
}

double ParFE_Evolution::ConvergenceCheck(double dt, double tol,
                                         const Vector &u) const
{
   z = u;
   z -= uOld;

   double res, resMPI = 0.;
   if (scheme == 0) // Standard, i.e. use consistent mass matrix.
   {
      MassMat->Mult(z, uOld);
      for (int i = 0; i < u.Size(); i++)
      {
         resMPI += uOld(i) * uOld(i);
      }
      MPI_Allreduce(&resMPI, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      res = sqrt(res) / dt;
   }
   else // use lumped mass matrix.
   {
      for (int i = 0; i < u.Size(); i++)
      {
         resMPI += pow(LumpedMassMat(i) * z(i), 2.);
      }
      MPI_Allreduce(&resMPI, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      res = sqrt(res) / dt;
   }

   uOld = u;
   return res;
}

void ParFE_Evolution::EvolveStandard(const Vector &x, Vector &y) const
{
   z = 0.;
   x_gf_MPI = x;
   x_gf_MPI.ExchangeFaceNbrData();
   Vector &xMPI = x_gf_MPI.FaceNbrData();
   hyp->b.SetTime(t);
   x_gf_MPI.ProjectCoefficient(hyp->b); // TODO Fallunterscheidung: bc zeitabh.?

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
            FaceEval(x, uEval, uNbrEval, xMPI, e, i, k);

            LaxFriedrichs(uEval, uNbrEval, normal, NumFlux, e, k, i);
            NumFlux *= BdrInt(i, k, e);

            for (int n = 0; n < hyp->NumEq; n++)
            {
               for (int j = 0; j < dofs.NumFaceDofs; j++)
               {
                  z(vdofs[n * nd + dofs.BdrDofs(j, i)]) -= ShapeEvalFace(i, j, k) * NumFlux(n);
               }
            }
         }
      }
   }

   InvMassMat->Mult(z, y);
}

void ParFE_Evolution::EvolveMCL(const Vector &x, Vector &y) const
{
   MFEM_ABORT("TODO.");
}
