#include "pfe_evol.hpp"

ParFE_Evolution::ParFE_Evolution(ParFiniteElementSpace *pfes_,
                                 HyperbolicSystem *hyp_,
                                 DofInfo &dofs_, EvolutionScheme scheme_,
                                 const Vector &LumpedMassMat_)
   : FE_Evolution(pfes_, hyp_, dofs_,scheme_,
                  LumpedMassMat_), pfes(pfes_),x_gf_MPI(pfes_),
     xSizeMPI(pfes_->GetTrueVSize()) { }

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
         resMPI += uOld(i)*uOld(i);
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

            uEval = uNbrEval = 0.;

            for (int j = 0; j < dofs.NumFaceDofs; j++)
            {
               nbr = dofs.NbrDofs(i,j,e);
               if (nbr < 0)
               {
                  DofInd = e*nd+dofs.BdrDofs(j,i);
                  uNbr(0) = hyp->inflow(DofInd);
               }
               else
               {
                  // nbr in different MPI task?
                  uNbr(0) = (nbr < xSizeMPI) ? x(nbr) : xMPI(nbr-xSizeMPI);
               }

               uEval(0) += uElem(dofs.BdrDofs(j,i)) * ShapeEvalFace(i,j,k);
               uNbrEval(0) += uNbr(0) * ShapeEvalFace(i,j,k);
            }

            // Lax-Friedrichs flux (equals full upwinding for Advection).
            tmp = 0.5 * ( tmp * (uEval(0) + uNbrEval(0)) + abs(tmp) * (uEval(0) - uNbrEval(
                                                                          0)) ) * QuadWeightFace(k);

            for (int j = 0; j < dofs.NumFaceDofs; j++)
            {
               z(vdofs[dofs.BdrDofs(j,i)]) -= ShapeEvalFace(i,j,k) * tmp;
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
