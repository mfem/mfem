#include "pmonolithic_convex_limiting.hpp"

ParMCL_Evolution::ParMCL_Evolution(ParFiniteElementSpace *pfes_,
                                   HyperbolicSystem *hyp_,
                                   DofInfo &dofs_)
   : MCL_Evolution(pfes_, hyp_, dofs_), x_gf_MPI(pfes_)
{
   H1_FECollection fec(fes->GetFE(0)->GetOrder(), dim);
   pfesH1 = new ParFiniteElementSpace(pfes_->GetParMesh(), &fec);
   delete bounds; // Serial version is deleted.
   bounds = new ParTightBounds(pfes_, pfesH1);
}

void ParMCL_Evolution::Mult(const Vector &x, Vector &y) const
{
   x_gf_MPI = x;
   x_gf_MPI.ExchangeFaceNbrData();
   Vector &xMPI = x_gf_MPI.FaceNbrData();

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
   ComputeTimeDerivative(x, y, xMPI);
}

double ParMCL_Evolution::ConvergenceCheck(double dt, double tol,
                                          const Vector &u) const
{
   z = u;
   z -= uOld;

   double res, resMPI = 0.;
   if (!hyp->SteadyState) // Use consistent mass matrix.
   {
      MassMat->Mult(z, uOld);
      for (int i = 0; i < u.Size(); i++)
      {
         resMPI += uOld(i) * uOld(i);
      }
      MPI_Allreduce(&resMPI, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      res = sqrt(res) / dt;
   }
   else // Use lumped mass matrix.
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
