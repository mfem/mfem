#include "pfe_evol_tempplate.hpp"

PAR_TEMPLATE::PAR_TEMPLATE(ParFiniteElementSpace *pfes_,
                           HyperbolicSystem *hyp_,
                           DofInfo &dofs_)
   : TEMPLATE(fes_, hyp_, dofs_),
     SCHEME(fes_, hyp_, dofs_), x_gf_MPI(pfes_) { }

void PAR_TEMPLATE::Mult(const Vector &x, Vector &y) const
{
   x_gf_MPI = x;
   x_gf_MPI.ExchangeFaceNbrData();
   Vector &xMPI = x_gf_MPI.FaceNbrData();
   ComputeTimeDerivative(x, y, xMPI);
}

double PAR_TEMPLATE::ConvergenceCheck(double dt, double tol,
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
