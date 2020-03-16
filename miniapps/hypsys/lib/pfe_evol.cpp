#include "pfe_evol.hpp"

ParFE_Evolution::ParFE_Evolution(ParFiniteElementSpace *pfes_,
                                 HyperbolicSystem *hyp_,
                                 DofInfo &dofs_, EvolutionScheme scheme_,
                                 const Vector &LumpedMassMat_)
   : FE_Evolution(pfes_, hyp_, dofs_, scheme_, LumpedMassMat_),
      pfes(pfes_), x_gf_MPI(pfes_) { }

void ParFE_Evolution::Mult(const Vector &x, Vector &y) const
{
   x_gf_MPI = x;
   x_gf_MPI.ExchangeFaceNbrData();
   Vector &xMPI = x_gf_MPI.FaceNbrData();

   switch (scheme)
   {
      case 0: // Standard Finite Element Approximation.
      {
         EvolveStandard(x, xMPI, y);
         break;
      }
      case 1: // Monolithic Convex Limiting.
      {
         EvolveMCL(x, xMPI, y);
         break;
      }
      default:
         MFEM_ABORT("Unknown Evolution Scheme.");
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
