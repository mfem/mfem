#include "pfe_evol.hpp"

ParFE_Evolution::ParFE_Evolution(ParFiniteElementSpace *pfes_,
                                 HyperbolicSystem *hyp_,
                                 DofInfo &dofs_, EvolutionScheme scheme_)
   : TimeDependentOperator(pfes_->GetVSize()), pfes(pfes_), x_gf_MPI(pfes_),
      scheme(scheme_)

{
   // Compute the lumped mass matrix.
   ParBilinearForm ml(pfes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(LumpedMassMat);
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
