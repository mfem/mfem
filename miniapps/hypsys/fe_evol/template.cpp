#include "template.hpp"

TEMPLATE::TEMPLATE(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                   DofInfo &dofs_)
   : FE_Evolution(fes_, hyp_, dofs_)
{
   // TODO
}

void TEMPLATE::Mult(const Vector &x, Vector &y) const
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
   ComputeTimeDerivative(x, y);
}

void TEMPLATE::ComputeTimeDerivative(const Vector &x, Vector &y,
                                     const Vector &xMPI) const
{
   // TODO
}
