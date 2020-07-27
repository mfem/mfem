#include "kpp.hpp"

Configuration ConfigKPP;

double InitialConditionKPP(const Vector &x);
void InflowFunctionKPP(const Vector &x, double t, Vector &u);

KPP::KPP(FiniteElementSpace *fes_, BlockVector &u_block,
         Configuration &config_)
   : HyperbolicSystem(fes_, u_block, 1, config_,
                      VectorFunctionCoefficient(1, InflowFunctionKPP))
{
   ConfigKPP = config_;

   FunctionCoefficient ic(InitialConditionKPP);

   switch (ConfigKPP.ConfigNum)
   {
      case 1:
      {
         ProblemName = "KPP Equation - Riemann Problem";
         glvis_scale = "on";
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}

void KPP::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                       int e, int k, int i) const
{
   FluxEval(0,0) = sin(u(0));
   FluxEval(0,1) = cos(u(0));
}

double KPP::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                         int i) const
{
   return 1.;
}

double InitialConditionKPP(const Vector &x)
{
   if (x.Size() != 2)
   {
      MFEM_ABORT("Test case only implemented in 2D.");
   }

   // Map to test case specific domain [-2,2] x [-2.5,1.5].
   Vector X(2);
   X(0) = ( 4.0 * x(0) - 2.0 * (ConfigKPP.bbMin(0) + ConfigKPP.bbMax(0)) ) / (ConfigKPP.bbMax(0) - ConfigKPP.bbMin(0));
   X(1) = ( 4.0 * x(1) - 1.5 * ConfigKPP.bbMin(1) - 2.5 * ConfigKPP.bbMax(1) ) / (ConfigKPP.bbMax(1) - ConfigKPP.bbMin(1));

   return X.Norml2() <= 1. ? 3.5 * M_PI : 0.25 * M_PI;
}

void InflowFunctionKPP(const Vector &x, double t, Vector &u)
{
   u(0) = 0.25 * M_PI;
}
