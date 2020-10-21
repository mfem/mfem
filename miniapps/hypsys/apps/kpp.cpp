#include "kpp.hpp"

Configuration ConfigKPP;

void InitialConditionKPP(const Vector &x, Vector &u);
void InflowFunctionKPP(const Vector &x, double t, Vector &u);

KPP::KPP(FiniteElementSpace *fes_, BlockVector &u_block,
         Configuration &config_)
   : HyperbolicSystem(fes_, u_block, 1, config_,
                      VectorFunctionCoefficient(1, InflowFunctionKPP))
{
   ConfigKPP = config_;
   VectorFunctionCoefficient ic(NumEq, InitialConditionKPP);

   switch (ConfigKPP.ConfigNum)
   {
      case 1:
      {
         ProblemName = "KPP Equation - 2D Spiral";
         glvis_scale = "on";
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 2:
      {
         ProblemName = "KPP Equation - 1D";
         glvis_scale = "on";
         SolutionKnown = false; // There is a solution, but I don't have an analytical expression.
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
   if (dim==1)
   {
      double coef = u(0)*(1.0-u(0));
      FluxEval(0,0) = u(0) < 0.5 ? (0.25*coef) : (0.1875 - 0.5*coef);
   }
   else if (dim==2)
   {
      FluxEval(0,0) = sin(u(0));
      FluxEval(0,1) = cos(u(0));
   }
   else { MFEM_ABORT("Not implemented."); }
}

double KPP::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                         int i) const
{
   return 1.0; // Tighter bound exists.
}

void InitialConditionKPP(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   Vector X(dim);

   for (int i = 0; i < dim; i++)
   {
      switch (ConfigKPP.ConfigNum)
      {
         case 1: // Map to the reference domain [-1,1]^d.
         {
            double center = 0.5 * (ConfigKPP.bbMin(i) + ConfigKPP.bbMax(i));
            X(i) = 2.0 * (x(i) - center) / (ConfigKPP.bbMax(i) - ConfigKPP.bbMin(i));
            break;
         }
         case 2: // Map to the reference domain [0,1]^d.
         {
            X(i) = (x(i) - ConfigKPP.bbMin(i)) / (ConfigKPP.bbMax(i) - ConfigKPP.bbMin(i));
            break;
         }
      }
   }

   switch (ConfigKPP.ConfigNum)
   {
      case 1:
      {
         // Map to test case specific domain [-2,2] x [-2.5,1.5].
         X *= 2.0;
         X(1) -= 0.5;

         u(0) = X.Norml2() <= 1. ? 3.5 * M_PI : 0.25 * M_PI;
         break;
      }
      case 2:
      {
         u(0) = X.Norml2() <= 0.25 ? 0.0 : 1.0; // According to the original KPP paper, not Ern and Guermond.
         break;
      }
   }
}

void InflowFunctionKPP(const Vector &x, double t, Vector &u)
{
   switch (ConfigKPP.ConfigNum)
   {
      case 1: { u(0) = 0.25 * M_PI;  break; }

      // This definition is consistent with the problem and assures correct
      // handling of inflow (left) and outflow (right) boundaries.
      case 2: { u(0) = x(0); break; }
   }
}
