#include "buckley_leverett.hpp"

Configuration ConfigBL;

void InitialConditionBuckleyLeverett(const Vector &x, Vector &u);
void InflowFunctionBuckleyLeverett(const Vector &x, double t, Vector &u);

BuckleyLeverett::BuckleyLeverett(FiniteElementSpace *fes_, BlockVector &u_block,
                                 Configuration &config_)
   : HyperbolicSystem(fes_, u_block, 1, config_,
                      VectorFunctionCoefficient(1, InflowFunctionBuckleyLeverett))
{
   ConfigBL = config_;
   VectorFunctionCoefficient ic(NumEq, InitialConditionBuckleyLeverett);

   switch (ConfigBL.ConfigNum)
   {
      case 1:
      {
         ProblemName = "Buckley-Leverett - 1D";
         glvis_scale = "on";
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 0;
         L2_Projection(ic, u0);
         break;
      }
      case 2:
      {
         ProblemName = "Buckley-Leverett - 2D";
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

void BuckleyLeverett::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                                   int e, int k, int i) const
{
   if (dim == 1)
   {
      FluxEval(0,0) = 4.*u(0)*u(0) / (4.*u(0)*u(0) + (1.-u(0))*(1.-u(0)));
   }
   else if (dim == 2)
   {
      double coef = u(0)*u(0) / (u(0)*u(0) + (1.-u(0))*(1.-u(0)));
      FluxEval(0,0) = coef;
      FluxEval(0,1) = coef * (1. -5.*(1.-u(0))*(1.-u(0)));
   }
   else { MFEM_ABORT("Not implemented."); }
}

double BuckleyLeverett::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                                     int i) const
{
   if (dim == 1)
   {
      return abs( 8.*u(0)*(1.-u(0)) / pow(4.*u(0)*u(0) + (1.-u(0))*(1.-u(0)), 2.) );
   }
   else if (dim == 2)
   {
      return 3.4;
   }
   else { MFEM_ABORT("Not implemented."); }
}

void InitialConditionBuckleyLeverett(const Vector &x, Vector &u)
{
   const int dim = x.Size();

   // Map to the reference domain [-1,1]^d.
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = 0.5 * (ConfigBL.bbMin(i) + ConfigBL.bbMax(i));
      X(i) = 2. * (x(i) - center) / (ConfigBL.bbMax(i) - ConfigBL.bbMin(i));
   }

   switch (ConfigBL.ConfigNum)
   {
      case 1:
      {
         u(0) =  X(0) < 0.0 ? -3.0 : 3.0;
         break;
      }
      case 2:
      {
         u(0) = X.Norml2()*X.Norml2() < 2.0 / 9.0 ? 1.0 : 0.0;
         break;
      }
   }
}

void InflowFunctionBuckleyLeverett(const Vector &x, double t, Vector &u)
{
   switch (ConfigBL.ConfigNum)
   {
      case 1:
      {
         u(0) = x(0) < 0.0 ? 3.0 : -3.0;
         break;
      }
      case 2:
      {
         u(0) = 0.0;
         break;
      }
   }
}
