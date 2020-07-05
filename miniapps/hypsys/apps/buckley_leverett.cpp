#include "buckley_leverett.hpp"

Configuration ConfigBuckleyLeverett;

void AnalyticalSolutionBuckleyLeverett(const Vector &x, double t, Vector &u);
void InitialConditionBuckleyLeverett(const Vector &x, Vector &u);
void InflowFunctionBuckleyLeverett(const Vector &x, double t, Vector &u);

BuckleyLeverett::BuckleyLeverett(FiniteElementSpace *fes_, BlockVector &u_block,
                                 Configuration &config_)
   : HyperbolicSystem(fes_, u_block, 1, config_,
                      VectorFunctionCoefficient(1, InflowFunctionBuckleyLeverett))
{
   ConfigBuckleyLeverett = config_;

   VectorFunctionCoefficient ic(1, InitialConditionBuckleyLeverett);

   switch (ConfigBuckleyLeverett.ConfigNum)
   {
      case 0:
      {
         ProblemName = "Buckley-Leverett - Smooth solution";
         glvis_scale = "on";
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 0;
         L2_Projection(ic, u0);
         break;
      }
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
   const int dim = FluxEval.Width();

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
   const int dim = n.Size();

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

void BuckleyLeverett::SetBdrCond(const Vector &y1, Vector &y2, const Vector &normal,
                                 int attr) const
{
   return;
}

void BuckleyLeverett::ComputeErrors(Array<double> &errors, const GridFunction &u,
                             double DomainSize, double t) const
{
   errors.SetSize(3);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionBuckleyLeverett);
   uAnalytic.SetTime(t);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


void AnalyticalSolutionBuckleyLeverett(const Vector &x, double t, Vector &u)
{

}

void InitialConditionBuckleyLeverett(const Vector &x, Vector &u)
{
   const int dim = x.Size();

   // Map to the reference domain [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigBuckleyLeverett.bbMin(i) + ConfigBuckleyLeverett.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigBuckleyLeverett.bbMax(i) - ConfigBuckleyLeverett.bbMin(i));
   }

   switch (ConfigBuckleyLeverett.ConfigNum)
   {
      case 1:
      {
         u(0) =  X(0) < 0. ? -3. : 3.;
         break;
      }
      case 2:
      {
         X *= 1.5;
         u(0) = X.Norml2()*X.Norml2() < 0.5 ? 1. : 0.;
         break;
      }
   }
}

void InflowFunctionBuckleyLeverett(const Vector &x, double t, Vector &u)
{
   switch (ConfigBuckleyLeverett.ConfigNum)
   {
      case 1:
      {
         u(0) = x(0) > 0. ? 3. : -3;
         break;
      }
      case 2:
      {
         u(0) = 0.;
         break;
      }
   }
}
