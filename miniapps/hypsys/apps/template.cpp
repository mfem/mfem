#include "template.hpp"

Configuration ConfigTEMPLATE;

void AnalyticalSolutionTEMPLATE(const Vector &x, double t, Vector &u);
void InitialConditionTEMPLATE(const Vector &x, Vector &u);
void InflowFunctionTEMPLATE(const Vector &x, double t, Vector &u);

TEMPLATE::TEMPLATE(FiniteElementSpace *fes_, BlockVector &u_block,
                   Configuration &config_)
   : HyperbolicSystem(fes_, u_block, NUMEQ, config_,
                      VectorFunctionCoefficient(NUMEQ, InflowFunctionTEMPLATE))
{
   ConfigTEMPLATE = config_;

   VectorFunctionCoefficient ic(NumEq, InitialConditionTEMPLATE);

   if (ConfigTEMPLATE.ConfigNum == 0)
   {
      SolutionKnown = ;
      SteadyState = ;
      TimeDepBC = ;
      ProjType = 0;
      L2_Projection(ic, u0);
      ProblemName = "TEMPLATE - ";
   }
   else
   {
      SolutionKnown = ;
      SteadyState = ;
      TimeDepBC = ;
      ProjType = 1;
      u0.ProjectCoefficient(ic);
      ProblemName = "TEMPLATE - ";
   }
}

void TEMPLATE::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                            int e, int k, int i) const
{
   // TODO
}

double TEMPLATE::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                              int i) const
{
   //TODO
}

void TEMPLATE::ComputeErrors(Array<double> &errors, const GridFunction &u,
                             double DomainSize, double t) const
{
   errors.SetSize(3);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionTEMPLATE);
   uAnalytic.SetTime(t);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


void AnalyticalSolutionTEMPLATE(const Vector &x, double t, Vector &u)
{
   // TODO
}

void InitialConditionTEMPLATE(const Vector &x, Vector &u)
{
   AnalyticalSolutionTEMPLATE(x, 0., u);
}

void InflowFunctionTEMPLATE(const Vector &x, double t, Vector &u)
{
   AnalyticalSolutionTEMPLATE(x, t, u);
}
