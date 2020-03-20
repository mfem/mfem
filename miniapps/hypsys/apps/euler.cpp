#include "euler.hpp"

Configuration ConfigEuler;

const double SpHeatRatio = 1.4;

void AnalyticalSolutionEuler(const Vector &x, double t, Vector &u);
void InitialConditionEuler(const Vector &x, Vector &u);
void InflowFunctionEuler(const Vector &x, double t, Vector &u);

Euler::Euler(FiniteElementSpace *fes_, BlockVector &u_block,
             Configuration &config_)
   : HyperbolicSystem(fes_, u_block, fes_->GetMesh()->Dimension() + 2, config_,
                      VectorFunctionCoefficient(fes_->GetMesh()->Dimension() + 2,
                                                InflowFunctionEuler))
{
   ConfigEuler = config_;

   VectorFunctionCoefficient ic(NumEq, InitialConditionEuler);

   switch (ConfigEuler.ConfigNum)
   {
      case 0:
      {
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false; // Usage of periodic meshes is required.
         ProjType = 0;
         L2_Projection(ic, u0);
         valuerange = "0.493807323 1";
         ProblemName = "Euler Equations of Gas dynamics - Smooth Vortex";
         break;
      }
      case 1:
      {
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC =
            false; // TODO Choose a boundary condition for shock tube and adjust this.
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         valuerange = "0 1";
         ProblemName = "Euler Equations of Gas dynamics - SOD Shock Tube";
         break;
      }
      case 2:
      {
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         valuerange = "0 7";
         ProblemName = "Euler Equations of Gas dynamics - Woodward Colella";
         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}

double Euler::EvaluatePressure(const Vector &u) const
{
   const int dim = u.Size() - 2;
   double aux = 0.;
   for (int i = 0; i < dim; i++)
   {
      aux += u(1+i) * u(1+i);
   }
   double pressure = (SpHeatRatio - 1.) * (u(dim + 1) - 0.5 * aux / u(0));
   if (pressure < 0.)
   {
      MFEM_ABORT("Negative pressure.");
   }
   return pressure;
}

void Euler::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                         int e, int k, int i) const
{
   const int dim = u.Size() - 2;
   double RhoMin = 0.001;

   if (u.Size() != NumEq)
   {
      MFEM_ABORT("Invalid solution vector.");
   }
   if (u(0) < RhoMin)
   {
      MFEM_ABORT("Density too small.");
   }

   double pressure = EvaluatePressure(u);

   switch (dim)
   {
      case 1:
      {
         FluxEval(0,0) = u(1);
         FluxEval(1,0) = u(1) * u(1) / u(0) + pressure;
         FluxEval(2,0) = u(1) * u(2) / u(0) + pressure * u(1) / u(0);
         break;
      }
      case 2:
      {
         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);

         FluxEval(1,0) = u(1) * u(1) / u(0) + pressure;
         FluxEval(1,1) = u(1) * u(2) / u(0);

         FluxEval(2,0) = u(2) * u(1) / u(0);
         FluxEval(2,1) = u(2) * u(2) / u(0) + pressure;

         FluxEval(3,0) = (u(3) + pressure) * u(1) / u(0);
         FluxEval(3,1) = (u(3) + pressure) * u(2) / u(0);
         break;
      }
      case 3:
      {
         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);
         FluxEval(0,2) = u(3);

         FluxEval(1,0) = u(1) * u(1) / u(0) + pressure;
         FluxEval(1,1) = u(1) * u(2) / u(0);
         FluxEval(1,2) = u(1) * u(3) / u(0);

         FluxEval(2,0) = u(2) * u(1) / u(0);
         FluxEval(2,1) = u(2) * u(2) / u(0) + pressure;
         FluxEval(2,2) = u(2) * u(3) / u(0);

         FluxEval(3,0) = u(3) * u(1) / u(0);
         FluxEval(3,1) = u(3) * u(2) / u(0);
         FluxEval(3,2) = u(3) * u(3) / u(0) + pressure;

         FluxEval(4,0) = (u(4) + pressure) * u(1) / u(0);
         FluxEval(4,1) = (u(4) + pressure) * u(2) / u(0);
         FluxEval(4,2) = (u(4) + pressure) * u(3) / u(0);
         break;
      }
      default:
         MFEM_ABORT("Invalid space dimension.");
   }
}

double Euler::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                           int i) const
{
   switch (u.Size())
   {
      case 3:
         return abs(u(1)*n(0)) / u(0) + sqrt(SpHeatRatio * EvaluatePressure(u) / u(0));
      case 4:
         return abs(u(1)*n(0) + u(2)*n(1)) / u(0)
                + sqrt(SpHeatRatio * EvaluatePressure(u) / u(0));
      case 5:
         return abs(u(1)*n(0) + u(2)*n(1) + u(3)*n(2)) / u(0) + sqrt(
                   SpHeatRatio * EvaluatePressure(u) / u(0));
      default:
         MFEM_ABORT("Invalid solution vector.");
   }
}

void Euler::SetBdrCond(const Vector &y1, Vector &y2, const Vector &normal,
                       int attr) const
{
   switch (attr)
   {
      case -1: // wall boundary
      {
         int dim = normal.Size();
         if (dim == 1)
         {
            y2(0) = y1(0);
            y2(1) = -y1(1);
            y2(2) = y1(2);
         }
         else if (dim == 2)
         {
            double NormalComponent = y1(1) * normal(0) + y1(2) * normal(1);
            y2(0) = y1(0);
            y2(1) = y1(1) - 2. * NormalComponent * normal(0);
            y2(2) = y1(2) - 2. * NormalComponent * normal(1);
            y2(3) = y1(3);
         }
         else
         {
            double NormalComponent = y1(1) * normal(0) + y1(2) * normal(1) + y1(3) * normal(
                                        2);
            y2(0) = y1(0);
            y2(1) = y1(1) - 2. * NormalComponent * normal(0);
            y2(2) = y1(2) - 2. * NormalComponent * normal(1);
            y2(3) = y1(3) - 2. * NormalComponent * normal(2);
            y2(4) = y1(4);
         }
         return;
      }
      case -2: // supersonic outlet
      {
         y2 = y1;
         return;
      }
      case -3: // supersonic inlet
      {
         return;
      }
      // TODO subsonic in- and outlet
      default:
         MFEM_ABORT("Invalid boundary attribute.");
   }
}

void Euler::ComputeErrors(Array<double> & errors, const GridFunction &u,
                          double DomainSize, double t) const
{
   errors.SetSize(3);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionEuler);
   // Right now we use initial condition = solution due to periodic mesh.
   uAnalytic.SetTime(0);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}

// TODO use this everywhere.
void EvaluateEnergy(Vector &u, const double &pressure)
{
   const int dim = u.Size() - 2;
   double aux = 0.;
   for (int i = 0; i < dim; i++)
   {
      aux += u(1 + i) * u(1 + i);
   }
   u(dim + 1) = pressure / (SpHeatRatio - 1.) + 0.5 * aux / u(0);
}

void AnalyticalSolutionEuler(const Vector &x, double t, Vector &u)
{
   const int dim = x.Size();
   u.SetSize(dim + 2);

   // Map to the reference domain [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigEuler.bbMin(i) + ConfigEuler.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigEuler.bbMax(i) - ConfigEuler.bbMin(i));
   }

   switch (ConfigEuler.ConfigNum)
   {
      case 0:
      {
         if (dim != 2)
         {
            MFEM_ABORT("Test case only implemented in 2D.");
         }

         X *= 5.; // Map to test case specific domain [-5,5].

         // Parameters
         double beta = 5.;
         double r = X.Norml2();
         double T0 = 1. - (SpHeatRatio - 1.) * beta * beta / (8. * SpHeatRatio * M_PI *
                                                              M_PI) * exp(1 - r * r);

         u(0) = pow(T0, 1. / (SpHeatRatio - 1.));
         u(1) = (1. - beta / (2. * M_PI) * exp(0.5 * (1 - r * r)) * X(1)) * u(0);
         u(2) = (1. + beta / (2. * M_PI) * exp(0.5 * (1 - r * r)) * X(0)) * u(0);
         u(3) = u(0) * T0/(SpHeatRatio-1.) + 0.5 * (u(1) * u(1) + u(2) * u(2)) / u(0);
         break;
      }
      case 1:
      {
         u = 0.;
         u(0) = X.Norml2() < 0.5 ? 1. : .125;
         u(dim + 1) = X.Norml2() < 0.5 ? 1. : .1;
         break;
      }
      case 2:
      {
         if (dim != 1)
         {
            MFEM_ABORT("Test case only implemented in 1D.");
         }

         X(0) = 0.5 * (X(0) + 1.);

         u = 0.;
         u(0) = 1.;
         if (X(0) < 0.1)
         {
            EvaluateEnergy(u, 1000.);
         }
         else if (X(0) < 0.9)
         {
            EvaluateEnergy(u, 0.01);
         }
         else
         {
            EvaluateEnergy(u, 100.);
         }
         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}

void InitialConditionEuler(const Vector &x, Vector &u)
{
   AnalyticalSolutionEuler(x, 0., u);
}

void InflowFunctionEuler(const Vector &x, double t, Vector &u)
{
   AnalyticalSolutionEuler(x, t, u);
}
