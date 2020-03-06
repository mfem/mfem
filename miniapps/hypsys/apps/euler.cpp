#include "euler.hpp"

Configuration ConfigEuler;

const double SpHeatRatio = 1.4; // TODO

void AnalyticalSolutionEuler(const Vector &x, double t, Vector &u);
void InitialConditionEuler(const Vector &x, Vector &u);
void InflowFunctionEuler(const Vector &x, double t, Vector &u);

Euler::Euler(FiniteElementSpace *fes_, BlockVector &u_block,
             Configuration &config_)
    : HyperbolicSystem(fes_, u_block, fes_->GetMesh()->Dimension() + 2, config_,
                       VectorFunctionCoefficient(fes_->GetMesh()->Dimension() + 2, InflowFunctionEuler))
{
   ConfigEuler = config_;
   SteadyState = false;
   SolutionKnown = true;

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();

   VectorFunctionCoefficient ic(NumEq, InitialConditionEuler);

   if (ConfigEuler.ConfigNum == 0)
   {
      ProjType = 0;
      L2_Projection(ic, u0);
   }
   else
   {
      ProjType = 1;
      u0.ProjectCoefficient(ic);
   }
}

double Euler::EvaluatePressure(const Vector &u) const
{
   const int dim = u.Size() - 2;
   double aux = 0.;
   for (int i = 0; i < dim; i++)
   {
      aux += u(1 + i) * u(1 + i);
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
   double H0 = 0.001;

   if (u.Size() != NumEq)
   {
      MFEM_ABORT("Invalid solution vector.");
   }
   if (u(0) < H0)
   {
      MFEM_ABORT("Water height too small.");
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
         return abs(u(1)*n(0) + u(2)*n(1)) / u(0) + sqrt(SpHeatRatio * EvaluatePressure(u) / u(0));
      case 5:
         return abs(u(1)*n(0) + u(2)*n(1) + u(3)*n(2)) / u(0) + sqrt(SpHeatRatio * EvaluatePressure(u) / u(0));
      default:
         MFEM_ABORT("Invalid solution vector.");
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


void AnalyticalSolutionEuler(const Vector &x, double t, Vector &u)
{
   const int dim = x.Size();
   u.SetSize(dim + 2);

   // Map to the reference domain [-1,1] x [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigEuler.bbMin(i) + ConfigEuler.bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigEuler.bbMax(i) - ConfigEuler.bbMin(i));
   }

   switch (ConfigEuler.ConfigNum)
   {
      case 0: // Vorticity advection
      {
         X *= 5.; // Map to test case specific domain [-5,5] x [-5,5].

         if (dim != 2)
         {
            MFEM_ABORT("Test case only implemented in 2D.");
         }

         // Parameters
         double beta = 5.;
         double r = X.Norml2();
         double T0 = 1. - (SpHeatRatio - 1.) * beta * beta / (8. * SpHeatRatio * M_PI * M_PI) * exp(1 - r * r);

         u(0) = pow(T0, 1. / (SpHeatRatio - 1.));
         u(1) = (1. - beta / (2. * M_PI) * exp(0.5 * (1 - r * r)) * X(1)) * u(0);
         u(2) = (1. + beta / (2. * M_PI) * exp(0.5 * (1 - r * r)) * X(0)) * u(0);
         u(3) = u(0) * T0 / (SpHeatRatio - 1.) + 0.5 * (u(1) * u(1) + u(2) * u(2)) / u(0);
         break;
      }
      case 1:
      {
         u = 0.;
         u(0) = X.Norml2() < 0.5 ? 1. : .125;
         u(dim + 1) = X.Norml2() < 0.5 ? 1. : .1;
         break;
      }
      default:
      {
         MFEM_ABORT("No such test case implemented.");
      }
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
