#include "euler.hpp"

Configuration ConfigEuler;

double SpHeatRatio;

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
         ProblemName = "Euler Equations of Gas dynamics - Smooth Vortex";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false; // Usage of periodic meshes is required.
         ProjType = 0;
         L2_Projection(ic, u0);
         break;
      }
      case 1:
      {
         ProblemName = "Euler Equations of Gas dynamics - SOD Shock Tube";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         L2_Projection(ic, u0);
         break;
      }
      case 2:
      {
         ProblemName = "Euler Equations of Gas dynamics - Woodward Colella";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 3:
      {
         ProblemName = "Euler Equations of Gas dynamics - Double Mach Reflection";
         glvis_scale = "off valuerange 1.4 22";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = true;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 4:
      {
         ProblemName = "Euler Equations of Gas dynamics - Sedov Blast";
         glvis_scale = "off valuerange 0 2.5"; // TODO
         SpHeatRatio = 5.0 / 3.0;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = true;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 5:
      {
         ProblemName = "Euler Equations of Gas dynamics - Noh Problem";
         glvis_scale = "off valuerange 1 16";
         SpHeatRatio = 5.0 / 3.0;
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = true;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 6:
      {
         ProblemName = "Euler Equations of Gas dynamics - MoST Gimmick";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;

         Mesh *mesh = fes->GetMesh();
         const int nd = fes->GetFE(0)->GetDof();
         const int ne = fes->GetNE();
         if (mesh->Dimension() != 2) { MFEM_ABORT("Test case is 2D."); }
         u0 = 0.;

         for (int e = 0; e < ne; e++)
         {
            int id = mesh->GetElement(e)->GetAttribute();
            for (int j = 0; j < nd; j++)
            {
               switch (id)
               {
                  case 1:
                  {
                     u0(e*nd+j) = 1.;
                     u0(3*ne*nd + e*nd+j) = 1. / SpHeatRatio;
                     break;
                  }
                  case 2:
                  case 3:
                  case 4:
                  {
                     u0(e*nd+j) = 0.125;
                     u0(3*ne*nd + e*nd+j) = 0.1 / SpHeatRatio;
                     break;
                  }
                  default:
                     MFEM_ABORT("Too many element IDs.");
               }
            }
         }

         break;
      }
      case 7:
      {
         ProblemName = "Euler Equations of Gas dynamics - Constricted Channel";
         glvis_scale = "on";
         SpHeatRatio = 1.4;
         SolutionKnown = false;
         SteadyState = true;
         TimeDepBC = false;
         ProjType = 0;
         u0.ProjectCoefficient(ic);
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
      ostringstream press_str;
      press_str << pressure;
      string err_msg = "Negative pressure p = ";
      MFEM_ABORT(err_msg << press_str.str());
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
      ostringstream rho_str;
      rho_str << u(0);
      string err_msg = "Density too small rho = ";
      MFEM_ABORT(err_msg << rho_str.str());
   }

   double pressure = EvaluatePressure(u);

   switch (dim)
   {
      case 1:
      {
         double vx = u(1) / u(0);
         FluxEval(0,0) = u(1);
         FluxEval(1,0) = u(1) * vx + pressure;
         FluxEval(2,0) = (u(2) + pressure) * vx;
         break;
      }
      case 2:
      {
         double vx = u(1) / u(0);
         double vy = u(2) / u(0);
         double energy = u(3) + pressure;

         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);

         FluxEval(1,0) = u(1) * vx + pressure;
         FluxEval(1,1) = u(1) * vy;

         FluxEval(2,0) = u(2) * vx;
         FluxEval(2,1) = u(2) * vy + pressure;

         FluxEval(3,0) = energy * vx;
         FluxEval(3,1) = energy * vy;
         break;
      }
      case 3:
      {
         double vx = u(1) / u(0);
         double vy = u(2) / u(0);
         double vz = u(3) / u(0);
         double energy = u(4) + pressure;

         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);
         FluxEval(0,2) = u(3);

         FluxEval(1,0) = u(1) * vx + pressure;
         FluxEval(1,1) = u(1) * vy;
         FluxEval(1,2) = u(1) * vz;

         FluxEval(2,0) = u(2) * vx;
         FluxEval(2,1) = u(2) * vy + pressure;
         FluxEval(2,2) = u(2) * vz;

         FluxEval(3,0) = u(3) * vx;
         FluxEval(3,1) = u(3) * vy;
         FluxEval(3,2) = u(3) * vz + pressure;

         FluxEval(4,0) = energy * vx;
         FluxEval(4,1) = energy * vy;
         FluxEval(4,2) = energy * vz;
         break;
      }
      default:
         MFEM_ABORT("Invalid space dimension.");
   }
}

double Euler::GetGMS(const Vector &uL, const Vector &uR, const Vector &normal) const
{
   double pL = EvaluatePressure(uL);
   double pR = EvaluatePressure(uR);
   double aL = sqrt(SpHeatRatio * pL / uL(0));
   double aR = sqrt(SpHeatRatio * pR / uR(0));
   double vL = uL(1)/uL(0) * normal(0);
   double vR = uR(1)/uR(0) * normal(0);

   double p = pow( (aL+aR-0.5*(SpHeatRatio-1.)*(vR-vL)) / (aL*pow(pL, (1.-SpHeatRatio)/(2.*SpHeatRatio)) + aR*pow(pR, (1.-SpHeatRatio)/(2.*SpHeatRatio)) ) , 2.*SpHeatRatio/(SpHeatRatio-1.) );

   double lambda1 = vL - aL * sqrt( 1. + (SpHeatRatio+1.)/(2.*SpHeatRatio) * max(0., (p-pL)/pL) );
   double lambda3 = vR + aR * sqrt( 1. + (SpHeatRatio+1.)/(2.*SpHeatRatio) * max(0., (p-pR)/pR) );
   return max(abs(lambda1), abs(lambda3));
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
         return abs(u(1)*n(0) + u(2)*n(1) + u(3)*n(2)) / u(0)
                + sqrt(SpHeatRatio * EvaluatePressure(u) / u(0));
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
            double Mom_x_Norm = y1(1) * normal(0) + y1(2) * normal(1);
            y2(0) = y1(0);
            y2(1) = y1(1) - 2. * Mom_x_Norm * normal(0);
            y2(2) = y1(2) - 2. * Mom_x_Norm * normal(1);
            y2(3) = y1(3);
         }
         else
         {
            double Mom_x_Norm = y1(1) * normal(0) + y1(2) * normal(1) + y1(3) * normal(2);
            y2(0) = y1(0);
            y2(1) = y1(1) - 2. * Mom_x_Norm * normal(0);
            y2(2) = y1(2) - 2. * Mom_x_Norm * normal(1);
            y2(3) = y1(3) - 2. * Mom_x_Norm * normal(2);
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

void EvaluateEnergy(Vector &u, const double &pressure)
{
   const int dim = u.Size() - 2;
   double aux = 0.;
   for (int i = 0; i < dim; i++)
   {
      aux += u(1+i) * u(1+i);
   }
   u(dim+1) = pressure / (SpHeatRatio - 1.) + 0.5 * aux / u(0);
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
         double T0 = 1. - (SpHeatRatio - 1.) * beta * beta
                     / (8. * SpHeatRatio * M_PI * M_PI) * exp(1. - r * r);

         u(0) = pow(T0, 1. / (SpHeatRatio - 1.));
         u(1) = (1. - beta / (2. * M_PI) * exp(0.5 * (1 - r * r)) * X(1)) * u(0);
         u(2) = (1. + beta / (2. * M_PI) * exp(0.5 * (1 - r * r)) * X(0)) * u(0);
         EvaluateEnergy(u, u(0) * T0);
         break;
      }
      case 3:
      {
         if (dim != 2)
         {
            MFEM_ABORT("Test case only implemented in 2D.");
         }

         X(0) = 2. * (X(0) + 1.);
         X(1) = 0.5 * (X(1) + 1.);

         bool left = X(0) < 1. / 6. + (X(1) + 20.*t) / sqrt(3.);

         if (left)
         {
            u(0) = 8.;
            u(1) = 66. * cos(M_PI / 6.);
            u(2) = -66. * sin(M_PI / 6.);
            EvaluateEnergy(u, 116.5);
         }
         else
         {
            u = 0.;
            u(0) = 1.4;
            EvaluateEnergy(u, 1.);
         }

         break;
      }
      case 5:
      {
         if (dim != 2)
         {
            MFEM_ABORT("Test case only implemented in 2D.");
         }

         X(0) = 0.5 * (X(0) + 1.0);
         X(1) = 0.5 * (X(1) + 1.0);

         double r = X.Norml2();

         if (r > t / 3.)
         {
            u(0) = 1.0 + t / r;
            u(1) = -X(0) / r * u(0);
            u(2) = -X(1) / r * u(0);
            EvaluateEnergy(u, 1.0E-6);
         }
         else
         {
            u(0) = 16.0;
            u(1) = 0.0;
            u(2) = 0.0;
            EvaluateEnergy(u, 16.0 / 3.0);
         }

         break;
      }
      case 6:
      {
         if (dim != 2)
         {
            MFEM_ABORT("Test case only implemented in 2D.");
         }

         bool left = X(0) < 0. && X(0)*X(0) + 0.24 * (X(1)+1.)*(X(1)+1.) > 1.05;
         X(0) = 0.5 * (X(0) + 1.);
         X(1) = 0.25 * (X(1) + 1.);

         if (left)
         {
            u(0) = 8.;
            u(1) = 66. * cos(M_PI / 6.);
            u(2) = -66. * sin(M_PI / 6.);
            EvaluateEnergy(u, 116.5);
         }
         else
         {
            u = 0.;
            u(0) = 1.4;
            EvaluateEnergy(u, 1.);
         }

         break;
      }
      default:
         MFEM_ABORT("Analytical solution not known.");
   }
}

void InitialConditionEuler(const Vector &x, Vector &u)
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
      case 3:
      case 5:
      case 6:
      {
         AnalyticalSolutionEuler(x, 0.0, u);
         break;
      }
      case 1:
      {
         for (int i = 0; i < dim; i++)
         {
            X(i) = 0.5 * (X(i) + 1.);
         }

         u = 0.0;
         u(0) = X.Norml2() < 0.5 ? 1.0 : 0.125;
         EvaluateEnergy(u, X.Norml2() < 0.5 ? 1.0 : 0.1);
         break;
      }
      case 2:
      {
         if (dim != 1)
         {
            MFEM_ABORT("Test case only implemented in 1D.");
         }

         X(0) = 0.5 * (X(0) + 1.0);

         u = 0.0;
         u(0) = 1.0;
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
      case 4:
      {
         X *= 5.0;

         u = 0.0;
         u(0) = 1.0;
         // TODO make sure that energy is essentially a delta distribution.
         u(dim+1) = X.Norml2() < 1.0E-1 ? 100.0 : 1.0E-8;
         break;
      }
      case 7:
      {
         u(0) = 1.;
         u(1) = 1.;
         u(2) = 0.;
         EvaluateEnergy(u, 0.1);
         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}

void InflowFunctionEuler(const Vector &x, double t, Vector &u)
{
   switch (ConfigEuler.ConfigNum)
   {
      case 0:
      case 3:
      case 5:
      case 6:
      {
         AnalyticalSolutionEuler(x, t, u);
         break;
      }
      case 7:
      {
         InitialConditionEuler(x, u);
         break;
      }
      case 1:
      case 2:
      case 4: break; // No boundary condition needed.
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}
