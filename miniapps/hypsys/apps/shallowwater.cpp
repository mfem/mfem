#include "shallowwater.hpp"

Configuration ConfigShallowWater;

double GravConst;

void AnalyticalSolutionSWE(const Vector &x, double t, Vector &u);
void InitialConditionSWE(const Vector &x, Vector &u);
void InflowFunctionSWE(const Vector &x, double t, Vector &u);

ShallowWater::ShallowWater(FiniteElementSpace *fes_, BlockVector &u_block,
                           Configuration &config_)
   : HyperbolicSystem(fes_, u_block, fes_->GetMesh()->Dimension() + 1, config_,
                      VectorFunctionCoefficient(fes_->GetMesh()->Dimension() + 1,
                                                InflowFunctionSWE))
{
   ConfigShallowWater = config_;

   VectorFunctionCoefficient ic(NumEq, InitialConditionSWE);

   switch (ConfigShallowWater.ConfigNum)
   {
      case 0:
      {
         ProblemName = "Shallow Water Equations - Vorticity Advection";
         glvis_scale = "on";
         GravConst = 1.0;
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false; // Usage of periodic meshes is required.
         ProjType = 0;
         L2_Projection(ic, u0);
         break;
      }
      case 1:
      {
         ProblemName = "Shallow Water Equations - Dam Break";
         glvis_scale = "off \n valuerange 0 1";
         GravConst = 1.0;
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

void ShallowWater::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                                int e, int k, int i) const
{
   const int dim = u.Size() - 1;
   double HMin = 0.001;

   if (u.Size() != NumEq)
   {
      MFEM_ABORT("Invalid solution vector.");
   }
   if (u(0) < HMin)
   {
      ostringstream height_str;
      height_str << u(0);
      string err_msg = "Water height too small H = ";
      MFEM_ABORT(err_msg << height_str.str());
   }

   switch (dim)
   {
      case 1:
      {
         FluxEval(0,0) = u(1);
         FluxEval(1,0) = u(1)*u(1)/u(0) + GravConst / 2. * u(0)*u(0);
         break;
      }
      case 2:
      {
         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);

         FluxEval(1,0) = u(1)*u(1)/u(0) + 0.5 * GravConst * u(0)*u(0);
         FluxEval(1,1) = u(1)*u(2)/u(0);

         FluxEval(2,0) = u(2)*u(1)/u(0);
         FluxEval(2,1) = u(2)*u(2)/u(0) + 0.5 * GravConst * u(0)*u(0);
         break;
      }
      default:
         MFEM_ABORT("Invalid space dimension.");
   }
}

double ShallowWater::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                                  int i) const
{
   switch (u.Size())
   {
      case 2:
         return abs(u(1)*n(0)) / u(0) + sqrt(GravConst * u(0));
      case 3:
         return abs(u(1)*n(0) + u(2)*n(1)) / u(0) + sqrt(GravConst * u(0));
      default: MFEM_ABORT("Invalid solution vector.");
   }
}

void ShallowWater::SetBdrCond(const Vector &y1, Vector &y2,
                              const Vector &normal, int attr) const
{
   switch (attr)
   {
      case -1: // land boundary
      {
         if (normal.Size() == 1)
         {
            y2(0) = y1(0);
            y2(1) = -y1(1);
         }
         else
         {
            double Mom_x_Norm = y1(1) * normal(0) + y1(2) * normal(1);
            y2(0) = y1(0);
            y2(1) = y1(1) - 2. * Mom_x_Norm * normal(0);
            y2(2) = y1(2) - 2. * Mom_x_Norm * normal(1);
         }
         return;
      }
      case -2: // radiation boundary
      {
         y2 = y1;
         return;
      }
      case -3: // river boundary
      {
         return;
      }
      case -4: // open sea boundary
      {
         double tmp = y2(0);
         y2 = y1;
         y2(0) = tmp;
         return;
      }
      default:
         MFEM_ABORT("Invalid boundary attribute.");
   }
}

void ShallowWater::ComputeErrors(Array<double> &errors, const GridFunction &u,
                                 double DomainSize, double t) const
{
   errors.SetSize(3);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionSWE);
   // Right now we use initial condition = solution due to periodic mesh.
   uAnalytic.SetTime(0);
   errors[0] = u.ComputeLpError(1., uAnalytic) / DomainSize;
   errors[1] = u.ComputeLpError(2., uAnalytic) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
}


void AnalyticalSolutionSWE(const Vector &x, double t, Vector &u)
{
   const int dim = x.Size();
   u.SetSize(dim+1);

   // Map to the reference domain [-1,1].
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (ConfigShallowWater.bbMin(i) + ConfigShallowWater.bbMax(
                          i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (ConfigShallowWater.bbMax(
                                        i) - ConfigShallowWater.bbMin(i));
   }

   switch (ConfigShallowWater.ConfigNum)
   {
      case 0:
      {
         if (dim != 2)
         {
            MFEM_ABORT("Test case is only implemented in 2D.");
         }

         X *= 50.; // Map to test case specific domain [-50,50].

         // Parameters
         double M = .5;
         double c1 = -.04;
         double c2 = .02;
         double a = M_PI / 4.;
         double x0 = 0.;
         double y0 = 0.;

         double f = -c2 * ( pow(X(0) - x0 - M*t*cos(a), 2.)
                            + pow(X(1) - y0 - M*t*sin(a), 2.) );

         u(0) = 1.;
         u(1) = M*cos(a) + c1 * (X(1) - y0 - M*t*sin(a)) * exp(f);
         u(2) = M*sin(a) - c1 * (X(0) - x0 - M*t*cos(a)) * exp(f);
         u *= 1. - c1*c1 / (4.*c2*GravConst) * exp(2.*f);

         break;
      }
      case 1:
      {
         u = 0.;
         u(0) = X.Norml2() < 0.5 ? 1. : .125;
         break;
      }
      default:
         MFEM_ABORT("No such test case implemented.");
   }
}

void InitialConditionSWE(const Vector &x, Vector &u)
{
   AnalyticalSolutionSWE(x, 0., u);
}

void InflowFunctionSWE(const Vector &x, double t, Vector &u)
{
   AnalyticalSolutionSWE(x, t, u);
}
