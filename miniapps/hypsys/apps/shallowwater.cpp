#include "shallowwater.hpp"

Configuration ConfigSWE;

double GravConst;
double Depth;

void AnalyticalSolutionSWE(const Vector &x, double t, Vector &u);
void InitialConditionSWE(const Vector &x, Vector &u);
void InflowFunctionSWE(const Vector &x, double t, Vector &u);

ShallowWater::ShallowWater(FiniteElementSpace *fes_, BlockVector &u_block,
                           Configuration &config_)
   : HyperbolicSystem(fes_, u_block, fes_->GetMesh()->Dimension() + 1, config_,
                      VectorFunctionCoefficient(fes_->GetMesh()->Dimension() + 1,
                                                InflowFunctionSWE))
{
   ConfigSWE = config_;
   VectorFunctionCoefficient ic(NumEq, InitialConditionSWE);

   switch (ConfigSWE.ConfigNum)
   {
      case 0:
      {
         // Periodic meshes must be used for this problem.
         ProblemName = "Shallow Water Equations - Vorticity Advection";
         glvis_scale = "on";
         GravConst = 1.0;
         Depth = 1.0;
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 0;
         L2_Projection(ic, u0);
         break;
      }
      case 1:
      {
         ProblemName = "Shallow Water Equations - Dam Break";
         glvis_scale = "on";
         GravConst = 9.81;
         Depth = 1.0;
         SolutionKnown = true;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         L2_Projection(ic, u0);
         break;
      }
      case 2:
      {
         ProblemName = "Shallow Water Equations - Radial Dam Break";
         glvis_scale = "off valuerange 0.1 1";
         GravConst = 9.81;
         Depth = 0.1;
         SolutionKnown = false;
         SteadyState = false;
         TimeDepBC = false;
         ProjType = 1;
         u0.ProjectCoefficient(ic);
         break;
      }
      case 3:
      {
         ProblemName = "Shallow Water Equations - Constricted Channel";
         glvis_scale = "on";
         GravConst = 0.16;
         Depth = 1.0;
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

void ShallowWater::EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                                int e, int k, int i) const
{
   CheckAdmissibility(u);
   switch (dim)
   {
      case 1:
      {
         FluxEval(0,0) = u(1);
         FluxEval(1,0) = u(1) * u(1) / u(0) + 0.5 * GravConst * u(0) * u(0);
         break;
      }
      case 2:
      {
         double vx = u(1) / u(0);
         double vy = u(2) / u(0);
         double gravitation = 0.5 * GravConst * u(0) * u(0);

         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);

         FluxEval(1,0) = u(1) * vx + gravitation;
         FluxEval(1,1) = u(1) * vy;

         FluxEval(2,0) = u(2) * vx;
         FluxEval(2,1) = u(2) * vy + gravitation;
         break;
      }
      default:
         MFEM_ABORT("Invalid space dimension.");
   }
}

double ShallowWater::GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                                  int i) const
{
   CheckAdmissibility(u);
   switch (u.Size())
   {
      case 2:
         return abs( u(1)*n(0) / u(0) ) + sqrt(GravConst * u(0));
      case 3:
         return abs( (u(1)*n(0) + u(2)*n(1)) / u(0) ) + sqrt(GravConst * u(0));
   }
}

void ShallowWater::CheckAdmissibility(const Vector &u) const
{
   double HMin = 1.e-12;

   if (u.Size() != NumEq) { MFEM_ABORT("Invalid solution vector."); }

   if (u(0) < HMin)
   {
      ostringstream height_str;
      height_str << u(0);
      string err_msg = "Water height too small H = ";
      MFEM_ABORT(err_msg << height_str.str() << ".");
   }
}

void ShallowWater::SetBdrCond(const Vector &y1, Vector &y2,
                              const Vector &normal, int attr) const
{
   switch (attr)
   {
      case -1: // Land boundary
      {
         if (normal.Size() == 1)
         {
            y2(0) = y1(0);
            y2(1) = -y1(1);
         }
         else
         {
            double MomTimesNor = y1(1) * normal(0) + y1(2) * normal(1);
            y2(0) = y1(0);
            y2(1) = y1(1) - 2. * MomTimesNor * normal(0);
            y2(2) = y1(2) - 2. * MomTimesNor * normal(1);
         }
         break;
      }
      case -2: // Radiation boundary
      {
         y2 = y1;
         break;
      }
      case -3: // River boundary
      {
         break;
      }
      case -4: // Open sea boundary
      {
         double tmp = y2(0);
         y2 = y1;
         y2(0) = tmp;
         break;
      }
      default:
         MFEM_ABORT("Invalid boundary attribute.");
   }
}

void ShallowWater::ComputeDerivedQuantities(const GridFunction &u, GridFunction &d1, GridFunction &d2) const
{
   double height, momentum;
   const IntegrationRule ir = u.FESpace()->GetFE(0)->GetNodes();

   for (int e = 0; e < ne; e++)
   {
      for (int i = 0; i < nd; i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         height = u.GetValue(e, ip, 1);
         momentum = u.GetValue(e, ip, 2);
         d1(e*nd + i) = pow(momentum / height, 2.0);

         if (dim==2)
         {
            momentum = u.GetValue(e, ip, 3);
            d1(e*nd + i) += pow(momentum / height, 2.0);
         }
         d1(e*nd + i) = sqrt(d1(e*nd + i));
      }
   }
}

void ShallowWater::ComputeErrors(Array<double> &errors, const GridFunction &u,
                                 double DomainSize, double t) const
{
   errors.SetSize(NumEq*3);
   Vector component(dim+1);
   VectorFunctionCoefficient uAnalytic(NumEq, AnalyticalSolutionSWE);

   if (ConfigSWE.ConfigNum == 0) { uAnalytic.SetTime(0); }
   else { uAnalytic.SetTime(t); }

   component = 0.0;
   component(0) = 1.0;
   VectorConstantCoefficient weight1(component);
   errors[0] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight1) / DomainSize;
   errors[1] = u.ComputeLpError(2.0, uAnalytic, NULL, &weight1) / DomainSize;
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight1);

   component = 0.0;
   component(1) = 1.0;
   VectorConstantCoefficient weight2(component);
   errors[3] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight2) / DomainSize;
   errors[4] = u.ComputeLpError(2.0, uAnalytic, NULL, &weight2) / DomainSize;
   errors[5] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight2);

   if (dim ==  2)
   {
      component = 0.0;
      component(2) = 1.0;
      VectorConstantCoefficient weight3(component);
      errors[6] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight3) / DomainSize;
      errors[7] = u.ComputeLpError(2.0, uAnalytic, NULL, &weight3) / DomainSize;
      errors[8] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight3);
   }
}


void AnalyticalSolutionSWE(const Vector &x, double t, Vector &u)
{
   const int dim = x.Size();
   Vector X(dim);

   for (int i = 0; i < dim; i++)
   {
      switch (ConfigSWE.ConfigNum)
      {
         case 0: // Map to the reference domain [-1,1]^d.
         {
            double center = 0.5 * (ConfigSWE.bbMin(i) + ConfigSWE.bbMax(i));
            double factor = 2.0 / (ConfigSWE.bbMax(i) - ConfigSWE.bbMin(i));
            X(i) = factor * (x(i) - center);
            t *= pow(factor, 1.0 / (double(dim)));
            break;
         }
         case 1: // Map to the reference domain [0,1]^d.
         {
            double factor = 1.0 / (ConfigSWE.bbMax(i) - ConfigSWE.bbMin(i));
            X(i) = factor * (x(i) - ConfigSWE.bbMin(i));
            t *= pow(factor, 1.0 / (double(dim)));
            break;
         }
      }
   }

   switch (ConfigSWE.ConfigNum)
   {
      case 0:
      {
         if (dim != 2) { MFEM_ABORT("Test case works only in 2D."); }

         // Map to test case specific domain [-50,50].
         X *= 50.;
         t *= 50.;

         double M = sqrt(2);
         double c1 = -0.1;
         double c2 = 0.005;
         double a = M_PI / 4.0;
         double x0 = 0.0;
         double y0 = 0.0;

         double f = -c2 * ( pow(X(0) - x0 - M*t*cos(a), 2.0)
                          + pow(X(1) - y0 - M*t*sin(a), 2.0) );

         u(0) = 1.0;
         u(1) = M*cos(a) + c1 * (X(1) - y0 - M*t*sin(a)) * exp(f);
         u(2) = M*sin(a) - c1 * (X(0) - x0 - M*t*cos(a)) * exp(f);
         u *= Depth - c1*c1 / (4.0*c2*GravConst) * exp(2.0*f);

         break;
      }
      case 1:
      {
         // Map to test case specific domain [0,1000]^d.
         X *= 1000;
         t *= 1000;

         double r =  X(0);
         u = 0.;

         if (t==0)
         {
            u(0) = r < 500.0 ? Depth + 9.0 : Depth;
            return;
         }

         double cm = 6.23416;
         double aux = sqrt(10.0 * GravConst);
         double xA = 500.0 - t*aux;
         double xB = 500.0 + t*(2.0*aux - 3.0*cm);
         double xC = 500.0 + t*(2.0*cm*cm*(aux - cm))/(cm*cm - GravConst);

         u(0) = 9.0 * (r<xA) + (4.0/(9.0*GravConst) * pow( aux - (r-500.0)/(2.0*t), 2.0 ) - 1.0) * (r>=xA) * (r<xB)
               + (cm*cm/GravConst - 1.) * (r>=xB) * (r<xC);
         u(1) = 2.0/3.0 * ((r-500.0)/t + aux) * (r >= xA) * (r < xB) + 2.0 * (aux - cm) * (r >= xB) * (r < xC);
         u(0) += Depth;
         u(1) *= u(0);

         break;
      }
      case 3:
      {
         if (dim != 2) { MFEM_ABORT("Test case works only in 2D."); }

         const double x1[2]={-10., 0.}, x2[2]={-10., 40.},
                      x3[2]={53.8622, 5.5872}, x4[2]={53.8622, 34.4128},
                      slope0=0.53886, slope1=0.79893;
         int sign_top, sign_bot;

         if (x(0)>x3[0])
         {
            sign_top = -(x(0)-x4[0])*slope1-(x(1)-x4[1])>0 ? 1 : -1;
            sign_bot = (x(0)-x3[0])*slope1-(x(1)-x3[1])>0 ? 1 : -1;
            u(0) = sign_top*sign_bot>0. ? 0.8350436: 0.5273361;
         }
         else
         {
            sign_top = -(x(0)-x2[0])*slope0-(x(1)-x2[1])>0 ? 1 : -1;
            sign_bot = (x(0)-x1[0])*slope0-(x(1)-x1[1])>0 ? 1 : -1;
            u(0) = sign_top*sign_bot>0. ? 0.250133 : (sign_top>0 ? 1. : 0.5273361);
         }

         u(0) += Depth;
         break;
      }
   }
}

void InitialConditionSWE(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   Vector X(dim);

   // Map to the reference domain [-1,1]^d.
   for (int i = 0; i < dim; i++)
   {
      double center = 0.5 * (ConfigSWE.bbMin(i) + ConfigSWE.bbMax(i));
      double factor = 2.0 / (ConfigSWE.bbMax(i) - ConfigSWE.bbMin(i));
      X(i) = factor * (x(i) - center);
   }

   switch (ConfigSWE.ConfigNum)
   {
      case 0:
      case 1:
      {
         AnalyticalSolutionSWE(x, 0., u);
         break;
      }
      case 2:
      {
         u = 0.0;
         u(0) = X.Norml2() < 0.5 ? 0.9 : 0.0;
         u(0) += Depth;
         break;
      }
      case 3:
      {
         u(0) = Depth;
         u(1) = Depth;
         u(2) = 0.;
         break;
      }
   }
}

void InflowFunctionSWE(const Vector &x, double t, Vector &u)
{
   switch (ConfigSWE.ConfigNum)
   {
      case 0:
      case 2:
      {
         // Do not impose inflow values in this setup.
         break;
      }
      case 1:
      {
         AnalyticalSolutionSWE(x, 0., u);
         break;
      }
      case 3:
      {
         InitialConditionSWE(x, u);
         break;
      }
   }
}
