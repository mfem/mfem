// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// MFEM Mesh Optimizer Miniapp - Serial/Parallel Shared Code

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

class DiscreteSize2D : public Coefficient
{
protected:
   const int ref_levels, type;

public:
   DiscreteSize2D(int ref_levels_, int type_ = 2)
      : ref_levels(ref_levels_), type(type_)  { }

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      const double small = 0.016/pow(4., ref_levels), big = 0.16/pow(4., ref_levels);
      double val = 0.;
      Vector x(3);
      T.Transform(ip, x);

      if (type == 1) // sine wave.
      {
         const double X = x(0), Y = x(1);
         val = std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
               std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);
      }
      else if (type == 2) // semi-circle
      {
         const double xc = x(0) - 0.0, yc = x(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         double r1 = 0.45; double r2 = 0.55; double sf=30.0;
         val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
      }
      else if (type == 3)
      {
         double dxyl = 0.25;
         const double xc = fabs(x(0) - 0.5), yc = fabs(x(1) - 0.5);
         const double dxy = max(xc, yc);
         if (dxy <= dxyl) { val = 1.0; }
         else { val = 0.0; }
         return val * (dxyl*dxyl/16) + (1.0 - val)*(1 - dxyl*dxyl)/(64-16);
      }

      val = std::max(0.,val);
      val = std::min(1.,val);

      return val * small + (1.0 - val) * big;
   }
};


class DiscreteSize3D : public Coefficient
{
protected:
   const int ref_levels;

public:
   DiscreteSize3D(int ref_levels_)
      : ref_levels(ref_levels_) { }

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);

      double small = 0.0064/pow(8., ref_levels),
             big   =   0.64/pow(8., ref_levels);

      double val = 0.;

      // semi-circle
      const double xc = x(0) - 0.0, yc = x(1) - 0.5, zc = x(2) - 0.5;
      const double r = sqrt(xc*xc + yc*yc + zc*zc);
      double r1 = 0.45; double r2 = 0.55; double sf=30.0;
      val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));

      val = std::max(0.,val);
      val = std::min(1.,val);

      return val * small + (1.0 - val) * big;
   }
};

double discrete_size_3d(const Vector &x)
{
   const double small = 0.0001, big = 0.01;
   double val = 0.;

   // semi-circle
   const double xc = x(0) - 0.0, yc = x(1) - 0.5, zc = x(2) - 0.5;
   const double r = sqrt(xc*xc + yc*yc + zc*zc);
   double r1 = 0.45; double r2 = 0.55; double sf=30.0;
   val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));

   val = std::max(0.,val);
   val = std::min(1.,val);

   return val * small + (1.0 - val) * big;
}

double material_indicator_2d(const Vector &x)
{
   double xc = x(0)-0.5, yc = x(1)-0.5;
   double th = 22.5*M_PI/180.;
   double xn =  cos(th)*xc + sin(th)*yc;
   double yn = -sin(th)*xc + cos(th)*yc;
   double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
   double stretch = 1/cos(th2);
   xc = xn/stretch; yc = yn/stretch;
   double tfac = 20;
   double s1 = 3;
   double s2 = 3;
   double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1);
   if (wgt > 1) { wgt = 1; }
   if (wgt < 0) { wgt = 0; }
   return wgt;
}

double discrete_ori_2d(const Vector &x)
{
   return M_PI * x(1) * (1.0 - x(1)) * cos(2 * M_PI * x(0));
}

double discrete_aspr_2d(const Vector &x)
{
   double xc = x(0)-0.5, yc = x(1)-0.5;
   double th = 22.5*M_PI/180.;
   double xn =  cos(th)*xc + sin(th)*yc;
   double yn = -sin(th)*xc + cos(th)*yc;
   // double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
   // double stretch = 1/cos(th2);
   xc = xn; yc = yn;

   double tfac = 20;
   double s1 = 3;
   double s2 = 2;
   double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1)
                - std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) - 1);
   if (wgt > 1) { wgt = 1; }
   if (wgt < 0) { wgt = 0; }
   return 0.1 + 1*(1-wgt)*(1-wgt);
}

void discrete_aspr_3d(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim);
   double l1, l2, l3;
   l1 = 1.;
   l2 = 1. + 5*x(1);
   l3 = 1. + 10*x(2);
   v[0] = l1/pow(l2*l3,0.5);
   v[1] = l2/pow(l1*l3,0.5);
   v[2] = l3/pow(l2*l1,0.5);
}

class HessianCoefficient : public TMOPMatrixCoefficient
{
private:
   int metric;

public:
   HessianCoefficient(int dim, int metric_id)
      : TMOPMatrixCoefficient(dim), metric(metric_id) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      if (metric != 14 && metric != 36 && metric != 85)
      {
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;
         const double eps = 0.5;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         K(0, 0) = eps + 1.0 * (tan1 - tan2);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
      }
      else if (metric == 14 || metric == 36) // Size + Alignment
      {
         const double xc = pos(0), yc = pos(1);
         double theta = M_PI * yc * (1.0 - yc) * cos(2 * M_PI * xc);
         double alpha_bar = 0.1;

         K(0, 0) =  cos(theta);
         K(1, 0) =  sin(theta);
         K(0, 1) = -sin(theta);
         K(1, 1) =  cos(theta);

         K *= alpha_bar;
      }
      else if (metric == 85) // Shape + Alignment
      {
         Vector x = pos;
         double xc = x(0)-0.5, yc = x(1)-0.5;
         double th = 22.5*M_PI/180.;
         double xn =  cos(th)*xc + sin(th)*yc;
         double yn = -sin(th)*xc + cos(th)*yc;
         xc = xn; yc=yn;

         double tfac = 20;
         double s1 = 3;
         double s2 = 2;
         double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1)
                      - std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         xc = pos(0), yc = pos(1);
         double theta = M_PI * (yc) * (1.0 - yc) * cos(2 * M_PI * xc);

         K(0, 0) =  cos(theta);
         K(1, 0) =  sin(theta);
         K(0, 1) = -sin(theta);
         K(1, 1) =  cos(theta);

         double asp_ratio_tar = 0.1 + 1*(1-wgt)*(1-wgt);

         K(0, 0) *=  1/pow(asp_ratio_tar,0.5);
         K(1, 0) *=  1/pow(asp_ratio_tar,0.5);
         K(0, 1) *=  pow(asp_ratio_tar,0.5);
         K(1, 1) *=  pow(asp_ratio_tar,0.5);
      }
   }

   virtual void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip, int comp)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      K = 0.;
      if (metric != 14 && metric != 85)
      {
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));
         double tan1d = 0., tan2d = 0.;
         if (r > 0.001)
         {
            tan1d = (1.-tan1*tan1)*(sf)/r,
            tan2d = (1.-tan2*tan2)*(sf)/r;
         }

         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
         if (comp == 0) { K(0, 0) = tan1d*xc - tan2d*xc; }
         else if (comp == 1) { K(0, 0) = tan1d*yc - tan2d*yc; }
      }
   }
};

// Additional IntegrationRules that can be used with the --quad-type option.
IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

// Defined with respect to the icf mesh.
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}

// Used for the adaptive limiting examples.
double adapt_lim_fun(const Vector &x)
{
   const double xc = x(0) - 0.1, yc = x(1) - 0.2;
   const double r = sqrt(xc*xc + yc*yc);
   double r1 = 0.45; double r2 = 0.55; double sf=30.0;
   double val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));

   val = std::max(0.,val);
   val = std::min(1.,val);
   return val;
}

void DiffuseField(GridFunction &field, int smooth_steps)
{
   //Setup the Laplacian operator
   BilinearForm *Lap = new BilinearForm(field.FESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();

   //Setup the smoothing operator
   DSmoother *S = new DSmoother(0,1.0,smooth_steps);
   S->iterative_mode = true;
   S->SetOperator(Lap->SpMat());

   Vector tmp(field.Size());
   tmp = 0.0;
   S->Mult(tmp, field);

   delete S;
   delete Lap;
}

#ifdef MFEM_USE_MPI
void DiffuseField(ParGridFunction &field, int smooth_steps)
{
   //Setup the Laplacian operator
   ParBilinearForm *Lap = new ParBilinearForm(field.ParFESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();
   HypreParMatrix *A = Lap->ParallelAssemble();

   HypreSmoother *S = new HypreSmoother(*A,0,smooth_steps);
   S->iterative_mode = true;

   Vector tmp(A->Width());
   field.SetTrueVector();
   Vector fieldtrue = field.GetTrueVector();
   tmp = 0.0;
   S->Mult(tmp, fieldtrue);

   field.SetFromTrueDofs(fieldtrue);

   delete S;
   delete Lap;
}
#endif


// 1D transformation at the right boundary.
double right(const double eps, const double x)
{
   return (x <= 0.5) ? (2-eps) * x : 1 + eps*(x-1);
}

// 1D transformation at the left boundary
double left(const double eps, const double x)
{
   return 1-right(eps,1-x);
}

// Transition from a value of "a" for x=0, to a value of "b" for x=1.  Optionally
// smooth -- see the commented versions at the end.
double step(const double a, const double b, double x)
{
   if (x <= 0) { return a; }
   if (x >= 1) { return b; }
   //return a + (b-a) * (x);
   //return a + (b-a) * (x*x*(3-2*x));
   return a + (b-a) * (x*x*x*(x*(6*x-15)+10));
}

// 3D version of a generalized Kershaw mesh transformation, see D. Kershaw,
// "Differencing of the diffusion equation in Lagrangian hydrodynamic codes",
// JCP, 39:375–395, 1981.
//
// The input mesh should be Cartesian nx x ny x nz with nx divisible by 6 and
// ny, nz divisible by 2.
//
// The eps parameters are in (0, 1]. Uniform mesh is recovered for epsy=epsz=1.
void kershaw(const double epsy, const double epsz,
             const double x, const double y, const double z,
             double &X, double &Y, double &Z)
{
   X = x;

   int layer = x*6.0;
   double lambda = (x-layer/6.0)*6;

   // The x-range is split in 6 layers going from left-to-left, left-to-right,
   // right-to-left (2 layers), left-to-right and right-to-right yz-faces.
   switch (layer)
   {
      case 0:
         Y = left(epsy, y);
         Z = left(epsz, z);
         break;
      case 1:
      case 4:
         Y = step(left(epsy, y), right(epsy, y), lambda);
         Z = step(left(epsz, z), right(epsz, z), lambda);
         break;
      case 2:
         Y = step(right(epsy, y), left(epsy, y), lambda/2);
         Z = step(right(epsz, z), left(epsz, z), lambda/2);
         break;
      case 3:
         Y = step(right(epsy, y), left(epsy, y), (1+lambda)/2);
         Z = step(right(epsz, z), left(epsz, z), (1+lambda)/2);
         break;
      default:
         Y = right(epsy, y);
         Z = right(epsz, z);
         break;
   }
}

void stretching2D(const double x, const double y,
                  double &X, double &Y)
{
   double amp = 0.3;
   double frq = 8.;

   double dx1 = amp*cos(frq*M_PI*(y-0.5)*(-0.5)),
          dxf = x*(1-x),
          dx2 = amp*(1-2*y);
   X = x + dx1*dxf + 0.0*dx2*dxf;

   double dy1 = amp*cos(frq*M_PI*(x-0.5)*(-0.5)),
          dyf = y*(1-y),
          dy2 = amp*(1-2*x);
   Y = y + dy1*dyf + 0.0*dy2*dyf;
}

static double rad2D(double x, double y,
                    double const xc = 0.0,
                    double const yc = 0.0)
{
   double dx = x - xc,
          dy = y - yc;
   double dr = dx*dx + dy*dy;
   if (dr > 0.0) { dr = std::pow(dr, 0.5); }
   return dr;
}

void rotation2D_substep(const double xc, const double yc,
                        const double r1, const double r2,
                        double &ux, double &uy)
{
   double rad = rad2D(xc, yc, 0.0, 0.0);
   if (rad < r1)
   {
      ux =  5.0 * yc;
      uy = -5.0 * xc;
   }
   else if (rad < r2)
   {
      ux =  2.0 * yc / rad - 5.0 * yc;
      uy = -2.0 * xc / rad + 5.0 * xc;
   }
}

void rotation2D(const double x, const double y,
                double &X, double &Y)
{
   double dt = 0.001;

   double ux, uy;
   double xc = x-0.5,
          yc = y-0.5;
   double r1 = 0.2,
          r2 = 0.4;

   for (int i = 0; i < 300; i++)
   {

      double xs = xc, ys = yc;

      rotation2D_substep(xc, yc, r1, r2, ux, uy);

      xc = xc + 0.5*dt*ux;
      yc = yc + 0.5*dt*uy;

      rotation2D_substep(xc, yc, r1, r2, ux, uy);

      xc = xs + dt*ux;
      yc = ys + dt*uy;
   }

   X = xc + 0.5;
   Y = yc + 0.5;
}

void stretching3D(const double x, const double y, const double z,
                  double &X, double &Y, double &Z)
{
   double amp = 0.3;
   double frq = 8.;

   double dx1 = amp*cos(frq*M_PI*(y-0.5)*(z-0.5)),
          dxf = x*(1-x),
          dx2 = amp*(1-2*y);
   X = x + dx1*dxf + 0.0*dx2*dxf;

   double dy1 = amp*cos(frq*M_PI*(x-0.5)*(z-0.5)),
          dyf = y*(1-y),
          dy2 = amp*(1-2*z);
   Y = y + dy1*dyf + 0.0*dy2*dyf;

   double dz1 = amp*cos(frq*M_PI*(y-0.5)*(x-0.5)),
          dzf = z*(1-z),
          dz2 = amp*(1-2*x);
   Z = z + dz1*dzf + 0.0*dz2*dzf;
}
