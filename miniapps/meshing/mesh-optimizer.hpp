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
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

double discrete_size_2d(const Vector &x)
{
   int opt = 2;
   const double small = 0.001, big = 0.01;
   double val = 0.;

   if (opt == 1) // sine wave.
   {
      const double X = x(0), Y = x(1);
      val = std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
            std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);
   }
   else if (opt == 2) // semi-circle
   {
      const double xc = x(0) - 0.0, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      double r1 = 0.45; double r2 = 0.55; double sf=30.0;
      val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   }

   val = std::max(0.,val);
   val = std::min(1.,val);

   return val * small + (1.0 - val) * big;
}

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

class HRHessianCoefficient : public TMOPMatrixCoefficient
{
private:
   int dim;
   // 0 - size target in an annular region,
   // 1 - size+aspect-ratio in an annular region,
   // 2 - size+aspect-ratio target for a rotate sine wave.
   int hr_target_type;

public:
   HRHessianCoefficient(int dim_, int hr_target_type_ = 0)
      : TMOPMatrixCoefficient(dim_), dim(dim_),
        hr_target_type(hr_target_type_) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      if (hr_target_type == 0) // size only circle
      {
         double small = 0.001, big = 0.01;
         if (dim == 3) { small = 0.005, big = 0.1; }
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         double zc;
         if (dim == 3) { zc = pos(2) - 0.5; }
         double r = sqrt(xc*xc + yc*yc);
         if (dim == 3) { r = sqrt(xc*xc + yc*yc + zc*zc); }
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         double ind = (tan1 - tan2);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double val = ind * small + (1.0 - ind) * big;
         K = 0.0;
         K(0, 0) = 1.0;
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
         K(0, 0) *= pow(val,0.5);
         K(1, 1) *= pow(val,0.5);
         if (dim == 3) { K(2, 2) = pow(val,0.5); }
      }
      else if (hr_target_type == 1) // circle with size and AR
      {
         const double small = 0.001, big = 0.01;
         const double xc = pos(0)-0.5, yc = pos(1)-0.5;
         const double rv = xc*xc + yc*yc;
         double r = 0;
         if (rv>0.) {r = sqrt(rv);}

         double r1 = 0.2; double r2 = 0.3; double sf=30.0;
         const double szfac = 1;
         const double asfac = 4;
         const double eps2 = szfac/asfac;
         const double eps1 = szfac;

         double tan1 = std::tanh(sf*(r-r1)+1),
                tan2 = std::tanh(sf*(r-r2)-1);
         double wgt = 0.5*(tan1-tan2);

         tan1 = std::tanh(sf*(r-r1)),
         tan2 = std::tanh(sf*(r-r2));

         double ind = (tan1 - tan2);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double szval = ind * small + (1.0 - ind) * big;

         double th = std::atan2(yc,xc)*180./M_PI;
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         double maxval = eps2 + eps1*(1-wgt)*(1-wgt);
         double minval = eps1;
         double avgval = 0.5*(maxval+minval);
         double ampval = 0.5*(maxval-minval);
         double val1 = avgval + ampval*sin(2.*th*M_PI/180.+90*M_PI/180.);
         double val2 = avgval + ampval*sin(2.*th*M_PI/180.-90*M_PI/180.);

         K(0,1) = 0.0;
         K(1,0) = 0.0;
         K(0,0) = val1;
         K(1,1) = val2;

         K(0,0) *= pow(szval,0.5);
         K(1,1) *= pow(szval,0.5);
      }
      else if (hr_target_type == 2) // sharp rotated sine wave
      {
         double xc = pos(0)-0.5, yc = pos(1)-0.5;
         double th = 15.5*M_PI/180.;
         double xn =  cos(th)*xc + sin(th)*yc;
         double yn = -sin(th)*xc + cos(th)*yc;
         double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
         double stretch = 1/cos(th2);
         xc = xn/stretch;
         yc = yn;
         double tfac = 20;
         double s1 = 3;
         double s2 = 2;
         double yl1 = -0.025;
         double yl2 =  0.025;
         double wgt = std::tanh((tfac*(yc-yl1) + s2*std::sin(s1*M_PI*xc)) + 1) -
                      std::tanh((tfac*(yc-yl2) + s2*std::sin(s1*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         const double eps2 = 25;
         const double eps1 = 1;
         K(1,1) = eps1/eps2 + eps1*(1-wgt)*(1-wgt);
         K(0,0) = eps1;
         K(0,1) = 0.0;
         K(1,0) = 0.0;
      }
      else { MFEM_ABORT("Unsupported option / wrong input."); }
   }

   virtual void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip, int comp)
   {
      K = 0.;
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

// Used for exact surface alignment
double circle_level_set(const Vector &x)
{
   const int dim = x.Size();
   if (dim == 2)
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      return r-0.2; // circle of radius 0.1
   }
   else
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
      const double r = sqrt(xc*xc + yc*yc + zc*zc);
      return std::tanh(2.0*(r-0.3));
   }
}

double squircle_level_set(const Vector &x)
{
   double power = 4.0;
   const int dim = x.Size();
   if (dim == 2)
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      const double r2 = pow(xc, power) + pow(yc, power);
      return r2 - pow(0.1, power);
   }
   else
   {
      MFEM_ABORT("Squircle level set implemented for only 2D right now.");
      return 0.0;
   }
}

double butterfly_level_set(const Vector &x)
{
   const int dim = x.Size();
   if (dim == 2)
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      double theta = atan2(yc, xc);
      if (theta < 0) { theta += 2.0*M_PI; }

      return r - (1./80.)*(12.0 - sin(theta) -2.0*cos(4.0*theta));
   }
   else
   {
      MFEM_ABORT("Butterfy level set implemented for only 2D right now.");
      return 0.0;
   }
}

double in_circle(const Vector &x, const Vector &x_center, double radius)
{
   Vector x_current = x;
   x_current -= x_center;
   double dist = x_current.Norml2();
   if (dist < radius)
   {
      return 1.0;
   }
   else if (dist == radius)
   {
      return 0.0;
   }
   else
   {
      return -1.0;
   }
   return dist <= radius ? 1.0 : 0.0;
}

double in_triangle(const Vector px, Vector p1, Vector p2, Vector p3)
{
   Vector v0 = p1;
   Vector v1 = p2; v1 -=p1;
   Vector v2 = p3; v2 -=p1;
   double p, q;
   p = ((px(0)*v2(1)-px(1)*v2(0))-(v0(0)*v2(1)-v0(1)*v2(0))) /
       (v1(0)*v2(1)-v1(1)*v2(0));
   q = -((px(0)*v1(1)-px(1)*v1(0))-(v0(0)*v1(1)-v0(1)*v1(0))) /
       (v1(0)*v2(1)-v1(1)*v2(0));

   return (p > 0 && q > 0 && 1-p-q > 0) ? 1.0 : -1.0;
}

double in_propeller(double a, double b, double d, const Vector &x, double theta)
{
   Vector xr = x;
   xr(0) = x(0)*cos(theta) - x(1)*sin(theta);
   xr(1) = x(0)*sin(theta) + x(1)*cos(theta);

   double val;
   val = b*b*xr(0)*xr(0) + a*a*xr(1)*xr(1) + 2*xr(0)*xr(1)*xr(1) + xr(1)*xr(
            1) - a*a*b*b;
   val = xr(0)*xr(0)/(a*a) + xr(1)*xr(1)*exp(2*xr(0)*log(2.4))/(b*b) - 1.0;
   return val < 0.0 ? 1.0 : -1.0;
}

double snowman(const Vector &x)
{
   // Base
   Vector x_circle1(2);
   x_circle1(0) = 0.5;
   x_circle1(1) = 0.35;
   double in_circle1_val = in_circle(x, x_circle1, 0.25);
   // Head
   Vector x_circle(x.Size());
   x_circle(0) = 0.5;
   x_circle(1) = 0.7;
   double circle_radius = 0.15;
   double in_circle2_val = in_circle(x, x_circle, circle_radius);
   return max(in_circle1_val, in_circle2_val);
}

double mickeymouse(const Vector &x)
{
   // Big circle
   Vector x_circle1(2);
   x_circle1(0) = 0.5;
   x_circle1(1) = 0.5;
   double in_circle1_val = in_circle(x, x_circle1, 0.25);
   // Circle 2
   Vector x_circle(x.Size());
   x_circle(0) = 0.3;
   x_circle(1) = 0.65;
   double circle_radius = 0.1;
   double in_circle2_val = in_circle(x, x_circle, circle_radius);
   // Circle 3
   x_circle(0) = 0.7;
   double in_circle3_val = in_circle(x, x_circle, circle_radius);
   return max(max(in_circle1_val, in_circle2_val), in_circle3_val);
}

double ball_balance(const Vector &x)
{
   // Circle
   double dx = 0.0;
   Vector x_circle1(2);
   x_circle1(0) = 0.5+dx;
   x_circle1(1) = 0.6;
   double in_circle1_val = in_circle(x, x_circle1, 0.171);

   Vector p1(x.Size()), p2(x.Size()), p3(x.Size());
   p1(0) = 0.5+dx; p1(1) = 0.51;
   p2(0) = 0.71+dx; p2(1) = 0.24;
   p3(0) = 0.29+dx; p3(1) = 0.24;
   double in_triangle_val = in_triangle(x, p1, p2, p3);

   return max(in_triangle_val, in_circle1_val);
}

double propeller(const Vector &x)
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.5;
   x_circle1(1) = 0.5;
   double in_circle1_val = in_circle(x, x_circle1, 0.151);

   // Blade 1
   double a = 0.35,
          b = 0.11,
          d = 1.0;
   Vector xcc = x;
   xcc -= x_circle1;
   xcc *= 1.5;
   xcc(0) -= 0.25;
   double in_propeller_val = in_propeller(a, b, d, xcc, 0.0);

   double final_val = max(in_propeller_val, in_circle1_val);

   // Blade 2
   double theta = -120.0*M_PI/180.0;
   xcc = x;
   xcc -= x_circle1;
   xcc *= 1.5;
   xcc(0) -= 0.25*cos(theta);
   xcc(1) += 0.25*sin(theta);

   in_propeller_val = in_propeller(a, b, d, xcc, theta);
   final_val = max(final_val, in_propeller_val);

   // Blade 3
   theta = 120.0*M_PI/180.0;
   xcc = x;
   xcc -= x_circle1;
   xcc *= 1.5;
   xcc(0) -= 0.25*cos(theta);
   xcc(1) += 0.25*sin(theta);

   in_propeller_val = in_propeller(a, b, d, xcc, theta);

   return max(final_val, in_propeller_val);
}

double in_trapezium(const Vector &x, double a, double b, double l)
{
   double phi_t = x(1) + (a-b)*x(0)/l - a;
   if (phi_t <= 0.0)
   {
      return 1.0;
   }
   return -1.0;
}

double in_parabola(const Vector &x, double h, double k, double t)
{
   double phi_p1 = (x(0)-h-t/2) - k*x(1)*x(1);
   double phi_p2 = (x(0)-h+t/2) - k*x(1)*x(1);
   if (phi_p1 <= 0.0 && phi_p2 >= 0.0)
   {
      return 1.0;
   }
   return -1.0;
}

// xc, yc = center, w,h = width and height
double in_rectangle(const Vector &x, double xc, double yc, double w, double h)
{
   double dx = fabs(x(0) - xc);
   double dy = fabs(x(1) - yc);
   if (dx <= w/2 && dy <= h/2)
   {
      return 1.0;
   }
   else
   {
      return -1.0;
   }
}

double reactor(const Vector &x)
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   double in_circle1_val = in_circle(x, x_circle1, 0.2);

   double r1 = 0.2;
   double r2 = 1.0;
   double in_trapezium_val = in_trapezium(x,0.05, 0.1, r2-r1);

   double return_val = max(in_circle1_val, in_trapezium_val);

   double h = 0.4;
   double k = 2;
   double t = 0.15;
   double in_parabola_val = in_parabola(x, h, k, t);
   return_val = max(return_val, in_parabola_val);

   double in_rectangle_val = in_rectangle(x, 1.0, 0.0, 0.12, 0.35);
   return_val = max(return_val, in_rectangle_val);

   double in_rectangle_val2 = in_rectangle(x, 1.0, 0.5, 0.12, 0.28);
   return_val = max(return_val, in_rectangle_val2);
   return return_val;
}

double in_parabola_xy(const Vector &x,
                      double hx, double hy,
                      double kx, double ky,
                      double tx, double ty,
                      double powerx, double powery,
                      double minv, double maxv,
                      int idir, int check_flip)
{
   double phi_p1 = kx*pow(x(0)-hx-tx/2, powerx) - ky*pow(x(1)-hy-ty/2, powery);
   double phi_p2 = kx*pow(x(0)-hx+tx/2, powerx) - ky*pow(x(1)-hy+ty/2, powery);
   if (check_flip == 0 && phi_p1 >= 0.0 && phi_p2 <= 0.0 && x(idir) >= minv &&
       x(idir) <= maxv)
   {
      return 1.0;
   }
   else if (check_flip == 1 && phi_p1 <= 0.0 && phi_p2 >= 0.0 && x(idir) >= minv &&
            x(idir) <= maxv)
   {
      return 1.0;
   }
   return -1.0;
}

int material_id(int el_id, const GridFunction &g)
{
   const FiniteElementSpace *fes = g.FESpace();
   const FiniteElement *fe = fes->GetFE(el_id);
   Vector g_vals;
   const IntegrationRule &ir =
      IntRules.Get(fe->GetGeomType(), fes->GetOrder(el_id) + 2);

   double integral = 0.0;
   g.GetValues(el_id, ir, g_vals);
   ElementTransformation *Tr = fes->GetMesh()->GetElementTransformation(el_id);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      integral += ip.weight * g_vals(q) * Tr->Weight();
   }
   return (integral > 0.0) ? 1.0 : 0.0;
}

void DiffuseField(GridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator
   BilinearForm *Lap = new BilinearForm(field.FESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();

   // Setup the smoothing operator
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
   // Setup the Laplacian operator
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

void ComputeScalarDistanceFromLevelSet(ParMesh &pmesh,
                                       FunctionCoefficient &ls_coeff,
                                       int amr_iter, ParGridFunction &distance_s)
{
   mfem::H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                               pmesh.Dimension());
   mfem::ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   mfem::ParGridFunction x(&h1fespace);

   mfem::L2_FECollection l2fec(0, pmesh.Dimension());
   mfem::ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);

   mfem::H1_FECollection lhfec(1, pmesh.Dimension());
   mfem::ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   x.ProjectCoefficient(ls_coeff);
   x.ExchangeFaceNbrData();

   for (int iter = 0; iter < amr_iter; iter++)
   {
      el_to_refine = 0.0;
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         h1fespace.GetElementDofs(e, dofs);
         x.GetSubVector(dofs, x_vals);
         int refine = 0;
         double min_val = 100;
         double max_val = -100;
         for (int j = 0; j < x_vals.Size(); j++)
         {
            double x_dof_val = x_vals(j);
            min_val = min(x_dof_val, min_val);
            max_val = max(x_dof_val, max_val);
         }
         if (min_val < 0 && max_val > 0)
         {
            refine = 1;
            el_to_refine(e) = 1.0;
         }
      }

      //Refine an element if its neighbor will be refined
      el_to_refine.ExchangeFaceNbrData();
      GridFunctionCoefficient field_in_dg(&el_to_refine);
      lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         lhfespace.GetElementDofs(e, dofs);
         lhx.GetSubVector(dofs, x_vals);
         int refine = 0;
         double max_val = -100;
         for (int j = 0; j < x_vals.Size(); j++)
         {
            double x_dof_val = x_vals(j);
            max_val = max(x_dof_val, max_val);
         }
         if (max_val > 0)
         {
            refine = 1;
            el_to_refine(e) = 1.0;
         }
      }

      //make the list of elements to be refined
      Array<int> el_to_refine_list;
      for (int e = 0; e < el_to_refine.Size(); e++)
      {
         if (el_to_refine(e) > 0.0)
         {
            el_to_refine_list.Append(e);
         }
      }

      int loc_count = el_to_refine_list.Size();
      int glob_count = loc_count;
      MPI_Allreduce(&loc_count, &glob_count, 1, MPI_INT, MPI_SUM,
                    pmesh.GetComm());
      MPI_Barrier(pmesh.GetComm());
      if (glob_count > 0)
      {
         pmesh.GeneralRefinement(el_to_refine_list, 1);
      }
      h1fespace.Update();
      x.Update();
      x.ProjectCoefficient(ls_coeff);

      l2fespace.Update();
      el_to_refine.Update();

      lhfespace.Update();
      lhx.Update();

      distance_s.ParFESpace()->Update();
      distance_s.Update();
   }

   //Now determine distance
   const double dx = AvgElementSize(pmesh);
   DistanceSolver *dist_solver = NULL;
   int solver_type = 1;
   double t_param = 1.0;

   if (solver_type == 0)
   {
      auto ds = new HeatDistanceSolver(t_param * dx * dx);
      ds->smooth_steps = 0;
      ds->vis_glvis = false;
      dist_solver = ds;
   }
   else if (solver_type == 1)
   {
      const int p = 5;
      const int newton_iter = 50;
      auto ds = new PLapDistanceSolver(p, newton_iter);
      dist_solver = ds;
   }
   else { MFEM_ABORT("Wrong solver option."); }
   dist_solver->print_level = 1;

   ParFiniteElementSpace pfes_s(*distance_s.ParFESpace());

   // Smooth-out Gibbs oscillations from the input level set. The smoothing
   // parameter here is specified to be mesh dependent with length scale dx.
   ParGridFunction filt_gf(&pfes_s);
   PDEFilter filter(pmesh, 1.0 * dx);
   filter.Filter(ls_coeff, filt_gf);
   GridFunctionCoefficient ls_filt_coeff(&filt_gf);

   dist_solver->ComputeScalarDistance(ls_filt_coeff, distance_s);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();

   DiffuseField(distance_s, 5);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();
   for (int i = 0; i < distance_s.Size(); i++)
   {
      distance_s(i) *= distance_s(i);
   }
}
#endif
