// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

// x = t.
// y = 0.
// The distance is the error in y.
class Line_Bottom : public Analytic2DCurve
{
   public:
   Line_Bottom(const Array<bool> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = x;
      dist = y - 0.0;
   }
   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      x = t;
      y = 0.0 + dist;
   }

   virtual double dx_dt(double t) const override { return 1.0; }
   virtual double dy_dt(double t) const override { return 0.0; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = 0.
// y = t.
// The distance is the error in x.
class Line_Left : public Analytic2DCurve
{
   public:
   Line_Left(const Array<bool> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = y;
      dist = x - 0.0;
   }
   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      x = 0.0 + dist;
      y = t;
   }

   virtual double dx_dt(double t) const override { return 0.0; }
   virtual double dy_dt(double t) const override { return 1.0; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = 1.5 t.
// y = 1 + 0.5 t.
// The distance is the error in y.
class Line_Top : public Analytic2DCurve
{
   public:
   Line_Top(const Array<bool> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = x / 1.5;
      dist = y - (1 + 0.5 * t);
   }
   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      x = 1.5 * t;
      y = dist + 1.0 + 0.5 * t;
   }

   virtual double dx_dt(double t) const override { return 1.5; }
   virtual double dy_dt(double t) const override { return 0.5; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = 1 + 0.5 t.
// y = 1.5 t.
// The distance is the error in x.
class Line_Right : public Analytic2DCurve
{
   public:
   Line_Right(const Array<bool> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = y / 1.5;
      dist = x - (1 + 0.5 * t);
   }
   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      x = dist + 1.0 + 0.5 * t;
      y = 1.5 * t;
   }

   virtual double dx_dt(double t) const override { return 0.5; }
   virtual double dy_dt(double t) const override { return 1.5; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = [1.0 + a sin(c pi) + b] t.
// y = 1 + a sin(c pi t) + b t.
// The distance is the error in y.
class Curve_Sine_Top : public Analytic2DCurve
{
private:
   const double a, b, c, x_scale;

public:
   Curve_Sine_Top(const Array<bool> &marker, double a_, double b_, double c_)
    : Analytic2DCurve(marker),
    a(a_), b(b_), c(c_), x_scale(1.0 + a * sin(c * M_PI) + b) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = x / x_scale;
      dist = y - (1.0 + a * sin(c * M_PI * t) + b * t);
   }
   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      x = x_scale * t;
      y = dist + 1.0 + a * sin(c * M_PI * t) + b * t;
   }

   virtual double dx_dt(double t) const override
   { return x_scale; }
   virtual double dy_dt(double t) const override
   { return a * c * M_PI * cos(c * M_PI * t) + b; }

   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * t); }
};

// x = 1 + a sin(s pi t) + b t.
// y = [1.0 + a sin(s pi) + b] t.
// The distance is the error in x.
class Curve_Sine_Right : public Analytic2DCurve
{
private:
   const double a, b, c, y_scale;

public:
   Curve_Sine_Right(const Array<bool> &marker, double a_, double b_, double c_)
    : Analytic2DCurve(marker),
      a(a_), b(b_), c(c_), y_scale(1.0 + a * sin(c * M_PI) + b) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = y / y_scale;
      dist = x - (1.0 + a * sin(c * M_PI * t) + b * t);
   }
   void xy_of_t(double t,  double dist, double &x, double &y) const override
   {
      x = dist + 1.0 + a * sin(c * M_PI * t) + b * t;
      y = y_scale * t;
   }

   virtual double dx_dt(double t) const override
   { return a * c * M_PI * cos(c * M_PI * t) + b; }
   virtual double dy_dt(double t) const override
   { return y_scale; }


   virtual double dx_dtdt(double t) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * t); }
   virtual double dy_dtdt(double t) const override
   { return 0.0; }
};

class AxisAlignedEdge : public Analytic3DCurve
{
private:
   const int axis;
   const double fixed_1;
   const double fixed_2;

public:
   // Creates straight edge aligned with an axis (0 <-> x, 1 <-> y, 2 <-> z).
   // fixed_1 and fixed_2 give the coordinates on the normal plane to the axis.
   AxisAlignedEdge(const Array<bool> &marker, int axis_,
                   double fixed_1_, double fixed_2_)
      : Analytic3DCurve(marker),
        axis(axis_),
        fixed_1(fixed_1_),
        fixed_2(fixed_2_) { }

   void t_of_xyz(double x, double y, double z,
                 double &dist1, double &dist2, double &t) const override
   {
      if (axis == 0)
      {
         t = x;
         dist1 = y - fixed_1;
         dist2 = z - fixed_2;
      }
      else if (axis == 1)
      {
         t = y;
         dist1 = x - fixed_1;
         dist2 = z - fixed_2;
      }
      else
      {
         t = z;
         dist1 = x - fixed_1;
         dist2 = y - fixed_2;
      }
   }

   void xyz_of_t(double t, double dist1, double dist2,
                 double &x, double &y, double &z) const override
   {
      if (axis == 0)
      {
         x = t;
         y = fixed_1 + dist1;
         z = fixed_2 + dist2;
      }
      else if (axis == 1)
      {
         x = fixed_1 + dist1;
         y = t;
         z = fixed_2 + dist2;
      }
      else
      {
         x = fixed_1 + dist1;
         y = fixed_2 + dist2;
         z = t;
      }
   }

   double dx_dt(double) const override { return axis == 0 ? 1.0 : 0.0; }
   double dy_dt(double) const override { return axis == 1 ? 1.0 : 0.0; }
   double dz_dt(double) const override { return axis == 2 ? 1.0 : 0.0; }

   double dx_dtdt(double) const override { return 0.0; }
   double dy_dtdt(double) const override { return 0.0; }
   double dz_dtdt(double) const override { return 0.0; }
};

class AxisAlignedPlane : public Analytic3DSurface
{
private:
   const int normal_axis;
   const double fixed_value;

public:
   AxisAlignedPlane(const Array<bool> &marker, int normal_axis_,
                    double fixed_value_)
      : Analytic3DSurface(marker),
        normal_axis(normal_axis_),
        fixed_value(fixed_value_) { }

   void uv_of_xyz(double x, double y, double z,
                  double &dist, double &u, double &v) const override
   {
      if (normal_axis == 0)
      {
         dist = x - fixed_value;
         u = y;
         v = z;
      }
      else if (normal_axis == 1)
      {
         dist = y - fixed_value;
         u = x;
         v = z;
      }
      else
      {
         dist = z - fixed_value;
         u = x;
         v = y;
      }
   }

   void xyz_of_uv(double u, double v, double dist,
                  double &x, double &y, double &z) const override
   {
      if (normal_axis == 0)
      {
         x = fixed_value + dist;
         y = u;
         z = v;
      }
      else if (normal_axis == 1)
      {
         x = u;
         y = fixed_value + dist;
         z = v;
      }
      else
      {
         x = u;
         y = v;
         z = fixed_value + dist;
      }
   }

   double dx_du(double, double) const override { return normal_axis == 1 || normal_axis == 2 ? 1.0 : 0.0; }
   double dy_du(double, double) const override { return normal_axis == 0 ? 1.0 : 0.0; }
   double dz_du(double, double) const override { return 0.0; }
   double dx_dv(double, double) const override { return 0.0; }
   double dy_dv(double, double) const override { return normal_axis == 2 ? 1.0 : 0.0; }
   double dz_dv(double, double) const override { return normal_axis == 0 || normal_axis == 1 ? 1.0 : 0.0; }

   double dx_dudu(double, double) const override { return 0.0; }
   double dy_dudu(double, double) const override { return 0.0; }
   double dz_dudu(double, double) const override { return 0.0; }
   double dx_dudv(double, double) const override { return 0.0; }
   double dy_dudv(double, double) const override { return 0.0; }
   double dz_dudv(double, double) const override { return 0.0; }
   double dx_dvdv(double, double) const override { return 0.0; }
   double dy_dvdv(double, double) const override { return 0.0; }
   double dz_dvdv(double, double) const override { return 0.0; }
};

class CubeEdge_XY : public Analytic3DCurve
{
private:
   const double a, b, c;
   const double s = sin(c * M_PI);

public:
   CubeEdge_XY(const Array<bool> &marker, double a_, double b_, double c_)
      : Analytic3DCurve(marker), a(a_), b(b_), c(c_) { }

   void t_of_xyz(double x, double y, double z,
                 double &dist1, double &dist2, double &t) const override
   {
      t = std::max(0.0, std::min(1.0, z));
      for (int it = 0; it < 12; it++)
      {
         const double f = z_of_t(t) - z;
         const double df = dz_dt(t);
         if (std::abs(df) < 1e-12) { break; }
         const double dt = -f / df;
         t = std::max(0.0, std::min(1.0, t + dt));
         if (std::abs(dt) < 1e-12) { break; }
      }
      dist1 = x - x_of_t(t);
      dist2 = y - y_of_t(t);
   }

   void xyz_of_t(double t, double dist1, double dist2,
                 double &x, double &y, double &z) const override
   {
      x = x_of_t(t) + dist1;
      y = y_of_t(t) + dist2;
      z = z_of_t(t);
   }

   double dx_dt(double t) const override
   { return a * s * c * M_PI * cos(c * M_PI * t) + b; }
   double dy_dt(double t) const override
   { return a * s * c * M_PI * cos(c * M_PI * t) + b; }
   double dz_dt(double t) const override
   { return 1.0 + 0.5 * a * s * s * M_PI * cos(0.5 * M_PI * t) + b; }

   double dx_dtdt(double t) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * t); }
   double dy_dtdt(double t) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * t); }
   double dz_dtdt(double t) const override
   { return -0.25 * a * s * s * M_PI * M_PI * sin(0.5 * M_PI * t); }

private:
   double x_of_t(double t) const
   { return 1.0 + a * s * sin(c * M_PI * t) + b * t; }
   double y_of_t(double t) const
   { return 1.0 + a * s * sin(c * M_PI * t) + b * t; }
   double z_of_t(double t) const
   { return t + a * s * s * sin(0.5 * M_PI * t) + b * t; }
};

class CubeEdge_XZ : public Analytic3DCurve
{
private:
   const double a, b, c;
   const double s = sin(c * M_PI);

public:
   CubeEdge_XZ(const Array<bool> &marker, double a_, double b_, double c_)
      : Analytic3DCurve(marker), a(a_), b(b_), c(c_) { }

   void t_of_xyz(double x, double y, double z,
                 double &dist1, double &dist2, double &t) const override
   {
      // Becuase the transformation is nonlinear, the problem to find t
      // to parameterize the curve is solved using Newton's method.
      t = std::max(0.0, std::min(1.0, y));
      for (int it = 0; it < 12; it++)
      {
         const double f = y_of_t(t) - y;
         const double df = dy_dt(t);
         if (std::abs(df) < 1e-12) { break; }
         const double dt = -f / df;
         t = std::max(0.0, std::min(1.0, t + dt));
         if (std::abs(dt) < 1e-12) { break; }
      }
      dist1 = x - x_of_t(t);
      dist2 = z - z_of_t(t);
   }

   void xyz_of_t(double t, double dist1, double dist2,
                 double &x, double &y, double &z) const override
   {
      x = x_of_t(t) + dist1;
      y = y_of_t(t);
      z = z_of_t(t) + dist2;
   }

   double dx_dt(double t) const override
   { return a * s * c * M_PI * cos(c * M_PI * t) + b; }
   double dy_dt(double t) const override
   { return 1.0 + 0.5 * a * s * s * M_PI * cos(0.5 * M_PI * t) + b; }
   double dz_dt(double t) const override
   { return a * s * c * M_PI * cos(c * M_PI * t) + b; }

   double dx_dtdt(double t) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * t); }
   double dy_dtdt(double t) const override
   { return -0.25 * a * s * s * M_PI * M_PI * sin(0.5 * M_PI * t); }
   double dz_dtdt(double t) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * t); }

private:
   double x_of_t(double t) const
   { return 1.0 + a * s * sin(c * M_PI * t) + b * t; }
   double y_of_t(double t) const
   { return t + a * s * s * sin(0.5 * M_PI * t) + b * t; }
   double z_of_t(double t) const
   { return 1.0 + a * s * sin(c * M_PI * t) + b * t; }
};

class CubeEdge_YZ : public Analytic3DCurve
{
private:
   const double a, b, c;
   const double s = sin(c * M_PI);

public:
   CubeEdge_YZ(const Array<bool> &marker, double a_, double b_, double c_)
      : Analytic3DCurve(marker), a(a_), b(b_), c(c_) { }

   void t_of_xyz(double x, double y, double z,
                 double &dist1, double &dist2, double &t) const override
   {
      // Becuase the transformation is nonlinear, the problem to find t
      // to parameterize the curve is solved using Newton's method.
      t = std::max(0.0, std::min(1.0, x));
      for (int it = 0; it < 12; it++)
      {
         const double f = x_of_t(t) - x;
         const double df = dx_dt(t);
         if (std::abs(df) < 1e-12) { break; }
         const double dt = -f / df;
         t = std::max(0.0, std::min(1.0, t + dt));
         if (std::abs(dt) < 1e-12) { break; }
      }
      dist1 = y - y_of_t(t);
      dist2 = z - z_of_t(t);
   }

   void xyz_of_t(double t, double dist1, double dist2,
                 double &x, double &y, double &z) const override
   {
      x = x_of_t(t);
      y = y_of_t(t) + dist1;
      z = z_of_t(t) + dist2;
   }

   double dx_dt(double t) const override
   { return 1.0 + 0.5 * a * s * s * M_PI * cos(0.5 * M_PI * t) + b; }
   double dy_dt(double t) const override
   { return a * s * c * M_PI * cos(c * M_PI * t) + b; }
   double dz_dt(double t) const override
   { return a * s * c * M_PI * cos(c * M_PI * t) + b; }

   double dx_dtdt(double t) const override
   { return -0.25 * a * s * s * M_PI * M_PI * sin(0.5 * M_PI * t); }
   double dy_dtdt(double t) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * t); }
   double dz_dtdt(double t) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * t); }

private:
   double x_of_t(double t) const
   { return t + a * s * s * sin(0.5 * M_PI * t) + b * t; }
   double y_of_t(double t) const
   { return 1.0 + a * s * sin(c * M_PI * t) + b * t; }
   double z_of_t(double t) const
   { return 1.0 + a * s * sin(c * M_PI * t) + b * t; }
};

class CubeFace_X : public Analytic3DSurface
{
private:
   const double a, b, c;
   const double s = sin(c * M_PI);

public:
   CubeFace_X(const Array<bool> &marker, double a_, double b_, double c_)
      : Analytic3DSurface(marker), a(a_), b(b_), c(c_) { }

   void uv_of_xyz(double x, double y, double z,
                  double &dist, double &u, double &v) const override
   {
      // Becuase the transformation is nonlinear, the problem to find u and v 
      // to parameterize the surface is solved using Newton's method.
      u = std::max(0.0, std::min(1.0, y));
      v = std::max(0.0, std::min(1.0, z));
      for (int it = 0; it < 15; it++)
      {
         const double r1 = y_of_uv(u, v) - y;
         const double r2 = z_of_uv(u, v) - z;
         const double j11 = dy_du(u, v);
         const double j12 = dy_dv(u, v);
         const double j21 = dz_du(u, v);
         const double j22 = dz_dv(u, v);
         const double det = j11 * j22 - j12 * j21;
         if (std::abs(det) < 1e-12) { break; }
         const double du = (-r1 * j22 + r2 * j12) / det;
         const double dv = (-j11 * r2 + j21 * r1) / det;
         u = std::max(0.0, std::min(1.0, u + du));
         v = std::max(0.0, std::min(1.0, v + dv));
         if (std::max(std::abs(du), std::abs(dv)) < 1e-12) { break; }
      }
      dist = x - x_of_uv(u, v);
   }

   void xyz_of_uv(double u, double v, double dist,
                  double &x, double &y, double &z) const override
   {
      x = x_of_uv(u, v) + dist;
      y = y_of_uv(u, v);
      z = z_of_uv(u, v);
   }

   double dx_du(double u, double v) const override
   { return a * c * M_PI * cos(c * M_PI * u) * sin(c * M_PI * v) + b * v; }
   double dy_du(double u, double v) const override
   { return 1.0 + 0.5 * a * s * M_PI * cos(0.5 * M_PI * u) * sin(c * M_PI * v) + b * v; }
   double dz_du(double u, double v) const override
   { return a * s * c * M_PI * cos(c * M_PI * u) * sin(0.5 * M_PI * v) + b * v; }
   double dx_dv(double u, double v) const override
   { return a * c * M_PI * sin(c * M_PI * u) * cos(c * M_PI * v) + b * u; }
   double dy_dv(double u, double v) const override
   { return a * s * c * M_PI * sin(0.5 * M_PI * u) * cos(c * M_PI * v) + b * u; }
   double dz_dv(double u, double v) const override
   { return 1.0 + 0.5 * a * s * M_PI * sin(c * M_PI * u) * cos(0.5 * M_PI * v) + b * u; }

   double dx_dudu(double u, double v) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(c * M_PI * v); }
   double dy_dudu(double u, double v) const override
   { return -0.25 * a * s * M_PI * M_PI * sin(0.5 * M_PI * u) * sin(c * M_PI * v); }
   double dz_dudu(double u, double v) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(0.5 * M_PI * v); }
   double dx_dudv(double u, double v) const override
   { return a * c * c * M_PI * M_PI * cos(c * M_PI * u) * cos(c * M_PI * v) + b; }
   double dy_dudv(double u, double v) const override
   { return 0.5 * a * s * c * M_PI * M_PI * cos(0.5 * M_PI * u) * cos(c * M_PI * v) + b; }
   double dz_dudv(double u, double v) const override
   { return 0.5 * a * s * c * M_PI * M_PI * cos(c * M_PI * u) * cos(0.5 * M_PI * v) + b; }
   double dx_dvdv(double u, double v) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(c * M_PI * v); }
   double dy_dvdv(double u, double v) const override
   { return -a * s * c * c * M_PI * M_PI * sin(0.5 * M_PI * u) * sin(c * M_PI * v); }
   double dz_dvdv(double u, double v) const override
   { return -0.25 * a * s * M_PI * M_PI * sin(c * M_PI * u) * sin(0.5 * M_PI * v); }

private:
   double x_of_uv(double u, double v) const
   { return 1.0 + a * sin(c * M_PI * u) * sin(c * M_PI * v) + b * u * v; }
   double y_of_uv(double u, double v) const
   { return u + a * s * sin(0.5 * M_PI * u) * sin(c * M_PI * v) + b * u * v; }
   double z_of_uv(double u, double v) const
   { return v + a * s * sin(c * M_PI * u) * sin(0.5 * M_PI * v) + b * u * v; }
};

class CubeFace_Y : public Analytic3DSurface
{
private:
   const double a, b, c;
   const double s = sin(c * M_PI);

public:
   CubeFace_Y(const Array<bool> &marker, double a_, double b_, double c_)
      : Analytic3DSurface(marker), a(a_), b(b_), c(c_) { }

   void uv_of_xyz(double x, double y, double z,
                  double &dist, double &u, double &v) const override
   {
      // Becuase the transformation is nonlinear, the problem to find u and v 
      // to parameterize the surface is solved using Newton's method.
      u = std::max(0.0, std::min(1.0, x));
      v = std::max(0.0, std::min(1.0, z));
      for (int it = 0; it < 15; it++)
      {
         const double r1 = x_of_uv(u, v) - x;
         const double r2 = z_of_uv(u, v) - z;
         const double j11 = dx_du(u, v);
         const double j12 = dx_dv(u, v);
         const double j21 = dz_du(u, v);
         const double j22 = dz_dv(u, v);
         const double det = j11 * j22 - j12 * j21;
         if (std::abs(det) < 1e-12) { break; }
         const double du = (-r1 * j22 + r2 * j12) / det;
         const double dv = (-j11 * r2 + j21 * r1) / det;
         u = std::max(0.0, std::min(1.0, u + du));
         v = std::max(0.0, std::min(1.0, v + dv));
         if (std::max(std::abs(du), std::abs(dv)) < 1e-12) { break; }
      }
      dist = y - y_of_uv(u, v);
   }

   void xyz_of_uv(double u, double v, double dist,
                  double &x, double &y, double &z) const override
   {
      x = x_of_uv(u, v);
      y = y_of_uv(u, v) + dist;
      z = z_of_uv(u, v);
   }

   double dx_du(double u, double v) const override
   { return 1.0 + 0.5 * a * s * M_PI * cos(0.5 * M_PI * u) * sin(c * M_PI * v) + b * v; }
   double dy_du(double u, double v) const override
   { return a * c * M_PI * cos(c * M_PI * u) * sin(c * M_PI * v) + b * v; }
   double dz_du(double u, double v) const override
   { return a * s * c * M_PI * cos(c * M_PI * u) * sin(0.5 * M_PI * v) + b * v; }
   double dx_dv(double u, double v) const override
   { return a * s * c * M_PI * sin(0.5 * M_PI * u) * cos(c * M_PI * v) + b * u; }
   double dy_dv(double u, double v) const override
   { return a * c * M_PI * sin(c * M_PI * u) * cos(c * M_PI * v) + b * u; }
   double dz_dv(double u, double v) const override
   { return 1.0 + 0.5 * a * s * M_PI * sin(c * M_PI * u) * cos(0.5 * M_PI * v) + b * u; }

   double dx_dudu(double u, double v) const override
   { return -0.25 * a * s * M_PI * M_PI * sin(0.5 * M_PI * u) * sin(c * M_PI * v); }
   double dy_dudu(double u, double v) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(c * M_PI * v); }
   double dz_dudu(double u, double v) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(0.5 * M_PI * v); }
   double dx_dudv(double u, double v) const override
   { return 0.5 * a * s * c * M_PI * M_PI * cos(0.5 * M_PI * u) * cos(c * M_PI * v) + b; }
   double dy_dudv(double u, double v) const override
   { return a * c * c * M_PI * M_PI * cos(c * M_PI * u) * cos(c * M_PI * v) + b; }
   double dz_dudv(double u, double v) const override
   { return 0.5 * a * s * c * M_PI * M_PI * cos(c * M_PI * u) * cos(0.5 * M_PI * v) + b; }
   double dx_dvdv(double u, double v) const override
   { return -a * s * c * c * M_PI * M_PI * sin(0.5 * M_PI * u) * sin(c * M_PI * v); }
   double dy_dvdv(double u, double v) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(c * M_PI * v); }
   double dz_dvdv(double u, double v) const override
   { return -0.25 * a * s * M_PI * M_PI * sin(c * M_PI * u) * sin(0.5 * M_PI * v); }

private:
   double x_of_uv(double u, double v) const
   { return u + a * s * sin(0.5 * M_PI * u) * sin(c * M_PI * v) + b * u * v; }
   double y_of_uv(double u, double v) const
   { return 1.0 + a * sin(c * M_PI * u) * sin(c * M_PI * v) + b * u * v; }
   double z_of_uv(double u, double v) const
   { return v + a * s * sin(c * M_PI * u) * sin(0.5 * M_PI * v) + b * u * v; }
};

class CubeFace_Z : public Analytic3DSurface
{
private:
   const double a, b, c;
   const double s = sin(c * M_PI);

public:
   CubeFace_Z(const Array<bool> &marker, double a_, double b_, double c_)
      : Analytic3DSurface(marker), a(a_), b(b_), c(c_) { }

   void uv_of_xyz(double x, double y, double z,
                  double &dist, double &u, double &v) const override
   {
      // Becuase the transformation is nonlinear, the problem to find u and v 
      // to parameterize the surface is solved using Newton's method.
      u = std::max(0.0, std::min(1.0, x));
      v = std::max(0.0, std::min(1.0, y));
      for (int it = 0; it < 15; it++)
      {
         const double r1 = x_of_uv(u, v) - x;
         const double r2 = y_of_uv(u, v) - y;
         const double j11 = dx_du(u, v);
         const double j12 = dx_dv(u, v);
         const double j21 = dy_du(u, v);
         const double j22 = dy_dv(u, v);
         const double det = j11 * j22 - j12 * j21;
         if (std::abs(det) < 1e-12) { break; }
         const double du = (-r1 * j22 + r2 * j12) / det;
         const double dv = (-j11 * r2 + j21 * r1) / det;
         u = std::max(0.0, std::min(1.0, u + du));
         v = std::max(0.0, std::min(1.0, v + dv));
         if (std::max(std::abs(du), std::abs(dv)) < 1e-12) { break; }
      }
      dist = z - z_of_uv(u, v);
   }

   void xyz_of_uv(double u, double v, double dist,
                  double &x, double &y, double &z) const override
   {
      x = x_of_uv(u, v);
      y = y_of_uv(u, v);
      z = z_of_uv(u, v) + dist;
   }

   double dx_du(double u, double v) const override
   { return 1.0 + 0.5 * a * s * M_PI * cos(0.5 * M_PI * u) * sin(c * M_PI * v) + b * v; }
   double dy_du(double u, double v) const override
   { return a * s * c * M_PI * cos(c * M_PI * u) * sin(0.5 * M_PI * v) + b * v; }
   double dz_du(double u, double v) const override
   { return a * c * M_PI * cos(c * M_PI * u) * sin(c * M_PI * v) + b * v; }
   double dx_dv(double u, double v) const override
   { return a * s * c * M_PI * sin(0.5 * M_PI * u) * cos(c * M_PI * v) + b * u; }
   double dy_dv(double u, double v) const override
   { return 1.0 + 0.5 * a * s * M_PI * sin(c * M_PI * u) * cos(0.5 * M_PI * v) + b * u; }
   double dz_dv(double u, double v) const override
   { return a * c * M_PI * sin(c * M_PI * u) * cos(c * M_PI * v) + b * u; }

   double dx_dudu(double u, double v) const override
   { return -0.25 * a * s * M_PI * M_PI * sin(0.5 * M_PI * u) * sin(c * M_PI * v); }
   double dy_dudu(double u, double v) const override
   { return -a * s * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(0.5 * M_PI * v); }
   double dz_dudu(double u, double v) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(c * M_PI * v); }
   double dx_dudv(double u, double v) const override
   { return 0.5 * a * s * c * M_PI * M_PI * cos(0.5 * M_PI * u) * cos(c * M_PI * v) + b; }
   double dy_dudv(double u, double v) const override
   { return 0.5 * a * s * c * M_PI * M_PI * cos(c * M_PI * u) * cos(0.5 * M_PI * v) + b; }
   double dz_dudv(double u, double v) const override
   { return a * c * c * M_PI * M_PI * cos(c * M_PI * u) * cos(c * M_PI * v) + b; }
   double dx_dvdv(double u, double v) const override
   { return -a * s * c * c * M_PI * M_PI * sin(0.5 * M_PI * u) * sin(c * M_PI * v); }
   double dy_dvdv(double u, double v) const override
   { return -0.25 * a * s * M_PI * M_PI * sin(c * M_PI * u) * sin(0.5 * M_PI * v); }
   double dz_dvdv(double u, double v) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * u) * sin(c * M_PI * v); }

private:
   double x_of_uv(double u, double v) const
   { return u + a * s * sin(0.5 * M_PI * u) * sin(c * M_PI * v) + b * u * v; }
   double y_of_uv(double u, double v) const
   { return v + a * s * sin(c * M_PI * u) * sin(0.5 * M_PI * v) + b * u * v; }
   double z_of_uv(double u, double v) const
   { return 1.0 + a * sin(c * M_PI * u) * sin(c * M_PI * v) + b * u * v; }
};
