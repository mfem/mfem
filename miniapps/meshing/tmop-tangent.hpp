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
   Line_Bottom(const Array<int> &marker) : Analytic2DCurve(marker) { }

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
   Line_Left(const Array<int> &marker) : Analytic2DCurve(marker) { }

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
   Line_Top(const Array<int> &marker) : Analytic2DCurve(marker) { }

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
   Line_Right(const Array<int> &marker) : Analytic2DCurve(marker) { }

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
   Curve_Sine_Top(const Array<int> &marker, double a_, double b_, double c_)
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
   Curve_Sine_Right(const Array<int> &marker, double a_, double b_, double c_)
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
   { return 0.2 * c * M_PI * cos(c * M_PI * t) + b; }
   virtual double dy_dt(double t) const override
   { return y_scale; }


   virtual double dx_dtdt(double t) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * t); }
   virtual double dy_dtdt(double t) const override
   { return 0.0; }
};

