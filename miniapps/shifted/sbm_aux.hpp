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

#include "../../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
double dist_value(const Vector &x, const int type)
{
   double ring_radius = 0.2;
   if (type == 1 || type == 2) // circle of radius 0.2 - centered at 0.5, 0.5
   {
      double dx = x(0) - 0.5,
             dy = x(1) - 0.5,
             rv = dx*dx + dy*dy;
      if (x.Size() == 3)
      {
         double dz = x(2) - 0.5;
         rv += dz*dz;
      }
      rv = rv > 0 ? pow(rv, 0.5) : 0;
      return rv - ring_radius; // positive is the domain
   }
   else if (type == 3) // walls at y = 0.0
   {
      return x(1);
   }
   else if (type == 4)
   {
      const int num_circ = 3;
      double rad[num_circ] = {0.3, 0.15, 0.2};
      double c[num_circ][2] = { {0.6, 0.6}, {0.3, 0.3}, {0.25, 0.75} };

      const double xc = x(0), yc = x(1);

      // circle 0
      double r0 = (xc-c[0][0])*(xc-c[0][0]) + (yc-c[0][1])*(yc-c[0][1]);
      r0 = (r0 > 0) ? std::sqrt(r0) : 0.0;
      if (r0 <= 0.2) { return -1.0; }

      for (int i = 0; i < num_circ; i++)
      {
         double r = (xc-c[i][0])*(xc-c[i][0]) + (yc-c[i][1])*(yc-c[i][1]);
         r = (r > 0) ? std::sqrt(r) : 0.0;
         if (r <= rad[i]) { return 1.0; }
      }

      // rectangle 1
      if (0.7 <= xc && xc <= 0.8 && 0.1 <= yc && yc <= 0.8) { return 1.0; }

      // rectangle 2
      if (0.3 <= xc && xc <= 0.8 && 0.15 <= yc && yc <= 0.2) { return 1.0; }
      return -1.0;
   }
   else
   {
      MFEM_ABORT(" Function type not implement yet.");
   }
   return 0.;
}

/// Level set coefficient - +1 inside the domain, -1 outside, 0 at the boundary.
class Dist_Level_Set_Coefficient : public Coefficient
{
private:
   int type;

public:
   Dist_Level_Set_Coefficient(int type_)
      : Coefficient(), type(type_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);
      double dist = dist_value(x, type);
      if (dist >= 0.) { return 1.; }
      else { return -1.; }
   }
};

/// Distance vector to the zero level-set.
class Dist_Vector_Coefficient : public VectorCoefficient
{
private:
   int type;

public:
   Dist_Vector_Coefficient(int dim_, int type_)
      : VectorCoefficient(dim_), type(type_) { }

   virtual void Eval(Vector &p, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector x;
      T.Transform(ip, x);
      const int dim = x.Size();
      p.SetSize(dim);
      if (type == 1 || type == 2)
      {
         double dist0 = dist_value(x, type);
         for (int i = 0; i < dim; i++) { p(i) = 0.5 - x(i); }
         double length = p.Norml2();
         p *= dist0/length;
      }
      else if (type == 3)
      {
         double dist0 = dist_value(x, type);
         p(0) = 0.;
         p(1) = -dist0;
      }
   }
};

/// Boundary conditions
double dirichlet_velocity_circle(const Vector &x)
{
   return 0.;
}

double dirichlet_velocity_xy_exponent(const Vector &x)
{
   double xy_p = 2.; // exponent for level set 2 where u = x^p+y^p;
   return pow(x(0), xy_p) + pow(x(1), xy_p);
}

double dirichlet_velocity_xy_sinusoidal(const Vector &x)
{
   return 1./(M_PI*M_PI)*std::sin(M_PI*x(0)*x(1));
}


/// `f` for the Poisson problem (-nabla^2 u = f).
double rhs_fun_circle(const Vector &x)
{
   return 1;
}

double rhs_fun_xy_exponent(const Vector &x)
{
   double xy_p = 2.; // exponent for level set 2 where u = x^p+y^p;
   double coeff = std::max(xy_p*(xy_p-1), 1.);
   double expon = std::max(0., xy_p-2);
   if (xy_p == 1)
   {
      return 0.;
   }
   else
   {
      return -coeff*std::pow(x(0), expon) - coeff*std::pow(x(1), expon);
   }
}

double rhs_fun_xy_sinusoidal(const Vector &x)
{
   return std::sin(M_PI*x(0)*x(1))*(x(0)*x(0)+x(1)*x(1));
}
