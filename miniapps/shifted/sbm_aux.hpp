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

double point_inside_trigon(const Vector px, Vector p1, Vector p2, Vector p3)
{
   Vector v0 = p1;
   Vector v1 = p2; v1 -=p1;
   Vector v2 = p3; v2 -=p1;
   double p, q;
   p = ((px(0)*v2(1)-px(1)*v2(0))-(v0(0)*v2(1)-v0(1)*v2(0)))/(v1(0)*v2(1)-v1(1)*v2(
                                                                 0));
   q = -((px(0)*v1(1)-px(1)*v1(0))-(v0(0)*v1(1)-v0(1)*v1(0)))/(v1(0)*v2(1)-v1(
                                                                  1)*v2(0));

   if (p > 0 && q > 0 && 1-p-q > 0)
   {
      return -1.0;
   }
   return 1.0;
}

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
double dist_value(const Vector &x, const int type)
{

   double ring_radius = 0.2;
   if (type == 1 || type == 2) // circle of radius 0.2 - centered at 0.5, 0.5
   {
      Vector xc(x.Size());
      xc = 0.5;
      xc -= x;
      return xc.Norml2() - ring_radius; // positive is the domain
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
   else if (type == 5) // square of side 0.2 centered at 0.75, 0.25
   {
      double square_side = 0.2;
      Vector xc(x.Size());
      xc = 0.75; xc(1) = 0.25;
      xc -= x;
      if (abs(xc(0)) > 0.5*square_side || abs(xc(1)) > 0.5*square_side)
      {
         return 1.0;
      }
      else
      {
         return -1.0;
      }
      return 0.0;
   }
   else if (type == 6) // Triangle
   {
      Vector p1(x.Size()), p2(x.Size()), p3(x.Size());
      p1(0) = 0.25; p1(1) = 0.4;
      p2(0) = 0.1; p2(1) = 0.1;
      p3(0) = 0.4; p3(1) = 0.1;
      return point_inside_trigon(x, p1, p2, p3);
   }
   else if (type == 7) // circle of radius 0.2 - centered at 0.5, 0.6
   {
      Vector xc(x.Size());
      xc = 0.5; xc(1) = 0.6;
      xc -= x;
      return xc.Norml2() - 0.2;
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

/// Level set coefficient - +1 inside the domain, -1 outside, 0 at the boundary.
class Combo_Level_Set_Coefficient : public Coefficient
{
private:
   Array<Dist_Level_Set_Coefficient *> dls;

public:
   Combo_Level_Set_Coefficient() : Coefficient() { }

   void Add_Level_Set_Coefficient(Dist_Level_Set_Coefficient &dls_)
   { dls.Append(&dls_); }

   int GetNLevelSets() { return dls.Size(); }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      MFEM_VERIFY(dls.Size() > 0, "Add at-least 1 Dist_level_Set_Coefficient to"
                  " the Combo.");
      double dist = dls[0]->Eval(T, ip);
      for (int j = 1; j < dls.Size(); j++)
      {
         dist = min(dist, dls[j]->Eval(T, ip));
      }
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

   using VectorCoefficient::Eval;

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

/// Boundary conditions - Dirichlet
double dirichlet_velocity_circle(const Vector &x)
{
   return 0.;
}

double unity(const Vector &x)
{
   return 0.015;
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

/// Boundary conditions - Neumann
double neumann_velocity_circle(const Vector &x)
{
   return 0.;
}

/// Normal vector for level_set_type = 1. Circle centered at [0.5 , 0.5]
void normal_vector_1(const Vector &x, Vector &p)
{
   p.SetSize(x.Size());
   p(0) = x(0)-0.5;
   p(1) = x(1)-0.5; // center of circle at [0.5, 0.5]
   p /= p.Norml2();
   p *= -1;
}

/// Normal vector for level_set_type = 7. Circle centered at [0.75 , 0.25]
void normal_vector_2(const Vector &x, Vector &p)
{
   p.SetSize(x.Size());
   p(0) = x(0)-0.5;
   p(1) = x(1)-0.6; // center of circle at [0.5, 0.6]
   p /= p.Norml2();
   p *= -1;
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
