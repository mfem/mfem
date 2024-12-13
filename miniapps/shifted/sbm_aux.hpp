// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"

using namespace std;
using namespace mfem;

real_t point_inside_trigon(const Vector px, Vector p1, Vector p2, Vector p3)
{
   Vector v0 = p1;
   Vector v1 = p2; v1 -=p1;
   Vector v2 = p3; v2 -=p1;
   real_t p, q;
   p = ((px(0)*v2(1)-px(1)*v2(0))-(v0(0)*v2(1)-v0(1)*v2(0))) /
       (v1(0)*v2(1)-v1(1)*v2(0));
   q = -((px(0)*v1(1)-px(1)*v1(0))-(v0(0)*v1(1)-v0(1)*v1(0))) /
       (v1(0)*v2(1)-v1(1)*v2(0));

   return (p > 0 && q > 0 && 1-p-q > 0) ? -1.0 : 1.0;
}

// 1 is inside the doughnut, -1 is outside.
real_t doughnut_cheese(const Vector &coord)
{
   // map [0,1] to [-1,1].
   real_t x = 2*coord(0)-1.0, y = 2*coord(1)-1.0, z = 2*coord(2)-1.0;

   bool doughnut;
   const real_t R = 0.8, r = 0.15;
   const real_t t = R - std::sqrt(x*x + y*y);
   doughnut = t*t + z*z - r*r <= 0;

   bool cheese;
   x = 3.0*x, y = 3.0*y, z = 3.0*z;
   cheese = (x*x + y*y - 4.0) * (x*x + y*y - 4.0) +
            (z*z - 1.0) * (z*z - 1.0) +
            (y*y + z*z - 4.0) * (y*y + z*z - 4.0) +
            (x*x - 1.0) * (x*x - 1.0) +
            (z*z + x*x - 4.0) * (z*z + x*x - 4.0) +
            (y*y - 1.0) * (y*y - 1.0) - 15.0 <= 0.0;

   return (doughnut || cheese) ? 1.0 : -1.0;
}

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
real_t dist_value(const Vector &x, const int type)
{
   if (type == 1 || type == 2) // circle of radius 0.2 - centered at 0.5, 0.5
   {
      const real_t ring_radius = 0.2;
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
      real_t rad[num_circ] = {0.3, 0.15, 0.2};
      real_t c[num_circ][2] = { {0.6, 0.6}, {0.3, 0.3}, {0.25, 0.75} };

      const real_t xc = x(0), yc = x(1);

      // circle 0
      real_t r0 = (xc-c[0][0])*(xc-c[0][0]) + (yc-c[0][1])*(yc-c[0][1]);
      r0 = (r0 > 0) ? std::sqrt(r0) : 0.0;
      if (r0 <= 0.2) { return -1.0; }

      for (int i = 0; i < num_circ; i++)
      {
         real_t r = (xc-c[i][0])*(xc-c[i][0]) + (yc-c[i][1])*(yc-c[i][1]);
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
      real_t square_side = 0.2;
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
   else if (type == 8) { return doughnut_cheese(x); }
   else
   {
      MFEM_ABORT(" Function type not implement yet.");
   }
   return 0.;
}

/// Level set coefficient: +1 inside the true domain, -1 outside.
class Dist_Level_Set_Coefficient : public Coefficient
{
private:
   int type;

public:
   Dist_Level_Set_Coefficient(int type_)
      : Coefficient(), type(type_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(3);
      T.Transform(ip, x);
      real_t dist = dist_value(x, type);
      return (dist >= 0.0) ? 1.0 : -1.0;
   }
};

/// Combination of level sets: +1 inside the true domain, -1 outside.
class Combo_Level_Set_Coefficient : public Coefficient
{
private:
   Array<Dist_Level_Set_Coefficient *> dls;

public:
   Combo_Level_Set_Coefficient() : Coefficient() { }

   void Add_Level_Set_Coefficient(Dist_Level_Set_Coefficient &dls_)
   { dls.Append(&dls_); }

   int GetNLevelSets() { return dls.Size(); }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      MFEM_VERIFY(dls.Size() > 0,
                  "Add at least 1 Dist_level_Set_Coefficient to the Combo.");
      real_t dist = dls[0]->Eval(T, ip);
      for (int j = 1; j < dls.Size(); j++)
      {
         dist = min(dist, dls[j]->Eval(T, ip));
      }
      return (dist >= 0.0) ? 1.0 : -1.0;
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

   void Eval(Vector &p, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector x;
      T.Transform(ip, x);
      const int dim = x.Size();
      p.SetSize(dim);
      if (type == 1 || type == 2)
      {
         real_t dist0 = dist_value(x, type);
         for (int i = 0; i < dim; i++) { p(i) = 0.5 - x(i); }
         real_t length = p.Norml2();
         p *= dist0/length;
      }
      else if (type == 3)
      {
         real_t dist0 = dist_value(x, type);
         p(0) = 0.;
         p(1) = -dist0;
      }
   }
};

/// Boundary conditions - Dirichlet
real_t homogeneous(const Vector &x)
{
   return 0.0;
}

real_t dirichlet_velocity_xy_exponent(const Vector &x)
{
   real_t xy_p = 2.; // exponent for level set 2 where u = x^p+y^p;
   return pow(x(0), xy_p) + pow(x(1), xy_p);
}

real_t dirichlet_velocity_xy_sinusoidal(const Vector &x)
{
   return 1./(M_PI*M_PI)*std::sin(M_PI*x(0)*x(1));
}

/// Boundary conditions - Neumann
/// Normal vector for level_set_type = 1. Circle centered at [0.5 , 0.5]
void normal_vector_1(const Vector &x, Vector &p)
{
   p.SetSize(x.Size());
   p(0) = x(0)-0.5;
   p(1) = x(1)-0.5; // center of circle at [0.5, 0.5]
   p /= p.Norml2();
   p *= -1;
}

/// Normal vector for level_set_type = 7. Circle centered at [0.5 , 0.6]
void normal_vector_2(const Vector &x, Vector &p)
{
   p.SetSize(x.Size());
   p(0) = x(0)-0.5;
   p(1) = x(1)-0.6; // center of circle at [0.5, 0.6]
   p /= p.Norml2();
   p *= -1;
}

/// Neumann condition for exponent based solution
real_t traction_xy_exponent(const Vector &x)
{
   real_t xy_p = 2;
   Vector gradient(2);
   gradient(0) = xy_p*x(0);
   gradient(1) = xy_p*x(1);
   Vector normal(2);
   normal_vector_1(x, normal);
   return 1.0*(gradient*normal);
}

/// `f` for the Poisson problem (-nabla^2 u = f).
real_t rhs_fun_circle(const Vector &x)
{
   return 1;
}

real_t rhs_fun_xy_exponent(const Vector &x)
{
   real_t xy_p = 2.; // exponent for level set 2 where u = x^p+y^p;
   real_t coeff = std::max(xy_p*(xy_p-1), real_t(1));
   real_t expon = std::max(real_t(0), xy_p-2);
   if (xy_p == 1)
   {
      return 0.;
   }
   else
   {
      return -coeff*std::pow(x(0), expon) - coeff*std::pow(x(1), expon);
   }
}

real_t rhs_fun_xy_sinusoidal(const Vector &x)
{
   return std::sin(M_PI*x(0)*x(1))*(x(0)*x(0)+x(1)*x(1));
}
