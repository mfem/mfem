//                       MFEM
//
// Compile with: make ex1p
//
// Sample runs:
// mpirun -np 1 ex1p -m ../../data/inline-quad.mesh  -rs 0 -vis -o 2

#include "../../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double dist_value(const Vector &x, const int type)
{
    double ring_radius = 0.2;
    if (type == 1 || type == 2) { // circle of radius 0.2 - centered at 0.5, 0.5
        double dx = x(0) - 0.5,
           dy = x(1) - 0.5,
           rv = dx*dx + dy*dy;
        if (x.Size() == 3) {
            double dz = x(2) - 0.5;
            rv += dz*dz;
        }
        rv = rv > 0 ? pow(rv, 0.5) : 0;
        return rv - ring_radius; // +ve is the domain
    }
    else if (type == 3) { // walls at y = 0.0 and y = 1.0
        if (x(1) > 0.5) {
            return 1. - x(1);
        }
        else {
            return x(1);
        }
    }
    else {
        MFEM_ABORT(" Function type not implement yet.");
    }
    return 0.;
}

/// Level set coefficient - 1 inside the domain, -1 outside, 0 at the boundary.
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
      if (type == 1 || type == 2) {
          double dist0 = dist_value(x, type);
          for (int i = 0; i < dim; i++) { p(i) = 0.5 - x(i); }
          double length = p.Norml2();
          p *= dist0/length;
      }
      else if (type == 3) {
          double dist0 = dist_value(x, type);
          p(0) = 0.;
          if (x(1) > 1. || x(1) < 0.5) {
              p(1) = -dist0;
          }
          else {
              p(1) = dist0;
          }
      }
   }
};

/// exponent for level set 2 where u = x^p+y^p;
#define xy_p 2.

/// Boundary conditions
double dirichlet_velocity_circle(const Vector &x)
{
    return 0.;
}

double dirichlet_velocity_xy_exponent(const Vector &x)
{
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
    double coeff = std::max(xy_p*(xy_p-1), 1.);
    double expon = std::max(0., xy_p-2);
    if (xy_p == 1) {
        return 0.;
    }
    else {
        return -coeff*std::pow(x(0), expon) - coeff*std::pow(x(1), expon);
    }
}

double rhs_fun_xy_sinusoidal(const Vector &x)
{
    return std::sin(M_PI*x(0)*x(1))*(x(0)*x(0)+x(1)*x(1));
}
