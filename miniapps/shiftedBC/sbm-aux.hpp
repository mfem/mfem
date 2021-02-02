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
    if (type == 1) { // circle of radius 0.2 - centered at 0.5, 0.5
        double dx = x(0) - 0.5,
           dy = x(1) - 0.5,
           rv = dx*dx + dy*dy;
        rv = rv > 0 ? pow(rv, 0.5) : 0;
        double dist0 = rv - ring_radius; // +ve is the domain
        return dist0;
    }
    else if (type == 2) { // square hole of size 0.2 at the center of domain
        double dx1 = 0.4 - x(0),
               dx2 = x(0) - 0.6;
        double dy1 =  0.4 - x(1),
               dy2 = x(1) - 0.6;

        double dx = std::max(dx1, dx2);
               dx = std::max(dx, 0.);
       double dy = std::max(dy1, dy2);
              dy = std::max(dy, 0.);


       double dist = std::pow(dx, 2.) + std::pow(dy, 2.);

       if (dist > 0.) { dist = std::pow(dist, 0.5); }

        if (x(0) > 0.4 && x(0) < 0.6 && x(1) > 0.4 && x(1) < 0.6) {
            dx1 = x(0) - 0.4;
            dx2 = 0.6 - x(0);
            dx = std::min(dx1, dx2);

            dy1 = x(1) - 0.4;
            dy2 = 0.6 - x(1);
            dy = std::min(dy1, dy2);
            dist = std::min(dx, dy);
            dist = -dist;
        }

        return dist;
    }
    else if (type == 3) { // circle of radius 0.2 at 0.5,0.5
        double dx = x(0) - 0.5,
           dy = x(1) - 0.5,
           rv = dx*dx + dy*dy;
        rv = rv > 0 ? pow(rv, 0.5) : 0;
        double dist0 = rv - ring_radius; // +ve is the domain
        return dist0;
    }
    else if (type == 4) { // walls at y=0.0 and y=1.0
        if (x(1) > 0.5) {
            return 1. - x(1);
        }
        else {
            return x(1);
        }
    }
    else if (type == 5) { //bicuspid
        double a = 4*x(0)-2,
               b = 4*x(1)-2;
        return -pow(b*b-1, 2.)+(a+1)*pow(1-a, 3.);
    }
    else {
        MFEM_ABORT(" Function type not implement yet.");
    }
    return 0.;
}

///returns distance from 0 level set. If dist +ve, point is
/// inside the domain, otherwise outside.
class Dist_Value_Coefficient : public Coefficient
{
private:
   int type;

public:
   Dist_Value_Coefficient(int type_)
      : Coefficient(), type(type_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);
      return dist_value(x, type);
   }
};

/// Level set coefficient - 0 at the boundary, 1 otherwise
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

class Dist_Vector_Coefficient : public VectorCoefficient
{
private:
   int type;

public:
   Dist_Vector_Coefficient(int dim_, int type_)
      : VectorCoefficient(dim_), type(type_) { }

   virtual void Eval(Vector &p, ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);
      p.SetSize(x.Size());
      if (type == 1 || type == 3) {
          double dist0 = dist_value(x, type);
          double theta = std::atan2(x(1)-0.5, x(0)-0.5);
          p(0) = -dist0*std::cos(theta);
          p(1) = -dist0*std::sin(theta);
      }
      else if (type == 4) {
          double dist0 = dist_value(x, type);
          p(0) = 0.;
          if (x(1) > 1. || x(1) < 0.5) {
              p(1) = -dist0;
          }
          else {
              p(1) = dist0;
          }
      }
      else if (type == 2) {
          double dist0 = dist_value(x, type);
          if (x(0) >= 0.6 && x(1) >= 0.4 && x(1) <= 0.6) {
              p(0) = 0.6 - x(0);
              p(1) = 0;
          }
          else if (x(0) >= 0.6 && x(1) >= 0.6) {
              p(0) = 0.6 - x(0);
              p(1) = 0.6 - x(1);
          }
          else if (x(0) >= 0.4 && x(0) <= 0.6 && x(1) >= 0.6) {
              p(0) = 0.;
              p(1) = 0.6 - x(1);
          }
          else if (x(0) <= 0.4 && x(1) >= 0.6) {
              p(0) = 0.4 - x(0);
              p(1) = 0.6 - x(1);
          }
          else if (x(0) <= 0.4 && x(1) >= 0.4 && x(1) <= 0.6) {
              p(0) = 0.4 - x(0);
              p(1) = 0;
          }
          else if (x(0) <= 0.4 && x(1) <= 0.4) {
              p(0) = 0.4 - x(0);
              p(1) = 0.4 - x(1);
          }
          else if (x(0) >= 0.4 && x(0) <= 0.6 && x(1) <= 0.4) {
              p(0) = 0.;
              p(1) = 0.4 - x(1);
          }
          else if (x(0) >= 0.6 && x(1) <= 0.4) {
              p(0) = 0.6 - x(0);
              p(1) = 0.4 - x(1);
          }
          else if (x(0) >= 0.4 && x(0) <= 0.6 && x(1) >= 0.4 && x(1) <= 0.6 ) {
              double dx, dy;
              if (x(0) >= 0.5) { dx = 0.6 - x(0); }
              else { dx = x(0) - 0.4; }
              if (x(1) >= 0.5) { dy = 0.6 - x(1); }
              else { dy = x(1) - 0.4; }
              if (dx <= dy) {
                  p(0) = dx;
                  p(1) = 0;
              }
              else {
                  p(0) = 0;
                  p(1) = dy;
              }
          }
      }
   }
};

double dirichlet_velocity(const Vector &x, int type)
{
    if (type == 1) {
        return 0.;
    }
    else if (type == 2) {
        return 0.;
    }
    else if (type == 3) {
        double power = 2.;
        return pow(x(0), power) + pow(x(1), power);
    }
    else if (type == 4 ){
        //return 0.5/(M_PI*M_PI)*std::sin(M_PI*x(0))*std::sin(M_PI*x(1));
        return 1./(M_PI*M_PI)*std::sin(M_PI*x(0)*x(1)); //cross terms in solution
    }
    else if (type == 5 ){
        return 0.;
    }
    else {
        MFEM_ABORT(" Function type not implement yet.");
    }
    return 0.;
}

double rhs_fun_t(const Vector &x, int type)
{
    if (type == 3) {
        double power = 2;
        double coeff = std::max(power*(power-1), 1.);
        double expon = std::max(0., power-2);
        if (power == 1) {
            return 0.;
        }
        else {
            return -coeff*std::pow(x(0), expon) - coeff*std::pow(x(1), expon);
        }
    }
    else if (type == 4) {
        //    return std::sin(M_PI*x(0))*sin(M_PI*x(1)); //no cross terms
        return std::sin(M_PI*x(0)*x(1))*(x(0)*x(0)+x(1)*x(1));
    }
    else {
        return 1.;
    }
}

double dirichlet_velocity_init(const Vector &x, int type)
{
    double val = dirichlet_velocity(x, type);
    if (x(0) <= 0.00001 || x(0) >= 0.9999) {
        return val;
    }
    else {
        return 0.1*val+0.1;
    }
}

