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
      else { return 0.; }
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
        double yv1 = x(1);
        double val = 0.0 + (yv1-0.1)/(0.9-0.1);
        return val;
    }
    else if (type == 4 ){
        return 0.5/(M_PI*M_PI)*std::sin(M_PI*x(0))*std::sin(M_PI*x(1));
    }
    else {
        MFEM_ABORT(" Function type not implement yet.");
    }
    return 0.;
}


