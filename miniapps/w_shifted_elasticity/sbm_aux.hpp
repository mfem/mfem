// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
double relativePosition(const Vector &x, const int type)
{
   if (type == 1) // circle of radius 0.2 - centered at 0.5, 0.5
   {
     Vector center(2);
     center(0) = 0.5;
     center(1) = 0.5;
     double radiusOfPt = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
     const double radius = 0.2;
     return radiusOfPt - radius; // positive is the domain
   }
   else
     {
      MFEM_ABORT(" Function type not implement yet.");
   }
   return 0.;
}

// Distance to circle of radius 0.2 - centered at 0.5, 0.5 
void Circle_Dist(const Vector &x, Vector &D){
  double radius = 0.2;
  Vector center(2);
  center(0) = 0.5;
  center(1) = 0.5;
  double r = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
  double distX = ((x(0)-center(0))/r)*(radius-r);
  double distY = ((x(1)-center(1))/r)*(radius-r);
  D(0) = distX;
  D(1) = distY;
}

// Unit normal of circle of radius 0.2 - centered at 0.5, 0.5
void Circle_Normal(const Vector &x, Vector &tN){
  double radius = 0.2;
  Vector center(2);
  center(0) = 0.5;
  center(1) = 0.5;
  
  double r = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
  double distX = ((x(0)-center(0))/r)*(radius-r);
  double distY = ((x(1)-center(1))/r)*(radius-r);
  double normD = sqrt(distX * distX + distY * distY);
  if (r < radius){
    tN(0) = -distX / normD;
    tN(1) = -distY / normD;
  }
  else if (r > radius){
    tN(0) = distX / normD;
    tN(1) = distY / normD;
  }
  else{
    tN(0) = (center(0) - x(0))/radius;
    tN(1) = (center(1) - x(1))/radius;
  }
}

/// Analytic distance to the 0 level set.
void dist_value(const Vector &x, Vector &D, const int type)
{
   if (type == 1) {
     return Circle_Dist(x, D);
   }
   else
   {
      MFEM_ABORT(" Function type not implement yet.");
   }
}

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
void normal_value(const Vector &x, Vector &tN, const int type)
{
   if (type == 1) {
     return Circle_Normal(x, tN);
   }
   else
   {
      MFEM_ABORT(" Function type not implement yet.");
   }
}

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
    dist_value(x, p, type);  
  }
};
  
  /// Normal vector to the zero level-set.
class Normal_Vector_Coefficient : public VectorCoefficient
{
private:
  int type;
  
public:
  Normal_Vector_Coefficient(int dim_, int type_)
    : VectorCoefficient(dim_), type(type_) { }
  
  using VectorCoefficient::Eval;
  
  virtual void Eval(Vector &p, ElementTransformation &T,
		    const IntegrationPoint &ip)
  {
    Vector x;
    T.Transform(ip, x);
    const int dim = x.Size();
    p.SetSize(dim);
    normal_value(x, p, type);  
  }
};

/// Level set coefficient: +1 inside the true domain, -1 outside.
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
      double dist = relativePosition(x, type);
      //      return (dist >= 0.0) ? 1.0 : -1.0;
      return dist;
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

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      MFEM_VERIFY(dls.Size() > 0,
                  "Add at least 1 Dist_level_Set_Coefficient to the Combo.");
      double dist = dls[0]->Eval(T, ip);
      for (int j = 1; j < dls.Size(); j++)
      {
         dist = min(dist, dls[j]->Eval(T, ip));
      }
      //      return (dist >= 0.0) ? 1.0 : -1.0;
      return dist;
   }
};
