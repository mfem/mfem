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
      return (radiusOfPt - radius)/fabs(radiusOfPt - radius); // positive is the domain
    }
  else if (type == 2) // sphere of radius 0.2 - centered at 0.5, 0.5
    {
      Vector center(3);
      center(0) = 0.5;
      center(1) = 0.5;
      center(2) = 0.5;
      double radiusOfPt = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0)+pow(x(2)-center(2),2.0),0.5);
      const double radius = 0.4;
      return (radiusOfPt - radius)/fabs(radiusOfPt - radius); // positive is the domain
    }
  else if (type == 4) // sphere of radius 0.2 - centered at 0.5, 0.5
    {
      Vector point(3);
      point(0) = 0.51;
      point(1) = 0.51;
      point(2) = 0.51;
      Vector normal(3);
      normal(0) = 0.0;
      normal(1) = 0.0;
      normal(2) = 1.0;
      double location = normal(2) * (x(2)-point(2));
      return -location/fabs(location); // positive is the domain
    }  
  else if (type == 3)
    {
      double pi = 3.141592653589793e0;
      
      double a = 0.5;
      double surface = sin((2*pi/a)*x(0))*cos((2*pi/a)*x(1))+sin((2*pi/a)*x(1))*cos((2*pi/a)*x(2))+sin((2*pi/a)*x(2))*cos((2*pi/a)*x(0));
      double sign = 0.0;
      if (surface != 0.0){
	sign = surface / fabs(surface);
      }
      //  return sign * mag;
      return sign;   
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
  double isIn = (pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0)-radius*radius) / std::fabs(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0)-radius*radius);
  if (isIn != 0.0){
    tN(0) = isIn * distX / normD;
    tN(1) = isIn * distY / normD;
  }
  else{
    tN(0) = (center(0) - x(0))/radius;
    tN(1) = (center(1) - x(1))/radius;
  }
}

// Distance to sphere of radius 0.2 - centered at 0.5, 0.5 
void Sphere_Dist(const Vector &x, Vector &D){
  double radius = 0.4;
  Vector center(3);
  center(0) = 0.5;
  center(1) = 0.5;
  center(2) = 0.5;
  double r = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0)+pow(x(2)-center(2),2.0),0.5);
  double distX = ((x(0)-center(0))/r)*(radius-r);
  double distY = ((x(1)-center(1))/r)*(radius-r);
  double distZ = ((x(2)-center(2))/r)*(radius-r);  
  D(0) = distX;
  D(1) = distY;
  D(2) = distZ;
}

// Unit normal of sphere of radius 0.2 - centered at 0.5, 0.5
void Sphere_Normal(const Vector &x, Vector &tN){
  double radius = 0.4;
  Vector center(3);
  center(0) = 0.5;
  center(1) = 0.5;
  center(2) = 0.5;
  double r = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0)+pow(x(2)-center(2),2.0),0.5);
  double distX = ((x(0)-center(0))/r)*(radius-r);
  double distY = ((x(1)-center(1))/r)*(radius-r);
  double distZ = ((x(2)-center(2))/r)*(radius-r);  

  double normD = sqrt(distX * distX + distY * distY + distZ * distZ);
  
  double isIn = (pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0)+pow(x(2)-center(2),2.0)-radius*radius) / std::fabs(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0)+pow(x(2)-center(2),2.0)-radius*radius);
  if (isIn != 0.0){
    tN(0) = isIn * distX / normD;
    tN(1) = isIn * distY / normD;
    tN(2) = isIn * distZ / normD;
  }
  else{
    tN(0) = (center(0) - x(0))/radius;
    tN(1) = (center(1) - x(1))/radius;
    tN(2) = (center(2) - x(2))/radius;
  }
}

// Distance to circle of radius 0.2 - centered at 0.5, 0.5 
void Plane_Dist(const Vector &x, Vector &D){
  Vector point(3);
  point(0) = 0.51;
  point(1) = 0.51;
  point(2) = 0.51;
  
  D(0) = 0.0;
  D(1) = 0.0;
  D(2) = point(2) - x(2); 
}

// Unit normal of circle of radius 0.2 - centered at 0.5, 0.5
void Plane_Normal(const Vector &x, Vector &tN){
  Vector normal(3);
  normal(0) = 0.0;
  normal(1) = 0.0;
  normal(2) = 1.0;
  
  tN(0) = 0.0;
  tN(1) = 0.0;
  tN(2) = 1.0;
}

// Distance to sphere of radius 0.2 - centered at 0.5, 0.5 
void Gyroid_Dist(const Vector &x, Vector &D){
  double pi = 3.141592653589793e0;

  double a = 0.5;
  DenseMatrix jacobian(4);
  jacobian = 0.0;
  double CtrialNp1_x = 0.0;
  double CtrialNp1_y = 0.0;
  double CtrialNp1_z = 0.0;
  double CtrialNp1_lambda = 0.0;
  int iter = 0;
  
  double residual_x = 0.0;
  double residual_y = 0.0;
  double residual_z = 0.0;
  double residual_lambda = 0.0;
  double err = 1;
  
  double CtrialN_x = x(0);
  double CtrialN_y = x(1);
  double CtrialN_z = x(2);
  double CtrialN_lambda = 0.0;
  
  while ((iter < 1000) && (err > 1e-12)){
    jacobian = 0.0;
 
    jacobian(0,0) = 2 + CtrialN_lambda * pow(2*pi/a,2.0) * (sin((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y)+sin((2*pi/a)*CtrialN_z)*cos((2*pi/a)*CtrialN_x));
    jacobian(0,1) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_x)*sin((2*pi/a)*CtrialN_y);
    jacobian(0,2) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x);
    jacobian(0,3) = -1.0 * (2*pi/a) * (cos((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y) - sin((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x));

    jacobian(1,0) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_x)*sin((2*pi/a)*CtrialN_y);
    jacobian(1,1) = 2 + CtrialN_lambda * pow(2*pi/a,2.0) * (sin((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y) + sin((2*pi/a)*CtrialN_y)*cos((2*pi/a)*CtrialN_z));
    jacobian(1,2) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_y)*sin((2*pi/a)*CtrialN_z);
    jacobian(1,3) = (2*pi/a) * (sin((2*pi/a)*CtrialN_x)*sin((2*pi/a)*CtrialN_y) - cos((2*pi/a)*CtrialN_y)*cos((2*pi/a)*CtrialN_z));
    
    jacobian(2,0) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x);
    jacobian(2,1) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_y)*sin((2*pi/a)*CtrialN_z);
    jacobian(2,2) = 2 + CtrialN_lambda * pow(2*pi/a,2.0) * (sin((2*pi/a)*CtrialN_y)*cos((2*pi/a)*CtrialN_z) + sin((2*pi/a)*CtrialN_z)*cos((2*pi/a)*CtrialN_x));
    jacobian(2,3) = (2*pi/a) * (sin((2*pi/a)*CtrialN_y)*sin((2*pi/a)*CtrialN_z) - cos((2*pi/a)*CtrialN_z)*cos((2*pi/a)*CtrialN_x));
      
    jacobian(3,0) = -(2*pi/a) * (cos((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y) - sin((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x));
    jacobian(3,1) = -(2*pi/a) * (-sin((2*pi/a)*CtrialN_x)*sin((2*pi/a)*CtrialN_y) + cos((2*pi/a)*CtrialN_y)*cos((2*pi/a)*CtrialN_z));
    jacobian(3,2) = -(2*pi/a) * (-sin((2*pi/a)*CtrialN_y)*sin((2*pi/a)*CtrialN_z) + cos((2*pi/a)*CtrialN_z)*cos((2*pi/a)*CtrialN_x));
    jacobian(3,3) = 0.0;      
    
    jacobian.Invert();
    //    
    residual_x = -2*(CtrialN_x-x(0))+CtrialN_lambda*(2*pi/a)*(cos((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y)-sin((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x));
    residual_y = -2*(CtrialN_y-x(1))+CtrialN_lambda*(2*pi/a)*( -sin((2*pi/a)*CtrialN_x) * sin((2*pi/a)*CtrialN_y) + cos((2*pi/a)*CtrialN_y) * cos((2*pi/a)*CtrialN_z));
    residual_z = -2*(CtrialN_z-x(2))+CtrialN_lambda*(2*pi/a)*( -sin((2*pi/a)*CtrialN_y) * sin((2*pi/a)*CtrialN_z) + cos((2*pi/a)*CtrialN_z) * cos((2*pi/a)*CtrialN_x));
    residual_lambda = sin((2*pi/a)*CtrialN_x) * cos((2*pi/a)*CtrialN_y) + sin((2*pi/a)*CtrialN_y) * cos((2*pi/a)*CtrialN_z) + sin((2*pi/a)*CtrialN_z) * cos((2*pi/a)*CtrialN_x);

    
    CtrialNp1_x = CtrialN_x + jacobian(0,0)*residual_x + jacobian(0,1)*residual_y + jacobian(0,2)*residual_z +jacobian(0,3)*residual_lambda;
    CtrialNp1_y = CtrialN_y + jacobian(1,0)*residual_x + jacobian(1,1)*residual_y + jacobian(1,2)*residual_z +jacobian(1,3)*residual_lambda;
    CtrialNp1_z = CtrialN_z + jacobian(2,0)*residual_x + jacobian(2,1)*residual_y + jacobian(2,2)*residual_z +jacobian(2,3)*residual_lambda;
    CtrialNp1_lambda = CtrialN_lambda + jacobian(3,0)*residual_x + jacobian(3,1)*residual_y + jacobian(3,2)*residual_z +jacobian(3,3)*residual_lambda;

    err = std::pow(std::pow((CtrialNp1_x - CtrialN_x),2) +std::pow((CtrialNp1_y - CtrialN_y),2) +std::pow((CtrialNp1_z - CtrialN_z),2) + std::pow((CtrialNp1_lambda - CtrialN_lambda),2),0.5);
   
    CtrialN_x = CtrialNp1_x;
    CtrialN_y = CtrialNp1_y;
    CtrialN_z = CtrialNp1_z;
    CtrialN_lambda = CtrialNp1_lambda;
    iter++;
  }
  if (iter == 1000){
    std::cout << " shit intersect " << std::endl;
  }
  
  double distX = CtrialNp1_x - x(0);
  double distY = CtrialNp1_y - x(1);
  double distZ = CtrialNp1_z - x(2);  
  D(0) = distX;
  D(1) = distY;
  D(2) = distZ;
}

// Unit normal of sphere of radius 0.2 - centered at 0.5, 0.5
void Gyroid_Normal(const Vector &x, Vector &tN){
  double pi = 3.141592653589793e0;

  double a = 0.5;
  DenseMatrix jacobian(4);
  jacobian = 0.0;
  double CtrialNp1_x = 0.0;
  double CtrialNp1_y = 0.0;
  double CtrialNp1_z = 0.0;
  double CtrialNp1_lambda = 0.0;
  int iter = 0;
  
  double residual_x = 0.0;
  double residual_y = 0.0;
  double residual_z = 0.0;
  double residual_lambda = 0.0;
  double err = 1;
  
  double CtrialN_x = x(0);
  double CtrialN_y = x(1);
  double CtrialN_z = x(2);
  double CtrialN_lambda = 0.0;
  
  while ((iter < 1000) && (err > 1e-12)){
    jacobian  = 0.0;
    jacobian(0,0) = 2 + CtrialN_lambda * pow(2*pi/a,2.0) * (sin((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y)+sin((2*pi/a)*CtrialN_z)*cos((2*pi/a)*CtrialN_x));
    jacobian(0,1) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_x)*sin((2*pi/a)*CtrialN_y);
    jacobian(0,2) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x);
    jacobian(0,3) = -1.0 * (2*pi/a) * (cos((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y) - sin((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x));

    jacobian(1,0) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_x)*sin((2*pi/a)*CtrialN_y);
    jacobian(1,1) = 2 + CtrialN_lambda * pow(2*pi/a,2.0) * (sin((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y) + sin((2*pi/a)*CtrialN_y)*cos((2*pi/a)*CtrialN_z));
    jacobian(1,2) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_y)*sin((2*pi/a)*CtrialN_z);
    jacobian(1,3) = (2*pi/a) * (sin((2*pi/a)*CtrialN_x)*sin((2*pi/a)*CtrialN_y) - cos((2*pi/a)*CtrialN_y)*cos((2*pi/a)*CtrialN_z));
    
    jacobian(2,0) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x);
    jacobian(2,1) = CtrialN_lambda * pow(2*pi/a,2.0) * cos((2*pi/a)*CtrialN_y)*sin((2*pi/a)*CtrialN_z);
    jacobian(2,2) = 2 + CtrialN_lambda * pow(2*pi/a,2.0) * (sin((2*pi/a)*CtrialN_y)*cos((2*pi/a)*CtrialN_z) + sin((2*pi/a)*CtrialN_z)*cos((2*pi/a)*CtrialN_x));
    jacobian(2,3) = (2*pi/a) * (sin((2*pi/a)*CtrialN_y)*sin((2*pi/a)*CtrialN_z) - cos((2*pi/a)*CtrialN_z)*cos((2*pi/a)*CtrialN_x));
      
    jacobian(3,0) = -(2*pi/a) * (cos((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y) - sin((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x));
    jacobian(3,1) = -(2*pi/a) * (-sin((2*pi/a)*CtrialN_x)*sin((2*pi/a)*CtrialN_y) + cos((2*pi/a)*CtrialN_y)*cos((2*pi/a)*CtrialN_z));
    jacobian(3,2) = -(2*pi/a) * (-sin((2*pi/a)*CtrialN_y)*sin((2*pi/a)*CtrialN_z) + cos((2*pi/a)*CtrialN_z)*cos((2*pi/a)*CtrialN_x));
    jacobian(3,3) = 0.0;      
    
    jacobian.Invert();
    //    
    residual_x = -2*(CtrialN_x-x(0))+CtrialN_lambda*(2*pi/a)*(cos((2*pi/a)*CtrialN_x)*cos((2*pi/a)*CtrialN_y)-sin((2*pi/a)*CtrialN_z)*sin((2*pi/a)*CtrialN_x));
    residual_y = -2*(CtrialN_y-x(1))+CtrialN_lambda*(2*pi/a)*( -sin((2*pi/a)*CtrialN_x) * sin((2*pi/a)*CtrialN_y) + cos((2*pi/a)*CtrialN_y) * cos((2*pi/a)*CtrialN_z));
    residual_z = -2*(CtrialN_z-x(2))+CtrialN_lambda*(2*pi/a)*( -sin((2*pi/a)*CtrialN_y) * sin((2*pi/a)*CtrialN_z) + cos((2*pi/a)*CtrialN_z) * cos((2*pi/a)*CtrialN_x));
    residual_lambda = sin((2*pi/a)*CtrialN_x) * cos((2*pi/a)*CtrialN_y) + sin((2*pi/a)*CtrialN_y) * cos((2*pi/a)*CtrialN_z) + sin((2*pi/a)*CtrialN_z) * cos((2*pi/a)*CtrialN_x);

    
    CtrialNp1_x = CtrialN_x + jacobian(0,0)*residual_x + jacobian(0,1)*residual_y + jacobian(0,2)*residual_z +jacobian(0,3)*residual_lambda;
    CtrialNp1_y = CtrialN_y + jacobian(1,0)*residual_x + jacobian(1,1)*residual_y + jacobian(1,2)*residual_z +jacobian(1,3)*residual_lambda;
    CtrialNp1_z = CtrialN_z + jacobian(2,0)*residual_x + jacobian(2,1)*residual_y + jacobian(2,2)*residual_z +jacobian(2,3)*residual_lambda;
    CtrialNp1_lambda = CtrialN_lambda + jacobian(3,0)*residual_x + jacobian(3,1)*residual_y + jacobian(3,2)*residual_z +jacobian(3,3)*residual_lambda;

    err = std::pow(std::pow((CtrialNp1_x - CtrialN_x),2) +std::pow((CtrialNp1_y - CtrialN_y),2) +std::pow((CtrialNp1_z - CtrialN_z),2) + std::pow((CtrialNp1_lambda - CtrialN_lambda),2),0.5);
    CtrialN_x = CtrialNp1_x;
    CtrialN_y = CtrialNp1_y;
    CtrialN_z = CtrialNp1_z;
    CtrialN_lambda = CtrialNp1_lambda;
    iter++;
  }
  if (iter == 1000){
    std::cout << " shit intersect " << std::endl;
  }
  double normD = std::pow(std::pow(CtrialNp1_x - x(0),2) + std::pow(CtrialNp1_y - x(1),2) + std::pow(CtrialNp1_z - x(2),2),0.5);

  double distX = CtrialNp1_x - x(0);
  double distY = CtrialNp1_y - x(1);
  double distZ = CtrialNp1_z - x(2);

  double surface = sin((2*pi/a)*x(0))*cos((2*pi/a)*x(1))+sin((2*pi/a)*x(1))*cos((2*pi/a)*x(2))+sin((2*pi/a)*x(2))*cos((2*pi/a)*x(0));
  double sign = surface / fabs(surface);

  tN(0) = sign * distX / normD;
  tN(1) = sign * distY / normD;
  tN(2) = sign * distZ / normD;
  
}

/// Analytic distance to the 0 level set.
void dist_value(const Vector &x, Vector &D, const int type)
{
  if (type == 1) {
    return Circle_Dist(x, D);
  }
  else if (type == 2) {
    return Sphere_Dist(x, D);
  }
  else if (type == 3) {
    return Gyroid_Dist(x, D);
  }
  else if (type == 4) {
    return Plane_Dist(x, D);
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
  else if (type == 2) {
    return Sphere_Normal(x, tN);
  }
  else if (type == 3) {
    return Gyroid_Dist(x, tN);
  }
  else if (type == 4) {
    return Plane_Dist(x, tN);
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
    return (dist >= 0.0) ? -1.0 : 1.0;
  }
};

/// Level set coefficient: +1 inside the true domain, -1 outside.
class Dist_Coefficient : public Coefficient
{
private:
  int type;

public:
  Dist_Coefficient(int type_)
    : Coefficient(), type(type_) { }

  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
  {
    Vector x(3);
    T.Transform(ip, x);
    double dist = relativePosition(x, type);
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
