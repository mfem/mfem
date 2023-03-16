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

#include "sbm_aux.hpp"

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
      return location/fabs(location); // positive is the domain
    }  
  else if (type == 3)
    {
      double pi = 3.141592653589793e0;
      
      double a = 0.5;
      double surface = sin((2*pi/a)*x(0))*cos((2*pi/a)*x(1))+sin((2*pi/a)*x(1))*cos((2*pi/a)*x(2))+sin((2*pi/a)*x(2))*cos((2*pi/a)*x(0));
      double sign = 1.0;
      if ( std::abs(surface) >= 1e-10){
	sign = surface / fabs(surface);
      }
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

  void WeightedDiffusionIntegrator::AssembleElementMatrix(const FiniteElement &el,
							   ElementTransformation &Trans,
							   DenseMatrix &elmat)
  {
    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    int NEproc = pmesh->GetNE();

    const int dim = el.GetDim();
    int dof = el.GetDof();

    elmat.SetSize(dof);
    elmat = 0.0;
    
    DenseMatrix dshape(dof,dim), dshape_ps(dof,dim), adjJ(dim);
    dshape = 0.0;
    dshape_ps = 0.0;
 
    adjJ = 0.0;
    
    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);
    //    std::cout << " begin " << " ppid " << myid << " elemNo " << Trans.ElementNo << std::endl;
    for (int q = 0; q < ir->GetNPoints(); q++)
      {
	const IntegrationPoint &ip = ir->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Trans.SetIntPoint(&ip);

	CalcAdjugate(Trans.Jacobian(), adjJ);

	el.CalcDShape(ip, dshape);

	Mult(dshape, adjJ, dshape_ps);
	double w = ip.weight/Trans.Weight();
	double volumeFraction = alpha->GetValue(Trans, ip);
	double val = w*volumeFraction;

	AddMult_a_AAt(val, dshape_ps, elmat);
      }
    // std::cout << " end " << " ppid " << myid << " elemNo " << Trans.ElementNo << std::endl;
  }

  const IntegrationRule &WeightedDiffusionIntegrator::GetRule(
						       const FiniteElement &trial_fe,
						       const FiniteElement &test_fe,
						       ElementTransformation &Trans)
  {
    int order = Trans.OrderGrad(&trial_fe) /*+ test_fe.GetOrder() + Trans.OrderJ()*/;
    return IntRules.Get(trial_fe.GetGeomType(), 5*order);
  }

void DiffuseH1(ParGridFunction &g, double c)
{
  //  g = 0.0;
  ParFiniteElementSpace &pfes = *g.ParFESpace();
  auto check_h1 = dynamic_cast<const H1_FECollection *>(pfes.FEColl());
  MFEM_VERIFY(check_h1 && pfes.GetVDim() == 1,
	      "This solver supports only scalar H1 spaces.");
  // Compute average mesh size (assumes similar cells).
  ParMesh &pmesh = *pfes.GetParMesh();
  double dx, loc_area = 0.0;
  for (int i = 0; i < pmesh.GetNE(); i++)
    {
      loc_area += pmesh.GetElementVolume(i);
    }
  double glob_area;
  MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE,
		MPI_SUM, pmesh.GetComm());
  const int glob_zones = pmesh.GetGlobalNE();
  switch (pmesh.GetElementBaseGeometry(0))
    {
    case Geometry::SEGMENT:
      dx = glob_area / glob_zones; break;
    case Geometry::SQUARE:
      dx = sqrt(glob_area / glob_zones); break;
    case Geometry::TRIANGLE:
      dx = sqrt(2.0 * glob_area / glob_zones); break;
    case Geometry::CUBE:
      dx = pow(glob_area / glob_zones, 1.0/3.0); break;
    case Geometry::TETRAHEDRON:
      dx = pow(6.0 * glob_area / glob_zones, 1.0/3.0); break;
    default: MFEM_ABORT("Unknown zone type!"); dx = 0.0;
    }
  int myid;
  MPI_Comm_rank(pmesh.GetComm(), &myid);  
  // Set up RHS.
  ParLinearForm b(&pfes);
  GridFunctionCoefficient g_old_coeff(&g);
  b.AddDomainIntegrator(new DomainLFIntegrator(g_old_coeff));
  b.Assemble();
  
  // Diffusion and mass terms in the LHS.
  ParBilinearForm a_n(&pfes);
  a_n.AddDomainIntegrator(new MassIntegrator);
  L2_FECollection alpha_fec(0, pmesh.Dimension());
  ParFiniteElementSpace alpha_fes(&pmesh, &alpha_fec);
  ParGridFunction c_coeff(&alpha_fes);
  c_coeff = c * dx * dx;
  a_n.AddDomainIntegrator(new WeightedDiffusionIntegrator(&pmesh, c_coeff));
  
  // ConstantCoefficient c_coeff(c * dx * dx);
  // a_n.AddDomainIntegrator(new DiffusionIntegrator(c_coeff));
  
  a_n.Assemble();
  // Solver.
  CGSolver cg(pmesh.GetComm());
  cg.SetRelTol(1e-12);
  //  cg.SetAbsTol(0.0);   
  cg.SetMaxIter(100);
  cg.SetPrintLevel(-1);
  OperatorPtr A;
  Vector B, X;
  // Solve with Neumann BC.
  Array<int> ess_tdof_list;
  a_n.FormLinearSystem(ess_tdof_list, g, b, A, X, B);
 
  auto *prec = new HypreBoomerAMG;
  prec->SetPrintLevel(-1);
  cg.SetPreconditioner(*prec);
  cg.SetOperator(*A);
  //  X = 0.0;
  cg.Mult(B, X); 
  a_n.RecoverFEMSolution(X, b, g);

  delete prec;
}
