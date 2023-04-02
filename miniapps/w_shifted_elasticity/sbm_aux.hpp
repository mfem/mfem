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
#ifndef SBM_AUX_HPP
#define SBM_AUX_HPP

#include "mfem.hpp"

using namespace std;
using namespace mfem;

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
double relativePosition(const Vector &x, const int type);

// Distance to circle of radius 0.2 - centered at 0.5, 0.5 
void Circle_Dist(const Vector &x, Vector &D);

// Unit normal of circle of radius 0.2 - centered at 0.5, 0.5
void Circle_Normal(const Vector &x, Vector &tN);

// Distance to sphere of radius 0.2 - centered at 0.5, 0.5 
void Sphere_Dist(const Vector &x, Vector &D);

// Unit normal of sphere of radius 0.2 - centered at 0.5, 0.5
void Sphere_Normal(const Vector &x, Vector &tN);

// Distance to circle of radius 0.2 - centered at 0.5, 0.5 
void Plane_Dist(const Vector &x, Vector &D);

// Unit normal of circle of radius 0.2 - centered at 0.5, 0.5
void Plane_Normal(const Vector &x, Vector &tN);

// Distance to sphere of radius 0.2 - centered at 0.5, 0.5 
void Gyroid_Dist(const Vector &x, Vector &D);

// Unit normal of sphere of radius 0.2 - centered at 0.5, 0.5
void Gyroid_Normal(const Vector &x, Vector &tN);

/// Analytic distance to the 0 level set.
void dist_value(const Vector &x, Vector &D, const int type);

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
void normal_value(const Vector &x, Vector &tN, const int type);

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
    return (dist >= 0.0) ? 1.0 : -1.0;
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

  class WeightedDiffusionIntegrator : public BilinearFormIntegrator
  {
  private:
    const ParMesh *pmesh;  
    ParGridFunction *alpha;
    
  public:
    WeightedDiffusionIntegrator(const ParMesh *pmesh, ParGridFunction &alphaF) : pmesh(pmesh), alpha(&alphaF) {}
    virtual void AssembleElementMatrix(const FiniteElement &el,
					ElementTransformation &Trans,
					DenseMatrix &elmat);
    const IntegrationRule &GetRule(const FiniteElement &trial_fe,
				   const FiniteElement &test_fe,
				   ElementTransformation &Trans);
  };

void DiffuseH1(ParGridFunction &g, double c);
#endif
