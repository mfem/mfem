// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_COEFFICIENT
#define MFEM_COEFFICIENT

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "eltrans.hpp"

namespace mfem
{

class Mesh;

#ifdef MFEM_USE_MPI
class ParMesh;
#endif


/// Base class Coefficient that may optionally depend on time.
class Coefficient
{
protected:
   double time;

public:
   Coefficient() { time = 0.; }

   void SetTime(double t) { time = t; }
   double GetTime() { return time; }

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) = 0;

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip, double t)
   {
      SetTime(t);
      return Eval(T, ip);
   }

   virtual ~Coefficient() { }
};


/// Subclass constant coefficient.
class ConstantCoefficient : public Coefficient
{
public:
   double constant;

   /// c is value of constant function
   explicit ConstantCoefficient(double c = 1.0) { constant=c; }

   /// Evaluate the coefficient
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   { return (constant); }
};

/// class for piecewise constant coefficient
class PWConstCoefficient : public Coefficient
{
private:
   Vector constants;

public:

   /// Constructs a piecewise constant coefficient in NumOfSubD subdomains
   explicit PWConstCoefficient(int NumOfSubD = 0) : constants(NumOfSubD)
   { constants = 0.0; }

   /** c should be a vector defined by attributes, so for region with
       attribute i  c[i] is the coefficient in that region */
   PWConstCoefficient(Vector &c)
   { constants.SetSize(c.Size()); constants=c; }

   /// Member function to access or modify the value of the i-th constant
   double &operator()(int i) { return constants(i-1); }

   /// Set domain constants equal to the same constant c
   void operator=(double c) { constants = c; }

   /// Returns the number of constants
   int GetNConst() { return constants.Size(); }

   /// Evaluate the coefficient function
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

/// class for C-function coefficient
class FunctionCoefficient : public Coefficient
{
protected:
   double (*Function)(const Vector &);
   double (*TDFunction)(const Vector &, double);

public:
   /// Define a time-independent coefficient from a C-function
   FunctionCoefficient(double (*f)(const Vector &))
   {
      Function = f;
      TDFunction = NULL;
   }

   /// Define a time-dependent coefficient from a C-function
   FunctionCoefficient(double (*tdf)(const Vector &, double))
   {
      Function = NULL;
      TDFunction = tdf;
   }

   /// (DEPRECATED) Define a time-independent coefficient from a C-function
   FunctionCoefficient(double (*f)(Vector &))
   {
      Function = reinterpret_cast<double(*)(const Vector&)>(f);
      TDFunction = NULL;
   }

   /// (DEPRECATED) Define a time-dependent coefficient from a C-function
   FunctionCoefficient(double (*tdf)(Vector &, double))
   {
      Function = NULL;
      TDFunction = reinterpret_cast<double(*)(const Vector&,double)>(tdf);
   }

   /// Evaluate coefficient
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

class GridFunction;

/// Coefficient defined by a GridFunction. This coefficient is mesh dependent.
class GridFunctionCoefficient : public Coefficient
{
private:
   GridFunction *GridF;
   int Component;

public:
   /** Construct GridFunctionCoefficient from a given GridFunction
       and optionally which component to use if it is a vector
       gridfunction. */
   GridFunctionCoefficient (GridFunction *gf, int comp = 1)
   { GridF = gf; Component = comp; }

   void SetGridFunction(GridFunction *gf) { GridF = gf; }

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

class TransformedCoefficient : public Coefficient
{
private:
   Coefficient * Q1;
   Coefficient * Q2;
   double (*Transform1)(double);
   double (*Transform2)(double,double);

public:
   TransformedCoefficient (Coefficient * q,double (*F)(double))
      : Q1(q), Transform1(F) { Q2 = 0; Transform2 = 0; }
   TransformedCoefficient (Coefficient * q1,Coefficient * q2,
                           double (*F)(double,double))
      : Q1(q1), Q2(q2), Transform2(F) { Transform1 = 0; }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

/// Delta function coefficient
class DeltaCoefficient : public Coefficient
{
private:
   double center[3], scale, tol;
   Coefficient *weight;

public:
   DeltaCoefficient();
   DeltaCoefficient(double x, double y, double s)
   {
      center[0] = x; center[1] = y; center[2] = 0.; scale = s; tol = 1e-12;
      weight = NULL;
   }
   DeltaCoefficient(double x, double y, double z, double s)
   {
      center[0] = x; center[1] = y; center[2] = z; scale = s; tol = 1e-12;
      weight = NULL;
   }
   void SetTol(double _tol) { tol = _tol; }
   void SetWeight(Coefficient *w) { weight = w; }
   const double *Center() { return center; }
   double Scale() { return scale; }
   double Tol() { return tol; }
   Coefficient *Weight() { return weight; }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   { mfem_error("DeltaCoefficient::Eval"); return 0.; }
   virtual ~DeltaCoefficient() { delete weight; }
};

/// Coefficient defined on a subset of domain or boundary attributes
class RestrictedCoefficient : public Coefficient
{
private:
   Coefficient *c;
   Array<int> active_attr;

public:
   RestrictedCoefficient(Coefficient &_c, Array<int> &attr)
   { c = &_c; attr.Copy(active_attr); }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   { return active_attr[T.Attribute-1] ? c->Eval(T, ip, GetTime()) : 0.0; }
};

class VectorCoefficient
{
protected:
   int vdim;
   double time;

public:
   VectorCoefficient(int vd) { vdim = vd; time = 0.; }

   void SetTime(double t) { time = t; }
   double GetTime() { return time; }

   /// Returns dimension of the vector.
   int GetVDim() { return vdim; }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip) = 0;

   // General implementation using the Eval method for one IntegrationPoint.
   // Can be overloaded for more efficient implementation.
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);

   virtual ~VectorCoefficient() { }
};

class VectorConstantCoefficient : public VectorCoefficient
{
private:
   Vector vec;
public:
   VectorConstantCoefficient(const Vector &v)
      : VectorCoefficient(v.Size()), vec(v) { }
   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip) { V = vec; }
};

class VectorFunctionCoefficient : public VectorCoefficient
{
private:
   void (*Function)(const Vector &, Vector &);
   void (*TDFunction)(const Vector &, double, Vector &);
   Coefficient *Q;

public:
   /// Construct a time-independent vector coefficient from a C-function
   VectorFunctionCoefficient(int dim, void (*F)(const Vector &, Vector &),
                             Coefficient *q = NULL)
      : VectorCoefficient(dim), Q(q)
   {
      Function = F;
      TDFunction = NULL;
   }

   /// Construct a time-dependent vector coefficient from a C-function
   VectorFunctionCoefficient(int dim,
                             void (*TDF)(const Vector &, double, Vector &),
                             Coefficient *q = NULL)
      : VectorCoefficient(dim), Q(q)
   {
      Function = NULL;
      TDFunction = TDF;
   }

   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~VectorFunctionCoefficient() { }
};

/// Vector coefficient defined by an array of scalar coefficients.
class VectorArrayCoefficient : public VectorCoefficient
{
private:
   Array<Coefficient*> Coeff;

public:
   /// Construct vector of dim coefficients.
   explicit VectorArrayCoefficient(int dim);

   /// Returns i'th coefficient.
   Coefficient &GetCoeff(int i) { return *Coeff[i]; }

   Coefficient **GetCoeffs() { return Coeff; }

   /// Sets coefficient in the vector.
   void Set(int i, Coefficient *c) { Coeff[i] = c; }

   /// Evaluates i'th component of the vector.
   double Eval(int i, ElementTransformation &T, IntegrationPoint &ip)
   { return Coeff[i]->Eval(T, ip, GetTime()); }

   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   /// Destroys vector coefficient.
   virtual ~VectorArrayCoefficient();
};

/// Vector coefficient defined by a vector GridFunction
class VectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   GridFunction *GridFunc;

public:
   VectorGridFunctionCoefficient(GridFunction *gf);

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);

   virtual ~VectorGridFunctionCoefficient() { }
};

/// VectorCoefficient defined on a subset of domain or boundary attributes
class VectorRestrictedCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient *c;
   Array<int> active_attr;

public:
   VectorRestrictedCoefficient(VectorCoefficient &vc, Array<int> &attr)
      : VectorCoefficient(vc.GetVDim())
   { c = &vc; attr.Copy(active_attr); }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);
};


class MatrixCoefficient
{
protected:
   int vdim;
   double time;

public:
   explicit MatrixCoefficient(int dim) { vdim = dim; time = 0.; }

   void SetTime(double t) { time = t; }
   double GetTime() { return time; }

   int GetVDim() { return vdim; }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip) = 0;

   virtual ~MatrixCoefficient() { }
};

class MatrixFunctionCoefficient : public MatrixCoefficient
{
private:
   void (*Function)(const Vector &, DenseMatrix &);
   void (*TDFunction)(const Vector &, double, DenseMatrix &);

public:
   /// Construct a time-independent matrix coefficient from a C-function
   MatrixFunctionCoefficient(int dim, void (*F)(const Vector &, DenseMatrix &))
      : MatrixCoefficient(dim)
   {
      Function = F;
      TDFunction = NULL;
   }

   /// Construct a time-dependent matrix coefficient from a C-function
   MatrixFunctionCoefficient(int dim,
                             void (*TDF)(const Vector &, double, DenseMatrix &))
      : MatrixCoefficient(dim)
   {
      Function = NULL;
      TDFunction = TDF;
   }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~MatrixFunctionCoefficient() { }
};

class MatrixArrayCoefficient : public MatrixCoefficient
{
private:
   Array<Coefficient *> Coeff;

public:

   explicit MatrixArrayCoefficient (int dim);

   Coefficient &GetCoeff (int i, int j) { return *Coeff[i*vdim+j]; }

   void Set(int i, int j, Coefficient * c) { Coeff[i*vdim+j] = c; }

   double Eval(int i, int j, ElementTransformation &T, IntegrationPoint &ip)
   { return Coeff[i*vdim+j] -> Eval(T, ip, GetTime()); }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~MatrixArrayCoefficient();
};

/** Compute the Lp norm of a function f.
    \f$ \| f \|_{Lp} = ( \int_\Omega | f |^p d\Omega)^{1/p} \f$ */
double ComputeLpNorm(double p, Coefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[]);

/** Compute the Lp norm of a vector function f = {f_i}_i=1...N.
    \f$ \| f \|_{Lp} = ( \sum_i \| f_i \|_{Lp}^p )^{1/p} \f$ */
double ComputeLpNorm(double p, VectorCoefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[]);

#ifdef MFEM_USE_MPI
/** Compute the global Lp norm of a function f.
    \f$ \| f \|_{Lp} = ( \int_\Omega | f |^p d\Omega)^{1/p} \f$ */
double ComputeGlobalLpNorm(double p, Coefficient &coeff, ParMesh &pmesh,
                           const IntegrationRule *irs[]);

/** Compute the global Lp norm of a vector function f = {f_i}_i=1...N.
    \f$ \| f \|_{Lp} = ( \sum_i \| f_i \|_{Lp}^p )^{1/p} \f$ */
double ComputeGlobalLpNorm(double p, VectorCoefficient &coeff, ParMesh &pmesh,
                           const IntegrationRule *irs[]);
#endif

}

#endif
