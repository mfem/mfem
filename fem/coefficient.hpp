// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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


/** @brief Base class Coefficients that optionally depend on space and time.
    These are used by the BilinearFormIntegrator, LinearFormIntegrator, and
    NonlinearFormIntegrator classes to represent the physical coefficients in
    the PDEs that are being discretized. */
class Coefficient
{
protected:
   double time;

public:
   Coefficient() { time = 0.; }

   /// Set the time for time dependent coefficients
   void SetTime(double t) { time = t; }

   /// Get the time for time dependent coefficients
   double GetTime() { return time; }

   /** @brief Evaluate the coefficient in the element described by @a T at the
       point @a ip. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) = 0;

   /** @brief Evaluate the coefficient in the element described by @a T at the
       point @a ip at time @a t. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip, double t)
   {
      SetTime(t);
      return Eval(T, ip);
   }

   virtual ~Coefficient() { }
};


/// A coefficient that is constant across space and time
class ConstantCoefficient : public Coefficient
{
public:
   double constant;

   /// c is value of constant function
   explicit ConstantCoefficient(double c = 1.0) { constant=c; }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   { return (constant); }
};

/** @brief A piecewise constant coefficient with the constants keyed
    off the element attribute numbers. */
class PWConstCoefficient : public Coefficient
{
private:
   Vector constants;

public:

   /// Constructs a piecewise constant coefficient in NumOfSubD subdomains
   explicit PWConstCoefficient(int NumOfSubD = 0) : constants(NumOfSubD)
   { constants = 0.0; }

   /// Construct the constant coefficient using a vector of constants.
   /** @a c should be a vector defined by attributes, so for region with
       attribute @a i @a c[i-1] is the coefficient in that region */
   PWConstCoefficient(Vector &c)
   { constants.SetSize(c.Size()); constants=c; }

   /// Update the constants with vector @a c.
   void UpdateConstants(Vector &c) { constants.SetSize(c.Size()); constants=c; }

   /// Return a reference to the i-th constant
   double &operator()(int i) { return constants(i-1); }

   /// Set the constants for all attributes to constant @a c.
   void operator=(double c) { constants = c; }

   /// Returns the number of constants representing different attributes.
   int GetNConst() { return constants.Size(); }

   /// Evaluate the coefficient.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};


/// A general C-function coefficient
class FunctionCoefficient : public Coefficient
{
protected:
   double (*Function)(const Vector &);
   double (*TDFunction)(const Vector &, double);

public:
   /// Define a time-independent coefficient from a pointer to a C-function
   FunctionCoefficient(double (*f)(const Vector &))
   {
      Function = f;
      TDFunction = NULL;
   }

   /// Define a time-dependent coefficient from a pointer to a C-function
   FunctionCoefficient(double (*tdf)(const Vector &, double))
   {
      Function = NULL;
      TDFunction = tdf;
   }

   /// (DEPRECATED) Define a time-independent coefficient from a C-function
   /** @deprecated Use the method where the C-function, @a f, uses a const
       Vector argument instead of Vector. */
   MFEM_DEPRECATED FunctionCoefficient(double (*f)(Vector &))
   {
      Function = reinterpret_cast<double(*)(const Vector&)>(f);
      TDFunction = NULL;
   }

   /// (DEPRECATED) Define a time-dependent coefficient from a C-function
   /** @deprecated Use the method where the C-function, @a tdf, uses a const
       Vector argument instead of Vector. */
   MFEM_DEPRECATED FunctionCoefficient(double (*tdf)(Vector &, double))
   {
      Function = NULL;
      TDFunction = reinterpret_cast<double(*)(const Vector&,double)>(tdf);
   }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

class GridFunction;

/// Coefficient defined by a GridFunction. This coefficient is mesh dependent.
class GridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *GridF;
   int Component;

public:
   GridFunctionCoefficient() : GridF(NULL), Component(1) { }
   /** Construct GridFunctionCoefficient from a given GridFunction, and
       optionally specify a component to use if it is a vector GridFunction. */
   GridFunctionCoefficient (const GridFunction *gf, int comp = 1)
   { GridF = gf; Component = comp; }

   /// Set the internal GridFunction
   void SetGridFunction(const GridFunction *gf) { GridF = gf; }

   /// Get the internal GridFunction
   const GridFunction * GetGridFunction() const { return GridF; }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};


/** @brief A coefficient that depends on 1 or 2 parent coefficients and a
    transformation rule represented by a C-function.

    \f$ C(x,t) = T(Q1(x,t)) \f$ or \f$ C(x,t) = T(Q1(x,t), Q2(x,t)) \f$

    where T is the transformation rule, and Q1/Q2 are the parent coefficients.*/
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

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

/** @brief Delta function coefficient optionally multiplied by a weight
    coefficient and a scaled time dependent C-function.

    \f$ F(x,t) = w(x,t) s T(t) d(x - xc) \f$

    where w is the optional weight coefficient, @a s is a scale factor
    T is an optional time-dependent function and d is a delta function.

    WARNING this cannot be used as a normal coefficient.  The usual Eval
    method is disabled. */
class DeltaCoefficient : public Coefficient
{
protected:
   double center[3], scale, tol;
   Coefficient *weight;
   int sdim;
   double (*tdf)(double);

public:

   /// Construct a unit delta function centered at (0.0,0.0,0.0)
   DeltaCoefficient()
   {
      center[0] = center[1] = center[2] = 0.; scale = 1.; tol = 1e-12;
      weight = NULL; sdim = 0; tdf = NULL;
   }

   /// Construct a delta function scaled by @a s and centered at (x,0.0,0.0)
   DeltaCoefficient(double x, double s)
   {
      center[0] = x; center[1] = 0.; center[2] = 0.; scale = s; tol = 1e-12;
      weight = NULL; sdim = 1; tdf = NULL;
   }

   /// Construct a delta function scaled by @a s and centered at (x,y,0.0)
   DeltaCoefficient(double x, double y, double s)
   {
      center[0] = x; center[1] = y; center[2] = 0.; scale = s; tol = 1e-12;
      weight = NULL; sdim = 2; tdf = NULL;
   }

   /// Construct a delta function scaled by @a s and centered at (x,y,z)
   DeltaCoefficient(double x, double y, double z, double s)
   {
      center[0] = x; center[1] = y; center[2] = z; scale = s; tol = 1e-12;
      weight = NULL; sdim = 3; tdf = NULL;
   }

   /// Set the center location of the delta function.
   void SetDeltaCenter(const Vector& center);

   /// Set the scale value multiplying the delta function.
   void SetScale(double _s) { scale = _s; }

   /// Set a time-dependent function that multiplies the Scale().
   void SetFunction(double (*f)(double)) { tdf = f; }

   /** @brief Set the tolerance used during projection onto GridFunction to
       identify the Mesh vertex where the Center() of the delta function
       lies. (default 1e-12)*/
   void SetTol(double _tol) { tol = _tol; }

   /// Set a weight Coefficient that multiplies the DeltaCoefficient.
   /** The weight Coefficient multiplies the value returned by EvalDelta() but
       not the value returned by Scale().
       The weight Coefficient is also used as the L2-weight function when
       projecting the DeltaCoefficient onto a GridFunction, so that the weighted
       integral of the projection is exactly equal to the Scale(). */
   void SetWeight(Coefficient *w) { weight = w; }

   /// Return a pointer to a c-array representing the center of the delta
   /// function.
   const double *Center() { return center; }

   /** @brief Return the scale factor times the optional time dependent
       function.  Returns \f$ s T(t) \f$ with \f$ T(t) = 1 \f$ when
       not set by the user. */
   double Scale() { return tdf ? (*tdf)(GetTime())*scale : scale; }

   /// Return the tolerance used to identify the mesh vertices
   double Tol() { return tol; }

   /// See SetWeight() for description of the weight Coefficient.
   Coefficient *Weight() { return weight; }

   /// Write the center of the delta function into @a center.
   void GetDeltaCenter(Vector& center);

   /// The value of the function assuming we are evaluating at the delta center.
   virtual double EvalDelta(ElementTransformation &T, const IntegrationPoint &ip);
   /** @brief A DeltaFunction cannot be evaluated. Calling this method will
       cause an MFEM error, terminating the application. */
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   { mfem_error("DeltaCoefficient::Eval"); return 0.; }
   virtual ~DeltaCoefficient() { delete weight; }
};

/** @brief Derived coefficient that takes the value of the parent coefficient
    for the active attributes and is zero otherwise. */
class RestrictedCoefficient : public Coefficient
{
private:
   Coefficient *c;
   Array<int> active_attr;

public:
   /** @brief Construct with a parent coefficient and an array with
       ones marking the attributes on which this coefficient should be
       active. */
   RestrictedCoefficient(Coefficient &_c, Array<int> &attr)
   { c = &_c; attr.Copy(active_attr); }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   { return active_attr[T.Attribute-1] ? c->Eval(T, ip, GetTime()) : 0.0; }
};

/// Base class for vector Coefficients that optionally depend on time and space.
class VectorCoefficient
{
protected:
   int vdim;
   double time;

public:
   /// Initialize the VectorCoefficient with vector dimension @a vd.
   VectorCoefficient(int vd) { vdim = vd; time = 0.; }

   /// Set the time for time dependent coefficients
   void SetTime(double t) { time = t; }

   /// Get the time for time dependent coefficients
   double GetTime() { return time; }

   /// Returns dimension of the vector.
   int GetVDim() { return vdim; }

   /** @brief Evaluate the vector coefficient in the element described by @a T
       at the point @a ip, storing the result in @a V. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip) = 0;

   /** @brief Evaluate the vector coefficient in the element described by @a T
       at all points of @a ir, storing the result in @a M. */
   /** The dimensions of @a M are GetVDim() by ir.GetNPoints() and they must be
       set by the implementation of this method.

       The general implementation provided by the base class (using the Eval
       method for one IntegrationPoint at a time) can be overloaded for more
       efficient implementation.

       @note The IntegrationPoint associated with @a T is not used, and this
       method will generally modify this IntegrationPoint associated with @a T.
   */
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);

   virtual ~VectorCoefficient() { }
};


/// Vector coefficient that is constant in space and time.
class VectorConstantCoefficient : public VectorCoefficient
{
private:
   Vector vec;
public:
   /// Construct the coefficient with constant vector @a v.
   VectorConstantCoefficient(const Vector &v)
      : VectorCoefficient(v.Size()), vec(v) { }
   using VectorCoefficient::Eval;

   ///  Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip) { V = vec; }

   /// Return a reference to the constant vector in this class.
   const Vector& GetVec() { return vec; }
};

/// A general C-function vector coefficient
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
   /// Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~VectorFunctionCoefficient() { }
};

/** @brief Vector coefficient defined by an array of scalar coefficients.
    Coefficients that are not set will evaluate to zero in the vector. This
    object takes ownership of the array of coefficients inside it and deletes
    them at object destruction. */
class VectorArrayCoefficient : public VectorCoefficient
{
private:
   Array<Coefficient*> Coeff;
   Array<bool> ownCoeff;

public:
   /** @brief Construct vector of dim coefficients.  The actual coefficients
       still need to be added with Set(). */
   explicit VectorArrayCoefficient(int dim);

   /// Returns i'th coefficient.
   Coefficient* GetCoeff(int i) { return Coeff[i]; }

   /// Returns the entire array of coefficients.
   Coefficient **GetCoeffs() { return Coeff; }

   /// Sets coefficient in the vector.
   void Set(int i, Coefficient *c, bool own=true);

   /// Evaluates i'th component of the vector of coefficients and returns the
   /// value.
   double Eval(int i, ElementTransformation &T, const IntegrationPoint &ip)
   { return Coeff[i] ? Coeff[i]->Eval(T, ip, GetTime()) : 0.0; }

   using VectorCoefficient::Eval;
   /** @brief Evaluate the coefficient. Each element of vector V comes from the
       associated array of scalar coefficients. */
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   /// Destroys vector coefficient.
   virtual ~VectorArrayCoefficient();
};

/// Vector coefficient defined by a vector GridFunction
class VectorGridFunctionCoefficient : public VectorCoefficient
{
protected:
   const GridFunction *GridFunc;

public:
   /** @brief Construct an empty coefficient.  Calling Eval() before the grid
       function is set will cause a segfault. */
   VectorGridFunctionCoefficient() : VectorCoefficient(0), GridFunc(NULL) { }

   /** @brief  Construct the coefficient with grid function @a gf.  The
       grid function is not owned by the coefficient. */
   VectorGridFunctionCoefficient(const GridFunction *gf);

   /** @brief Set the grid function for this coefficient. Also sets the Vector
       dimension to match that of the @a gf. */
   void SetGridFunction(const GridFunction *gf);

   ///  Returns a pointer to the grid function in this Coefficient
   const GridFunction * GetGridFunction() const { return GridFunc; }

   /// Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   /** @brief Evaluate the vector coefficients at all of the locations in the
       integration rule and write the vectors into the columns of matrix @a
       M. */
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);

   virtual ~VectorGridFunctionCoefficient() { }
};

/// Vector coefficient defined as the Gradient of a scalar GridFunction
class GradientGridFunctionCoefficient : public VectorCoefficient
{
protected:
   const GridFunction *GridFunc;

public:

   /** @brief Construct the coefficient with a scalar grid function @a gf. The
       grid function is not owned by the coefficient. */
   GradientGridFunctionCoefficient(const GridFunction *gf);

   ///Set the scalar grid function.
   void SetGridFunction(const GridFunction *gf);

   ///Get the scalar grid function.
   const GridFunction * GetGridFunction() const { return GridFunc; }

   /// Evaluate the gradient vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   /** @brief Evaluate the gradient vector coefficient at all of the locations
       in the integration rule and write the vectors into columns of matrix @a
       M. */
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);

   virtual ~GradientGridFunctionCoefficient() { }
};

/// Vector coefficient defined as the Curl of a vector GridFunction
class CurlGridFunctionCoefficient : public VectorCoefficient
{
protected:
   const GridFunction *GridFunc;

public:
   /** @brief Construct the coefficient with a vector grid function @a gf. The
       grid function is not owned by the coefficient. */
   CurlGridFunctionCoefficient(const GridFunction *gf);

   /// Set the vector grid function.
   void SetGridFunction(const GridFunction *gf);

   /// Get the vector grid function.
   const GridFunction * GetGridFunction() const { return GridFunc; }

   using VectorCoefficient::Eval;
   /// Evaluate the vector curl coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~CurlGridFunctionCoefficient() { }
};

/// Scalar coefficient defined as the Divergence of a vector GridFunction
class DivergenceGridFunctionCoefficient : public Coefficient
{
protected:
   const GridFunction *GridFunc;

public:
   /** @brief Construct the coefficient with a vector grid function @a gf. The
       grid function is not owned by the coefficient. */
   DivergenceGridFunctionCoefficient(const GridFunction *gf);

   /// Set the vector grid function.
   void SetGridFunction(const GridFunction *gf) { GridFunc = gf; }

   /// Get the vector grid function.
   const GridFunction * GetGridFunction() const { return GridFunc; }

   /// Evaluate the scalar divergence coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);

   virtual ~DivergenceGridFunctionCoefficient() { }
};

/** @brief Vector coefficient defined by a scalar DeltaCoefficient and a
    constant vector direction.

    WARNING this cannot be used as a normal coefficient. The usual Eval method
    is disabled. */
class VectorDeltaCoefficient : public VectorCoefficient
{
protected:
   Vector dir;
   DeltaCoefficient d;

public:
   /// Construct with a vector of dimension @a _vdim.
   VectorDeltaCoefficient(int _vdim)
      : VectorCoefficient(_vdim), dir(_vdim), d() { }

   /** @brief Construct with a Vector object representing the direction and a
       unit delta function centered at (0.0,0.0,0.0) */
   VectorDeltaCoefficient(const Vector& _dir)
      : VectorCoefficient(_dir.Size()), dir(_dir), d() { }

   /** @brief Construct with a Vector object representing the direction and a
       delta function scaled by @a s and centered at (x,0.0,0.0) */
   VectorDeltaCoefficient(const Vector& _dir, double x, double s)
      : VectorCoefficient(_dir.Size()), dir(_dir), d(x,s) { }

   /** @brief Construct with a Vector object representing the direction and a
       delta function scaled by @a s and centered at (x,y,0.0) */
   VectorDeltaCoefficient(const Vector& _dir, double x, double y, double s)
      : VectorCoefficient(_dir.Size()), dir(_dir), d(x,y,s) { }

   /** @brief Construct with a Vector object representing the direction and a
       delta function scaled by @a s and centered at (x,y,z) */
   VectorDeltaCoefficient(const Vector& _dir, double x, double y, double z,
                          double s)
      : VectorCoefficient(_dir.Size()), dir(_dir), d(x,y,z,s) { }

   /// Replace the associated DeltaCoefficient with a new DeltaCoefficient.
   /** The new DeltaCoefficient cannot have a specified weight Coefficient, i.e.
       DeltaCoefficient::Weight() should return NULL. */
   void SetDeltaCoefficient(const DeltaCoefficient& _d) { d = _d; }

   /// Return the associated scalar DeltaCoefficient.
   DeltaCoefficient& GetDeltaCoefficient() { return d; }

   void SetScale(double s) { d.SetScale(s); }
   void SetDirection(const Vector& _d);

   void SetDeltaCenter(const Vector& center) { d.SetDeltaCenter(center); }
   void GetDeltaCenter(Vector& center) { d.GetDeltaCenter(center); }

   /** @brief Return the specified direction vector multiplied by the value
       returned by DeltaCoefficient::EvalDelta() of the associated scalar
       DeltaCoefficient. */
   virtual void EvalDelta(Vector &V, ElementTransformation &T,
                          const IntegrationPoint &ip);

   using VectorCoefficient::Eval;
   /** @brief A VectorDeltaFunction cannot be evaluated. Calling this method
       will cause an MFEM error, terminating the application. */
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   { mfem_error("VectorDeltaCoefficient::Eval"); }
   virtual ~VectorDeltaCoefficient() { }
};

/** @brief Derived vector coefficient that has the value of the parent vector
    where it is active and is zero otherwise. */
class VectorRestrictedCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient *c;
   Array<int> active_attr;

public:
   /** @brief Construct with a parent vector coefficient and an array of zeros
       and ones representing the attributes for which this coefficient should be
       active. */
   VectorRestrictedCoefficient(VectorCoefficient &vc, Array<int> &attr)
      : VectorCoefficient(vc.GetVDim())
   { c = &vc; attr.Copy(active_attr); }

   /// Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   /** @brief Evaluate the vector coefficient at all of the locations in the
       integration rule and write the vectors into the columns of matrix @a
       M. */
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir);
};


/// Base class for Matrix Coefficients that optionally depend on time and space.
class MatrixCoefficient
{
protected:
   int height, width;
   double time;

public:
   /// Construct a dim x dim matrix coefficient.
   explicit MatrixCoefficient(int dim) { height = width = dim; time = 0.; }

   /// Construct a h x w matrix coefficient.
   MatrixCoefficient(int h, int w) : height(h), width(w), time(0.) { }

   /// Set the time for time dependent coefficients
   void SetTime(double t) { time = t; }

   /// Get the time for time dependent coefficients
   double GetTime() { return time; }

   /// Get the height of the matrix.
   int GetHeight() const { return height; }

   /// Get the width of the matrix.
   int GetWidth() const { return width; }

   /// For backward compatibility get the width of the matrix.
   int GetVDim() const { return width; }

   /** @brief Evaluate the matrix coefficient in the element described by @a T
       at the point @a ip, storing the result in @a K. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip) = 0;

   virtual ~MatrixCoefficient() { }
};


/// A matrix coefficient that is constant in space and time.
class MatrixConstantCoefficient : public MatrixCoefficient
{
private:
   DenseMatrix mat;
public:
   ///Construct using matrix @a m for the constant.
   MatrixConstantCoefficient(const DenseMatrix &m)
      : MatrixCoefficient(m.Height(), m.Width()), mat(m) { }
   using MatrixCoefficient::Eval;
   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip) { M = mat; }
};


/** @brief A matrix coefficient with an optional scalar coefficient multiplier
    \a q.  The matrix function can either be represented by a C-function or a
    constant matrix provided when constructing this object.  */
class MatrixFunctionCoefficient : public MatrixCoefficient
{
private:
   void (*Function)(const Vector &, DenseMatrix &);
   void (*TDFunction)(const Vector &, double, DenseMatrix &);
   Coefficient *Q;
   DenseMatrix mat;

public:
   /// Construct a square matrix coefficient from a C-function without time
   /// dependence.
   MatrixFunctionCoefficient(int dim, void (*F)(const Vector &, DenseMatrix &),
                             Coefficient *q = NULL)
      : MatrixCoefficient(dim), Q(q)
   {
      Function = F;
      TDFunction = NULL;
      mat.SetSize(0);
   }

   /// Construct a constant matrix coefficient times a scalar Coefficient
   MatrixFunctionCoefficient(const DenseMatrix &m, Coefficient &q)
      : MatrixCoefficient(m.Height(), m.Width()), Q(&q)
   {
      Function = NULL;
      TDFunction = NULL;
      mat = m;
   }

   /// Construct a square matrix coefficient from a C-function with
   /// time-dependence.
   MatrixFunctionCoefficient(int dim,
                             void (*TDF)(const Vector &, double, DenseMatrix &),
                             Coefficient *q = NULL)
      : MatrixCoefficient(dim), Q(q)
   {
      Function = NULL;
      TDFunction = TDF;
      mat.SetSize(0);
   }

   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~MatrixFunctionCoefficient() { }
};



/** @brief Matrix coefficient defined by a matrix of scalar coefficients.
    Coefficients that are not set will evaluate to zero in the vector. The
    coefficient is stored as a flat Array with indexing (i,j) -> i*width+j. */
class MatrixArrayCoefficient : public MatrixCoefficient
{
private:
   Array<Coefficient *> Coeff;
   Array<bool> ownCoeff;

public:
   /** @brief Construct a coefficient matrix of dimensions @a dim * @a dim. The
       actual coefficients still need to be added with Set(). */
   explicit MatrixArrayCoefficient (int dim);

   /// Get the coefficient located at (i,j) in the matrix.
   Coefficient* GetCoeff (int i, int j) { return Coeff[i*width+j]; }

   /** @brief Set the coefficient located at (i,j) in the matrix.  By default by
       default this will take ownership of the Coefficient passed in, but this
       can be overridden with the @a own parameter. */
   void Set(int i, int j, Coefficient * c, bool own=true);

   /// Evaluate coefficient located at (i,j) in the matrix using integration
   /// point @a ip.
   double Eval(int i, int j, ElementTransformation &T, const IntegrationPoint &ip)
   { return Coeff[i*width+j] ? Coeff[i*width+j] -> Eval(T, ip, GetTime()) : 0.0; }

   /// Evaluate the matrix coefficient @a ip.
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~MatrixArrayCoefficient();
};


/** @brief Derived matrix coefficient that has the value of the parent matrix
    coefficient where it is active and is zero otherwise. */
class MatrixRestrictedCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient *c;
   Array<int> active_attr;

public:
   /** @brief Construct with a parent matrix coefficient and an array of zeros
       and ones representing the attributes for which this coefficient should be
       active. */
   MatrixRestrictedCoefficient(MatrixCoefficient &mc, Array<int> &attr)
      : MatrixCoefficient(mc.GetHeight(), mc.GetWidth())
   { c = &mc; attr.Copy(active_attr); }

   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

/// Coefficients based on sums and products of other coefficients

/// Scalar coefficient defined as the sum of two scalar coefficients
class SumCoefficient : public Coefficient
{
private:
   Coefficient * a;
   Coefficient * b;

   double alpha;
   double beta;

public:
   /// Construct with the two coefficients.  Result is _alpha * A + _beta * B.
   SumCoefficient(Coefficient &A, Coefficient &B,
                  double _alpha = 1.0, double _beta = 1.0)
      : a(&A), b(&B), alpha(_alpha), beta(_beta) { }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   { return alpha * a->Eval(T, ip) + beta * b->Eval(T, ip); }
};

/// Scalar coefficient defined as the product of two scalar coefficients
class ProductCoefficient : public Coefficient
{
private:
   Coefficient * a;
   Coefficient * b;

public:
   /// Construct with the two coefficients.  Result is A * B.
   ProductCoefficient(Coefficient &A, Coefficient &B)
      : a(&A), b(&B) { }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   { return a->Eval(T, ip) * b->Eval(T, ip); }
};

/// Scalar coefficient defined as a scalar raised to a power
class PowerCoefficient : public Coefficient
{
private:
   Coefficient * a;

   double p;

public:
   /// Construct with a coefficient and a constant power @a _p.  Result is A^p.
   PowerCoefficient(Coefficient &A, double _p)
      : a(&A), p(_p) { }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   { return pow(a->Eval(T, ip), p); }
};


/// Scalar coefficient defined as the inner product of two vector coefficients
class InnerProductCoefficient : public Coefficient
{
private:
   VectorCoefficient * a;
   VectorCoefficient * b;

   mutable Vector va;
   mutable Vector vb;
public:
   /// Construct with the two vector coefficients.  Result is \f$ A \cdot B \f$.
   InnerProductCoefficient(VectorCoefficient &A, VectorCoefficient &B);

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

/// Scalar coefficient defined as a cross product of two vectors in the xy-plane.
class VectorRotProductCoefficient : public Coefficient
{
private:
   VectorCoefficient * a;
   VectorCoefficient * b;

   mutable Vector va;
   mutable Vector vb;

public:
   /// Construct with the two vector coefficients.  Result is \f$ A_x B_y - A_y * B_x; \f$.
   VectorRotProductCoefficient(VectorCoefficient &A, VectorCoefficient &B);

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

/// Scalar coefficient defined as the determinant of a matrix coefficient
class DeterminantCoefficient : public Coefficient
{
private:
   MatrixCoefficient * a;

   mutable DenseMatrix ma;

public:
   /// Construct with the matrix.
   DeterminantCoefficient(MatrixCoefficient &A);

   /// Evaluate the determinant coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

/// Vector coefficient defined as the sum of two vector coefficients
class VectorSumCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * a;
   VectorCoefficient * b;

   double alpha;
   double beta;

   mutable Vector va;

public:
   /// Construct with the two vector coefficients.  Result is _alpha * A + _beta * B.
   VectorSumCoefficient(VectorCoefficient &A, VectorCoefficient &B,
                        double _alpha = 1.0, double _beta = 1.0);

   /// Evaluate the coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
   using VectorCoefficient::Eval;
};

/// Vector coefficient defined as a product of scalar and vector coefficients.
class ScalarVectorProductCoefficient : public VectorCoefficient
{
private:
   Coefficient * a;
   VectorCoefficient * b;

public:
   /// Construct with the two coefficients.  Result is A * B.
   ScalarVectorProductCoefficient(Coefficient &A, VectorCoefficient &B);

   /// Evaluate the coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
   using VectorCoefficient::Eval;
};

/// Vector coefficient defined as a cross product of two vectors
class VectorCrossProductCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * a;
   VectorCoefficient * b;

   mutable Vector va;
   mutable Vector vb;

public:
   /// Construct with the two coefficients.  Result is A x B.
   VectorCrossProductCoefficient(VectorCoefficient &A, VectorCoefficient &B);

   /// Evaluate the coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
   using VectorCoefficient::Eval;
};

/** @brief Vector coefficient defined as a product of a matrix coefficient and
    a vector coefficient. */
class MatVecCoefficient : public VectorCoefficient
{
private:
   MatrixCoefficient * a;
   VectorCoefficient * b;

   mutable DenseMatrix ma;
   mutable Vector vb;

public:
   /// Construct with the two coefficients.  Result is A*B.
   MatVecCoefficient(MatrixCoefficient &A, VectorCoefficient &B);

   /// Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
   using VectorCoefficient::Eval;
};

/// Constant matrix coefficient defined as the identity of dimension d
class IdentityMatrixCoefficient : public MatrixCoefficient
{
private:
   int dim;

public:
   /// Construct with the dimension of the square identity matrix.
   IdentityMatrixCoefficient(int d)
      : MatrixCoefficient(d, d), dim(d) { }

   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

/// Matrix coefficient defined as the sum of two matrix coefficients.
class MatrixSumCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient * a;
   MatrixCoefficient * b;

   double alpha;
   double beta;

   mutable DenseMatrix ma;

public:
   /// Construct with the two coefficients.  Result is _alpha * A + _beta * B.
   MatrixSumCoefficient(MatrixCoefficient &A, MatrixCoefficient &B,
                        double _alpha = 1.0, double _beta = 1.0);

   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

/** @brief Matrix coefficient defined as a product of a scalar coefficient and a
    matrix coefficient.*/
class ScalarMatrixProductCoefficient : public MatrixCoefficient
{
private:
   Coefficient * a;
   MatrixCoefficient * b;

public:
   /// Construct with the two coefficients.  Result is A*B.
   ScalarMatrixProductCoefficient(Coefficient &A, MatrixCoefficient &B);

   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

/// Matrix coefficient defined as the transpose a matrix coefficient
class TransposeMatrixCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient * a;

public:
   /// Construct with the matrix coefficient.  Result is \f$ A^T \f$.
   TransposeMatrixCoefficient(MatrixCoefficient &A);

   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

/// Matrix coefficient defined as the inverse a matrix coefficient.
class InverseMatrixCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient * a;

public:
   /// Construct with the matrix coefficient.  Result is \f$ A^{-1} \f$.
   InverseMatrixCoefficient(MatrixCoefficient &A);

   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

/// Matrix coefficient defined as the outer product of two vector coefficients.
class OuterProductCoefficient : public MatrixCoefficient
{
private:
   VectorCoefficient * a;
   VectorCoefficient * b;

   mutable Vector va;
   mutable Vector vb;

public:
   /// Construct with two vector coefficients.  Result is \f$ A B^T \f$.
   OuterProductCoefficient(VectorCoefficient &A, VectorCoefficient &B);

   /// Evaluate the matrix coefficient at @a ip.
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip);
};


class QuadratureFunction;

/** @brief Vector quadrature function coefficient which requires that the
    quadrature rules used for this vector coefficient be the same as those that
    live within the supplied QuadratureFunction. */
class VectorQuadratureFunctionCoefficient : public VectorCoefficient
{
private:
   const QuadratureFunction &QuadF; //do not own
   int index;

public:
   /// Constructor with a quadrature function as input
   VectorQuadratureFunctionCoefficient(QuadratureFunction &qf);

   /** Set the starting index within the QuadFunc that'll be used to project
       outwards as well as the corresponding length. The projected length should
       have the bounds of 1 <= length <= (length QuadFunc - index). */
   void SetComponent(int _index, int _length);

   const QuadratureFunction& GetQuadFunction() const { return QuadF; }

   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   virtual ~VectorQuadratureFunctionCoefficient() { }
};

/** @brief Quadrature function coefficient which requires that the quadrature
    rules used for this coefficient be the same as those that live within the
    supplied QuadratureFunction. */
class QuadratureFunctionCoefficient : public Coefficient
{
private:
   const QuadratureFunction &QuadF;

public:
   /// Constructor with a quadrature function as input
   QuadratureFunctionCoefficient(QuadratureFunction &qf);

   const QuadratureFunction& GetQuadFunction() const { return QuadF; }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);

   virtual ~QuadratureFunctionCoefficient() { }
};

/** @brief Compute the Lp norm of a function f.
    \f$ \| f \|_{Lp} = ( \int_\Omega | f |^p d\Omega)^{1/p} \f$ */
double ComputeLpNorm(double p, Coefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[]);

/** @brief Compute the Lp norm of a vector function f = {f_i}_i=1...N.
    \f$ \| f \|_{Lp} = ( \sum_i \| f_i \|_{Lp}^p )^{1/p} \f$ */
double ComputeLpNorm(double p, VectorCoefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[]);

#ifdef MFEM_USE_MPI
/** @brief Compute the global Lp norm of a function f.
    \f$ \| f \|_{Lp} = ( \int_\Omega | f |^p d\Omega)^{1/p} \f$ */
double ComputeGlobalLpNorm(double p, Coefficient &coeff, ParMesh &pmesh,
                           const IntegrationRule *irs[]);

/** @brief Compute the global Lp norm of a vector function f = {f_i}_i=1...N.
    \f$ \| f \|_{Lp} = ( \sum_i \| f_i \|_{Lp}^p )^{1/p} \f$ */
double ComputeGlobalLpNorm(double p, VectorCoefficient &coeff, ParMesh &pmesh,
                           const IntegrationRule *irs[]);
#endif

}

#endif
