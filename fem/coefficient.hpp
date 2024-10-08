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

#ifndef MFEM_COEFFICIENT
#define MFEM_COEFFICIENT

#include <functional>

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "intrules.hpp"
#include "eltrans.hpp"

namespace mfem
{

class Mesh;
class QuadratureSpaceBase;
class QuadratureFunction;

#ifdef MFEM_USE_MPI
class ParMesh;
#endif


/** @brief Base class Coefficients that optionally depend on space and time.
    These are used by the BilinearFormIntegrator, LinearFormIntegrator, and
    NonlinearFormIntegrator classes to represent the physical coefficients in
    the PDEs that are being discretized. This class can also be used in a more
    general way to represent functions that don't necessarily belong to a FE
    space, e.g., to project onto GridFunctions to use as initial conditions,
    exact solutions, etc. See, e.g., ex4 or ex22 for these uses. */
class Coefficient
{
protected:
   real_t time;

public:
   Coefficient() { time = 0.; }

   /// Set the time for time dependent coefficients
   virtual void SetTime(real_t t) { time = t; }

   /// Get the time for time dependent coefficients
   real_t GetTime() { return time; }

   /** @brief Evaluate the coefficient in the element described by @a T at the
       point @a ip. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) = 0;

   /** @brief Evaluate the coefficient in the element described by @a T at the
       point @a ip at time @a t. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip, real_t t)
   {
      SetTime(t);
      return Eval(T, ip);
   }

   /// @brief Fill the QuadratureFunction @a qf by evaluating the coefficient at
   /// the quadrature points.
   virtual void Project(QuadratureFunction &qf);

   virtual ~Coefficient() { }
};


/// A coefficient that is constant across space and time
class ConstantCoefficient : public Coefficient
{
public:
   real_t constant;

   /// c is value of constant function
   explicit ConstantCoefficient(real_t c = 1.0) { constant=c; }

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   { return (constant); }

   /// Fill the QuadratureFunction @a qf with the constant value.
   void Project(QuadratureFunction &qf) override;
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
   real_t &operator()(int i) { return constants(i-1); }

   /// Set the constants for all attributes to constant @a c.
   void operator=(real_t c) { constants = c; }

   /// Returns the number of constants representing different attributes.
   int GetNConst() { return constants.Size(); }

   /// Evaluate the coefficient.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/** @brief A piecewise coefficient with the pieces keyed off the element
    attribute numbers.

    A value of zero will be returned for any missing attribute numbers.

    This object will not assume ownership of any Coefficient objects
    passed to it. Consequently, the caller must ensure that the
    individual Coefficient objects are not deleted while this
    PWCoefficient is still in use.

    \note The keys may either be domain attribute numbers or boundary
    attribute numbers. If the PWCoefficient is used with a domain
    integrator the keys are assumed to be domain attribute
    numbers. Similarly, if the PWCoefficient is used with a boundary
    integrator the keys are assumed to be boundary attribute numbers.
*/
class PWCoefficient : public Coefficient
{
private:
   /** Internal data structure to store pointers to the appropriate
       coefficients for different regions of the mesh. The keys used
       in the map are the mesh attribute numbers (either element
       attribute or boundary element attribute depending upon
       context). The values returned for any missing attributes will
       be zero. The coefficient pointers may be NULL in which case a
       value of zero is returned.

       The Coefficient objects contained in this map are NOT owned by
       this PWCoefficient object. This means that they will not be
       deleted when this object is deleted also the caller must ensure
       that the various Coefficient objects are not deleted while this
       PWCoefficient is still needed.
   */
   std::map<int, Coefficient*> pieces;

   /** Convenience function to check for compatible array lengths,
       loop over the arrays, and add their attribute/Coefficient pairs
       to the internal data structure.
   */
   void InitMap(const Array<int> & attr,
                const Array<Coefficient*> & coefs);

public:

   /// Constructs a piecewise coefficient
   explicit PWCoefficient() {}

   /// Construct the coefficient using arrays describing the pieces
   /** \param attr - an array of attribute numbers for each piece
       \param coefs - the corresponding array of Coefficient pointers
       Any missing attributes or NULL coefficient pointers will result in a
       value of zero being returned for that attribute.

       \note Ownership of the Coefficient objects will NOT be
       transferred to this object.
   */
   PWCoefficient(const Array<int> & attr,
                 const Array<Coefficient*> & coefs)
   { InitMap(attr, coefs); }

   /// Set the time for time dependent coefficients
   void SetTime(real_t t) override;

   /// Replace a set of coefficients
   void UpdateCoefficients(const Array<int> & attr,
                           const Array<Coefficient*> & coefs)
   { InitMap(attr, coefs); }

   /// Replace a single Coefficient for a particular attribute
   void UpdateCoefficient(int attr, Coefficient & coef)
   { pieces[attr] = &coef; }

   /// Remove a single Coefficient for a particular attribute
   void ZeroCoefficient(int attr)
   { pieces.erase(attr); }

   /// Evaluate the coefficient.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// A general function coefficient
class FunctionCoefficient : public Coefficient
{
protected:
   std::function<real_t(const Vector &)> Function;
   std::function<real_t(const Vector &, real_t)> TDFunction;

public:
   /// Define a time-independent coefficient from a std function
   /** \param F time-independent std::function */
   FunctionCoefficient(std::function<real_t(const Vector &)> F)
      : Function(std::move(F))
   { }

   /// Define a time-dependent coefficient from a std function
   /** \param TDF time-dependent function */
   FunctionCoefficient(std::function<real_t(const Vector &, real_t)> TDF)
      : TDFunction(std::move(TDF))
   { }

   /// (DEPRECATED) Define a time-independent coefficient from a C-function
   /** @deprecated Use the method where the C-function, @a f, uses a const
       Vector argument instead of Vector. */
   MFEM_DEPRECATED FunctionCoefficient(real_t (*f)(Vector &))
   {
      Function = reinterpret_cast<real_t(*)(const Vector&)>(f);
      TDFunction = NULL;
   }

   /// (DEPRECATED) Define a time-dependent coefficient from a C-function
   /** @deprecated Use the method where the C-function, @a tdf, uses a const
       Vector argument instead of Vector. */
   MFEM_DEPRECATED FunctionCoefficient(real_t (*tdf)(Vector &, real_t))
   {
      Function = NULL;
      TDFunction = reinterpret_cast<real_t(*)(const Vector&,real_t)>(tdf);
   }

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// A common base class for returning individual components of the domain's
/// Cartesian coordinates.
class CartesianCoefficient : public Coefficient
{
protected:
   int comp;
   mutable Vector transip;

   /// @a comp_ index of the desired component (0 -> x, 1 -> y, 2 -> z)
   CartesianCoefficient(int comp_) : comp(comp_), transip(3) {}

public:
   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// Scalar coefficient which returns the x-component of the evaluation point
class CartesianXCoefficient : public CartesianCoefficient
{
public:
   CartesianXCoefficient() : CartesianCoefficient(0) {}
};

/// Scalar coefficient which returns the y-component of the evaluation point
class CartesianYCoefficient : public CartesianCoefficient
{
public:
   CartesianYCoefficient() : CartesianCoefficient(1) {}
};

/// Scalar coefficient which returns the z-component of the evaluation point
class CartesianZCoefficient : public CartesianCoefficient
{
public:
   CartesianZCoefficient() : CartesianCoefficient(2) {}
};

/// Scalar coefficient which returns the radial distance from the axis of
/// the evaluation point in the cylindrical coordinate system
class CylindricalRadialCoefficient : public Coefficient
{
private:
   mutable Vector transip;

public:
   CylindricalRadialCoefficient() : transip(3) {}

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// Scalar coefficient which returns the angular position or azimuth (often
/// denoted by theta) of the evaluation point in the cylindrical coordinate
/// system
class CylindricalAzimuthalCoefficient : public Coefficient
{
private:
   mutable Vector transip;

public:
   CylindricalAzimuthalCoefficient() : transip(3) {}

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// Scalar coefficient which returns the height or altitude of
/// the evaluation point in the cylindrical coordinate system
typedef CartesianZCoefficient CylindricalZCoefficient;

/// Scalar coefficient which returns the radial distance from the origin of
/// the evaluation point in the spherical coordinate system
class SphericalRadialCoefficient : public Coefficient
{
private:
   mutable Vector transip;

public:
   SphericalRadialCoefficient() : transip(3) {}

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// Scalar coefficient which returns the azimuthal angle (often denoted by phi)
/// of the evaluation point in the spherical coordinate system
class SphericalAzimuthalCoefficient : public Coefficient
{
private:
   mutable Vector transip;

public:
   SphericalAzimuthalCoefficient() : transip(3) {}

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// Scalar coefficient which returns the polar angle (often denoted by theta)
/// of the evaluation point in the spherical coordinate system
class SphericalPolarCoefficient : public Coefficient
{
private:
   mutable Vector transip;

public:
   SphericalPolarCoefficient() : transip(3) {}

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
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
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;

   /// @brief Fill the QuadratureFunction @a qf by evaluating the coefficient at
   /// the quadrature points.
   ///
   /// This function uses the efficient QuadratureFunction::ProjectGridFunction
   /// to fill the QuadratureFunction.
   void Project(QuadratureFunction &qf) override;
};


/** @brief A coefficient that depends on 1 or 2 parent coefficients and a
    transformation rule represented by a C-function.

    $ C(x,t) = T(Q1(x,t)) $ or $ C(x,t) = T(Q1(x,t), Q2(x,t)) $

    where T is the transformation rule, and Q1/Q2 are the parent coefficients.*/
class TransformedCoefficient : public Coefficient
{
private:
   Coefficient * Q1;
   Coefficient * Q2;
   std::function<real_t(real_t)> Transform1;
   std::function<real_t(real_t, real_t)> Transform2;

public:
   TransformedCoefficient (Coefficient * q, std::function<real_t(real_t)> F)
      : Q1(q), Transform1(std::move(F)) { Q2 = 0; Transform2 = 0; }
   TransformedCoefficient (Coefficient * q1,Coefficient * q2,
                           std::function<real_t(real_t, real_t)> F)
      : Q1(q1), Q2(q2), Transform2(std::move(F)) { Transform1 = 0; }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

/** @brief Delta function coefficient optionally multiplied by a weight
    coefficient and a scaled time dependent C-function.

    $ F(x,t) = w(x,t) s T(t) d(x - xc) $

    where w is the optional weight coefficient, @a s is a scale factor
    T is an optional time-dependent function and d is a delta function.

    WARNING this cannot be used as a normal coefficient.  The usual Eval
    method is disabled. */
class DeltaCoefficient : public Coefficient
{
protected:
   real_t center[3], scale, tol;
   Coefficient *weight;
   int sdim;
   real_t (*tdf)(real_t);

public:

   /// Construct a unit delta function centered at (0.0,0.0,0.0)
   DeltaCoefficient()
   {
      center[0] = center[1] = center[2] = 0.; scale = 1.; tol = 1e-12;
      weight = NULL; sdim = 0; tdf = NULL;
   }

   /// Construct a delta function scaled by @a s and centered at (x,0.0,0.0)
   DeltaCoefficient(real_t x, real_t s)
   {
      center[0] = x; center[1] = 0.; center[2] = 0.; scale = s; tol = 1e-12;
      weight = NULL; sdim = 1; tdf = NULL;
   }

   /// Construct a delta function scaled by @a s and centered at (x,y,0.0)
   DeltaCoefficient(real_t x, real_t y, real_t s)
   {
      center[0] = x; center[1] = y; center[2] = 0.; scale = s; tol = 1e-12;
      weight = NULL; sdim = 2; tdf = NULL;
   }

   /// Construct a delta function scaled by @a s and centered at (x,y,z)
   DeltaCoefficient(real_t x, real_t y, real_t z, real_t s)
   {
      center[0] = x; center[1] = y; center[2] = z; scale = s; tol = 1e-12;
      weight = NULL; sdim = 3; tdf = NULL;
   }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Set the center location of the delta function.
   void SetDeltaCenter(const Vector& center);

   /// Set the scale value multiplying the delta function.
   void SetScale(real_t s_) { scale = s_; }

   /// Set a time-dependent function that multiplies the Scale().
   void SetFunction(real_t (*f)(real_t)) { tdf = f; }

   /** @brief Set the tolerance used during projection onto GridFunction to
       identify the Mesh vertex where the Center() of the delta function
       lies. (default 1e-12)*/
   void SetTol(real_t tol_) { tol = tol_; }

   /// Set a weight Coefficient that multiplies the DeltaCoefficient.
   /** The weight Coefficient multiplies the value returned by EvalDelta() but
       not the value returned by Scale().
       The weight Coefficient is also used as the L2-weight function when
       projecting the DeltaCoefficient onto a GridFunction, so that the weighted
       integral of the projection is exactly equal to the Scale(). */
   void SetWeight(Coefficient *w) { weight = w; }

   /// Return a pointer to a c-array representing the center of the delta
   /// function.
   const real_t *Center() { return center; }

   /** @brief Return the scale factor times the optional time dependent
       function.  Returns $ s T(t) $ with $ T(t) = 1 $ when
       not set by the user. */
   real_t Scale() { return tdf ? (*tdf)(GetTime())*scale : scale; }

   /// Return the tolerance used to identify the mesh vertices
   real_t Tol() { return tol; }

   /// See SetWeight() for description of the weight Coefficient.
   Coefficient *Weight() { return weight; }

   /// Write the center of the delta function into @a center.
   void GetDeltaCenter(Vector& center);

   /// The value of the function assuming we are evaluating at the delta center.
   virtual real_t EvalDelta(ElementTransformation &T, const IntegrationPoint &ip);
   /** @brief A DeltaFunction cannot be evaluated. Calling this method will
       cause an MFEM error, terminating the application. */
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
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
   RestrictedCoefficient(Coefficient &c_, Array<int> &attr)
   { c = &c_; attr.Copy(active_attr); }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   { return active_attr[T.Attribute-1] ? c->Eval(T, ip, GetTime()) : 0.0; }
};

/// Base class for vector Coefficients that optionally depend on time and space.
class VectorCoefficient
{
protected:
   int vdim;
   real_t time;

public:
   /// Initialize the VectorCoefficient with vector dimension @a vd.
   VectorCoefficient(int vd) { vdim = vd; time = 0.; }

   /// Set the time for time dependent coefficients
   virtual void SetTime(real_t t) { time = t; }

   /// Get the time for time dependent coefficients
   real_t GetTime() { return time; }

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

   /// @brief Fill the QuadratureFunction @a qf by evaluating the coefficient at
   /// the quadrature points.
   ///
   /// The @a vdim of the VectorCoefficient should be equal to the @a vdim of
   /// the QuadratureFunction.
   virtual void Project(QuadratureFunction &qf);

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
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override { V = vec; }

   /// Return a reference to the constant vector in this class.
   const Vector& GetVec() const { return vec; }
};

/** @brief A piecewise vector-valued coefficient with the pieces keyed off the
    element attribute numbers.

    A value of zero will be returned for any missing attribute numbers.

    This object will not assume ownership of any VectorCoefficient
    objects passed to it. Consequently, the caller must ensure that
    the individual VectorCoefficient objects are not deleted while
    this PWVectorCoefficient is still in use.

    \note The keys may either be domain attribute numbers or boundary
    attribute numbers. If the PWVectorCoefficient is used with a
    domain integrator the keys are assumed to be domain attribute
    numbers. Similarly, if the PWVectorCoefficient is used with a
    boundary integrator the keys are assumed to be boundary attribute
    numbers.
*/
class PWVectorCoefficient : public VectorCoefficient
{
private:
   /** Internal data structure to store pointers to the appropriate
       coefficients for different regions of the mesh. The keys used
       in the map are the mesh attribute numbers (either element
       attribute or boundary element attribute depending upon
       context). The values returned for any missing attributes will
       be zero. The coefficient pointers may be NULL in which case a
       value of zero is returned.

       The VectorCoefficient objects contained in this map are NOT
       owned by this PWVectorCoefficient object. This means that they
       will not be deleted when this object is deleted also the caller
       must ensure that the various VectorCoefficient objects are not
       deleted while this PWVectorCoefficient is still needed.
   */
   std::map<int, VectorCoefficient*> pieces;

   /** Convenience function to check for compatible array lengths,
       loop over the arrays, and add their attribute/VectorCoefficient
       pairs to the internal data structure.
   */
   void InitMap(const Array<int> & attr,
                const Array<VectorCoefficient*> & coefs);

public:

   /// Constructs a piecewise vector coefficient of dimension vd
   explicit PWVectorCoefficient(int vd): VectorCoefficient(vd) {}

   /// Construct the coefficient using arrays describing the pieces
   /** \param vd - dimension of the vector-valued result
       \param attr - an array of attribute numbers for each piece
       \param coefs - the corresponding array of VectorCoefficient pointers
       Any missing attributes or NULL coefficient pointers will result in a
       zero vector being returned for that attribute.

       \note Ownership of the VectorCoefficient objects will NOT be
       transferred to this object.
   */
   PWVectorCoefficient(int vd, const Array<int> & attr,
                       const Array<VectorCoefficient*> & coefs)
      : VectorCoefficient(vd) { InitMap(attr, coefs); }

   /// Set the time for time dependent coefficients
   void SetTime(real_t t) override;

   /// Replace a set of coefficients
   void UpdateCoefficients(const Array<int> & attr,
                           const Array<VectorCoefficient*> & coefs)
   { InitMap(attr, coefs); }

   /// Replace a single Coefficient for a particular attribute
   void UpdateCoefficient(int attr, VectorCoefficient & coef);

   /// Remove a single VectorCoefficient for a particular attribute
   void ZeroCoefficient(int attr)
   { pieces.erase(attr); }

   /// Evaluate the coefficient.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;
   using VectorCoefficient::Eval;
};

/// A vector coefficient which returns the physical location of the
/// evaluation point in the Cartesian coordinate system.
class PositionVectorCoefficient : public VectorCoefficient
{
public:

   PositionVectorCoefficient(int dim) : VectorCoefficient(dim) {}

   using VectorCoefficient::Eval;
   /// Evaluate the vector coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   virtual ~PositionVectorCoefficient() { }
};

/// A general vector function coefficient
class VectorFunctionCoefficient : public VectorCoefficient
{
private:
   std::function<void(const Vector &, Vector &)> Function;
   std::function<void(const Vector &, real_t, Vector &)> TDFunction;
   Coefficient *Q;

public:
   /// Define a time-independent vector coefficient from a std function
   /** \param dim - the size of the vector
       \param F - time-independent function
       \param q - optional scalar Coefficient to scale the vector coefficient */
   VectorFunctionCoefficient(int dim,
                             std::function<void(const Vector &, Vector &)> F,
                             Coefficient *q = nullptr)
      : VectorCoefficient(dim), Function(std::move(F)), Q(q)
   { }

   /// Define a time-dependent vector coefficient from a std function
   /** \param dim - the size of the vector
       \param TDF - time-dependent function
       \param q - optional scalar Coefficient to scale the vector coefficient */
   VectorFunctionCoefficient(int dim,
                             std::function<void(const Vector &, real_t, Vector &)> TDF,
                             Coefficient *q = nullptr)
      : VectorCoefficient(dim), TDFunction(std::move(TDF)), Q(q)
   { }

   using VectorCoefficient::Eval;
   /// Evaluate the vector coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

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

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Returns i'th coefficient.
   Coefficient* GetCoeff(int i) { return Coeff[i]; }

   /// Returns the entire array of coefficients.
   Coefficient **GetCoeffs() { return Coeff; }

   /// Sets coefficient in the vector.
   void Set(int i, Coefficient *c, bool own=true);

   /// Evaluates i'th component of the vector of coefficients and returns the
   /// value.
   real_t Eval(int i, ElementTransformation &T, const IntegrationPoint &ip)
   { return Coeff[i] ? Coeff[i]->Eval(T, ip, GetTime()) : 0.0; }

   using VectorCoefficient::Eval;
   /** @brief Evaluate the coefficient. Each element of vector V comes from the
       associated array of scalar coefficients. */
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

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
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   /** @brief Evaluate the vector coefficients at all of the locations in the
       integration rule and write the vectors into the columns of matrix @a
       M. */
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationRule &ir) override;

   /// @brief Fill the QuadratureFunction @a qf by evaluating the coefficient at
   /// the quadrature points.
   ///
   /// This function uses the efficient QuadratureFunction::ProjectGridFunction
   /// to fill the QuadratureFunction.
   void Project(QuadratureFunction &qf) override;

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
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   /** @brief Evaluate the gradient vector coefficient at all of the locations
       in the integration rule and write the vectors into columns of matrix @a
       M. */
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationRule &ir) override;

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
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

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
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;

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
   /// Construct with a vector of dimension @a vdim_.
   VectorDeltaCoefficient(int vdim_)
      : VectorCoefficient(vdim_), dir(vdim_), d() { }

   /** @brief Construct with a Vector object representing the direction and a
       unit delta function centered at (0.0,0.0,0.0) */
   VectorDeltaCoefficient(const Vector& dir_)
      : VectorCoefficient(dir_.Size()), dir(dir_), d() { }

   /** @brief Construct with a Vector object representing the direction and a
       delta function scaled by @a s and centered at (x,0.0,0.0) */
   VectorDeltaCoefficient(const Vector& dir_, real_t x, real_t s)
      : VectorCoefficient(dir_.Size()), dir(dir_), d(x,s) { }

   /** @brief Construct with a Vector object representing the direction and a
       delta function scaled by @a s and centered at (x,y,0.0) */
   VectorDeltaCoefficient(const Vector& dir_, real_t x, real_t y, real_t s)
      : VectorCoefficient(dir_.Size()), dir(dir_), d(x,y,s) { }

   /** @brief Construct with a Vector object representing the direction and a
       delta function scaled by @a s and centered at (x,y,z) */
   VectorDeltaCoefficient(const Vector& dir_, real_t x, real_t y, real_t z,
                          real_t s)
      : VectorCoefficient(dir_.Size()), dir(dir_), d(x,y,z,s) { }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Replace the associated DeltaCoefficient with a new DeltaCoefficient.
   /** The new DeltaCoefficient cannot have a specified weight Coefficient, i.e.
       DeltaCoefficient::Weight() should return NULL. */
   void SetDeltaCoefficient(const DeltaCoefficient& d_) { d = d_; }

   /// Return the associated scalar DeltaCoefficient.
   DeltaCoefficient& GetDeltaCoefficient() { return d; }

   void SetScale(real_t s) { d.SetScale(s); }
   void SetDirection(const Vector& d_);

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
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
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

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Evaluate the vector coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   /** @brief Evaluate the vector coefficient at all of the locations in the
       integration rule and write the vectors into the columns of matrix @a
       M. */
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationRule &ir) override;
};

typedef VectorCoefficient DiagonalMatrixCoefficient;

/// Base class for Matrix Coefficients that optionally depend on time and space.
class MatrixCoefficient
{
protected:
   int height, width;
   real_t time;
   bool symmetric;  // deprecated

public:
   /// Construct a dim x dim matrix coefficient.
   explicit MatrixCoefficient(int dim, bool symm=false)
   { height = width = dim; time = 0.; symmetric = symm; }

   /// Construct a h x w matrix coefficient.
   MatrixCoefficient(int h, int w, bool symm=false) :
      height(h), width(w), time(0.), symmetric(symm) { }

   /// Set the time for time dependent coefficients
   virtual void SetTime(real_t t) { time = t; }

   /// Get the time for time dependent coefficients
   real_t GetTime() { return time; }

   /// Get the height of the matrix.
   int GetHeight() const { return height; }

   /// Get the width of the matrix.
   int GetWidth() const { return width; }

   /// For backward compatibility get the width of the matrix.
   int GetVDim() const { return width; }

   /** @deprecated Use SymmetricMatrixCoefficient instead */
   bool IsSymmetric() const { return symmetric; }

   /** @brief Evaluate the matrix coefficient in the element described by @a T
       at the point @a ip, storing the result in @a K. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip) = 0;

   /// @brief Fill the QuadratureFunction @a qf by evaluating the coefficient at
   /// the quadrature points. The matrix will be transposed or not according to
   /// the boolean argument @a transpose.
   ///
   /// The @a vdim of the QuadratureFunction should be equal to the height times
   /// the width of the matrix.
   virtual void Project(QuadratureFunction &qf, bool transpose=false);

   /// (DEPRECATED) Evaluate a symmetric matrix coefficient.
   /** @brief Evaluate the upper triangular entries of the matrix coefficient
       in the symmetric case, similarly to Eval. Matrix entry (i,j) is stored
       in K[j - i + os_i] for 0 <= i <= j < width, os_0 = 0,
       os_{i+1} = os_i + width - i. That is, K = {M(0,0), ..., M(0,w-1),
       M(1,1), ..., M(1,w-1), ..., M(w-1,w-1) with w = width.
       @deprecated Use Eval() instead. */
   virtual void EvalSymmetric(Vector &K, ElementTransformation &T,
                              const IntegrationPoint &ip)
   { mfem_error("MatrixCoefficient::EvalSymmetric"); }

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
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override { M = mat; }
   /// Return a reference to the constant matrix.
   const DenseMatrix& GetMatrix() { return mat; }
};


/** @brief A piecewise matrix-valued coefficient with the pieces keyed off the
    element attribute numbers.

    A value of zero will be returned for any missing attribute numbers.

    This object will not assume ownership of any MatrixCoefficient
    objects passed to it. Consequently, the caller must ensure that
    the individual MatrixCoefficient objects are not deleted while
    this PWMatrixCoefficient is still in use.

    \note The keys may either be domain attribute numbers or boundary
    attribute numbers. If the PWMatrixCoefficient is used with a
    domain integrator the keys are assumed to be domain attribute
    numbers. Similarly, if the PWMatrixCoefficient is used with a
    boundary integrator the keys are assumed to be boundary attribute
    numbers.
*/
class PWMatrixCoefficient : public MatrixCoefficient
{
private:
   /** Internal data structure to store pointers to the appropriate
       coefficients for different regions of the mesh. The keys used
       in the map are the mesh attribute numbers (either element
       attribute or boundary element attribute depending upon
       context). The values returned for any missing attributes will
       be zero. The coefficient pointers may be NULL in which case a
       value of zero is returned.

       The MatrixCoefficient objects contained in this map are NOT
       owned by this PWMatrixCoefficient object. This means that they
       will not be deleted when this object is deleted also the caller
       must ensure that the various MatrixCoefficient objects are not
       deleted while this PWMatrixCoefficient is still needed.
   */
   std::map<int, MatrixCoefficient*> pieces;

   /** Convenience function to check for compatible array lengths,
       loop over the arrays, and add their attribute/MatrixCoefficient
       pairs to the internal data structure.
   */
   void InitMap(const Array<int> & attr,
                const Array<MatrixCoefficient*> & coefs);

public:

   /// Constructs a piecewise matrix coefficient of dimension dim by dim
   explicit PWMatrixCoefficient(int dim, bool symm = false)
      : MatrixCoefficient(dim, symm) {}

   /// Constructs a piecewise matrix coefficient of dimension h by w
   explicit PWMatrixCoefficient(int h, int w, bool symm = false)
      : MatrixCoefficient(h, w, symm) {}

   /// Construct the coefficient using arrays describing the pieces
   /** \param dim - size of the square matrix-valued result
       \param attr - an array of attribute numbers for each piece
       \param coefs - the corresponding array of MatrixCoefficient pointers
       \param symm - true if the result will be symmetric, false otherwise
       Any missing attributes or NULL coefficient pointers will result in a
       zero matrix being returned.

       \note Ownership of the MatrixCoefficient objects will NOT be
       transferred to this object.
   */
   PWMatrixCoefficient(int dim, const Array<int> & attr,
                       const Array<MatrixCoefficient*> & coefs,
                       bool symm=false)
      : MatrixCoefficient(dim, symm) { InitMap(attr, coefs); }

   /// Construct the coefficient using arrays describing the pieces
   /** \param h - height of the matrix-valued result
       \param w - width of the matrix-valued result
       \param attr - an array of attribute numbers for each piece
       \param coefs - the corresponding array of MatrixCoefficient pointers
       \param symm - true if the result will be symmetric, false otherwise
       Any missing attributes or NULL coefficient pointers will result in a
       zero matrix being returned for that attribute.

       \note Ownership of the MatrixCoefficient objects will NOT be
       transferred to this object.
   */
   PWMatrixCoefficient(int h, int w, const Array<int> & attr,
                       const Array<MatrixCoefficient*> & coefs,
                       bool symm=false)
      : MatrixCoefficient(h, w, symm) { InitMap(attr, coefs); }

   /// Set the time for time dependent coefficients
   void SetTime(real_t t) override;

   /// Replace a set of coefficients
   void UpdateCoefficients(const Array<int> & attr,
                           const Array<MatrixCoefficient*> & coefs)
   { InitMap(attr, coefs); }

   /// Replace a single coefficient for a particular attribute
   void UpdateCoefficient(int attr, MatrixCoefficient & coef);

   /// Remove a single MatrixCoefficient for a particular attribute
   void ZeroCoefficient(int attr)
   { pieces.erase(attr); }

   /// Evaluate the coefficient.
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/** @brief A matrix coefficient with an optional scalar coefficient multiplier
    \a q.  The matrix function can either be represented by a std function or
    a constant matrix provided when constructing this object.  */
class MatrixFunctionCoefficient : public MatrixCoefficient
{
private:
   std::function<void(const Vector &, DenseMatrix &)> Function;
   std::function<void(const Vector &, Vector &)> SymmFunction;  // deprecated
   std::function<void(const Vector &, real_t, DenseMatrix &)> TDFunction;

   Coefficient *Q;
   DenseMatrix mat;

public:
   /// Define a time-independent square matrix coefficient from a std function
   /** \param dim - the size of the matrix
       \param F - time-independent function
       \param q - optional scalar Coefficient to scale the matrix coefficient */
   MatrixFunctionCoefficient(int dim,
                             std::function<void(const Vector &, DenseMatrix &)> F,
                             Coefficient *q = nullptr)
      : MatrixCoefficient(dim), Function(std::move(F)), Q(q), mat(0)
   { }

   /// Define a constant matrix coefficient times a scalar Coefficient
   /** \param m - constant matrix
       \param q - optional scalar Coefficient to scale the matrix coefficient */
   MatrixFunctionCoefficient(const DenseMatrix &m, Coefficient &q)
      : MatrixCoefficient(m.Height(), m.Width()), Q(&q), mat(m)
   { }

   /** @brief Define a time-independent symmetric square matrix coefficient from
       a std function */
   /** \param dim - the size of the matrix
       \param SymmF - function used in EvalSymmetric
       \param q - optional scalar Coefficient to scale the matrix coefficient
       @deprecated Use another constructor without setting SymmFunction. */
   MatrixFunctionCoefficient(int dim,
                             std::function<void(const Vector &, Vector &)> SymmF,
                             Coefficient *q = NULL)
      : MatrixCoefficient(dim, true), SymmFunction(std::move(SymmF)), Q(q), mat(0)
   { }

   /// Define a time-dependent square matrix coefficient from a std function
   /** \param dim - the size of the matrix
       \param TDF - time-dependent function
       \param q - optional scalar Coefficient to scale the matrix coefficient */
   MatrixFunctionCoefficient(int dim,
                             std::function<void(const Vector &, real_t, DenseMatrix &)> TDF,
                             Coefficient *q = nullptr)
      : MatrixCoefficient(dim), TDFunction(std::move(TDF)), Q(q)
   { }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   /// (DEPRECATED) Evaluate the symmetric matrix coefficient at @a ip.
   /** @deprecated Use Eval() instead. */
   void EvalSymmetric(Vector &K, ElementTransformation &T,
                      const IntegrationPoint &ip) override;

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

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Get the coefficient located at (i,j) in the matrix.
   Coefficient* GetCoeff (int i, int j) { return Coeff[i*width+j]; }

   /** @brief Set the coefficient located at (i,j) in the matrix.  By default by
       default this will take ownership of the Coefficient passed in, but this
       can be overridden with the @a own parameter. */
   void Set(int i, int j, Coefficient * c, bool own=true);

   using MatrixCoefficient::Eval;

   /// Evaluate coefficient located at (i,j) in the matrix using integration
   /// point @a ip.
   real_t Eval(int i, int j, ElementTransformation &T, const IntegrationPoint &ip)
   { return Coeff[i*width+j] ? Coeff[i*width+j] -> Eval(T, ip, GetTime()) : 0.0; }

   /// Evaluate the matrix coefficient @a ip.
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   virtual ~MatrixArrayCoefficient();
};

/** @brief Matrix coefficient defined row-wise by an array of vector
    coefficients. Rows that are not set will evaluate to zero. The
    matrix coefficient is stored as an array indexing the rows of
    the matrix. */
class MatrixArrayVectorCoefficient : public MatrixCoefficient
{
private:
   Array<VectorCoefficient *> Coeff;
   Array<bool> ownCoeff;

public:
   /** @brief Construct a coefficient matrix of dimensions @a dim * @a dim. The
       actual coefficients still need to be added with Set(). */
   explicit MatrixArrayVectorCoefficient (int dim);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Get the vector coefficient located at the i-th row of the matrix
   VectorCoefficient* GetCoeff (int i) { return Coeff[i]; }

   /** @brief Set the coefficient located at the i-th row of the matrix.
       By this will take ownership of the Coefficient passed in, but this
       can be overridden with the @a own parameter. */
   void Set(int i, VectorCoefficient * c, bool own=true);

   using MatrixCoefficient::Eval;

   /// Evaluate coefficient located at the i-th row of the matrix using integration
   /// point @a ip.
   void Eval(int i, Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

   /// Evaluate the matrix coefficient @a ip.
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   virtual ~MatrixArrayVectorCoefficient();
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

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/// Coefficients based on sums, products, or other functions of coefficients.
///@{
/** @brief Scalar coefficient defined as the linear combination of two scalar
    coefficients or a scalar and a scalar coefficient */
class SumCoefficient : public Coefficient
{
private:
   real_t aConst;
   Coefficient * a;
   Coefficient * b;

   real_t alpha;
   real_t beta;

public:
   /// Constructor with one coefficient.  Result is alpha_ * A + beta_ * B
   SumCoefficient(real_t A, Coefficient &B,
                  real_t alpha_ = 1.0, real_t beta_ = 1.0)
      : aConst(A), a(NULL), b(&B), alpha(alpha_), beta(beta_) { }

   /// Constructor with two coefficients.  Result is alpha_ * A + beta_ * B.
   SumCoefficient(Coefficient &A, Coefficient &B,
                  real_t alpha_ = 1.0, real_t beta_ = 1.0)
      : aConst(0.0), a(&A), b(&B), alpha(alpha_), beta(beta_) { }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the first term in the linear combination as a constant
   void SetAConst(real_t A) { a = NULL; aConst = A; }
   /// Return the first term in the linear combination
   real_t GetAConst() const { return aConst; }

   /// Reset the first term in the linear combination
   void SetACoef(Coefficient &A) { a = &A; }
   /// Return the first term in the linear combination
   Coefficient * GetACoef() const { return a; }

   /// Reset the second term in the linear combination
   void SetBCoef(Coefficient &B) { b = &B; }
   /// Return the second term in the linear combination
   Coefficient * GetBCoef() const { return b; }

   /// Reset the factor in front of the first term in the linear combination
   void SetAlpha(real_t alpha_) { alpha = alpha_; }
   /// Return the factor in front of the first term in the linear combination
   real_t GetAlpha() const { return alpha; }

   /// Reset the factor in front of the second term in the linear combination
   void SetBeta(real_t beta_) { beta = beta_; }
   /// Return the factor in front of the second term in the linear combination
   real_t GetBeta() const { return beta; }

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return alpha * ((a == NULL ) ? aConst : a->Eval(T, ip) )
             + beta * b->Eval(T, ip);
   }
};


/// Base class for symmetric matrix coefficients that optionally depend on time and space.
class SymmetricMatrixCoefficient : public MatrixCoefficient
{
protected:

   /// Internal matrix used when evaluating this coefficient as a DenseMatrix.
   mutable DenseSymmetricMatrix mat_aux;
public:
   /// Construct a dim x dim matrix coefficient.
   explicit SymmetricMatrixCoefficient(int dimension)
      : MatrixCoefficient(dimension, true), mat_aux(height) { }

   /// Get the size of the matrix.
   int GetSize() const { return height; }

   /// @brief Fill the QuadratureFunction @a qf by evaluating the coefficient at
   /// the quadrature points.
   ///
   /// @note As opposed to MatrixCoefficient::Project, this function stores only
   /// the @a symmetric part of the matrix at each quadrature point.
   ///
   /// The @a vdim of the coefficient should be equal to height*(height+1)/2.
   virtual void ProjectSymmetric(QuadratureFunction &qf);

   /** @brief Evaluate the matrix coefficient in the element described by @a T
       at the point @a ip, storing the result as a symmetric matrix @a K. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual void Eval(DenseSymmetricMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip) = 0;

   /** @brief Evaluate the matrix coefficient in the element described by @a T
       at the point @a ip, storing the result as a dense matrix @a K. */
   /** This function allows the use of SymmetricMatrixCoefficient in situations
       where the symmetry is not taken advantage of.

       @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;


   /// @deprecated Return a reference to the internal matrix used when evaluating this coefficient as a DenseMatrix.
   MFEM_DEPRECATED const DenseSymmetricMatrix& GetMatrix() { return mat_aux; }

   virtual ~SymmetricMatrixCoefficient() { }
};


/// A matrix coefficient that is constant in space and time.
class SymmetricMatrixConstantCoefficient : public SymmetricMatrixCoefficient
{
private:
   DenseSymmetricMatrix mat;

public:
   ///Construct using matrix @a m for the constant.
   SymmetricMatrixConstantCoefficient(const DenseSymmetricMatrix &m)
      : SymmetricMatrixCoefficient(m.Height()), mat(m) { }
   using SymmetricMatrixCoefficient::Eval;
   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseSymmetricMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override { M = mat; }

   /// Return a reference to the constant matrix.
   const DenseSymmetricMatrix& GetMatrix() { return mat; }

};


/** @brief A matrix coefficient with an optional scalar coefficient multiplier
    \a q.  The matrix function can either be represented by a std function or
    a constant matrix provided when constructing this object.  */
class SymmetricMatrixFunctionCoefficient : public SymmetricMatrixCoefficient
{
private:
   std::function<void(const Vector &, DenseSymmetricMatrix &)> Function;
   std::function<void(const Vector &, real_t, DenseSymmetricMatrix &)> TDFunction;

   Coefficient *Q;
   DenseSymmetricMatrix mat;

public:
   /// Define a time-independent symmetric matrix coefficient from a std function
   /** \param dim - the size of the matrix
       \param F - time-independent function
       \param q - optional scalar Coefficient to scale the matrix coefficient */
   SymmetricMatrixFunctionCoefficient(int dim,
                                      std::function<void(const Vector &, DenseSymmetricMatrix &)> F,
                                      Coefficient *q = nullptr)
      : SymmetricMatrixCoefficient(dim), Function(std::move(F)), Q(q), mat(0)
   { }

   /// Define a constant matrix coefficient times a scalar Coefficient
   /** \param m - constant matrix
       \param q - optional scalar Coefficient to scale the matrix coefficient */
   SymmetricMatrixFunctionCoefficient(const DenseSymmetricMatrix &m,
                                      Coefficient &q)
      : SymmetricMatrixCoefficient(m.Height()), Q(&q), mat(m)
   { }

   /// Define a time-dependent square matrix coefficient from a std function
   /** \param dim - the size of the matrix
       \param TDF - time-dependent function
       \param q - optional scalar Coefficient to scale the matrix coefficient */
   SymmetricMatrixFunctionCoefficient(int dim,
                                      std::function<void(const Vector &, real_t, DenseSymmetricMatrix &)> TDF,
                                      Coefficient *q = nullptr)
      : SymmetricMatrixCoefficient(dim), TDFunction(std::move(TDF)), Q(q)
   { }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   using SymmetricMatrixCoefficient::Eval;
   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseSymmetricMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   virtual ~SymmetricMatrixFunctionCoefficient() { }
};


/** @brief Scalar coefficient defined as the product of two scalar coefficients
    or a scalar and a scalar coefficient. */
class ProductCoefficient : public Coefficient
{
private:
   real_t aConst;
   Coefficient * a;
   Coefficient * b;

public:
   /// Constructor with one coefficient.  Result is A * B.
   ProductCoefficient(real_t A, Coefficient &B)
      : aConst(A), a(NULL), b(&B) { }

   /// Constructor with two coefficients.  Result is A * B.
   ProductCoefficient(Coefficient &A, Coefficient &B)
      : aConst(0.0), a(&A), b(&B) { }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the first term in the product as a constant
   void SetAConst(real_t A) { a = NULL; aConst = A; }
   /// Return the first term in the product
   real_t GetAConst() const { return aConst; }

   /// Reset the first term in the product
   void SetACoef(Coefficient &A) { a = &A; }
   /// Return the first term in the product
   Coefficient * GetACoef() const { return a; }

   /// Reset the second term in the product
   void SetBCoef(Coefficient &B) { b = &B; }
   /// Return the second term in the product
   Coefficient * GetBCoef() const { return b; }

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   { return ((a == NULL ) ? aConst : a->Eval(T, ip) ) * b->Eval(T, ip); }
};

/** @brief Scalar coefficient defined as the ratio of two scalars where one or
    both scalars are scalar coefficients. */
class RatioCoefficient : public Coefficient
{
private:
   real_t aConst;
   real_t bConst;
   Coefficient * a;
   Coefficient * b;

public:
   /** Initialize a coefficient which returns A / B where @a A is a
       constant and @a B is a scalar coefficient */
   RatioCoefficient(real_t A, Coefficient &B)
      : aConst(A), bConst(1.0), a(NULL), b(&B) { }
   /** Initialize a coefficient which returns A / B where @a A and @a B are both
       scalar coefficients */
   RatioCoefficient(Coefficient &A, Coefficient &B)
      : aConst(0.0), bConst(1.0), a(&A), b(&B) { }
   /** Initialize a coefficient which returns A / B where @a A is a
       scalar coefficient and @a B is a constant */
   RatioCoefficient(Coefficient &A, real_t B)
      : aConst(0.0), bConst(B), a(&A), b(NULL) { }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the numerator in the ratio as a constant
   void SetAConst(real_t A) { a = NULL; aConst = A; }
   /// Return the numerator of the ratio
   real_t GetAConst() const { return aConst; }

   /// Reset the denominator in the ratio as a constant
   void SetBConst(real_t B) { b = NULL; bConst = B; }
   /// Return the denominator of the ratio
   real_t GetBConst() const { return bConst; }

   /// Reset the numerator in the ratio
   void SetACoef(Coefficient &A) { a = &A; }
   /// Return the numerator of the ratio
   Coefficient * GetACoef() const { return a; }

   /// Reset the denominator in the ratio
   void SetBCoef(Coefficient &B) { b = &B; }
   /// Return the denominator of the ratio
   Coefficient * GetBCoef() const { return b; }

   /// Evaluate the coefficient
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      real_t den = (b == NULL ) ? bConst : b->Eval(T, ip);
      MFEM_ASSERT(den != 0.0, "Division by zero in RatioCoefficient");
      return ((a == NULL ) ? aConst : a->Eval(T, ip) ) / den;
   }
};

/// Scalar coefficient defined as a scalar raised to a power
class PowerCoefficient : public Coefficient
{
private:
   Coefficient * a;

   real_t p;

public:
   /// Construct with a coefficient and a constant power @a p_.  Result is A^p.
   PowerCoefficient(Coefficient &A, real_t p_)
      : a(&A), p(p_) { }

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the base coefficient
   void SetACoef(Coefficient &A) { a = &A; }
   /// Return the base coefficient
   Coefficient * GetACoef() const { return a; }

   /// Reset the exponent
   void SetExponent(real_t p_) { p = p_; }
   /// Return the exponent
   real_t GetExponent() const { return p; }

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
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
   /// Construct with the two vector coefficients.  Result is $ A \cdot B $.
   InnerProductCoefficient(VectorCoefficient &A, VectorCoefficient &B);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the first vector in the inner product
   void SetACoef(VectorCoefficient &A) { a = &A; }
   /// Return the first vector coefficient in the inner product
   VectorCoefficient * GetACoef() const { return a; }

   /// Reset the second vector in the inner product
   void SetBCoef(VectorCoefficient &B) { b = &B; }
   /// Return the second vector coefficient in the inner product
   VectorCoefficient * GetBCoef() const { return b; }

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
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
   /// Constructor with two vector coefficients.  Result is $ A_x B_y - A_y * B_x; $.
   VectorRotProductCoefficient(VectorCoefficient &A, VectorCoefficient &B);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the first vector in the product
   void SetACoef(VectorCoefficient &A) { a = &A; }
   /// Return the first vector of the product
   VectorCoefficient * GetACoef() const { return a; }

   /// Reset the second vector in the product
   void SetBCoef(VectorCoefficient &B) { b = &B; }
   /// Return the second vector of the product
   VectorCoefficient * GetBCoef() const { return b; }

   /// Evaluate the coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
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

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the matrix coefficient
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the matrix coefficient
   MatrixCoefficient * GetACoef() const { return a; }

   /// Evaluate the determinant coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// Scalar coefficient defined as the trace of a matrix coefficient
class TraceCoefficient : public Coefficient
{
private:
   MatrixCoefficient * a;

   mutable DenseMatrix ma;

public:
   /// Construct with the matrix.
   TraceCoefficient(MatrixCoefficient &A);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the matrix coefficient
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the matrix coefficient
   MatrixCoefficient * GetACoef() const { return a; }

   /// Evaluate the trace coefficient at @a ip.
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override;
};

/// Vector coefficient defined as the linear combination of two vectors
class VectorSumCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * ACoef;
   VectorCoefficient * BCoef;

   Vector A;
   Vector B;

   Coefficient * alphaCoef;
   Coefficient * betaCoef;

   real_t alpha;
   real_t beta;

   mutable Vector va;

public:
   /** Constructor with no coefficients.
       To be used with the various "Set" methods */
   VectorSumCoefficient(int dim);

   /** Constructor with two vector coefficients.
       Result is alpha_ * A + beta_ * B */
   VectorSumCoefficient(VectorCoefficient &A, VectorCoefficient &B,
                        real_t alpha_ = 1.0, real_t beta_ = 1.0);

   /** Constructor with scalar coefficients.
       Result is alpha_ * A_ + beta_ * B_ */
   VectorSumCoefficient(VectorCoefficient &A_, VectorCoefficient &B_,
                        Coefficient &alpha_, Coefficient &beta_);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the first vector coefficient
   void SetACoef(VectorCoefficient &A_) { ACoef = &A_; }
   /// Return the first vector coefficient
   VectorCoefficient * GetACoef() const { return ACoef; }

   /// Reset the second vector coefficient
   void SetBCoef(VectorCoefficient &B_) { BCoef = &B_; }
   /// Return the second vector coefficient
   VectorCoefficient * GetBCoef() const { return BCoef; }

   /// Reset the factor in front of the first vector coefficient
   void SetAlphaCoef(Coefficient &A_) { alphaCoef = &A_; }
   /// Return the factor in front of the first vector coefficient
   Coefficient * GetAlphaCoef() const { return alphaCoef; }

   /// Reset the factor in front of the second vector coefficient
   void SetBetaCoef(Coefficient &B_) { betaCoef = &B_; }
   /// Return the factor in front of the second vector coefficient
   Coefficient * GetBetaCoef() const { return betaCoef; }

   /// Reset the first vector as a constant
   void SetA(const Vector &A_) { A = A_; ACoef = NULL; }
   /// Return the first vector constant
   const Vector & GetA() const { return A; }

   /// Reset the second vector as a constant
   void SetB(const Vector &B_) { B = B_; BCoef = NULL; }
   /// Return the second vector constant
   const Vector & GetB() const { return B; }

   /// Reset the factor in front of the first vector coefficient as a constant
   void SetAlpha(real_t alpha_) { alpha = alpha_; alphaCoef = NULL; }
   /// Return the factor in front of the first vector coefficient
   real_t GetAlpha() const { return alpha; }

   /// Reset the factor in front of the second vector coefficient as a constant
   void SetBeta(real_t beta_) { beta = beta_; betaCoef = NULL; }
   /// Return the factor in front of the second vector coefficient
   real_t GetBeta() const { return beta; }

   /// Evaluate the coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;
   using VectorCoefficient::Eval;
};

/// Vector coefficient defined as a product of scalar and vector coefficients.
class ScalarVectorProductCoefficient : public VectorCoefficient
{
private:
   real_t aConst;
   Coefficient * a;
   VectorCoefficient * b;

public:
   /// Constructor with constant and vector coefficient.  Result is A * B.
   ScalarVectorProductCoefficient(real_t A, VectorCoefficient &B);

   /// Constructor with two coefficients.  Result is A * B.
   ScalarVectorProductCoefficient(Coefficient &A, VectorCoefficient &B);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the scalar factor as a constant
   void SetAConst(real_t A) { a = NULL; aConst = A; }
   /// Return the scalar factor
   real_t GetAConst() const { return aConst; }

   /// Reset the scalar factor
   void SetACoef(Coefficient &A) { a = &A; }
   /// Return the scalar factor
   Coefficient * GetACoef() const { return a; }

   /// Reset the vector factor
   void SetBCoef(VectorCoefficient &B) { b = &B; }
   /// Return the vector factor
   VectorCoefficient * GetBCoef() const { return b; }

   /// Evaluate the coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;
   using VectorCoefficient::Eval;
};

/// Vector coefficient defined as a normalized vector field (returns v/|v|)
class NormalizedVectorCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * a;

   real_t tol;

public:
   /** @brief Return a vector normalized to a length of one

       This class evaluates the vector coefficient @a A and, if |A| > @a tol,
       returns the normalized vector A / |A|.  If |A| <= @a tol, the zero
       vector is returned.
   */
   NormalizedVectorCoefficient(VectorCoefficient &A, real_t tol = 1e-6);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the vector coefficient
   void SetACoef(VectorCoefficient &A) { a = &A; }
   /// Return the vector coefficient
   VectorCoefficient * GetACoef() const { return a; }

   /// Evaluate the coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;
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

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the first term in the product
   void SetACoef(VectorCoefficient &A) { a = &A; }
   /// Return the first term in the product
   VectorCoefficient * GetACoef() const { return a; }

   /// Reset the second term in the product
   void SetBCoef(VectorCoefficient &B) { b = &B; }
   /// Return the second term in the product
   VectorCoefficient * GetBCoef() const { return b; }

   /// Evaluate the coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;
   using VectorCoefficient::Eval;
};

/** @brief Vector coefficient defined as a product of a matrix coefficient and
    a vector coefficient. */
class MatrixVectorProductCoefficient : public VectorCoefficient
{
private:
   MatrixCoefficient * a;
   VectorCoefficient * b;

   mutable DenseMatrix ma;
   mutable Vector vb;

public:
   /// Constructor with two coefficients.  Result is A*B.
   MatrixVectorProductCoefficient(MatrixCoefficient &A, VectorCoefficient &B);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the matrix coefficient
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the matrix coefficient
   MatrixCoefficient * GetACoef() const { return a; }

   /// Reset the vector coefficient
   void SetBCoef(VectorCoefficient &B) { b = &B; }
   /// Return the vector coefficient
   VectorCoefficient * GetBCoef() const { return b; }

   /// Evaluate the vector coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;
   using VectorCoefficient::Eval;
};

/// Convenient alias for the MatrixVectorProductCoefficient
typedef MatrixVectorProductCoefficient MatVecCoefficient;

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
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/// Matrix coefficient defined as the linear combination of two matrices
class MatrixSumCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient * a;
   MatrixCoefficient * b;

   real_t alpha;
   real_t beta;

   mutable DenseMatrix ma;

public:
   /// Construct with the two coefficients.  Result is alpha_ * A + beta_ * B.
   MatrixSumCoefficient(MatrixCoefficient &A, MatrixCoefficient &B,
                        real_t alpha_ = 1.0, real_t beta_ = 1.0);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the first matrix coefficient
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the first matrix coefficient
   MatrixCoefficient * GetACoef() const { return a; }

   /// Reset the second matrix coefficient
   void SetBCoef(MatrixCoefficient &B) { b = &B; }
   /// Return the second matrix coefficient
   MatrixCoefficient * GetBCoef() const { return b; }

   /// Reset the factor in front of the first matrix coefficient
   void SetAlpha(real_t alpha_) { alpha = alpha_; }
   /// Return the factor in front of the first matrix coefficient
   real_t GetAlpha() const { return alpha; }

   /// Reset the factor in front of the second matrix coefficient
   void SetBeta(real_t beta_) { beta = beta_; }
   /// Return the factor in front of the second matrix coefficient
   real_t GetBeta() const { return beta; }

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/// Matrix coefficient defined as the product of two matrices
class MatrixProductCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient * a;
   MatrixCoefficient * b;

   mutable DenseMatrix ma;
   mutable DenseMatrix mb;

public:
   /// Construct with the two coefficients.  Result is A * B.
   MatrixProductCoefficient(MatrixCoefficient &A, MatrixCoefficient &B);

   /// Reset the first matrix coefficient
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the first matrix coefficient
   MatrixCoefficient * GetACoef() const { return a; }

   /// Reset the second matrix coefficient
   void SetBCoef(MatrixCoefficient &B) { b = &B; }
   /// Return the second matrix coefficient
   MatrixCoefficient * GetBCoef() const { return b; }

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/** @brief Matrix coefficient defined as a product of a scalar coefficient and a
    matrix coefficient.*/
class ScalarMatrixProductCoefficient : public MatrixCoefficient
{
private:
   real_t aConst;
   Coefficient * a;
   MatrixCoefficient * b;

public:
   /// Constructor with one coefficient.  Result is A*B.
   ScalarMatrixProductCoefficient(real_t A, MatrixCoefficient &B);

   /// Constructor with two coefficients.  Result is A*B.
   ScalarMatrixProductCoefficient(Coefficient &A, MatrixCoefficient &B);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the scalar factor as a constant
   void SetAConst(real_t A) { a = NULL; aConst = A; }
   /// Return the scalar factor
   real_t GetAConst() const { return aConst; }

   /// Reset the scalar factor
   void SetACoef(Coefficient &A) { a = &A; }
   /// Return the scalar factor
   Coefficient * GetACoef() const { return a; }

   /// Reset the matrix factor
   void SetBCoef(MatrixCoefficient &B) { b = &B; }
   /// Return the matrix factor
   MatrixCoefficient * GetBCoef() const { return b; }

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/// Matrix coefficient defined as the transpose of a matrix coefficient
class TransposeMatrixCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient * a;

public:
   /// Construct with the matrix coefficient.  Result is $ A^T $.
   TransposeMatrixCoefficient(MatrixCoefficient &A);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the matrix coefficient
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the matrix coefficient
   MatrixCoefficient * GetACoef() const { return a; }

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/// Matrix coefficient defined as the inverse of a matrix coefficient.
class InverseMatrixCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient * a;

public:
   /// Construct with the matrix coefficient.  Result is $ A^{-1} $.
   InverseMatrixCoefficient(MatrixCoefficient &A);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the matrix coefficient
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the matrix coefficient
   MatrixCoefficient * GetACoef() const { return a; }

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/// Matrix coefficient defined as the exponential of a matrix coefficient.
class ExponentialMatrixCoefficient : public MatrixCoefficient
{
private:
   MatrixCoefficient * a;

public:
   /// Construct the matrix coefficient.  Result is $ \exp(A) $.
   ExponentialMatrixCoefficient(MatrixCoefficient &A);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the matrix coefficient
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the matrix coefficient
   MatrixCoefficient * GetACoef() const { return a; }

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
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
   /// Construct with two vector coefficients.  Result is $ A B^T $.
   OuterProductCoefficient(VectorCoefficient &A, VectorCoefficient &B);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the first vector in the outer product
   void SetACoef(VectorCoefficient &A) { a = &A; }
   /// Return the first vector coefficient in the outer product
   VectorCoefficient * GetACoef() const { return a; }

   /// Reset the second vector in the outer product
   void SetBCoef(VectorCoefficient &B) { b = &B; }
   /// Return the second vector coefficient in the outer product
   VectorCoefficient * GetBCoef() const { return b; }

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

/** @brief Matrix coefficient defined as -a k x k x, for a vector k and scalar a

    This coefficient returns $a * (|k|^2 I - k \otimes k)$, where I is
    the identity matrix and $\otimes$ indicates the outer product.  This
    can be evaluated for vectors of any dimension but in three
    dimensions it corresponds to computing the cross product with k twice.
*/
class CrossCrossCoefficient : public MatrixCoefficient
{
private:
   real_t aConst;
   Coefficient * a;
   VectorCoefficient * k;

   mutable Vector vk;

public:
   CrossCrossCoefficient(real_t A, VectorCoefficient &K);
   CrossCrossCoefficient(Coefficient &A, VectorCoefficient &K);

   /// Set the time for internally stored coefficients
   void SetTime(real_t t) override;

   /// Reset the scalar factor as a constant
   void SetAConst(real_t A) { a = NULL; aConst = A; }
   /// Return the scalar factor
   real_t GetAConst() const { return aConst; }

   /// Reset the scalar factor
   void SetACoef(Coefficient &A) { a = &A; }
   /// Return the scalar factor
   Coefficient * GetACoef() const { return a; }

   /// Reset the vector factor
   void SetKCoef(VectorCoefficient &K) { k = &K; }
   /// Return the vector factor
   VectorCoefficient * GetKCoef() const { return k; }

   /// Evaluate the matrix coefficient at @a ip.
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};
///@}

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
   VectorQuadratureFunctionCoefficient(const QuadratureFunction &qf);

   /** Set the starting index within the QuadFunc that'll be used to project
       outwards as well as the corresponding length. The projected length should
       have the bounds of 1 <= length <= (length QuadFunc - index). */
   void SetComponent(int index_, int length_);

   const QuadratureFunction& GetQuadFunction() const { return QuadF; }

   using VectorCoefficient::Eval;
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   void Project(QuadratureFunction &qf) override;

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
   QuadratureFunctionCoefficient(const QuadratureFunction &qf);

   const QuadratureFunction& GetQuadFunction() const { return QuadF; }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;

   void Project(QuadratureFunction &qf) override;

   virtual ~QuadratureFunctionCoefficient() { }
};

/// Flags that determine what storage optimizations to use in CoefficientVector
enum class CoefficientStorage : int
{
   FULL = 0, ///< Store the coefficient as a full QuadratureFunction.
   CONSTANTS = 1 << 0, ///< Store constants using only @a vdim entries.
   SYMMETRIC = 1 << 1, ///< Store the triangular part of symmetric matrices.
   COMPRESSED = CONSTANTS | SYMMETRIC ///< Enable all above compressions.
};

inline CoefficientStorage operator|(CoefficientStorage a, CoefficientStorage b)
{
   return CoefficientStorage(int(a) | int(b));
}

inline int operator&(CoefficientStorage a, CoefficientStorage b)
{
   return int(a) & int(b);
}


/// @brief Class to represent a coefficient evaluated at quadrature points.
///
/// In the general case, a CoefficientVector is the same as a QuadratureFunction
/// with a coefficient projected onto it.
///
/// This class allows for some "compression" of the coefficient data, according
/// to the storage flags given by CoefficientStorage. For example, constant
/// coefficients can be stored using only @a vdim values, and symmetric matrices
/// can be stored using e.g. the upper triangular part of the matrix.
class CoefficientVector : public Vector
{
protected:
   CoefficientStorage storage; ///< Storage optimizations (see CoefficientStorage).
   int vdim; ///< Number of values per quadrature point.
   QuadratureSpaceBase &qs; ///< Associated QuadratureSpaceBase.
   QuadratureFunction *qf; ///< Internal QuadratureFunction (owned, may be NULL).
public:
   /// Create an empty CoefficientVector.
   CoefficientVector(QuadratureSpaceBase &qs_,
                     CoefficientStorage storage_ = CoefficientStorage::FULL);

   /// @brief Create a CoefficientVector from the given Coefficient and
   /// QuadratureSpaceBase.
   ///
   /// If @a coeff is NULL, it will be interpreted as a constant with value one.
   /// @sa CoefficientStorage for a description of @a storage_.
   CoefficientVector(Coefficient *coeff, QuadratureSpaceBase &qs,
                     CoefficientStorage storage_ = CoefficientStorage::FULL);

   /// @brief Create a CoefficientVector from the given Coefficient and
   /// QuadratureSpaceBase.
   ///
   /// @sa CoefficientStorage for a description of @a storage_.
   CoefficientVector(Coefficient &coeff, QuadratureSpaceBase &qs,
                     CoefficientStorage storage_ = CoefficientStorage::FULL);

   /// @brief Create a CoefficientVector from the given VectorCoefficient and
   /// QuadratureSpaceBase.
   ///
   /// @sa CoefficientStorage for a description of @a storage_.
   CoefficientVector(VectorCoefficient &coeff, QuadratureSpaceBase &qs,
                     CoefficientStorage storage_ = CoefficientStorage::FULL);

   /// @brief Create a CoefficientVector from the given MatrixCoefficient and
   /// QuadratureSpaceBase.
   ///
   /// @sa CoefficientStorage for a description of @a storage_.
   CoefficientVector(MatrixCoefficient &coeff, QuadratureSpaceBase &qs,
                     CoefficientStorage storage_ = CoefficientStorage::FULL);

   /// @brief Evaluate the given Coefficient at the quadrature points defined by
   /// @ref qs.
   void Project(Coefficient &coeff);

   /// @brief Evaluate the given VectorCoefficient at the quadrature points
   /// defined by @ref qs.
   ///
   /// @sa CoefficientVector for a description of the @a compress argument.
   void Project(VectorCoefficient &coeff);

   /// @brief Evaluate the given MatrixCoefficient at the quadrature points
   /// defined by @ref qs.
   ///
   /// @sa CoefficientVector for a description of the @a compress argument.
   void Project(MatrixCoefficient &coeff, bool transpose=false);

   /// @brief Project the transpose of @a coeff.
   ///
   /// @sa Project(MatrixCoefficient&, QuadratureSpace&, bool, bool)
   void ProjectTranspose(MatrixCoefficient &coeff);

   /// Make this vector a reference to the given QuadratureFunction.
   void MakeRef(const QuadratureFunction &qf_);

   /// Set this vector to the given constant.
   void SetConstant(real_t constant);

   /// Set this vector to the given constant vector.
   void SetConstant(const Vector &constant);

   /// Set this vector to the given constant matrix.
   void SetConstant(const DenseMatrix &constant);

   /// Set this vector to the given constant symmetric matrix.
   void SetConstant(const DenseSymmetricMatrix &constant);

   /// Return the number of values per quadrature point.
   int GetVDim() const;

   ~CoefficientVector();
};

/** @brief Compute the Lp norm of a function f.
    $ \| f \|_{Lp} = ( \int_\Omega | f |^p d\Omega)^{1/p} $ */
real_t ComputeLpNorm(real_t p, Coefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[]);

/** @brief Compute the Lp norm of a vector function f = {f_i}_i=1...N.
    $ \| f \|_{Lp} = ( \sum_i \| f_i \|_{Lp}^p )^{1/p} $ */
real_t ComputeLpNorm(real_t p, VectorCoefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[]);

#ifdef MFEM_USE_MPI
/** @brief Compute the global Lp norm of a function f.
    $ \| f \|_{Lp} = ( \int_\Omega | f |^p d\Omega)^{1/p} $ */
real_t ComputeGlobalLpNorm(real_t p, Coefficient &coeff, ParMesh &pmesh,
                           const IntegrationRule *irs[]);

/** @brief Compute the global Lp norm of a vector function f = {f_i}_i=1...N.
    $ \| f \|_{Lp} = ( \sum_i \| f_i \|_{Lp}^p )^{1/p} $ */
real_t ComputeGlobalLpNorm(real_t p, VectorCoefficient &coeff, ParMesh &pmesh,
                           const IntegrationRule *irs[]);
#endif

}

#endif
