// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_COMPLEX_COEFFICIENT
#define MFEM_COMPLEX_COEFFICIENT

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "coefficient.hpp"
#include "intrules.hpp"
#include "eltrans.hpp"

namespace mfem
{

class ComplexCoefficient;
class ComplexVectorCoefficient;

/// Standard Coefficient which returns the real part of a ComplexCoefficient
class RealPartCoefficient : public Coefficient
{
private:
   ComplexCoefficient &complex_coef_;

public:
   RealPartCoefficient(ComplexCoefficient & complex_coef)
      : complex_coef_(complex_coef) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

/// Standard Coefficient which returns the imaginary part of a
/// ComplexCoefficient
class ImagPartCoefficient : public Coefficient
{
private:
   ComplexCoefficient &complex_coef_;

public:
   ImagPartCoefficient(ComplexCoefficient & complex_coef)
      : complex_coef_(complex_coef) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip);
};

typedef ImagPartCoefficient ImaginaryPartCoefficient;

class RealPartVectorCoefficient : public VectorCoefficient
{
private:
   ComplexVectorCoefficient &complex_vcoef_;
   mutable ComplexVector val_;

public:
   RealPartVectorCoefficient(ComplexVectorCoefficient & complex_vcoef);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);
};

class ImagPartVectorCoefficient : public VectorCoefficient
{
private:
   ComplexVectorCoefficient &complex_vcoef_;
   mutable ComplexVector val_;

public:
   ImagPartVectorCoefficient(ComplexVectorCoefficient & complex_vcoef);

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);
};

typedef ImagPartVectorCoefficient ImaginaryPartVectorCoefficient;

/** @brief Base class ComplexCoefficients that optionally depend on space and
    time. These are used by the SesquilinearForm, ComplexLinearForm, and
    ComplexGridFunction classes to represent the physical coefficients in
    the PDEs that are being discretized. This class can also be used in a more
    general way to represent functions that don't necessarily belong to a FE
    space, e.g., to project onto ComplexGridFunctions to use as initial
    conditions, exact solutions, etc. See, e.g., ex22 for these uses. */
class ComplexCoefficient
{
protected:
   real_t time;

private:
   RealPartCoefficient re_part_coef_;
   ImagPartCoefficient im_part_coef_;

protected:
   Coefficient &real_coef_;
   Coefficient &imag_coef_;

public:

   ComplexCoefficient();
   ComplexCoefficient(Coefficient &c_r, Coefficient &c_i);

   /// Set the time for time dependent coefficients
   virtual void SetTime(real_t t)
   { time = t; real_coef_.SetTime(t); imag_coef_.SetTime(t); }

   /// Get the time for time dependent coefficients
   real_t GetTime() { return time; }

   /** @brief Evaluate the coefficient in the element described by @a T at the
       point @a ip. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual std::complex<real_t> Eval(ElementTransformation &T,
                                     const IntegrationPoint &ip);

   /** @brief Evaluate the coefficient in the element described by @a T at the
       point @a ip at time @a t. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   std::complex<real_t> Eval(ElementTransformation &T,
                             const IntegrationPoint &ip, real_t t)
   {
      SetTime(t);
      return Eval(T, ip);
   }

   /** @brief Access a standard Coefficient object reproducing the real part of
       the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its real part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual Coefficient & real() { return real_coef_; }

   /** @brief Access a standard Coefficient object reproducing the imaginary
       part of the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its imaginary part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual Coefficient & imag() { return imag_coef_; }

   virtual ~ComplexCoefficient() { }
};

/** @brief Base class ComplexVectorCoefficients that optionally depend
    on space and time. These are used by the SesquilinearForm,
    ComplexLinearForm, and ComplexGridFunction classes to represent
    the physical vector-valued coefficients in the PDEs that are being
    discretized. This class can also be used in a more general way to
    represent functions that don't necessarily belong to a FE space,
    e.g., to project onto ComplexGridFunctions to use as initial
    conditions, exact solutions, etc. See, e.g., ex22 for these
    uses. */
class ComplexVectorCoefficient
{
protected:
   int vdim;
   real_t time;

private:
   RealPartVectorCoefficient re_part_vcoef_;
   ImagPartVectorCoefficient im_part_vcoef_;

protected:
   VectorCoefficient &real_vcoef_;
   VectorCoefficient &imag_vcoef_;

   mutable Vector V_r_;
   mutable Vector V_i_;

public:
   ComplexVectorCoefficient(int vd)
      : vdim(vd), time(0.),
        re_part_vcoef_(*this), im_part_vcoef_(*this),
        real_vcoef_(re_part_vcoef_), imag_vcoef_(im_part_vcoef_)
   { }

   ComplexVectorCoefficient(VectorCoefficient &v_r, VectorCoefficient &v_i);


   /// Set the time for time dependent coefficients
   virtual void SetTime(real_t t)
   { time = t; real_vcoef_.SetTime(t); imag_vcoef_.SetTime(t); }

   /// Get the time for time dependent coefficients
   real_t GetTime() { return time; }

   /// Returns dimension of the vector.
   int GetVDim() { return vdim; }

   /** @brief Evaluate the vector coefficient in the element described by @a T
       at the point @a ip, storing the result in @a V. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   virtual void Eval(ComplexVector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

   /** @brief Evaluate the vector coefficient in the element described by @a T
       at the point @a ip at time @a t, storing the result in @a V. */
   /** @note When this method is called, the caller must make sure that the
       IntegrationPoint associated with @a T is the same as @a ip. This can be
       achieved by calling T.SetIntPoint(&ip). */
   void Eval(ComplexVector &V, ElementTransformation &T,
             const IntegrationPoint &ip, real_t t)
   {
      SetTime(t);
      Eval(V, T, ip);
   }

   /** @brief Access a standard Coefficient object reproducing the real part of
       the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its real part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual VectorCoefficient & real() { return real_vcoef_; }

   /** @brief Access a standard Coefficient object reproducing the imaginary
       part of the complex-valued field */
   /** @note By default this method returns an internal object which
       computes the complex value using the above Eval method and
       returns its imaginary part. Custom implementations may choose to
       override this method with a more efficient real-valued
       coefficient. */
   virtual VectorCoefficient & imag() { return imag_vcoef_; }

   virtual ~ComplexVectorCoefficient() { }
};

/// A complex-valued coefficient that is constant across space and time
class ComplexConstantCoefficient : public ComplexCoefficient
{
private:
   std::complex<real_t> val;

   ConstantCoefficient real_coef;
   ConstantCoefficient imag_coef;

public:
   ComplexConstantCoefficient(const std::complex<real_t> z);

   ComplexConstantCoefficient(real_t z_r, real_t z_i = 0.);

   std::complex<real_t> Eval(ElementTransformation &T,
                             const IntegrationPoint &ip) { return val; }
};

/// Complex-valued vector coefficient that is constant in space and time.
class ComplexVectorConstantCoefficient : public ComplexVectorCoefficient
{
private:
   ComplexVector vec;

public:
   /// Construct the coefficient with constant vector @a v.
   ComplexVectorConstantCoefficient(const ComplexVector &v)
      : ComplexVectorCoefficient(v.Size()), vec(v) { }

   /// Construct the coefficient with constant vector @a v.
   ComplexVectorConstantCoefficient(const Vector &v)
      : ComplexVectorCoefficient(v.Size()), vec(v) { }

   using ComplexVectorCoefficient::Eval;

   ///  Evaluate the vector coefficient at @a ip.
   void Eval(ComplexVector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override { V = vec; }

   /// Return a reference to the constant vector in this class.
   const ComplexVector& GetVec() const { return vec; }
};

/// A general complex-valued function coefficient
class ComplexFunctionCoefficient : public ComplexCoefficient
{
protected:
   std::function<std::complex<real_t>(const Vector &)> Function;
   std::function<std::complex<real_t>(const Vector &, real_t)> TDFunction;

public:
   /// Define a time-independent coefficient from a std function
   /** \param F time-independent std::function */
   ComplexFunctionCoefficient(std::function<std::complex<real_t>
                              (const Vector &)> F)
      : Function(std::move(F))
   { }

   /// Define a time-dependent coefficient from a std function
   /** \param TDF time-dependent function */
   ComplexFunctionCoefficient(std::function<std::complex<real_t>
                              (const Vector &, real_t)> TDF)
      : TDFunction(std::move(TDF))
   { }

   /// (DEPRECATED) Define a time-independent coefficient from a C-function
   /** @deprecated Use the method where the C-function, @a f, uses a const
       Vector argument instead of Vector. */
   MFEM_DEPRECATED ComplexFunctionCoefficient(std::complex<real_t>
                                              (*f)(Vector &))
   {
      // Cast first to (void*) to suppress a warning from newer version of
      // Clang when using -Wextra.
      Function = reinterpret_cast<std::complex<real_t>(*)
                 (const Vector&)>((void*)f);
      TDFunction = NULL;
   }

   /// (DEPRECATED) Define a time-dependent coefficient from a C-function
   /** @deprecated Use the method where the C-function, @a tdf, uses a const
       Vector argument instead of Vector. */
   MFEM_DEPRECATED ComplexFunctionCoefficient(std::complex<real_t>
                                              (*tdf)(Vector &, real_t))
   {
      Function = NULL;
      // Cast first to (void*) to suppress a warning from newer version of
      // Clang when using -Wextra.
      TDFunction =
         reinterpret_cast<std::complex<real_t>(*)(const Vector&,
                                                  real_t)>((void*)tdf);
   }

   /// Evaluate the coefficient at @a ip.
   std::complex<real_t> Eval(ElementTransformation &T,
                             const IntegrationPoint &ip) override;
};

/// A general vector function coefficient
class ComplexVectorFunctionCoefficient : public ComplexVectorCoefficient
{
private:
   std::function<void(const Vector &, ComplexVector &)> Function;
   std::function<void(const Vector &, real_t, ComplexVector &)> TDFunction;
   ComplexCoefficient *Q;

public:
   /// Define a time-independent complex-valued vector coefficient
   /// from a std function
   /** \param dim - the size of the vector
       \param F - time-independent function
       \param q - optional scalar Coefficient to scale the vector coefficient */
   ComplexVectorFunctionCoefficient(int dim,
                                    std::function<void(const Vector &,
                                                       ComplexVector &)> F,
                                    ComplexCoefficient *q = nullptr)
      : ComplexVectorCoefficient(dim), Function(std::move(F)), Q(q)
   { }

   /// Define a time-dependent complex-valued vector coefficient from
   /// a std function
   /** \param dim - the size of the vector
       \param TDF - time-dependent function
       \param q - optional scalar ComplexCoefficient to scale the vector coefficient */
   ComplexVectorFunctionCoefficient(int dim,
                                    std::function<void(const Vector &, real_t,
                                                       ComplexVector &)> TDF,
                                    ComplexCoefficient *q = nullptr)
      : ComplexVectorCoefficient(dim), TDFunction(std::move(TDF)), Q(q)
   { }

   using ComplexVectorCoefficient::Eval;
   /// Evaluate the vector coefficient at @a ip.
   void Eval(ComplexVector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   virtual ~ComplexVectorFunctionCoefficient() { }
};

} // end namespace mfem

#endif
