// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef FDUAL_H
#define FDUAL_H

#include <cmath>
#include <type_traits>

namespace mfem
{
namespace ad
{
/** The FDualNumber template class provides forward
  <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">automatic differentiation</a>
   implementation based on dual numbers. The derivative of an arbitrary
 function double f(double a)  can be obtained by replacing the double
 type for the return value and the argument a  with FDualNumber<double>, i.e.,
 FDualNumber<double> f(FDualNumber<double> a). The derivative is evaluated automatically
 by calling the function r=f(a). The value of the function is stored in r.pr
 and the derivative in r.du. These can be extracted by the corresponding
  methods real()/prim() and dual(). Internally, the function f can be
 composed of standard functions predefined for FDualNumber type. These consist
 of a large set of functions replicating the functionality of the standard
 math library, i.e., sin, cos, exp, log, ...

New functions (non-member) can be eaily added to the class. Example:
\code{.cpp}
template<typename tbase>
inline FDualNumber<tbase> cos(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(cos(f.real()), -f.dual() * sin(f.real()));
}
\endcode
The real part of the return value consists of the standard real value
 of the function, i.e., cos(f.real()).

The dual part of the return value consists of the first derivative of
 the function with respect to the real part of the argument -sin(f.reaf)
 multiplied with the dual part of the argument f.dual().
*/
template<typename tbase>
class FDualNumber
{
private:
   /// Real value
   tbase pr;
   /// Dual value holding derivative information
   tbase du;

public:
   /// Standard contructor - both values are set to zero.
   FDualNumber() : pr(0), du(0) {}

   /// The constructor utilized in nested definition of dual numbers.
   /// It is used for second and higher order derivatives.
   template<class fltyp,
            class = typename std::enable_if<std::is_arithmetic<fltyp>::value>::type>
   FDualNumber(fltyp &f) : pr(f), du(0)
   {}

   /// The constructor utilized in nested definition of dual numbers.
   /// It is used for second and higher order derivatives.
   template<class fltyp,
            class = typename std::enable_if<std::is_arithmetic<fltyp>::value>::type>
   FDualNumber(const fltyp &f) : pr(f), du(0)
   {}

   /// Standard constructor with user supplied input for both parts of the dual number.
   FDualNumber(tbase &pr_, tbase &du_) : pr(pr_), du(du_) {}

   /// Standard constructor with user supplied input for both parts of the dual number.
   FDualNumber(const tbase &pr_, const tbase &du_) : pr(pr_), du(du_) {}

   /// Standard constructor with user supplied dual number.
   FDualNumber(FDualNumber<tbase> &nm) : pr(nm.pr), du(nm.du) {}

   /// Standard constructor with user supplied dual number.
   FDualNumber(const FDualNumber<tbase> &nm) : pr(nm.pr), du(nm.du) {}

   /// Return the real value of the dual number.
   tbase prim() const { return pr; }

   /// Same as prim(). Return the real value of the dual number.
   tbase real() const { return pr; }

   /// Return the dual value of the dual number.
   tbase dual() const { return du; }

   /// Set the primal and the dual values.
   void set(const tbase &pr_, const tbase &du_)
   {
      pr = pr_;
      du = du_;
   }

   /// Set the primal value.
   void prim(const tbase &pr_) { pr = pr_; }

   /// Set the primal value.
   void real(const tbase &pr_) { pr = pr_; }

   /// Set the dual value.
   void dual(const tbase &du_) { du = du_; }

   /// Set the primal value.
   void setReal(const tbase &pr_) { pr = pr_; }

   /// Set the dual value.
   void setDual(const tbase &du_) { du = du_; }

   /// operator =
   FDualNumber<tbase> &operator=(tbase sc_)
   {
      pr = sc_;
      du = tbase(0);
      return *this;
   }

   /// operator +=
   FDualNumber<tbase> &operator+=(tbase sc_)
   {
      pr = pr + sc_;
      return *this;
   }

   /// operator -=
   FDualNumber<tbase> &operator-=(tbase sc_)
   {
      pr = pr - sc_;
      return *this;
   }

   /// operator *=
   FDualNumber<tbase> &operator*=(tbase sc_)
   {
      pr = pr * sc_;
      du = du * sc_;
      return *this;
   }

   /// operator /=
   FDualNumber<tbase> &operator/=(tbase sc_)
   {
      pr = pr / sc_;
      du = du / sc_;
      return *this;
   }

   /// operator =
   FDualNumber<tbase> &operator=(const FDualNumber<tbase> &f)
   {
      pr = f.real();
      du = f.dual();
      return *this;
   }

   /// operator +=
   FDualNumber<tbase> &operator+=(const FDualNumber<tbase> &f)
   {
      pr += f.real();
      du += f.dual();
      return *this;
   }

   /// operator -=
   FDualNumber<tbase> &operator-=(const FDualNumber<tbase> &f)
   {
      pr -= f.real();
      du -= f.dual();
      return *this;
   }

   /// operator *=
   FDualNumber<tbase> &operator*=(const FDualNumber<tbase> &f)
   {
      du = du * f.real();
      du = du + pr * f.dual();
      pr = pr * f.real();
      return *this;
   }

   /// operator /=
   FDualNumber<tbase> &operator/=(const FDualNumber<tbase> &f_)
   {
      pr = pr / f_.real();
      du = du - pr * f_.dual();
      du = du / f_.real();
      return *this;
   }
};

/// non-member functions
/// boolean operation ==
template<typename tbase>
inline bool operator==(const FDualNumber<tbase> &a1, const FDualNumber<tbase> &a2)
{
   return a1.real() == a2.real();
}

/// boolean operation ==
template<typename tbase>
inline bool operator==(tbase a, const FDualNumber<tbase> &f_)
{
   return a == f_.real();
}

/// boolean operation ==
template<typename tbase>
inline bool operator==(const FDualNumber<tbase> &a, tbase b)
{
   return a.real() == b;
}

/// boolean operation <
template<typename tbase>
inline bool operator<(const FDualNumber<tbase> &f1, const FDualNumber<tbase> &f2)
{
   return f1.real() < f2.real();
}

/// boolean operation <
template<typename tbase>
inline bool operator<(const FDualNumber<tbase> &f, tbase a)
{
   return f.real() < a;
}

/// boolean operation <
template<typename tbase>
inline bool operator<(tbase a, const FDualNumber<tbase> &f)
{
   return a < f.real();
}

/// boolean operation >
template<typename tbase>
inline bool operator>(const FDualNumber<tbase> &f1, const FDualNumber<tbase> &f2)
{
   return f1.real() > f2.real();
}

/// boolean operation >
template<typename tbase>
inline bool operator>(const FDualNumber<tbase> &f, tbase a)
{
   return f.real() > a;
}

/// boolean operation >
template<typename tbase>
inline bool operator>(tbase a, const FDualNumber<tbase> &f)
{
   return (a > f.real());
}

/// Negate the real and the dual parts.
template<typename tbase>
inline FDualNumber<tbase> operator-(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(-f.real(), -f.dual());
}

/// [dual number] - [base number]
template<typename tbase>
inline FDualNumber<tbase> operator-(const FDualNumber<tbase> &f, tbase a)
{
   return FDualNumber<tbase>(f.real() - a, f.dual());
}

/// [dual number<dual number>] - [base number]
template<typename tbase>
inline FDualNumber<FDualNumber<tbase>> operator-(const FDualNumber<FDualNumber<tbase>> &f, tbase a)
{
   return FDualNumber<FDualNumber<tbase>>(f.real() - a, f.dual());
}

/// [dual number] + [base number]
template<typename tbase>
inline FDualNumber<tbase> operator+(const FDualNumber<tbase> &f, tbase a)
{
   return FDualNumber<tbase>(f.real() + a, f.dual());
}

/// [dual number<dual number>] + [base number]
template<typename tbase>
inline FDualNumber<FDualNumber<tbase>> operator+(const FDualNumber<FDualNumber<tbase>> &f, tbase a)
{
   return FDualNumber<FDualNumber<tbase>>(f.real() + a, f.dual());
}

/// [dual number] * [base number]
template<typename tbase>
inline FDualNumber<tbase> operator*(const FDualNumber<tbase> &f, tbase a)
{
   return FDualNumber<tbase>(f.real() * a, f.dual() * a);
}

/// [dual number] / [base number]
template<typename tbase>
inline FDualNumber<tbase> operator/(const FDualNumber<tbase> &f, tbase a)
{
   return FDualNumber<tbase>(f.real() / a, f.dual() / a);
}

/// [dual number<dual number>] / [base number]
template<typename tbase>
inline FDualNumber<FDualNumber<tbase>> operator/(const FDualNumber<FDualNumber<tbase>> &f, tbase a)
{
   return FDualNumber<FDualNumber<tbase>>(f.real() / a, f.dual() / a);
}

/// [base number] + [dual number]
template<typename tbase>
inline FDualNumber<tbase> operator+(tbase a, const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(a + f.real(), f.dual());
}

/// [base number] + [dual number<dual number>]
template<typename tbase>
inline FDualNumber<FDualNumber<tbase>> operator+(tbase a, const FDualNumber<FDualNumber<tbase>> &f)
{
   return FDualNumber<FDualNumber<tbase>>(a + f.real(), f.dual());
}

/// [base number] - [dual number]
template<typename tbase>
inline FDualNumber<tbase> operator-(tbase a, const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(a - f.real(), -f.dual());
}

/// [base number] - [dual number<dual number>]
template<typename tbase>
inline FDualNumber<FDualNumber<tbase>> operator-(tbase a, const FDualNumber<FDualNumber<tbase>> &f)
{
   return FDualNumber<FDualNumber<tbase>>(a - f.real(), -f.dual());
}

/// [base number] * [dual number]
template<typename tbase>
inline FDualNumber<tbase> operator*(tbase a, const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(f.real() * a, f.dual() * a);
}

/// [base number] * [dual number<dual number>]
template<typename tbase>
inline FDualNumber<FDualNumber<tbase>> operator*(tbase a, const FDualNumber<FDualNumber<tbase>> &f)
{
   return FDualNumber<FDualNumber<tbase>>(f.real() * a, f.dual() * a);
}

/// [base number] / [dual number]
template<typename tbase>
inline FDualNumber<tbase> operator/(tbase a, const FDualNumber<tbase> &f)
{
   a = a / f.real();
   return FDualNumber<tbase>(a, -a * f.dual() / f.real());
}

/// [dual number] + [dual number]
template<typename tbase>
inline FDualNumber<tbase> operator+(const FDualNumber<tbase> &f1, const FDualNumber<tbase> &f2)
{
   return FDualNumber<tbase>(f1.real() + f2.real(), f1.dual() + f2.dual());
}

/// [dual number] - [dual number]
template<typename tbase>
inline FDualNumber<tbase> operator-(const FDualNumber<tbase> &f1, const FDualNumber<tbase> &f2)
{
   return FDualNumber<tbase>(f1.real() - f2.real(), f1.dual() - f2.dual());
}

/// [dual number] * [dual number]
template<typename tbase>
inline FDualNumber<tbase> operator*(const FDualNumber<tbase> &f1, const FDualNumber<tbase> &f2)
{
   return FDualNumber<tbase>(f1.real() * f2.real(),
                       f1.real() * f2.dual() + f1.dual() * f2.real());
}

/// [dual number] / [dual number]
template<typename tbase>
inline FDualNumber<tbase> operator/(const FDualNumber<tbase> &f1, const FDualNumber<tbase> &f2)
{
   tbase a = tbase(1) / f2.real();
   tbase b = f1.real() * a;
   return FDualNumber<tbase>(b, (f1.dual() - f2.dual() * b) * a);
}

/// acos([dual number])
template<typename tbase>
inline FDualNumber<tbase> acos(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(acos(f.real()),
                       -f.dual() / sqrt(tbase(1) - f.real() * f.real()));
}

/// acos([dual number<double>])
template<>
inline FDualNumber<double> acos(const FDualNumber<double> &f)
{
   return FDualNumber<double>(std::acos(f.real()),
                        -f.dual() / std::sqrt(double(1) - f.real() * f.real()));
}

/// asin([dual number])
template<typename tbase>
inline FDualNumber<tbase> asin(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(asin(f.real()),
                       f.dual() / sqrt(tbase(1) - f.real() * f.real()));
}

/// asin([dual number<double>])
template<>
inline FDualNumber<double> asin(const FDualNumber<double> &f)
{
   return FDualNumber<double>(std::asin(f.real()),
                        f.dual() / std::sqrt(double(1) - f.real() * f.real()));
}

/// atan([dual number])
template<typename tbase>
inline FDualNumber<tbase> atan(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(atan(f.real()),
                       f.dual() / (tbase(1) + f.real() * f.real()));
}

/// atan([dual number<double>])
template<>
inline FDualNumber<double> atan(const FDualNumber<double> &f)
{
   return FDualNumber<double>(std::atan(f.real()),
                        f.dual() / (double(1) + f.real() * f.real()));
}

/// cos([dual number])
template<typename tbase>
inline FDualNumber<tbase> cos(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(cos(f.real()), -f.dual() * sin(f.real()));
}

/// cos([dual number<double>])
template<>
inline FDualNumber<double> cos(const FDualNumber<double> &f)
{
   return FDualNumber<double>(std::cos(f.real()), -f.dual() * std::sin(f.real()));
}

/// cosh([dual number])
template<typename tbase>
inline FDualNumber<tbase> cosh(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(cosh(f.real()), f.dual() * sinh(f.real()));
}

/// cosh([dual number<double>])
template<>
inline FDualNumber<double> cosh(const FDualNumber<double> &f)
{
   return FDualNumber<double>(std::cosh(f.real()), f.dual() * std::sinh(f.real()));
}

/// exp([dual number])
template<typename tbase>
inline FDualNumber<tbase> exp(const FDualNumber<tbase> &f)
{
   tbase x = exp(f.real());
   return FDualNumber<tbase>(x, f.dual() * x);
}

/// exp([dual number<double>])
template<>
inline FDualNumber<double> exp(const FDualNumber<double> &f)
{
   double x = std::exp(f.real());
   return FDualNumber<double>(x, f.dual() * x);
}

/// log([dual number])
template<typename tbase>
inline FDualNumber<tbase> log(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(log(f.real()), f.dual() / f.real());
}

/// log([dual number<double>])
template<>
inline FDualNumber<double> log(const FDualNumber<double> &f)
{
   return FDualNumber<double>(std::log(f.real()), f.dual() / f.real());
}

/// log10([dual number])
template<typename tbase>
inline FDualNumber<tbase> log10(const FDualNumber<tbase> &f)
{
   return log(f) / log(tbase(10));
}

/// log10([dual number<double>])
template<>
inline FDualNumber<double> log10(const FDualNumber<double> &f)
{
   return log(f) / std::log(double(10));
}

/// pow([dual number],[dual number])
template<typename tbase>
inline FDualNumber<tbase> pow(const FDualNumber<tbase> &a, const FDualNumber<tbase> &b)
{
   return exp(log(a) * b);
}

/// pow([dual number], [base number])
template<typename tbase, typename tbase1>
inline FDualNumber<tbase> pow(const FDualNumber<tbase> &a, const tbase1 &b)
{
   return exp(log(a) * tbase(b));
}

/// pow([base number], [dual number])
template<typename tbase, typename tbase1>
inline FDualNumber<tbase> pow(const tbase1 &a, const FDualNumber<tbase> &b)
{
   return exp(log(tbase(a)) * b);
}

/// pow([base number], [dual number<double>])
template<>
inline FDualNumber<double> pow(const double &a, const FDualNumber<double> &b)
{
   return exp(std::log(a) * b);
}

/// sin([dual number])
template<typename tbase>
inline FDualNumber<tbase> sin(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(sin(f.real()), f.dual() * cos(f.real()));
}

/// sin([dual number<double>])
template<>
inline FDualNumber<double> sin(const FDualNumber<double> &f)
{
   return FDualNumber<double>(std::sin(f.real()), f.dual() * std::cos(f.real()));
}

/// sinh([dual number])
template<typename tbase>
inline FDualNumber<tbase> sinh(const FDualNumber<tbase> &f)
{
   return FDualNumber<tbase>(sinh(f.real()), f.dual() * cosh(f.real()));
}

/// sinh([dual number<double>])
template<>
inline FDualNumber<double> sinh(const FDualNumber<double> &f)
{
   return FDualNumber<double>(std::sinh(f.real()), f.dual() * std::cosh(f.real()));
}

/// sqrt([dual number])
template<typename tbase>
inline FDualNumber<tbase> sqrt(const FDualNumber<tbase> &f)
{
   tbase a = sqrt(f.real());
   return FDualNumber<tbase>(a, f.dual() / (tbase(2) * a));
}

/// sqrt([dual number<double>])
template<>
inline FDualNumber<double> sqrt(const FDualNumber<double> &f)
{
   double a = std::sqrt(f.real());
   return FDualNumber<double>(a, f.dual() / (double(2) * a));
}

/// tan([dual number])
template<typename tbase>
inline FDualNumber<tbase> tan(const FDualNumber<tbase> &f)
{
   tbase a = tan(f.real());
   return FDualNumber<tbase>(a, f.dual() * (tbase(1) + a * a));
}

/// tan([dual number<double>])
template<>
inline FDualNumber<double> tan(const FDualNumber<double> &f)
{
   double a = std::tan(f.real());
   return FDualNumber<double>(a, f.dual() * (double(1) + a * a));
}

/// tanh([dual number])
template<typename tbase>
inline FDualNumber<tbase> tanh(const FDualNumber<tbase> &f)
{
   tbase a = tanh(f.real());
   return FDualNumber<tbase>(a, f.dual() * (tbase(1) - a * a));
}

/// tanh([dual number<double>])
template<>
inline FDualNumber<double> tanh(const FDualNumber<double> &f)
{
   double a = std::tanh(f.real());
   return FDualNumber<double>(a, f.dual() * (double(1) - a * a));
}

} // namespace ad

} // namespace mfem

#endif
