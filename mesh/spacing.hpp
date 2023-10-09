// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SPACING
#define MFEM_SPACING

#include "../linalg/vector.hpp"

namespace mfem
{

typedef enum {UNIFORM_SPACING, LINEAR, GEOMETRIC, BELL,
              GAUSSIAN, LOGARITHMIC, PIECEWISE
             } SPACING_TYPE;

/// Class for spacing functions that define meshes in a dimension, using a
/// formula or method implemented in a derived class.
class SpacingFunction
{
public:
   /** @brief Base class constructor.
   @param[in] n_  Size or number of intervals, which defines elements.
   @param[in] r   Whether to reverse the spacings, false by default.
   @param[in] s   Whether to scale parameters by the refinement or coarsening
                  factor, in the function @a SpacingFunction::ScaleParameters.
   */
   SpacingFunction(int n_, bool r=false, bool s=false)
   {
      n = n_;
      reverse = r;
      scale = s;
   }

   /// Returns the size, or number of intervals (elements).
   inline int Size() const { return n; }

   /// Sets the size, or number of intervals (elements).
   virtual void SetSize(int size) = 0;

   /// Sets the property that determines whether the spacing is reversed.
   void SetReverse(bool r) { reverse = r; }

   /// Returns the width of interval @a p (between 0 and @a Size() - 1).
   virtual double Eval(int p) = 0;

   /// Returns the width of all intervals, resizing @a s to @a Size().
   void EvalAll(Vector & s)
   {
      s.SetSize(n);
      for (int i=0; i<n; ++i)
      {
         s[i] = Eval(i);
      }
   }

   /** @brief Scales parameters by the factor @a a associated with @a Size().

       Note that parameters may be scaled inversely during coarsening and
       refining, so the scaling should be linear in the sense that scaling by a
       number followed by scaling by its inverse has no effect on parameters. */
   virtual void ScaleParameters(double a) { }

   /// Returns the spacing type, indicating the derived class.
   virtual SPACING_TYPE SpacingType() const = 0;

   /* @brief Prints all the data necessary to define the spacing function and
      its current state (size and other parameters).

      The format is generally
      SPACING_TYPE numIntParam numDoubleParam {int params} {double params} */
   virtual void Print(std::ostream &os) const = 0;

   /// Returns the number of integer parameters defining the spacing function.
   virtual int NumIntParameters() const = 0;

   /// Returns the number of double parameters defining the spacing function.
   virtual int NumDoubleParameters() const = 0;

   /// Returns the array of integer parameters defining the spacing function.
   /// @param[out] p  Array of integer parameters, resized appropriately.
   virtual void GetIntParameters(Array<int> & p) const = 0;

   /// Returns the array of double parameters defining the spacing function.
   /// @param[out] p  Array of double parameters, resized appropriately.
   virtual void GetDoubleParameters(Vector & p) const = 0;

   /// Returns true if the spacing function is nested during refinement.
   virtual bool Nested() const = 0;

   /// Returns a clone of this spacing function.
   virtual SpacingFunction *Clone() const;

   virtual ~SpacingFunction() { }

protected:
   int n;  // Size, or number of intervals (elements)
   bool reverse;  // Whether to reverse the spacing
   bool scale;  // Whether to scale parameters in ScaleParameters.
};

/** @brief Uniform spacing function, dividing the unit interval into @a Size()
    equally spaced intervals (elements).

    This function is nested and has no scaled parameters. */
class UniformSpacingFunction : public SpacingFunction
{
public:
   UniformSpacingFunction(int n_)
      : SpacingFunction(n_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual double Eval(int p) override
   {
      return s;
   }

   virtual void Print(std::ostream &os) const override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << UNIFORM_SPACING << " 1 0 " << n << "\n";
   }

   virtual SPACING_TYPE SpacingType() const override { return UNIFORM_SPACING; }
   virtual int NumIntParameters() const override { return 1; }
   virtual int NumDoubleParameters() const override { return 0; }

   virtual void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(1);
      p[0] = n;
   }

   virtual void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(0);
   }

   virtual bool Nested() const override { return true; }

   virtual SpacingFunction *Clone() const override
   {
      return new UniformSpacingFunction(n);
   }

private:
   double s; // Width of each interval (element)

   void CalculateSpacing()
   {
      // Spacing is 1 / n
      s = 1.0 / ((double) n);
   }
};

/** @brief Linear spacing function, defining the width of interval i as
    s + i * d.

    The initial interval width s is prescribed as a parameter, which can be
    scaled, and d is computed as a function of Size(), to ensure that the widths
    sum to 1. This function is not nested. */
class LinearSpacingFunction : public SpacingFunction
{
public:
   LinearSpacingFunction(int n_, bool r_, double s_, bool scale_)
      : SpacingFunction(n_, r_, scale_), s(s_)
   {
      MFEM_VERIFY(0.0 < s && s < 1.0, "Initial spacing must be in (0,1)");
      CalculateDifference();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateDifference();
   }

   virtual void ScaleParameters(double a) override
   {
      if (scale)
      {
         s *= a;
         CalculateDifference();
      }
   }

   virtual double Eval(int p) override
   {
      MFEM_ASSERT(0 <= p && p < n, "Access element " << p
                  << " of spacing function, size = " << n);
      const int i = reverse ? n - 1 - p : p;
      return s + (i * d);
   }

   virtual void Print(std::ostream &os) const override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << LINEAR << " 3 1 " << n << " " << (int) reverse << " "
         << (int) scale << " " << s << "\n";
   }

   virtual SPACING_TYPE SpacingType() const override { return LINEAR; }
   virtual int NumIntParameters() const override { return 3; }
   virtual int NumDoubleParameters() const override { return 1; }

   virtual void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   virtual void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(1);
      p[0] = s;
   }

   virtual bool Nested() const override { return false; }

   virtual SpacingFunction *Clone() const override
   {
      return new LinearSpacingFunction(n, reverse, s, scale);
   }

private:
   double s, d;

   void CalculateDifference()
   {
      // Spacings are s, s + d, ..., s + (n-1)d, which must sum to 1:
      // 1 = ns + dn(n-1)/2
      d = 2.0 * (1.0 - (n * s)) / ((double) (n*(n-1)));

      if (s + ((n-1) * d) <= 0.0)
      {
         MFEM_ABORT("Invalid linear spacing parameters");
      }
   }
};

/** @brief Geometric spacing function.

    The spacing of interval i is s*r^i for 0 <= i < n, with
       s + s*r + s*r^2 + ... + s*r^(n-1) = 1
       s * (r^n - 1) / (r - 1) = 1
    The initial spacing s and number of intervals n are inputs, and r is solved
    for by Newton's method. The parameter s can be scaled. This function is not
    nested. */
class GeometricSpacingFunction : public SpacingFunction
{
public:
   GeometricSpacingFunction(int n_, bool r_, double s_, bool scale_)
      : SpacingFunction(n_, r_, scale_), s(s_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual void ScaleParameters(double a) override
   {
      if (scale)
      {
         s *= a;
         CalculateSpacing();
      }
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return n == 1 ? 1.0 : s * std::pow(r, i);
   }

   virtual void Print(std::ostream &os) const override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << GEOMETRIC << " 3 1 " << n << " " << (int) reverse << " "
         << (int) scale << " " << s << "\n";
   }

   virtual SPACING_TYPE SpacingType() const override { return GEOMETRIC; }
   virtual int NumIntParameters() const override { return 3; }
   virtual int NumDoubleParameters() const override { return 1; }

   virtual void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   virtual void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(1);
      p[0] = s;
   }

   virtual bool Nested() const override { return false; }

   virtual SpacingFunction *Clone() const override
   {
      return new GeometricSpacingFunction(n, reverse, s, scale);
   }

private:
   double s;  // Initial spacing
   double r;  // Ratio

   void CalculateSpacing();
};

/** @brief Bell spacing function, which produces spacing resembling a Bell
    curve.

    The widths of the first and last intervals (elements) are prescribed, and
    the remaining interior spacings are computed by an algorithm that minimizes
    the ratios of adjacent spacings. If the first and last intervals are wide
    enough, the spacing may decrease in the middle of the domain. The first and
    last interval widths can be scaled. This function is not nested.
 */
class BellSpacingFunction : public SpacingFunction
{
public:
   /** @brief Constructor for BellSpacingFunction.
   @param[in] n_  Size or number of intervals, which defines elements.
   @param[in] r_  Whether to reverse the spacings.
   @param[in] s0_ Width of the first interval (element).
   @param[in] s1_ Width of the last interval (element).
   @param[in] s_  Whether to scale parameters by the refinement or coarsening
                  factor, in the function @a SpacingFunction::ScaleParameters.
   */
   BellSpacingFunction(int n_, bool r_, double s0_, double s1_, bool s_)
      : SpacingFunction(n_, r_, s_), s0(s0_), s1(s1_)
   {
      CalculateSpacing();
   }

   BellSpacingFunction(const BellSpacingFunction &sf)
      : SpacingFunction(sf.n, sf.reverse, sf.scale), s0(sf.s0), s1(sf.s1)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual void ScaleParameters(double a) override
   {
      if (scale)
      {
         s0 *= a;
         s1 *= a;
         CalculateSpacing();
      }
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   virtual void Print(std::ostream &os) const override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << BELL << " 3 2 " << n << " " << (int) reverse << " " << (int) scale
         << " " << s0 << " " << s1 << "\n";
   }

   virtual SPACING_TYPE SpacingType() const override { return BELL; }
   virtual int NumIntParameters() const override { return 3; }
   virtual int NumDoubleParameters() const override { return 2; }

   virtual void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   virtual void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(2);
      p[0] = s0;
      p[1] = s1;
   }

   virtual bool Nested() const override { return false; }

   virtual SpacingFunction *Clone() const override
   {
      return new BellSpacingFunction(n, reverse, s0, s1, scale);
   }

private:
   double s0, s1; // First and last interval widths
   Vector s;

   void CalculateSpacing();
};

/** @brief Gaussian spacing function of the general form
    g(x) = a exp(-(x-m)^2 / c^2) for some scalar parameters a, m, c.

    The widths of the first and last intervals (elements) are prescribed, and
    the remaining interior spacings are computed by using Newton's method to
    compute parameters that fit the endpoint widths. The results of this spacing
    function are very similar to those of @a BellSpacingFunction, but they may
    differ by about 1%. If the first and last intervals are wide enough, the
    spacing may decrease in the middle of the domain. The first and last
    interval widths can be scaled. This function is not nested. */
class GaussianSpacingFunction : public SpacingFunction
{
public:
   /** @brief Constructor for BellSpacingFunction.
   @param[in] n_  Size or number of intervals, which defines elements.
   @param[in] r_  Whether to reverse the spacings.
   @param[in] s0_ Width of the first interval (element).
   @param[in] s1_ Width of the last interval (element).
   @param[in] s_  Whether to scale parameters by the refinement or coarsening
                  factor, in the function @a SpacingFunction::ScaleParameters.
   */
   GaussianSpacingFunction(int n_, bool r_, double s0_, double s1_, bool s_)
      : SpacingFunction(n_, r_, s_), s0(s0_), s1(s1_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual void ScaleParameters(double a) override
   {
      if (scale)
      {
         s0 *= a;
         s1 *= a;
         CalculateSpacing();
      }
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   virtual void Print(std::ostream &os) const override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << GAUSSIAN << " 3 2 " << n << " " << (int) reverse << " "
         << (int) scale << " " << s0 << " " << s1 << "\n";
   }

   virtual SPACING_TYPE SpacingType() const override { return GAUSSIAN; }
   virtual int NumIntParameters() const override { return 3; }
   virtual int NumDoubleParameters() const override { return 2; }

   virtual void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   virtual void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(2);
      p[0] = s0;
      p[1] = s1;
   }

   virtual bool Nested() const override { return false; }

   virtual SpacingFunction *Clone() const override
   {
      return new GaussianSpacingFunction(n, reverse, s0, s1, scale);
   }

private:
   double s0, s1; // First and last interval widths
   Vector s;

   void CalculateSpacing();
};

/** @brief Logarithmic spacing function, uniform in log base 10 by default.

    The log base can be changed as an input parameter. Decreasing it makes the
    distribution more uniform, whereas increasing it makes the spacing vary
    more. Another input option is a flag to make the distribution symmetric
    (default is non-symmetric). There are no scaled parameters. This function is
    nested. */
class LogarithmicSpacingFunction : public SpacingFunction
{
public:
   LogarithmicSpacingFunction(int n_, bool r_, bool sym_=false, double b_=10.0)
      : SpacingFunction(n_, r_), sym(sym_), logBase(b_)
   {
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   virtual void Print(std::ostream &os) const override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << LOGARITHMIC << " 3 1 " << n << " " << (int) reverse << " "
         << (int) sym << " " << logBase << "\n";
   }

   virtual SPACING_TYPE SpacingType() const override { return LOGARITHMIC; }
   virtual int NumIntParameters() const override { return 3; }
   virtual int NumDoubleParameters() const override { return 1; }

   virtual void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) sym;
   }

   virtual void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(1);
      p[0] = logBase;
   }

   virtual bool Nested() const override { return true; }

   virtual SpacingFunction *Clone() const override
   {
      return new LogarithmicSpacingFunction(n, reverse, sym, logBase);
   }

private:
   bool sym;
   double logBase;
   Vector s;

   void CalculateSpacing();
   void CalculateSymmetric();
   void CalculateNonsymmetric();
};

/** @brief Piecewise spacing function, with spacing functions defining spacing
    within arbitarily many fixed subdomains of the unit interval.

    The scaling of parameters is done for the spacing function on each
    subinterval separately. This function is nested if and only if the functions
    on all subintervals are nested.
 */
class PiecewiseSpacingFunction : public SpacingFunction
{
public:
   /** @brief Constructor for PiecewiseSpacingFunction.
   @param[in] n_   Size or number of intervals, which defines elements.
   @param[in] np_  Number of pieces (subintervals of unit interval).
   @param[in] ipar Integer parameters for all np_ spacing functions. For each
                   piece, these parameters are type, number of integer
                   parameters, number of double parameters, integer parameters.
   @param[in] dpar Double parameters for all np_ spacing functions. The first
                   np_ - 1 entries define the partition of the unit interval,
                   and the remaining are for the pieces.
   */
   PiecewiseSpacingFunction(int n_, int np_, bool r_,
                            Array<int> const& ipar, Vector const& dpar)
      : SpacingFunction(n_, r_), np(np_), partition(np - 1)
   {
      SetupPieces(ipar, dpar);
      CalculateSpacing();
   }

   virtual void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   virtual double Eval(int p) override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   virtual void ScaleParameters(double a) override;

   virtual void Print(std::ostream &os) const override;

   virtual SpacingFunction *Clone() const override;

   void SetupPieces(Array<int> const& ipar, Vector const& dpar);

   virtual SPACING_TYPE SpacingType() const override { return PIECEWISE; }
   virtual int NumIntParameters() const override { return 3; }
   virtual int NumDoubleParameters() const override { return np - 1; }

   virtual void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = np;
      p[2] = (int) reverse;
   }

   virtual void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(np - 1);
      p = partition;
   }

   // PiecewiseSpacingFunction is nested if and only if all pieces are nested.
   virtual bool Nested() const override;

private:
   int np;  // Number of pieces
   Vector partition;
   Array<int> npartition;
   Array<SpacingFunction*> pieces;

   int n0 = 0;

   Vector s;

   void CalculateSpacing();
};

SpacingFunction* GetSpacingFunction(const SPACING_TYPE type,
                                    Array<int> const& ipar,
                                    Vector const& dpar);
}
#endif
