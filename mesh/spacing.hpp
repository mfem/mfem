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

typedef enum {UNIFORM, LINEAR, GEOMETRIC, BELL,
              GAUSSIAN, LOGARITHMIC, PIECEWISE
             } SPACING_TYPE;

class SpacingFunction
{
public:
   SpacingFunction(int n_, bool r=false, bool s=false)
   {
      n = n_;
      reverse = r;
      scale = s;
   }

   inline int Size() const { return n; }

   virtual double Eval(int p) = 0;

   virtual void SetSize(int size) = 0;

   void SetReverse(bool r) { reverse = r; }

   void EvalAll(Vector & s)
   {
      s.SetSize(n);
      for (int i=0; i<n; ++i)
      {
         s[i] = Eval(i);
      }
   }

   virtual void ScaleParameters(double a) { }

   // The format is
   // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
   virtual void Print(std::ostream &os) = 0;

   virtual SPACING_TYPE SpacingType() = 0;

   virtual int NumIntParameters() = 0;
   virtual int NumDoubleParameters() = 0;
   virtual void GetIntParameters(Array<int> & p) = 0;
   virtual void GetDoubleParameters(Vector & p) = 0;

   virtual SpacingFunction *Clone() const;

   virtual ~SpacingFunction() { }

protected:
   int n;
   bool reverse, scale;
};

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

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << UNIFORM << " 1 0 " << n << "\n";
   }

   virtual SPACING_TYPE SpacingType() override { return UNIFORM; }
   virtual int NumIntParameters() override { return 1; }
   virtual int NumDoubleParameters() override { return 0; }

   virtual void GetIntParameters(Array<int> & p) override
   {
      p.SetSize(1);
      p[0] = n;
   }

   virtual void GetDoubleParameters(Vector & p) override
   {
      p.SetSize(0);
   }

   virtual SpacingFunction *Clone() const override
   {
      return new UniformSpacingFunction(n);
   }

private:
   double s;

   void CalculateSpacing()
   {
      // Spacing is 1 / n
      s = 1.0 / ((double) n);
   }
};

class LinearSpacingFunction : public SpacingFunction
{
public:
   LinearSpacingFunction(int n_, bool r_, double s_, bool scale_)
      : SpacingFunction(n_, r_, scale_), s(s_)
   {
      MFEM_ASSERT(0.0 < s && s < 1.0, "Initial spacing must be in (0,1)");
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
      MFEM_ASSERT(p>=0 && p<n, "Access element " << p
                  << " of spacing function, size = " << n);
      const int i = reverse ? n - 1 - p : p;
      return s + (i * d);
   }

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << LINEAR << " 3 1 " << n << " " << (int) reverse << " "
         << (int) scale << " " << s << "\n";
   }

   virtual SPACING_TYPE SpacingType() override { return LINEAR; }
   virtual int NumIntParameters() override { return 3; }
   virtual int NumDoubleParameters() override { return 1; }

   virtual void GetIntParameters(Array<int> & p) override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   virtual void GetDoubleParameters(Vector & p) override
   {
      p.SetSize(1);
      p[0] = s;
   }

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

// Spacing of interval i is s*r^i for 0 <= i < n, with
//     s + s*r + s*r^2 + ... + s*r^(n-1) = 1
//     s * (r^n - 1) / (r - 1) = 1
// The initial spacing s and number of intervals n are inputs, and r is solved
// for by Newton's method.
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
      return s * std::pow(r, i);
   }

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << GEOMETRIC << " 3 1 " << n << " " << (int) reverse << " "
         << (int) scale << " " << s << "\n";
   }

   virtual SPACING_TYPE SpacingType() override { return GEOMETRIC; }
   virtual int NumIntParameters() override { return 3; }
   virtual int NumDoubleParameters() override { return 1; }

   virtual void GetIntParameters(Array<int> & p) override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   virtual void GetDoubleParameters(Vector & p) override
   {
      p.SetSize(1);
      p[0] = s;
   }

   virtual SpacingFunction *Clone() const override
   {
      return new GeometricSpacingFunction(n, reverse, s, scale);
   }

private:
   double s;  // Initial spacing
   double r;  // Ratio

   void CalculateSpacing();
};

class BellSpacingFunction : public SpacingFunction
{
public:
   BellSpacingFunction(int n_, bool r_, double s0_, double s1_, bool scale_)
      : SpacingFunction(n_, r_, scale_), s0(s0_), s1(s1_)
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

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << BELL << " 3 2 " << n << " " << (int) reverse << " " << (int) scale
         << " " << s0 << " " << s1 << "\n";
   }

   virtual SPACING_TYPE SpacingType() override { return BELL; }
   virtual int NumIntParameters() override { return 3; }
   virtual int NumDoubleParameters() override { return 2; }

   virtual void GetIntParameters(Array<int> & p) override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   virtual void GetDoubleParameters(Vector & p) override
   {
      p.SetSize(2);
      p[0] = s0;
      p[1] = s1;
   }

   virtual SpacingFunction *Clone() const override
   {
      return new BellSpacingFunction(n, reverse, s0, s1, scale);
   }

private:
   double s0, s1;
   Vector s;

   void CalculateSpacing();
};

// GaussianSpacingFunction fits a Gaussian function of the general form
// g(x) = a exp(-(x-m)^2 / c^2) for some scalar parameters a, m, c.
class GaussianSpacingFunction : public SpacingFunction
{
public:
   GaussianSpacingFunction(int n_, bool r_, double s0_, double s1_, bool scale_)
      : SpacingFunction(n_, r_, scale_), s0(s0_), s1(s1_)
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

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << GAUSSIAN << " 3 2 " << n << " " << (int) reverse << " "
         << (int) scale << " " << s0 << " " << s1 << "\n";
   }

   virtual SPACING_TYPE SpacingType() override { return GAUSSIAN; }
   virtual int NumIntParameters() override { return 3; }
   virtual int NumDoubleParameters() override { return 2; }

   virtual void GetIntParameters(Array<int> & p) override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   virtual void GetDoubleParameters(Vector & p) override
   {
      p.SetSize(2);
      p[0] = s0;
      p[1] = s1;
   }

   virtual SpacingFunction *Clone() const override
   {
      return new GaussianSpacingFunction(n, reverse, s0, s1, scale);
   }

private:
   double s0, s1;
   Vector s;

   void CalculateSpacing();
};

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

   virtual void Print(std::ostream &os) override
   {
      // SPACING_TYPE numIntParam numDoubleParam {int params} {double params}
      os << LOGARITHMIC << " 3 1 " << n << " " << (int) reverse << " "
         << (int) sym << " " << logBase << "\n";
   }

   virtual SPACING_TYPE SpacingType() override { return LOGARITHMIC; }
   virtual int NumIntParameters() override { return 3; }
   virtual int NumDoubleParameters() override { return 1; }

   virtual void GetIntParameters(Array<int> & p) override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) sym;
   }

   virtual void GetDoubleParameters(Vector & p) override
   {
      p.SetSize(1);
      p[0] = logBase;
   }

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

class PiecewiseSpacingFunction : public SpacingFunction
{
public:
   // Parameters in ipar, for np pieces with index i:
   //   piece i: type, number of integer parameters, number of double
   //            parameters, integer parameters
   // Parameters in dpar start with np-1 partition entries, and the rest are
   // ordered by piece:
   //   piece i: double parameters
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

   virtual void Print(std::ostream &os) override;

   virtual SpacingFunction *Clone() const override;

   void SetupPieces(Array<int> const& ipar, Vector const& dpar);

   virtual SPACING_TYPE SpacingType() override { return PIECEWISE; }
   virtual int NumIntParameters() override { return 3; }
   virtual int NumDoubleParameters() override { return np - 1; }

   virtual void GetIntParameters(Array<int> & p) override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = np;
      p[2] = (int) reverse;
   }

   virtual void GetDoubleParameters(Vector & p) override
   {
      p.SetSize(np - 1);
      p = partition;
   }

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
