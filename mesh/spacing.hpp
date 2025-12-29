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

#ifndef MFEM_SPACING
#define MFEM_SPACING

#include "../linalg/vector.hpp"

#include <memory>
#include <vector>

namespace mfem
{

enum class SpacingType {UNIFORM_SPACING, LINEAR, GEOMETRIC, BELL,
                        GAUSSIAN, LOGARITHMIC, PIECEWISE, PARTIAL
                       };

/// Class for spacing functions that define meshes in a dimension, using a
/// formula or method implemented in a derived class.
class SpacingFunction
{
public:
   /** @brief Base class constructor.
   @param[in] n  Size or number of intervals, which defines elements.
   @param[in] r  Whether to reverse the spacings, false by default.
   @param[in] s  Whether to scale parameters by the refinement or coarsening
                 factor, in the function @a SpacingFunction::ScaleParameters.
   */
   SpacingFunction(int n, bool r=false, bool s=false) : n(n), reverse(r), scale(s)
   { }

   /// Returns the size, or number of intervals (elements).
   inline int Size() const { return n; }

   /// Sets the size, or number of intervals (elements).
   virtual void SetSize(int size) = 0;

   /// Sets the property that determines whether the spacing is reversed.
   void SetReverse(bool r) { reverse = r; }

   bool GetReverse() const { return reverse; }

   void Flip() { reverse = !reverse; }

   /// Returns the width of interval @a p (between 0 and @a Size() - 1).
   virtual real_t Eval(int p) const = 0;

   /// Returns the width of all intervals, resizing @a s to @a Size().
   void EvalAll(Vector & s) const
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
   virtual void ScaleParameters(real_t a) { }

   /// Returns the spacing type, indicating the derived class.
   virtual SpacingType GetSpacingType() const = 0;

   /** @brief Prints all the data necessary to define the spacing function and
       its current state (size and other parameters).

       The format is generally
       SpacingType numIntParam numDoubleParam {int params} {double params} */
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

   /// Returns a clone (deep-copy) of this spacing function.
   virtual std::unique_ptr<SpacingFunction> Clone() const;

   void FullyCoarsen()
   {
      const int nfine = n;
      SetSize(1);
      ScaleParameters(nfine);
   }

   virtual ~SpacingFunction() = default;

protected:
   int n;  ///< Size, or number of intervals (elements)
   bool reverse;  ///< Whether to reverse the spacing
   bool scale;  ///< Whether to scale parameters in ScaleParameters.
};

/** @brief Uniform spacing function, dividing the unit interval into @a Size()
    equally spaced intervals (elements).

    This function is nested and has no scaled parameters. */
class UniformSpacingFunction : public SpacingFunction
{
public:
   UniformSpacingFunction(int n)
      : SpacingFunction(n)
   {
      CalculateSpacing();
   }

   void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   real_t Eval(int p) const override
   {
      return s;
   }

   void Print(std::ostream &os) const override
   {
      // SpacingType numIntParam numDoubleParam {int params} {double params}
      os << int(SpacingType::UNIFORM_SPACING) << " 1 0 " << n << "\n";
   }

   SpacingType GetSpacingType() const override
   { return SpacingType::UNIFORM_SPACING; }
   int NumIntParameters() const override { return 1; }
   int NumDoubleParameters() const override { return 0; }

   void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(1);
      p[0] = n;
   }

   void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(0);
   }

   bool Nested() const override { return true; }

   std::unique_ptr<SpacingFunction> Clone() const override
   {
      return std::unique_ptr<SpacingFunction>(
                new UniformSpacingFunction(*this));
   }

private:
   real_t s; ///< Width of each interval (element)

   /// Calculate interval width @a s
   void CalculateSpacing()
   {
      // Spacing is 1 / n
      s = 1.0 / ((real_t) n);
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
   LinearSpacingFunction(int n, bool r, real_t s, bool scale)
      : SpacingFunction(n, r, scale), s(s)
   {
      MFEM_VERIFY(0.0 < s && s < 1.0, "Initial spacing must be in (0,1)");
      CalculateDifference();
   }

   void SetSize(int size) override
   {
      n = size;
      CalculateDifference();
   }

   void ScaleParameters(real_t a) override
   {
      if (scale)
      {
         s *= a;
         CalculateDifference();
      }
   }

   real_t Eval(int p) const override
   {
      MFEM_ASSERT(0 <= p && p < n, "Access element " << p
                  << " of spacing function, size = " << n);
      const int i = reverse ? n - 1 - p : p;
      return s + (i * d);
   }

   void Print(std::ostream &os) const override
   {
      // SpacingType numIntParam numDoubleParam {int params} {double params}
      os << int(SpacingType::LINEAR) << " 3 1 " << n << " " << (int) reverse
         << " " << (int) scale << " " << s << "\n";
   }

   SpacingType GetSpacingType() const override { return SpacingType::LINEAR; }
   int NumIntParameters() const override { return 3; }
   int NumDoubleParameters() const override { return 1; }

   void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(1);
      p[0] = s;
   }

   bool Nested() const override { return false; }

   std::unique_ptr<SpacingFunction> Clone() const override
   {
      return std::unique_ptr<SpacingFunction>(
                new LinearSpacingFunction(*this));
   }

private:
   real_t s, d;  ///< Spacing parameters, set by @a CalculateDifference

   void CalculateDifference()
   {
      if (n < 2)
      {
         d = 0.0;
         return;
      }

      // Spacings are s, s + d, ..., s + (n-1)d, which must sum to 1:
      // 1 = ns + dn(n-1)/2
      d = 2.0 * (1.0 - (n * s)) / ((real_t) (n*(n-1)));

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
   GeometricSpacingFunction(int n, bool r, real_t s, bool scale)
      : SpacingFunction(n, r, scale), s(s)
   {
      CalculateSpacing();
   }

   void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   void ScaleParameters(real_t a) override
   {
      if (scale)
      {
         s *= a;
         CalculateSpacing();
      }
   }

   real_t Eval(int p) const override
   {
      const int i = reverse ? n - 1 - p : p;
      return n == 1 ? 1.0 : s * std::pow(r, i);
   }

   void Print(std::ostream &os) const override
   {
      // SpacingType numIntParam numDoubleParam {int params} {double params}
      os << int(SpacingType::GEOMETRIC) << " 3 1 " << n << " "
         << (int) reverse << " " << (int) scale << " " << s << "\n";
   }

   SpacingType GetSpacingType() const override
   { return SpacingType::GEOMETRIC; }
   int NumIntParameters() const override { return 3; }
   int NumDoubleParameters() const override { return 1; }

   void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(1);
      p[0] = s;
   }

   bool Nested() const override { return false; }

   std::unique_ptr<SpacingFunction> Clone() const override
   {
      return std::unique_ptr<SpacingFunction>(
                new GeometricSpacingFunction(*this));
   }

private:
   real_t s;  ///< Initial spacing
   real_t r;  ///< Ratio

   /// Calculate parameters used by @a Eval and @a EvalAll
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
   @param[in] n  Size or number of intervals, which defines elements.
   @param[in] r  Whether to reverse the spacings.
   @param[in] s0 Width of the first interval (element).
   @param[in] s1 Width of the last interval (element).
   @param[in] s  Whether to scale parameters by the refinement or coarsening
                  factor, in the function @a SpacingFunction::ScaleParameters.
   */
   BellSpacingFunction(int n, bool r, real_t s0, real_t s1, bool s)
      : SpacingFunction(n, r, s), s0(s0), s1(s1)
   {
      CalculateSpacing();
   }

   void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   void ScaleParameters(real_t a) override
   {
      if (scale)
      {
         s0 *= a;
         s1 *= a;
         CalculateSpacing();
      }
   }

   real_t Eval(int p) const override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   void Print(std::ostream &os) const override
   {
      // SpacingType numIntParam numDoubleParam {int params} {double params}
      os << int(SpacingType::BELL) << " 3 2 " << n << " " << (int) reverse
         << " " << (int) scale << " " << s0 << " " << s1 << "\n";
   }

   SpacingType GetSpacingType() const override { return SpacingType::BELL; }
   int NumIntParameters() const override { return 3; }
   int NumDoubleParameters() const override { return 2; }

   void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(2);
      p[0] = s0;
      p[1] = s1;
   }

   bool Nested() const override { return false; }

   std::unique_ptr<SpacingFunction> Clone() const override
   {
      return std::unique_ptr<SpacingFunction>(
                new BellSpacingFunction(*this));
   }

private:
   real_t s0, s1; ///< First and last interval widths
   Vector s;  ///< Stores the spacings calculated by @a CalculateSpacing

   /// Calculate parameters used by @a Eval and @a EvalAll
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
   /** @brief Constructor for GaussianSpacingFunction.
   @param[in] n  Size or number of intervals, which defines elements.
   @param[in] r  Whether to reverse the spacings.
   @param[in] s0 Width of the first interval (element).
   @param[in] s1 Width of the last interval (element).
   @param[in] s  Whether to scale parameters by the refinement or coarsening
                  factor, in the function @a SpacingFunction::ScaleParameters.
   */
   GaussianSpacingFunction(int n, bool r, real_t s0, real_t s1, bool s)
      : SpacingFunction(n, r, s), s0(s0), s1(s1)
   {
      CalculateSpacing();
   }

   void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   void ScaleParameters(real_t a) override
   {
      if (scale)
      {
         s0 *= a;
         s1 *= a;
         CalculateSpacing();
      }
   }

   real_t Eval(int p) const override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   void Print(std::ostream &os) const override
   {
      // SpacingType numIntParam numDoubleParam {int params} {double params}
      os << int(SpacingType::GAUSSIAN) << " 3 2 " << n << " " << (int) reverse
         << " " << (int) scale << " " << s0 << " " << s1 << "\n";
   }

   SpacingType GetSpacingType() const override { return SpacingType::GAUSSIAN; }
   int NumIntParameters() const override { return 3; }
   int NumDoubleParameters() const override { return 2; }

   void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) scale;
   }

   void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(2);
      p[0] = s0;
      p[1] = s1;
   }

   bool Nested() const override { return false; }

   std::unique_ptr<SpacingFunction> Clone() const override
   {
      return std::unique_ptr<SpacingFunction>(
                new GaussianSpacingFunction(*this));
   }

private:
   real_t s0, s1; ///< First and last interval widths
   Vector s;  ///< Stores the spacings calculated by @a CalculateSpacing

   /// Calculate parameters used by @a Eval and @a EvalAll
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
   LogarithmicSpacingFunction(int n, bool r, bool sym=false, real_t b=10.0)
      : SpacingFunction(n, r), sym(sym), logBase(b)
   {
      CalculateSpacing();
   }

   void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   real_t Eval(int p) const override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   void Print(std::ostream &os) const override
   {
      // SpacingType numIntParam numDoubleParam {int params} {double params}
      os << int(SpacingType::LOGARITHMIC) << " 3 1 " << n << " " <<
         (int) reverse << " " << (int) sym << " " << logBase << "\n";
   }

   SpacingType GetSpacingType() const override
   { return SpacingType::LOGARITHMIC; }
   int NumIntParameters() const override { return 3; }
   int NumDoubleParameters() const override { return 1; }

   void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(3);
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = (int) sym;
   }

   void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(1);
      p[0] = logBase;
   }

   bool Nested() const override { return true; }

   std::unique_ptr<SpacingFunction> Clone() const override
   {
      return std::unique_ptr<SpacingFunction>(
                new LogarithmicSpacingFunction(*this));
   }

private:
   bool sym;  ///< Whether to make the spacing symmetric
   real_t logBase;  ///< Base of the logarithmic function
   Vector s;  ///< Stores the spacings calculated by @a CalculateSpacing

   /// Calculate parameters used by @a Eval and @a EvalAll
   void CalculateSpacing();
   /// Symmetric case for @a CalculateSpacing
   void CalculateSymmetric();
   /// Nonsymmetric case for @a CalculateSpacing
   void CalculateNonsymmetric();
};

/** @brief Piecewise spacing function, with spacing functions defining spacing
    within arbitarily many fixed subintervals of the unit interval.

    The number of elements in each piece (or subinterval) is determined by the
    constructor input @a relN, which is the relative number of intervals. For
    equal numbers, relN would be all 1's. The total number of elements for this
    spacing function must be an integer multiple of the sum of entries in relN
    (stored in n0).

    The scaling of parameters is done for the spacing function on each
    subinterval separately. This function is nested if and only if the functions
    on all subintervals are nested.
 */
class PiecewiseSpacingFunction : public SpacingFunction
{
public:
   /** @brief Constructor for PiecewiseSpacingFunction.
   @param[in] n   Size or number of intervals, which defines elements.
   @param[in] np  Number of pieces (subintervals of unit interval).
   @param[in] r   Whether to reverse the spacings.
   @param[in] relN Relative number of elements per piece.
   @param[in] ipar Integer parameters for all np spacing functions. For each
                   piece, these parameters are type, number of integer
                   parameters, number of double parameters, integer parameters.
   @param[in] dpar Double parameters for all np spacing functions. The first
                   np - 1 entries define the partition of the unit interval,
                   and the remaining are for the pieces.
   */
   PiecewiseSpacingFunction(int n, int np, bool r, Array<int> const& relN,
                            Array<int> const& ipar, Vector const& dpar)
      : SpacingFunction(n, r), np(np), partition(np - 1)
   {
      npartition = relN;
      SetupPieces(ipar, dpar);
      CalculateSpacing();
   }

   /// Copy constructor (deep-copy all data, including SpacingFunction pieces)
   PiecewiseSpacingFunction(const PiecewiseSpacingFunction &sf)
      : SpacingFunction(sf.n, sf.reverse), np(sf.np), partition(sf.partition),
        npartition(sf.npartition), pieces(), n0(sf.n0), s(sf.s)
   {
      // To copy, the pointers must be cloned.
      for (const auto &f : sf.pieces) { pieces.emplace_back(f->Clone()); }
   }

   PiecewiseSpacingFunction& operator=(const PiecewiseSpacingFunction &sf)
   {
      PiecewiseSpacingFunction tmp(sf);
      std::swap(tmp, *this);
      return *this;
   }

   PiecewiseSpacingFunction(PiecewiseSpacingFunction &&sf) = default;
   PiecewiseSpacingFunction& operator=(PiecewiseSpacingFunction &&sf) = default;

   void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   real_t Eval(int p) const override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   void ScaleParameters(real_t a) override;

   void Print(std::ostream &os) const override;

   std::unique_ptr<SpacingFunction> Clone() const override
   {
      return std::unique_ptr<SpacingFunction>(
                new PiecewiseSpacingFunction(*this));
   }

   void SetupPieces(Array<int> const& ipar, Vector const& dpar);

   SpacingType GetSpacingType() const override { return SpacingType::PIECEWISE; }
   int NumIntParameters() const override
   {
      int count = 3 + np;
      for (const auto &p : pieces)
      {
         // Add three for the type and the integer and double parameter counts.
         count += p->NumIntParameters() + 3;
      }
      return count;
   }

   int NumDoubleParameters() const override
   {
      int count = np - 1;
      for (const auto &p : pieces)
      {
         count += p->NumDoubleParameters();
      }
      return count;
   }

   Array<int> RelativePieceSizes() const { return npartition; }

   void ScalePartition(Array<int> const& f, bool reorient)
   {
      MFEM_VERIFY(npartition.Size() == f.Size(), "");
      n0 = 0;
      for (int i=0; i<f.Size(); ++i)
      {
         const int ir = reorient && reverse ? f.Size() - 1 - i : i;
         npartition[i] *= f[ir];
         n0 += npartition[i];
      }
   }

   void GetIntParameters(Array<int> & p) const override
   {
      p.SetSize(NumIntParameters());
      p[0] = n;
      p[1] = np;
      p[2] = (int) reverse;
      for (int i=0; i<np; ++i) { p[3 + i] = npartition[i]; }

      Array<int> ipar;
      int os = 3 + np;
      for (int i=0; i<np; ++i)
      {
         p[os++] = (int) pieces[i]->GetSpacingType();
         p[os++] = pieces[i]->NumIntParameters();
         p[os++] = pieces[i]->NumDoubleParameters();

         pieces[i]->GetIntParameters(ipar);
         for (auto ip : ipar)
         {
            p[os++] = ip;
         }
      }
   }

   void GetDoubleParameters(Vector & p) const override
   {
      p.SetSize(NumDoubleParameters());
      for (int i=0; i<partition.Size(); ++i) { p[i] = partition[i]; }
      int os = partition.Size();
      Vector dpar;
      for (int i=0; i<np; ++i)
      {
         pieces[i]->GetDoubleParameters(dpar);
         for (auto dp : dpar)
         {
            p[os++] = dp;
         }
      }
   }

   // PiecewiseSpacingFunction is nested if and only if all pieces are nested.
   bool Nested() const override;

private:
   int np;  ///< Number of pieces
   Vector partition;  ///< Partition of the unit interval
   Array<int> npartition;  ///< Relative number of intervals in each partition
   std::vector<std::unique_ptr<SpacingFunction>> pieces;

   int n0 = 0;  ///< Sum of npartition

   Vector s;  ///< Stores the spacings calculated by @a CalculateSpacing

   /// Calculate parameters used by @a Eval and @a EvalAll
   void CalculateSpacing();
};

/** @brief Partial spacing function, defined as part of an existing spacing
    function.
 */
class PartialSpacingFunction : public SpacingFunction
{
public:
   PartialSpacingFunction(int n, bool r, int rel_first_elem, int rel_num_elems,
                          int rel_num_elems_full,
                          Array<int> const& ipar, Vector const& dpar,
                          SpacingType typeFull)
      : SpacingFunction(n, r), num_elems_full(rel_num_elems_full),
        num_elems(rel_num_elems), first_elem(rel_first_elem)
   {
      SetupFull(typeFull, ipar, dpar);
      CalculateSpacing();
   }

   /// Copy constructor (deep-copy all data, including SpacingFunction pieces)
   PartialSpacingFunction(const PartialSpacingFunction &sf)
      : SpacingFunction(sf.n, sf.reverse), fullSpacing(sf.fullSpacing->Clone()),
        num_elems_full(sf.num_elems_full), num_elems(sf.num_elems),
        first_elem(sf.first_elem), s(sf.s)
   { }

   PartialSpacingFunction& operator=(const PartialSpacingFunction &sf)
   {
      PartialSpacingFunction tmp(sf);
      std::swap(tmp, *this);
      return *this;
   }

   PartialSpacingFunction(PartialSpacingFunction &&sf) = default;
   PartialSpacingFunction& operator=(PartialSpacingFunction &&sf) = default;

   void SetSize(int size) override
   {
      n = size;
      CalculateSpacing();
   }

   real_t Eval(int p) const override
   {
      const int i = reverse ? n - 1 - p : p;
      return s[i];
   }

   void ScaleParameters(real_t a) override;

   void Print(std::ostream &os) const override;

   std::unique_ptr<SpacingFunction> Clone() const override
   {
      return std::unique_ptr<SpacingFunction>(
                new PartialSpacingFunction(*this));
   }

   void SetupFull(SpacingType typeFull,
                  Array<int> const& ipar, Vector const& dpar);

   SpacingType GetSpacingType() const override { return SpacingType::PARTIAL; }
   int NumIntParameters() const override { return 8 + fullSpacing->NumIntParameters(); }
   int NumDoubleParameters() const override { return fullSpacing->NumDoubleParameters(); }

   void GetIntParameters(Array<int> & p) const override
   {
      Array<int> ipar;
      fullSpacing->GetIntParameters(ipar);
      p.SetSize(8 + ipar.Size());
      p[0] = n;
      p[1] = (int) reverse;
      p[2] = first_elem;
      p[3] = num_elems;
      p[4] = num_elems_full;
      p[5] = (int) fullSpacing->GetSpacingType();
      p[6] = ipar.Size();
      p[7] = NumDoubleParameters();
      for (int i=0; i<ipar.Size(); ++i) { p[8 + i] = ipar[i]; }
   }

   void GetDoubleParameters(Vector & p) const override
   {
      fullSpacing->GetDoubleParameters(p);
   }

   // PartialSpacingFunction is nested if and only if the full spacing is nested.
   bool Nested() const override { return fullSpacing->Nested(); }

private:
   std::unique_ptr<SpacingFunction> fullSpacing;

   // The following numbers and indices may be relative, not absolute.
   int num_elems_full; ///< Reference number of elements in fullSpacing
   int num_elems; ///< Number of elements, relative to num_elems_full
   int first_elem; ///< Index of the first element, relative to num_elems_full

   Vector s;  ///< Stores the spacings calculated by @a CalculateSpacing

   /// Calculate parameters used by @a Eval and @a EvalAll
   void CalculateSpacing();
};

/// Returns a new SpacingFunction instance defined by the type and parameters
std::unique_ptr<SpacingFunction> GetSpacingFunction(const SpacingType type,
                                                    Array<int> const& ipar,
                                                    Vector const& dpar);
}
#endif
