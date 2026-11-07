#ifndef MFEM_CUT_QUADRATURE_ALGOIM
#define MFEM_CUT_QUADRATURE_ALGOIM

#include "cut_quadrature.hpp"

#ifdef MFEM_USE_ALGOIM
#include <algoim/interval.hpp>
#include <utility>

namespace mfem
{

/** Optional interval capability for general ALGOIM quadrature.
    Include this header explicitly; ALGOIM types are not part of the common API.
    Evaluations must preserve ALGOIM's affine interval/remainder semantics.
    Callbacks must not invoke nested interval algorithms or quadrature. */
template <int Dimension>
class AlgoimIntervalLevelSet : public ReferenceLevelSet
{
public:
   static_assert(Dimension == 2 || Dimension == 3, "Quads and hexes only.");
   using Interval = algoim::Interval<Dimension>;
   using IntervalVector = algoim::uvector<Interval, Dimension>;

   Geometry::Type GetGeometry() const override
   { return Dimension == 2 ? Geometry::SQUARE : Geometry::CUBE; }

   virtual Interval EvalInterval(const IntervalVector &point) const = 0;
   virtual IntervalVector EvalIntervalGradient(
      const IntervalVector &point) const = 0;
};

/** Own a functor with scalar-templated operator() and grad() methods.
    Both methods must accept algoim::uvector<Scalar, Dimension> for real and
    interval scalars. Referenced objects captured by the functor must outlive
    this level set. No polynomial approximation is made. */
template <int Dimension, typename Functor>
class AlgoimFunctionLevelSet : public AlgoimIntervalLevelSet<Dimension>
{
public:
   using Interval = typename AlgoimIntervalLevelSet<Dimension>::Interval;
   using IntervalVector =
      typename AlgoimIntervalLevelSet<Dimension>::IntervalVector;

   explicit AlgoimFunctionLevelSet(Functor function)
      : function(std::move(function)) { }

   real_t Eval(const IntegrationPoint &point) const override
   { return function(Coordinates(point)); }

   void EvalGradient(const IntegrationPoint &point, Vector &gradient) const override
   {
      const auto values = function.grad(Coordinates(point));
      gradient.SetSize(Dimension);
      for (int axis = 0; axis < Dimension; ++axis)
      { gradient(axis) = values(axis); }
   }

   Interval EvalInterval(const IntervalVector &point) const override
   { return function(point); }

   IntervalVector EvalIntervalGradient(const IntervalVector &point) const override
   { return function.grad(point); }

private:
   Functor function;

   static algoim::uvector<real_t, Dimension> Coordinates(
      const IntegrationPoint &point)
   {
      algoim::uvector<real_t, Dimension> result;
      result(0) = point.x;
      result(1) = point.y;
      if constexpr (Dimension == 3) { result(2) = point.z; }
      return result;
   }
};

}
#endif
#endif
