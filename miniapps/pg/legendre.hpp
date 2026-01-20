#pragma once

#include "mfem.hpp"
#if __cplusplus < 201402L
// define make_unique for c++11
namespace std
{
template<class T> struct _Unique_if
{
   typedef unique_ptr<T> _Single_object;
};

template<class T> struct _Unique_if<T[]>
{
   typedef unique_ptr<T[]> _Unknown_bound;
};

template<class T, size_t N> struct _Unique_if<T[N]>
{
   typedef void _Known_bound;
};

template<class T, class... Args>
typename _Unique_if<T>::_Single_object
make_unique(Args&&... args)
{
   return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template<class T>
typename _Unique_if<T>::_Unknown_bound
make_unique(size_t n)
{
   typedef typename remove_extent<T>::type U;
   return unique_ptr<T>(new U[n]());
}

template<class T, class... Args>
typename _Unique_if<T>::_Known_bound
make_unique(Args&&...) = delete;
} // namespace std
#endif

namespace mfem
{

class ScaledLegendreFunction; // f(x) = g((x - shift) / scale), dom(f) = dom(g)*scale + shift
class CoefficientScaledLegendreFunction; // f(x) = g((x - shift(x)) / scale(x))
class TransformedLegendreFunction; // f(x) = g(A(x - b)), dom(f) = A*dom(g) + b
class Shannon; // f(x) = x ln(x) - x, dom(f) = [0, inf)
class FermiDirac; // f(x) = x ln(x) + (1-x) ln(1-x), dom(f) = [0, 1]
class Hellinger; // f(x) = -sqrt(1 - x^2), dom(f) = unit ball
class Gibbs; // f(x) = sum(x ln(x)), dom(f) = standard n-simplex
class SeperableLegendreFunction; // f(x) = sum g_i(x_i), dom(f) = cartesian product of dom(g_i)

// @brief Base class for Legendre functions
// Here, Legendre functions are
// - strictly convex at int(dom(f))
// - continuously differentiable at int(dom(f))
// - divergent gradient at boundary of dom(f)
//
// grad f is injective from int(dom(f)) to R^n
// If f is superlinear (f(x) / |x| -> +inf as |x| -> +inf), then grad f is
// a bijection from int(dom(f)) to R^n, and gradinv is well-defined everywhere.
//
// In fact, the inverse of the gradient is the gradient of the Fenchel conjugate
// (grad f)^{-1} = grad (f^*) : R^n -> int(dom(f))
// where f^*(y) = sup_x (x dot y - f(x))
//
// If f is R -> R, then f(x) with vector input x is defined as
// f(x) = sum f(x_i)
class LegendreFunction
{
public:
   LegendreFunction() = default;
   virtual ~LegendreFunction() = default;

   // scalar interface

   // @brief Evaluate the function at a point x
   // @param[in] x The point at which to evaluate the function
   // @return The value of the function at x
   virtual real_t operator()(real_t x) const
   { MFEM_ABORT("LegendreFunction::operator() not implemented."); }

   // @brief Evaluate the function at a point x
   // @param[in] x The point at which to evaluate the function
   // @param[in] Tr the element transformation with integration point
   // @return The value of the function at x
   virtual real_t operator()(real_t x, ElementTransformation &Tr) const
   { return this->operator()(x); }

   // @brief Evaluate the function at a point x
   // @param[in] x The point at which to evaluate the function
   // @return The value of the function at x
   virtual real_t operator()(const Vector &x) const
   {
      real_t sum = 0.0;
      for (auto val : x) { sum += this->operator()(val); }
      return sum;
   }

   // @brief Evaluate the function at a point x
   // @param[in] x The point at which to evaluate the function
   // @param[in] Tr the element transformation with integration point
   // @return The value of the function at x
   virtual real_t operator()(const Vector &x, ElementTransformation &Tr) const
   { return this->operator()(x); }

   // @brief Evaluate the gradient of the function at a point x
   // @param[in] x The point at which to evaluate the gradient
   // @return The value of the gradient at x
   virtual real_t grad(const real_t x) const
   { MFEM_ABORT("LegendreFunction::grad not implemented."); }

   // @brief Evaluate the gradient of the space-dependent function at a point x
   // and element Tr.
   // @note Integration point should be set in Tr before calling this function.
   // @note Unless overridden, this function ignores Tr and calls grad(x).
   // @param[in] x The point at which to evaluate the gradient
   // @param[in] Tr the element transformation with integration point
   // @return The value of the gradient at x
   virtual real_t grad(const real_t x, ElementTransformation &Tr) const
   { return this->grad(x); }

   // @brief Evaluate the gradient of the function at a point x
   // @param[in] x The point at which to evaluate the gradient
   // @param[out] g The vector to store the gradient
   virtual void grad(const Vector &x, Vector &g) const
   {
      g.SetSize(x.Size());
      for (int i=0; i<x.Size(); i++)
      {
         g[i] = this->grad(x[i]);
      }
   }

   // @brief Evaluate the gradient of the function at a point x
   // @param[in] x The point at which to evaluate the gradient
   // @param[in] Tr the element transformation with integration point
   // @param[out] g The vector to store the gradient
   virtual void grad(const Vector &x, ElementTransformation &Tr, Vector &g) const
   { this->grad(x, g); }

   // @brief Evaluate the inverse gradient of the function at a point x
   // @param[in] x The point at which to evaluate the inverse gradient
   // @return The value of the inverse gradient at x
   virtual real_t gradinv(const real_t x) const
   { MFEM_ABORT("LegendreFunction::gradinv not implemented."); }
   // @brief Evaluate the inverse gradient of the function at a point x
   // with given index i. This typcailly used in finite-dimensional context,
   // where the function varies per dimension.
   // @param[in] x The point at which to evaluate the inverse gradient
   // @return The value of the inverse gradient at x
   virtual real_t gradinv(const real_t x, int i) const
   { return this->gradinv(x); }

   // @brief Evaluate the inverse gradient of the function at a point x
   // @param[in] x The point at which to evaluate the inverse gradient
   // @param[in] Tr the element transformation with integration point
   // @return The value of the inverse gradient at x
   virtual real_t gradinv(const real_t x, ElementTransformation &Tr) const
   { return this->gradinv(x); }

   // @brief Evaluate the inverse gradient of the function at a point x
   // @param[in] x The point at which to evaluate the inverse gradient
   // @param[out] invg The vector to store the inverse gradient
   virtual void gradinv(const Vector &x, Vector &invg) const
   {
      invg.SetSize(x.Size());
      for (int i=0; i<x.Size(); i++) { invg[i] = this->gradinv(x[i]); }
   }

   // @brief Evaluate the inverse gradient of the function at a point x
   // @param[in] x The point at which to evaluate the inverse gradient
   // @param[in] Tr the element transformation with integration point
   // @param[out] invg The vector to store the inverse gradient
   virtual void gradinv(const Vector &x, ElementTransformation &Tr,
                        Vector &invg) const
   { this->gradinv(x, invg); }
   virtual void gradinv(const Vector &x, int i, Vector &invg) const
   { this->gradinv(x, invg); }

   // @brief Evaluate the inverse Hessian of the function at a point x
   // @param[in] x The point at which to evaluate the inverse Hessian
   // @return The value of the inverse Hessian at x
   virtual real_t hessinv(const real_t x) const
   { MFEM_ABORT("LegendreFunction::hessinv not implemented."); }

   // @brief Evaluate the inverse Hessian of the function at a point x
   // @param[in] x The point at which to evaluate the inverse Hessian
   // @param[in] Tr the element transformation with integration point
   // @return The value of the inverse Hessian at x
   // @note Unless overridden, this function ignores Tr and calls hessinv(x).
   virtual real_t hessinv(const real_t x, ElementTransformation &Tr) const
   { return this->hessinv(x); }
   virtual real_t hessinv(const real_t x, int i) const
   { return this->hessinv(x); }

   // @brief Evaluate the inverse Hessian of the function at a point x
   // @param[in] x The point at which to evaluate the inverse Hessian
   // @param[out] H The matrix to store the inverse Hessian
   // @note Defaults to use diagonal Hessian with H(i, i) = hessinv(x[i])
   virtual void hessinv(const Vector &x, DenseMatrix &H) const
   {
      H.SetSize(x.Size());
      for (int i=0; i<x.Size(); i++) { H(i,i) = this->hessinv(x[i]); }
   }

   // @brief Evaluate the inverse Hessian of the function at a point x
   // @param[in] x The point at which to evaluate the inverse Hessian
   // @param[in] Tr the element transformation with integration point
   // @param[out] H The matrix to store the inverse Hessian
   // @note Defaults to use diagonal Hessian with H(i, i) = hessinv(x[i])
   virtual void hessinv(const Vector &x, ElementTransformation &Tr,
                        DenseMatrix &H) const
   { this->hessinv(x, H); }
   virtual void hessinv(const Vector &x, int i,
                        DenseMatrix &H) const
   { this->hessinv(x, H); }

};

// @brief A Legendre function scaled (and shifted) by scalar parameters
// f(x) = g((x - shift) / scale)
// dom(f) = dom(g) * scale + shift
// where g is the base Legendre function
// For example,
// Shannon g(); // dom(g) = [0, 1]
// TransformedLegendreFunction f(g, 2.0, -1.0); // dom(f) = [-1, 1]
class ScaledLegendreFunction : public LegendreFunction
{
protected:
   std::unique_ptr<LegendreFunction> f;
   mutable real_t scale,
           shift; // for space-dependent scaling, marked mutable. See, CoefficientScaledLegendreFunction
   mutable Vector y;
public:
   ScaledLegendreFunction(std::unique_ptr<LegendreFunction> &func,
                          const real_t scale_=1.0, const real_t shift_=0.0)
      : LegendreFunction()
      , f(std::move(func))
      , scale(scale_), shift(shift_) {}
   ScaledLegendreFunction(LegendreFunction *func,
                          const real_t scale_=1.0, const real_t shift_=0.0)
      : LegendreFunction()
      , f(std::unique_ptr<LegendreFunction>(func))
      , scale(scale_), shift(shift_) {}

   ScaledLegendreFunction &SetScale(const real_t s) { scale = s; return *this; }
   ScaledLegendreFunction &SetShift(const real_t s) { shift = s; return *this; }
   ScaledLegendreFunction &SetAffine(const real_t s, const real_t sh)
   { scale = s; shift = sh; return *this; }
   real_t GetScale() const { return scale; }
   real_t GetShift() const { return shift; }

   ~ScaledLegendreFunction() = default;

   // @brief Evaluate the function at a point x
   // @param[in] x The point at which to evaluate the function
   // @return The value of the function at x
   real_t operator()(const real_t x) const override
   { return f->operator()((x - shift) / scale); }
   real_t operator()(const real_t x, ElementTransformation &Tr) const override
   { return f->operator()((x - shift) / scale, Tr); }

   // @brief Evaluate the gradient of the function at a point x
   // @param[in] x The point at which to evaluate the gradient
   // @return The value of the gradient at x
   real_t grad(const real_t x) const override
   { return f->grad((x - shift) / scale) / scale; }
   real_t grad(const real_t x, ElementTransformation &Tr) const override
   { return f->grad((x - shift) / scale, Tr) / scale; }

   // @brief Evaluate the inverse gradient of the function at a point x
   // @param[in] x The point at which to evaluate the inverse gradient
   // @return The value of the inverse gradient at x
   real_t gradinv(const real_t x) const override
   { return f->gradinv(scale * x) * scale + shift; }
   real_t gradinv(const real_t x, ElementTransformation &Tr) const override
   { return f->gradinv(scale * x, Tr) * scale + shift; }

   real_t hessinv(const real_t x) const override
   { return f->hessinv(x*scale) * scale * scale; }
   real_t hessinv(const real_t x, ElementTransformation &Tr) const override
   { return f->hessinv(x*scale, Tr) * scale * scale; }

   real_t operator()(const Vector &x) const override
   { y = x; y -= shift; y /= scale; return f->operator()(y); }
   real_t operator()(const Vector &x, ElementTransformation &Tr) const override
   { y = x; y -= shift; y /= scale; return f->operator()(y, Tr); }

   void grad(const Vector &x, Vector &g) const override
   { y = x; y -= shift; y /= scale; f->grad(y, g); g /= scale; }
   void grad(const Vector &x, ElementTransformation &Tr, Vector &g) const override
   { y = x; y -= shift; y /= scale; f->grad(y, Tr, g); g /= scale; }

   void gradinv(const Vector &x, Vector &invg) const override
   { y = x; y *= scale; f->gradinv(y, invg); invg *= scale; invg += shift; }
   void gradinv(const Vector &x, ElementTransformation &Tr,
                Vector &invg) const override
   { y = x; y *= scale; f->gradinv(y, Tr, invg); invg *= scale; invg += shift; }

   void hessinv(const Vector &x, DenseMatrix &H) const override
   { y = x; y *= scale; f->hessinv(y, H); H *= scale * scale; }
   void hessinv(const Vector &x, ElementTransformation &Tr,
                DenseMatrix &H) const override
   { y = x; y *= scale; f->hessinv(y, Tr, H); H *= scale * scale; }
};

class CoefficientScaledLegendreFunction : public ScaledLegendreFunction
{
   Coefficient * scale_cf;
   Coefficient * shift_cf;
   bool own_scale = true;
   bool own_shift = true;
   void update_scale_shift(ElementTransformation &Tr) const
   {
      MFEM_ASSERT(scale_cf != nullptr && shift_cf != nullptr,
                  "Scale and shift Coefficients must be set.");
      scale = scale_cf->Eval(Tr, Tr.GetIntPoint());
      shift = shift_cf->Eval(Tr, Tr.GetIntPoint());
   }
public:
   CoefficientScaledLegendreFunction(std::unique_ptr<LegendreFunction> &func)
      : ScaledLegendreFunction(func, 1.0, 0.0)
      , scale_cf(new ConstantCoefficient(1.0)),
        shift_cf(new ConstantCoefficient(0.0)) {}
   CoefficientScaledLegendreFunction(LegendreFunction *func)
      : ScaledLegendreFunction(func, 1.0, 0.0)
      , scale_cf(new ConstantCoefficient(1.0)),
        shift_cf(new ConstantCoefficient(0.0)) {}
   CoefficientScaledLegendreFunction(std::unique_ptr<LegendreFunction> &func,
                                     Coefficient& scale_, Coefficient &shift_)
      : ScaledLegendreFunction(func, 1.0, 0.0)
      , scale_cf(&scale_), shift_cf(&shift_) {}
   CoefficientScaledLegendreFunction(std::unique_ptr<LegendreFunction> &func,
                                     Coefficient* scale_, Coefficient *shift_,
                                     bool own_scale_ = false,
                                     bool own_shift_ = false)
      : ScaledLegendreFunction(func, 1.0, 0.0)
      , scale_cf(scale_), shift_cf(shift_)
      , own_scale(own_scale_), own_shift(own_shift_)
   {
      MFEM_VERIFY(scale_cf != nullptr || own_scale,
                  "Scale Coefficient pointer is null but own_scale is false.");
      if (!scale_) { scale_cf = new ConstantCoefficient(1.0); own_scale = true; }
      if (!shift_) { shift_cf = new ConstantCoefficient(0.0); own_shift = true; }
   }
   CoefficientScaledLegendreFunction(LegendreFunction *func,
                                     Coefficient &scale_, Coefficient &shift_)
      : ScaledLegendreFunction(func)
      , scale_cf(&scale_), shift_cf(&shift_)
      , own_scale(false), own_shift(false)
   {}

   CoefficientScaledLegendreFunction &SetScale(const real_t s)
   {
      scale_cf = new ConstantCoefficient(s);
      own_scale = true;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetScale(Coefficient &s)
   {
      scale_cf = &s;
      own_scale = false;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetScale(Coefficient *s, bool own = false)
   {
      scale_cf = s;
      own_scale = own;
      return *this;
   }

   CoefficientScaledLegendreFunction &SetShift(const real_t s)
   {
      shift_cf = new ConstantCoefficient(s);
      own_shift = true;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetShift(Coefficient &s)
   {
      shift_cf = &s;
      own_shift = false;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetShift(Coefficient *s, bool own = false)
   {
      shift_cf = s;
      own_shift = own;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetAffine(const real_t s, const real_t sh)
   { return SetScale(s).SetShift(sh);}
   CoefficientScaledLegendreFunction &SetAffine(Coefficient &s, Coefficient &sh)
   { return SetScale(s).SetShift(sh); }
   CoefficientScaledLegendreFunction &SetAffine(Coefficient *s, Coefficient *sh,
         bool own_scale_ = false, bool own_shift_ = false)
   { return SetScale(s, own_scale_).SetShift(sh, own_shift_); }

   ~CoefficientScaledLegendreFunction()
   {
      if (own_scale && scale_cf) { delete scale_cf; }
      if (own_shift && shift_cf) { delete shift_cf; }
   }

   // @brief Evaluate the function at a point x
   // @param[in] x The point at which to evaluate the function
   // @return The value of the function at x
   real_t operator()(const real_t x) const override
   { return ScaledLegendreFunction::operator()(x); }
   real_t operator()(const real_t x, ElementTransformation &Tr) const override
   { update_scale_shift(Tr); return ScaledLegendreFunction::operator()(x, Tr); }
   real_t operator()(const Vector &x) const override
   { return ScaledLegendreFunction::operator()(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr) const override
   { update_scale_shift(Tr); return ScaledLegendreFunction::operator()(x, Tr); }

   // @brief Evaluate the gradient of the function at a point x
   // @param[in] x The point at which to evaluate the gradient
   // @return The value of the gradient at x
   real_t grad(const real_t x) const override
   { return ScaledLegendreFunction::grad(x); }
   real_t grad(const real_t x, ElementTransformation &Tr) const override
   { update_scale_shift(Tr); return ScaledLegendreFunction::grad(x); }
   void grad(const Vector &x, Vector &g) const override
   { return ScaledLegendreFunction::grad(x, g); }
   void grad(const Vector &x, ElementTransformation &Tr, Vector &g) const override
   { update_scale_shift(Tr); return ScaledLegendreFunction::grad(x, g); }


   // @brief Evaluate the inverse gradient of the function at a point x
   // @param[in] x The point at which to evaluate the inverse gradient
   // @return The value of the inverse gradient at x
   real_t gradinv(const real_t x) const override
   { return ScaledLegendreFunction::gradinv(x); }
   real_t gradinv(const real_t x, ElementTransformation &Tr) const override
   { update_scale_shift(Tr); return ScaledLegendreFunction::gradinv(x); }
   void gradinv(const Vector &x, Vector &g) const override
   { return ScaledLegendreFunction::gradinv(x, g); }
   void gradinv(const Vector &x, ElementTransformation &Tr,
                Vector &g) const override
   { update_scale_shift(Tr); return ScaledLegendreFunction::gradinv(x, g); }

   real_t hessinv(const real_t x) const override
   { return ScaledLegendreFunction::hessinv(x); }
   real_t hessinv(const real_t x, ElementTransformation &Tr) const override
   { update_scale_shift(Tr); return ScaledLegendreFunction::hessinv(x); }
   void hessinv(const Vector &x, DenseMatrix &H) const override
   { return ScaledLegendreFunction::hessinv(x, H); }
   void hessinv(const Vector &x, ElementTransformation &Tr,
                DenseMatrix &H) const override
   { update_scale_shift(Tr); return ScaledLegendreFunction::hessinv(x, H); }
};

class TranslatedLegendreFunction : public LegendreFunction
{
protected:
   std::unique_ptr<LegendreFunction> f;
   mutable DenseMatrix V;
   mutable Vector shift;
   mutable Vector Vtx;
   mutable Vector invgVtx;
public:
   TranslatedLegendreFunction(std::unique_ptr<LegendreFunction> &func,
                              const DenseMatrix V_, const Vector shift_= {})
      : LegendreFunction()
      , f(std::move(func))
      , V(V_), shift(shift_)
   {
      if (shift.Size() == 0)
      {
         shift.SetSize(V.Height());
         shift = 0.0;
      }
   }

   TranslatedLegendreFunction(LegendreFunction *func,
                              const DenseMatrix V_, const Vector shift_= {})
      : LegendreFunction()
      , f(std::unique_ptr<LegendreFunction>(func))
      , V(V_), shift(shift_)
   {
      if (shift.Size() == 0)
      {
         shift.SetSize(V.Height());
         shift = 0.0;
      }
   }

   TranslatedLegendreFunction &SetV(const DenseMatrix &V_) { V = V_; return *this; }
   TranslatedLegendreFunction &SetShift(const Vector &shift_) { shift = shift_; return *this; }
   TranslatedLegendreFunction &SetAffine(const DenseMatrix &V_,
                                         const Vector &shift_)
   { return SetV(V_).SetShift(shift_); }
   const DenseMatrix &GetV() const { return V; }
   const Vector &GetShift() const { return shift; }

   ~TranslatedLegendreFunction() = default;

   // @brief Evaluate the function at a point x
   // @param[in] x The point at which to evaluate the function
   // @return The value of the function at x

   void gradinv(const Vector &x, Vector &invg) const override
   {
      V.MultTranspose(x, Vtx);
      f->gradinv(Vtx, invgVtx);
      V.Mult(invgVtx, invg);
   }
   void gradinv(const Vector &x, ElementTransformation &Tr,
                Vector &invg) const override
   { this->gradinv(x, invg); }

   // void hessinv(const Vector &x, DenseMatrix &H) const override
   // { y = x; y *= scale; f->hessinv(y, H); H *= scale * scale; }
   // void hessinv(const Vector &x, ElementTransformation &Tr,
   //              DenseMatrix &H) const override
   // { y = x; y *= scale; f->hessinv(y, Tr, H); H *= scale * scale; }
};

class SeperableLegendreFunction : public LegendreFunction
{
   std::vector<std::unique_ptr<LegendreFunction>> funcs;
   Array<int> offsets;
   mutable Vector x_sub, g_sub, invg_sub;
public:
   // @brief A Legendre function that is the sum of several Legendre functions
   // f(x) = sum_i g_i(x_i), here, x_i is a subvector of x
   // Use AddFunction(g, dim) to add Legendre functions with given input dimension
   SeperableLegendreFunction(): funcs(), offsets({0}) {}
   ~SeperableLegendreFunction() = default;

   SeperableLegendreFunction &AddFunction(std::unique_ptr<LegendreFunction> &func,
                                          int dim=1)
   {
      MFEM_VERIFY(func.get() != nullptr, "Null pointer passed to AddFunction.");
      offsets.Append(offsets.Last() + dim);
      funcs.push_back(std::move(func));
      return *this;
   }

   // @brief Add a Legendre function with given dimension (defaults to 1)
   // @note The ownership of the function will be moved to the SeperableLegendreFunction
   // @param[in] func The Legendre function to add. The ownership will be moved.
   // @param[in] dim The dimension of the input space of the function
   SeperableLegendreFunction &AddFunction(LegendreFunction *func, int dim=1)
   {
      MFEM_VERIFY(func != nullptr, "Null pointer passed to AddFunction.");
      offsets.Append(offsets.Last() + dim);
      funcs.push_back(std::unique_ptr<LegendreFunction>(func));
      return *this;
   }

   real_t operator()(const Vector &x, ElementTransformation &Tr) const override
   {
      real_t sum = 0.0;
      for (int i=0; i<funcs.size(); i++)
      {
         const int dim = offsets[i+1] - offsets[i];
         x_sub.MakeRef(const_cast<Vector&>(x), offsets[i], dim);
         sum += funcs[i]->operator()(x_sub, Tr);
      }
      return sum;
   }
   void grad(const Vector &x, ElementTransformation &Tr, Vector &g) const override
   {
      g.SetSize(x.Size());
      for (int i=0; i<funcs.size(); i++)
      {
         const int dim = offsets[i+1] - offsets[i];
         x_sub.MakeRef(const_cast<Vector&>(x), offsets[i], dim);
         g_sub.MakeRef(g, offsets[i], dim);
         funcs[i]->grad(x_sub, Tr, g_sub);
      }
   }
   void gradinv(const Vector &x, ElementTransformation &Tr,
                Vector &invg) const override
   {
      invg.SetSize(x.Size());
      for (int i=0; i<funcs.size(); i++)
      {
         const int dim = offsets[i+1] - offsets[i];
         x_sub.MakeRef(const_cast<Vector&>(x), offsets[i], dim);
         invg_sub.MakeRef(invg, offsets[i], dim);
         funcs[i]->gradinv(x_sub, Tr, invg_sub);
      }
   }
};



class Shannon : public LegendreFunction
{
   real_t tol;
public:
   Shannon(real_t tol=1e-10) : tol(tol) {}
   ~Shannon() = default;
   using LegendreFunction::operator();
   real_t operator()(real_t x) const override
   {
      return x > tol ? x * std::log(x) - x : 0.0;
   }
   using LegendreFunction::grad;
   real_t grad(const real_t x) const override
   {
      return x > tol ? std::log(x) : std::log(tol);
   }
   using LegendreFunction::gradinv;
   real_t gradinv(const real_t x) const override
   {
      return std::exp(x);
   }
   using LegendreFunction::hessinv;
   real_t hessinv(const real_t x) const override
   {
      return std::exp(x);
   }
};

class FermiDirac : public LegendreFunction
{
   real_t tol;
public:
   FermiDirac(real_t tol=1e-10) : tol(tol) {}
   ~FermiDirac() = default;
   using LegendreFunction::operator();
   real_t operator()(real_t x) const override
   {
      return (x > tol ? x < 1.0 - tol ? x * std::log(x) + (1.0 - x) * std::log(
                 1.0 - x) : 0.0 : 0.0);
   }
   using LegendreFunction::grad;
   real_t grad(const real_t x) const override
   {
      return (
                x > tol
                ? x < 1.0 - tol
                ? std::log(x / (1.0 - x))
                : std::log((1.0 - tol) / tol)
                : std::log(tol / (1.0 - tol)));
   }
   using LegendreFunction::gradinv;
   real_t gradinv(const real_t x) const override
   {
      return x > 0.0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
   }
   using LegendreFunction::hessinv;
   real_t hessinv(const real_t x) const override
   {
      real_t sig = gradinv(x);
      MFEM_VERIFY(sig >= 0.0 && sig <= 1.0,
                  "Fermi-Dirac inverse gradient out of bounds.");
      return sig*(1.0-sig);
   }
};

class PointwiseFermiDirac : public FermiDirac
{
   const Vector &xmin;
   const Vector &xmax;
public:
   PointwiseFermiDirac(const Vector &xmin_, const Vector &xmax_,
                       real_t tol=1e-10)
      : FermiDirac(tol)
      , xmin(xmin_), xmax(xmax_)
   {}
   ~PointwiseFermiDirac() = default;
   using FermiDirac::operator();
   using FermiDirac::grad;
   using FermiDirac::gradinv;
   real_t gradinv(const real_t x, int i) const override
   {
      real_t val = FermiDirac::grad(x);
      return xmin[i] + (xmax[i] - xmin[i]) * val;

   }
   using FermiDirac::hessinv;
   real_t hessinv(const real_t x, int i) const override
   {
      real_t val = FermiDirac::hessinv(x);
      return (xmax[i] - xmin[i]) * val;
   }
};

class Hellinger : public LegendreFunction
{
public:
   Hellinger() {}
   ~Hellinger() = default;
   using LegendreFunction::operator();
   real_t operator()(const Vector &x) const override
   {
      return -std::sqrt(std::max(1.0 - (x*x), 0.0));
   }
   using LegendreFunction::grad;
   void grad(const Vector &x, Vector &g) const override
   {
      g = x;
      g /= std::sqrt(1.0 - std::max(1.0 - (x*x), 0.0));
   }
   using LegendreFunction::gradinv;
   void gradinv(const Vector &x, Vector &invg) const override
   {
      invg = x;
      invg /= std::sqrt(1.0 + (x*x));
   }
   using LegendreFunction::hessinv;
   void hessinv(const Vector &x, DenseMatrix &H) const override
   {
      real_t denom = std::sqrt(1.0 + (x*x));
      H.Diag(1.0 / denom, x.Size());
      AddMult_a_VVt(-1.0 / std::pow(denom, 3.0), x, H);
   }
};

class Gibbs : public LegendreFunction
{
   real_t tol;
   mutable Vector y;
public:
   Gibbs(real_t tol=1e-10) : tol(tol) {}
   ~Gibbs() = default;

   using LegendreFunction::operator();
   real_t operator()(const Vector &x) const override;
   using LegendreFunction::grad;
   void grad(const Vector &x, Vector &g) const override;
   using LegendreFunction::gradinv;
   void gradinv(const Vector &x, Vector &invg) const override;
   using LegendreFunction::hessinv;
   void hessinv(const Vector &x, DenseMatrix &H) const override;
};
class PointwiseGibbs : public Gibbs
{
   const Array<DenseMatrix*> &Vs;
   mutable Vector y_buffer;
   mutable Vector invgy_buffer;
   mutable Vector y;
   mutable Vector invgy;
   mutable Vector H_buffer;
   mutable Vector VH_buffer;
   mutable DenseMatrix invhessy;
   mutable DenseMatrix Vinvhessy;
public:
   PointwiseGibbs(const Array<DenseMatrix*> &Vs_, real_t tol=1e-10);
   ~PointwiseGibbs() = default;

   using Gibbs::operator();
   using Gibbs::grad;
   using Gibbs::gradinv;
   void gradinv(const Vector &x, int i, Vector &invg) const override;
   using Gibbs::hessinv;
   void hessinv(const Vector &x, int i, DenseMatrix &H) const override;
};

class PrimalCoefficient : public Coefficient
{
   LegendreFunction &f;
   std::unique_ptr<Coefficient> latent;
public:
   PrimalCoefficient(GridFunction &psi, LegendreFunction &func)
      : f(func), latent(std::make_unique<GridFunctionCoefficient>(&psi)) {}
   PrimalCoefficient(QuadratureFunction &psi, LegendreFunction &func)
      : f(func), latent(std::make_unique<QuadratureFunctionCoefficient>(psi)) {}
   virtual ~PrimalCoefficient() {}
   real_t Eval(ElementTransformation &Tr,
               const IntegrationPoint &ip) override
   {
      return f.gradinv(latent->Eval(Tr, ip), Tr);
   }
};
class PrimalJacobianCoefficient : public Coefficient
{
   LegendreFunction &f;
   std::unique_ptr<Coefficient> latent;
public:
   PrimalJacobianCoefficient(GridFunction &psi, LegendreFunction &func)
      : f(func), latent(std::make_unique<GridFunctionCoefficient>(&psi)) {}
   PrimalJacobianCoefficient(QuadratureFunction &psi, LegendreFunction &func)
      : f(func), latent(std::make_unique<QuadratureFunctionCoefficient>(psi)) {}
   virtual ~PrimalJacobianCoefficient() {}
   real_t Eval(ElementTransformation &Tr,
               const IntegrationPoint &ip) override
   {
      return f.hessinv(latent->Eval(Tr, ip), Tr);
   }
};

class PrimalVectorCoefficient : public VectorCoefficient
{
   LegendreFunction &f;
   std::unique_ptr<VectorCoefficient> latent;
   mutable Vector val;
public:
   PrimalVectorCoefficient(GridFunction &psi, LegendreFunction &func)
      : VectorCoefficient(psi.VectorDim()), f(func)
   {
      latent = std::make_unique<VectorGridFunctionCoefficient>(&psi);
   }
   PrimalVectorCoefficient(QuadratureFunction &psi, LegendreFunction &func)
      : VectorCoefficient(psi.GetVDim()), f(func)
   {
      latent = std::make_unique<VectorQuadratureFunctionCoefficient>(psi);
   }
   virtual ~PrimalVectorCoefficient() {}
   void Eval(Vector &u, ElementTransformation &Tr,
             const IntegrationPoint &ip) override
   {
      u.SetSize(GetVDim());
      latent->Eval(val, Tr, ip);
      f.gradinv(val, Tr, u);
   }
};
class PrimalVectorJacobianCoefficient : public MatrixCoefficient
{
   LegendreFunction &f;
   std::unique_ptr<VectorCoefficient> latent;
   mutable Vector val;
public:
   PrimalVectorJacobianCoefficient(GridFunction &psi, LegendreFunction &func)
      : MatrixCoefficient(psi.VectorDim()), f(func)
   {
      latent = std::make_unique<VectorGridFunctionCoefficient>(&psi);
   }

   PrimalVectorJacobianCoefficient(QuadratureFunction &psi, LegendreFunction &func)
      : MatrixCoefficient(psi.GetVDim()), f(func)
   {
      latent = std::make_unique<VectorQuadratureFunctionCoefficient>(psi);
   }
   virtual ~PrimalVectorJacobianCoefficient() {}
   void Eval(DenseMatrix &H, ElementTransformation &Tr,
             const IntegrationPoint &ip) override
   {
      H.SetSize(GetHeight());
      latent->Eval(val, Tr, ip);
      f.hessinv(val, Tr, H);
   }
};

class LegendreFunctional : public Operator
{
protected:
   LegendreFunction &f;
public:
   LegendreFunctional(LegendreFunction &func, int size)
      : Operator(size), f(func) {}
   virtual ~LegendreFunctional() = default;
};

class LegendreFEFunctional : public LegendreFunctional
{
   bool use_lambda_form = false;
   real_t &alpha;
   FiniteElementSpace &fes;
   std::unique_ptr<GridFunction> latent_gf;
   mutable Vector latent_x;
   std::unique_ptr<PrimalCoefficient> primal_cf;
   std::unique_ptr<PrimalJacobianCoefficient> dprimal_cf;
   std::unique_ptr<PrimalVectorCoefficient> vector_primal_cf;
   std::unique_ptr<PrimalVectorJacobianCoefficient> vector_dprimal_cf;
   std::unique_ptr<LinearForm> grad;
   std::unique_ptr<BilinearForm> hess;
#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
   bool IsParallel() const { return comm != MPI_COMM_NULL; }
   mutable std::unique_ptr<HypreParMatrix> hess_mat;
#else
   bool IsParallel() const { return false; }
#endif
public:
   void UseLambdaForm() { use_lambda_form = true; };
   LegendreFEFunctional(LegendreFunction &func,
                        FiniteElementSpace &fes_,
                        real_t &alpha_,
                        bool use_lambda_form_=false)
      : LegendreFunctional(func, fes_.GetTrueVSize())
      , fes(fes_)
      , alpha(alpha_)
      , use_lambda_form(use_lambda_form_)
   {
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace *>(&fes);
      if (pfes)
      {
         comm = pfes->GetComm();
         grad = std::make_unique<ParLinearForm>(pfes);
         hess = std::make_unique<ParBilinearForm>(pfes);
      }
#endif
      if (!IsParallel())
      {
         grad = std::make_unique<LinearForm>(&fes);
         hess = std::make_unique<BilinearForm>(&fes);
      }
      latent_gf = std::make_unique<GridFunction>(&fes);
      if (fes.GetVectorDim() == 1)
      {
         primal_cf = std::make_unique<PrimalCoefficient>(*latent_gf, f);
         dprimal_cf = std::make_unique<PrimalJacobianCoefficient>(*latent_gf, f);
         grad->AddDomainIntegrator(new DomainLFIntegrator(*primal_cf));
         hess->AddDomainIntegrator(new MassIntegrator(*dprimal_cf));
      }
      else
      {
         vector_primal_cf = std::make_unique<PrimalVectorCoefficient>(*latent_gf, f);
         vector_dprimal_cf = std::make_unique<PrimalVectorJacobianCoefficient>
                             (*latent_gf, f);
         auto maptype = fes.GetTypicalFE()->GetMapType();
         if (maptype == FiniteElement::MapType::VALUE ||
             maptype == FiniteElement::MapType::INTEGRAL)
         {
            grad->AddDomainIntegrator(new VectorDomainLFIntegrator(*vector_primal_cf));
            auto hess_intg = new VectorMassIntegrator(*vector_dprimal_cf);
            hess_intg->SetVDim(vector_primal_cf->GetVDim());
            hess->AddDomainIntegrator(hess_intg);
         }
         else
         {
            grad->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*vector_primal_cf));
            hess->AddDomainIntegrator(new VectorFEMassIntegrator(*vector_dprimal_cf));
         }
      }
   }
   void SetLatentSolution(const Vector &x)
   {
      if (!use_lambda_form)
      {
         MFEM_WARNING("LegendreFEFunctional::SetLatentSolution: "
                      "Setting latent solution without using lambda form is redundant. "
                      "The latent solution will be set in Mult().");
      }
      MFEM_VERIFY(x.Size() == fes.GetTrueVSize(),
                  "Size mismatch in SetLatentSolution. "
                  << "Use true vector of size " << fes.GetTrueVSize()
                  << ", got size " << x.Size() << ".");
      latent_x = x;
   }
   virtual ~LegendreFEFunctional() = default;
   void Mult(const Vector &x, Vector &y) const override
   {
      if (use_lambda_form)
      {
         add(latent_x, alpha, x, latent_gf->GetTrueVector());
         latent_gf->SetFromTrueVector();
      }
      else
      {
         latent_gf->SetFromTrueDofs(x);
      }
      if (!IsParallel())
      {
         y.SetSize(x.Size());
         grad->SetData(y.GetData());
         grad->Assemble();
      }
      else
      {
#ifdef MFEM_USE_MPI
         auto pgrad = dynamic_cast<ParLinearForm*>(grad.get());
         MFEM_VERIFY(pgrad != nullptr, "Failed to cast grad to ParLinearForm.");
         y.SetSize(x.Size());
         pgrad->Assemble();
         pgrad->ParallelAssemble(y);
#else
         MFEM_ABORT("This should never be reached.");
#endif
      }
   }
   Operator &GetGradient(const Vector &x) const override
   {
      if (use_lambda_form)
      {
         add(latent_x, alpha, x, latent_gf->GetTrueVector());
         latent_gf->SetFromTrueVector();
      }
      else
      {
         latent_gf->SetFromTrueDofs(x);
      }
      if (!IsParallel())
      {
         hess->Assemble();
         hess->Finalize();
         hess->SpMat() *= alpha;
         return hess->SpMat();
      }
      else
      {
#ifdef MFEM_USE_MPI
         auto phess = dynamic_cast<ParBilinearForm*>(hess.get());
         MFEM_VERIFY(phess != nullptr, "Failed to cast hess to ParBilinearForm.");
         phess->Assemble();
         phess->Finalize();
         phess->SpMat() *= alpha;
         hess_mat.reset(phess->ParallelAssemble());
         return *hess_mat;
#endif
      }
   }
};
class LegendreQFunctional : public LegendreFunctional
{
   bool use_lambda_form = false;
   real_t &alpha;
   QuadratureSpace &qs;
   const int vdim;
   std::unique_ptr<QuadratureFunction> latent_qf;
   mutable Vector latent_x;
   std::unique_ptr<PrimalCoefficient> primal_cf;
   std::unique_ptr<PrimalJacobianCoefficient> dprimal_cf;
   std::unique_ptr<PrimalVectorCoefficient> vector_primal_cf;
   std::unique_ptr<PrimalVectorJacobianCoefficient> vector_dprimal_cf;
   mutable Array<int> i= {};
   mutable Array<int> j= {};
   mutable Vector d= {};
   mutable std::unique_ptr<SparseMatrix> hess;
#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
   bool IsParallel() const { return comm != MPI_COMM_NULL; }
   mutable std::unique_ptr<HypreParMatrix> hess_mat;
   mutable std::unique_ptr<HypreParMatrix> hess_par;
#else
   bool IsParallel() const { return false; }
#endif
public:
   void UseLambdaForm() { use_lambda_form = true; };
   LegendreQFunctional(LegendreFunction &func,
                       QuadratureSpace &qs_,
                       const int vdim,
                       real_t &alpha_,
                       bool use_lambda_form_=false)
      : LegendreFunctional(func, qs_.GetSize())
      , qs(qs_)
      , vdim(vdim)
      , alpha(alpha_)
      , use_lambda_form(use_lambda_form_)
   {
      latent_qf = std::make_unique<QuadratureFunction>(qs);
      if (vdim == 1)
      {
         primal_cf = std::make_unique<PrimalCoefficient>(*latent_qf, f);
         dprimal_cf = std::make_unique<PrimalJacobianCoefficient>(*latent_qf, f);
      }
      else
      {
         vector_primal_cf = std::make_unique<PrimalVectorCoefficient>(*latent_qf, f);
         vector_dprimal_cf = std::make_unique<PrimalVectorJacobianCoefficient>
                             (*latent_qf, f);
      }
   }
   void SetLatentSolution(const Vector &x)
   {
      if (!use_lambda_form)
      {
         MFEM_WARNING("LegendreFEFunctional::SetLatentSolution: "
                      "Setting latent solution without using lambda form is redundant. "
                      "The latent solution will be set in Mult().");
      }
      MFEM_VERIFY(x.Size() == qs.GetSize(),
                  "Size mismatch in SetLatentSolution. "
                  << "Use true vector of size " << qs.GetSize()
                  << ", got size " << x.Size() << ".");
      latent_x = x;
   }
   virtual ~LegendreQFunctional() = default;
   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == qs.GetSize()*vdim,
                  "Size mismatch in LegendreQFunctional::Mult. "
                  << "Expected size " << qs.GetSize()*vdim
                  << ", got size " << x.Size() << ".");
      if (use_lambda_form)
      {
         add(latent_x, alpha, x, *latent_qf);
      }
      else
      {
         *latent_qf = x;
      }
      y.SetSize(x.Size());
      if (vdim == 1)
      {
         const Vector &w = qs.GetWeights();
         int offset = 0;
         for (int i=0; i<qs.GetNE(); i++)
         {
            ElementTransformation &Tr = *qs.GetTransformation(i);
            const IntegrationRule &ir = qs.GetElementIntRule(i);
            for (int j=0; j<ir.GetNPoints(); j++, offset++)
            {
               const IntegrationPoint &ip = ir.IntPoint(j);
               Tr.SetIntPoint(&ip);
               const real_t val = f.gradinv((*latent_qf)[offset], Tr);
               y[offset] = val * w[offset];
            }
         }
      }
      else
      {
         const Vector &w = qs.GetWeights();
         int offset = 0;
         Vector psi_view(vdim), y_view(vdim);
         for (int i=0; i<qs.GetNE(); i++)
         {
            ElementTransformation &Tr = *qs.GetTransformation(i);
            const IntegrationRule &ir = qs.GetElementIntRule(i);
            for (int j=0; j<ir.GetNPoints(); j++)
            {
               const IntegrationPoint &ip = ir.IntPoint(j);
               Tr.SetIntPoint(&ip);
               psi_view.SetData(&(*latent_qf)[offset]);
               y_view.SetData(&y[offset]);
               f.gradinv(psi_view, Tr, y_view);
               y_view *= w[offset];
               offset += vdim;
            }
         }
      }
   }

   Operator &GetGradient(const Vector &x) const override
   {
      if (use_lambda_form)
      {
         add(latent_x, alpha, x, *latent_qf);
      }
      else
      {
         *latent_qf = x;
      }
      d.SetSize(qs.GetSize()*vdim);
      const Vector &w = qs.GetWeights();
      if (vdim == 1)
      {
         for (int i=0; i<d.Size(); i++)
         {
            d[i] = f.hessinv((*latent_qf)[i])*w[i];
         }
      }
      else
      {
         MFEM_ABORT("Not implemented yet.");
      }
      d *= alpha;
      if (i.IsEmpty())
      {
         // Construct CSR format for diagonal matrix
         i.SetSize(d.Size() + 1);
         std::iota(i.begin(), i.end(), 0);
      }
      if (IsParallel())
      {
         MFEM_ABORT("Not implemented yet.");
      }
      else
      {
         if (j.IsEmpty())
         {
            j.SetSize(d.Size());
            std::iota(j.begin(), j.end(), 0);
            hess = std::make_unique<SparseMatrix>(i.GetData(), j.GetData(), d.GetData(),
                                                  d.Size(), d.Size(), false,
                                                  false, true);
         }
      }
      return *hess;
   }
};


} // namespace mfem
