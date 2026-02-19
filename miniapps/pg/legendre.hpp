#pragma once

#include "mfem.hpp"

namespace mfem
{

class ScaledLegendreFunction; // f(x) = g((x - shift) / scale), dom(f) = dom(g)*scale + shift
class CoefficientScaledLegendreFunction; // f(x) = g((x - shift(x)) / scale(x))
class TransformedLegendreFunction; // f(x) = g(A(x - b)), dom(f) = A*dom(g) + b
class Euclidean;
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
      if (scale_cf && own_scale) { delete scale_cf; }
      scale_cf = new ConstantCoefficient(s);
      own_scale = true;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetScale(Coefficient &s)
   {
      if (scale_cf && own_scale) { delete scale_cf; }
      scale_cf = &s;
      own_scale = false;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetScale(Coefficient *s, bool own = false)
   {
      if (scale_cf && own_scale) { delete scale_cf; }
      scale_cf = s;
      own_scale = own;
      return *this;
   }

   CoefficientScaledLegendreFunction &SetShift(const real_t s)
   {
      if (shift_cf && own_shift) { delete shift_cf; }
      shift_cf = new ConstantCoefficient(s);
      own_shift = true;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetShift(Coefficient &s)
   {
      if (shift_cf && own_shift) { delete shift_cf; }
      shift_cf = &s;
      own_shift = false;
      return *this;
   }
   CoefficientScaledLegendreFunction &SetShift(Coefficient *s, bool own = false)
   {
      if (shift_cf && own_shift) { delete shift_cf; }
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


class Euclidean : public LegendreFunction
{
public:
   Euclidean() {}
   ~Euclidean() = default;
   using LegendreFunction::operator();
   real_t operator()(real_t x) const override
   {
      return x*x*0.5;
   }
   using LegendreFunction::grad;
   real_t grad(const real_t x) const override
   {
      return x;
   }
   using LegendreFunction::gradinv;
   real_t gradinv(const real_t x) const override
   {
      return x;
   }
   using LegendreFunction::hessinv;
   real_t hessinv(const real_t x) const override
   {
      return 1;
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
      return x > 0.0 ? 1.0 / (1.0 + std::exp(-x)) : 1.0 / (1.0 + std::exp(-x));
   }
   using LegendreFunction::hessinv;
   real_t hessinv(const real_t x) const override
   {
      real_t sig = gradinv(x);
      return sig*(1.0-sig);
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

class PrimalCoefficient : public Coefficient
{
   const LegendreFunction &f;
   Coefficient *psi_cf;
   bool own_psi_cf;
   // GridFunction &latent;
public:
   PrimalCoefficient(GridFunction &psi, const LegendreFunction &func)
      : f(func), psi_cf(new GridFunctionCoefficient(&psi)), own_psi_cf(true) {}
   PrimalCoefficient(QuadratureFunction &psi, const LegendreFunction &func)
      : f(func), psi_cf(new QuadratureFunctionCoefficient(psi)), own_psi_cf(true) {}
   PrimalCoefficient(Coefficient &psi, const LegendreFunction &func)
      : f(func), psi_cf(&psi), own_psi_cf(false) {}
   virtual ~PrimalCoefficient() { if (own_psi_cf && psi_cf) { delete psi_cf; } }
   real_t Eval(ElementTransformation &Tr,
               const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(psi_cf != nullptr, "Psi Coefficient is null.");
      return f.gradinv(psi_cf->Eval(Tr, ip), Tr);
   }
};
class PrimalGradientCoefficient : public Coefficient
{
   const LegendreFunction &f;
   Coefficient *psi_cf;
   bool own_psi_cf;
   // GridFunction &latent;
public:
   PrimalGradientCoefficient(GridFunction &psi, const LegendreFunction &func)
      : f(func), psi_cf(new GridFunctionCoefficient(&psi)), own_psi_cf(true) {}
   PrimalGradientCoefficient(QuadratureFunction &psi, const LegendreFunction &func)
      : f(func), psi_cf(new QuadratureFunctionCoefficient(psi)), own_psi_cf(true) {}
   PrimalGradientCoefficient(Coefficient &psi, const LegendreFunction &func)
      : f(func), psi_cf(&psi), own_psi_cf(false) {}
   virtual ~PrimalGradientCoefficient() { if (own_psi_cf && psi_cf) { delete psi_cf; } }
   real_t Eval(ElementTransformation &Tr,
               const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(psi_cf != nullptr, "Psi Coefficient is null.");
      return f.hessinv(psi_cf->Eval(Tr, ip), Tr);
   }
};

class PrimalVectorCoefficient : public VectorCoefficient
{
   const LegendreFunction &f;
   VectorCoefficient *latent;
   bool own_coeff;
   mutable Vector val;
public:
   PrimalVectorCoefficient(GridFunction &psi, const LegendreFunction &func)
      : VectorCoefficient(psi.VectorDim()), f(func),
        latent(new VectorGridFunctionCoefficient(&psi)), own_coeff(true) {}
   PrimalVectorCoefficient(VectorCoefficient &psi, const LegendreFunction &func)
      : VectorCoefficient(psi.GetVDim()), f(func),
        latent(&psi), own_coeff(false) {}
   virtual ~PrimalVectorCoefficient() { if (own_coeff && latent) { delete latent; } }
   void Eval(Vector &u, ElementTransformation &Tr,
             const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(latent != nullptr, "Latent VectorCoefficient is null.");
      u.SetSize(GetVDim());
      latent->Eval(val, Tr, ip);
      f.gradinv(val, Tr, u);
   }
};
class PrimalGradientVectorCoefficient : public MatrixCoefficient
{
   const LegendreFunction &f;
   VectorCoefficient *latent;
   bool own_coeff;
   mutable Vector val;
public:
   PrimalGradientVectorCoefficient(GridFunction &psi, const LegendreFunction &func)
      : MatrixCoefficient(psi.VectorDim()), f(func),
        latent(new VectorGridFunctionCoefficient(&psi)), own_coeff(true) {}
   PrimalGradientVectorCoefficient(VectorCoefficient &psi,
                                   const LegendreFunction &func)
      : MatrixCoefficient(psi.GetVDim()), f(func),
        latent(&psi), own_coeff(false) {}
   virtual ~PrimalGradientVectorCoefficient() { if (own_coeff && latent) { delete latent; } }
   void Eval(DenseMatrix &H, ElementTransformation &Tr,
             const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(latent != nullptr, "Latent VectorCoefficient is null.");
      H.SetSize(GetVDim());
      latent->Eval(val, Tr, ip);
      f.hessinv(val, Tr, H);
   }
};


} // namespace mfem
