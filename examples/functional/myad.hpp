#ifndef MFEM_HESS_AD_HPP
#define MFEM_HESS_AD_HPP
#include "mfem.hpp"
#include "miniapps/autodiff/admfem.hpp"

// Provide a make_unique implementation for C++11
#if __cplusplus <= 201103L
#ifndef MAKE_UNIQUE_EXISTS
#define MAKE_UNIQUE_EXISTS
#include <memory>
#include <utility>

// define make_unique for C++11
namespace std
{
template<typename T, typename... Args>
std::unique_ptr<T> make_unique( Args&&... args )
{
   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}
#endif
#endif // __cplusplus <= 201103L

namespace mfem
{
/** @brief implementation of max function for dual numbers */
template <typename value_type> MFEM_HOST_DEVICE
inline internal::dual<value_type, real_t> max(
   internal::dual<value_type, real_t> a, value_type b)
{
   if (a.value > b)
   {
      return a;
   }
   else if (a.value < b)
   {
      return internal::dual<value_type, real_t> {b, 0.0};
   }
   else
   {
      // At a.value == b: derivative is undefined — choose subgradient 0.5 * grad
      return internal::dual<value_type, real_t> {b, 0.5 * a.gradient};
   }
}
template <typename value_type, typename gradient_type> MFEM_HOST_DEVICE
inline internal::dual<value_type, gradient_type> max(
   internal::dual<value_type, gradient_type> a,
   internal::dual<value_type, gradient_type> b)
{
   if (a.value > b.value)
   {
      return a;
   }
   else if (a.value < b.value)
   {
      return b;
   }
   else
   {
      // At a.value == b: derivative is undefined — choose subgradient 0.5 * grad
      return (a+b)*0.5;
   }
}
inline real_t max(real_t a, real_t b) { return std::max(a, b); }

struct BaseADFunction
{
   using ADReal_t = internal::dual<real_t, real_t>;
   virtual ADReal_t operator()(TAutoDiffVector<ADReal_t> &var,
                               const Vector &param) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   virtual real_t operator()(const Vector &var, const Vector &param) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }
};

// Make Autodiff Functor
// @param name will be the name of the structure
// @param T will be the name of the templated type
// @param param is additional parameter name (will not be differentiated)
// @param body is the main function body. Use T() to create 0 T-typed value.
#define MAKE_AD_FUNCTION(name, T, var, param, body)                     \
struct name : public BaseADFunction                                     \
{                                                                       \
using ADReal_t = internal::dual<real_t, real_t>;                        \
   ADReal_t operator()(TAutoDiffVector<ADReal_t> &var,                  \
                       const Vector &param) const                       \
   {                                                                    \
      using T = ADReal_t;                                               \
      body                                                              \
   }                                                                    \
                                                                        \
   real_t operator()(const Vector &var, const Vector &param) const      \
   {                                                                    \
      using T = real_t;                                                 \
      body                                                              \
   }                                                                    \
}

// Autodiff of f:R^n -> R where f is a derived structure of BaseADFunction
// To evaluate, use (). To evaluate Jacobian, use Jacobian()
class ADFunctor
{
   using ADReal_t = internal::dual<real_t, real_t>;
   using ADVector = TAutoDiffVector<ADReal_t>;
protected:
   BaseADFunction &f;
public:
   ADFunctor(BaseADFunction &f): f(f) { }
   /// Evaluates the function for arguments x with given parameters param.
   virtual real_t operator()(const Vector &x, const Vector &param) const
   {
      return f(x, param);
   }

   /// Evaluate the Jacobian of function with respect to x with given parameters param.
   virtual void Jacobian(const Vector &x, const Vector &param, Vector &jac) const
   {
      jac.SetSize(x.Size());
      ADVector ad_x(x);
      // for (int i=0; i<x.Size(); i++)
      // { ad_x[i].value = std::max(1e-10, ad_x[i].value); }
      for (int i=0; i<x.Size(); i++)
      {
         ad_x[i].gradient = 1.0;
         ADReal_t result = f(ad_x, param);
         jac[i] = result.gradient;
         ad_x[i].gradient = 0.0;
      }
   }
};

// Autodiff of f(u(x)) but only for f. That is, Jacobian will evaluate J_f(u(x))
// where u is defined in Preprocess.
class PreprocessedADFunctor : public ADFunctor
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector val;
#endif
public:
   PreprocessedADFunctor(BaseADFunction &f): ADFunctor(f) { }
   virtual void Preprocess(const Vector &x, const Vector &param,
                           Vector &y) const = 0;
   /// Evaluates the function for arguments vparam and uu. The evaluation is
   real_t operator()(const Vector &x, const Vector &param) const override
   {
#ifdef MFEM_THREAD_SAFE
      Vector val;
#endif
      val.SetSize(x.Size());
      Preprocess(x, param, val);
      return ADFunctor::operator()(val, param);
   }

   void Jacobian(const Vector &x, const Vector &param, Vector &jac) const override
   {
#ifdef MFEM_THREAD_SAFE
      Vector val;
#endif
      val.SetSize(x.Size());
      Preprocess(x, param, val);
      return ADFunctor::Jacobian(val, param, jac);
   }
};

inline void exp_sum_to_one(const Vector &x, Vector &y)
{
   const int n = x.Size();
   y.SetSize(n);
   for (int i=0; i<n; i++)
   {
      MFEM_ASSERT(x[i] < 700, "Overflow");
      y[i] = std::exp(std::min(std::max(x[i], -40.0), 40.0));
   }
   y /= y.Sum();
}
// AutoDiff Functor with preprocessing, u_i(x) = exp(x_i) / sum_j exp(x_j)
class SumToOnePGFunctor : public PreprocessedADFunctor
{
public:
   SumToOnePGFunctor(BaseADFunction &f) : PreprocessedADFunctor(f) {}
   // Preprocess values, y[i] = exp(x[i]) / sum_j exp(x[j])
   void Preprocess(const Vector &x, const Vector &, Vector &y) const
   {
      const int n = x.Size();
      y.SetSize(n);
      for (int i=0; i<n; i++)
      {
         MFEM_ASSERT(x[i] < 700, "Overflow");
         y[i] = std::exp(std::min(std::max(x[i], -40.0), 40.0));
      }
      y /= y.Sum();
   }
};

// Differentiable GridFunction Coefficient f(u)
// where u is a grid function with vector dim n
//       f is an ADFunctor, Autodiff enabled function.
// MyADVecGridFunCF::Eval() evaluates f(u(x))
// MyADVecGridFunCF::GetJacobian() returns a vector coefficient
// that evaluates J_f(u(x)) where J_f is the Jacobian of f.
class ADVecGridFuncCF : public Coefficient
{
   class JacobianCF : public VectorCoefficient
   {
   public:
      JacobianCF(ADVecGridFuncCF &cf)
         : VectorCoefficient(cf.vdim)
         , cf(cf), val(cf.vdim)
      {}

      void Eval(Vector &jac, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         jac.SetSize(cf.vdim);
         cf.gf->GetVectorValue(T, ip, val);
         cf.f.Jacobian(val, cf.param, jac);
      }

   private:
      ADVecGridFuncCF &cf;
      mutable Vector val;
   };
   friend JacobianCF;

public:
   ADVecGridFuncCF(ADFunctor &f, GridFunction * gf = nullptr)
      : gf(gf)
      , f(f)
      , vdim(gf ? gf->VectorDim() : 0)
   {}

   void SetGridFunction(GridFunction &gf)
   {
      this->gf = &gf;
      vdim = gf.VectorDim();
      val.SetSize(vdim);
   }

   void SetParam(const Vector &param)
   {
      this->param.SetDataAndSize(param.GetData(), param.Size());
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(gf != nullptr, "Set grid function before evaluate.");
      gf->GetVectorValue(T, ip, val);
      return f(val, param);
   }

   // This should be called after grid function is set. Otherwise, vdim == 0.
   JacobianCF GetJacobian() { return JacobianCF(*this); }

protected:
   GridFunction *gf;
   Vector param;
   ADFunctor &f;
   mutable Vector val;
   int vdim;
};

} // namespace mfem
#endif
