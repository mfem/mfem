#pragma once

#include "mfem.hpp"
#include "miniapps/autodiff/tadvector.hpp"
#include "miniapps/autodiff/taddensemat.hpp"

namespace mfem
{
template <typename value_type, typename gradient_type, typename other_type>
MFEM_HOST_DEVICE
inline future::dual<value_type, gradient_type> max(
   future::dual<value_type, gradient_type> a, other_type b);

inline real_t max(const real_t a, const real_t b) { return std::max(a,b); }

template <typename value_type, typename gradient_type, typename other_type>
MFEM_HOST_DEVICE
inline future::dual<value_type, gradient_type> min(
   future::dual<value_type, gradient_type> a, other_type b);

MFEM_HOST_DEVICE
inline real_t min(const real_t a, const real_t b) { return std::min(a,b); }

// Use mfem-native autodiff types
// If other autodiff libraries are used,
// define ADReal_t, ADVector, ADMatrix, ... types accordingly.

// First order dual
typedef future::dual<real_t, real_t> ADReal_t;
typedef TAutoDiffVector<ADReal_t> ADVector;
typedef TAutoDiffDenseMatrix<ADReal_t> ADMatrix;

// second order dual (nested dual)
typedef future::dual<ADReal_t, ADReal_t> AD2Real_t;
typedef TAutoDiffVector<AD2Real_t> AD2Vector;
typedef TAutoDiffDenseMatrix<AD2Real_t> AD2Matrix;

struct ADFunction
{

public:
   virtual void ProcessParameters(ElementTransformation &Tr,
                                  const IntegrationPoint &ip) const
   {
      // DO nothing by default
   }

   int n_input;
   ADFunction(int n_input)
      : n_input(n_input) { }
   // default evaluator
   virtual real_t operator()(const Vector &x) const
   { MFEM_ABORT("Not implemented. Use AD_IMPL macro to implement all path"); }
   virtual real_t operator()(const Vector &x, ElementTransformation &Tr,
                             const IntegrationPoint &ip) const
   {
      ProcessParameters(Tr, ip);
      return (*this)(x);
   }

   // default Jacobian evaluator
   virtual ADReal_t operator()(const ADVector &x) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // default Hessian evaluator
   virtual AD2Real_t operator()(const AD2Vector &x) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // Evaluate the gradient, using forward mode autodiff
   virtual void Gradient(const Vector &x, ElementTransformation &Tr,
                         const IntegrationPoint &ip, Vector &J) const;
   virtual void Gradient(const Vector &x, Vector &J) const;
   // Evaluate the Hessian, using forward over forward autodiff
   // The Hessian assumed to be symmetric.
   virtual void Hessian(const Vector &x, ElementTransformation &Tr,
                        const IntegrationPoint &ip,
                        DenseMatrix &H) const;
   virtual void Hessian(const Vector &x, DenseMatrix &H) const;
};

// We currently only support Jacobian.
// To consistent with ADFunction, which returns
// evaluate: scalar, Gradient: vector, Hessian: matrix,
// we overrode the Gradient for evaulation, and Hessian for Jacobian
// To be used with ADNonlinearFormIntegrator or ADBlockNonlinearFormIntegrator,
// n_input and n_output must be the same.
struct ADVectorFunction : public ADFunction
{

   int n_output;
   ADVectorFunction(int n_input, int n_output)
      : ADFunction(n_input), n_output(n_output)
   {
      MFEM_ASSERT(n_input > 0 && n_output > 0,
                  "ADVectorFunction: n_input and n_output must be positive");
   }

   void operator()(const Vector &x, ElementTransformation &Tr,
                   const IntegrationPoint &ip,
                   Vector &F) const
   {
      ProcessParameters(Tr, ip);
      (*this)(x, F);
   }
   // Derived struct should implement the following methods.
   // Use AD_VEC_IMPL macro to implement them.
   virtual void operator()(const Vector &x, Vector &F) const = 0;
   virtual void operator()(const ADVector &x, ADVector &F) const = 0;
   virtual void operator()(const AD2Vector &x, AD2Vector &F) const = 0;

   void Gradient(const Vector &x, ElementTransformation &Tr,
                 const IntegrationPoint &ip, DenseMatrix &J) const
   {
      MFEM_ASSERT(x.Size() == n_input,
                  "ADVectorFunction::Gradient: x.Size() must match n_input");
      ProcessParameters(Tr, ip);
      Gradient(x, J);
   }
   void Gradient(const Vector &x, DenseMatrix &J) const
   {
      ADVector x_ad(x);
      ADVector Fx(n_output);
      J.SetSize(n_output, n_input);
      for (int i=0; i<x.Size(); i++)
      {
         x_ad[i].gradient = 1.0;
         Fx = ADReal_t();
         (*this)(x_ad, Fx);
         for (int j=0; j<n_output; j++)
         {
            J(j,i) = Fx[j].gradient;
         }
         x_ad[i].gradient = 0.0; // Reset gradient for next iteration
      }
   }
   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip,
                DenseTensor &H) const
   {
      ProcessParameters(Tr, ip);
      Hessian(x, H);
   }
   void Hessian(const Vector &x, DenseTensor &H) const
   {
      AD2Vector x_ad(x);
      AD2Vector Fx(n_output);
      H.SetSize(n_input, n_input, n_output);
      for (int i=0; i<n_input; i++) // Loop for the first derivative
      {
         x_ad[i].value.gradient = 1.0;
         for (int j=0; j<=i; j++)
         {
            x_ad[j].gradient.value = 1.0;
            Fx = AD2Real_t();
            (*this)(x_ad, Fx);
            for (int k=0; k<n_output; k++)
            {
               H(j, i, k) = Fx[k].gradient.gradient;
               H(i, j, k) = Fx[k].gradient.gradient;
            }
            x_ad[j].gradient.value = 0.0; // Reset gradient for next iteration
         }
         x_ad[i].value.gradient = 0.0;
      }
   }

   // To support ADNonlinearFormIntegrator and ADVectorNonlinearFormIntegrator
   void Gradient(const Vector &x, ElementTransformation &Tr,
                 const IntegrationPoint &ip, Vector &F) const override final
   { (*this)(x, Tr, ip, F); }
   void Gradient(const Vector &x, Vector &F) const override final
   {
      (*this)(x, F);
   }

   // To support ADNonlinearFormIntegrator and ADVectorNonlinearFormIntegrator
   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip,
                DenseMatrix &J) const override final
   {
      MFEM_ASSERT(n_input == n_output,
                  "ADVectorFunction::Hessian: n_input must match n_output");
      this->Gradient(x, Tr, ip, J);
   }
   void Hessian(const Vector &x, DenseMatrix &J) const override final
   {
      MFEM_ASSERT(n_input == n_output,
                  "ADVectorFunction::Hessian: n_input must match n_output");
      this->Gradient(x, J);
   }

   real_t operator()(const Vector &x) const override final
   {
      MFEM_ABORT("ADVectorFunction::operator(): This method should not be called. "
                 "Use ADVectorFunction::operator(const Vector &x, Vector &F) instead.");
   }
   ADReal_t operator()(const ADVector &x) const override final
   {
      MFEM_ABORT("ADVectorFunction::operator(): This method should not be called. "
                 "Use ADVectorFunction::operator(const ADVector &x, ADVector &F) instead.");
   }
   AD2Real_t operator()(const AD2Vector &x) const override final
   {
      MFEM_ABORT("ADVectorFunction::operator(): This method should not be called. "
                 "Use ADVectorFunction::operator(const AD2Vector &x, AD2Vector &F) instead.");
   }
};

class DifferentiableCoefficient : public Coefficient
{
private:
   int idx;
   class GradientCoefficient : public VectorCoefficient
   {
      DifferentiableCoefficient &c;
   public:
      GradientCoefficient(int dim, DifferentiableCoefficient &c)
         : VectorCoefficient(dim), c(c) { }
      void Eval(Vector &J, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         c.EvalInput(T, ip);
         c.f.Gradient(c.x, T, ip, J);
      }
   };
   friend class GradientCoefficient;
   GradientCoefficient grad_cf;
   class HessianCoefficient : public MatrixCoefficient
   {
      DifferentiableCoefficient &c;
   public:
      HessianCoefficient(int dim, DifferentiableCoefficient &c)
         : MatrixCoefficient(dim), c(c) { }
      void Eval(DenseMatrix &H, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         c.EvalInput(T, ip);
         c.f.Hessian(c.x, T, ip, H);
      }
   };
   friend class HessianCoefficient;
   HessianCoefficient hess_cf;
protected:

   ADFunction &f;
   mutable Vector x;
   std::vector<Coefficient*> cfs;
   std::vector<int> cfs_idx;
   std::vector<VectorCoefficient*> v_cfs;
   std::vector<int> v_cfs_idx;
   std::vector<GridFunction*> gfs;
   std::vector<int> gfs_idx;
   std::vector<QuadratureFunction*> qfs;
   std::vector<int> qfs_idx;
public:
   DifferentiableCoefficient(ADFunction &f)
      : f(f), x(1), idx(0)
      , grad_cf(f.n_input, *this)
      , hess_cf(f.n_input, *this)
   {}
   DifferentiableCoefficient& AddInput(Coefficient &cf)
   {
      cfs.push_back(&cf);
      cfs_idx.push_back(idx++);
      x.SetSize(idx);
      return *this;
   }
   DifferentiableCoefficient& AddInput(VectorCoefficient &vcf)
   {
      v_cfs.push_back(&vcf);
      v_cfs_idx.push_back(idx);
      idx += vcf.GetVDim();
      x.SetSize(idx);
      return *this;
   }
   DifferentiableCoefficient& AddInput(GridFunction &gf)
   {
      gfs.push_back(&gf);
      gfs_idx.push_back(idx);
      idx += gf.FESpace()->GetVDim();
      x.SetSize(idx);
      return *this;
   }
   DifferentiableCoefficient& AddInput(QuadratureFunction &qf)
   {
      qfs.push_back(&qf);
      qfs_idx.push_back(idx);
      idx += qf.GetVDim();
      x.SetSize(idx);
      return *this;
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      EvalInput(T, ip);
      return f(x, T, ip);
   }
   GradientCoefficient& Gradient() { return grad_cf; }
   HessianCoefficient& Hessian() { return hess_cf; }

protected:
   void EvalInput(ElementTransformation &T,
                  const IntegrationPoint &ip) const
   {
      Vector x_view;
      for (int i=0; i<cfs.size(); i++)
      {
         x[cfs_idx[i]] = cfs[i]->Eval(T, ip);
      }
      for (int i=0; i<v_cfs.size(); i++)
      {
         x_view.SetDataAndSize(x.GetData() + v_cfs_idx[i],
                               v_cfs[i]->GetVDim());
         v_cfs[i]->Eval(x_view, T, ip);
      }
      for (int i=0; i<gfs.size(); i++)
      {
         x_view.SetDataAndSize(x.GetData() + gfs_idx[i],
                               gfs[i]->FESpace()->GetVDim());
         gfs[i]->GetVectorValue(T, ip, x_view);
      }
      for (int i=0; i<qfs.size(); i++)
      {
         x_view.SetDataAndSize(x.GetData() + qfs_idx[i],
                               qfs[i]->GetVDim());
         qfs[i]->GetValues(T.ElementNo, ip.index, x_view);
      }
   }
};

// Macro to generate type-varying implementation for ADFunction.
// See, DiffusionEnergy, ..., for example of usage.
// @param SCALAR is the name of templated scalar type
// @param VEC is the name of templated vector type
// @param MAT is the name of templated matrix type
// @param var is the input variable name
// @param body is the main function body. Use T() to create T-typed 0.
#define AD_IMPL(SCALAR, VEC, MAT, var, body)                                           \
   using ADFunction::operator();                                                       \
   real_t operator()(const Vector &var) const override                                 \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = real_t;                                                           \
      using VEC = Vector;                                                              \
      using MAT = DenseMatrix;                                                         \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   ADReal_t operator()(const ADVector &var) const override                             \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = ADReal_t;                                                         \
      using VEC = ADVector;                                                            \
      using MAT = ADMatrix;                                                            \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   AD2Real_t operator()(const AD2Vector &var) const override                           \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = AD2Real_t;                                                        \
      using VEC = AD2Vector;                                                           \
      using MAT = AD2Matrix;                                                           \
      body                                                                             \
   }


// Macro to generate type-varying implementation for ADVectorFunction.
// @param SCALAR is the name of templated scalar type
// @param VEC is the name of templated vector type
// @param MAT is the name of templated matrix type
// @param var is the input variable name
// @param result is the output variable name
// @param body is the main function body. Use T() to create T-typed 0.
#define AD_VEC_IMPL(SCALAR, VEC, MAT, var, result, body)                               \
   using ADVectorFunction::operator();                                                 \
   using ADVectorFunction::Gradient;                                                   \
   using ADVectorFunction::Hessian;                                                    \
                                                                                       \
   void operator()(const Vector &var, Vector &result) const override                   \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = real_t;                                                           \
      using VEC = Vector;                                                              \
      using MAT = DenseMatrix;                                                         \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   void operator()(const ADVector &var, ADVector &result) const override               \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = ADReal_t;                                                         \
      using VEC = ADVector;                                                            \
      using MAT = ADMatrix;                                                            \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   void operator()(const AD2Vector &var, AD2Vector &result) const override             \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = AD2Real_t;                                                        \
      using VEC = AD2Vector;                                                           \
      using MAT = AD2Matrix;                                                           \
      body                                                                             \
   }

struct MassEnergy : public ADFunction
{
   MassEnergy(int n_var)
      : ADFunction(n_var)
   {}
   AD_IMPL(T, V, M, x, return 0.5*(x*x););
};
struct DiffusionEnergy : public ADFunction
{
   DiffusionEnergy(int dim)
      : ADFunction(dim)
   {}
   AD_IMPL(T, V, M, gradu, return 0.5*(gradu*gradu););
};
struct HeteroDiffusionEnergy : public ADFunction
{
   Coefficient &K;
   mutable real_t kappa;
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      kappa = K.Eval(Tr, ip);
   }
   HeteroDiffusionEnergy(int dim, Coefficient &K)
      : ADFunction(dim), K(K)
   {}

   AD_IMPL(T, V, M, gradu, return (kappa*0.5)*(gradu*gradu);)
};
struct AnisoDiffuionEnergy : public ADFunction
{
   MatrixCoefficient &K;
   mutable DenseMatrix kappa;
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      K.Eval(kappa, Tr, ip);
   }
   AnisoDiffuionEnergy(int dim, MatrixCoefficient &K)
      : ADFunction(dim), K(K), kappa(K.GetHeight(), K.GetWidth())
   {
      MFEM_VERIFY(dim == K.GetHeight() && dim == K.GetWidth(),
                  "AnisoDiffuionEnergy: K must be a square matrix of size dim");
   }

   AD_IMPL(T, V, M, gradu,
   {
      T result = T();
      const int dim = gradu.Size();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            result += kappa(i,j)*gradu[i]*gradu[j];
         }
      }
      return result;
   });
};

struct DiffEnergy : public ADFunction
{
   const ADFunction &energy;
   VectorCoefficient *other;
   std::unique_ptr<VectorCoefficient> owned_cf;
   mutable Vector other_v;
   DiffEnergy(const ADFunction &energy)
      : ADFunction(energy.n_input)
      , energy(energy), other_v(n_input)
   { }
   DiffEnergy(const ADFunction &energy, VectorCoefficient &other)
      : DiffEnergy(energy)
   {
      MFEM_VERIFY(other.GetVDim() == n_input,
                  "DiffEnergy: other must have the same dimension as energy");
      this->other = &other;
   }
   DiffEnergy(const ADFunction &energy, GridFunction &other)
      : DiffEnergy(energy)
   {
      MFEM_VERIFY(other.FESpace()->GetVDim() == n_input,
                  "DiffEnergy: other must have the same dimension as energy");
      SetTarget(other);
   }
   DiffEnergy(const ADFunction &energy, QuadratureFunction &other)
      : DiffEnergy(energy)
   {
      MFEM_VERIFY(other.GetVDim() == n_input,
                  "DiffEnergy: other must have the same dimension as energy");
      SetTarget(other);
   }
   DiffEnergy(const ADFunction &energy, Coefficient &other)
      : DiffEnergy(energy)
   {
      MFEM_VERIFY(n_input == 1,
                  "DiffEnergy: other must have the same dimension as energy");
      SetTarget(other);
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      MFEM_ASSERT(other != nullptr,
                  "DiffEnergy: other is not set. Use SetTarget() to set it.");
      energy.ProcessParameters(Tr, ip);
      other->Eval(other_v, Tr, ip);
   }

   AD_IMPL(T, V, M, x,
   {
      V diff(x);
      for (int i=0; i<n_input; i++) { diff[i] -= other_v[i]; }
      return energy(diff);
   });

   void SetTarget(VectorCoefficient &other) { this->other = &other; }
   void SetTarget(Coefficient &other)
   {
      auto cf = new VectorArrayCoefficient(1);
      cf->Set(0, &other, false);
      owned_cf.reset(cf);
      this->other = cf;
   }
   void SetTarget(GridFunction &other)
   {
      if (other.FESpace()->GetVDim() == 1)
      {
         auto cf = new VectorArrayCoefficient(1);
         cf->Set(0, new GridFunctionCoefficient(&other), true);
         owned_cf.reset(cf);
         this->other = cf;
      }
      else
      {
         owned_cf = std::make_unique<VectorGridFunctionCoefficient>(&other);
         this->other = owned_cf.get();
      }
   }
   void SetTarget(QuadratureFunction &other)
   {
      if (other.GetVDim() == 1)
      {
         auto cf = new VectorArrayCoefficient(1);
         cf->Set(0, new QuadratureFunctionCoefficient(other), true);
         owned_cf.reset(cf);
         this->other = cf;
      }
      else
      {
         owned_cf = std::make_unique<VectorQuadratureFunctionCoefficient>(other);
         this->other = owned_cf.get();
      }
   }
};

struct LinearElasticityEnergy : public ADFunction
{
   Coefficient *lambda_cf;
   Coefficient *mu_cf;
   const int dim;
   mutable real_t lambda;
   mutable real_t mu;
   std::vector<std::unique_ptr<Coefficient>> owned_cf;
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      lambda = lambda_cf->Eval(Tr, ip);
      mu = mu_cf->Eval(Tr, ip);
   }
   LinearElasticityEnergy(int dim, Coefficient &lambda, Coefficient &mu)
      : ADFunction(dim*dim), lambda_cf(&lambda), mu_cf(&mu), dim(dim)
   {}
   LinearElasticityEnergy(int dim, real_t lambda, real_t mu)
      : ADFunction(dim*dim), dim(dim)
   {
      owned_cf.resize(2);
      owned_cf[0] = std::make_unique<ConstantCoefficient>(lambda);
      lambda_cf = owned_cf[0].get();
      owned_cf[1] = std::make_unique<ConstantCoefficient>(mu);
      mu_cf = owned_cf[1].get();
   }
   AD_IMPL(T, V, M, gradu,
   {
      T divnorm = T();
      for (int i=0; i<dim; i++) { divnorm += gradu[i*dim + i]; }
      divnorm = divnorm*divnorm;
      T h1_norm = T();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            T symm = 0.5*(gradu[i*dim + j] + gradu[j*dim + i]);
            h1_norm += symm*symm;
         }
      }
      return 0.5*lambda*divnorm + mu*h1_norm;
   });
};
// ----------------------------------------------------------------
// Operator overloading for ADFunction arithmetics
// Using shared_ptr to manage memory.
// Not ideal, but good enough for now.
// ----------------------------------------------------------------
struct ProductADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   const std::shared_ptr<ADFunction> f2;

   ProductADFunction(const std::shared_ptr<ADFunction> &f1,
                     const std::shared_ptr<ADFunction> &f2)
      : ADFunction(0), f1(f1), f2(f2)
   {
      MFEM_ASSERT(f1.get() != nullptr && f2.get() != nullptr,
                  "ProductADFunction: f1 and f2 must not be null");
      MFEM_ASSERT(f1->n_input == f2->n_input,
                  "ProductADFunction: f1 and f2 must have the same n_input");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
      f2->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x)*(*f2)(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) * (*f2)(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) * (*f2)(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) * (*f2)(x); }
};
inline std::shared_ptr<ADFunction>
operator*(const std::shared_ptr<ADFunction> &f1,
          const std::shared_ptr<ADFunction> &f2)
{ return std::make_shared<ProductADFunction>(f1, f2); }

struct ScaledADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   real_t a;

   ScaledADFunction(const std::shared_ptr<ADFunction> &f1,
                    real_t a)
      : ADFunction(0), f1(f1), a(a)
   {
      MFEM_ASSERT(f1.get() != nullptr,
                  "ProductADFunction: f1 and f2 must not be null");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x)*a; }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) * a; }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) * a; }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) * a; }
};
inline std::shared_ptr<ADFunction>
operator*(const std::shared_ptr<ADFunction> &f1,
          real_t a)
{ return std::make_shared<ScaledADFunction>(f1, a); }
inline std::shared_ptr<ADFunction>
operator*(real_t a,
          const std::shared_ptr<ADFunction> &f1)
{ return std::make_shared<ScaledADFunction>(f1, a); }
inline std::shared_ptr<ADFunction>
operator/(const std::shared_ptr<ADFunction> &f1,
          real_t a)
{ return std::make_shared<ScaledADFunction>(f1, 1.0/a); }

struct SumADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   const std::shared_ptr<ADFunction> f2;
   real_t b; // scaling factor for f2

   SumADFunction(const std::shared_ptr<ADFunction> &f1,
                 const std::shared_ptr<ADFunction> &f2, real_t b)
      : ADFunction(0), f1(f1), f2(f2)
   {
      MFEM_ASSERT(f1.get() != nullptr && f2.get() != nullptr,
                  "ProductADFunction: f1 and f2 must not be null");
      MFEM_ASSERT(f1->n_input == f2->n_input,
                  "ProductADFunction: f1 and f2 must have the same n_input");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
      f2->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x) + b*(*f2)(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) + b*(*f2)(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) + b*(*f2)(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) + b* (*f2)(x); }
};
inline std::shared_ptr<ADFunction>
operator+(const std::shared_ptr<ADFunction> &f1,
          const std::shared_ptr<ADFunction> &f2)
{ return std::make_shared<SumADFunction>(f1, f2, 1.0); }
inline std::shared_ptr<ADFunction>
operator-(const std::shared_ptr<ADFunction> &f1,
          const std::shared_ptr<ADFunction> &f2)
{ return std::make_shared<SumADFunction>(f1, f2, -1.0); }

struct ShiftedADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   real_t b; // scaling factor for f2

   ShiftedADFunction(const std::shared_ptr<ADFunction> &f1, real_t b)
      : ADFunction(0), f1(f1), b(b)
   {
      MFEM_ASSERT(f1.get() != nullptr,
                  "ProductADFunction: f1 must not be null");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x) + b; }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) + b; }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) + b; }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) + b; }
};
inline std::shared_ptr<ADFunction>
operator+(const std::shared_ptr<ADFunction> &f1, real_t b)
{ return std::make_shared<ShiftedADFunction>(f1, b); }
inline std::shared_ptr<ADFunction>
operator+(real_t b,const std::shared_ptr<ADFunction> &f1)
{ return std::make_shared<ShiftedADFunction>(f1, b); }

inline std::shared_ptr<ADFunction>
operator-(const std::shared_ptr<ADFunction> &f1, real_t b)
{ return std::make_shared<ShiftedADFunction>(f1, -b); }

struct QuatiendADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   const std::shared_ptr<ADFunction> f2;

   QuatiendADFunction(const std::shared_ptr<ADFunction> &f1,
                      const std::shared_ptr<ADFunction> &f2)
      : ADFunction(0), f1(f1), f2(f2)
   {
      MFEM_ASSERT(f1.get() != nullptr && f2.get() != nullptr,
                  "QuatiendADFunction: f1 and f2 must not be null");
      MFEM_ASSERT(f1->n_input == f2->n_input,
                  "QuatiendADFunction: f1 and f2 must have the same n_input");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
      f2->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x) / (*f2)(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) / (*f2)(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) / (*f2)(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) / (*f2)(x); }
};
inline std::shared_ptr<ADFunction>
operator/(const std::shared_ptr<ADFunction> &f1,
          const std::shared_ptr<ADFunction> &f2)
{ return std::make_shared<QuatiendADFunction>(f1, f2); }

struct ReciprocalADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   real_t a;

   ReciprocalADFunction(const std::shared_ptr<ADFunction> &f1, real_t a)
      : ADFunction(0), f1(f1)
   {
      MFEM_ASSERT(f1.get() != nullptr,
                  "ReciprocalADFunction: f1 must not be null");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return a / (*f1)(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return a / (*f1)(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return a / (*f1)(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return a / (*f1)(x); }
};
inline std::shared_ptr<ADFunction>
operator/(real_t a, const std::shared_ptr<ADFunction> &f1)
{ return std::make_shared<ReciprocalADFunction>(f1, a); }

struct ReferenceConstantADFunction : public ADFunction
{
   real_t &a;

   ReferenceConstantADFunction(real_t &a, int n_input)
      : ADFunction(n_input)
      , a(a)
   { }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return a; }

   // default Jacobian evaluator
   real_t operator()(const Vector &x) const override
   { return a; }
   ADReal_t operator()(const ADVector &x) const override
   { return ADReal_t{a, 0.0}; }

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override
   { return AD2Real_t{a, 0.0}; }

   void Gradient(const Vector &x, Vector &J) const override
   {
      J.SetSize(x.Size());
      J = 0.0; // Gradient is zero for constant function
   }
   void Hessian(const Vector &x, DenseMatrix &H) const override
   {
      H.SetSize(x.Size(), x.Size());
      H = 0.0; // Hessian is zero for constant function
   }
};

// ------------------------------------------------------------------------------
// Implement dual max/min
// ------------------------------------------------------------------------------
template <typename value_type, typename gradient_type, typename other_type>
MFEM_HOST_DEVICE
inline future::dual<value_type, gradient_type> max(
   future::dual<value_type, gradient_type> a,
   other_type b)
{
   if (a > b)
   {
      return a;
   }
   else if (a < b)
   {
      if constexpr (std::is_same<other_type, real_t>::value)
      {
         return future::dual<value_type, gradient_type> {b};
      }
      else
      {
         return b;
      }
   }
   else
   {
      // If values are equal, return the average (subgradient)
      return 0.5*(a + b);
   }
}

template <typename value_type, typename gradient_type, typename other_type>
MFEM_HOST_DEVICE
inline future::dual<value_type, gradient_type> min(
   future::dual<value_type, gradient_type> a,
   other_type b)
{
   if (a < b)
   {
      return a;
   }
   else if (a > b)
   {
      if constexpr (std::is_same<other_type, real_t>::value)
      {
         return future::dual<value_type, gradient_type> {b};
      }
      else
      {
         return b;
      }
   }
   else
   {
      // If values are equal, return the average (subgradient)
      return 0.5*(a + b);
   }
}
}
