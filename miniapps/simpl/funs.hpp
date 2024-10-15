#ifndef FUNS_HPP
#define FUNS_HPP
#include "mfem.hpp"

namespace mfem
{

// Some useful function for topology optimization
#define LOGMIN 2e-50
#define LOGMIN_VAL -34.65735902799726547086160607290882840377500671801276270603400047
real_t safe_log(const real_t x);
real_t sigmoid(const real_t x);
real_t invsigmoid(const real_t x);
real_t der_sigmoid(const real_t x);
real_t simp(const real_t x, const real_t exponent, const real_t rho0);
real_t der_simp(const real_t x, const real_t exponent, const real_t rho0);

class CompositeCoefficient : public Coefficient
{
   typedef std::function<real_t(const real_t)> fun_type;
private:
   Coefficient *coeff;
   bool own_coeff;
   fun_type fun;
public:
   CompositeCoefficient(Coefficient &coeff, fun_type fun):coeff(&coeff),
      own_coeff(false), fun(fun) {}
   CompositeCoefficient(Coefficient *coeff, fun_type fun,
                        bool own_coeff=true)
      :coeff(coeff), own_coeff(own_coeff), fun(fun) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return fun(coeff->Eval(T, ip));
   };

   void SetCoefficient(Coefficient &cf)
   {
      if (own_coeff && coeff) {delete coeff;}
      coeff = &cf;
      own_coeff = false;
   }

   void SetCoefficient(Coefficient *cf, bool own_cf=true)
   {
      if (own_coeff && coeff) {delete coeff;}
      coeff = cf;
      own_coeff = own_cf;
   }

   void SetFunction(fun_type new_fun) { fun = new_fun; }
};

// A coefficient that maps a given gridfunction with a given function.
// x |-> f(gf(x))
class MappedGFCoefficient : public GridFunctionCoefficient
{
   typedef std::function<real_t(const real_t)> fun_type;
private:
   fun_type fun;

public:
   MappedGFCoefficient():GridFunctionCoefficient(nullptr) {}
   MappedGFCoefficient(GridFunction &gf, fun_type fun, int comp=1)
      : GridFunctionCoefficient(&gf, comp), fun(fun) {}
   MappedGFCoefficient(fun_type fun)
      : GridFunctionCoefficient(nullptr), fun(fun) {}
   MappedGFCoefficient(GridFunction *gf, int comp=1)
      : GridFunctionCoefficient(gf, comp) {}

   void SetFunction(fun_type newfun) { fun=newfun; }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return fun(GridFunctionCoefficient::Eval(T, ip));
   }
};

// A coefficient that maps given gridfunctions with a given function.
// x |-> f(gf(x), other_gf(x))
class MappedPairedGFCoefficient : public GridFunctionCoefficient

{
   typedef std::function<real_t(const real_t, const real_t)> fun_type;
private:
   fun_type fun;
   GridFunction *other_gf;
   int other_gf_comp;

public:
   // Create a coefficient that returns f(v1,v2) where v1=gf(x) and v2=other_gf(x)
   MappedPairedGFCoefficient(GridFunction &gf, GridFunction &other_gf,
                             fun_type fun)
      : GridFunctionCoefficient(&gf), fun(fun), other_gf(&other_gf),
        other_gf_comp(1) {}

   // Create only with function. Use SetGridFunction to set gridfunctions.
   // By default, the object takes the ownership.
   MappedPairedGFCoefficient(fun_type fun)
      : GridFunctionCoefficient(nullptr), fun(fun), other_gf(nullptr) {}

   // Create an empty object. Use SetFunction and SetGridFunction
   MappedPairedGFCoefficient():GridFunctionCoefficient(nullptr) {}

   void SetGridFunction(GridFunction *new_gf, GridFunction *new_other_gf)
   {
      GridFunctionCoefficient::SetGridFunction(new_gf);
      other_gf=new_other_gf;
   }

   void SetOtherGridFunction(GridFunction *new_other_gf, int new_other_comp=-1)
   {
      other_gf = new_other_gf;
      other_gf_comp=new_other_comp;
   }

   void SetFunction(fun_type newfun)
   {
      fun=newfun;
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return fun(GridFunctionCoefficient::Eval(T, ip),
                 other_gf->GetValue(T, ip, other_gf_comp));
   }
};

// An entropy defined by Legendre function
// With Bregman divergence, this function can generate
// mapping between a convex set to a vector space
class LegendreEntropy
{
   typedef std::function<real_t(const real_t)> fun_type;
private:
   real_t lower_bound;
   real_t upper_bound;
   real_t finite_lower_bound;
   real_t finite_upper_bound;
public:
   fun_type entropy;
   fun_type forward; // primal to dual
   fun_type backward; // dual to primal
   LegendreEntropy(fun_type entropy, fun_type forward, fun_type backward,
                   real_t lower_bound, real_t upper_bound,
                   real_t finite_lower_bound, real_t finite_upper_bound)
      :entropy(entropy), forward(forward), backward(backward),
       lower_bound(lower_bound), upper_bound(upper_bound),
       finite_lower_bound(finite_lower_bound),
       finite_upper_bound(finite_upper_bound) {}
   MappedGFCoefficient GetForwardCoeff();
   MappedGFCoefficient GetBackwardCoeff();
   MappedGFCoefficient GetEntropyCoeff();
   real_t GetLowerBound() {return lower_bound;};
   real_t GetUpperBound() {return upper_bound;};
   real_t GetFiniteLowerBound() {return finite_lower_bound;};
   real_t GetFiniteUpperBound() {return finite_upper_bound;};
   void SetFiniteLowerBound(real_t new_lower_bound) { finite_lower_bound = new_lower_bound; };
   void SetFiniteUpperBound(real_t new_upper_bound) { finite_upper_bound = new_upper_bound;};
   MappedGFCoefficient GetForwardCoeff(GridFunction &x);
   MappedGFCoefficient GetBackwardCoeff(GridFunction &psi);
   MappedGFCoefficient GetEntropyCoeff(GridFunction &x);
   // Get Bregman divergence with primal variables
   MappedPairedGFCoefficient GetBregman(GridFunction &x, GridFunction &y);
   // Get Bregman divergence with dual variables
   MappedPairedGFCoefficient GetBregman_dual(GridFunction &psi, GridFunction &chi);
};

class PrimalEntropy : public LegendreEntropy
{
public:
   PrimalEntropy():LegendreEntropy(
         [](const real_t x) {return x*x; }, [](const real_t x) {return x;}, [](
         const real_t x) {return x;}, 0, 1, 0, 1
   ) {}
};

// Fermi-Dirac Entropy with effective domain (0,1)
class FermiDiracEntropy : public LegendreEntropy
{
public:
   FermiDiracEntropy():LegendreEntropy(
         [](const real_t x) {return x*safe_log(x)+(1.0-x)*safe_log(1.0 - x);},
   invsigmoid, sigmoid, -mfem::infinity(), mfem::infinity(), -1e09, 1e09) {}
};

// Shannon Entropy with effective domain (0,1)
class ShannonEntropy : public LegendreEntropy
{
public:
   ShannonEntropy():LegendreEntropy(
         [](const real_t x) {return x*safe_log(x)-x;},
   safe_log, [](const real_t x) {return std::exp(x);}, -mfem::infinity(), 0.0,
   -1e09, 0.0) {}
};


} // end of namespace mfem
#endif
