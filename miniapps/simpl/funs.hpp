#include "mfem.hpp"

namespace mfem
{

// Some useful function for topology optimization
#define LOGMIN 2e-50
#define LOGMIN_VAL -50
real_t safe_log(const real_t x);
real_t sigmoid(const real_t x);
real_t invsigmoid(const real_t x);
real_t der_sigmoid(const real_t x);
real_t simp(const real_t x, const real_t exponent, const real_t rho0);
real_t der_simp(const real_t x, const real_t exponent, const real_t rho0);


// A coefficient that maps a given gridfunction with a given function.
// x |-> f(gf(x))
class MappedGFCoefficient : public GridFunctionCoefficient
{
   typedef std::function<real_t(const real_t)> fun_type;
private:
   fun_type *fun;
   bool own_function;

public:
   MappedGFCoefficient(GridFunction &gf, fun_type &fun)
      : GridFunctionCoefficient(&gf), fun(&fun), own_function(false) {}
   MappedGFCoefficient(fun_type *fun)
      : GridFunctionCoefficient(nullptr), fun(fun), own_function(true) {}
   MappedGFCoefficient():GridFunctionCoefficient(nullptr), fun(nullptr),
      own_function(false) {}
   ~MappedGFCoefficient()
   {
      if (own_function && fun) { delete fun; }
   }

   void SetFunction(fun_type &newfun)
   {
      if (own_function && fun) {delete fun;}
      fun=&newfun; own_function = false;
   }

   void SetFunction(fun_type *newfun, bool makeOwner=true)
   {
      if (own_function && fun) {delete fun;}
      fun=newfun; own_function = makeOwner;
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return fun->operator()(GridFunctionCoefficient::Eval(T, ip));
   }
};

// A coefficient that maps given gridfunctions with a given function.
// x |-> f(gf(x), other_gf(x))
class MappedPairedGFCoefficient : public GridFunctionCoefficient

{
   typedef std::function<real_t(const real_t, const real_t)> fun_type;
private:
   fun_type *fun;
   GridFunction *other_gf;
   int other_gf_comp;
   bool own_function;

public:
   // Create a coefficient that returns f(v1,v2) where v1=gf(x) and v2=other_gf(x)
   MappedPairedGFCoefficient(GridFunction &gf, GridFunction &other_gf,
                             fun_type &fun)
      : GridFunctionCoefficient(&gf), fun(&fun), other_gf(&other_gf),
        other_gf_comp(1), own_function(false) {}

   // Create only with function. Use SetGridFunction to set gridfunctions.
   // By default, the object takes the ownership.
   MappedPairedGFCoefficient(fun_type *fun, bool makeOwner=true)
      : GridFunctionCoefficient(nullptr), fun(fun), other_gf(nullptr),
        own_function(makeOwner) {}

   // Create an empty object. Use SetFunction and SetGridFunction
   MappedPairedGFCoefficient():GridFunctionCoefficient(nullptr), fun(nullptr),
      own_function(false) {}

   ~MappedPairedGFCoefficient()
   {
      if (own_function && fun) { delete fun; }
   }

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

   void SetFunction(fun_type &newfun)
   {
      if (own_function && fun) {delete fun;}
      fun=&newfun; own_function = false;
   }

   void SetFunction(fun_type *newfun, bool makeOwner=true)
   {
      if (own_function && fun) {delete fun;}
      fun=newfun; own_function = makeOwner;
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return fun->operator()(GridFunctionCoefficient::Eval(T, ip),
                             other_gf->GetValue(T, ip, other_gf_comp));
   }
};

// An entropy defined by Legendre function
// With Bregman divergence, this function can generate
// mapping between a convex set to a vector space
class LegendreEntropy : public Coefficient
{
   typedef std::function<real_t(const real_t)> fun_type;
public:
   fun_type entropy;
   fun_type forward; // primal to dual
   fun_type backward; // dual to primal
   LegendreEntropy(fun_type entropy, fun_type forward, fun_type backward)
      :entropy(entropy), forward(forward), backward(backward) {}
   MappedGFCoefficient GetForwardCoeff();
   MappedGFCoefficient GetBackwardCoeff();
   MappedGFCoefficient GetEntropyCoeff();
   MappedGFCoefficient GetForwardCoeff(GridFunction &x);
   MappedGFCoefficient GetBackwardCoeff(GridFunction &psi);
   MappedGFCoefficient GetEntropyCoeff(GridFunction &x);
   // Get Bregman divergence with primal variables
   MappedPairedGFCoefficient GetBregman(GridFunction &x, GridFunction &y);
   // Get Bregman divergence with dual variables
   MappedPairedGFCoefficient GetBregman_dual(GridFunction &psi, GridFunction &chi);
};

// Fermi-Dirac Entropy with effective domain (0,1)
class FermiDiracEntropy : public LegendreEntropy
{
public:
   FermiDiracEntropy():LegendreEntropy(
         [](const real_t x) {return x*safe_log(x)+(1.0-x)*safe_log(x);},
   invsigmoid, sigmoid) {}
};

// Shannon Entropy with effective domain (0,1)
class Shannon : public LegendreEntropy
{
public:
   Shannon():LegendreEntropy(
         [](const real_t x) {return x*safe_log(x)-1;},
   safe_log, [](const real_t x) {return std::exp(x);}) {}
};


} // end of namespace mfem
