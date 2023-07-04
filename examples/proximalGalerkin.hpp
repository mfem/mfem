#ifndef PROXIMAL_GALERKIN_HPP
#define PROXIMAL_GALERKIN_HPP
#include "mfem.hpp"


double sigmoid(const double x)
{
   if (x < 0)
   {
      const double exp_x = std::exp(x);
      return exp_x / (1.0 + exp_x);
   }
   return 1.0 / (1.0 + std::exp(-x));
}

double dsigmoiddx(const double x)
{
   const double tmp = sigmoid(x);
   return tmp - std::pow(tmp, 2);
}

double logit(const double x)
{
   return std::log(x / (1.0 - x));
}

double simpRule(const double rho, const int exponent, const double rho0)
{
   return rho0 + (1.0 - rho0)*std::pow(rho, exponent);
}

double dsimpRuledx(const double rho, const int exponent, const double rho0)
{
   return exponent*(1.0 - rho0)*std::pow(rho, exponent - 1);
}

double d2simpRuledx2(const double rho, const int exponent, const double rho0)
{
   return exponent*(exponent - 1)*(1.0 - rho0)*std::pow(rho, exponent - 2);
}


namespace mfem
{
class MappedGridFunctionCoefficient :public GridFunctionCoefficient
{
   typedef std::__1::function<double (const double)> Mapping;
public:
   MappedGridFunctionCoefficient(GridFunction *gf, Mapping f_x, int comp=1)
      :GridFunctionCoefficient(gf, comp), map(f_x) { }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return map(GridFunctionCoefficient::Eval(T, ip));
   }

   inline void SetMapping(Mapping &f_x) { map = f_x; }
private:
   Mapping &map;
};

MappedGridFunctionCoefficient SIMPCoefficient(GridFunction *gf,
                                              const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return simpRule(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient DerSIMPCoefficient(GridFunction *gf,
                                                 const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return dsimpRuledx(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient Der2SIMPCoefficient(GridFunction *gf,
                                                  const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return d2simpRuledx2(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient SigmoidCoefficient(GridFunction *gf)
{
   return MappedGridFunctionCoefficient(gf, sigmoid);
}

MappedGridFunctionCoefficient DerSigmoidCoefficient(GridFunction *gf)
{
   return MappedGridFunctionCoefficient(gf, dsigmoiddx);
}
} // end of namespace mfem

#endif // end of proximalGalerkin.hpp