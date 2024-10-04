#include "funs.hpp"

namespace mfem
{

real_t safe_log(const real_t x)
{
   return x<LOGMIN ? LOGMIN_VAL : std::log(x);
}

real_t sigmoid(const real_t x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+std::exp(-x));
   }
   else
   {
      const real_t expx = std::exp(x);
      return expx/(1.0+expx);
   }
}

real_t invsigmoid(const real_t x)
{
   return x < 0.5 ? safe_log(x/(1.0-x)) : -safe_log((1.0-x)/x);
}

real_t der_sigmoid(const real_t x)
{
   const real_t sigmoidx = sigmoid(x);
   return sigmoidx*(1.0-sigmoidx);
}

real_t simp(const real_t x, const real_t exponent, const real_t rho0)
{
   return rho0 + (1.0-rho0)*std::pow(x, exponent);
}

real_t der_simp(const real_t x, const real_t exponent, const real_t rho0)
{
   return exponent*(1.0-rho0)*std::pow(x, exponent-1.0);
}

MappedGFCoefficient LegendreEntropy::GetForwardCoeff()
{
   MappedGFCoefficient coeff;
   coeff.SetFunction(forward);
   return coeff;
}

MappedGFCoefficient LegendreEntropy::GetForwardCoeff(GridFunction &gf)
{
   auto coeff = GetForwardCoeff();
   coeff.SetGridFunction(&gf);
   return coeff;
}

MappedGFCoefficient LegendreEntropy::GetBackwardCoeff()
{
   MappedGFCoefficient coeff;
   coeff.SetFunction(backward);
   return coeff;
}

MappedGFCoefficient LegendreEntropy::GetBackwardCoeff(GridFunction &gf)
{
   auto coeff = GetBackwardCoeff();
   coeff.SetGridFunction(&gf);
   return coeff;
}


MappedGFCoefficient LegendreEntropy::GetEntropyCoeff()
{
   MappedGFCoefficient coeff;
   coeff.SetFunction(entropy);
   return coeff;
}

MappedGFCoefficient LegendreEntropy::GetEntropyCoeff(GridFunction &gf)
{
   auto coeff = GetEntropyCoeff();
   coeff.SetGridFunction(&gf);
   return coeff;
}

MappedPairedGFCoefficient LegendreEntropy::GetBregman(GridFunction &x,
                                                      GridFunction &y)
{
   MappedPairedGFCoefficient coeff;
   coeff.SetGridFunction(&x, &y);
   coeff.SetFunction([this](const real_t p, const real_t q)
   {
      return std::max(0.0, this->entropy(p) - this->entropy(q) - this->forward(q)*(p-q));
   });
   return coeff;
}

MappedPairedGFCoefficient LegendreEntropy::GetBregman_dual(GridFunction &x,
                                                           GridFunction &y)
{
   MappedPairedGFCoefficient coeff;
   coeff.SetGridFunction(&x, &y);
   coeff.SetFunction([this](const real_t p_dual, const real_t q_dual)
   {
      const real_t p=this->backward(p_dual);
      const real_t q=this->backward(q_dual);
      return std::max(0.0, this->entropy(p) - this->entropy(q) - q_dual*(p-q));
   });
   return coeff;
}

}
