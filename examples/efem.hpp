#ifndef MFEM_EFEM_HPP
#define MFEM_EFEM_HPP

#include "mfem.hpp"

using namespace std;
using namespace mfem;

/**
 * @brief Inverse sigmoid, log(x/(1-x))
 *
 * @param x -
 * @param tol tolerance to force x ∈ (tol, 1 - tol)
 * @return double log(x/(1-x))
 */
double invsigmoid(const double x, const double tol=1e-12)
{
   // forcing x to be in (0, 1)
   const double clipped_x = min(max(tol,x),1.0-tol);
   return log(clipped_x/(1.0-clipped_x));
}

// Sigmoid function
double sigmoid(const double x)
{
   return x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}

// Derivative of sigmoid function d(sigmoid)/dx
double dsigdx(const double x)
{
   double tmp = sigmoid(-x);
   return tmp - pow(tmp,2);
}


/**
 * @brief A coefficient that maps u to f(u).
 *
 */
class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
   // lambda function maps double to double
   typedef std::__1::function<double(const double)> __LambdaFunction;
private:
   __LambdaFunction fun; // a lambda function f(u(x))
public:
   /**
    * @brief Construct a mapped grid function coefficient with given gridfunction and lambda function
    *
    * @param[in] gf u
    * @param[in] double_to_double lambda function, f(x)
    * @param[in] comp (Optional) index of a vector if u is a vector
    */
   MappedGridFunctionCoefficient(const GridFunction *gf,
                                 __LambdaFunction double_to_double, int comp = 1): GridFunctionCoefficient(gf,
                                          comp), fun(double_to_double) {};

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      const double value = GridFunctionCoefficient::Eval(T, ip);
      return fun(value);
   }
};

/**
 * @brief GridFunctionCoefficient that returns exp(u)
 *
 */
class ExponentialGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
   ExponentialGridFunctionCoefficient(const GridFunction *gf,
                                      int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return exp(x);},
   comp) {};
};


/**
 * @brief GridFunctionCoefficient that returns log(u)
 *
 */
class LogarithmicGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
   LogarithmicGridFunctionCoefficient(const GridFunction *gf,
                                      int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return log(x);},
   comp) {};
};


/**
 * @brief GridFunctionCoefficient that returns sigmoid(u)
 *
 */
class SigmoidGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
   SigmoidGridFunctionCoefficient(const GridFunction *gf,
                                  int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return sigmoid(x);},
   comp) {};
};


/**
 * @brief GridFunctionCoefficient that returns invsigmoid(u)
 *
 */
class InvSigmoidGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
   InvSigmoidGridFunctionCoefficient(const GridFunction *gf,
                                     int comp=1):MappedGridFunctionCoefficient(gf, [](const double x) {return invsigmoid(x);},
   comp) {};
};


/**
 * @brief GridFunctionCoefficient that returns pow(u, exponent)
 *
 */
class PowerGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
   PowerGridFunctionCoefficient(const GridFunction *gf, int exponent, int comp=1)
      : MappedGridFunctionCoefficient(gf, [exponent](double x) {return pow(x, exponent);},
   comp) {}
};


/**
 * @brief GridFunctionCoefficient that returns u^2
 *
 */
class PowerGridFunctionCoefficient : public MappedGridFunctionCoefficient
{
   PowerGridFunctionCoefficient(const GridFunction *gf, int exponent, int comp=1)
      : MappedGridFunctionCoefficient(gf, [](const double x) {return x*x;},
   comp) {}
};

/**
 * @brief SIMP Rule, r(ρ) = ρ_0 + (1-ρ_0)ρ^p
 * 
 */
class SIMPCoefficient : public MappedGridFunctionCoefficient
{
   /**
    * @brief Make a GridFunctionCoefficient that computes r(ρ) = ρ_0 + (1-ρ_0)ρ^p
    * 
    * @param gf Density, ρ
    * @param exponent Exponent, p
    * @param rho_min minimum density, ρ_0
    */
   SIMPCoefficient(const const GridFunction *gf, const double exponent, const double rho_min=1e-12)
   : MappedGridFunctionCoefficient(gf, [rho_min, exponent](const double x){return rho_min + (1-rho_min)*pow(x, exponent);}){};
};



#endif