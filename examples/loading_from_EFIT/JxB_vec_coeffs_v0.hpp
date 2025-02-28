#include "mfem.hpp"
#include <iostream>
#include <functional>

using namespace std;
using namespace mfem;

/// @brief Input $f$ and return $f/r$
class JPerpRVectorGridFunctionCoefficient : public VectorGridFunctionCoefficient
{
public:
    JPerpRVectorGridFunctionCoefficient() : VectorGridFunctionCoefficient() {}

    JPerpRVectorGridFunctionCoefficient(const GridFunction *gf) : VectorGridFunctionCoefficient(gf)
    {
    }

    void Eval(Vector &V, ElementTransformation &T,
              const IntegrationPoint &ip) override
    {
        // get r, z coordinates
        Vector x;
        T.Transform(ip, x);
        real_t r = x[0];

        VectorGridFunctionCoefficient::Eval(V, T, ip);
        V *= r;
    }
};

/// @brief Return $r$
class RGridFunctionCoefficient : public Coefficient
{
private:
   bool flip_sign;
public:
   int counter = 0;
   RGridFunctionCoefficient(bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      counter++;
      return r * (flip_sign ? -1 : 1);
   }
};

/// @brief Return $r^2$
class RSquareGridFunctionCoefficient : public Coefficient
{
private:
   bool flip_sign;
public:
   int counter = 0;
   RSquareGridFunctionCoefficient(bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      counter++;
      return r * r * (flip_sign ? -1 : 1);
   }
};