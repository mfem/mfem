#include "mfem.hpp"
#include <iostream>
#include <functional>

using namespace std;
using namespace mfem;

/// @brief Input $J_perp$ and return $J_perp*r$
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
   real_t scale;
public:
   int counter = 0;
   RGridFunctionCoefficient(real_t scale = 1.0, bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign), scale(scale)
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
      return r * scale * (flip_sign ? -1 : 1);
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

/// @brief Compute r*(curl B_perp × B_perp_perp)
class RCurlBPerpBPerpPerpGridFunctionCoefficient : public VectorCoefficient
{
private:
   VectorGridFunctionCoefficient B_perp;
   GridFunctionCoefficient R_Curl_B_perp;

public:
   RCurlBPerpBPerpPerpGridFunctionCoefficient(GridFunction *R_Curl_B_perp, GridFunction *B_perp)
       : VectorCoefficient(2), B_perp(B_perp), R_Curl_B_perp(R_Curl_B_perp) {}
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector x;
      Vector B_perp_val;
      B_perp.Eval(B_perp_val, T, ip);
      real_t R_Curl_B_perp_val = R_Curl_B_perp.Eval(T, ip);
      V(0) = -R_Curl_B_perp_val * B_perp_val(1);
      V(1) = R_Curl_B_perp_val * B_perp_val(0);
   }
};
