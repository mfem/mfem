#ifndef HEATTRANSFER_OPT_HPP
#define HEATTRANSFER_OPT_HPP

#include "mfem.hpp"
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>

namespace mfem
{

// =============================================================================
// SIMP MATERIAL INTERPOLATION
// =============================================================================
// Computes r(ρ̃) = r_min + ρ̃^p (r_max - r_min)
class SIMPCoefficient : public Coefficient
{
private:
   ParGridFunction *rho_filter;  // Filtered density ρ̃
   real_t r_min, r_max;
   real_t exponent;

public:
   SIMPCoefficient(ParGridFunction *rho_filt, real_t rmin, real_t rmax, real_t p)
      : rho_filter(rho_filt), r_min(rmin), r_max(rmax), exponent(p) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);  // Clamp to [0,1]
      real_t rho_pow = std::pow(rho_val, exponent);
      return r_min + rho_pow * (r_max - r_min);
   }
 
   virtual real_t Eval_Derivative(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);  // Clamp to [0,1]
      real_t rho_pow = std::pow(rho_val, exponent-1.0);
      return exponent * rho_pow * (r_max - r_min);
   }
};

// =============================================================================
// Brinkman Coefficient
// =============================================================================
// 
class BrinkmanCoefficient : public Coefficient
{
private:
   ParGridFunction *rho_filter;  // Filtered density ρ̃
   real_t b;
   real_t a;

public:
   BrinkmanCoefficient(ParGridFunction *rho_filt, real_t b_, real_t a_)
      : rho_filter(rho_filt), b(b_), a(a_) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip); 
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);
      real_t val = a*(1 - rho_val) / (1 + b*rho_val);
      if(val < 1e-12){val = 1e-12;}
      else if(val > a){val = a;}
      return val;
   }

   virtual real_t Eval_Derivative(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      //rho_val = std::min(std::max(rho_val, 0.0), 1.0);
      return -a*((1+b*rho_val) + (1-rho_val)*b) / ((1 + b*rho_val)*(1 + b*rho_val));
   }

   void UpdateRho(ParGridFunction &rho_filt_new)
   {
      *rho_filter = rho_filt_new;
   }
};

// =============================================================================
//  RAMP INTERPOLATION
// =============================================================================
class RAMPCoefficient : public Coefficient
{
private:
   ParGridFunction *rho_filter;  // Filtered density ρ̃
   real_t f, s;
   real_t b;


public:
   RAMPCoefficient(ParGridFunction *rho_filt, real_t f_, real_t s_, real_t b_)
      : rho_filter(rho_filt), f(f_), s(s_), b(b_) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      real_t c = f / s;
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);  // Clamp to [0,1]
      real_t out = rho_val*f*((c*(1+b)-1)+1) / (c*(1+b*rho_val));
      return out;
   }
 
   virtual real_t Eval_Derivative(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t rho_val = rho_filter->GetValue(T, ip);
      real_t c = f / s;
      rho_val = std::min(std::max(rho_val, 0.0), 1.0);  // Clamp to [0,1]
      real_t low = (c*(1+b*rho_val));
      real_t high = rho_val*f*((c*(1+b)-1)+1);
      real_t dlow = c*b;
      real_t dhigh = f*((c*(1+b)-1)+1);
      return (low*dhigh - high*dlow) / (low * low);
   }
};



}
#endif 