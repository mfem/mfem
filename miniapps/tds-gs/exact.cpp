#include "mfem.hpp"
#include "exact.hpp"
#include <math.h>

using namespace std;
using namespace mfem;


double psi_exact(double r, double z, double r0, double z0, double k) {
  return - cos(k * (z - r - z0 + r0)) * cos(k * (z + r - z0 - r0)) + 1.0;
}




double ExactCoefficient::Eval(ElementTransformation & T,
                              const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double r(x(0));
   double z(x(1));

   if (true) {
     return 1.0;
   }
   return psi_exact(r, z, r0, z0, k);
}

double ExactForcingCoefficient::Eval(ElementTransformation & T,
                                     const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double r(x(0));
   double z(x(1));

   double mu = model.get_mu();
   double ans;

   if (true) {
     return 0.0;
   }
   if (true) {
     // I u = ...
     double psi_N = psi_exact(r, z, r0, z0, k);
     ans = psi_N;
     // ans -= r * model.S_p_prime(psi_N);
     // ans -= model.S_ff_prime(psi_N) / (r * mu);
     ans -= pow(psi_N, 2.0);
     if ((T.Attribute > 832) && (T.Attribute <= 838)){
       // coil region
       ans -= 1.0;
     }
     return ans;
   }

   
   // if (true) {
   //   ans = - 4 * k * k * cos(k * (z - r - z0 + r0)) * cos(k * (z + r - z0 - r0));
   //   return ans;
   // }

   ans = - 4.0 * pow(k, 2.0) / r / mu * cos(k * (z - r - z0 + r0)) * cos(k * (z + r - z0 - r0)) \
     - k / pow(r, 2.0) / mu * sin(k * (z - r - z0 + r0)) * cos(k * (z + r - z0 - r0)) \
     + k / pow(r, 2.0) / mu * cos(k * (z - r - z0 + r0)) * sin(k * (z + r - z0 - r0));

   double L = M_PI/(2.0*k);
   if (abs(r - r0) + abs(z - z0) <= L) {
     // inside plasma region
     double psi_N = psi_exact(r, z, r0, z0, k);
     ans -= r * model.S_p_prime(psi_N);
     ans -= model.S_ff_prime(psi_N) / (r * mu);
   }

   if ((T.Attribute > 832) && (T.Attribute <= 838)){
     // coil region
     // TODO, handle this better...
     ans -= 1.0;
   }
   
   
   return ans;
}
