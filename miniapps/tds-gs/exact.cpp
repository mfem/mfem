#include "mfem.hpp"
#include "exact.hpp"
#include <math.h>

using namespace std;
using namespace mfem;


double psi_exact(double r, double z, double r0, double z0, double k) {
  return - cos(k * (z - r - z0 + r0)) * cos(k * (z + r - z0 - r0)) + 1.0;
}
double psi_r_exact(double r, double z, double r0, double z0, double k) {
  return k * cos(k * (z - r - z0 + r0)) * sin(k * (z + r - z0 - r0))
    - k * sin(k * (z - r - z0 + r0)) * cos(k * (z + r - z0 - r0));
}
double psi_rr_exact(double r, double z, double r0, double z0, double k) {
  return 2 * k * k * sin(k * (z - r - z0 + r0)) * sin(k * (z + r - z0 - r0))
    + 2 * k * k * cos(k * (z - r - z0 + r0)) * cos(k * (z + r - z0 - r0));
}
double psi_zz_exact(double r, double z, double r0, double z0, double k) {
  return - 2 * k * k * sin(k * (z - r - z0 + r0)) * sin(k * (z + r - z0 - r0))
    + 2 * k * k * cos(k * (z - r - z0 + r0)) * cos(k * (z + r - z0 - r0));
}



double ExactCoefficient::Eval(ElementTransformation & T,
                              const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double r(x(0));
   double z(x(1));

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
   double coeff_u2 = model.get_coeff_u2();
   double ans;

   if (false) {
     return 0.0;
   }

   if (false) {
     // I u = ...
     double psi_N = psi_exact(r, z, r0, z0, k);
     ans = psi_N;
     // ans -= r * model.S_p_prime(psi_N);
     // ans -= model.S_ff_prime(psi_N) / (r * mu);
     ans -= coeff_u2 * pow(psi_N, 2.0);
     if ((T.Attribute > 832) && (T.Attribute <= 838)){
       // coil region
       // ans -= 1.0;
     }
     // cout << ans << endl;
     return ans;
   }
   if (true) {
     // - 1 / (r mu) Delta* u = - 1 / (r mu) u_rr + 1 / (r^2 mu) u_r  - 1 / (r mu) u_zz = ...
     double psi_N = psi_exact(r, z, r0, z0, k);
     ans =
       - 1 / (r * mu) * psi_rr_exact(r, z, r0, z0, k)
       + 1 / (r * r * mu) * psi_r_exact(r, z, r0, z0, k)
       - 1 / (r * mu) * psi_zz_exact(r, z, r0, z0, k);
     ans -= coeff_u2 * pow(psi_N, 2.0);

     return ans;
   }

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
