#ifndef __LOR_MMS_HPP__
#define __LOR_MMS_HPP__

extern int dim, problem;

namespace mfem
{

double f_exact(const Vector &xvec)
{
   double x=xvec[0], y=xvec[1], z;
   if (dim == 3) { z = xvec[2]; }
   if (problem == 0)
   {
      if (dim == 3)
      {
         return y*(1.0 - y) + z*(1.0 - z) + x*y*(1.0 - y)*z*(1.0 - z);
      }
      else
      {
         return y*(1.0 - y) + 2.0;
      }
   }
   else
   {
      constexpr double kappa = 2.0*M_PI;
      if (dim == 3)
      {
         return sin(kappa*x)*sin(kappa*y)*sin(kappa*z);
      }
      else
      {
         return sin(kappa*x)*sin(kappa*y);
      }
   }
}

void f_exact_vec(const Vector &xvec, Vector &f)
{
   double x=xvec[0], y=xvec[1], z;
   if (dim == 3) { z = xvec[2]; }

   if (problem == 0)
   {
      if (dim == 3)
      {
         f(0) = x*y*(1.0 - y)*z*(1.0 - z);
         f(1) = x*y*(1.0 - x)*z*(1.0 - z);
         f(2) = x*z*(1.0 - x)*y*(1.0 - y);

         f(0) += y*(1.0 - y) + z*(1.0 - z);
         f(1) += x*(1.0 - x) + z*(1.0 - z);
         f(2) += x*(1.0 - x) + y*(1.0 - y);
      }
      else
      {
         f(0) = y*(1.0 - y);
         f(1) = x*(1.0 - x);

         f(0) += 2.0;
         f(1) += 2.0;
      }
   }
   else
   {
      constexpr double kappa = 2.0*M_PI;
      if (dim == 3)
      {
         f(0) = (1.0 + 2.0*kappa*kappa)*sin(kappa*y)*sin(kappa*z);
         f(1) = (1.0 + 2.0*kappa*kappa)*sin(kappa*x)*sin(kappa*z);
         f(2) = (1.0 + 2.0*kappa*kappa)*sin(kappa*x)*sin(kappa*y);
      }
      else
      {
         f(0) = (1. + kappa*kappa)*sin(kappa*y);
         f(1) = (1. + kappa*kappa)*sin(kappa*x);
         if (xvec.Size() == 3) { f(2) = 0.0; }
      }
   }
}

} // namespace mfem

#endif
