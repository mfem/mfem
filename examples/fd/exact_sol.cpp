
#include "exact_sol.hpp"

double lshape_exact(const Vector & pt)
{
   double x = pt[0];
   double y = pt[1];
   double r = sqrt(x*x + y*y);
   double alpha = 2. / 3.;

   double theta = atan2(y, x);
   if (y < 0) { theta += 2 * M_PI; }

   return pow(r,alpha) * sin(alpha * theta);
}

void lshape_grad(const Vector & x, Vector & grad)
{

}

double lshape_rhs(const Vector & x)
{
   return 0.0;
}

double wavefront_exact(const Vector & x)
{

}

void wavefront_grad(const Vector & x, Vector & grad)
{

}
double wavefront_rhs(const Vector & x)
{

}