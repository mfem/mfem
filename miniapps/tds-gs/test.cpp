#include "mfem.hpp"
#include "test.hpp"
#include <math.h>

using namespace std;
using namespace mfem;

int test()
{
  double a = 1.0;
  // double b = S_p_prime(a);
  // cout << "a" << a << "b" << b << endl;
  return 1;
}

double TestCoefficient::Eval(ElementTransformation & T,
                             const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double x1(x(0));
   double x2(x(1));
   double k = 4.0;
   
   // return pow(x1, 2.0) * pow(x2, 3.0);
   return cos(k * x1) * cos(k * x2) * exp(- pow(x1, 2.0) - pow(x2, 2.0));
   // return pow(x1 - 1.0, 2.0) + pow(x2, 2.0);
   // return pow(x1 - 1.0, 2.0) - pow(x2, 2.0);
}
