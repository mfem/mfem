//--------------------------------------------------------------
// To compile run, for example:
// mpicc -shared -o libfun.so -fPIC airy.c
// gcc   -shared -o libfun.so -fPIC airy.c
//--------------------------------------------------------------

#include <math.h>

double h = 1.0;
double g = 9.81;
double k = 2*M_PI/1.0;
double a = 0.001;

double Omega(double k)
{
   return sqrt(g*k*tanh(k*h));
}

double phi(double *x, int dim, double time)
{
   double omega = Omega(k);
   return a*omega/k *((cosh(k*(h+x[1])))/(sinh(k*h)))*sin(k*x[0]);
}


double dphidt(double *x, int dim, double time)
{
   double omega = Omega(k);
   return a*omega*omega/k *((cosh(k*(h+x[1])))/(sinh(k*h)))*cos(k*x[0]);
}

double eta(double *x, int dim, double time)
{
   return -(1.0/9.81)*dphidt(x, dim, time);
}

double motion(double *x, int dim, double time)
{
   double omega = Omega(k);
   return 0.0;
}
 
