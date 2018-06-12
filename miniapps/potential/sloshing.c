//--------------------------------------------------------------
// To compile run, for example:
// mpicc -shared -o libfun.so -fPIC sloshing.c
// gcc   -shared -o libfun.so -fPIC sloshing.c
//--------------------------------------------------------------

#include <math.h>

double phi(double *x, int dim, double time)
{
   return 0.001*cos(2*M_PI*x[0])*cosh(2*M_PI*x[1]) ;
}

double dphidt(double *x, int dim, double time)
{
   return 0.0;
}

double eta(double *x, int dim, double time)
{
   return 0.0;
}

double motion(double *x, int dim, double time)
{
   return 0.0;
}

