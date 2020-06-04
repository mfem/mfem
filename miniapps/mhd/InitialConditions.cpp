#include "InitialConditions.hpp"

double vari_coeff=8.;
double visc_bdy=5e-4;
double tau=200.;

double InitialPhi(const Vector &x)
{
    return 0.0;
}

double InitialW(const Vector &x)
{
    return 0.0;
}

double InitialJ(const Vector &x)
{
   return -M_PI*M_PI*(1.0+4.0/Lx/Lx)*beta*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double InitialPsi(const Vector &x)
{
   return -x(1)+beta*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double BackPsi(const Vector &x)
{
   //this is the background psi (for post-processing/plotting only)
   return -x(1);
}

double InitialJ2(const Vector &x)
{
   return lambda/pow(cosh(lambda*(x(1)-.5)),2)
       -M_PI*M_PI*(1.0+4.0/Lx/Lx)*beta*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double InitialPsi2(const Vector &x)
{
   return log(cosh(lambda*(x(1)-.5)))/lambda
       +beta*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double BackPsi2(const Vector &x)
{
   //this is the background psi (for post-processing/plotting only)
   return log(cosh(lambda*(x(1)-.5)))/lambda;
}

double E0rhs(const Vector &x)
{
   //for icase 2 only, there is a rhs
   return resiG*lambda/pow(cosh(lambda*(x(1)-.5)),2);
}

double InitialJ3(const Vector &x)
{
   return (ep*ep-1.)/lambda/pow(cosh(x(1)/lambda) +ep*cos(x(0)/lambda), 2)
        -M_PI*M_PI*1.25*beta*cos(.5*M_PI*x(1))*cos(M_PI*x(0));
}

double InitialPsi3(const Vector &x)
{
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) )
          +beta*cos(M_PI*.5*x(1))*cos(M_PI*x(0));
}

double InitialJ6(const Vector &x)
{
   double tmp=beta*exp(-vari_coeff*x(1)*x(1))*cos(.5*M_PI*x(1))*cos(M_PI*x(0));
   return (ep*ep-1.)/lambda/pow(cosh(x(1)/lambda) +ep*cos(x(0)/lambda), 2)
        -M_PI*M_PI*1.25*tmp + (4.*vari_coeff*vari_coeff*x(1)*x(1)-2.*vari_coeff)*tmp
        +beta*exp(-vari_coeff*x(1)*x(1))*sin(.5*M_PI*x(1))*cos(M_PI*x(0))*2.*M_PI*x(1)*vari_coeff;
}

double InitialPsi6(const Vector &x)
{
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) )
          +beta*exp(-vari_coeff*x(1)*x(1))*cos(M_PI*.5*x(1))*cos(M_PI*x(0));
}

double BackPsi3(const Vector &x)
{
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) );
}

double E0rhs3(const Vector &x)
{
   return resiG*(ep*ep-1.)/lambda/pow(cosh(x(1)/lambda) +ep*cos(x(0)/lambda), 2);
}

double resiVari(const Vector &x)
{
   return resiG+exp(-vari_coeff*(1.-x(1)*x(1)))*(visc_bdy-resiG);
}

double resiVari_y(const Vector &x)
{
   return vari_coeff*2.*x(1)*exp(-vari_coeff*(1.-x(1)*x(1)))*(visc_bdy-resiG);
}

double E0rhs5(const Vector &x)
{
   //E0=grad.(coeff grad psi0)
   return resiVari(x)*(ep*ep-1.)/lambda/pow(cosh(x(1)/lambda) +ep*cos(x(0)/lambda), 2)
         +resiVari_y(x)*(-1.)*sinh(x(1)/lambda)/(cosh(x(1)/lambda) +ep*cos(x(0)/lambda));
}

double InitialJ4(const Vector &x)
{
   return (ep*ep-1.)/lambda/pow(cosh(x(1)/lambda) +ep*cos(x(0)/lambda), 2)
         +beta*exp(-tau*x(1)*x(1))*cos(M_PI*x(0))*(pow(2.*tau*x(1), 2)-M_PI*M_PI-2.*tau);
}

double InitialPsi4(const Vector &x)
{
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) )
          +beta*exp(-tau*x(1)*x(1))*cos(M_PI*x(0));
}


