#include "InitialConditions.hpp"

double vari_coeff=10.;
double visc_bdy=1e-2;
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
    const double k=2.*M_PI/Lx;
    return -(M_PI*M_PI+k*k)*beta*sin(M_PI*x(1))*cos(k*x(0));
}

double InitialPsi(const Vector &x)
{
    const double k=2.*M_PI/Lx;
    return -x(1)+beta*sin(M_PI*x(1))*cos(k*x(0));
}

double exactPhi1(const Vector &x, double t)
{
    const double k=2.*M_PI/Lx;
    return -beta*sin(M_PI*x(1))*sin(k*x(0))*sin(k*t);
}

double exactW1(const Vector &x, double t)
{
    const double k=2.*M_PI/Lx;
    return beta*(M_PI*M_PI+k*k)*sin(M_PI*x(1))*sin(k*x(0))*sin(k*t);
}

double exactPsi1(const Vector &x, double t)
{
    const double k=2.*M_PI/Lx;
    return -x(1)+beta*sin(M_PI*x(1))*cos(k*x(0))*cos(k*t);
}


double exactPhiRe(const Vector &x, double t)
{
    double d=1, nu=1, eta=1, U=1;
    double rho=4e-5, B0=1.4494e-4, mu0=1.256636e-6, A0=B0/sqrt(mu0*rho);
    double Y=x(1), X=x(0);

    //the second erf needs to be replace by erfc to avoid numerical issue
    return U/4/A0*( (t*A0*A0+d*exp(-A0*Y/d)-A0*Y+d)*erf((A0*t-Y)/2./sqrt(d*t))          
                 -  (t*A0*A0+d*exp( A0*Y/d)+A0*Y+d)*erfc((A0*t+Y)/2./sqrt(d*t)) + t*A0*A0 + d*exp(-A0*Y/d)-A0*Y+d )+ U*sqrt(d*t)/2./sqrt(M_PI)*(exp(-(A0*t-Y)*(A0*t-Y)/4./d/t) + exp(-(A0*t+Y)*(A0*t+Y)/4./d/t) );
}

double exactWRe(const Vector &x, double t)
{
    double d=1, nu=1, eta=1, U=1;
    double rho=4e-5, B0=1.4494e-4, mu0=1.256636e-6, A0=B0/sqrt(mu0*rho);
    double Y=x(1), X=x(0);

    return U*A0/4./d*( 2*exp(-A0*Y/d)  - exp(A0*Y/d)*erfc(  ( Y+A0*t)/2/sqrt(d*t))  - exp(-A0*Y/d)*erfc( (-Y+A0*t)/2/sqrt(d*t))) 
      + U/2./sqrt(M_PI*d*t)*(exp(-(A0*t-Y)*(A0*t-Y)/4/d/t) + exp(-(A0*t+Y)*(A0*t+Y)/4/d/t) );
}

double exactPsiRe(const Vector &x, double t)
{
    double d=1, nu=1, eta=1, U=1;
    double rho=4e-5, B0=1.4494e-4, mu0=1.256636e-6, A0=B0/sqrt(mu0*rho);
    double Y=x(1), X=x(0);

    return B0*X/sqrt(mu0*rho)-U*sqrt(d*t)/2/sqrt(M_PI)*( exp(-(A0*t-Y)*(A0*t-Y)/4/d/t) - exp(-(A0*t+Y)*(A0*t+Y)/4/d/t) ) 
      - U/4/A0*(t*A0*A0+d)*( erf((A0*t-Y)/2/sqrt(d*t)) - erf((A0*t+Y)/2/sqrt(d*t) ) )    
      + U/4/A0*( (d*exp(-A0*Y/d) + A0*Y)*erfc((Y-A0*t)/2./sqrt(d*t)) + (d*exp(A0*Y/d)-A0*Y)*erfc((A0*t+Y)/2./sqrt(d*t)));
}



double BackPsi(const Vector &x)
{
    //this is the background psi (for post-processing/plotting only)
    return -x(1);
}

double E0rhs1(const Vector &x, double t)
{
    //E0=d^2*sin(k*t)*sin(Pi*y)*k*cos(Pi*y)*cos(k*t)*Pi
    const double k=2.*M_PI/Lx;
    return beta*beta*sin(k*t)*sin(M_PI*x(1))*k*cos(M_PI*x(1))*cos(k*t)*M_PI;
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

double InitialPsi32(const Vector &x)
{
    double qty0=cosh(1./lambda);
    return -lambda*log( (cosh(x(1)/lambda) +ep*cos(x(0)/lambda))/qty0 )
           +beta*cos(M_PI*.5*x(1))*cos(M_PI*x(0));
}

double BackPsi32(const Vector &x)
{
    double qty0=cosh(1./lambda);
    return -lambda*log( (cosh(x(1)/lambda) +ep*cos(x(0)/lambda))/qty0 );
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


