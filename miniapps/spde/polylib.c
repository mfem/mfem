#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "polylib.h"
#include <float.h>
#define STOP  30 
#define EPS   100*DBL_EPSILON
#define sign(a,b) ((b)<0 ? -fabs(a) : fabs(a))

/*
LIBRARY ROUTINES FOR ORTHOGONAL POLYNOMIAL CALCULUS AND INTERPOLATION

  Spencer Sherwin
  Aeronautics, Imperial College London
  
  Based on codes by Einar Ronquist and Ron Henderson

  Abbreviations
  - z    -   Set of collocation/quadrature points
  - w    -   Set of quadrature weights
  - D    -   Derivative matrix
  - h    -   Lagrange Interpolant
  - I    -   Interpolation matrix
  - g    -   Gauss
  - gr   -   Gauss-Radau
  - gl   -   Gauss-Lobatto
  - j    -   Jacobi
  - l    -   Legendre  (Jacobi with alpha = beta =  0.0)
  - c    -   Chebychev (Jacobi with alpha = beta = -0.5)
  - m    -   point at minus 1 in Radau rules
  - p    -   point at plus  1 in Radau rules

  -----------------------------------------------------------------------
                         M A I N     R O U T I N E S
  -----------------------------------------------------------------------

  Points and Weights:

  zwgj        Compute Gauss-Jacobi         points and weights
  zwgrjm      Compute Gauss-Radau-Jacobi   points and weights (z=-1)
  zwgrjp      Compute Gauss-Radau-Jacobi   points and weights (z= 1)
  zwglj       Compute Gauss-Lobatto-Jacobi points and weights

  Derivative Matrices:

  Dgj         Compute Gauss-Jacobi         derivative matrix
  Dgrjm       Compute Gauss-Radau-Jacobi   derivative matrix (z=-1)
  Dgrjp       Compute Gauss-Radau-Jacobi   derivative matrix (z= 1)
  Dglj        Compute Gauss-Lobatto-Jacobi derivative matrix

  Lagrange Interpolants:

  hgj         Compute Gauss-Jacobi         Lagrange interpolants
  hgrjm       Compute Gauss-Radau-Jacobi   Lagrange interpolants (z=-1)
  hgrjp       Compute Gauss-Radau-Jacobi   Lagrange interpolants (z= 1)
  hglj        Compute Gauss-Lobatto-Jacobi Lagrange interpolants

  Interpolation Operators:

  Imgj        Compute interpolation operator gj->m
  Imgrjm      Compute interpolation operator grj->m (z=-1)
  Imgrjp      Compute interpolation operator grj->m (z= 1)
  Imglj       Compute interpolation operator glj->m

  Polynomial Evaluation:

  jacobfd     Returns value and derivative of Jacobi poly. at point z
  jacobd      Returns derivative of Jacobi poly. at point z (valid at z=-1,1)

  -----------------------------------------------------------------------
                     L O C A L      R O U T I N E S
  -----------------------------------------------------------------------

  jacobz      Returns Jacobi polynomial zeros
  gammaf      Gamma function for integer values and halves

  -----------------------------------------------------------------------
                         M A C R O S
  -----------------------------------------------------------------------
  
  Legendre  polynomial alpha = beta = 0
  Chebychev polynomial alpha = beta = -0.5

  Points and Weights:

  zwgl        Compute Gauss-Legendre          points and weights
  zwgrlm      Compute Gauss-Radau-Legendre    points and weights (z=-1)
  zwgrlp      Compute Gauss-Radau-Legendre    points and weights (z=+1)
  zwgll       Compute Gauss-Lobatto-Legendre  points and weights

  zwgc        Compute Gauss-Chebychev         points and weights
  zwgrcm      Compute Gauss-Radau-Chebychev   points and weights (z=-1)
  zwgrcp      Compute Gauss-Radau-Chebychev   points and weights (z=+1)
  zwglc       Compute Gauss-Lobatto-Chebychev points and weights

  Derivative Operators:

  Dgl         Compute Gauss-Legendre          derivative matrix
  Dgrlm       Compute Gauss-Radau-Legendre    derivative matrix (z=-1)
  Dgrlp       Compute Gauss-Radau-Legendre    derivative matrix (z=+1)
  Dgll        Compute Gauss-Lobatto-Legendre  derivative matrix

  Dgc         Compute Gauss-Chebychev         derivative matrix
  Dgrcm       Compute Gauss-Radau-Chebychev   derivative matrix (z=-1)
  Dgrcp       Compute Gauss-Radau-Chebychev   derivative matrix (z=+1)
  Dglc        Compute Gauss-Lobatto-Chebychev derivative matrix

  Lagrangian Interpolants:

  hgl         Compute Gauss-Legendre          Lagrange interpolants
  hgrlm       Compute Gauss-Radau-Legendre    Lagrange interpolants (z=-1)
  hgrlp       Compute Gauss-Radau-Legendre    Lagrange interpolants (z=+1)
  hgll        Compute Gauss-Lobatto-Legendre  Lagrange interpolants

  hgc         Compute Gauss-Chebychev         Lagrange interpolants
  hgrcm       Compute Gauss-Radau-Chebychev   Lagrange interpolants (z=-1)
  hgrcp       Compute Gauss-Radau-Chebychev   Lagrange interpolants (z=+1)
  hglc        Compute Gauss-Lobatto-Chebychev Lagrange interpolants

  Interpolation Operators:

  Imgl        Compute interpolation operator gl->m
  Imgrlm      Compute interpolation operator grl->m (z=-1)
  Imgrlp      Compute interpolation operator grl->m (z=+1)
  Imgll       Compute interpolation operator gll->m

  Imgc        Compute interpolation operator gc->m
  Imgrcm      Compute interpolation operator grc->m (z=-1)
  Imgrcp      Compute interpolation operator grc->m (z=+1)
  Imglc       Compute interpolation operator glc->m

  ------------------------------------------------------------------------

  Useful references:

  - [1] Gabor Szego: Orthogonal Polynomials, American Mathematical Society,
      Providence, Rhode Island, 1939.
  - [2] Abramowitz \& Stegun: Handbook of Mathematical Functions,
      Dover, New York, 1972.
  - [3] Canuto, Hussaini, Quarteroni \& Zang: Spectral Methods in Fluid
      Dynamics, Springer-Verlag, 1988.
  - [4] Ghizzetti \& Ossicini: Quadrature Formulae, Academic Press, 1970.
  - [5] Karniadakis \& Sherwin: Spectral/hp element methods for CFD, 1999


  NOTES
  -----
  (1) All routines are double precision.  
  (2) All array subscripts start from zero, i.e. vector[0..N-1] 
*/

#ifdef __cplusplus
namespace polylib {
#endif


#if 0 
  /// zero determination using Newton iteration with polynomial deflation
#define jacobz(n,z,alpha,beta) Jacobz(n,z,alpha,beta)
#else
  /// zero determination using eigenvalues of tridiagaonl matrix 
#define jacobz(n,z,alpha,beta) JacZeros(n,z,alpha,beta)
#endif



/* local functions */
static void   Jacobz   (int n, double *z, double alpha, double beta);
static void   JacZeros (int n, double *a, double alpha, double beta);
static void   TriQL    (int n, double *d, double *e);

static double gammaF (double);

/**
   \brief  Gauss-Jacobi zeros and weights.

   \li Generate \a np Gauss Jacobi zeros, \a z, and weights,\a w,
   associated with the Jacobi polynomial \f$ P^{\alpha,\beta}_{np}(z)
   \f$,

   \li Exact for polynomials of order \a 2np-1 or less  
*/

void zwgj (double *z, double *w, int np, double alpha, double beta){
  register int i;
  double fac, one = 1.0, two = 2.0, apb = alpha + beta;

  jacobz (np,z,alpha,beta);
  jacobd (np,z,w,np,alpha,beta);

  fac  = pow(two,apb + one)*gammaF(alpha + np + one)*gammaF(beta + np + one);
  fac /= gammaF(np + one)*gammaF(apb + np + one);
  
  for(i = 0; i < np; ++i) w[i] = fac/(w[i]*w[i]*(one-z[i]*z[i]));
  
  return;
}


/** 
  \brief  Gauss-Radau-Jacobi zeros and weights with end point at \a z=-1.

  \li Generate \a np Gauss-Radau-Jacobi zeros, \a z, and weights,\a w,
  associated with the  polynomial \f$(1+z) P^{\alpha,\beta+1}_{np-1}(z)
  \f$.

  \li  Exact for polynomials of order \a 2np-2 or less    
*/

void zwgrjm(double *z, double *w, int np, double alpha, double beta){

  if(np == 1){
    z[0] = 0.0;
    w[0] = 2.0;
  }
  else{
    register int i;
    double fac, one = 1.0, two = 2.0, apb = alpha + beta;
    
    z[0] = -one;
    jacobz  (np-1,z+1,alpha,beta+1);
    jacobfd (np,z,w,NULL,np-1,alpha,beta);
    
    fac  = pow(two,apb)*gammaF(alpha + np)*gammaF(beta + np);
    fac /= gammaF(np)*(beta + np)*gammaF(apb + np + 1);

    for(i = 0; i < np; ++i) w[i] = fac*(1-z[i])/(w[i]*w[i]);
    w[0] *= (beta + one);
  }

  return;
}


/** 
  \brief  Gauss-Radau-Jacobi zeros and weights with end point at \a z=1
 

  \li Generate \a np Gauss-Radau-Jacobi zeros, \a z, and weights,\a w,
  associated with the  polynomial \f$(1-z) P^{\alpha+1,\beta}_{np-1}(z)
  \f$.

  \li Exact for polynomials of order \a 2np-2 or less    
*/

void zwgrjp(double *z, double *w, int np, double alpha, double beta){

  if(np == 1){
    z[0] = 0.0;
    w[0] = 2.0;
  }
  else{
    register int i;
    double fac, one = 1.0, two = 2.0, apb = alpha + beta;
    
    jacobz  (np-1,z,alpha+1,beta);
    z[np-1] = one;
    jacobfd (np,z,w,NULL,np-1,alpha,beta);
    
    fac  = pow(two,apb)*gammaF(alpha + np)*gammaF(beta + np);
    fac /= gammaF(np)*(alpha + np)*gammaF(apb + np + 1);

    for(i = 0; i < np; ++i) w[i] = fac*(1+z[i])/(w[i]*w[i]);
    w[np-1] *= (alpha + one);
  }

  return;
}


/** 
  \brief  Gauss-Lobatto-Jacobi zeros and weights with end point at \a z=-1,\a 1


  \li Generate \a np Gauss-Lobatto-Jacobi points, \a z, and weights, \a w,
  associated with polynomial \f$ (1-z)(1+z) P^{\alpha+1,\beta+1}_{np-2}(z) \f$
  \li Exact for polynomials of order \a 2np-3 or less
*/

void zwglj(double *z, double *w, int np, double alpha, double beta){

  if( np == 1 ){
    z[0] = 0.0;
    w[0] = 2.0;
  }
  else{
    register int i;
    double   fac, one = 1.0, apb = alpha + beta, two = 2.0;
  
    z[0]    = -one;
    z[np-1] =  one;
    jacobz  (np-2,z + 1,alpha + one,beta + one); 
    jacobfd (np,z,w,NULL,np-1,alpha,beta);

    fac  = pow(two,apb + 1)*gammaF(alpha + np)*gammaF(beta + np);
    fac /= (np-1)*gammaF(np)*gammaF(alpha + beta + np + one);
    
    for(i = 0; i < np; ++i) w[i] = fac/(w[i]*w[i]);
    w[0]    *= (beta  + one);
    w[np-1] *= (alpha + one);
  }
      
  return;
}


/** 
    \brief Compute the Derivative Matrix and its transpose associated
    with the Gauss-Jacobi zeros.
             
    \li Compute the derivative matrix, \a d, and its transpose, \a dt,
    associated with the n_th order Lagrangian interpolants through the
    \a np Gauss-Jacobi points \a z such that \n
    \f$  \frac{du}{dz}(z[i]) =  \sum_{j=0}^{np-1} D[i*np+j] u(z[j]) \f$

    \li d and dt are both square matrices.
*/

void Dgj(double *D, double *Dt, double *z, int np,double alpha, double beta){

  double one = 1.0, two = 2.0;

  if (np <= 0){
    D[0] = Dt[0] = 0.0;
  }
  else{
    register int i,j; 
    double *pd;
    
    pd = (double *)malloc(np*sizeof(double));
    jacobd(np,z,pd,np,alpha,beta);
    
    for (i = 0; i < np; i++){
      for (j = 0; j < np; j++){

	if (i != j) 
	  D[i*np+j] = pd[i]/(pd[j]*(z[i]-z[j]));
	else    
	  D[i*np+j] = (alpha - beta + (alpha + beta + two)*z[i])/
		     (two*(one - z[i]*z[i]));

	Dt[j*np+i] = D[i*np+j];
      }
    }
    free(pd);
  }
  return;
}


/** 
    \brief Compute the Derivative Matrix and its transpose associated
    with the Gauss-Radau-Jacobi zeros with a zero at \a z=-1.
             
    \li Compute the derivative matrix, \a d, and its transpose, \a dt,
    associated with the n_th order Lagrangian interpolants through the
    \a np Gauss-Radau-Jacobi points \a z such that \n
    \f$  \frac{du}{dz}(z[i]) =  \sum_{j=0}^{np-1} D[i*np+j] u(z[j]) \f$

    \li d and dt are both square matrices.
*/

void Dgrjm(double *D, double *Dt, double *z, int np,
	   double alpha, double beta){

  if (np <= 0){
    D[0] = Dt[0] = 0.0;
  }
  else{
    register int i, j; 
    double   one = 1.0, two = 2.0;
    double   *pd;

    pd  = (double *)malloc(np*sizeof(double));

    pd[0] = pow(-one,np-1)*gammaF(np+beta+one);
    pd[0] /= gammaF(np)*gammaF(beta+two);
    jacobd(np-1,z+1,pd+1,np-1,alpha,beta+1);
    for(i = 1; i < np; ++i) pd[i] *= (1+z[i]);

    for (i = 0; i < np; i++)
      for (j = 0; j < np; j++){
	if (i != j) 
	  D[i*np+j] = pd[i]/(pd[j]*(z[i]-z[j]));
	else { 
	  if(i == 0)
	    D[i*np+j] = -(np + alpha + beta + one)*(np - one)/
	      (two*(beta + two));
	  else
	    D[i*np+j] = (alpha - beta + one + (alpha + beta + one)*z[i])/
	      (two*(one - z[i]*z[i]));
	}
	
	Dt[j*np+i] = D[i*np+j];
      }
    free(pd);
  }

  return;
}


/** 
    \brief Compute the Derivative Matrix and its transpose associated
    with the Gauss-Radau-Jacobi zeros with a zero at \a z=1.
             
    \li Compute the derivative matrix, \a d, and its transpose, \a dt,
    associated with the n_th order Lagrangian interpolants through the
    \a np Gauss-Radau-Jacobi points \a z such that \n
    \f$  \frac{du}{dz}(z[i]) =  \sum_{j=0}^{np-1} D[i*np+j] u(z[j]) \f$

    \li d and dt are both square matrices.
*/

void Dgrjp(double *D, double *Dt, double *z, int np,
	   double alpha, double beta){

  if (np <= 0){
    D[0] = Dt[0] = 0.0;
  }
  else{
    register int i, j; 
    double   one = 1.0, two = 2.0;
    double   *pd;

    pd  = (double *)malloc(np*sizeof(double));


    jacobd(np-1,z,pd,np-1,alpha+1,beta);
    for(i = 0; i < np-1; ++i) pd[i] *= (1-z[i]);
    pd[np-1] = -gammaF(np+alpha+one);
    pd[np-1] /= gammaF(np)*gammaF(alpha+two);

    for (i = 0; i < np; i++)
      for (j = 0; j < np; j++){
	if (i != j) 
	  D[i*np+j] = pd[i]/(pd[j]*(z[i]-z[j]));
	else { 
	  if(i == np-1)
	    D[i*np+j] = (np + alpha + beta + one)*(np - one)/
	      (two*(alpha + two));
	  else
	    D[i*np+j] = (alpha - beta - one + (alpha + beta + one)*z[i])/
	      (two*(one - z[i]*z[i]));
	}
	
	Dt[j*np+i] = D[i*np+j];
      }
    free(pd);
  }

  return;
}

/** 
    \brief Compute the Derivative Matrix and its transpose associated
    with the Gauss-Lobatto-Jacobi zeros.
             
    \li Compute the derivative matrix, \a d, and its transpose, \a dt,
    associated with the n_th order Lagrangian interpolants through the
    \a np Gauss-Lobatto-Jacobi points \a z such that \n
    \f$  \frac{du}{dz}(z[i]) =  \sum_{j=0}^{np-1} D[i*np+j] u(z[j]) \f$

    \li d and dt are both square matrices.
*/

void Dglj(double *D, double *Dt, double *z, int np,
	  double alpha, double beta){
     
  if (np <= 0){
    D[0] = Dt[0] = 0.0;
  }
  else{
    register int i, j; 
    double   one = 1.0, two = 2.0;
    double   *pd;

    pd  = (double *)malloc(np*sizeof(double));

    pd[0]  = two*pow(-one,np)*gammaF(np + beta);
    pd[0] /= gammaF(np - one)*gammaF(beta + two);
    jacobd(np-2,z+1,pd+1,np-2,alpha+1,beta+1);
    for(i = 1; i < np-1; ++i) pd[i] *= (one-z[i]*z[i]);
    pd[np-1]  = -two*gammaF(np + alpha);
    pd[np-1] /= gammaF(np - one)*gammaF(alpha + two);

    for (i = 0; i < np; i++)
      for (j = 0; j < np; j++){
	if (i != j) 
	  D[i*np+j] = pd[i]/(pd[j]*(z[i]-z[j]));
	else { 
	  if      (i == 0)
	    D[i*np+j] = (alpha - (np-1)*(np + alpha + beta))/(two*(beta+ two));
	  else if (i == np-1)
	    D[i*np+j] =-(beta - (np-1)*(np + alpha + beta))/(two*(alpha+ two));
	  else
	    D[i*np+j] = (alpha - beta + (alpha + beta)*z[i])/
	                        (two*(one - z[i]*z[i]));
	}
	
	Dt[j*np+i] = D[i*np+j];
      }
    free(pd);
  }

  return;
}


/** 
    \brief Compute the value of the \a i th Lagrangian interpolant through  
    the \a np Gauss-Jacobi points \a zgj at the arbitrary location \a z.     

    \li \f$ -1 \leq z \leq 1 \f$

    \li Uses the defintion of the Lagrangian interpolant:\n
%
    \f$ \begin{array}{rcl}
    h_j(z) =  \left\{ \begin{array}{ll}
    \displaystyle \frac{P_{np}^{\alpha,\beta}(z)}
    {[P_{np}^{\alpha,\beta}(z_j)]^\prime
    (z-z_j)} & \mbox{if $z \ne z_j$}\\ 
    & \\
    1 & \mbox{if $z=z_j$}
    \end{array}
    \right.
    \end{array}   \f$ 
*/

double hgj (int i, double z, double *zgj, int np, double alpha, double beta)
{

  double zi, dz, p, pd, h;

  zi  = *(zgj+i);
  dz  = z - zi;
  if (fabs(dz) < EPS) return 1.0;

  jacobd (1, &zi, &pd , np, alpha, beta);
  jacobfd(1, &z , &p, NULL , np, alpha, beta);
  h = p/(pd*dz);

  return h;
}

/** 
    \brief Compute the value of the \a i th Lagrangian interpolant through the
    \a np Gauss-Radau-Jacobi points \a zgrj at the arbitrary location
    \a z. This routine assumes \a zgrj includes the point \a -1.

    \li \f$ -1 \leq z \leq 1 \f$

    \li Uses the defintion of the Lagrangian interpolant:\n
%
    \f$ \begin{array}{rcl}
    h_j(z) = \left\{ \begin{array}{ll}
    \displaystyle \frac{(1+z) P_{np-1}^{\alpha,\beta+1}(z)}
    {((1+z_j) [P_{np-1}^{\alpha,\beta+1}(z_j)]^\prime +
    P_{np-1}^{\alpha,\beta+1}(z_j) ) (z-z_j)} & \mbox{if $z \ne z_j$}\\ 
    & \\
    1 & \mbox{if $z=z_j$}
    \end{array}
    \right.
    \end{array}   \f$ 
*/

double hgrjm (int i, double z, double *zgrj, int np, double alpha, double beta)
{

  double zi, dz, p, pd, h;

  zi  = *(zgrj+i);
  dz  = z - zi;
  if (fabs(dz) < EPS) return 1.0;

  jacobfd (1, &zi, &p , NULL, np-1, alpha, beta + 1);
  // need to use this routine in caes zi = -1 or 1
  jacobd  (1, &zi, &pd, np-1, alpha, beta + 1);
  h = (1.0 + zi)*pd + p;
  jacobfd (1, &z, &p, NULL,  np-1, alpha, beta + 1);
  h = (1.0 + z )*p/(h*dz);

  return h;
}


/** 
    \brief Compute the value of the \a i th Lagrangian interpolant through the
    \a np Gauss-Radau-Jacobi points \a zgrj at the arbitrary location
    \a z. This routine assumes \a zgrj includes the point \a +1.

    \li \f$ -1 \leq z \leq 1 \f$

    \li Uses the defintion of the Lagrangian interpolant:\n
%
    \f$ \begin{array}{rcl}
    h_j(z) = \left\{ \begin{array}{ll}
    \displaystyle \frac{(1-z) P_{np-1}^{\alpha+1,\beta}(z)}
    {((1-z_j) [P_{np-1}^{\alpha+1,\beta}(z_j)]^\prime -
    P_{np-1}^{\alpha+1,\beta}(z_j) ) (z-z_j)} & \mbox{if $z \ne z_j$}\\ 
    & \\
    1 & \mbox{if $z=z_j$}
    \end{array}
    \right.
    \end{array}   \f$ 
*/

double hgrjp (int i, double z, double *zgrj, int np, double alpha, double beta)
{

  double zi, dz, p, pd, h;

  zi  = *(zgrj+i);
  dz  = z - zi;
  if (fabs(dz) < EPS) return 1.0;

  jacobfd (1, &zi, &p , NULL, np-1, alpha+1, beta );
  // need to use this routine in caes z = -1 or 1
  jacobd  (1, &zi, &pd, np-1, alpha+1, beta );
  h = (1.0 - zi)*pd - p;
  jacobfd (1, &z, &p, NULL,  np-1, alpha+1, beta);
  h = (1.0 - z )*p/(h*dz);

  return h;
}


/** 
    \brief Compute the value of the \a i th Lagrangian interpolant through the
    \a np Gauss-Lobatto-Jacobi points \a zgrj at the arbitrary location
    \a z. 

    \li \f$ -1 \leq z \leq 1 \f$

    \li Uses the defintion of the Lagrangian interpolant:\n
%
    \f$ \begin{array}{rcl}
    h_j(z) = \left\{ \begin{array}{ll}
    \displaystyle \frac{(1-z^2) P_{np-2}^{\alpha+1,\beta+1}(z)}
    {((1-z^2_j) [P_{np-2}^{\alpha+1,\beta+1}(z_j)]^\prime -
    2 z_j P_{np-2}^{\alpha+1,\beta+1}(z_j) ) (z-z_j)}&\mbox{if $z \ne z_j$}\\ 
    & \\
    1 & \mbox{if $z=z_j$}
    \end{array}
    \right.
    \end{array}   \f$ 
*/

double hglj (int i, double z, double *zglj, int np, double alpha, double beta)
{
  double one = 1., two = 2.;
  double zi, dz, p, pd, h;

  zi  = *(zglj+i);
  dz  = z - zi;
  if (fabs(dz) < EPS) return 1.0;

  jacobfd(1, &zi, &p , NULL, np-2, alpha + one, beta + one);
  // need to use this routine in caes z = -1 or 1
  jacobd (1, &zi, &pd, np-2, alpha + one, beta + one);
  h = (one - zi*zi)*pd - two*zi*p;
  jacobfd(1, &z, &p, NULL, np-2, alpha + one, beta + one);
  h = (one - z*z)*p/(h*dz);

  return h;
}


/** 
    \brief Interpolation Operator from Gauss-Jacobi points to an
    arbitrary distrubtion at points \a zm
                                                                        
    \li Computes the one-dimensional interpolation matrix, \a im, to
    interpolate a function from at Gauss-Jacobi distribution of \a nz
    zeros \a zgrj to an arbitrary distribution of \a mz points \a zm, i.e.\n
    \f$ 
    u(zm[i]) = \sum_{j=0}^{nz-1} im[i*nz+j] \ u(zgj[j]) 
    \f$

*/

void Imgj(double *im,double *zgj, double *zm, int nz, int mz,
	  double alpha, double beta){
  double zp;
  register int i, j;

  for (i = 0; i < mz; ++i) {
    zp = zm[i];
    for (j = 0; j < nz; ++j)
      im [i*nz+j] = hgj(j, zp, zgj, nz, alpha, beta);
  }
  
  return;
}

/** 
    \brief Interpolation Operator from Gauss-Radau-Jacobi points
    (including \a z=-1) to an arbitrary distrubtion at points \a zm
                                                                        
    \li Computes the one-dimensional interpolation matrix, \a im, to
    interpolate a function from at Gauss-Radau-Jacobi distribution of
    \a nz zeros \a zgrj (where \a zgrj[0]=-1) to an arbitrary
    distribution of \a mz points \a zm, i.e.
    \n 
    \f$ u(zm[i]) =    \sum_{j=0}^{nz-1} im[i*nz+j] \ u(zgj[j]) \f$

*/

void Imgrjm(double *im,double *zgrj, double *zm, int nz, int mz,
	   double alpha, double beta){
  double zp;
  register int i, j;

  for (i = 0; i < mz; i++) {
    zp = zm[i];
    for (j = 0; j < nz; j++)
      im [i*nz+j] = hgrjm(j, zp, zgrj, nz, alpha, beta);
  }
  
  return;
}

/** 
    \brief Interpolation Operator from Gauss-Radau-Jacobi points
    (including \a z=1) to an arbitrary distrubtion at points \a zm
                                                                        
    \li Computes the one-dimensional interpolation matrix, \a im, to
    interpolate a function from at Gauss-Radau-Jacobi distribution of
    \a nz zeros \a zgrj (where \a zgrj[nz-1]=1) to an arbitrary
    distribution of \a mz points \a zm, i.e.
    \n 
    \f$ u(zm[i]) =    \sum_{j=0}^{nz-1} im[i*nz+j] \ u(zgj[j]) \f$

*/

void Imgrjp(double *im,double *zgrj, double *zm, int nz, int mz,
	   double alpha, double beta){
  double zp;
  register int i, j;

  for (i = 0; i < mz; i++) {
    zp = zm[i];
    for (j = 0; j < nz; j++)
      im [i*nz+j] = hgrjp(j, zp, zgrj, nz, alpha, beta);
  }
  
  return;
}


/** 
    \brief Interpolation Operator from Gauss-Lobatto-Jacobi points
    to an arbitrary distrubtion at points \a zm
                                                                        
    \li Computes the one-dimensional interpolation matrix, \a im, to
    interpolate a function from at Gauss-Lobatto-Jacobi distribution of
    \a nz zeros \a zgrj (where \a zgrj[0]=-1) to an arbitrary
    distribution of \a mz points \a zm, i.e.
    \n 
    \f$ u(zm[i]) =    \sum_{j=0}^{nz-1} im[i*nz+j] \ u(zgj[j]) \f$

*/

void Imglj(double *im, double *zglj, double *zm, int nz, int mz,
	   double alpha, double beta)
{
  double zp;
  register int i, j;
  
  for (i = 0; i < mz; i++) {
    zp = zm[i];
    for (j = 0; j < nz; j++)
      im[i*nz+j] = hglj(j, zp, zglj, nz, alpha, beta);
  }
  
  return;
}

/** 
    \brief Routine to calculate Jacobi polynomials, \f$
    P^{\alpha,\beta}_n(z) \f$, and their first derivative, \f$
    \frac{d}{dz} P^{\alpha,\beta}_n(z) \f$.
   
    \li This function returns the vectors \a poly_in and \a poly_d
    containing the value of the \f$ n^th \f$ order Jacobi polynomial
    \f$ P^{\alpha,\beta}_n(z) \alpha > -1, \beta > -1 \f$ and its
    derivative at the \a np points in \a z[i]
    
    - If \a poly_in = NULL then only calculate derivatice

    - If \a polyd   = NULL then only calculate polynomial

    - To calculate the polynomial this routine uses the recursion
    relationship (see appendix A ref [4]) :
    \f$ \begin{array}{rcl}
    P^{\alpha,\beta}_0(z) &=& 1 \\
    P^{\alpha,\beta}_1(z) &=& \frac{1}{2} [ \alpha-\beta+(\alpha+\beta+2)z] \\
    a^1_n P^{\alpha,\beta}_{n+1}(z) &=& (a^2_n + a^3_n z) 
    P^{\alpha,\beta}_n(z) - a^4_n P^{\alpha,\beta}_{n-1}(z) \\
    a^1_n &=& 2(n+1)(n+\alpha + \beta + 1)(2n + \alpha + \beta) \\
    a^2_n &=& (2n + \alpha + \beta + 1)(\alpha^2 - \beta^2)  \\
    a^3_n &=& (2n + \alpha + \beta)(2n + \alpha + \beta + 1)
    (2n + \alpha + \beta + 2)  \\
    a^4_n &=& 2(n+\alpha)(n+\beta)(2n + \alpha + \beta + 2)
    \end{array} \f$
    
    - To calculate the derivative of the polynomial this routine uses
    the relationship (see appendix A ref [4]) :
    \f$ \begin{array}{rcl}
    b^1_n(z)\frac{d}{dz} P^{\alpha,\beta}_n(z)&=&b^2_n(z)P^{\alpha,\beta}_n(z)
    + b^3_n(z) P^{\alpha,\beta}_{n-1}(z) \hspace{2.2cm} \\
    b^1_n(z) &=& (2n+\alpha + \beta)(1-z^2) \\
    b^2_n(z) &=& n[\alpha - \beta - (2n+\alpha + \beta)z]\\
    b^3_n(z) &=& 2(n+\alpha)(n+\beta) 
    \end{array} \f$

    - Note the derivative from this routine is only valid for -1 < \a z < 1.
*/
void jacobfd(int np, double *z, double *poly_in, double *polyd, int n, 
	     double alpha, double beta){
  register int i;
  double  zero = 0.0, one = 1.0, two = 2.0;

  if(!np)
    return;

  if(n == 0){
    if(poly_in)
      for(i = 0; i < np; ++i) 
	poly_in[i] = one;
    if(polyd)
      for(i = 0; i < np; ++i) 
	polyd[i] = zero; 
  }
  else if (n == 1){
    if(poly_in)
      for(i = 0; i < np; ++i) 
	poly_in[i] = 0.5*(alpha - beta + (alpha + beta + two)*z[i]);
    if(polyd)
      for(i = 0; i < np; ++i) 
	polyd[i] = 0.5*(alpha + beta + two);
  }
  else{
    register int k;
    double   a1,a2,a3,a4;
    double   two = 2.0, apb = alpha + beta;
    double   *poly, *polyn1,*polyn2;
    
    if(poly_in){ // switch for case of no poynomial function return
      polyn1 = (double *)malloc(2*np*sizeof(double));
      polyn2 = polyn1+np; 
      poly   = poly_in;
    }
    else{
      polyn1 = (double *)malloc(3*np*sizeof(double));
      polyn2 = polyn1+np; 
      poly   = polyn2+np;      
    }

    for(i = 0; i < np; ++i){
      polyn2[i] = one;
      polyn1[i] = 0.5*(alpha - beta + (alpha + beta + two)*z[i]);
    }
    
    for(k = 2; k <= n; ++k){
      a1 =  two*k*(k + apb)*(two*k + apb - two);
      a2 = (two*k + apb - one)*(alpha*alpha - beta*beta);
      a3 = (two*k + apb - two)*(two*k + apb - one)*(two*k + apb);
      a4 =  two*(k + alpha - one)*(k + beta - one)*(two*k + apb);
      
      a2 /= a1;
      a3 /= a1;
      a4 /= a1;
	
      for(i = 0; i < np; ++i){
	poly  [i] = (a2 + a3*z[i])*polyn1[i] - a4*polyn2[i];
	polyn2[i] = polyn1[i];
	polyn1[i] = poly  [i];
      }
    }
    
    if(polyd){
      a1 = n*(alpha - beta);
      a2 = n*(two*n + alpha + beta);
      a3 = two*(n + alpha)*(n + beta);
      a4 = (two*n + alpha + beta);
      a1 /= a4;  a2 /= a4;   a3 /= a4;

      // note polyn2 points to polyn1 at end of poly iterations
      for(i = 0; i < np; ++i){
	polyd[i]  = (a1- a2*z[i])*poly[i] + a3*polyn2[i];
	polyd[i] /= (one - z[i]*z[i]);
      }
    }
    
    free(polyn1);
  }
  
  return;
}


/**
   \brief Calculate the  derivative of Jacobi polynomials 
  
   \li Generates a vector \a poly of values of the derivative of the
  \a n th order Jacobi polynomial \f$ P^(\alpha,\beta)_n(z)\f$ at the
  \a np points \a z.

  \li To do this we have used the relation 
  \n
  \f$ \frac{d}{dz} P^{\alpha,\beta}_n(z) 
  = \frac{1}{2} (\alpha + \beta + n + 1)  P^{\alpha,\beta}_n(z) \f$

  \li This formulation is valid for \f$ -1 \leq z \leq 1 \f$ 

*/

void jacobd(int np, double *z, double *polyd, int n, double alpha, double beta)
{
  register int i;
  double one = 1.0;
  if(n == 0)
    for(i = 0; i < np; ++i) polyd[i] = 0.0;
  else{
    //jacobf(np,z,polyd,n-1,alpha+one,beta+one);
    jacobfd(np,z,polyd,NULL,n-1,alpha+one,beta+one);
    for(i = 0; i < np; ++i) polyd[i] *= 0.5*(alpha + beta + (double)n + one);
  }
  return;
}


/** 
 \brief Calculate the Gamma function , \f$ \Gamma(n)\f$, for integer
 values and halves.

 Determine the value of \f$\Gamma(n)\f$ using:

 \f$ \Gamma(n) = (n-1)!  \mbox{ or  }  \Gamma(n+1/2) = (n-1/2)\Gamma(n-1/2)\f$

 where \f$ \Gamma(1/2) = \sqrt(\pi)\f$
 */

static double gammaF(double x){
  double gamma = 1.0;
  
  if     (x == -0.5) gamma = -2.0*sqrt(M_PI);
  else if (!x) return gamma;
  else if ((x-(int)x) == 0.5){ 
    int n = (int) x;
    double tmp = x;

    gamma = sqrt(M_PI);
    while(n--){
      tmp   -= 1.0;
      gamma *= tmp;
    }
  }
  else if ((x-(int)x) == 0.0){
    int n = (int) x;
    double tmp = x;

    while(--n){
      tmp   -= 1.0;
      gamma *= tmp;
    }
  }  
  else
    fprintf(stderr,"%lf is not of integer or half order\n",x);
  return gamma;
}
    
/** 
    \brief  Calculate the \a n zeros, \a z, of the Jacobi polynomial, i.e.
    \f$ P_n^{\alpha,\beta}(z) = 0 \f$
                                                             
    This routine is only value for \f$( \alpha > -1, \beta > -1)\f$
    and uses polynomial deflation in a Newton iteration 
*/

static void Jacobz(int n, double *z, double alpha, double beta){
  register int i,j,k;
  double   dth = M_PI/(2.0*(double)n);
  double   poly,pder,rlast=0.0;
  double   sum,delr,r;
  double one = 1.0, two = 2.0;
  
  if(!n)
    return;
  
  for(k = 0; k < n; ++k){
    r = -cos((two*(double)k + one) * dth);
    if(k) r = 0.5*(r + rlast);
    
    for(j = 1; j < STOP; ++j){
      jacobfd(1,&r,&poly, &pder, n, alpha, beta);
      
      for(i = 0, sum = 0.0; i < k; ++i) sum += one/(r - z[i]);
      
      delr = -poly / (pder - sum * poly);
      r   += delr;
      if( fabs(delr) < EPS ) break;
    }
    z[k]  = r;
    rlast = r;
  }
  return;
}


/**
   \brief Zero determination through the eigenvalues of a tridiagonal
   matrix from teh three term recursion relationship.
   
   Set up a symmetric tridiagonal matrix

   \f$ \left [  \begin{array}{ccccc}
   a[0] & b[0]   &        &        & \\
   b[0] & a[1]   & b[1]   &        & \\
    0   & \ddots & \ddots & \ddots &  \\
        &        & \ddots & \ddots & b[n-2] \\
        &        &        & b[n-2] & a[n-1] \end{array} \right ] \f$

   Where the coefficients a[n], b[n] come from the  recurrence relation
   
   \f$  b_j p_j(z) = (z - a_j ) p_{j-1}(z) - b_{j-1}   p_{j-2}(z) \f$
   
   where \f$ j=n+1\f$ and \f$p_j(z)\f$ are the Jacobi (normalized)
   orthogonal polynomials \f$ \alpha,\beta > -1\f$( integer values and
   halves). Since the polynomials are orthonormalized, the tridiagonal
   matrix is guaranteed to be symmetric. The eigenvalues of this
   matrix are the zeros of the Jacobi polynomial.
*/

static void JacZeros(int n, double *a, double alpha, double beta){
  int i;
  double apb, apbi,a2b2;
  double *b;
  
  if(!n)
    return;

  b = (double *) malloc(n*sizeof(double));
  
  // generate normalised terms 
  apb  = alpha + beta;
  apbi = 2.0 + apb;

  b[n-1] = pow(2.0,apb+1.0)*gammaF(alpha+1.0)*gammaF(beta+1.0)/gammaF(apbi);
  a[0]   = (beta-alpha)/apbi;
  b[0]   = sqrt(4.0*(1.0+alpha)*(1.0+beta)/((apbi+1.0)*apbi*apbi));

  a2b2 = beta*beta-alpha*alpha;
  for(i = 1; i < n-1; ++i){
    apbi = 2.0*(i+1) + apb;
    a[i] = a2b2/((apbi-2.0)*apbi);
    b[i] = sqrt(4.0*(i+1)*(i+1+alpha)*(i+1+beta)*(i+1+apb)/
		((apbi*apbi-1)*apbi*apbi));
  }

  apbi   = 2.0*n + apb;
  a[n-1] = a2b2/((apbi-2.0)*apbi);
  
  // find eigenvalues 
  TriQL(n, a, b);

  free(b);
  return;
}


/** \brief QL algorithm for symmetric tridiagonal matrix 

    This subroutine is a translation of an algol procedure,
    num. math. \b 12, 377-383(1968) by martin and wilkinson, as modified
    in num. math. \b 15, 450(1970) by dubrulle.  Handbook for
    auto. comp., vol.ii-linear algebra, 241-248(1971).  This is a
    modified version from numerical recipes.

    This subroutine finds the eigenvalues and first components of the
    eigenvectors of a symmetric tridiagonal matrix by the implicit QL
    method.

    on input:
    - n is the order of the matrix;
    - d contains the diagonal elements of the input matrix;
    - e contains the subdiagonal elements of the input matrix
    in its first n-1 positions. e(n) is arbitrary;

    on output:

    - d contains the eigenvalues in ascending order.  
    - e has been destroyed;
*/

static void TriQL(int n, double *d,double *e){
  int m,l,iter,i,k;
  double s,r,p,g,f,dd,c,b;
  
  for (l=0;l<n;l++) {
    iter=0;
    do {
      for (m=l;m<n-1;m++) {
	dd=fabs(d[m])+fabs(d[m+1]);
	if (fabs(e[m])+dd == dd) break;
      }
      if (m != l) {
	if (iter++ == STOP){
	  fprintf(stderr,"triQL: Too many iterations in TQLI");
	  exit(1);
	}
	g=(d[l+1]-d[l])/(2.0*e[l]);
	r=sqrt((g*g)+1.0);
	g=d[m]-d[l]+e[l]/(g+sign(r,g));
	s=c=1.0;
	p=0.0;
	for (i=m-1;i>=l;i--) {
	  f=s*e[i];
	  b=c*e[i];
	  if (fabs(f) >= fabs(g)) {
	    c=g/f;
	    r=sqrt((c*c)+1.0);
	    e[i+1]=f*r;
	    c *= (s=1.0/r);
	  } else {
	    s=f/g;
	    r=sqrt((s*s)+1.0);
	    e[i+1]=g*r;
	    s *= (c=1.0/r);
	  }
	  g=d[i+1]-p;
	  r=(d[i]-g)*s+2.0*c*b;
	  p=s*r;
	  d[i+1]=g+p;
	  g=c*r-b;
	}
	d[l]=d[l]-p;
	e[l]=g;
	e[m]=0.0;
      }
    } while (m != l);
  }

  // order eigenvalues
  for(i = 0; i < n-1; ++i){ 
    k = i;
    p = d[i];
    for(l = i+1; l < n; ++l)
      if (d[l] < p) {
	k = l;
	p = d[l];
      }
    d[k] = d[i]; 
    d[i] = p;
  }
}



#ifdef __cplusplus
} // end of namespace
#endif
