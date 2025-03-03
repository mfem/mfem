// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//    ---------------------------------------------------------------------
//    Mesh Optimizer NLP Miniapp: Optimize high-order meshes - Parallel Version
//    ---------------------------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., it used non-linear programming techniques
// to solve the proble,
//
// Compile with: make pmesh-optimizer_NLP
// mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-4 -ni 100 -ft 2 -w1 5e1 -w2 1e-2
// mpirun -np 10 pmesh-optimizer_NLP -met 1 -ch 2e-3 -ni 200 -ft 2 --qtype 3 -w1 5e3 -w2 1e-2
// WIP mpirun -np 10 pmesh-optimizer_NLP -met 1 -ch 2e-3 -ni 200 -ft 2 --qtype 4 -w1 1e-4 -w2 1e-2


// K10 -  TMOP solver based run
// order 2, shock wave around origin
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 500 -ft 2 --qtype 4 -w1 5e-2 -w2 5e-2 -m square01.mesh -rs 2 -o 2 -lsn 1.05 -lse 1.05
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 500 -ft 2 --qtype 4 -w1 1e-1 -w2 5 -m square01-tri.mesh -rs 1 -alpha 20 -o 2 -mid 2 -tid 4
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 400 -ft 2 --qtype 3 -w1 2e3 -w2 30 -m square01-tri.mesh -rs 1 -alpha 20 -o 2 -mid 2 -tid 4
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-4 -ni 200 -ft 2 -w1 1e1 -w2 0.5 -qt 1 -rs 3 -m square01.mesh -lsn 1.01 -o 1

// order 1, cube mesh
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 100 -ft 2 --qtype 3 -w1 5e3 -w2 1e-2 -m cube.mesh -o 1 -rs 4 -mid 303

// sinusoidal wave for orientation and sharp inclined wave for solution
// working with energy
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 200 -ft 3 --qtype 4 -w1 5e-2 -w2 2e-2 -m square01.mesh -rs 2 -alpha 20 -o 2 -mid 107 -tid 5

// sinusoidal wave for orientation and gradient in solution
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 200 -ft 4 --qtype 4 -w1 1e-2 -w2 2e-2 -m square01.mesh -rs 2 -alpha 50 -o 2 -mid 107 -tid 5
// Long run (3rd order):
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 2000 -ft 4 --qtype 4 -w1 8e-3 -w2 2e-2 -m square01.mesh -rs 2 -alpha 50 -o 3 -mid 107 -tid 5

// L-shaped domain.


// l2 error
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-4 -ni 1000 -ft 1 --qtype 0 -w1 2e4 -w2 1e-1 -m square01.mesh -rs 2 -o 1 -lsn 1.01 -lse 1.01 -alpha 10 -bndrfree
// h1 error
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 1000 -ft 1 --qtype 1 -w1 2e2 -w2 8e-1 -m square01.mesh -rs 2 -o 1 -lsn 1.01 -lse 1.01 -alpha 10 -bndrfree

// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 1000 -w1 1e5 -w2 1e-2 -rs 3 -o 2 -lsn 1.01 -lse 1.01 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad 0.01


/*******************************/
// Presentation runs below:

// zz 2nd order - shock wave around corner - with filter
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 200 -w1 1e5 -w2 1e-2 -rs 2 -o 2 -lsn 1.01 -lse 1.01 -alpha 20 -bndrfree -qt 5 -ft 2 -vis -weakbc -filter -frad 0.01
// average error - 2nd order - shock wave around corner
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e3 -w2 1e-2 -rs 2 -o 2 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad 0.005
// same but with simplices
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 300 -w1 1e3 -w2 1e-2 -rs 1 -o 2 -lsn 2.01 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 2 -vis -weakbc -filter -frad 0.005 -m square01-tri.mesh
// l2 with wave around center - linear
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-4 -ni 400 -ft 1 --qtype 0 -w1 2e4 -w2 1e-1 -m square01.mesh -rs 2 -o 1 -lsn 1.01 -lse 1.01 -alpha 10 -bndrfree
// h1 with wave around center - linear
// make pmesh-optimizer_NLP -j && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 200 -ft 1 --qtype 1 -w1 2e2 -w2 15e-1 -m uare01.mesh -rs 2 -o 1 -lsn 1.01 -lse 1.01 -alpha 10 -bndrfree

// inclined wave with avg error
//  make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 1000 -w1 1e3 -w2 1e-2 -rs 2 -o 2 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 3 -vis -weakbc -filter -frad 0.005

// zz for wave around center
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 1e-3 -ni 500 -w1 1e5 -w2 1e-2 -rs 2 -o 2 -lsn 1.01 -lse 1.01 -alpha 20 -bndrfree -qt 5 -ft 1 -vis -weakbc -filter -frad 0.05

// avg error - inclined wave + analytic orientation
// make pmesh-optimizer_NLP -j4 && mpirun -np 10 pmesh-optimizer_NLP -met 0 -ch 2e-3 -ni 500 -w1 2e3 -w2 5e-1 -rs 2 -o 2 -lsn 2.0 -lse 1.01 -alpha 20 -bndrfree -qt 3 -ft 3 -vis -weakbc -filter -frad 0.005 -mid 107 -tid 5

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include "linalg/dual.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer_using_NLP.hpp"
#include "MMA.hpp"
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

/// MFEM native AD-type for first derivatives
typedef internal::dual<real_t, real_t> ADFType;
/// MFEM native AD-type for second derivatives
typedef internal::dual<ADFType, ADFType> ADSType;
/// MFEM native AD-type for third derivatives
typedef internal::dual<ADSType, ADSType> ADTType;

real_t ADVal_func( const Vector &x, std::function<ADFType( std::vector<ADFType>&)> func)
{
  int dim = x.Size();
  int matsize = dim;
   std::vector<ADFType> adinp(matsize);
   for (int i=0; i<matsize; i++) { adinp[i] = ADFType{x[i], 0.0}; }

   return func(adinp).value;
}


void ADGrad_func( const Vector &x, std::function<ADFType( std::vector<ADFType>&)> func, Vector &grad)
{
   int dim = x.Size();

   std::vector<ADFType> adinp(dim);

   for (int i=0; i<dim; i++) { adinp[i] = ADFType{x[i], 0.0}; }

   for (int i=0; i<dim; i++)
   {
      adinp[i] = ADFType{x[i], 1.0};
      ADFType rez = func(adinp);
      grad[i] = rez.gradient;
      adinp[i] = ADFType{x[i], 0.0};
   }
}

void ADHessian_func(const Vector &x, std::function<ADSType( std::vector<ADSType>&)> func, DenseMatrix &H)
{
   int dim = x.Size();

   //use forward-forward mode
   std::vector<ADSType> aduu(dim);
   for (int ii = 0; ii < dim; ii++)
   {
      aduu[ii].value = ADFType{x[ii], 0.0};
      aduu[ii].gradient = ADFType{0.0, 0.0};
   }

   for (int ii = 0; ii < dim; ii++)
   {
      aduu[ii].value = ADFType{x[ii], 1.0};
      for (int jj = 0; jj < (ii + 1); jj++)
      {
         aduu[jj].gradient = ADFType{1.0, 0.0};
         ADSType rez = func(aduu);
         H(ii,jj) = rez.gradient.gradient;
         H(jj,ii) = rez.gradient.gradient;
         aduu[jj].gradient = ADFType{0.0, 0.0};
      }
      aduu[ii].value = ADFType{x[ii], 0.0};
   }
   return;
}

void AD3rdDeric_func(const Vector &x, std::function<ADTType( std::vector<ADTType>&)> func, std::vector<DenseMatrix> &TRD)
{
   int dim = x.Size();

   //use forward-forward mode
   std::vector<ADTType> aduu(dim);
   for (int ii = 0; ii < dim; ii++)
   {
      aduu[ii].value.value = ADFType{x[ii], 0.0};
      aduu[ii].value.gradient = ADFType{0.0, 0.0};
      aduu[ii].gradient.value = ADFType{0.0, 0.0};
      aduu[ii].gradient.gradient = ADFType{0.0, 0.0};
   }

   for (int ii = 0; ii < dim; ii++)
   {
      aduu[ii].value.value = ADFType{x[ii], 1.0};
      for (int jj = 0; jj < dim; jj++)
      {
         aduu[jj].value.gradient = ADFType{1.0, 0.0};
         for (int kk = 0; kk < dim; kk++)                    // FIXME is ymmetric, only loop over half the possibilites
         {
            aduu[kk].gradient.value = ADFType{1.0, 0.0};
            //aduu[kk].gradient.gradient = ADFType{1.0, 0.0};
            ADTType rez = func(aduu);
            TRD[ii](jj,kk) = rez.gradient.gradient.gradient;
            aduu[kk].gradient.value = ADFType{0.0, 0.0};
            //aduu[kk].gradient.gradient = ADFType{0.0, 0.0};
         }
         aduu[jj].value.gradient = ADFType{0.0, 0.0};
      }
      aduu[ii].value.value = ADFType{x[ii], 0.0};
   }
   return;
}

int ftype = 1;
double kw = 10.0;
double alphaw = 50;

template <typename type>
auto func_0( std::vector<type>& x ) -> type
{
   return sin( 1.0*M_PI *x[0] )*sin(2.0*M_PI*x[1]);;
};

template <typename type>
auto func_1( std::vector<type>& x ) -> type
{
    double k_w = kw;
    double k_t = 0.5;
    double T_ref = 1.0;

    return 0.5+0.5*tanh(k_w*  ((sin( M_PI *x[0] )*sin(M_PI *x[1]))-k_t*T_ref)   );
};

template <typename type>
auto func_8( std::vector<type>& x ) -> type
{
   return sin( M_PI *x[0] );
};

template <typename type>
auto func_2( std::vector<type>& x ) -> type
{
    double theta = 0.0;
    auto xv = 1.0-x[0];
    auto yv = 1.0-x[1];
    double xc = -0.05,
           yc = -0.05,
           zc = -0.05,
           rc = 0.7,
           alpha = alphaw;
    auto dx = xv-xc,
         dy = yv-yc;
    auto val = dx*dx + dy*dy;
    if (val > 0.0) { val = sqrt(val); }
    val -= rc;
    val = alpha*val;
    return 5.0+atan(val);
};

template <typename type>
auto func_3( std::vector<type>& x ) -> type
{
    auto xv = x[0];
    auto yv = x[1];
    auto alpha = alphaw;
    auto dx = xv - 0.5-0.2*(yv-0.5);
    // auto dx = xv-0.5;
    // auto dx = xv - 0.5-0.2*(yv-0.5);
    return atan(alpha*dx);
}

double tanh_left_right_walls(const Vector &x)
{
  double xv = x(0);
  double yv = x(1);
  double beta = 20.0;
  double betay = 50.0;
  double yscale = 0.5*(std::tanh(betay*(yv-.2))-std::tanh(betay*(yv-0.8)));
  double xscale = 0.5*(std::tanh(beta*(xv-.2))-std::tanh(beta*(xv-0.8)));
  return xscale;
}

class OSCoefficient : public TMOPMatrixCoefficient
{
private:
   int metric, dd;

public:
   OSCoefficient(int dim, int metric_id)
      : TMOPMatrixCoefficient(dim), dd(dim), metric(metric_id) { }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector pos(dd);
      T.Transform(ip, pos);
      MFEM_VERIFY(dd == 2,"OSCoefficient does not support 3D\n");
      const real_t xc = pos(0), yc = pos(1);
      real_t theta = M_PI * yc * (1.0 - yc) * cos(2 * M_PI * xc);
      // real_t alpha_bar = 0.1;
      K(0, 0) =  cos(theta);
      K(1, 0) =  sin(theta);
      K(0, 1) = -sin(theta);
      K(1, 1) =  cos(theta);
      // K *= alpha_bar;
   }

    void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                 const IntegrationPoint &ip, int comp) override
   {
      Vector pos(dd);
      T.Transform(ip, pos);
      K = 0.;
   }
};

double trueSolFunc(const Vector & x)
{
  if (ftype == 0)
  {
    return ADVal_func(x, func_0<ADFType>);
  }
  else if (ftype == 1) // circular wave centered in domain
  {
      return ADVal_func(x, func_1<ADFType>);
  }
  else if (ftype == 2) // circular shock wave front centered at origin
  {
    return ADVal_func(x, func_2<ADFType>);
  }
  else if (ftype == 3) // incline shock
  {
    return ADVal_func(x, func_3<ADFType>);
  }
  else if (ftype == 4)
  {
    double xv = x[0], yv = x[1];
    double yc = yv-0.5;
    double delta = 0.1;
    return std::atan(alphaw*(yv - 0.5 - delta*sin(2*M_PI*xv)));
  }
  else if (ftype == 5)
  {
    real_t xv = x[0];
    real_t yv = x[1];
    real_t r = sqrt(xv*xv + yv*yv);
    real_t alpha = 2./3.;
    real_t phi = atan2(yv,xv);
    if (phi < 0) { phi += 2*M_PI; }
    return pow(r,alpha) * sin(alpha * phi);
  }
  else if (ftype == 6)
  {
    double xv = x[0];
    double yv = x[1];
    double alpha = alphaw;
    double dx = xv - 0.48;
    return std::atan(alpha*dx);
  }
  else if (ftype == 7) // circular wave centered in domain
  {
    return x[0]*x[0];
  }
  else if (ftype == 8)
  {
    double val = std::sin( M_PI *x[0] );
    return val;
  }
  return 0.0;
};

void trueSolGradFunc(const Vector & x,Vector & grad)
{
  if (ftype == 0)
  {
    ADGrad_func(x, func_0<ADFType>, grad);
  }
  else if (ftype == 1) // circular wave centered in domain
  {
    ADGrad_func(x, func_1<ADFType>, grad);
  }
  else if (ftype == 2) // circular shock wave front centered at origin
  {
    ADGrad_func(x, func_2<ADFType>, grad);
  }
  else if (ftype == 3)
  {
    ADGrad_func(x, func_3<ADFType>, grad);
  }
  else if (ftype == 4)
  {
    double xv = x[0], yv = x[1];
    double delta = 0.1;
    double phi = alphaw*(yv-0.5-delta*std::sin(2*M_PI*xv));
    double den = 1.0 + phi*phi;
    grad[0] = -2.0*M_PI*alphaw*delta*std::cos(2*M_PI*xv)/den;
    grad[1] = alphaw/den;
  }
  else if (ftype == 5)
  {
    real_t xv = x[0];
    real_t yv = x[1];
    real_t r = sqrt(xv*xv + yv*yv);
    real_t alpha = 2./3.;
    real_t phi = atan2(yv,xv);
    if (phi < 0) { phi += 2*M_PI; }

    real_t r_x = xv/r;
    real_t r_y = yv/r;
    real_t phi_x = - yv / (r*r);
    real_t phi_y = xv / (r*r);
    real_t beta = alpha * pow(r,alpha - 1.);
    grad[0] = beta*(r_x * sin(alpha*phi) + r * phi_x * cos(alpha*phi));
    grad[1] = beta*(r_y * sin(alpha*phi) + r * phi_y * cos(alpha*phi));
  }
  else if (ftype == 6)
  {
    double xv = x[0];
    double yv = x[1];
    double alpha = alphaw;
    double dx = xv - 0.48;
    grad[0] = alpha/(1.0+std::pow(dx*alpha,2.0));
    grad[1] = 0.0;
  }
  else if (ftype == 7)
  {
    grad[0] = 2.0*x[0];
    grad[1] = 0.0;
  }
  else if (ftype == 8)
  {
    grad[0] = M_PI*std::cos( M_PI *x[0] );
    grad[1] = 0.0;
  }
};

double loadFunc(const Vector & x)
{
  if (ftype == 0)
  {
    DenseMatrix Hessian(x.Size());
    ADHessian_func(x, func_0<ADSType>, Hessian);
    double val = -1.0*(Hessian(0,0)+Hessian(1,1));
    return val;
  }
  else if (ftype == 1)
  {
    DenseMatrix Hessian(x.Size());
    ADHessian_func(x, func_1<ADSType>, Hessian);
    double val = -1.0*(Hessian(0,0)+Hessian(1,1));
    return val;
  }
  else if (ftype == 2)
  {
    DenseMatrix Hessian(x.Size());
    ADHessian_func(x, func_2<ADSType>, Hessian);
    double val = -1.0*(Hessian(0,0)+Hessian(1,1));
    return val;
  }
  else if (ftype == 3)
  {
    DenseMatrix Hessian(x.Size());
    ADHessian_func(x, func_3<ADSType>, Hessian);
    double val = -1.0*(Hessian(0,0)+Hessian(1,1));
    return val;
  }
  else if (ftype == 4)
  {
    double xv = x[0], yv = x[1];
    double delta = 0.1;
    double phi = alphaw*(yv-0.5-delta*std::sin(2*M_PI*xv));
    double den = 1.0 + phi*phi;
    double phi_x = -2.0*M_PI*alphaw*delta*std::cos(2*M_PI*xv);
    double term1 = (2*phi/(den*den))*(phi_x*phi_x+alphaw*alphaw);
    double term2 = 4*M_PI*M_PI*alphaw*delta*std::sin(2*M_PI*xv)/den;
    return term1-term2;
  }
  else if (ftype == 5)
  {
    return 0.0;
  }
  else if (ftype == 6)
  {
    double xv = x[0];
    double yv = x[1];
    double alpha = alphaw;
    double dx = xv - 0.48;
    double num1 = std::pow(alpha,3.0)*dx;
    double den1 = std::pow((1.0+std::pow(dx*alpha,2.0)),2.0);
    return 2.0*num1/den1;
  }
  else if (ftype == 7)
  {
    return -2.0;
  }
  else if (ftype == 8)
  {
    double val = M_PI*M_PI * std::sin( M_PI *x[0] );
    return val;
  }
  else if (ftype == 9)
  {
    double val = 0.0;
    if(x[0]>0.99)
    {
      val = 1.0;
    }
    return val;
  }
  return 0.0;
};

void trueLoadFuncGrad(const Vector & x,Vector & grad)
{
  if (ftype == 0)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_0<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[0](1,1));
    grad[1] = -1.0*(TRD[1](0,0)+TRD[1](1,1));
  }
  else if (ftype == 1)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_1<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[1](1,0));
    grad[1] = -1.0*(TRD[0](0,1)+TRD[1](1,1));
  }
  else if (ftype == 2)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_2<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[0](1,1));
    grad[1] = -1.0*(TRD[1](0,0)+TRD[1](1,1));
  }
  else if (ftype == 3)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_3<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[0](1,1));
    grad[1] = -1.0*(TRD[1](0,0)+TRD[1](1,1));
  }
  else if (ftype == 8)
  {
    std::vector<DenseMatrix> TRD(x.Size());
    for(int i=0; i<x.Size();i++){TRD[i].SetSize(x.Size());}
    AD3rdDeric_func(x, func_8<ADTType>, TRD);
    grad[0] = -1.0*(TRD[0](0,0)+TRD[1](1,0));
    grad[1] = -1.0*(TRD[0](0,1)+TRD[1](1,1));

    // grad[0] = M_PI*M_PI*M_PI*std::cos( M_PI *x[0] );
    // grad[1] = 0.0;
  }
}

void trueHessianFunc(const Vector & x,DenseMatrix & Hessian)
{
  Hessian.SetSize(x.Size());
  if (ftype == 0)
  {
    ADHessian_func(x, func_0<ADSType>, Hessian);
  }
  else if (ftype == 1)
  {
    ADHessian_func(x, func_1<ADSType>, Hessian);
  }
  else if (ftype == 2)
  {
    ADHessian_func(x, func_2<ADSType>, Hessian);
  }
  else if (ftype == 3)
  {
    ADHessian_func(x, func_3<ADSType>, Hessian);
  }
  else
  {
    MFEM_ABORT("Not implemented\n");
  }
}

void trueHessianFunc_v(const Vector & x,Vector & Hessian)
{
  DenseMatrix HessianM;
  trueHessianFunc(x,HessianM);
  Hessian.SetSize(HessianM.Height()*HessianM.Width());
  Hessian = HessianM.GetData();
}

void VisVectorField(OSCoefficient *adapt_coeff, ParMesh *pmesh, ParGridFunction *orifield)
{
  ParFiniteElementSpace *pfespace = orifield->ParFESpace();
  int dim = pfespace->GetMesh()->Dimension();

    DenseMatrix mat(dim);
    Vector vec(dim);
    Array<int> dofs;
  // Loop over the elements and project the adapt_coeff to vector field
  for (int e = 0; e < pmesh->GetNE(); e++)
  {
    const FiniteElement *fe = pfespace->GetFE(e);
    const IntegrationRule ir = fe->GetNodes();
    const int dof = fe->GetDof();
    ElementTransformation *trans = pmesh->GetElementTransformation(e);
    Vector nodevals(dof*dim);
    for (int q = 0; q < ir.GetNPoints(); q++)
    {
      const IntegrationPoint &ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);
      adapt_coeff->Eval(mat, *trans, ip);
      mat.GetColumn(0, vec);
      nodevals[q + dof*0] = vec[0];
      nodevals[q + dof*1] = vec[1];
    }
    pfespace->GetElementVDofs(e, dofs);
    orifield->SetSubVector(dofs, nodevals);
  }
}

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   int nranks = Mpi::WorldSize();
   Hypre::Init();

#ifdef MFEM_USE_PETSC
   const char *petscrc_file = "";
   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
#endif

  int qoitype = static_cast<int>(QoIType::H1S_ERROR);
  bool weakBC = false;
  bool perturbMesh = false;
  double epsilon_pert =  0.006;
  int ref_ser = 2;
  int mesh_node_ordering = 0;
  int max_it = 100;
  double max_ch=0.002; //max design change
  double weight_1 = 1e4; //1e7; // 5e2;
  double weight_tmop = 1e-2;
  int metric_id   = 2;
  int target_id   = 1;
  int quad_order  = 8;
  int quad_order2 = 8;
  srand(9898975);
  bool visualization = false;
  double filterRadius = 0.000;
  int method = 0;
  int mesh_poly_deg     = 1;
  int nx                = 4;
  const char *mesh_file = "null.mesh";
  bool exactaction      = false;
  double ls_norm_fac    = 1.2;
  double ls_energy_fac  = 1.1;
  bool   bndr_fix       = true;
  bool   filter         = false;
  int    physics = 0;
  int physicsdim = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&ref_ser, "-rs", "--refine-serial",
                 "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&metric_id, "-mid", "--metric-id",
                "Mesh optimization metric:\n\t"
                "T-metrics\n\t"
                "1  : |T|^2                          -- 2D no type\n\t"
                "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                );
  args.AddOption(&target_id, "-tid", "--target-id",
                "Target (ideal element) type:\n\t"
                "1: Ideal shape, unit size\n\t"
                "2: Ideal shape, equal size\n\t"
                "3: Ideal shape, initial size\n\t"
                "4: Given full analytic Jacobian (in physical space)\n\t"
                "5: Ideal shape, given size (in physical space)");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
  args.AddOption(&quad_order2, "-qo2", "--quad_order2",
                  "Order of the quadrature rule for sensitivities.");
   args.AddOption(&method, "-met", "--method",
                  "0(Defaults to TMOP_MMA), 1 - MS");
   args.AddOption(&max_ch, "-ch", "--max-ch",
                  "max node movement");
   args.AddOption(&max_it, "-ni", "--newton-oter",
                  "number of iters");
   args.AddOption(&ftype, "-ft", "--ftype",
                  "function type");
   args.AddOption(&alphaw, "-alpha", "--alpha",
                  "alpha weight for functions");
   args.AddOption(&qoitype, "-qt", "--qtype",
                  "Quantity of interest type");
   args.AddOption(&weight_1, "-w1", "--weight1",
                  "Quantity of interest weight");
   args.AddOption(&weight_tmop, "-w2", "--weight2",
                  "Mesh quality weight type");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
    args.AddOption(&exactaction, "-ex", "--exact_action",
                  "-no-ex", "--no-exact-action",
                  "Enable exact action of TMOP_Integrator.");
   args.AddOption(&ls_norm_fac, "-lsn", "--ls-norm-fac",
                  "line-search norm factor");
   args.AddOption(&ls_energy_fac, "-lse", "--ls-energy-fac",
                  "line-search energy factor");
    args.AddOption(&bndr_fix, "-bndr", "--bndrfix",
                  "-bndrfree", "--bndr-free",
                  "Enable exact action of TMOP_Integrator.");
    args.AddOption(&visualization, "-vis", "--vis",
                  "-no-vis", "--no-vis",
                  "Enable/disable visualization.");
    args.AddOption(&weakBC, "-weakbc", "--weakbc",
                  "-no-weakbc", "--no-weakbc",
                  "Enable/disable weak boundary condition.");
    args.AddOption(&filter, "-filter", "--filter",
                  "-no-filter", "--no-filter",
                  "Use vector helmholtz filter.");
    args.AddOption(&filterRadius, "-frad", "--frad",
                    "Filter radius");
    args.AddOption(&physics, "-ph", "--physics",
                    "Physics");

                    

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

  enum QoIType qoiType  = static_cast<enum QoIType>(qoitype);
  bool dQduFD =false;
  bool dQdxFD =false;
  bool dQdxFD_global =false;
  bool BreakAfterFirstIt = false;

  // Create mesh
  Mesh *des_mesh = nullptr;
  if (strcmp(mesh_file, "null.mesh") == 0)
  {
     des_mesh = new Mesh(Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL,
                                        true, 1.0, 1.0));
  }
  else
  {
    des_mesh = new Mesh(mesh_file, 1, 1, false);
  }

  if(perturbMesh)
  {
     int tNumVertices  = des_mesh->GetNV();
     for (int i = 0; i < tNumVertices; ++i) {
        double * Coords = des_mesh->GetVertex(i);
        //if (Coords[ 0 ] != 0.0 && Coords[ 0 ] != 1.0 && Coords[ 1 ] != 0.0 && Coords[ 1 ] != 1.0) {
          //  Coords[ 0 ] = Coords[ 0 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* epsilon_pert);
          //  Coords[ 1 ] = Coords[ 1 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* epsilon_pert);

          Coords[ 0 ] = Coords[ 0 ] +0.5;
          //Coords[ 1 ] = Coords[ 1 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* epsilon_pert);
        //}
     }
  }

  // Refine mesh in serial
  for (int lev = 0; lev < ref_ser; lev++) { des_mesh->UniformRefinement(); }

  auto PMesh = new ParMesh(MPI_COMM_WORLD, *des_mesh);

  int dim = PMesh->SpaceDimension();

  if( physics ==1)
  {
    physicsdim = dim;
  }


  // -----------------------
  // Remaining mesh settings
  // -----------------------

  // Nodes are only active for higher order meshes, and share locations with
  // the vertices, plus all the higher- order control points within  the
  // element and along the edges and on the faces.
  if (nullptr == PMesh->GetNodes())
  {
    PMesh->SetCurvature(mesh_poly_deg, false, -1, 0);
  }

  // int mesh_poly_deg = PMesh->GetNodes()->FESpace()->GetElementOrder(0);

  // Create finite Element Spaces for analysis mesh
  // if ( dim != 2 ) {
  //   mfem_error("... This example only supports 2D meshes");
  // }

  // 4. Define a finite element space on the mesh. Here we use vector finite
  //    elements which are tensor products of quadratic finite elements. The
  //    number of components in the vector finite element space is specified by
  //    the last parameter of the FiniteElementSpace constructor.
  FiniteElementCollection *fec= new H1_FECollection(mesh_poly_deg, dim);
  ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(PMesh, fec, dim,
                                                               mesh_node_ordering);
  auto fespace_scalar = new ParFiniteElementSpace(PMesh, fec, 1);
  ParFiniteElementSpace pfespace_gf(PMesh, fec);
  ParGridFunction x_gf(&pfespace_gf);

  // 5. Make the mesh curved based on the above finite element space. This
  //    means that we define the mesh elements through a fespace-based
  //    transformation of the reference element.
  PMesh->SetNodalFESpace(pfespace);

  // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
  //    element space) as a finite element grid function in fespace. Note that
  //    changing x automatically changes the shapes of the mesh elements.
  ParGridFunction x(pfespace);
  PMesh->SetNodalGridFunction(&x);
  ParGridFunction x0(pfespace);
  x0 = x;
  ParGridFunction orifield(pfespace);
  int numOptVars = pfespace->GetTrueVSize();

  // TMOP Integrator setup
     TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 4: metric = new TMOP_Metric_004; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 36: metric = new TMOP_AMetric_036; break;
      case 50: metric = new TMOP_Metric_050; break;
      case 80: metric = new TMOP_Metric_080(0.8); break;
      case 85: metric = new TMOP_Metric_085; break;
      case 98: metric = new TMOP_Metric_098; break;
      case 107: metric = new TMOP_AMetric_107a; break;
      case 303: metric = new TMOP_Metric_303; break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   OSCoefficient *adapt_coeff = NULL;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE; break;
      case 5:
      {
         target_t = TargetConstructor::GIVEN_FULL;
         AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
         adapt_coeff = new OSCoefficient(dim, metric_id);
         tc->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tc;
         VisVectorField(adapt_coeff, PMesh, &orifield);
         break;
      }
      default:
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
   }
   if (target_c == NULL)
   {
    target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
   target_c->SetNodes(x0);


   IntegrationRules *irules = &IntRulesLo;
   auto tmop_integ = new TMOP_Integrator(metric, target_c);
   tmop_integ->SetIntegrationRules(*irules, quad_order);

   ConstantCoefficient metric_w(weight_tmop);
   tmop_integ->SetCoefficient(metric_w);
   tmop_integ->SetExactActionFlag(exactaction);

  // set esing variable bounds
  Vector objgrad(numOptVars); objgrad=0.0;
  Vector volgrad(numOptVars); volgrad=1.0;
  Vector xxmax(numOptVars);   xxmax=  0.001;
  Vector xxmin(numOptVars);   xxmin= -0.001;

  ParGridFunction gridfuncOptVar(pfespace);
  gridfuncOptVar = 0.0;
  ParGridFunction gridfuncLSBoundIndicator(pfespace);
  gridfuncLSBoundIndicator = 0.0;
  Array<int> vdofs;

  // Identify coordinate dofs perpendicular to BE
  if (strcmp(mesh_file, "null.mesh") == 0)
  {
    for (int i = 0; i < PMesh->GetNBE(); i++)
    {
      Element * tEle = PMesh->GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      pfespace->GetBdrElementVDofs(i, vdofs);
      const int nd = pfespace->GetBE(i)->GetDof();

      if (attribute == 1 || attribute == 3) // zero out motion in y
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
          // gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
        }
      }
      else if (attribute == 2 || attribute == 4) // zero out in x
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
          // gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
        }
      }
    }
  }
  else if(physics ==1)
  {
    for (int i = 0; i < PMesh->GetNBE(); i++)
    {
      Element * tEle = PMesh->GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      pfespace->GetBdrElementVDofs(i, vdofs);
      const int nd = pfespace->GetBE(i)->GetDof();

      if (attribute == 1 ||
          attribute == 3 ||
          attribute == 5 ) // zero out motion in y
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
        }
      }
      else if (attribute == 2 ||
          attribute == 4 ||
          attribute == 6 ) // zero out in x
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
        }
      }
    }
  }
  else
  {
    for (int i = 0; i < PMesh->GetNBE(); i++)
    {
      Element * tEle = PMesh->GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      pfespace->GetBdrElementVDofs(i, vdofs);
      const int nd = pfespace->GetBE(i)->GetDof();

      if (attribute == 2) // zero out motion in y
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0;
          if (bndr_fix) {
            gridfuncLSBoundIndicator[ vdofs[j+0*nd] ] = 1.0; // stops all motion
          }
        }
      }
      else if (attribute == 1) // zero out in x
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j] ] = 1.0;
          if (bndr_fix)
          {
            gridfuncLSBoundIndicator[ vdofs[j+nd] ] = 1.0; // stops all motion
          }
        }
      }
      else if (dim == 3 && attribute == 3) // zero out in z
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncLSBoundIndicator[ vdofs[j+2*nd] ] = 1.0;
        }
      }
    }
  }

  gridfuncOptVar.SetTrueVector();
  gridfuncLSBoundIndicator.SetTrueVector();

  Vector & trueOptvar = gridfuncOptVar.GetTrueVector();

  const int nbattr = PMesh->bdr_attributes.Max();
  std::vector<std::pair<int, double>> essentialBC(nbattr);
  std::vector<std::pair<int, int>> essentialBCfilter(nbattr);
  if( physics ==1 )
  {
    essentialBC.resize(1);
    essentialBC[0] = {4, 0};
  }
  else
  {
    for (int i = 0; i < nbattr; i++)
    {
      essentialBC[i] = {i+1, 0};
    }
  }

  if (strcmp(mesh_file, "null.mesh") == 0)
  {
    essentialBCfilter[0] = {1, 1};
    essentialBCfilter[1] = {2, 0};
    essentialBCfilter[2] = {3, 1};
    essentialBCfilter[3] = {4, 0};
  }
  else if( physics ==1 )
  {
    essentialBCfilter[0] = {1, 1};
    essentialBCfilter[1] = {2, 0};
    essentialBCfilter[2] = {3, 1};
    essentialBCfilter[3] = {4, 0};
    essentialBCfilter[4] = {5, 1};
    essentialBCfilter[5] = {6, 0};
  }
  else
  {
    essentialBCfilter[0] = {1, 0};
    essentialBCfilter[1] = {2, 1};
  }

  const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
  mfem::MMAOpt* mma = nullptr;
#ifdef MFEM_USE_PETSC
  mfem::NativeMMA* mmaPetsc = nullptr;
#endif
    // mfem::NativeMMA* mma = nullptr;
  TMOP_MMA *tmma = new TMOP_MMA(MPI_COMM_WORLD, trueOptvar.Size(), 0,
                                 trueOptvar, ir);
  {
#ifdef MFEM_USE_PETSC
    double a=0.0;
    double c=1000.0;
    double d=0.0;
    mmaPetsc=new mfem::NativeMMA(MPI_COMM_WORLD,1, objgrad,&a,&c,&d);
#else
    mma=new mfem::MMAOpt(MPI_COMM_WORLD, trueOptvar.Size(), 0, trueOptvar);
#endif
  }

if (myid == 0) {
  switch (qoiType) {
  case 0:
    std::cout<<" L2 Error"<<std::endl;
    break;
  case 1:
    std::cout<<" H1 semi-norm Error"<<std::endl;
    break;
  case 2:
    std::cout<<" ZZ Error"<<std::endl;
    break;
  case 3:
    std::cout<<" Avg Error"<<std::endl;;
    break;
  case 4:
    std::cout<<" Energy"<<std::endl;;
    break;
  case 5:
    std::cout<<" Global ZZ"<<std::endl;
    break;
  case 6:
    std::cout<<" L2+H1"<<std::endl;;
    break;
  case 7:
    std::cout<<" Struct Compliance"<<std::endl;;
    break;
  default:
    std::cout << "Unknown Error Coeff: " << qoiType << std::endl;
  }
}

  VectorCoefficient *loadFuncGrad = new VectorFunctionCoefficient(dim,
                                                              trueLoadFuncGrad);
  VectorCoefficient *trueSolutionGrad =
                          new VectorFunctionCoefficient(dim, trueSolGradFunc);
  MatrixCoefficient *trueSolutionHess = new
                                MatrixFunctionCoefficient(dim,trueHessianFunc);
  VectorCoefficient *trueSolutionHessV =
                          new VectorFunctionCoefficient(dim*dim, trueHessianFunc_v);
  PhysicsSolverBase * solver = nullptr;
  QuantityOfInterest QoIEvaluator(PMesh, qoiType, mesh_poly_deg,physicsdim);
  NodeAwareTMOPQuality MeshQualityEvaluator(PMesh, mesh_poly_deg);
  Coefficient *trueSolution = new FunctionCoefficient(trueSolFunc);
  Coefficient *QCoef = new FunctionCoefficient(loadFunc);


  mfem::VectorArrayCoefficient tractionLoad(PMesh->SpaceDimension());
  tractionLoad.Set(0, QCoef);
  tractionLoad.Set(1, new mfem::ConstantCoefficient(0.0));

  if( physics ==0)
  {
    Diffusion_Solver * diffsolver = new Diffusion_Solver(PMesh, essentialBC, mesh_poly_deg, trueSolution, weakBC, loadFuncGrad);
    diffsolver->SetManufacturedSolution(QCoef);
    diffsolver->setTrueSolGradCoeff(trueSolutionGrad);

    solver = diffsolver;

    QoIEvaluator.setTrueSolCoeff( trueSolution );
    if(qoiType == QoIType::ENERGY){QoIEvaluator.setTrueSolCoeff( QCoef );}
    QoIEvaluator.setTrueSolGradCoeff(trueSolutionGrad);
    QoIEvaluator.setTrueSolHessCoeff(trueSolutionHess);
    QoIEvaluator.setTrueSolHessCoeff(trueSolutionHessV);
  }
  else if(physics ==1)
  {
    //Elasticity_Solver * elasticitysolver = new Elasticity_Solver(PMesh, dirichletBC, tractionBC, mesh_poly_deg);
    Elasticity_Solver * elasticitysolver = new Elasticity_Solver(PMesh, essentialBC, mesh_poly_deg);
    elasticitysolver->SetLoad(&tractionLoad);

    solver = elasticitysolver;

    QoIEvaluator.setTractionCoeff(&tractionLoad);
  }

  //std::vector<std::pair<int, double>> essentialBC_filter(0);
  FunctionCoefficient leftrightwalls(&tanh_left_right_walls);
  ProductCoefficient filterRadiusCoeff(filterRadius, leftrightwalls);
  VectorHelmholtz *filterSolver;
  if (metric_id == 107)
  {
    // filterSolver = new VectorHelmholtz(PMesh, essentialBCfilter, &filterRadiusCoeff, mesh_poly_deg);
    filterSolver = new VectorHelmholtz(PMesh, essentialBCfilter, filterRadius, mesh_poly_deg);
  }
  else
  {
    filterSolver = new VectorHelmholtz(PMesh, essentialBCfilter, filterRadius, mesh_poly_deg);
  }

  QoIEvaluator.SetIntegrationRules(&IntRulesLo, quad_order2);
  x_gf.ProjectCoefficient(*trueSolution);

  Diffusion_Solver solver_FD1(PMesh, essentialBC, mesh_poly_deg, trueSolution, weakBC);
  Diffusion_Solver solver_FD2(PMesh, essentialBC, mesh_poly_deg, trueSolution, weakBC);
  solver_FD1.SetManufacturedSolution(QCoef);
  solver_FD2.SetManufacturedSolution(QCoef);

  QuantityOfInterest QoIEvaluator_FD1(PMesh, qoiType, 1);
  QuantityOfInterest QoIEvaluator_FD2(PMesh, qoiType, 1);

  ParaViewDataCollection paraview_dc("MeshOptimizer", PMesh);
  paraview_dc.SetLevelsOfDetail(1);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);

  //
  ParGridFunction & discretSol = solver->GetSolution();
  discretSol.ProjectCoefficient(*trueSolution);
  if (visualization)
  {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, discretSol,
                            "Initial Projected Solution", 0, 0, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
  }
  {
    solver->SetDesignVarFromUpdatedLocations(x);
    solver->FSolve();
    ParGridFunction & discretSol = solver->GetSolution();
    if (visualization)
    {
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, discretSol,
                              "Initial Solver Solution", 0, 480, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
    }
  }



  x.SetTrueVector();

  VisItDataCollection *visdc = new VisItDataCollection("tmop-pde", PMesh);
  visdc->RegisterField("solution", &(solver->GetSolution()));
  visdc->SetCycle(0);
  visdc->SetTime(0.0);
  visdc->Save();
  int save_freq = 10;


  if (method == 0)
  {
    auto init_l2_error = discretSol.ComputeL2Error(*trueSolution);
    auto init_grad_error = discretSol.ComputeGradError(trueSolutionGrad);
    auto init_h1_error = discretSol.ComputeH1Error(trueSolution, trueSolutionGrad);

    ParNonlinearForm a(pfespace);
    a.AddDomainIntegrator(tmop_integ);
    {
      Array<int> ess_bdr(PMesh->bdr_attributes.Max());
      ess_bdr = 1;
      //a.SetEssentialBC(ess_bdr);
    }
    IterativeSolver::PrintLevel newton_print;
    newton_print.Errors().Warnings().Iterations();
    // set the TMOP Integrator
    tmma->SetOperator(a);
    // Set change limits on dx
    tmma->SetUpperBound(max_ch);
    tmma->SetLowerBound(max_ch);
    // Set true vector so that it can be zeroed out
    {
      Vector & trueBounds = gridfuncLSBoundIndicator.GetTrueVector();
      tmma->SetTrueDofs(trueBounds);
    }
    // Set QoI and Solver and weight
    if (weight_1 > 0.0)
    {
      tmma->SetQuantityOfInterest(&QoIEvaluator);
      tmma->SetDiffusionSolver(reinterpret_cast<Diffusion_Solver*>(solver));       // TODO change to base class
      tmma->SetQoIWeight(weight_1);
      tmma->SetVectorHelmholtzFilter(filterSolver);
    }

    // Set min jac
    tmma->SetMinimumDeterminantThreshold(1e-7);

    // Set line search factors
    tmma->SetLineSearchNormFactor(ls_norm_fac);
    tmma->SetLineSearchEnergyFactor(ls_energy_fac);

    tmma->SetPrintLevel(newton_print);

    const real_t init_energy = tmma->GetEnergy(x.GetTrueVector(), true);
    const real_t init_metric_energy = tmma->GetEnergy(x.GetTrueVector(), false);
    const real_t init_qoi_energy = init_energy - init_metric_energy;

    // Set max # iterations
    bool save_after_every_iteration = true;
    if (save_after_every_iteration)
    {
      tmma->SetDataCollectionObjectandMesh(visdc, PMesh, save_freq);
    }
    tmma->SetMaxIter(max_it);
    if (filter)
    {
      tmma->MultFilter(x.GetTrueVector());
    }
    else
    {
      tmma->Mult(x.GetTrueVector());
    }
    x.SetFromTrueVector();
    if (!save_after_every_iteration)
    {
      visdc->SetCycle(1);
      visdc->SetTime(1.0);
      visdc->Save();
    }


    // Visualize the mesh displacement.
    if (visualization)
    {
      x0 -= x;
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, x0,
                              "Displacements", 800, 000, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");

      ParaViewDataCollection paraview_dc("NativeMeshOptimizer", PMesh);
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(1.0);
      //paraview_dc.RegisterField("Solution",&x_gf);
      paraview_dc.Save();
    }

    {
      ostringstream mesh_name;
      mesh_name << "optimized.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      PMesh->PrintAsSerial(mesh_ofs);
    }


    solver->SetDesignVarFromUpdatedLocations(x);
    solver->FSolve();
    ParGridFunction & discretSol = solver->GetSolution();
    if (visualization)
    {
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, discretSol,
                              "Final Solver Solution", 400, 480, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
    }

    auto final_l2_error = discretSol.ComputeL2Error(*trueSolution);
    auto final_grad_error = discretSol.ComputeGradError(trueSolutionGrad);
    auto final_h1_error = discretSol.ComputeH1Error(trueSolution, trueSolutionGrad);

    const real_t final_energy = tmma->GetEnergy(x.GetTrueVector(), true);
    const real_t final_metric_energy = tmma->GetEnergy(x.GetTrueVector(), false);
    const real_t final_qoi_energy = final_energy - final_metric_energy;

    discretSol.ProjectCoefficient(*trueSolution);
    if (visualization)
    {
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, discretSol,
                              "Final Projected Solution", 400, 000, 400, 400, "jRmclAppppppppppppp]]]]]]]]]]]]]]]");
    }
    if (myid == 0)
    {
      std::cout << "Initial L2 error: " << " " << init_l2_error << " " << std::endl;
      std::cout << "Final   L2 error: " << " " << final_l2_error << " " << std::endl;

      std::cout << "Initial Grad error: " << " " << init_grad_error << " " << std::endl;
      std::cout << "Final   Grad error: " << " " << final_grad_error << " " << std::endl;

      std::cout << "Initial H1 error: " << " " << init_h1_error << " " << std::endl;
      std::cout << "Final   H1 error: " << " " << final_h1_error << " " << std::endl;

      std::cout << "Initial Total/Metric/QOI Energy: " << init_energy << " " << init_metric_energy << " " << init_qoi_energy << std::endl;
      std::cout << "Final   Total/Metric/QOI Energy: " << final_energy << " " << final_metric_energy << " " << final_qoi_energy << std::endl;
    }

    if (visualization && adapt_coeff)
    {

         VisVectorField(adapt_coeff, PMesh, &orifield);
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, orifield,
                              "Orientation", 800, 480, 400, 400, "jRmclAevvppp]]]]]]]]]]]]]]]");
    }
  }
  else
  {
    int cycle_count = 1;
    for(int i=1;i<max_it;i++)
    {
      filterSolver->setLoadGridFunction(gridfuncOptVar);
      filterSolver->FSolve();
      ParGridFunction & filteredDesign = filterSolver->GetSolution();

      solver->SetDesign( filteredDesign );
      solver->FSolve();

      ParGridFunction & discretSol = solver->GetSolution();

      QoIEvaluator.SetDesign( filteredDesign );
      MeshQualityEvaluator.SetDesign( filteredDesign );

      QoIEvaluator.SetDiscreteSol( discretSol );
      QoIEvaluator.SetIntegrationRules(&IntRulesLo, quad_order);

      double ObjVal = QoIEvaluator.EvalQoI();
      double meshQualityVal = MeshQualityEvaluator.EvalQoI();

      double val = weight_1 * ObjVal+ weight_tmop * meshQualityVal;

      QoIEvaluator.EvalQoIGrad();
      MeshQualityEvaluator.EvalQoIGrad();

      ParLinearForm * dQdu = QoIEvaluator.GetDQDu();
      ParLinearForm * dQdxExpl = QoIEvaluator.GetDQDx();
      ParLinearForm * dMeshQdxExpl = MeshQualityEvaluator.GetDQDx();

      solver->ASolve( *dQdu );

      ParLinearForm * dQdxImpl = solver->GetImplicitDqDx();

      ParLinearForm dQdx(pfespace); dQdx = 0.0;
      ParLinearForm dQdx_physics(pfespace); dQdx_physics = 0.0;
      ParLinearForm dQdx_filtered(pfespace); dQdx_filtered = 0.0;
      dQdx_physics.Add(1.0, *dQdxExpl);
      dQdx_physics.Add(1.0, *dQdxImpl);

      dQdx_filtered.Add(weight_1, *dQdxExpl);
      dQdx_filtered.Add(weight_1, *dQdxImpl);
      dQdx_filtered.Add(weight_tmop, *dMeshQdxExpl);

      HypreParVector *truedQdx_physics = dQdx_physics.ParallelAssemble();
      mfem::ParGridFunction dQdx_physicsGF(pfespace, truedQdx_physics);

      //std::cout << dQdx_filtered.Norml2() << " k101-filt1\n";
      filterSolver->ASolve(dQdx_filtered);
      ParLinearForm * dQdxImplfilter = filterSolver->GetImplicitDqDx();

      dQdx.Add(1.0, *dQdxImplfilter);
      //std::cout << dQdxImplfilter->Norml2() << " k101-filt2\n";

      HypreParVector *truedQdx = dQdx.ParallelAssemble();

      HypreParVector *truedQdx_Expl = dQdxExpl->ParallelAssemble();
      HypreParVector *truedQdx_Impl = dQdxImpl->ParallelAssemble();

      // Construct grid function from hypre vector
      mfem::ParGridFunction dQdx_ExplGF(pfespace, truedQdx_Expl);
      mfem::ParGridFunction dQdx_ImplGF(pfespace, truedQdx_Impl);


      objgrad = *truedQdx;

      //----------------------------------------------------------------------------------------------------------

      if(dQduFD)
      {
        double epsilon = 1e-8;
        mfem::ParGridFunction tFD_sens(fespace_scalar); tFD_sens = 0.0;
        for( int Ia = 0; Ia<discretSol.Size(); Ia++)
        {
          if (myid == 0)
          {
            std::cout<<"iter: "<< Ia<< " out of: "<<discretSol.Size() <<std::endl;
          }
          discretSol[Ia] +=epsilon;

          // QoIEvaluator_FD1(PMesh, qoiType, 1);
          QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD1.SetDiscreteSol( discretSol );
          QoIEvaluator_FD1.SetNodes(x0);
          QoIEvaluator_FD1.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          discretSol[Ia] -=2.0*epsilon;

          //QuantityOfInterest QoIEvaluator_FD2(PMesh, qoiType, 1);
          QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD2.SetDiscreteSol( discretSol );
          QoIEvaluator_FD2.SetNodes(x0);
          QoIEvaluator_FD2.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          discretSol[Ia] +=epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }
        dQdu->Print();
        std::cout<<"  ----------  FD Diff ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdu Analytic - FD Diff ------------"<<std::endl;
        mfem::ParGridFunction tFD_diff(fespace_scalar); tFD_diff = 0.0;
        tFD_diff = *dQdu;
        tFD_diff -=tFD_sens;
        //tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      }

      if(dQdxFD)
      {
        // nodes are p
        // det(J) is order d*p-1
        double epsilon = 1e-8;
        mfem::ParGridFunction tFD_sens(pfespace); tFD_sens = 0.0;
        ConstantCoefficient zerocoeff(0.0);
        Array<double> GLLVec;
        int nqpts;
        {
          const IntegrationRule *ir = &IntRulesLo.Get(Geometry::SQUARE, 8);
          nqpts = ir->GetNPoints();
          // std::cout << nqpts << " k10c\n";
          for (int e = 0; e < PMesh->GetNE(); e++)
          {
            ElementTransformation *T = PMesh->GetElementTransformation(e);
            for (int q = 0; q < ir->GetNPoints(); q++)
            {
              const IntegrationPoint &ip = ir->IntPoint(q);
              T->SetIntPoint(&ip);
              double disc_val = discretSol.GetValue(e, ip);
              double exact_val = trueSolution->Eval( *T, ip );
              GLLVec.Append(disc_val-exact_val);
            }
          }
        }
        std::cout << nqpts << " " << GLLVec.Size() << " k10c\n";
        // MFEM_ABORT(" ");

        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          if(gridfuncLSBoundIndicator[Ia] == 1.0)
          {
            (*dQdxExpl)[Ia] = 0.0;

            continue;
          }

          std::cout<<"iter: "<< Ia<< " out of: "<<gridfuncOptVar.Size() <<std::endl;
          double fac = 1.0-gridfuncLSBoundIndicator[Ia];
          gridfuncOptVar[Ia] +=(fac)*epsilon;

          QuantityOfInterest QoIEvaluator_FD1(PMesh, qoiType, 1);
          QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD1.SetDiscreteSol( discretSol );
          QoIEvaluator_FD1.SetNodes(x0);
          QoIEvaluator_FD1.SetGLLVec(GLLVec);
          QoIEvaluator_FD1.SetNqptsPerEl(nqpts);
          QoIEvaluator_FD1.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          gridfuncOptVar[Ia] -=(fac)*2.0*epsilon;

          QuantityOfInterest QoIEvaluator_FD2(PMesh, qoiType, 1);
          QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
          // QoIEvaluator_FD2.setTrueSolCoeff(  &zerocoeff );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD2.SetDiscreteSol( discretSol );
          QoIEvaluator_FD2.SetNodes(x0);
          QoIEvaluator_FD2.SetGLLVec(GLLVec);
          QoIEvaluator_FD2.SetNqptsPerEl(nqpts);
          QoIEvaluator_FD2.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          gridfuncOptVar[Ia] +=(fac)*epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }

        dQdxExpl->Print();
        std::cout<<"  ----------  FD Diff ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdx Analytic - FD Diff ------------"<<std::endl;
        mfem::ParGridFunction tFD_diff(pfespace); tFD_diff = 0.0;
        tFD_diff = *dQdxExpl;
        tFD_diff -=tFD_sens;
        tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          tFD_diff[Ia] *= (1.0-gridfuncLSBoundIndicator[Ia]);
        }
        // tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      }

      if(dQdxFD_global)
      {
        double epsilon = 1e-8;
        mfem::ParGridFunction tFD_sens(pfespace); tFD_sens = 0.0;
        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          if(gridfuncLSBoundIndicator[Ia] == 1.0)
          {
            dQdx_physics[Ia] = 0.0;
            dQdx[Ia] = 0.0;

            continue;
          }
          std::cout<<"iter: "<< Ia<< " out of: "<<gridfuncOptVar.Size() <<std::endl;
          double fac = 1.0-gridfuncLSBoundIndicator[Ia];
          gridfuncOptVar[Ia] +=fac*epsilon;

          solver_FD1.SetDesign( gridfuncOptVar );
          solver_FD1.FSolve();
          ParGridFunction & discretSol_1 = solver_FD1.GetSolution();

          //QuantityOfInterest QoIEvaluator_FD1(PMesh, qoiType, 1);
          QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD1.SetDiscreteSol( discretSol_1 );
          QoIEvaluator_FD1.SetNodes(x0);
          QoIEvaluator_FD1.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          gridfuncOptVar[Ia] -=fac*2.0*epsilon;

          solver_FD2.SetDesign( gridfuncOptVar );
          solver_FD2.FSolve();
          ParGridFunction & discretSol_2 = solver_FD2.GetSolution();

          //QuantityOfInterest QoIEvaluator_FD2(PMesh, qoiType, 1);
          QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
          if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
          QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
          QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
          QoIEvaluator_FD2.SetDiscreteSol( discretSol_2 );
          QoIEvaluator_FD2.SetNodes(x0);
          QoIEvaluator_FD2.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          gridfuncOptVar[Ia] +=fac*epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }

        dQdx.Print();
        std::cout<<"  ----------  FD Diff - Global ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdx Analytic - FD Diff ------------"<<std::endl;
        mfem::ParGridFunction tFD_diff(pfespace); tFD_diff = 0.0;
        tFD_diff = dQdx;
        tFD_diff -=tFD_sens;
        tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
        for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
        {
          tFD_diff[Ia] *= (1.0-gridfuncLSBoundIndicator[Ia]);
        }
        // tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;

        paraview_dc.SetCycle(i);
        paraview_dc.SetTime(i*1.0);
        //paraview_dc.RegisterField("ObjGrad",&objGradGF);
        paraview_dc.RegisterField("Solution",&x_gf);
        paraview_dc.RegisterField("SolutionD",&discretSol   );
        paraview_dc.RegisterField("Sensitivity",&dQdx_physicsGF);
        paraview_dc.RegisterField("SensitivityFD",&tFD_sens);
        paraview_dc.RegisterField("SensitivityDiff",&tFD_diff);
        paraview_dc.RegisterField("SensitivityExpl",&dQdx_ExplGF);
        paraview_dc.RegisterField("SensitivityImpl",&dQdx_ImplGF);
        paraview_dc.Save();

        std::cout<<"expl: "<<dQdxExpl->Norml2()<<std::endl;
        std::cout<<"impl: "<<dQdxImpl->Norml2()<<std::endl;
      }

      if( BreakAfterFirstIt )
      {
        mfem::mfem_error("break before update");
      }

      //----------------------------------------------------------------------------------------------------------
      gridfuncOptVar.SetTrueVector();
      Vector & trueBounds = gridfuncLSBoundIndicator.GetTrueVector();

      // impose desing variable bounds - set xxmin and xxmax
      xxmin=trueOptvar; xxmin-=max_ch;
      xxmax=trueOptvar; xxmax+=max_ch;
      for(int li=0;li<xxmin.Size();li++){
        if( trueBounds[li] ==1.0)
        {
          xxmin[li] = -1e-8;
          xxmax[li] =  1e-8;
        }
      }

      Vector Xi = x0;
      Xi += filteredDesign;
      PMesh->SetNodes(Xi);
      PMesh->DeleteGeometricFactors();

      if (i % save_freq == 0)
      {
         visdc->SetCycle(cycle_count++);
         visdc->SetTime(cycle_count*1.0);
         visdc->Save();
      }


      x_gf.ProjectCoefficient(*trueSolution);
      //ParGridFunction objGradGF(pfespace); objGradGF = objgrad;
      paraview_dc.SetCycle(i);
      paraview_dc.SetTime(i*1.0);
      //paraview_dc.RegisterField("ObjGrad",&objGradGF);
      paraview_dc.RegisterField("SolutionD",&discretSol   );
      //paraview_dc.RegisterField("Solution",&x_gf);
      //paraview_dc.RegisterField("Sensitivity",&dQdx_physicsGF);
      paraview_dc.Save();

      double localGradNormSquared = std::pow(objgrad.Norml2(), 2);
      double globGradNorm;
  #ifdef MFEM_USE_MPI
    MPI_Allreduce(&localGradNormSquared, &globGradNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  #endif
    globGradNorm = std::sqrt(globGradNorm);

    if (myid == 0)
    {
      std:cout<<"Iter: "<<i<<" obj: "<<val<<" with: "<<ObjVal<<" | "<<meshQualityVal<<" objGrad_Norm: "<<globGradNorm<<std::endl;
    }

  #ifdef MFEM_USE_PETSC
      double  conDummy = -0.1;
      mmaPetsc->Update(trueOptvar,objgrad,&conDummy,&volgrad,xxmin,xxmax);
  #else
      mfem:Vector conDummy(1);  conDummy= -0.1;
      //std::cout << trueOptvar.Norml2() << " k10-dxpre\n";
      mma->Update(i, objgrad, conDummy, volgrad, xxmin,xxmax, trueOptvar);
      //std::cout << trueOptvar.Norml2() << " k10-dxpost\n";
  #endif

      gridfuncOptVar.SetFromTrueVector();

      // std::string tDesingName = "DesingVarVec";
      // desingVarVec.Save( tDesingName.c_str() );

      // std::string tFieldName = "FieldVec";
      // tPreassureGF.Save( tFieldName.c_str() );
    }

    if (visualization)
    {
        x0 -= x;
        socketstream vis;
        common::VisualizeField(vis, "localhost", 19916, x0,
                              "Displacements", 400, 400, 300, 300, "jRmclA");
    }

      {
        ostringstream mesh_name;
        mesh_name << "optimized.mesh";
        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        PMesh->PrintAsSerial(mesh_ofs);
    }
  }

  delete solver;
  delete PMesh;

  return 0;
}
