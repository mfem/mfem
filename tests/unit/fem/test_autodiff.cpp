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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace autodiff
{
void scalarFun(const Vector& param, const ADVector& x, ADVector& f)
{
   int d = x.size();

   real_t pi  = (real_t)(M_PI);
   f[0] = sin(pi*x[0]);

   if (d >= 2)
   {
      f[0] *= sin(2*pi*x[1]);
   }
   if (d >= 3)
   {
      f[0] *= sin(3*pi*x[2]);
   }
}

real_t scalarFunD(const Vector& x)
{
   int d = x.Size();

   real_t pi  = (real_t)(M_PI);
   real_t f = sin(pi*x[0]);

   if (d >= 2)
   {
      f *= sin(2*pi*x[1]);
   }
   if (d >= 3)
   {
      f *= sin(3*pi*x[2]);
   }
   return f;
}

void scalarGrad(const Vector & x, Vector & a)
{
   real_t pi  = (real_t)(M_PI);

   real_t sx = sin(pi*x[0]);
   real_t cx = cos(pi*x[0]);
   real_t sy = 1.0;
   real_t cy = 1.0;
   real_t sz = 1.0;
   real_t cz = 1.0;

   a[0] = pi*cx;
   int d = x.Size();
   if (d >= 2)
   {
      sy = sin(2*pi*x[1]);
      cy = cos(2*pi*x[1]);
      a[0] = pi*cx*sy;
      a[1] = 2*pi*sx*cy;
   }
   if (d >= 3)
   {
      sz = sin(3*pi*x[2]);
      cz = cos(3*pi*x[2]);
      a[0] = pi*cx*sy*sz;
      a[1] = 2*pi*sx*cy*sz;
      a[2] = 3*pi*sx*sy*cz;
   }
}

void scalarFun2(const Vector& param, const ADVector& x, ADVector& f)
{
   int d = x.size();

   real_t pi  = (real_t)(M_PI);
   f[0] = sin(param[0]*pi*x[0]);

   if (d >= 2)
   {
      f[0] *= sin(param[1]*pi*x[1]);
   }
   if (d >= 3)
   {
      f[0] *= sin(param[2]*pi*x[2]);
   }
}


TEST_CASE("Autodiff of scalar function for Coefficient",
          "[AD Scalar]")
{
   Vector param(3);
   param[0] = 1.0;
   param[1] = 2.0;
   param[2] = 3.0;
   ADVectorFunc fun(scalarFun);
   ADVectorFunc fun2(scalarFun2, param);

   SECTION("1D")
   {
      real_t ref_sol;
      Vector x(1), ref_grad(1), grad(1);

      for (int i = 0; i < 10; i++)
      {
         x[0] = 0.1*i;

         // Solution
         ref_sol = scalarFunD(x);

         REQUIRE(fun.ScalarSolution(x)- ref_sol == MFEM_Approx(0.0));
         REQUIRE(fun2.ScalarSolution(x) -ref_sol == MFEM_Approx(0.0));
#ifdef MFEM_USE_CODIPACK
         // Gradient -- > Implies Jacobian
         scalarGrad(x,ref_grad);

         fun.Gradient(x,grad);
         grad -= ref_grad;
         REQUIRE(grad.Norml2() == MFEM_Approx(0.0));

         fun2.Gradient(x,grad);
         grad -= ref_grad;
         REQUIRE(grad.Norml2() == MFEM_Approx(0.0));
#endif
      }
   }

   SECTION("2D")
   {
      real_t ref_sol;
      Vector x(2), ref_grad(2), grad(2);

      for (int i = 0; i < 10; i++)
      {
         for (int j = 0; j < 10; j++)
         {
            x[0] = 0.1*i;
            x[1] = 0.1*j;

            // Solution
            ref_sol = scalarFunD(x);

            REQUIRE(fun.ScalarSolution(x)- ref_sol == MFEM_Approx(0.0));
            REQUIRE(fun2.ScalarSolution(x) -ref_sol == MFEM_Approx(0.0));
#ifdef MFEM_USE_CODIPACK
            // Gradient -- > Implies Jacobian
            scalarGrad(x,ref_grad);

            fun.Gradient(x,grad);
            grad -= ref_grad;
            REQUIRE(grad.Norml2() == MFEM_Approx(0.0));

            fun2.Gradient(x,grad);
            grad -= ref_grad;
            REQUIRE(grad.Norml2() == MFEM_Approx(0.0));
#endif
         }
      }
   }

   SECTION("3D")
   {
      real_t ref_sol;
      Vector x(3), ref_grad(3), grad(3);

      for (int i = 0; i < 10; i++)
      {
         for (int j = 0; j < 10; j++)
         {
            for (int k = 0; k < 10; k++)
            {
               x[0] = 0.1*i;
               x[1] = 0.1*j;
               x[2] = 0.1*k;

               // Solution
               ref_sol = scalarFunD(x);

               REQUIRE(fun.ScalarSolution(x)- ref_sol == MFEM_Approx(0.0));
               REQUIRE(fun2.ScalarSolution(x) -ref_sol == MFEM_Approx(0.0));
#ifdef MFEM_USE_CODIPACK
               // Gradient -- > Implies Jacobian
               scalarGrad(x,ref_grad);

               fun.Gradient(x,grad);
               grad -= ref_grad;
               REQUIRE(grad.Norml2() == MFEM_Approx(0.0));

               fun2.Gradient(x,grad);
               grad -= ref_grad;
               REQUIRE(grad.Norml2() == MFEM_Approx(0.0));
#endif
            }
         }
      }
   }

   SECTION("2D Coefficient")
   {
      Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
      int order_quad = 6;
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      FunctionCoefficient sol_ref_coeff(scalarFunD);
      FunctionCoefficient sol_coeff(fun.GetScalarSolution());
      SumCoefficient sol_dif_coeff(sol_ref_coeff, sol_coeff, -1.0);

      real_t sol_norm = ComputeLpNorm(2.0, sol_dif_coeff, mesh, irs);
      REQUIRE(sol_norm == MFEM_Approx(0.0));

#ifdef MFEM_USE_CODIPACK
      VectorFunctionCoefficient grad_ref_coeff(2,scalarGrad);
      VectorFunctionCoefficient grad_coeff(2,fun.GetGradient());
      VectorSumCoefficient grad_dif_coeff(grad_ref_coeff, grad_coeff, -1.0);

      real_t grad_norm = ComputeLpNorm(2.0, sol_dif_coeff, mesh, irs);
      REQUIRE(grad_norm == MFEM_Approx(0.0));
#endif
   }

} // Test Case


void scalarTDFun(const Vector& param, const ADVector& x, ADFloat t, ADVector& f)
{
   int d = x.size();

   real_t pi  = (real_t)(M_PI);
   f[0] = (2.0*t + 3.0*t*t)*sin(3*pi*x[0]);

   if (d >= 2)
   {
      f[0] *= sin(5*pi*x[1]);
   }
   if (d >= 3)
   {
      f[0] *= sin(7*pi*x[2]);
   }
}

real_t scalarTDFunD(const Vector& x, const real_t t)
{
   int d = x.Size();

   real_t pi  = (real_t)(M_PI);
   real_t f= (2.0*t + 3.0*t*t)*sin(3*pi*x[0]);

   if (d >= 2)
   {
      f *= sin(5*pi*x[1]);
   }
   if (d >= 3)
   {
      f *= sin(7*pi*x[2]);
   }
   return f;
}

real_t  scalarTDRate(const Vector& x, const real_t t)
{
   int d = x.Size();

   real_t pi  = (real_t)(M_PI);
   real_t f = (2.0 + 6.0*t)*sin(3*pi*x[0]);

   if (d >= 2)
   {
      f *= sin(5*pi*x[1]);
   }
   if (d >= 3)
   {
      f *= sin(7*pi*x[2]);
   }
   return f;
}

void scalarTDGrad(const Vector & x, const real_t t, Vector & a)
{
   real_t amp = (2.0*t + 3.0*t*t);

   real_t pi  = (real_t)(M_PI);

   real_t sx = sin(3*pi*x[0]);
   real_t cx = cos(3*pi*x[0]);
   real_t sy = 1.0;
   real_t cy = 1.0;
   real_t sz = 1.0;
   real_t cz = 1.0;

   a[0] = amp*3*pi*cx;
   int d = x.Size();
   if (d >= 2)
   {
      sy = sin(5*pi*x[1]);
      cy = cos(5*pi*x[1]);
      a[0] = amp*3*pi*cx*sy;
      a[1] = amp*5*pi*sx*cy;
   }
   if (d >= 3)
   {
      sz = sin(7*pi*x[2]);
      cz = cos(7*pi*x[2]);
      a[0] = amp*3*pi*cx*sy*sz;
      a[1] = amp*5*pi*sx*cy*sz;
      a[2] = amp*7*pi*sx*sy*cz;
   }
}

void scalarTDFun2(const Vector& param, const ADVector& x, const ADFloat t,
                  ADVector& f)
{
   ADFloat amp = (2.0*t + 3.0*t*t);
   int d = x.size();

   real_t pi  = (real_t)(M_PI);
   f[0] = amp*sin(param[0]*pi*x[0]);

   if (d >= 2)
   {
      f[0] *= sin(param[1]*pi*x[1]);
   }
   if (d >= 3)
   {
      f[0] *= sin(param[2]*pi*x[2]);
   }
}

TEST_CASE("Autodiff of TD scalar function for Coefficient",
          "[AD TD Scalar]")
{
   Vector param(3);
   param[0] = 3.0;
   param[1] = 5.0;
   param[2] = 7.0;
   ADVectorTDFunc fun(scalarTDFun);
   ADVectorTDFunc fun2(scalarTDFun2,param);

   SECTION("Time dependent - 1D")
   {
      real_t t, ref_sol;
      Vector x(1);
#ifdef MFEM_USE_CODIPACK
      real_t ref_rate;
      Vector ref_grad(1), grad(1);
#endif
      for (int i = 0; i < 10; i++)
      {
         for (int j = 0; j < 10; j++)
         {
            t = 0.1*i;
            x[0] = 0.1*j;

            // Solution
            ref_sol = scalarTDFunD(x,t);

            REQUIRE(fun.ScalarSolution(x,t)- ref_sol  == MFEM_Approx(0.0));
            REQUIRE(fun2.ScalarSolution(x,t) -ref_sol == MFEM_Approx(0.0));
#ifdef MFEM_USE_CODIPACK
            // Rate
            ref_rate = scalarTDRate(x,t);

            REQUIRE(fun.ScalarRate(x,t)- ref_rate  == MFEM_Approx(0.0));
            REQUIRE(fun2.ScalarRate(x,t) -ref_rate == MFEM_Approx(0.0));

            // Gradient -- > Implies Jacobian
            scalarTDGrad(x,t,ref_grad);

            fun.Gradient(x,t,grad);
            grad -= ref_grad;
            REQUIRE(grad.Norml2() == MFEM_Approx(0.0));

            fun2.Gradient(x,t,grad);
            grad -= ref_grad;
            REQUIRE(grad.Norml2() == MFEM_Approx(0.0));
#endif
         }
      }
   }

   SECTION("Time dependent - 2D")
   {
      real_t t, ref_sol;
      Vector x(2);
#ifdef MFEM_USE_CODIPACK
      real_t ref_rate;
      Vector ref_grad(2), grad(2);
#endif
      for (int i = 0; i < 10; i++)
      {
         for (int j = 0; j < 10; j++)
         {
            for (int k = 0; k < 10; k++)
            {
               t = 0.1*i;
               x[0] = 0.1*j;
               x[1] = 0.1*k;

               // Solution
               ref_sol = scalarTDFunD(x,t);

               REQUIRE(fun.ScalarSolution(x,t)- ref_sol  == MFEM_Approx(0.0));
               REQUIRE(fun2.ScalarSolution(x,t) -ref_sol == MFEM_Approx(0.0));
#ifdef MFEM_USE_CODIPACK
               // Rate
               ref_rate = scalarTDRate(x,t);

               REQUIRE(fun.ScalarRate(x,t)- ref_rate  == MFEM_Approx(0.0));
               REQUIRE(fun2.ScalarRate(x,t) -ref_rate == MFEM_Approx(0.0));

               // Gradient -- > Implies Jacobian
               scalarTDGrad(x,t,ref_grad);

               fun.Gradient(x,t,grad);
               grad -= ref_grad;
               REQUIRE(grad.Norml2() == MFEM_Approx(0.0));

               fun2.Gradient(x,t,grad);
               grad -= ref_grad;
               REQUIRE(grad.Norml2() == MFEM_Approx(0.0));
#endif
            }
         }
      }
   }

   SECTION("Time dependent - 3D")
   {
      real_t t, ref_sol;
      Vector x(3);
#ifdef MFEM_USE_CODIPACK
      real_t ref_rate;
      Vector ref_grad(3), grad(3);
#endif

      for (int i = 0; i < 10; i++)
      {
         for (int j = 0; j < 10; j++)
         {
            for (int k = 0; k < 10; k++)
            {
               for (int l = 0; l < 10; l++)
               {

                  t = 0.1*i;
                  x[0] = 0.1*j;
                  x[1] = 0.1*k;
                  x[2] = 0.1*l;

                  // Solution
                  ref_sol = scalarTDFunD(x,t);

                  REQUIRE(fun.ScalarSolution(x,t)- ref_sol  == MFEM_Approx(0.0));
                  REQUIRE(fun2.ScalarSolution(x,t) -ref_sol == MFEM_Approx(0.0));
#ifdef MFEM_USE_CODIPACK
                  // Rate
                  ref_rate = scalarTDRate(x,t);

                  REQUIRE(fun.ScalarRate(x,t)- ref_rate  == MFEM_Approx(0.0));
                  REQUIRE(fun2.ScalarRate(x,t) -ref_rate == MFEM_Approx(0.0));

                  // Gradient -- > Implies Jacobian
                  scalarTDGrad(x,t,ref_grad);

                  fun.Gradient(x,t,grad);
                  grad -= ref_grad;
                  REQUIRE(grad.Norml2() == MFEM_Approx(0.0));

                  fun2.Gradient(x,t,grad);
                  grad -= ref_grad;
                  REQUIRE(grad.Norml2() == MFEM_Approx(0.0));
#endif
               }
            }
         }
      }
   }

   SECTION("2D Coefficient")
   {
      Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
      int order_quad = 6;
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      FunctionCoefficient sol_ref_coeff(scalarTDFunD);
      FunctionCoefficient sol_coeff(fun.GetScalarSolution());
      SumCoefficient sol_dif_coeff(sol_ref_coeff, sol_coeff, -1.0);
      for (int i = 0; i < 10; i++)
      {
         real_t t = 0.1*i;
         sol_ref_coeff.SetTime(t);
         sol_coeff.SetTime(t);
         real_t sol_norm = ComputeLpNorm(2.0, sol_dif_coeff, mesh, irs);
         REQUIRE(sol_norm == MFEM_Approx(0.0));
      }

#ifdef MFEM_USE_CODIPACK
      FunctionCoefficient rate_ref_coeff(scalarTDRate);
      FunctionCoefficient rate_coeff(fun.GetScalarRate());
      SumCoefficient rate_dif_coeff(rate_ref_coeff, rate_coeff, -1.0);
      for (int i = 0; i < 10; i++)
      {
         real_t t = 0.1*i;
         rate_ref_coeff.SetTime(t);
         rate_coeff.SetTime(t);
         real_t rate_norm = ComputeLpNorm(2.0, rate_dif_coeff, mesh, irs);
         REQUIRE(rate_norm == MFEM_Approx(0.0));
      }

      VectorFunctionCoefficient grad_ref_coeff(2,scalarTDGrad);
      VectorFunctionCoefficient grad_coeff(2,fun.GetGradient());
      VectorSumCoefficient grad_dif_coeff(grad_ref_coeff, grad_coeff, -1.0);

      real_t grad_norm = ComputeLpNorm(2.0, sol_dif_coeff, mesh, irs);
      REQUIRE(grad_norm == MFEM_Approx(0.0));
      for (int i = 0; i < 10; i++)
      {
         real_t t = 0.1*i;
         grad_ref_coeff.SetTime(t);
         grad_coeff.SetTime(t);
         real_t grad_norm = ComputeLpNorm(2.0, sol_dif_coeff, mesh, irs);
         REQUIRE(grad_norm == MFEM_Approx(0.0));
      }
#endif
   }

} // Test Case

void vectorFun(const Vector& param, const ADVector& x, ADVector& f)
{
   real_t pi  = (real_t)(M_PI);
   f[0] = sin(pi*x[0])*cos(pi*x[1])*3*x[2]*x[2];
   f[1] = cos(pi*x[0])*cos(pi*x[1])*2*x[2];
   f[2] = -pi*cos(pi*x[0])*cos(pi*x[1])*x[2]*x[2]*x[2]
          +pi*cos(pi*x[0])*sin(pi*x[1])*x[2]*x[2];
}

void vectorFunD(const Vector& x, Vector& f)
{
   real_t pi  = (real_t)(M_PI);
   f[0] = sin(pi*x[0])*cos(pi*x[1])*3*x[2]*x[2];
   f[1] = cos(pi*x[0])*cos(pi*x[1])*2*x[2];
   f[2] = -pi*cos(pi*x[0])*cos(pi*x[1])*x[2]*x[2]*x[2]
          +pi*cos(pi*x[0])*sin(pi*x[1])*x[2]*x[2];
}

void vectorJac(const Vector & x, DenseMatrix & j)
{
   real_t pi  = (real_t)(M_PI);
   j(0,0) = pi*cos(pi*x[0])*cos(pi*x[1])*3*x[2]*x[2];
   j(0,1) = -pi*sin(pi*x[0])*sin(pi*x[1])*3*x[2]*x[2];
   j(0,2) = sin(pi*x[0])*cos(pi*x[1])*6*x[2];

   j(1,0) = -pi*sin(pi*x[0])*cos(pi*x[1])*2*x[2];
   j(1,1) = -pi*cos(pi*x[0])*sin(pi*x[1])*2*x[2];
   j(1,2) = cos(pi*x[0])*cos(pi*x[1])*2;

   j(2,0) = pi*pi*sin(pi*x[0])*cos(pi*x[1])*x[2]*x[2]*x[2]
            -pi*pi*sin(pi*x[0])*sin(pi*x[1])*x[2]*x[2];
   j(2,1) = pi*pi*cos(pi*x[0])*sin(pi*x[1])*x[2]*x[2]*x[2]
            +pi*pi*cos(pi*x[0])*cos(pi*x[1])*x[2]*x[2];
   j(2,2) = -pi*cos(pi*x[0])*cos(pi*x[1])*3*x[2]*x[2]
            +pi*cos(pi*x[0])*sin(pi*x[1])*2*x[2];
}

void vectorCurl(const Vector & x, Vector& curl)
{
   DenseMatrix jac(x.Size());
   vectorJac(x,jac);

   // Curl
   curl[0] = jac(2,1) - jac(1,2);
   curl[1] = jac(0,2) - jac(2,0);
   curl[2] = jac(1,0) - jac(0,1);
}

TEST_CASE("Autodiff of vector function for Coefficient",
          "[AD Vector]")
{

   ADVectorFunc fun(vectorFun);

   SECTION("3D")
   {
      Vector x(3), ref_sol(3), sol(3);
#ifdef MFEM_USE_CODIPACK
      real_t ref_div, div;
      Vector  ref_curl(3), curl(3);
      DenseMatrix ref_jac(3,3), jac(3,3);
#endif

      for (int i = 0; i < 10; i++)
      {
         for (int j = 0; j < 10; j++)
         {
            for (int k = 0; k < 10; k++)
            {
               x[0] = 0.1*i;
               x[1] = 0.1*j;
               x[2] = 0.1*k;

               // Solution
               vectorFunD(x,ref_sol);

               fun.Solution(x,sol);
               sol -= ref_sol;
               REQUIRE(sol.Norml2() == MFEM_Approx(0.0));
#ifdef MFEM_USE_CODIPACK
               // Jacobian
               vectorJac(x,ref_jac);

               fun.Jacobian(x,jac);
               jac -= ref_jac;
               REQUIRE(jac.FNorm() == MFEM_Approx(0.0));

               // Curl
               ref_curl[0] = ref_jac(2,1) - ref_jac(1,2);
               ref_curl[1] = ref_jac(0,2) - ref_jac(2,0);
               ref_curl[2] = ref_jac(1,0) - ref_jac(0,1);
               fun.Curl(x,curl);
               curl -= ref_curl;
               // REQUIRE(curl.Norml2() == MFEM_Approx(0.0));

               // Divergence
               ref_div = 0.0;

               div = fun.Divergence(x);
               REQUIRE(div-ref_div == MFEM_Approx(0.0));
#endif
            }
         }
      }
   }

   SECTION("Coefficient")
   {
      Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2,Element::HEXAHEDRON);
      int order_quad = 6;
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      VectorFunctionCoefficient sol_ref_coeff(3,vectorFunD);
      VectorFunctionCoefficient sol_coeff(3,fun.GetSolution());
      VectorSumCoefficient sol_dif_coeff(sol_ref_coeff, sol_coeff, -1.0);

      real_t sol_norm = ComputeLpNorm(2.0, sol_dif_coeff, mesh, irs);
      REQUIRE(sol_norm == MFEM_Approx(0.0));

#ifdef MFEM_USE_CODIPACK
      FunctionCoefficient div_coeff(fun.GetDivergence());

      real_t rate_norm = ComputeLpNorm(2.0, div_coeff, mesh, irs);
      REQUIRE(rate_norm == MFEM_Approx(0.0));

      VectorFunctionCoefficient curl_ref_coeff(3,vectorCurl);
      VectorFunctionCoefficient curl_coeff(3,fun.GetCurl());
      VectorSumCoefficient curl_dif_coeff(curl_ref_coeff, curl_coeff, -1.0);

      real_t curl_norm = ComputeLpNorm(2.0, curl_dif_coeff, mesh, irs);
      REQUIRE(curl_norm == MFEM_Approx(0.0));

      MatrixFunctionCoefficient jac_ref_coeff(3,vectorJac);
      MatrixFunctionCoefficient jac_coeff(3,fun.GetJacobian());
      MatrixSumCoefficient jac_dif_coeff(jac_ref_coeff, jac_coeff, -1.0);

      real_t jac_norm = ComputeLpNorm(2.0, jac_dif_coeff, mesh, irs);
      REQUIRE(jac_norm == MFEM_Approx(0.0));
#endif

   }
} // Test Case

void vectorTDFun(const Vector& param, const ADVector& x, const ADFloat t,
                 ADVector& f)
{
   f[0] = x[1] * exp(2*t);
   f[1] = x[0] * exp(2*t);
   f[2] = x[2] * t;
}

void vectorTDFunD(const Vector& x, const real_t t, Vector& f)
{
   f[0] = x[1] * exp(2*t);
   f[1] = x[0] * exp(2*t);
   f[2] = x[2] * t;
}

void vectorTDRate(const Vector& x, const real_t t, Vector& r)
{
   r[0] = x[1] * 2*exp(2*t);
   r[1] = x[0] * 2*exp(2*t);
   r[2] = x[2];
}

void vectorTDJac(const Vector & x, const real_t t, DenseMatrix & j)
{
   real_t amp = exp(2*t);
   j(0,0) = 0.0;
   j(0,1) = amp;
   j(0,2) = 0.0;

   j(1,0) = amp;
   j(1,1) = 0.0;
   j(1,2) = 0.0;

   j(2,0) = 0.0;
   j(2,1) = 0.0;
   j(2,2) = t;
}

void vectorTDCurl(const Vector & x, const real_t t, Vector& curl)
{
   DenseMatrix jac(x.Size());
   vectorTDJac(x,t,jac);

   // Curl
   curl[0] = jac(2,1) - jac(1,2);
   curl[1] = jac(0,2) - jac(2,0);
   curl[2] = jac(1,0) - jac(0,1);
}

TEST_CASE("Autodiff of TD vector function for Coefficient",
          "[AD TD Vector]")
{
   ADVectorTDFunc fun(vectorTDFun);

   SECTION("3D")
   {
      real_t t;
      Vector x(3), ref_sol(3), sol(3);
#ifdef MFEM_USE_CODIPACK
      real_t ref_div, div;
      Vector ref_rate(3), rate(3), ref_curl(3), curl(3);
      DenseMatrix ref_jac(3,3), jac(3,3);
#endif
      for (int i = 0; i < 10; i++)
      {
         for (int j = 0; j < 10; j++)
         {
            for (int k = 0; k < 10; k++)
            {
               for (int l = 0; l < 10; l++)
               {

                  t = 0.1*i;
                  x[0] = 0.1*j;
                  x[1] = 0.1*k;
                  x[2] = 0.1*l;

                  // Solution
                  vectorTDFunD(x,t,ref_sol);

                  fun.Solution(x,t,sol);
                  sol -= ref_sol;
                  REQUIRE(sol.Norml2() == MFEM_Approx(0.0));
#ifdef MFEM_USE_CODIPACK
                  // Rate
                  vectorTDRate(x,t,ref_rate);

                  fun.Rate(x,t,rate);
                  rate -= ref_rate;
                  REQUIRE(rate.Norml2() == MFEM_Approx(0.0));

                  // Jacobian
                  vectorTDJac(x,t,ref_jac);

                  fun.Jacobian(x,t,jac);
                  jac -= ref_jac;
                  REQUIRE(jac.FNorm() == MFEM_Approx(0.0));

                  // Curl
                  ref_curl = 0.0;
                  fun.Curl(x,t,curl);
                  curl -= ref_curl;
                  REQUIRE(curl.Norml2() == MFEM_Approx(0.0));

                  // Divergence
                  ref_div = ref_jac(0,0) + ref_jac(1,1) + ref_jac(2,2);

                  div = fun.Divergence(x,t);
                  REQUIRE(div-ref_div == MFEM_Approx(0.0));
#endif
               }
            }
         }
      }
   }

   SECTION("Coefficient")
   {
      Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2,Element::HEXAHEDRON);
      int order_quad = 6;
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      VectorFunctionCoefficient sol_ref_coeff(3,vectorTDFunD);
      VectorFunctionCoefficient sol_coeff(3,fun.GetSolution());
      VectorSumCoefficient sol_dif_coeff(sol_ref_coeff, sol_coeff, -1.0);

      for (int i = 0; i < 10; i++)
      {
         real_t t = 0.1*i;
         sol_ref_coeff.SetTime(t);
         sol_coeff.SetTime(t);
         real_t sol_norm = ComputeLpNorm(2.0, sol_dif_coeff, mesh, irs);
         REQUIRE(sol_norm == MFEM_Approx(0.0));
      }

#ifdef MFEM_USE_CODIPACK
      FunctionCoefficient div_coeff(fun.GetDivergence());
      for (int i = 0; i < 10; i++)
      {
         real_t t = 0.1*i;
         sol_ref_coeff.SetTime(t);
         sol_coeff.SetTime(t);
         real_t rate_norm = ComputeLpNorm(2.0, div_coeff, mesh, irs);
         REQUIRE(rate_norm == MFEM_Approx(0.0));
      }

      VectorFunctionCoefficient curl_ref_coeff(3,vectorTDCurl);
      VectorFunctionCoefficient curl_coeff(3,fun.GetCurl());
      VectorSumCoefficient curl_dif_coeff(curl_ref_coeff, curl_coeff, -1.0);
      for (int i = 0; i < 10; i++)
      {
         real_t t = 0.1*i;
         curl_ref_coeff.SetTime(t);
         curl_coeff.SetTime(t);
         real_t curl_norm = ComputeLpNorm(2.0, curl_dif_coeff, mesh, irs);
         REQUIRE(curl_norm == MFEM_Approx(0.0));
      }

      MatrixFunctionCoefficient jac_ref_coeff(3,vectorTDJac);
      MatrixFunctionCoefficient jac_coeff(3,fun.GetJacobian());
      MatrixSumCoefficient jac_dif_coeff(jac_ref_coeff, jac_coeff, -1.0);
      for (int i = 0; i < 10; i++)
      {
         real_t t = 0.1*i;
         jac_ref_coeff.SetTime(t);
         jac_coeff.SetTime(t);
         real_t jac_norm = ComputeLpNorm(2.0, jac_dif_coeff, mesh, irs);
         REQUIRE(jac_norm == MFEM_Approx(0.0));
      }
#endif
   }
} // Test Case

} // namespace autodiff

