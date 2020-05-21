// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "catch.hpp"

using namespace mfem;

namespace bilininteg_2d
{

double f2(const Vector & x) { return 2.345 * x[0] + 3.579 * x[1]; }
void F2(const Vector & x, Vector & v)
{
   v.SetSize(2);
   v[0] = 1.234 * x[0] - 2.357 * x[1];
   v[1] = 3.572 * x[0] + 4.321 * x[1];
}

double q2(const Vector & x) { return 4.234 * x[0] + 3.357 * x[1]; }
void V2(const Vector & x, Vector & v)
{
   v.SetSize(2);
   v[0] = 2.234 * x[0] + 1.357 * x[1];
   v[1] = 4.572 * x[0] + 3.321 * x[1];
}
void M2(const Vector & x, DenseMatrix & m)
{
   m.SetSize(2);

   m(0,0) =  4.234 * x[0] + 3.357 * x[1];
   m(0,1) =  0.234 * x[0] + 0.357 * x[1];

   m(1,0) = -0.572 * x[0] - 0.321 * x[1];
   m(1,1) =  4.537 * x[0] + 1.321 * x[1];
}
void MT2(const Vector & x, DenseMatrix & m)
{
   M2(x, m); m.Transpose();
}

double qf2(const Vector & x) { return q2(x) * f2(x); }
void   qF2(const Vector & x, Vector & v) { F2(x, v); v *= q2(x); }
void   MF2(const Vector & x, Vector & v)
{
   DenseMatrix M(2);  M2(x, M);
   Vector F(2);       F2(x, F);
   v.SetSize(2);  M.Mult(F, v);
}
void   DF2(const Vector & x, Vector & v)
{
   Vector D(2);  V2(x, D);
   Vector F(2);  F2(x, v);
   v[0] *= D[0]; v[1] *= D[1];
}

void Grad_f2(const Vector & x, Vector & df)
{
   df.SetSize(2);
   df[0] = 2.345;
   df[1] = 3.579;
}

void Curl_zf2(const Vector & x, Vector & df)
{
   Grad_f2(x, df);
   double df0 = df[0];
   df[0] = df[1];
   df[1] = -df0;
}

double CurlF2(const Vector & x) { return 3.572 + 2.357; }
double DivF2(const Vector & x) { return 1.234 + 4.321; }

void qGrad_f2(const Vector & x, Vector & df)
{
   Grad_f2(x, df);
   df *= q2(x);
}
void DGrad_f2(const Vector & x, Vector & df)
{
   Vector D(2);  V2(x, D);
   Grad_f2(x, df); df[0] *= D[0]; df[1] *= D[1];
}
void MGrad_f2(const Vector & x, Vector & df)
{
   DenseMatrix M(2);  M2(x, M);
   Vector gradf(2);  Grad_f2(x, gradf);
   M.Mult(gradf, df);
}

double qCurlF2(const Vector & x)
{
   return CurlF2(x) * q2(x);
}

double qDivF2(const Vector & x)
{
   return q2(x) * DivF2(x);
}

void Vf2(const Vector & x, Vector & vf) { V2(x, vf); vf *= f2(x); }

double VdotF2(const Vector & x)
{
   Vector v;
   Vector f;
   V2(x, v);
   F2(x, f);
   return v * f;
}

double VdotGrad_f2(const Vector & x)
{
   Vector v;     V2(x, v);
   Vector gradf; Grad_f2(x, gradf);
   return v * gradf;
}

void Vcross_f2(const Vector & x, Vector & Vf)
{
   Vector v; V2(x, v);
   Vf.SetSize(2);
   Vf[0] = v[1]; Vf[1] = -v[0]; Vf *= f2(x);
}

double VcrossF2(const Vector & x)
{
   Vector v; V2(x, v);
   Vector f; F2(x, f);
   return v(0) * f(1) - v(1) * f(0);
}

double VcrossGrad_f2(const Vector & x)
{
   Vector  V; V2(x, V);
   Vector dF; Grad_f2(x, dF);
   return V(0) * dF(1) - V(1) * dF(0);
}

void VcrossCurlF2(const Vector & x, Vector & VF)
{
   Vector  V; V2(x, V);
   VF.SetSize(2);
   VF(0) = V(1); VF(1) = - V(0); VF *= CurlF2(x);
}

void VDivF2(const Vector & x, Vector & VF)
{
   V2(x, VF); VF *= DivF2(x);
}

TEST_CASE("2D Bilinear Mass Integrators",
          "[MixedScalarMassIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient f2_coef(f2);
   FunctionCoefficient q2_coef(q2);
   FunctionCoefficient qf2_coef(qf2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f2_coef);

      SECTION("Mapping H1 to L2")
      {
         L2_FECollection    fec_l2(order, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_l2);
            blf.AddDomainIntegrator(new MixedScalarMassIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(f2_coef) < tol );

            MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
            blfw.AddDomainIntegrator(new MixedScalarMassIntegrator());
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
            REQUIRE( diff->MaxNorm() < tol );
         }
         SECTION("With Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_l2);
            blf.AddDomainIntegrator(new MixedScalarMassIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(qf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
            blfw.AddDomainIntegrator(new MixedScalarMassIntegrator(q2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
            REQUIRE( diff->MaxNorm() < tol );
         }
      }
      SECTION("Mapping H1 to H1")
      {
         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_h1);
            blf.AddDomainIntegrator(new MixedScalarMassIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(f2_coef) < tol );
         }
         SECTION("With Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_h1);
            blf.AddDomainIntegrator(new MixedScalarMassIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(qf2_coef) < tol );
         }
      }
   }
   SECTION("Operators on L2")
   {
      L2_FECollection    fec_l2(order, dim);
      FiniteElementSpace fespace_l2(&mesh, &fec_l2);

      GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(f2_coef);

      SECTION("Mapping L2 to L2")
      {
         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_l2, &fespace_l2);
            blf.AddDomainIntegrator(new MixedScalarMassIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_l2,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(f2_coef) < tol );
         }
         SECTION("With Coefficient")
         {
            MixedBilinearForm blf(&fespace_l2, &fespace_l2);
            blf.AddDomainIntegrator(new MixedScalarMassIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_l2,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(qf2_coef) < tol );
         }
      }
      SECTION("Mapping L2 to H1")
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_l2, &fespace_h1);
            blf.AddDomainIntegrator(new MixedScalarMassIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_l2,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(f2_coef) < tol );

            MixedBilinearForm blfw(&fespace_h1, &fespace_l2);
            blfw.AddDomainIntegrator(new MixedScalarMassIntegrator());
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
            REQUIRE( diff->MaxNorm() < tol );
         }
         SECTION("With Coefficient")
         {
            MixedBilinearForm blf(&fespace_l2, &fespace_h1);
            blf.AddDomainIntegrator(new MixedScalarMassIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_l2,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(qf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_h1, &fespace_l2);
            blfw.AddDomainIntegrator(new MixedScalarMassIntegrator(q2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
            REQUIRE( diff->MaxNorm() < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Vector Mass Integrators",
          "[MixedVectorMassIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient  F2_coef(dim, F2);
   FunctionCoefficient        q2_coef(q2);
   VectorFunctionCoefficient  D2_coef(dim, V2);
   MatrixFunctionCoefficient  M2_coef(dim, M2);
   MatrixFunctionCoefficient MT2_coef(dim, MT2);
   VectorFunctionCoefficient qF2_coef(dim, qF2);
   VectorFunctionCoefficient DF2_coef(dim, DF2);
   VectorFunctionCoefficient MF2_coef(dim, MF2);

   SECTION("Operators on ND")
   {
      ND_FECollection    fec_nd(order, dim);
      FiniteElementSpace fespace_nd(&mesh, &fec_nd);

      GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F2_coef);

      SECTION("Mapping ND to RT")
      {
         {
            // Tests requiring an RT space with same order of
            // convergence as the ND space
            RT_FECollection    fec_rt(order - 1, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(F2_coef) < tol );

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(new MixedVectorMassIntegrator());
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MixedBilinearForm blfv(&fespace_nd, &fespace_rt);
               blfv.AddDomainIntegrator(new VectorFEMassIntegrator());
               blfv.Assemble();
               blfv.Finalize();

               SparseMatrix * diffv = Add(1.0,blf.SpMat(),-1.0,blfv.SpMat());

               REQUIRE( diffv->MaxNorm() < tol );

               delete diffv;
            }
         }
         {
            // Tests requiring a higher order RT space
            RT_FECollection    fec_rt(order, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(q2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(qF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(new MixedVectorMassIntegrator(q2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(D2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(DF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorMassIntegrator(D2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(M2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(MF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorMassIntegrator(MT2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
         }
      }
      SECTION("Mapping ND to ND")
      {
         {
            // Tests requiring an ND test space with same order of
            // convergence as the ND trial space
            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(F2_coef) < tol );

               MixedBilinearForm blfw(&fespace_nd, &fespace_nd);
               blfw.AddDomainIntegrator(new MixedVectorMassIntegrator());
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MixedBilinearForm blfv(&fespace_nd, &fespace_nd);
               blfv.AddDomainIntegrator(new VectorFEMassIntegrator());
               blfv.Assemble();
               blfv.Finalize();

               SparseMatrix * diffv = Add(1.0,blf.SpMat(),-1.0,blfv.SpMat());

               REQUIRE( diffv->MaxNorm() < tol );

               delete diffv;
            }
         }
         {
            // Tests requiring a higher order ND space
            ND_FECollection    fec_ndp(order+1, dim);
            FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

            BilinearForm m_ndp(&fespace_ndp);
            m_ndp.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_ndp.Assemble();
            m_ndp.Finalize();

            GridFunction g_ndp(&fespace_ndp);

            Vector tmp_ndp(fespace_ndp.GetNDofs());

            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(q2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
               CG(m_ndp, tmp_ndp, g_ndp, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_ndp.ComputeL2Error(qF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
               blfw.AddDomainIntegrator(new MixedVectorMassIntegrator(q2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(D2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
               CG(m_ndp, tmp_ndp, g_ndp, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_ndp.ComputeL2Error(DF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorMassIntegrator(D2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(M2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
               CG(m_ndp, tmp_ndp, g_ndp, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_ndp.ComputeL2Error(MF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorMassIntegrator(MT2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
         }
      }
   }
   SECTION("Operators on RT")
   {
      RT_FECollection    fec_rt(order - 1, dim);
      FiniteElementSpace fespace_rt(&mesh, &fec_rt);

      GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F2_coef);

      SECTION("Mapping RT to ND")
      {
         {
            // Tests requiring an ND test space with same order of
            // convergence as the RT trial space
            ND_FECollection    fec_nd(order, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_nd);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(F2_coef) < tol );

               MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
               blfw.AddDomainIntegrator(new MixedVectorMassIntegrator());
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MixedBilinearForm blfv(&fespace_rt, &fespace_nd);
               blfv.AddDomainIntegrator(new VectorFEMassIntegrator());
               blfv.Assemble();
               blfv.Finalize();

               SparseMatrix * diffv = Add(1.0,blf.SpMat(),-1.0,blfv.SpMat());

               REQUIRE( diffv->MaxNorm() < tol );

               delete diffv;
            }
         }
         {
            // Tests requiring a higher order ND space
            ND_FECollection    fec_nd(order + 1, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_nd);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(q2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(qF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
               blfw.AddDomainIntegrator(new MixedVectorMassIntegrator(q2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_nd);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(D2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(DF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedVectorMassIntegrator(D2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_nd);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(M2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(MF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedVectorMassIntegrator(MT2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
         }
      }
      SECTION("Mapping RT to RT")
      {
         {
            // Tests requiring an RT test space with same order of
            // convergence as the RT trial space
            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_rt);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(F2_coef) < tol );

               MixedBilinearForm blfw(&fespace_rt, &fespace_rt);
               blfw.AddDomainIntegrator(new MixedVectorMassIntegrator());
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MixedBilinearForm blfv(&fespace_rt, &fespace_rt);
               blfv.AddDomainIntegrator(new VectorFEMassIntegrator());
               blfv.Assemble();
               blfv.Finalize();

               SparseMatrix * diffv = Add(1.0,blf.SpMat(),-1.0,blfv.SpMat());

               REQUIRE( diffv->MaxNorm() < tol );

               delete diffv;
            }
         }
         {
            // Tests requiring a higher order RT space
            RT_FECollection    fec_rtp(order, dim);
            FiniteElementSpace fespace_rtp(&mesh, &fec_rtp);

            BilinearForm m_rtp(&fespace_rtp);
            m_rtp.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rtp.Assemble();
            m_rtp.Finalize();

            GridFunction g_rtp(&fespace_rtp);

            Vector tmp_rtp(fespace_rtp.GetNDofs());

            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(q2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
               CG(m_rtp, tmp_rtp, g_rtp, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rtp.ComputeL2Error(qF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
               blfw.AddDomainIntegrator(new MixedVectorMassIntegrator(q2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(D2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
               CG(m_rtp, tmp_rtp, g_rtp, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rtp.ComputeL2Error(DF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedVectorMassIntegrator(D2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
               blf.AddDomainIntegrator(new MixedVectorMassIntegrator(M2_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
               CG(m_rtp, tmp_rtp, g_rtp, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rtp.ComputeL2Error(MF2_coef) < tol );

               MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedVectorMassIntegrator(MT2_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
         }
      }
   }
}

TEST_CASE("2D Bilinear Gradient Integrator",
          "[MixedVectorGradientIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient         f2_coef(f2);
   FunctionCoefficient         q2_coef(q2);
   VectorFunctionCoefficient   D2_coef(dim, V2);
   MatrixFunctionCoefficient   M2_coef(dim, M2);
   VectorFunctionCoefficient  df2_coef(dim, Grad_f2);
   VectorFunctionCoefficient qdf2_coef(dim, qGrad_f2);
   VectorFunctionCoefficient Ddf2_coef(dim, DGrad_f2);
   VectorFunctionCoefficient Mdf2_coef(dim, MGrad_f2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f2_coef);

      SECTION("Mapping H1 to ND")
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         BilinearForm m_nd(&fespace_nd);
         m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_nd.Assemble();
         m_nd.Finalize();

         GridFunction g_nd(&fespace_nd);

         Vector tmp_nd(fespace_nd.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(new MixedVectorGradientIntegrator());
            blf.Assemble();
            blf.Finalize();
            g_nd.ProjectCoefficient(df2_coef);

            blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
            CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_nd.ComputeL2Error(df2_coef) < tol );
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
            CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_nd.ComputeL2Error(qdf2_coef) < tol );
         }
         SECTION("With Diagonal Matrix Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(D2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
            CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_nd.ComputeL2Error(Ddf2_coef) < tol );
         }
         SECTION("With Matrix Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(M2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
            CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_nd.ComputeL2Error(Mdf2_coef) < tol );
         }
      }
      SECTION("Mapping H1 to RT")
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         BilinearForm m_rt(&fespace_rt);
         m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_rt.Assemble();
         m_rt.Finalize();

         GridFunction g_rt(&fespace_rt);

         Vector tmp_rt(fespace_rt.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(new MixedVectorGradientIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(df2_coef) < tol );
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(qdf2_coef) < tol );
         }
         SECTION("With Diagonal Matrix Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(D2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(Ddf2_coef) < tol );
         }
         SECTION("With Matrix Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(M2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(Mdf2_coef) < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Scalar Curl Integrator",
          "[MixedScalarCurlIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient   F2_coef(dim, F2);
   FunctionCoefficient         q2_coef(q2);
   FunctionCoefficient        dF2_coef(CurlF2);
   FunctionCoefficient       qdF2_coef(qCurlF2);

   SECTION("Operators on ND")
   {
      ND_FECollection    fec_nd(order, dim);
      FiniteElementSpace fespace_nd(&mesh, &fec_nd);

      GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F2_coef);

      SECTION("Mapping ND to L2")
      {
         L2_FECollection    fec_l2(order - 1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_l2);
            blf.AddDomainIntegrator(new MixedScalarCurlIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(dF2_coef) < tol );
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedScalarCurlIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(qdF2_coef) < tol );
         }
      }
      SECTION("Mapping ND to H1")
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_h1);
            blf.AddDomainIntegrator(new MixedScalarCurlIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(dF2_coef) < tol );
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedScalarCurlIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(qdF2_coef) < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Scalar Cross Product Gradient Integrator",
          "[MixedScalarCrossGradIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient          f2_coef(f2);
   VectorFunctionCoefficient    V2_coef(dim, V2);
   FunctionCoefficient       Vxdf2_coef(VcrossGrad_f2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f2_coef);

      SECTION("Mapping H1 to L2")
      {
         L2_FECollection    fec_l2(order, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedScalarCrossGradIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(Vxdf2_coef) < tol );
         }
      }
      SECTION("Mapping H1 to H1")
      {
         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedScalarCrossGradIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(Vxdf2_coef) < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Divergence Integrator",
          "[MixedScalarDivergenceIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient F2_coef(dim, F2);
   FunctionCoefficient       q2_coef(q2);
   FunctionCoefficient      dF2_coef(DivF2);
   FunctionCoefficient     qdF2_coef(qDivF2);

   SECTION("Operators on RT")
   {
      RT_FECollection    fec_rt(order - 1, dim);
      FiniteElementSpace fespace_rt(&mesh, &fec_rt);

      GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F2_coef);

      SECTION("Mapping RT to L2")
      {
         L2_FECollection    fec_l2(order - 1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_l2);
            blf.AddDomainIntegrator(new MixedScalarDivergenceIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(dF2_coef) < tol );
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedScalarDivergenceIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(qdF2_coef) < tol );
         }
      }
      SECTION("Mapping RT to H1")
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_h1);
            blf.AddDomainIntegrator(new MixedScalarDivergenceIntegrator());
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(dF2_coef) < tol );
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedScalarDivergenceIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(qdF2_coef) < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Vector Divergence Integrator",
          "[MixedVectorDivergenceIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient   F2_coef(dim, F2);
   VectorFunctionCoefficient   V2_coef(dim, V2);
   VectorFunctionCoefficient VdF2_coef(dim, VDivF2);

   SECTION("Operators on RT")
   {
      RT_FECollection    fec_rt(order - 1, dim);
      FiniteElementSpace fespace_rt(&mesh, &fec_rt);

      GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F2_coef);

      SECTION("Mapping RT to RT")
      {
         BilinearForm m_rt(&fespace_rt);
         m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_rt.Assemble();
         m_rt.Finalize();

         GridFunction g_rt(&fespace_rt);

         Vector tmp_rt(fespace_rt.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorDivergenceIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(VdF2_coef) < tol );
         }
      }
      SECTION("Mapping RT to ND")
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         BilinearForm m_nd(&fespace_nd);
         m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_nd.Assemble();
         m_nd.Finalize();

         GridFunction g_nd(&fespace_nd);

         Vector tmp_nd(fespace_nd.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorDivergenceIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
            CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_nd.ComputeL2Error(VdF2_coef) < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Vector Product Integrators",
          "[MixedVectorProductIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient        f2_coef(f2);
   VectorFunctionCoefficient  V2_coef(dim, V2);
   VectorFunctionCoefficient Vf2_coef(dim, Vf2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f2_coef);

      SECTION("Mapping H1 to ND")
      {
         ND_FECollection    fec_nd(order + 1, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         BilinearForm m_nd(&fespace_nd);
         m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_nd.Assemble();
         m_nd.Finalize();

         GridFunction g_nd(&fespace_nd);

         Vector tmp_nd(fespace_nd.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(new MixedVectorProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
            CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_nd.ComputeL2Error(Vf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedDotProductIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
      SECTION("Mapping H1 to RT")
      {
         RT_FECollection    fec_rt(order, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         BilinearForm m_rt(&fespace_rt);
         m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_rt.Assemble();
         m_rt.Finalize();

         GridFunction g_rt(&fespace_rt);

         Vector tmp_rt(fespace_rt.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(new MixedVectorProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(Vf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedDotProductIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
   SECTION("Operators on L2")
   {
      L2_FECollection    fec_l2(order, dim);
      FiniteElementSpace fespace_l2(&mesh, &fec_l2);

      GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(f2_coef);

      SECTION("Mapping L2 to ND")
      {
         ND_FECollection    fec_nd(order + 1, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         BilinearForm m_nd(&fespace_nd);
         m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_nd.Assemble();
         m_nd.Finalize();

         GridFunction g_nd(&fespace_nd);

         Vector tmp_nd(fespace_nd.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_l2, &fespace_nd);
            blf.AddDomainIntegrator(new MixedVectorProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_l2,tmp_nd); g_nd = 0.0;
            CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_nd.ComputeL2Error(Vf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_nd, &fespace_l2);
            blfw.AddDomainIntegrator(
               new MixedDotProductIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
      SECTION("Mapping L2 to RT")
      {
         RT_FECollection    fec_rt(order, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         BilinearForm m_rt(&fespace_rt);
         m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_rt.Assemble();
         m_rt.Finalize();

         GridFunction g_rt(&fespace_rt);

         Vector tmp_rt(fespace_rt.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_l2, &fespace_rt);
            blf.AddDomainIntegrator(new MixedVectorProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_l2,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(Vf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_rt, &fespace_l2);
            blfw.AddDomainIntegrator(
               new MixedDotProductIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
}

TEST_CASE("2D Bilinear Scalar Cross Product Integrators",
          "[MixedScalarCrossProductIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient   F2_coef(dim, F2);
   VectorFunctionCoefficient   V2_coef(dim, V2);
   FunctionCoefficient       VxF2_coef(VcrossF2);

   SECTION("Operators on ND")
   {
      ND_FECollection    fec_nd(order, dim);
      FiniteElementSpace fespace_nd(&mesh, &fec_nd);

      GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F2_coef);

      SECTION("Mapping ND to H1")
      {
         H1_FECollection    fec_h1p(order + 1, dim);
         FiniteElementSpace fespace_h1p(&mesh, &fec_h1p);

         BilinearForm m_h1p(&fespace_h1p);
         m_h1p.AddDomainIntegrator(new MassIntegrator());
         m_h1p.Assemble();
         m_h1p.Finalize();

         GridFunction g_h1p(&fespace_h1p);

         Vector tmp_h1p(fespace_h1p.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_h1p);
            blf.AddDomainIntegrator(
               new MixedScalarCrossProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_h1p); g_h1p = 0.0;
            CG(m_h1p, tmp_h1p, g_h1p, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1p.ComputeL2Error(VxF2_coef) < tol );
         }
      }
      SECTION("Mapping ND to L2")
      {
         L2_FECollection    fec_l2(order, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedScalarCrossProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(VxF2_coef) < tol );
         }
      }
   }
   SECTION("Operators on RT")
   {
      RT_FECollection    fec_rt(order - 1, dim);
      FiniteElementSpace fespace_rt(&mesh, &fec_rt);

      GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F2_coef);

      SECTION("Mapping RT to H1")
      {
         H1_FECollection    fec_h1(order + 1, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedScalarCrossProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(VxF2_coef) < tol );
         }
      }
      SECTION("Mapping RT to L2")
      {
         L2_FECollection    fec_l2p(order, dim);
         FiniteElementSpace fespace_l2p(&mesh, &fec_l2p);

         BilinearForm m_l2p(&fespace_l2p);
         m_l2p.AddDomainIntegrator(new MassIntegrator());
         m_l2p.Assemble();
         m_l2p.Finalize();

         GridFunction g_l2p(&fespace_l2p);

         Vector tmp_l2p(fespace_l2p.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_l2p);
            blf.AddDomainIntegrator(
               new MixedScalarCrossProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_l2p); g_l2p = 0.0;
            CG(m_l2p, tmp_l2p, g_l2p, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2p.ComputeL2Error(VxF2_coef) < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Scalar Weak Cross Product Integrators",
          "[MixedScalarWeakCrossProductIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient         f2_coef(f2);
   VectorFunctionCoefficient   V2_coef(dim, V2);
   VectorFunctionCoefficient Vxf2_coef(dim, Vcross_f2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f2_coef);

      SECTION("Mapping H1 to ND")
      {
         ND_FECollection    fec_ndp(order + 1, dim);
         FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

         BilinearForm m_ndp(&fespace_ndp);
         m_ndp.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_ndp.Assemble();
         m_ndp.Finalize();

         GridFunction g_ndp(&fespace_ndp);

         Vector tmp_ndp(fespace_ndp.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_ndp);
            blf.AddDomainIntegrator(
               new MixedScalarWeakCrossProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_ndp); g_ndp = 0.0;
            CG(m_ndp, tmp_ndp, g_ndp, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_ndp.ComputeL2Error(Vxf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_ndp, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedScalarCrossProductIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
      SECTION("Mapping H1 to RT")
      {
         RT_FECollection    fec_rt(order, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         BilinearForm m_rt(&fespace_rt);
         m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_rt.Assemble();
         m_rt.Finalize();

         GridFunction g_rt(&fespace_rt);

         Vector tmp_rt(fespace_rt.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedScalarWeakCrossProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(Vxf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedScalarCrossProductIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
   SECTION("Operators on L2")
   {
      L2_FECollection    fec_l2(order - 1, dim);
      FiniteElementSpace fespace_l2(&mesh, &fec_l2);

      GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(f2_coef);

      SECTION("Mapping L2 to ND")
      {
         ND_FECollection    fec_ndp(order + 1, dim);
         FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

         BilinearForm m_ndp(&fespace_ndp);
         m_ndp.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_ndp.Assemble();
         m_ndp.Finalize();

         GridFunction g_ndp(&fespace_ndp);

         Vector tmp_ndp(fespace_ndp.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_l2, &fespace_ndp);
            blf.AddDomainIntegrator(
               new MixedScalarWeakCrossProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_l2,tmp_ndp); g_ndp = 0.0;
            CG(m_ndp, tmp_ndp, g_ndp, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_ndp.ComputeL2Error(Vxf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_ndp, &fespace_l2);
            blfw.AddDomainIntegrator(
               new MixedScalarCrossProductIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
      SECTION("Mapping L2 to RT")
      {
         RT_FECollection    fec_rt(order, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         BilinearForm m_rt(&fespace_rt);
         m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_rt.Assemble();
         m_rt.Finalize();

         GridFunction g_rt(&fespace_rt);

         Vector tmp_rt(fespace_rt.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_l2, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedScalarWeakCrossProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_l2,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(Vxf2_coef) < tol );

            MixedBilinearForm blfw(&fespace_rt, &fespace_l2);
            blfw.AddDomainIntegrator(
               new MixedScalarCrossProductIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
}

TEST_CASE("2D Bilinear Scalar Weak Curl Integrators",
          "[MixedScalarWeakCurlIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient q2_coef(q2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      SECTION("Mapping H1 to ND")
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedScalarCurlIntegrator());
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_h1, &fespace_nd);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakCurlIntegrator());
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedScalarCurlIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_h1, &fespace_nd);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakCurlIntegrator(q2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
   SECTION("Operators on L2")
   {
      L2_FECollection    fec_l2(order - 1, dim);
      FiniteElementSpace fespace_l2(&mesh, &fec_l2);

      SECTION("Mapping L2 to ND")
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedScalarCurlIntegrator());
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_l2, &fespace_nd);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakCurlIntegrator());
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedScalarCurlIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_l2, &fespace_nd);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakCurlIntegrator(q2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
}

TEST_CASE("2D Bilinear Scalar Weak Gradient Integrators",
          "[MixedScalarWeakGradientIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient q2_coef(q2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      SECTION("Mapping H1 to RT")
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedScalarDivergenceIntegrator());
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_h1, &fespace_rt);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakGradientIntegrator());
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedScalarDivergenceIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_h1, &fespace_rt);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakGradientIntegrator(q2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
   SECTION("Operators on L2")
   {
      L2_FECollection    fec_l2(order - 1, dim);
      FiniteElementSpace fespace_l2(&mesh, &fec_l2);

      SECTION("Mapping L2 to RT")
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedScalarDivergenceIntegrator());
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_l2, &fespace_rt);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakGradientIntegrator());
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedScalarDivergenceIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_l2, &fespace_rt);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakGradientIntegrator(q2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
}

TEST_CASE("2D Bilinear Directional Derivative Integrators",
          "[MixedDirectionalDerivativeIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient         f2_coef(f2);
   VectorFunctionCoefficient   V2_coef(dim, V2);
   FunctionCoefficient       Vdf2_coef(VdotGrad_f2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f2_coef);

      SECTION("Mapping H1 to ND")
      {
         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedDirectionalDerivativeIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(Vdf2_coef) < tol );
         }
      }
      SECTION("Mapping H1 to L2")
      {
         L2_FECollection    fec_l2(order - 1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedDirectionalDerivativeIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_h1,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(Vdf2_coef) < tol );
         }
      }
   }
}


TEST_CASE("2D Bilinear Scalar Weak Divergence Integrators",
          "[MixedScalarWeakDivergenceIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient  V2_coef(dim, V2);

   SECTION("Operators on H1")
   {
      H1_FECollection    fec_h1(order, dim);
      FiniteElementSpace fespace_h1(&mesh, &fec_h1);

      SECTION("Mapping H1 to H1")
      {
         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedDirectionalDerivativeIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_h1, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakDivergenceIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
   SECTION("Operators on L2")
   {
      L2_FECollection    fec_l2(order - 1, dim);
      FiniteElementSpace fespace_l2(&mesh, &fec_l2);

      SECTION("Mapping L2 to H1")
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedDirectionalDerivativeIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakDivergenceIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
}


TEST_CASE("2D Bilinear Vector Weak Divergence Integrators",
          "[MixedVectorWeakDivergenceIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   FunctionCoefficient        q2_coef(q2);
   VectorFunctionCoefficient  D2_coef(dim, V2);
   MatrixFunctionCoefficient  M2_coef(dim, M2);
   MatrixFunctionCoefficient MT2_coef(dim, MT2);

   SECTION("Operators on ND")
   {
      ND_FECollection    fec_nd(order, dim);
      FiniteElementSpace fespace_nd(&mesh, &fec_nd);

      SECTION("Mapping ND to H1")
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator());
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedVectorWeakDivergenceIntegrator());
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedVectorWeakDivergenceIntegrator(q2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Diagonal Matrix Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(D2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedVectorWeakDivergenceIntegrator(D2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Matrix Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(MT2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedVectorWeakDivergenceIntegrator(M2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
   SECTION("Operators on RT")
   {
      RT_FECollection    fec_rt(order - 1, dim);
      FiniteElementSpace fespace_rt(&mesh, &fec_rt);

      SECTION("Mapping RT to H1")
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         SECTION("Without Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator());
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedVectorWeakDivergenceIntegrator());
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Scalar Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(q2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedVectorWeakDivergenceIntegrator(q2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Diagonal Matrix Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(D2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedVectorWeakDivergenceIntegrator(D2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
         SECTION("With Matrix Coefficient")
         {
            MixedBilinearForm blf(&fespace_h1, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorGradientIntegrator(MT2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
            blfw.AddDomainIntegrator(
               new MixedVectorWeakDivergenceIntegrator(M2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
}


TEST_CASE("2D Bilinear Dot Product Integrators",
          "[MixedDotProductIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient  F2_coef(dim, F2);
   VectorFunctionCoefficient  V2_coef(dim, V2);
   FunctionCoefficient       VF2_coef(VdotF2);

   SECTION("Operators on ND")
   {
      ND_FECollection    fec_nd(order, dim);
      FiniteElementSpace fespace_nd(&mesh, &fec_nd);

      GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F2_coef);

      SECTION("Mapping ND to H1")
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedDotProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(VF2_coef) < tol );
         }
      }
      SECTION("Mapping ND to L2")
      {
         L2_FECollection    fec_l2(order, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedDotProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(VF2_coef) < tol );
         }
      }
   }
   SECTION("Operators on RT")
   {
      RT_FECollection    fec_rt(order - 1, dim);
      FiniteElementSpace fespace_rt(&mesh, &fec_rt);

      GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F2_coef);

      SECTION("Mapping RT to H1")
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         BilinearForm m_h1(&fespace_h1);
         m_h1.AddDomainIntegrator(new MassIntegrator());
         m_h1.Assemble();
         m_h1.Finalize();

         GridFunction g_h1(&fespace_h1);

         Vector tmp_h1(fespace_h1.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_h1);
            blf.AddDomainIntegrator(
               new MixedDotProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_h1); g_h1 = 0.0;
            CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_h1.ComputeL2Error(VF2_coef) < tol );
         }
      }
      SECTION("Mapping RT to L2")
      {
         L2_FECollection    fec_l2(order, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         BilinearForm m_l2(&fespace_l2);
         m_l2.AddDomainIntegrator(new MassIntegrator());
         m_l2.Assemble();
         m_l2.Finalize();

         GridFunction g_l2(&fespace_l2);

         Vector tmp_l2(fespace_l2.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_l2);
            blf.AddDomainIntegrator(
               new MixedDotProductIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_rt,tmp_l2); g_l2 = 0.0;
            CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_l2.ComputeL2Error(VF2_coef) < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Weak Gradient Dot Product Integrators",
          "[MixedWeakGradDotIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient  V2_coef(dim, V2);

   SECTION("Operators on ND")
   {
      ND_FECollection    fec_nd(order, dim);
      FiniteElementSpace fespace_nd(&mesh, &fec_nd);

      SECTION("Mapping ND to RT")
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedVectorDivergenceIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
            blfw.AddDomainIntegrator(
               new MixedWeakGradDotIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
   SECTION("Operators on RT")
   {
      RT_FECollection    fec_rt(order - 1, dim);
      FiniteElementSpace fespace_rt(&mesh, &fec_rt);

      SECTION("Mapping RT to RT")
      {
         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_rt, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedVectorDivergenceIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_rt, &fespace_rt);
            blfw.AddDomainIntegrator(
               new MixedWeakGradDotIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
}

TEST_CASE("2D Bilinear Scalar Cross Product Curl Integrator",
          "[MixedScalarCrossCurlIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient    F2_coef(dim, F2);
   VectorFunctionCoefficient    V2_coef(dim, V2);
   VectorFunctionCoefficient VxdF2_coef(dim, VcrossCurlF2);

   SECTION("Operators on ND")
   {
      ND_FECollection    fec_nd(order, dim);
      FiniteElementSpace fespace_nd(&mesh, &fec_nd);

      GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F2_coef);

      SECTION("Mapping ND to RT")
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         BilinearForm m_rt(&fespace_rt);
         m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_rt.Assemble();
         m_rt.Finalize();

         GridFunction g_rt(&fespace_rt);

         Vector tmp_rt(fespace_rt.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedScalarCrossCurlIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
            CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_rt.ComputeL2Error(VxdF2_coef) < tol );
         }
      }
      SECTION("Mapping ND to ND")
      {
         BilinearForm m_nd(&fespace_nd);
         m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
         m_nd.Assemble();
         m_nd.Finalize();

         GridFunction g_nd(&fespace_nd);

         Vector tmp_nd(fespace_nd.GetNDofs());

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedScalarCrossCurlIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
            CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

            REQUIRE( g_nd.ComputeL2Error(VxdF2_coef) < tol );
         }
      }
   }
}

TEST_CASE("2D Bilinear Scalar Weak Curl Cross Integrators",
          "[MixedScalarWeakCurlCrossIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);

   VectorFunctionCoefficient  V2_coef(dim, V2);

   SECTION("Operators on ND")
   {
      ND_FECollection    fec_nd(order, dim);
      FiniteElementSpace fespace_nd(&mesh, &fec_nd);

      SECTION("Mapping ND to ND")
      {
         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_nd);
            blf.AddDomainIntegrator(
               new MixedScalarCrossCurlIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_nd, &fespace_nd);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakCurlCrossIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
   SECTION("Operators on RT")
   {
      RT_FECollection    fec_rt(order - 1, dim);
      FiniteElementSpace fespace_rt(&mesh, &fec_rt);

      SECTION("Mapping RT to ND")
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         SECTION("With Vector Coefficient")
         {
            MixedBilinearForm blf(&fespace_nd, &fespace_rt);
            blf.AddDomainIntegrator(
               new MixedScalarCrossCurlIntegrator(V2_coef));
            blf.Assemble();
            blf.Finalize();

            MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
            blfw.AddDomainIntegrator(
               new MixedScalarWeakCurlCrossIntegrator(V2_coef));
            blfw.Assemble();
            blfw.Finalize();

            SparseMatrix * blfT = Transpose(blfw.SpMat());
            SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

            REQUIRE( diff->MaxNorm() < tol );

            delete blfT;
            delete diff;
         }
      }
   }
}

} // namespace bilininteg_2d
