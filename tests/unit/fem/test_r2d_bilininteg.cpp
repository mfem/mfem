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
#include "unit_tests.hpp"

using namespace mfem;

namespace bilininteg_p2d
{

double zero3(const Vector & x) { return 0.0; }
void Zero3(const Vector & x, Vector & v) { v.SetSize(3); v = 0.0; }

double f3(const Vector & x)
{ return 2.345 * x[0] + 3.579 * x[1]; }
void F3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] =  1.234 * x[0] - 2.357 * x[1];
   v[1] =  2.537 * x[0] + 4.321 * x[1];
   v[2] = -2.572 * x[0] + 1.321 * x[1];
}

double q3(const Vector & x)
{ return 4.234 * x[0] + 3.357 * x[1]; }

void V3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] = 4.234 * x[0] + 3.357 * x[1];
   v[1] = 4.537 * x[0] + 1.321 * x[1];
   v[2] = 1.572 * x[0] + 2.321 * x[1];
}
void M3(const Vector & x, DenseMatrix & m)
{
   m.SetSize(3);

   m(0,0) =  4.234 * x[0] + 3.357 * x[1];
   m(0,1) =  0.234 * x[0] + 0.357 * x[1];
   m(0,2) = -0.537 * x[0] + 0.321 * x[1];

   m(1,0) = -0.572 * x[0] - 0.321 * x[1];
   m(1,1) =  4.537 * x[0] + 1.321 * x[1];
   m(1,2) =  0.537 * x[0] + 0.321 * x[1];

   m(2,0) =  0.572 * x[0] + 0.321 * x[1];
   m(2,1) =  0.234 * x[0] - 0.357 * x[1];
   m(2,2) =  1.572 * x[0] + 2.321 * x[1];
}
void MT3(const Vector & x, DenseMatrix & m)
{
   M3(x, m); m.Transpose();
}

double qf3(const Vector & x) { return q3(x) * f3(x); }
void   qF3(const Vector & x, Vector & v) { F3(x, v); v *= q3(x); }
void   MF3(const Vector & x, Vector & v)
{
   DenseMatrix M(3);  M3(x, M);
   Vector F(3);       F3(x, F);
   v.SetSize(3);  M.Mult(F, v);
}
void   DF3(const Vector & x, Vector & v)
{
   Vector D(3);  V3(x, D);
   Vector F(3);  F3(x, v);
   v[0] *= D[0]; v[1] *= D[1]; v[2] *= D[2];
}

void Grad_f3(const Vector & x, Vector & df)
{
   df.SetSize(3);
   df[0] = 2.345;
   df[1] = 3.579;
   df[2] = 0.0;
}
void CurlF3(const Vector & x, Vector & df)
{
   df.SetSize(3);
   df[0] = 1.321;
   df[1] = 2.572;
   df[2] = 2.537 + 2.357;
}
double DivF3(const Vector & x)
{ return 1.234 + 4.321; }

void CurlV3(const Vector & x, Vector & dV)
{
   dV.SetSize(3);
   dV[0] =  2.321;
   dV[1] = -1.572;
   dV[2] =  4.537 - 3.357;
}

void qGrad_f3(const Vector & x, Vector & df)
{
   Grad_f3(x, df); df *= q3(x);
}
void DGrad_f3(const Vector & x, Vector & df)
{
   Vector D(3);  V3(x, D);
   Grad_f3(x, df); df[0] *= D[0]; df[1] *= D[1]; df[2] *= D[2];
}
void MGrad_f3(const Vector & x, Vector & df)
{
   DenseMatrix M(3);  M3(x, M);
   Vector gradf(3);  Grad_f3(x, gradf);
   M.Mult(gradf, df);
}

void qCurlF3(const Vector & x, Vector & df)
{
   CurlF3(x, df); df *= q3(x);
}
void DCurlF3(const Vector & x, Vector & df)
{
   Vector D(3);  V3(x, D);
   CurlF3(x, df); df[0] *= D[0]; df[1] *= D[1]; df[2] *= D[2];
}
void MCurlF3(const Vector & x, Vector & df)
{
   DenseMatrix M(3);  M3(x, M);
   Vector curlf(3);  CurlF3(x, curlf);
   M.Mult(curlf, df);
}

double qDivF3(const Vector & x)
{
   return q3(x) * DivF3(x);
}

void Vf3(const Vector & x, Vector & vf)
{
   V3(x, vf); vf *= f3(x);
}

void VcrossF3(const Vector & x, Vector & VF)
{
   Vector V; V3(x, V);
   Vector F; F3(x, F);
   VF.SetSize(3);
   VF(0) = V(1) * F(2) - V(2) * F(1);
   VF(1) = V(2) * F(0) - V(0) * F(2);
   VF(2) = V(0) * F(1) - V(1) * F(0);
}

double VdotF3(const Vector & x)
{
   Vector v; V3(x, v);
   Vector f; F3(x, f);
   return v * f;
}

double VdotGrad_f3(const Vector & x)
{
   Vector v;     V3(x, v);
   Vector gradf; Grad_f3(x, gradf);
   return v * gradf;
}

void VcrossGrad_f3(const Vector & x, Vector & VF)
{
   Vector  V; V3(x, V);
   Vector dF; Grad_f3(x, dF);
   VF.SetSize(3);
   VF(0) = V(1) * dF(2) - V(2) * dF(1);
   VF(1) = V(2) * dF(0) - V(0) * dF(2);
   VF(2) = V(0) * dF(1) - V(1) * dF(0);
}

void VcrossCurlF3(const Vector & x, Vector & VF)
{
   Vector  V; V3(x, V);
   Vector dF; CurlF3(x, dF);
   VF.SetSize(3);
   VF(0) = V(1) * dF(2) - V(2) * dF(1);
   VF(1) = V(2) * dF(0) - V(0) * dF(2);
   VF(2) = V(0) * dF(1) - V(1) * dF(0);
}

void VDivF3(const Vector & x, Vector & VF)
{
   V3(x, VF); VF *= DivF3(x);
}

void Grad_q3(const Vector & x, Vector & dq)
{
   dq.SetSize(3);
   dq[0] = 4.234;
   dq[1] = 3.357;
   dq[2] = 0.0;
}

void Grad_V3(const Vector & x, DenseMatrix & dv)
{
   dv.SetSize(3);
   dv(0,0) = 4.234; dv(0,1) = 3.357; dv(0,2) = 0.0;
   dv(1,0) = 4.537; dv(1,1) = 1.321; dv(1,2) = 0.0;
   dv(2,0) = 1.572; dv(2,1) = 2.321; dv(2,2) = 0.0;
}

double DivV3(const Vector & x)
{ return 4.234 + 1.321; }

void Grad_F3(const Vector & x, DenseMatrix & df)
{
   df.SetSize(3);
   df(0,0) =  1.234; df(0,1) = -2.357; df(0,2) = 0.0;
   df(1,0) =  2.537; df(1,1) =  4.321; df(1,2) = 0.0;
   df(2,0) = -2.572; df(2,1) =  1.321; df(2,2) = 0.0;
}

void Grad_M3(const Vector & x, DenseTensor & dm)
{
   dm.SetSize(3,3,3);
   dm(0,0,0) =  4.234; dm(0,0,1) =  3.357; dm(0,0,2) = 0.0;
   dm(0,1,0) =  0.234; dm(0,1,1) =  0.357; dm(0,1,2) = 0.0;
   dm(0,2,0) = -0.537; dm(0,2,1) =  0.321; dm(0,2,2) = 0.0;

   dm(1,0,0) = -0.572; dm(1,0,1) = -0.321; dm(1,0,2) = 0.0;
   dm(1,1,0) =  4.537; dm(1,1,1) =  1.321; dm(1,1,2) = 0.0;
   dm(1,2,0) =  0.537; dm(1,2,1) =  0.321; dm(1,2,2) = 0.0;

   dm(2,0,0) =  0.572; dm(2,0,1) =  0.321; dm(2,0,2) = 0.0;
   dm(2,1,0) =  0.234; dm(2,1,1) = -0.357; dm(2,1,2) = 0.0;
   dm(2,2,0) =  1.572; dm(2,2,1) =  2.321; dm(2,2,2) = 0.0;
}

void Grad_qf3(const Vector & x, Vector & v)
{
   Vector dq; Grad_q3(x, dq);
   Grad_f3(x, v);
   v *= q3(x);
   v.Add(f3(x), dq);
}


void GradVdotF3(const Vector & x, Vector & dvf)
{
   Vector V; V3(x, V);
   Vector F; F3(x, F);
   DenseMatrix dV; Grad_V3(x, dV);
   DenseMatrix dF; Grad_F3(x, dF);
   dvf.SetSize(3);
   dV.MultTranspose(F, dvf);

   Vector tmp(3);
   dF.MultTranspose(V, tmp);
   dvf += tmp;
}

void Curl_qF3(const Vector & x, Vector & dqF)
{
   Vector dq; Grad_q3(x, dq);
   Vector  F; F3(x, F);
   CurlF3(x, dqF);
   dqF *= q3(x);
   dqF[0] += dq[1]*F[2] - dq[2]*F[1];
   dqF[1] += dq[2]*F[0] - dq[0]*F[2];
   dqF[2] += dq[0]*F[1] - dq[1]*F[0];
}

double Div_qF3(const Vector & x)
{
   Vector dq; Grad_q3(x, dq);
   Vector  F; F3(x, F);
   return dq[0]*F[0] + dq[1]*F[1] + dq[2]*F[2] + q3(x)*DivF3(x);
}

double Div_Vf3(const Vector & x)
{
   Vector  V; V3(x, V);
   Vector df; Grad_f3(x, df);
   return DivV3(x)*f3(x) + V*df;
}

double Div_VcrossF3(const Vector & x)
{
   Vector  V; V3(x, V);
   Vector  F; F3(x, F);
   Vector dV; CurlV3(x, dV);
   Vector dF; CurlF3(x, dF);
   return dV*F - V*dF;
}

double Div_DF3(const Vector & x)
{
   DenseMatrix dV; Grad_V3(x, dV);
   DenseMatrix dF; Grad_F3(x, dF);
   Vector  V; V3(x, V);
   Vector  F; F3(x, F);
   return dV(0,0)*F[0] + dV(1,1)*F[1] + dV(2,2)*F[2] +
          V[0]*dF(0,0) + V[1]*dF(1,1) + V[2]*dF(2,2);
}

double Div_MF3(const Vector & x)
{
   DenseTensor dM; Grad_M3(x, dM);
   DenseMatrix dF; Grad_F3(x, dF);
   DenseMatrix M; M3(x, M);
   Vector  F; F3(x, F);
   return
      dM(0,0,0)*F[0] + dM(0,1,0)*F[1] + dM(0,2,0)*F[2] +
      dM(1,0,1)*F[0] + dM(1,1,1)*F[1] + dM(1,2,1)*F[2] +
      dM(2,0,2)*F[0] + dM(2,1,2)*F[1] + dM(2,2,2)*F[2] +
      M(0,0)*dF(0,0) + M(0,1)*dF(1,0) + M(0,2)*dF(2,0) +
      M(1,0)*dF(0,1) + M(1,1)*dF(1,1) + M(1,2)*dF(2,1) +
      M(2,0)*dF(0,2) + M(2,1)*dF(1,2) + M(2,2)*dF(2,2);
}

void Curl_VcrossF3(const Vector & x, Vector & dVxF)
{
   Vector V; V3(x, V);
   DenseMatrix dV; Grad_V3(x, dV);
   Vector F; F3(x, F);
   DenseMatrix dF; Grad_F3(x, dF);
   dVxF.SetSize(3);
   dVxF[0] =
      dV(0,1)*F[1] - V[1]*dF(0,1) +
      dV(0,2)*F[2] - V[2]*dF(0,2) -
      (dV(1,1) + dV(2,2))*F[0] + V[0]*(dF(1,1) + dF(2,2));
   dVxF[1] =
      dV(1,2)*F[2] - V[2]*dF(1,2) +
      dV(1,0)*F[0] - V[0]*dF(1,0) -
      (dV(2,2) + dV(0,0))*F[1] + V[1]*(dF(2,2) + dF(0,0));
   dVxF[2] =
      dV(2,0)*F[0] - V[0]*dF(2,0) +
      dV(2,1)*F[1] - V[1]*dF(2,1) -
      (dV(0,0) + dV(1,1))*F[2] + V[2]*(dF(0,0) + dF(1,1));
}

void Curl_DF3(const Vector & x, Vector & dDF)
{
   Vector D; V3(x, D);
   DenseMatrix dD; Grad_V3(x, dD);
   Vector  F; F3(x, F);
   DenseMatrix dF; Grad_F3(x, dF);
   dDF.SetSize(3);
   dDF[0] = dD(2,1)*F[2] - dD(1,2)*F[1] + D[2]*dF(2,1) - D[1]*dF(1,2);
   dDF[1] = dD(0,2)*F[0] - dD(2,0)*F[2] + D[0]*dF(0,2) - D[2]*dF(2,0);
   dDF[2] = dD(1,0)*F[1] - dD(0,1)*F[0] + D[1]*dF(1,0) - D[0]*dF(0,1);
}

void Curl_MF3(const Vector & x, Vector & dMF)
{
   DenseMatrix M; M3(x, M);
   DenseTensor dM; Grad_M3(x, dM);
   Vector  F; F3(x, F);
   DenseMatrix dF; Grad_F3(x, dF);
   dMF.SetSize(3);
   dMF[0] =
      (dM(2,0,1) - dM(1,0,2))*F[0] + M(2,0)*dF(0,1) - M(1,0)*dF(0,2) +
      (dM(2,2,1) - dM(1,2,2))*F[2] + M(2,1)*dF(1,1) - M(1,2)*dF(2,2) +
      (dM(2,1,1) - dM(1,1,2))*F[1] + M(2,2)*dF(2,1) - M(1,1)*dF(1,2);
   dMF[1] =
      (dM(0,0,2) - dM(2,0,0))*F[0] + M(0,0)*dF(0,2) - M(2,0)*dF(0,0) +
      (dM(0,1,2) - dM(2,1,0))*F[1] + M(0,1)*dF(1,2) - M(2,1)*dF(1,0) +
      (dM(0,2,2) - dM(2,2,0))*F[2] + M(0,2)*dF(2,2) - M(2,2)*dF(2,0);
   dMF[2] =
      (dM(1,0,0) - dM(0,0,1))*F[0] + M(1,0)*dF(0,0) - M(0,0)*dF(0,1) +
      (dM(1,1,0) - dM(0,1,1))*F[1] + M(1,1)*dF(1,0) - M(0,1)*dF(1,1) +
      (dM(1,2,0) - dM(0,2,1))*F[2] + M(1,2)*dF(2,0) - M(0,2)*dF(2,1);
}

double Div_qGrad_f3(const Vector & x)
{
   Vector dq, df;
   Grad_q3(x, dq);
   Grad_f3(x, df);
   return dq * df;
}

double Div_VcrossGrad_f3(const Vector & x)
{
   DenseMatrix dv;
   Vector df;
   Grad_V3(x, dv);
   Grad_f3(x, df);
   return
      (dv(2,1) - dv(1,2))*df[0] +
      (dv(0,2) - dv(2,0))*df[1] +
      (dv(1,0) - dv(0,1))*df[2];
}

double Div_DGrad_f3(const Vector & x)
{
   DenseMatrix dv;
   Vector df;
   Grad_V3(x, dv);
   Grad_f3(x, df);
   return dv(0,0) * df[0] + dv(1,1) * df[1] + dv(2,2) * df[2];
}

double Div_MGrad_f3(const Vector & x)
{
   DenseTensor dm;
   Vector df;
   Grad_M3(x, dm);
   Grad_f3(x, df);
   return
      (dm(0,0,0) + dm(1,0,1) + dm(2,0,2)) * df[0] +
      (dm(0,1,0) + dm(1,1,1) + dm(2,1,2)) * df[1] +
      (dm(0,2,0) + dm(1,2,1) + dm(2,2,2)) * df[2];
}

void Curl_qCurlF3(const Vector & x, Vector & ddF)
{
   Vector dq; Grad_q3(x, dq);
   Vector dF; CurlF3(x, dF);
   ddF.SetSize(3);
   ddF[0] = dq[1]*dF[2] - dq[2]*dF[1];
   ddF[1] = dq[2]*dF[0] - dq[0]*dF[2];
   ddF[2] = dq[0]*dF[1] - dq[1]*dF[0];
}

void Curl_VcrossGrad_f3(const Vector & x, Vector & ddf)
{
   DenseMatrix dV; Grad_V3(x, dV);
   Vector df;      Grad_f3(x, df);
   ddf.SetSize(3);
   ddf[0] = dV(0,1)*df[1] + dV(0,2)*df[2] - (dV(1,1)+dV(2,2))*df[0];
   ddf[1] = dV(1,2)*df[2] + dV(1,0)*df[0] - (dV(2,2)+dV(0,0))*df[1];
   ddf[2] = dV(2,0)*df[0] + dV(2,1)*df[1] - (dV(0,0)+dV(1,1))*df[2];
}

void Curl_VcrossCurlF3(const Vector & x, Vector & ddF)
{
   DenseMatrix dv; Grad_V3(x, dv);
   Vector dF; CurlF3(x, dF);
   ddF.SetSize(3);
   ddF[0] = dv(0,1)*dF[1] + dv(0,2)*dF[2] - (dv(1,1)+dv(2,2))*dF[0];
   ddF[1] = dv(1,2)*dF[2] + dv(1,0)*dF[0] - (dv(2,2)+dv(0,0))*dF[1];
   ddF[2] = dv(2,0)*dF[0] + dv(2,1)*dF[1] - (dv(0,0)+dv(1,1))*dF[2];
}

void Curl_DCurlF3(const Vector & x, Vector & ddF)
{
   DenseMatrix dv;
   Grad_V3(x, dv);
   Vector dF; CurlF3(x, dF);
   ddF.SetSize(3);
   ddF[0] = dv(2,1)*dF[2] - dv(1,2)*dF[1];
   ddF[1] = dv(0,2)*dF[0] - dv(2,0)*dF[2];
   ddF[2] = dv(1,0)*dF[1] - dv(0,1)*dF[0];
}

void Curl_MCurlF3(const Vector & x, Vector & ddF)
{
   DenseTensor dm;
   Grad_M3(x, dm);
   Vector dF; CurlF3(x, dF);
   ddF.SetSize(3);
   ddF[0] =
      (dm(2,0,1)-dm(1,0,2))*dF[0] +
      (dm(2,1,1)-dm(1,1,2))*dF[1] +
      (dm(2,2,1)-dm(1,2,2))*dF[2];
   ddF[1] =
      (dm(0,0,2)-dm(2,0,0))*dF[0] +
      (dm(0,1,2)-dm(2,1,0))*dF[1] +
      (dm(0,2,2)-dm(2,2,0))*dF[2];
   ddF[2] =
      (dm(1,0,0)-dm(0,0,1))*dF[0] +
      (dm(1,1,0)-dm(0,1,1))*dF[1] +
      (dm(1,2,0)-dm(0,2,1))*dF[2];
}

void Grad_qDivF3(const Vector & x, Vector & ddF)
{
   Grad_q3(x, ddF);
   ddF *= DivF3(x);
}

void GradVdotGrad_f3(const Vector & x, Vector & ddf)
{
   DenseMatrix dv; Grad_V3(x, dv);
   Vector df; Grad_f3(x, df);
   ddf.SetSize(3);
   dv.MultTranspose(df,ddf);
}

double DivVDivF3(const Vector & x)
{
   return DivV3(x)*DivF3(x);
}

double DivVcrossCurlF3(const Vector & x)
{
   Vector dV; CurlV3(x, dV);
   Vector dF; CurlF3(x, dF);

   return dV * dF;
}

TEST_CASE("P2D Bilinear Vector Mass Integrators",
          "[ND_R2D_FECollection]"
          "[RT_R2D_FECollection]"
          "[VectorFEMassIntegrator]"
          "[MixedVectorMassIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 2, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-5;

   VectorFunctionCoefficient  F3_coef(vdim, F3);
   FunctionCoefficient        q3_coef(q3);
   VectorFunctionCoefficient  D3_coef(vdim, V3);
   MatrixFunctionCoefficient  M3_coef(vdim, M3);
   MatrixFunctionCoefficient MT3_coef(vdim, MT3);
   VectorFunctionCoefficient qF3_coef(vdim, qF3);
   VectorFunctionCoefficient DF3_coef(vdim, DF3);
   VectorFunctionCoefficient MF3_coef(vdim, MF3);

   for (int type = (int)Element::TRIANGLE;
        type <= (int)Element::QUADRILATERAL; type++)
   {
      Mesh mesh(n, n, (Element::Type)type, 1, 2.0, 3.0);

      SECTION("Operators on ND_P2D for element type " + std::to_string(type))
      {
         ND_R2D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         REQUIRE( f_nd.ComputeL2Error(F3_coef) < tol );

         SECTION("Mapping ND_P2D to RT_P2D")
         {
            {
               // Tests requiring an RT space with same order of
               // convergence as the ND space
               RT_R2D_FECollection    fec_rt(order - 1, dim);
               FiniteElementSpace fespace_rt(&mesh, &fec_rt);

               BilinearForm m_rt(&fespace_rt);
               m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
               m_rt.Assemble();
               m_rt.Finalize();

               GSSmoother s_rt(m_rt.SpMat());

               GridFunction g_rt(&fespace_rt);

               Vector tmp_rt(fespace_rt.GetNDofs());

               SECTION("Without Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_rt);
                  blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(F3_coef) < tol );

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
               SECTION("Without Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_rt);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator());
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(F3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
                  blfw.AddDomainIntegrator(new VectorFEMassIntegrator());
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
            }
            {
               // Tests requiring a higher order RT space
               RT_R2D_FECollection    fec_rt(order, dim);
               FiniteElementSpace fespace_rt(&mesh, &fec_rt);

               BilinearForm m_rt(&fespace_rt);
               m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
               m_rt.Assemble();
               m_rt.Finalize();

               GSSmoother s_rt(m_rt.SpMat());

               GridFunction g_rt(&fespace_rt);

               Vector tmp_rt(fespace_rt.GetNDofs());

               SECTION("With Scalar Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_rt);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(q3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(qF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(q3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Scalar Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_rt);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator(q3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(qF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
                  blfw.AddDomainIntegrator(new VectorFEMassIntegrator(q3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Diagonal Matrix Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_rt);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(D3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(DF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(D3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Diagonal Matrix Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_rt);
                  blf.AddDomainIntegrator(
                     new VectorFEMassIntegrator(D3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(DF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new VectorFEMassIntegrator(D3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Matrix Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_rt);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(M3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(MF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(MT3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Matrix Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_rt);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator(M3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(MF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new VectorFEMassIntegrator(MT3_coef));
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
         SECTION("Mapping ND_P2D to ND_P2D")
         {
            {
               // Tests requiring an ND test space with same order of
               // convergence as the ND trial space

               BilinearForm m_nd(&fespace_nd);
               m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
               m_nd.Assemble();
               m_nd.Finalize();

               GSSmoother s_nd(m_nd.SpMat());

               GridFunction g_nd(&fespace_nd);

               Vector tmp_nd(fespace_nd.GetNDofs());

               SECTION("Without Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_nd);
                  blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(F3_coef) < tol );

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
               SECTION("Without Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_nd);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator());
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(F3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_nd, &fespace_nd);
                  blfw.AddDomainIntegrator(new VectorFEMassIntegrator());
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
            }
            {
               // Tests requiring a higher order ND space
               ND_R2D_FECollection    fec_ndp(order+1, dim);
               FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

               BilinearForm m_ndp(&fespace_ndp);
               m_ndp.AddDomainIntegrator(new VectorFEMassIntegrator());
               m_ndp.Assemble();
               m_ndp.Finalize();

               GSSmoother s_ndp(m_ndp.SpMat());

               GridFunction g_ndp(&fespace_ndp);

               Vector tmp_ndp(fespace_ndp.GetNDofs());

               SECTION("With Scalar Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(q3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
                  PCG(m_ndp, s_ndp, tmp_ndp, g_ndp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_ndp.ComputeL2Error(qF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(q3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Scalar Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator(q3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
                  PCG(m_ndp, s_ndp, tmp_ndp, g_ndp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_ndp.ComputeL2Error(qF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
                  blfw.AddDomainIntegrator(new VectorFEMassIntegrator(q3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Diagonal Matrix Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(D3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
                  PCG(m_ndp, s_ndp, tmp_ndp, g_ndp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_ndp.ComputeL2Error(DF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(D3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Diagonal Matrix Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
                  blf.AddDomainIntegrator(
                     new VectorFEMassIntegrator(D3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
                  PCG(m_ndp, s_ndp, tmp_ndp, g_ndp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_ndp.ComputeL2Error(DF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new VectorFEMassIntegrator(D3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Matrix Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(M3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
                  PCG(m_ndp, s_ndp, tmp_ndp, g_ndp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_ndp.ComputeL2Error(MF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(MT3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Matrix Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator(M3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
                  PCG(m_ndp, s_ndp, tmp_ndp, g_ndp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_ndp.ComputeL2Error(MF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
                  blfw.AddDomainIntegrator(
                     new VectorFEMassIntegrator(MT3_coef));
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
      SECTION("Operators on RT_P2D for element type " + std::to_string(type))
      {
         RT_R2D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         REQUIRE( f_rt.ComputeL2Error(F3_coef) < tol );

         SECTION("Mapping RT_P2D to ND_P2D")
         {
            {
               // Tests requiring an ND test space with same order of
               // convergence as the RT trial space
               ND_R2D_FECollection    fec_nd(order, dim);
               FiniteElementSpace fespace_nd(&mesh, &fec_nd);

               BilinearForm m_nd(&fespace_nd);
               m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
               m_nd.Assemble();
               m_nd.Finalize();

               GSSmoother s_nd(m_nd.SpMat());

               GridFunction g_nd(&fespace_nd);

               Vector tmp_nd(fespace_nd.GetNDofs());

               SECTION("Without Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_nd);
                  blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(F3_coef) < tol );

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
               SECTION("Without Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_nd);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator());
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(F3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
                  blfw.AddDomainIntegrator(new VectorFEMassIntegrator());
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
            }
            {
               // Tests requiring a higher order ND space
               ND_R2D_FECollection    fec_nd(order + 1, dim);
               FiniteElementSpace fespace_nd(&mesh, &fec_nd);

               BilinearForm m_nd(&fespace_nd);
               m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
               m_nd.Assemble();
               m_nd.Finalize();

               GSSmoother s_nd(m_nd.SpMat());

               GridFunction g_nd(&fespace_nd);

               Vector tmp_nd(fespace_nd.GetNDofs());

               SECTION("With Scalar Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_nd);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(q3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(qF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(q3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Scalar Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_nd);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator(q3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(qF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
                  blfw.AddDomainIntegrator(new VectorFEMassIntegrator(q3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Diagonal Matrix Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_nd);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(D3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(DF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(D3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Diagonal Matrix Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_nd);
                  blf.AddDomainIntegrator(
                     new VectorFEMassIntegrator(D3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(DF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new VectorFEMassIntegrator(D3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Matrix Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_nd);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(M3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(MF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(MT3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Matrix Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_nd);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator(M3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
                  PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_nd.ComputeL2Error(MF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new VectorFEMassIntegrator(MT3_coef));
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
         SECTION("Mapping RT_P2D to RT_P2D")
         {
            {
               // Tests requiring an RT test space with same order of
               // convergence as the RT trial space

               BilinearForm m_rt(&fespace_rt);
               m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
               m_rt.Assemble();
               m_rt.Finalize();

               GSSmoother s_rt(m_rt.SpMat());

               GridFunction g_rt(&fespace_rt);

               Vector tmp_rt(fespace_rt.GetNDofs());

               SECTION("Without Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_rt);
                  blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(F3_coef) < tol );

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
               SECTION("Without Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_rt);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator());
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_rt); g_rt = 0.0;
                  PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rt.ComputeL2Error(F3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rt, &fespace_rt);
                  blfw.AddDomainIntegrator(new VectorFEMassIntegrator());
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
            }
            {
               // Tests requiring a higher order RT space
               RT_R2D_FECollection    fec_rtp(order, dim);
               FiniteElementSpace fespace_rtp(&mesh, &fec_rtp);

               BilinearForm m_rtp(&fespace_rtp);
               m_rtp.AddDomainIntegrator(new VectorFEMassIntegrator());
               m_rtp.Assemble();
               m_rtp.Finalize();

               GSSmoother s_rtp(m_rtp.SpMat());

               GridFunction g_rtp(&fespace_rtp);

               Vector tmp_rtp(fespace_rtp.GetNDofs());

               SECTION("With Scalar Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(q3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
                  PCG(m_rtp, s_rtp, tmp_rtp, g_rtp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rtp.ComputeL2Error(qF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(q3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Scalar Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator(q3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
                  PCG(m_rtp, s_rtp, tmp_rtp, g_rtp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rtp.ComputeL2Error(qF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
                  blfw.AddDomainIntegrator(new VectorFEMassIntegrator(q3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Diagonal Matrix Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(D3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
                  PCG(m_rtp, s_rtp, tmp_rtp, g_rtp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rtp.ComputeL2Error(DF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(D3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Diagonal Matrix Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
                  blf.AddDomainIntegrator(
                     new VectorFEMassIntegrator(D3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
                  PCG(m_rtp, s_rtp, tmp_rtp, g_rtp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rtp.ComputeL2Error(DF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new VectorFEMassIntegrator(D3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Matrix Coefficient (MixedVector)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
                  blf.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(M3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
                  PCG(m_rtp, s_rtp, tmp_rtp, g_rtp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rtp.ComputeL2Error(MF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new MixedVectorMassIntegrator(MT3_coef));
                  blfw.Assemble();
                  blfw.Finalize();

                  SparseMatrix * blfT = Transpose(blfw.SpMat());
                  SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

                  REQUIRE( diff->MaxNorm() < tol );

                  delete blfT;
                  delete diff;
               }
               SECTION("With Matrix Coefficient (VectorFE)")
               {
                  MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
                  blf.AddDomainIntegrator(new VectorFEMassIntegrator(M3_coef));
                  blf.Assemble();
                  blf.Finalize();

                  blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
                  PCG(m_rtp, s_rtp, tmp_rtp, g_rtp, 0, 200,
                      cg_rtol * cg_rtol, 0.0);

                  REQUIRE( g_rtp.ComputeL2Error(MF3_coef) < tol );

                  MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
                  blfw.AddDomainIntegrator(
                     new VectorFEMassIntegrator(MT3_coef));
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
}

} // namespace bilininteg_p2d
