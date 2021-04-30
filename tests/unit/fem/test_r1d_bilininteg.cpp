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

namespace bilininteg_r1d
{

double zero3(const Vector & x) { return 0.0; }
void Zero3(const Vector & x, Vector & v) { v.SetSize(3); v = 0.0; }

double f3(const Vector & x)
{ return 1.234 + 2.345 * x[0]; }
void Grad_f3(const Vector & x, Vector & df)
{
   df.SetSize(3);
   df[0] = 2.345;
   df[1] = 0.0;
   df[2] = 0.0;
}
void F3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] = 2.537 + 1.234 * x[0];
   v[1] = 1.763 +  3.572 * x[0];
   v[2] = 3.156 - 2.752 * x[0];
}
void Grad_F3(const Vector & x, DenseMatrix & df)
{
   df.SetSize(3);
   df(0,0) =  1.234; df(0,1) = 0.0; df(0,2) = 0.0;
   df(1,0) =  3.572; df(1,1) = 0.0; df(1,2) = 0.0;
   df(2,0) = -2.752; df(2,1) = 0.0; df(2,2) = 0.0;
}
void CurlF3(const Vector & x, Vector & df)
{
   df.SetSize(3);
   df[0] = 0.0;
   df[1] = 2.752;
   df[2] = 3.572;
}
double DivF3(const Vector & x)
{ return 1.234; }

double q3(const Vector & x)
{ return 2.678 + 4.234 * x[0]; }
void Grad_q3(const Vector & x, Vector & dq)
{
   dq.SetSize(3);
   dq[0] = 4.234;
   dq[1] = 0.0;
   dq[2] = 0.0;
}

void V3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] = 4.254 + 2.234 * x[0];
   v[1] = 1.789 + 4.572 * x[0];
   v[2] = 2.658 + 1.357 * x[0];
}
void Grad_V3(const Vector & x, DenseMatrix & dv)
{
   dv.SetSize(3);
   dv(0,0) = 2.234; dv(0,1) = 0.0; dv(0,2) = 0.0;
   dv(1,0) = 4.572; dv(1,1) = 0.0; dv(1,2) = 0.0;
   dv(2,0) = 1.357; dv(2,1) = 0.0; dv(2,2) = 0.0;
}
void CurlV3(const Vector & x, Vector & dV)
{
   dV.SetSize(3);
   dV[0] =  0.0;
   dV[1] = -1.357;
   dV[2] =  4.572;
}
double DivV3(const Vector & x)
{ return 2.234; }

void M3(const Vector & x, DenseMatrix & m)
{
   m.SetSize(3);

   m(0,0) = 1.792 + 4.234 * x[0];
   m(0,1) = 0.116 + 0.234 * x[0];
   m(0,2) = 0.213 - 0.537 * x[0];

   m(1,0) = 0.324 - 0.572 * x[0];
   m(1,1) = 1.234 + 4.537 * x[0];
   m(1,2) = 0.132 + 0.537 * x[0];

   m(2,0) = 0.214 + 0.572 * x[0];
   m(2,1) = 0.314 + 0.234 * x[0];
   m(2,2) = 1.431 + 1.572 * x[0];
}
void MT3(const Vector & x, DenseMatrix & m)
{
   M3(x, m); m.Transpose();
}
void Grad_M3(const Vector & x, DenseTensor & dm)
{
   dm.SetSize(3,3,3);
   dm(0,0,0) =  4.234; dm(0,0,1) = 0.0; dm(0,0,2) = 0.0;
   dm(0,1,0) =  0.234; dm(0,1,1) = 0.0; dm(0,1,2) = 0.0;
   dm(0,2,0) = -0.537; dm(0,2,1) = 0.0; dm(0,2,2) = 0.0;

   dm(1,0,0) = -0.572; dm(1,0,1) = 0.0; dm(1,0,2) = 0.0;
   dm(1,1,0) =  4.537; dm(1,1,1) = 0.0; dm(1,1,2) = 0.0;
   dm(1,2,0) =  0.537; dm(1,2,1) = 0.0; dm(1,2,2) = 0.0;

   dm(2,0,0) =  0.572; dm(2,0,1) = 0.0; dm(2,0,2) = 0.0;
   dm(2,1,0) =  0.234; dm(2,1,1) = 0.0; dm(2,1,2) = 0.0;
   dm(2,2,0) =  1.572; dm(2,2,1) = 0.0; dm(2,2,2) = 0.0;
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

TEST_CASE("R1D Bilinear Vector Mass Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[VectorFEMassIntegrator]"
          "[MixedVectorMassIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
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

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         REQUIRE( f_nd.ComputeL2Error(F3_coef) < tol );

         SECTION("Mapping ND_R1D to RT_R1D")
         {
            {
               // Tests requiring an RT space with same order of
               // convergence as the ND space
               RT_R1D_FECollection    fec_rt(order - 1, dim);
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
               RT_R1D_FECollection    fec_rt(order, dim);
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
         SECTION("Mapping ND_R1D to ND_R1D")
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
               ND_R1D_FECollection    fec_ndp(order+1, dim);
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
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         REQUIRE( f_rt.ComputeL2Error(F3_coef) < tol );

         SECTION("Mapping RT_R1D to ND_R1D")
         {
            {
               // Tests requiring an ND test space with same order of
               // convergence as the RT trial space
               ND_R1D_FECollection    fec_nd(order, dim);
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
               ND_R1D_FECollection    fec_nd(order + 1, dim);
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
         SECTION("Mapping RT_R1D to RT_R1D")
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
               RT_R1D_FECollection    fec_rtp(order, dim);
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

TEST_CASE("R1D Bilinear Curl Integrator",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedVectorCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      VectorFunctionCoefficient   F3_coef(vdim, F3);
      FunctionCoefficient         q3_coef(q3);
      VectorFunctionCoefficient   D3_coef(vdim, V3);
      MatrixFunctionCoefficient   M3_coef(vdim, M3);
      VectorFunctionCoefficient  dF3_coef(vdim, CurlF3);
      VectorFunctionCoefficient qdF3_coef(vdim, qCurlF3);
      VectorFunctionCoefficient DdF3_coef(vdim, DCurlF3);
      VectorFunctionCoefficient MdF3_coef(vdim, MCurlF3);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         REQUIRE( f_nd.ComputeL2Error(F3_coef) < tol );

         SECTION("Mapping ND to RT")
         {
            RT_R1D_FECollection    fec_rt(order - 1, dim);
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
               blf.AddDomainIntegrator(new MixedVectorCurlIntegrator());
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(qdF3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(DdF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(M3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(MdF3_coef) < tol );
            }
         }
         SECTION("Mapping ND_R1D to ND_R1D")
         {
            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(new MixedVectorCurlIntegrator());
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(qdF3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(DdF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(M3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(MdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Cross Product Curl Integrator",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedCrossCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      VectorFunctionCoefficient    F3_coef(vdim, F3);
      VectorFunctionCoefficient    V3_coef(vdim, V3);
      VectorFunctionCoefficient VxdF3_coef(vdim, VcrossCurlF3);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to RT_R1D")
         {
            RT_R1D_FECollection    fec_rt(order - 1, dim);
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
                  new MixedCrossCurlIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(VxdF3_coef) < tol );
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
                  new MixedCrossCurlIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(VxdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Divergence Integrator",
          "[RT_R1D_FECollection]"
          "[MixedScalarDivergenceIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      VectorFunctionCoefficient F3_coef(vdim, F3);
      FunctionCoefficient       q3_coef(q3);
      FunctionCoefficient      dF3_coef(DivF3);
      FunctionCoefficient     qdF3_coef(qDivF3);

      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to L2")
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

               REQUIRE( g_l2.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_l2);
               blf.AddDomainIntegrator(
                  new MixedScalarDivergenceIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_l2); g_l2 = 0.0;
               CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_l2.ComputeL2Error(qdF3_coef) < tol );
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

               REQUIRE( g_h1.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedScalarDivergenceIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_h1); g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(qdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Vector Divergence Integrator",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedVectorDivergenceIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      VectorFunctionCoefficient   F3_coef(vdim, F3);
      VectorFunctionCoefficient   V3_coef(vdim, V3);
      VectorFunctionCoefficient VdF3_coef(vdim, VDivF3);

      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         REQUIRE( f_rt.ComputeL2Error(F3_coef) < tol );

         SECTION("Divergence of RT_R1D")
         {
            DivergenceGridFunctionCoefficient divF_coef(&f_rt);

            L2_FECollection fec_l2(order - 1, dim);
            FiniteElementSpace fespace_l2(&mesh, &fec_l2);

            GridFunction df_l2(&fespace_l2); df_l2.ProjectCoefficient(divF_coef);

            FunctionCoefficient divF3_coef(DivF3);
            REQUIRE( df_l2.ComputeL2Error(divF3_coef) < tol );
         }
         SECTION("Mapping RT_R1D to RT_R1D")
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
                  new MixedVectorDivergenceIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(VdF3_coef) < tol );
            }
         }
         SECTION("Mapping RT_R1D to ND_R1D")
         {
            ND_R1D_FECollection    fec_nd(order, dim);
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
                  new MixedVectorDivergenceIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(VdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Vector Product Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedVectorProductIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient        f3_coef(f3);
   VectorFunctionCoefficient  V3_coef(vdim, V3);
   VectorFunctionCoefficient Vf3_coef(vdim, Vf3);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

         SECTION("Mapping H1 to ND_R1D")
         {
            ND_R1D_FECollection    fec_nd(order + 1, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GSSmoother s_nd(m_nd.SpMat());

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
               PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(Vf3_coef) < tol );

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedDotProductIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
         }
         SECTION("Mapping H1 to RT_R1D")
         {
            RT_R1D_FECollection    fec_rt(order, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GSSmoother s_rt(m_rt.SpMat());

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
               PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(Vf3_coef) < tol );

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedDotProductIntegrator(V3_coef));
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
      SECTION("Operators on L2 for element type " + std::to_string(type))
      {
         L2_FECollection    fec_l2(order, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(f3_coef);

         SECTION("Mapping L2 to ND_R1D")
         {
            ND_R1D_FECollection    fec_nd(order + 1, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GSSmoother s_nd(m_nd.SpMat());

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_l2, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_l2,tmp_nd); g_nd = 0.0;
               PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(Vf3_coef) < tol );

               MixedBilinearForm blfw(&fespace_nd, &fespace_l2);
               blfw.AddDomainIntegrator(
                  new MixedDotProductIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
         }
         SECTION("Mapping L2 to RT_R1D")
         {
            RT_R1D_FECollection    fec_rt(order, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GSSmoother s_rt(m_rt.SpMat());

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_l2, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_l2,tmp_rt); g_rt = 0.0;
               PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(Vf3_coef) < tol );

               MixedBilinearForm blfw(&fespace_rt, &fespace_l2);
               blfw.AddDomainIntegrator(
                  new MixedDotProductIntegrator(V3_coef));
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

TEST_CASE("R1D Bilinear Vector Cross Product Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedCrossProductIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient   F3_coef(vdim, F3);
   VectorFunctionCoefficient   V3_coef(vdim, V3);
   VectorFunctionCoefficient VxF3_coef(vdim, VcrossF3);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to ND_R1D")
         {
            ND_R1D_FECollection    fec_ndp(order + 1, dim);
            FiniteElementSpace fespace_ndp(&mesh, &fec_ndp);

            BilinearForm m_ndp(&fespace_ndp);
            m_ndp.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_ndp.Assemble();
            m_ndp.Finalize();

            GSSmoother s_ndp(m_ndp.SpMat());

            GridFunction g_ndp(&fespace_ndp);

            Vector tmp_ndp(fespace_ndp.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_ndp);
               blf.AddDomainIntegrator(
                  new MixedCrossProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_ndp); g_ndp = 0.0;
               PCG(m_ndp, s_ndp, tmp_ndp, g_ndp, 0, 200,
                   cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_ndp.ComputeL2Error(VxF3_coef) < tol );

               MixedBilinearForm blfw(&fespace_ndp, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedCrossProductIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
         }
         SECTION("Mapping ND_R1D to RT_R1D")
         {
            RT_R1D_FECollection    fec_rt(order, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GSSmoother s_rt(m_rt.SpMat());

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedCrossProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_rt); g_rt = 0.0;
               PCG(m_rt, s_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(VxF3_coef) < tol );

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedCrossProductIntegrator(V3_coef));
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
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to ND_R1D")
         {
            ND_R1D_FECollection    fec_nd(order + 1, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GSSmoother s_nd(m_nd.SpMat());

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedCrossProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_nd); g_nd = 0.0;
               PCG(m_nd, s_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(VxF3_coef) < tol );

               MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedCrossProductIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
         }
         SECTION("Mapping RT_R1D to RT_R1D")
         {
            RT_R1D_FECollection    fec_rtp(order, dim);
            FiniteElementSpace fespace_rtp(&mesh, &fec_rtp);

            BilinearForm m_rtp(&fespace_rtp);
            m_rtp.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rtp.Assemble();
            m_rtp.Finalize();

            GSSmoother s_rtp(m_rtp.SpMat());

            GridFunction g_rtp(&fespace_rtp);

            Vector tmp_rtp(fespace_rtp.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_rtp);
               blf.AddDomainIntegrator(
                  new MixedCrossProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_rtp); g_rtp = 0.0;
               PCG(m_rtp, s_rtp, tmp_rtp, g_rtp, 0, 200,
                   cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rtp.ComputeL2Error(VxF3_coef) < tol );

               MixedBilinearForm blfw(&fespace_rtp, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedCrossProductIntegrator(V3_coef));
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
}

TEST_CASE("R1D Bilinear Vector Dot Product Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedDotProductIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient  F3_coef(vdim, F3);
   VectorFunctionCoefficient  V3_coef(vdim, V3);
   FunctionCoefficient       VF3_coef(VdotF3);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D_ for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to H1")
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
                  new MixedDotProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_h1); g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(VF3_coef) < tol );
            }
         }
         SECTION("Mapping ND_R1D to L2")
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
                  new MixedDotProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_l2); g_l2 = 0.0;
               CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_l2.ComputeL2Error(VF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to H1")
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
                  new MixedDotProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_h1); g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(VF3_coef) < tol );
            }
         }
         SECTION("Mapping RT_R1D to L2")
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
                  new MixedDotProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_l2); g_l2 = 0.0;
               CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_l2.ComputeL2Error(VF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Weak Gradient Integrators",
          "[RT_R1D_FECollection]"
          "[MixedScalarWeakGradientIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryFluxLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient         f3_coef(f3);
   FunctionCoefficient         q3_coef(q3);
   FunctionCoefficient        qf3_coef(qf3);
   VectorFunctionCoefficient  df3_coef(vdim, Grad_f3);
   VectorFunctionCoefficient dqf3_coef(vdim, Grad_qf3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ProductCoefficient nf3_coef(bcNormal, f3_coef);
   ProductCoefficient nqf3_coef(bcNormal, qf3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

         SECTION("Mapping H1 to RT_R1D")
         {
            RT_R1D_FECollection    fec_rt(order - 1, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

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

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(nf3_coef));
               lf.Assemble();

               blfw.Mult(f_h1,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(df3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedScalarDivergenceIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_h1, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedScalarWeakGradientIntegrator(q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(nqf3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_h1,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dqf3_coef) < tol );
            }
         }
      }
      SECTION("Operators on L2 for element type " + std::to_string(type))
      {
         L2_FECollection    fec_l2(order, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(f3_coef);

         SECTION("Mapping L2 to RT_R1D")
         {
            RT_R1D_FECollection    fec_rt(order - 1, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

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

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(nf3_coef));
               lf.Assemble();

               blfw.Mult(f_l2,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(df3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_l2);
               blf.AddDomainIntegrator(
                  new MixedScalarDivergenceIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_l2, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedScalarWeakGradientIntegrator(q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(nqf3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_l2,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dqf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Weak Divergence Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedVectorWeakDivergenceIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient  F3_coef(vdim, F3);
   FunctionCoefficient        q3_coef(q3);
   MatrixFunctionCoefficient  M3_coef(vdim, M3);
   MatrixFunctionCoefficient MT3_coef(vdim, MT3);
   VectorFunctionCoefficient qF3_coef(vdim, qF3);
   VectorFunctionCoefficient DF3_coef(vdim, DF3);
   VectorFunctionCoefficient MF3_coef(vdim, MF3);
   FunctionCoefficient       dF3_coef(DivF3);
   FunctionCoefficient      dqF3_coef(Div_qF3);
   FunctionCoefficient      dDF3_coef(Div_DF3);
   FunctionCoefficient      dMF3_coef(Div_MF3);

   DenseMatrix R13(3, 1); R13 = 0.0; R13(0,0) = 1.0;
   MatrixConstantCoefficient R13_coef(R13);
   TransposeMatrixCoefficient R31_coef(R13_coef);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient nF3_coef(bcNormal, F3_coef);
   ScalarVectorProductCoefficient nqF3_coef(bcNormal, qF3_coef);
   ScalarVectorProductCoefficient nMF3_coef(bcNormal, MF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to H1")
         {
            H1_FECollection    fec_h1(order + 1, dim);
            FiniteElementSpace fespace_h1(&mesh, &fec_h1);

            BilinearForm m_h1(&fespace_h1);
            m_h1.AddDomainIntegrator(new MassIntegrator());
            m_h1.Assemble();
            m_h1.Finalize();

            GridFunction g_h1(&fespace_h1);

            Vector tmp_h1(fespace_h1.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(R13_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(R31_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_F3_coef(R31_coef, nF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_F3_coef));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               ScalarMatrixProductCoefficient q3_R13_coef(q3_coef, R13_coef);
               ScalarMatrixProductCoefficient R31_q3_coef(q3_coef, R31_coef);

               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(q3_R13_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(R31_q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_qF3_coef(R31_coef, nqF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_qF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dqF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MatrixProductCoefficient MT3_R13_coef(MT3_coef, R13_coef);
               MatrixProductCoefficient R31_M3_coef(R31_coef, M3_coef);

               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(MT3_R13_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(R31_M3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_MF3_coef(R31_coef, nMF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_MF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dMF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to H1")
         {
            H1_FECollection    fec_h1(order + 1, dim);
            FiniteElementSpace fespace_h1(&mesh, &fec_h1);

            BilinearForm m_h1(&fespace_h1);
            m_h1.AddDomainIntegrator(new MassIntegrator());
            m_h1.Assemble();
            m_h1.Finalize();

            GridFunction g_h1(&fespace_h1);

            Vector tmp_h1(fespace_h1.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(R13_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(R31_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_F3_coef(R31_coef, nF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_F3_coef));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               ScalarMatrixProductCoefficient q3_R13_coef(q3_coef, R13_coef);
               ScalarMatrixProductCoefficient R31_q3_coef(q3_coef, R31_coef);

               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(q3_R13_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(R31_q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_qF3_coef(R31_coef, nqF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_qF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dqF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MatrixProductCoefficient MT3_R13_coef(MT3_coef, R13_coef);
               MatrixProductCoefficient R31_M3_coef(R31_coef, M3_coef);

               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(MT3_R13_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(R31_M3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_MF3_coef(R31_coef, nMF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_MF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dMF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Weak Curl Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedVectorWeakCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient   F3_coef(vdim, F3);
   FunctionCoefficient         q3_coef(q3);
   VectorFunctionCoefficient   D3_coef(vdim, V3);
   MatrixFunctionCoefficient   M3_coef(vdim, M3);
   MatrixFunctionCoefficient  MT3_coef(vdim, MT3);
   VectorFunctionCoefficient  qF3_coef(vdim, qF3);
   VectorFunctionCoefficient  DF3_coef(vdim, DF3);
   VectorFunctionCoefficient  MF3_coef(vdim, MF3);
   VectorFunctionCoefficient  dF3_coef(vdim, CurlF3);
   VectorFunctionCoefficient dqF3_coef(vdim, Curl_qF3);
   VectorFunctionCoefficient dDF3_coef(vdim, Curl_DF3);
   VectorFunctionCoefficient dMF3_coef(vdim, Curl_MF3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient nF3_coef(bcNormal, F3_coef);
   ScalarVectorProductCoefficient nqF3_coef(bcNormal, qF3_coef);
   ScalarVectorProductCoefficient nDF3_coef(bcNormal, DF3_coef);
   ScalarVectorProductCoefficient nMF3_coef(bcNormal, MF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to ND_R1D")
         {
            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator());
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakCurlIntegrator());
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nF3_coef));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakCurlIntegrator(q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nqF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dqF3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakCurlIntegrator(D3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nDF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dDF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(MT3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakCurlIntegrator(M3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nMF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dMF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to ND_R1D")
         {
            ND_R1D_FECollection    fec_nd(order, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator());
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakCurlIntegrator());
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nF3_coef));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakCurlIntegrator(q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nqF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dqF3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakCurlIntegrator(D3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nDF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dDF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorCurlIntegrator(MT3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakCurlIntegrator(M3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nMF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dMF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Weak Div Cross Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedWeakDivCrossIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient   F3_coef(vdim, F3);
   VectorFunctionCoefficient   V3_coef(vdim, V3);
   VectorFunctionCoefficient  VF3_coef(vdim, VcrossF3);
   FunctionCoefficient       dVF3_coef(Div_VcrossF3);

   DenseMatrix R13(3, 1); R13 = 0.0; R13(0,0) = 1.0;
   MatrixConstantCoefficient R13_coef(R13);
   TransposeMatrixCoefficient R31_coef(R13_coef);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient nVF3_coef(bcNormal, VF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND to H1")
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
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedCrossGradIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedWeakDivCrossIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_VF3_coef(R31_coef, nVF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_VF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dVF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to H1")
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
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedCrossGradIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedWeakDivCrossIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_VF3_coef(R31_coef, nVF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_VF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dVF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Weak Curl Cross Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedWeakCurlCrossIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient    F3_coef(vdim, F3);
   VectorFunctionCoefficient    V3_coef(vdim, V3);
   VectorFunctionCoefficient  VxF3_coef(vdim, VcrossF3);
   VectorFunctionCoefficient dVxF3_coef(vdim, Curl_VcrossF3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient nVxF3_coef(bcNormal, VxF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to ND_R1D")
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
                  new MixedCrossCurlIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedWeakCurlCrossIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nVxF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dVxF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to ND_R1D")
         {
            ND_R1D_FECollection    fec_nd(order, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedCrossCurlIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedWeakCurlCrossIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nVxF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dVxF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Weak Grad Dot Product Integrators",
          "[ND_R1D_FECollection]"
          "[RT_R1D_FECollection]"
          "[MixedWeakGradDotIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryFluxLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 2, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(vdim, F3);
   VectorFunctionCoefficient     V3_coef(vdim, V3);
   FunctionCoefficient       VdotF3_coef(VdotF3);
   VectorFunctionCoefficient   dVF3_coef(vdim, GradVdotF3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ProductCoefficient nVdotF3_coef(bcNormal, VdotF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to RT_R1D")
         {
            RT_R1D_FECollection    fec_rt(order - 1, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GridFunction g_rt(&fespace_rt);
            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorDivergenceIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedWeakGradDotIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(nVdotF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dVF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to RT_R1D")
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
                  new MixedVectorDivergenceIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedWeakGradDotIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(nVdotF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dVF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Grad Div Integrators",
          "[RT_R1D_FECollection]"
          "[MixedGradDivIntegrator]"
          "[MixedDivGradIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[VectorFEBoundaryFluxLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient          f3_coef(f3);
   VectorFunctionCoefficient    F3_coef(vdim, F3);
   VectorFunctionCoefficient    V3_coef(vdim, V3);
   FunctionCoefficient        Vdf3_coef(VdotGrad_f3);
   VectorFunctionCoefficient  VdF3_coef(vdim, VDivF3);
   VectorFunctionCoefficient dVdf3_coef(vdim, GradVdotGrad_f3);
   FunctionCoefficient       dVdF3_coef(DivVDivF3);

   DenseMatrix R13(3, 1); R13 = 0.0; R13(0,0) = 1.0;
   MatrixConstantCoefficient R13_coef(R13);
   TransposeMatrixCoefficient R31_coef(R13_coef);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ProductCoefficient nVdf3_coef(bcNormal, Vdf3_coef);
   ScalarVectorProductCoefficient nVdF3_coef(bcNormal, VdF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

         SECTION("Mapping H1 to RT_R1D")
         {
            RT_R1D_FECollection    fec_rt(order - 1, dim);
            FiniteElementSpace fespace_rt(&mesh, &fec_rt);

            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MatrixVectorProductCoefficient R31_V3_coef(R31_coef, V3_coef);

               MixedBilinearForm blf(&fespace_rt, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedDivGradIntegrator(R31_V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_h1, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedGradDivIntegrator(R31_V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(nVdf3_coef));
               lf.Assemble();

               blfw.Mult(f_h1,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dVdf3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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
               MatrixVectorProductCoefficient R31_V3_coef(R31_coef, V3_coef);

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedDivGradIntegrator(R31_V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               MatrixVectorProductCoefficient R31_VdF3_coef(R31_coef,
                                                            nVdF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_VdF3_coef));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dVdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Mixed Cross Curl Grad Integrators",
          "[ND_R1D_FECollection]"
          "[MixedCrossCurlGradIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient    F3_coef(vdim, F3);
   VectorFunctionCoefficient    V3_coef(vdim, V3);
   VectorFunctionCoefficient VxdF3_coef(vdim, VcrossCurlF3);
   FunctionCoefficient      dVxdF3_coef(DivVcrossCurlF3);

   DenseMatrix R13(3, 1); R13 = 0.0; R13(0,0) = 1.0;
   MatrixConstantCoefficient R13_coef(R13);
   TransposeMatrixCoefficient R31_coef(R13_coef);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient nVxdF3_coef(bcNormal, VxdF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to H1")
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
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedCrossGradCurlIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedCrossCurlGradIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               MatrixVectorProductCoefficient R31_VxdF3_coef(R31_coef,
                                                             nVxdF3_coef);

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(R31_VxdF3_coef));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);
               g_h1 *= -1.0;

               REQUIRE( g_h1.ComputeL2Error(dVxdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Curl Curl Integrators",
          "[ND_R1D_FECollection]"
          "[CurlCurlIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(vdim, F3);
   FunctionCoefficient           q3_coef(q3);
   FunctionCoefficient        zero3_coef(zero3);
   VectorFunctionCoefficient  Zero3_coef(vdim, Zero3);
   VectorFunctionCoefficient    dF3_coef(vdim, CurlF3);
   VectorFunctionCoefficient  qdF3_coef(vdim, qCurlF3);
   VectorFunctionCoefficient dqdF3_coef(vdim, Curl_qCurlF3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient ndF3_coef(bcNormal, dF3_coef);
   ScalarVectorProductCoefficient nqdF3_coef(bcNormal, qdF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to ND_R1D")
         {
            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("Without Coefficient")
            {
               BilinearForm blf(&fespace_nd);
               blf.AddDomainIntegrator(new CurlCurlIntegrator());
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(ndF3_coef));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(Zero3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               BilinearForm blf(&fespace_nd);
               blf.AddDomainIntegrator(new CurlCurlIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nqdF3_coef, 1, 2));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dqdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Mixed Curl Curl Integrators",
          "[ND_R1D_FECollection]"
          "[MixedCurlCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(vdim, F3);
   FunctionCoefficient           q3_coef(q3);
   VectorFunctionCoefficient     D3_coef(vdim, V3);
   MatrixFunctionCoefficient     M3_coef(vdim, M3);
   MatrixFunctionCoefficient    MT3_coef(vdim, MT3);
   FunctionCoefficient        zero3_coef(zero3);
   VectorFunctionCoefficient  Zero3_coef(vdim, Zero3);
   VectorFunctionCoefficient    dF3_coef(vdim, CurlF3);
   VectorFunctionCoefficient  qdF3_coef(vdim, qCurlF3);
   VectorFunctionCoefficient  DdF3_coef(vdim, DCurlF3);
   VectorFunctionCoefficient  MdF3_coef(vdim, MCurlF3);
   VectorFunctionCoefficient dqdF3_coef(vdim, Curl_qCurlF3);
   VectorFunctionCoefficient dDdF3_coef(vdim, Curl_DCurlF3);
   VectorFunctionCoefficient dMdF3_coef(vdim, Curl_MCurlF3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient ndF3_coef(bcNormal, dF3_coef);
   ScalarVectorProductCoefficient nqdF3_coef(bcNormal, qdF3_coef);
   ScalarVectorProductCoefficient nDdF3_coef(bcNormal, DdF3_coef);
   ScalarVectorProductCoefficient nMdF3_coef(bcNormal, MdF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to ND_R1D")
         {
            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("Without Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedCurlCurlIntegrator());
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(ndF3_coef));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(Zero3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedCurlCurlIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nqdF3_coef, 1, 2));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dqdF3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedCurlCurlIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nDdF3_coef));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dDdF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedCurlCurlIntegrator(M3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blft(&fespace_nd, &fespace_nd);
               blft.AddDomainIntegrator(
                  new MixedCurlCurlIntegrator(MT3_coef));
               blft.Assemble();
               blft.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blft.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nMdF3_coef));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dMdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Mixed Cross Curl Curl Integrators",
          "[ND_R1D_FECollection]"
          "[MixedCrossCurlCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(vdim, F3);
   VectorFunctionCoefficient     V3_coef(vdim, V3);
   VectorFunctionCoefficient    dF3_coef(vdim, CurlF3);
   VectorFunctionCoefficient  VdF3_coef(vdim, VcrossCurlF3);
   VectorFunctionCoefficient dVdF3_coef(vdim, Curl_VcrossCurlF3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient nVdF3_coef(bcNormal, VdF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on ND_R1D for element type " + std::to_string(type))
      {
         ND_R1D_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND_R1D to ND_R1D")
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
                  new MixedCrossCurlCurlIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nVdF3_coef));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dVdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Mixed Cross Grad Curl Integrators",
          "[ND_R1D_FECollection]"
          "[MixedCrossGradCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient          f3_coef(f3);
   VectorFunctionCoefficient    V3_coef(vdim, V3);
   VectorFunctionCoefficient  Vdf3_coef(vdim, VcrossGrad_f3);
   VectorFunctionCoefficient dVdf3_coef(vdim, Curl_VcrossGrad_f3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ScalarVectorProductCoefficient nVdf3_coef(bcNormal, Vdf3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

         SECTION("Mapping H1 to ND_R1D")
         {
            ND_R1D_FECollection    fec_nd(order, dim);
            FiniteElementSpace fespace_nd(&mesh, &fec_nd);

            BilinearForm m_nd(&fespace_nd);
            m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_nd.Assemble();
            m_nd.Finalize();

            GridFunction g_nd(&fespace_nd);

            Vector tmp_nd(fespace_nd.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_nd, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedCrossCurlGradIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_h1, &fespace_nd);
               blfw.AddDomainIntegrator(
                  new MixedCrossGradCurlIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_nd);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryTangentLFIntegrator(nVdf3_coef));
               lf.Assemble();

               blfw.Mult(f_h1,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dVdf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("R1D Bilinear Div Div Integrators",
          "[RT_R1D_FECollection]"
          "[DivDivIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryFluxLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 1, vdim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(vdim, F3);
   FunctionCoefficient           q3_coef(q3);
   VectorFunctionCoefficient     D3_coef(vdim, V3);
   MatrixFunctionCoefficient     M3_coef(vdim, M3);
   MatrixFunctionCoefficient    MT3_coef(vdim, MT3);
   FunctionCoefficient        zero3_coef(zero3);
   VectorFunctionCoefficient  Zero3_coef(vdim, Zero3);
   FunctionCoefficient          dF3_coef(DivF3);
   FunctionCoefficient         qdF3_coef(qDivF3);
   VectorFunctionCoefficient  dqdF3_coef(vdim, Grad_qDivF3);

   // Set normal directions for the two mesh boundary points
   PWConstCoefficient bcNormal(2);
   bcNormal(1) = -1.0;
   bcNormal(2) =  1.0;

   ProductCoefficient ndF3_coef(bcNormal, dF3_coef);
   ProductCoefficient nqdF3_coef(bcNormal, qdF3_coef);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      SECTION("Operators on RT_R1D for element type " + std::to_string(type))
      {
         RT_R1D_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT_R1D to RT_R1D")
         {
            BilinearForm m_rt(&fespace_rt);
            m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
            m_rt.Assemble();
            m_rt.Finalize();

            GridFunction g_rt(&fespace_rt);

            Vector tmp_rt(fespace_rt.GetNDofs());

            SECTION("Without Coefficient")
            {
               BilinearForm blf(&fespace_rt);
               blf.AddDomainIntegrator(new DivDivIntegrator());
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(ndF3_coef));
               lf.Assemble();

               blf.Mult(f_rt,tmp_rt); tmp_rt -= lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(Zero3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               BilinearForm blf(&fespace_rt);
               blf.AddDomainIntegrator(new DivDivIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(nqdF3_coef));
               lf.Assemble();

               blf.Mult(f_rt,tmp_rt); tmp_rt -= lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);
               g_rt *= -1.0;

               REQUIRE( g_rt.ComputeL2Error(dqdF3_coef) < tol );
            }
         }
      }
   }
}

} // namespace bilininteg_r2d
