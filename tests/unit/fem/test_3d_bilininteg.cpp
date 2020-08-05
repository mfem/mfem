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

namespace bilininteg_3d
{

double zero3(const Vector & x) { return 0.0; }
void Zero3(const Vector & x, Vector & v) { v.SetSize(3); v = 0.0; }

double f3(const Vector & x)
{ return 2.345 * x[0] + 3.579 * x[1] + 4.680 * x[2]; }
void F3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] =  1.234 * x[0] - 2.357 * x[1] + 3.572 * x[2];
   v[1] =  2.537 * x[0] + 4.321 * x[1] - 1.234 * x[2];
   v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
}

double q3(const Vector & x)
{ return 4.234 * x[0] + 3.357 * x[1] + 1.572 * x[2]; }

void V3(const Vector & x, Vector & v)
{
   v.SetSize(3);
   v[0] = 4.234 * x[0] + 3.357 * x[1] + 1.572 * x[2];
   v[1] = 4.537 * x[0] + 1.321 * x[1] + 2.234 * x[2];
   v[2] = 1.572 * x[0] + 2.321 * x[1] + 3.234 * x[2];
}
void M3(const Vector & x, DenseMatrix & m)
{
   m.SetSize(3);

   m(0,0) =  4.234 * x[0] + 3.357 * x[1] + 1.572 * x[2];
   m(0,1) =  0.234 * x[0] + 0.357 * x[1] + 0.572 * x[2];
   m(0,2) = -0.537 * x[0] + 0.321 * x[1] - 0.234 * x[2];

   m(1,0) = -0.572 * x[0] - 0.321 * x[1] + 0.234 * x[2];
   m(1,1) =  4.537 * x[0] + 1.321 * x[1] + 2.234 * x[2];
   m(1,2) =  0.537 * x[0] + 0.321 * x[1] + 0.234 * x[2];

   m(2,0) =  0.572 * x[0] + 0.321 * x[1] + 0.234 * x[2];
   m(2,1) =  0.234 * x[0] - 0.357 * x[1] - 0.572 * x[2];
   m(2,2) =  1.572 * x[0] + 2.321 * x[1] + 3.234 * x[2];
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
   df[2] = 4.680;
}
void CurlF3(const Vector & x, Vector & df)
{
   df.SetSize(3);
   df[0] = 1.321 + 1.234;
   df[1] = 3.572 + 2.572;
   df[2] = 2.537 + 2.357;
}
double DivF3(const Vector & x)
{ return 1.234 + 4.321 + 3.234; }

void CurlV3(const Vector & x, Vector & dV)
{
   dV.SetSize(3);
   dV[0] = 2.321 - 2.234;
   dV[1] = 1.572 - 1.572;
   dV[2] = 4.537 - 3.357;
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
   dq[2] = 1.572;
}

void Grad_V3(const Vector & x, DenseMatrix & dv)
{
   dv.SetSize(3);
   dv(0,0) = 4.234; dv(0,1) = 3.357; dv(0,2) = 1.572;
   dv(1,0) = 4.537; dv(1,1) = 1.321; dv(1,2) = 2.234;
   dv(2,0) = 1.572; dv(2,1) = 2.321; dv(2,2) = 3.234;
}

double DivV3(const Vector & x)
{ return 4.234 + 1.321 + 3.234; }

void Grad_F3(const Vector & x, DenseMatrix & df)
{
   df.SetSize(3);
   df(0,0) =  1.234; df(0,1) = -2.357; df(0,2) =  3.572;
   df(1,0) =  2.537; df(1,1) =  4.321; df(1,2) = -1.234;
   df(2,0) = -2.572; df(2,1) =  1.321; df(2,2) =  3.234;
}

void Grad_M3(const Vector & x, DenseTensor & dm)
{
   dm.SetSize(3,3,3);
   dm(0,0,0) =  4.234; dm(0,0,1) =  3.357; dm(0,0,2) =  1.572;
   dm(0,1,0) =  0.234; dm(0,1,1) =  0.357; dm(0,1,2) =  0.572;
   dm(0,2,0) = -0.537; dm(0,2,1) =  0.321; dm(0,2,2) = -0.234;

   dm(1,0,0) = -0.572; dm(1,0,1) = -0.321; dm(1,0,2) =  0.234;
   dm(1,1,0) =  4.537; dm(1,1,1) =  1.321; dm(1,1,2) =  2.234;
   dm(1,2,0) =  0.537; dm(1,2,1) =  0.321; dm(1,2,2) =  0.234;

   dm(2,0,0) =  0.572; dm(2,0,1) =  0.321; dm(2,0,2) =  0.234;
   dm(2,1,0) =  0.234; dm(2,1,1) = -0.357; dm(2,1,2) = -0.572;
   dm(2,2,0) =  1.572; dm(2,2,1) =  2.321; dm(2,2,2) =  3.234;
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

TEST_CASE("3D Bilinear Mass Integrators",
          "[MixedScalarMassIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient f3_coef(f3);
   FunctionCoefficient q3_coef(q3);
   FunctionCoefficient qf3_coef(qf3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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

               REQUIRE( g_l2.ComputeL2Error(f3_coef) < tol );

               MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
               blfw.AddDomainIntegrator(new MixedScalarMassIntegrator());
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_l2);
               blf.AddDomainIntegrator(new MixedScalarMassIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_l2); g_l2 = 0.0;
               CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_l2.ComputeL2Error(qf3_coef) < tol );

               MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
               blfw.AddDomainIntegrator(new MixedScalarMassIntegrator(q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
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

               REQUIRE( g_h1.ComputeL2Error(f3_coef) < tol );
            }
            SECTION("With Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_h1);
               blf.AddDomainIntegrator(new MixedScalarMassIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_h1); g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(qf3_coef) < tol );
            }
         }
      }
      SECTION("Operators on L2 for element type " + std::to_string(type))
      {
         L2_FECollection    fec_l2(order, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(f3_coef);

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

               REQUIRE( g_l2.ComputeL2Error(f3_coef) < tol );
            }
            SECTION("With Coefficient")
            {
               MixedBilinearForm blf(&fespace_l2, &fespace_l2);
               blf.AddDomainIntegrator(new MixedScalarMassIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_l2,tmp_l2); g_l2 = 0.0;
               CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_l2.ComputeL2Error(qf3_coef) < tol );
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

               REQUIRE( g_h1.ComputeL2Error(f3_coef) < tol );

               MixedBilinearForm blfw(&fespace_h1, &fespace_l2);
               blfw.AddDomainIntegrator(new MixedScalarMassIntegrator());
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;
            }
            SECTION("With Coefficient")
            {
               MixedBilinearForm blf(&fespace_l2, &fespace_h1);
               blf.AddDomainIntegrator(new MixedScalarMassIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_l2,tmp_h1); g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(qf3_coef) < tol );

               MixedBilinearForm blfw(&fespace_h1, &fespace_l2);
               blfw.AddDomainIntegrator(new MixedScalarMassIntegrator(q3_coef));
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

TEST_CASE("3D Bilinear Vector Mass Integrators",
          "[VectorFEMassIntegrator]"
          "[MixedVectorMassIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-5;

   VectorFunctionCoefficient  F3_coef(dim, F3);
   FunctionCoefficient        q3_coef(q3);
   VectorFunctionCoefficient  D3_coef(dim, V3);
   MatrixFunctionCoefficient  M3_coef(dim, M3);
   MatrixFunctionCoefficient MT3_coef(dim, MT3);
   VectorFunctionCoefficient qF3_coef(dim, qF3);
   VectorFunctionCoefficient DF3_coef(dim, DF3);
   VectorFunctionCoefficient MF3_coef(dim, MF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

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
               RT_FECollection    fec_rt(order, dim);
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
         SECTION("Mapping ND to ND")
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
               ND_FECollection    fec_ndp(order+1, dim);
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
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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
               ND_FECollection    fec_nd(order + 1, dim);
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
         SECTION("Mapping RT to RT")
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
               RT_FECollection    fec_rtp(order, dim);
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

TEST_CASE("3D Bilinear Gradient Integrator",
          "[MixedVectorGradientIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient         f3_coef(f3);
   FunctionCoefficient         q3_coef(q3);
   VectorFunctionCoefficient   D3_coef(dim, V3);
   MatrixFunctionCoefficient   M3_coef(dim, M3);
   VectorFunctionCoefficient  df3_coef(dim, Grad_f3);
   VectorFunctionCoefficient qdf3_coef(dim, qGrad_f3);
   VectorFunctionCoefficient Ddf3_coef(dim, DGrad_f3);
   VectorFunctionCoefficient Mdf3_coef(dim, MGrad_f3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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

               blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(df3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(qdf3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(Ddf3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(M3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(Mdf3_coef) < tol );
            }
         }
         SECTION("Mapping H1 to RT")
         {
            // Tests requiring an RT test space with same order of
            // convergence as the RT trial space

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

               REQUIRE( g_rt.ComputeL2Error(df3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(qdf3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(Ddf3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(M3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(Mdf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Curl Integrator",
          "[MixedVectorCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient   F3_coef(dim, F3);
      FunctionCoefficient         q3_coef(q3);
      VectorFunctionCoefficient   D3_coef(dim, V3);
      MatrixFunctionCoefficient   M3_coef(dim, M3);
      VectorFunctionCoefficient  dF3_coef(dim, CurlF3);
      VectorFunctionCoefficient qdF3_coef(dim, qCurlF3);
      VectorFunctionCoefficient DdF3_coef(dim, DCurlF3);
      VectorFunctionCoefficient MdF3_coef(dim, MCurlF3);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

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
         SECTION("Mapping ND to ND")
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

TEST_CASE("3D Bilinear Cross Product Gradient Integrator",
          "[MixedCrossGradIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      FunctionCoefficient          f3_coef(f3);
      VectorFunctionCoefficient    V3_coef(dim, V3);
      VectorFunctionCoefficient Vxdf3_coef(dim, VcrossGrad_f3);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedCrossGradIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(Vxdf3_coef) < tol );
            }
         }
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

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedCrossGradIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_nd); g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(Vxdf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Cross Product Curl Integrator",
          "[MixedCrossCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient    F3_coef(dim, F3);
      VectorFunctionCoefficient    V3_coef(dim, V3);
      VectorFunctionCoefficient VxdF3_coef(dim, VcrossCurlF3);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

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

TEST_CASE("3D Bilinear Divergence Integrator",
          "[MixedScalarDivergenceIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient F3_coef(dim, F3);
      FunctionCoefficient       q3_coef(q3);
      FunctionCoefficient      dF3_coef(DivF3);
      FunctionCoefficient     qdF3_coef(qDivF3);

      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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

TEST_CASE("3D Bilinear Vector Divergence Integrator",
          "[MixedVectorDivergenceIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      VectorFunctionCoefficient   F3_coef(dim, F3);
      VectorFunctionCoefficient   V3_coef(dim, V3);
      VectorFunctionCoefficient VdF3_coef(dim, VDivF3);

      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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
                  new MixedVectorDivergenceIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_rt,tmp_rt); g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(VdF3_coef) < tol );
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

TEST_CASE("3D Bilinear Vector Product Integrators",
          "[MixedVectorProductIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient        f3_coef(f3);
   VectorFunctionCoefficient  V3_coef(dim, V3);
   VectorFunctionCoefficient Vf3_coef(dim, Vf3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

         SECTION("Mapping H1 to ND")
         {
            ND_FECollection    fec_nd(order + 1, dim);
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
         SECTION("Mapping H1 to RT")
         {
            RT_FECollection    fec_rt(order, dim);
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

         SECTION("Mapping L2 to ND")
         {
            ND_FECollection    fec_nd(order + 1, dim);
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
         SECTION("Mapping L2 to RT")
         {
            RT_FECollection    fec_rt(order, dim);
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

TEST_CASE("3D Bilinear Vector Cross Product Integrators",
          "[MixedCrossProductIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient   F3_coef(dim, F3);
   VectorFunctionCoefficient   V3_coef(dim, V3);
   VectorFunctionCoefficient VxF3_coef(dim, VcrossF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND to ND")
         {
            ND_FECollection    fec_ndp(order + 1, dim);
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
         SECTION("Mapping ND to RT")
         {
            RT_FECollection    fec_rt(order, dim);
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
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT to ND")
         {
            ND_FECollection    fec_nd(order + 1, dim);
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
         SECTION("Mapping RT to RT")
         {
            RT_FECollection    fec_rtp(order, dim);
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

TEST_CASE("3D Bilinear Vector Dot Product Integrators",
          "[MixedDotProductIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient  F3_coef(dim, F3);
   VectorFunctionCoefficient  V3_coef(dim, V3);
   FunctionCoefficient       VF3_coef(VdotF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

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
                  new MixedDotProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_h1); g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(VF3_coef) < tol );
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
                  new MixedDotProductIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_nd,tmp_l2); g_l2 = 0.0;
               CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_l2.ComputeL2Error(VF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
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

TEST_CASE("3D Bilinear Directional Derivative Integrator",
          "[MixedDirectionalDerivativeIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient         f3_coef(f3);
   VectorFunctionCoefficient   V3_coef(dim, V3);
   FunctionCoefficient       Vdf3_coef(VdotGrad_f3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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
                  new MixedDirectionalDerivativeIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_h1); g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(Vdf3_coef) < tol );
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
                  new MixedDirectionalDerivativeIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               blf.Mult(f_h1,tmp_l2); g_l2 = 0.0;
               CG(m_l2, tmp_l2, g_l2, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_l2.ComputeL2Error(Vdf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Weak Gradient Integrators",
          "[MixedScalarWeakGradientIntegrator]"
          "[MixedScalarIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryFluxLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient         f3_coef(f3);
   FunctionCoefficient         q3_coef(q3);
   FunctionCoefficient        qf3_coef(qf3);
   VectorFunctionCoefficient  df3_coef(dim, Grad_f3);
   VectorFunctionCoefficient dqf3_coef(dim, Grad_qf3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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
                  new VectorFEBoundaryFluxLFIntegrator(f3_coef));
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
                  new VectorFEBoundaryFluxLFIntegrator(qf3_coef, 1, 2));
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

         SECTION("Mapping L2 to RT")
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
                  new VectorFEBoundaryFluxLFIntegrator(f3_coef));
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
                  new VectorFEBoundaryFluxLFIntegrator(qf3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_l2,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dqf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Scalar Weak Divergence Integrators",
          "[MixedScalarWeakDivergenceIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient         f3_coef(f3);
   VectorFunctionCoefficient   V3_coef(dim, V3);
   VectorFunctionCoefficient  Vf3_coef(dim, Vf3);
   FunctionCoefficient       dVf3_coef(Div_Vf3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

         SECTION("Mapping H1 to H1")
         {
            H1_FECollection    fec_h1p(order + 1, dim);
            FiniteElementSpace fespace_h1p(&mesh, &fec_h1p);

            BilinearForm m_h1(&fespace_h1p);
            m_h1.AddDomainIntegrator(new MassIntegrator());
            m_h1.Assemble();
            m_h1.Finalize();

            GridFunction g_h1(&fespace_h1p);

            Vector tmp_h1(fespace_h1p.GetNDofs());

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1p, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedDirectionalDerivativeIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_h1, &fespace_h1p);
               blfw.AddDomainIntegrator(
                  new MixedScalarWeakDivergenceIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1p);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(Vf3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_h1,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dVf3_coef) < tol );
            }
         }
      }
      SECTION("Operators on L2 for element type " + std::to_string(type))
      {
         L2_FECollection    fec_l2(order - 1, dim);
         FiniteElementSpace fespace_l2(&mesh, &fec_l2);

         GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(f3_coef);

         SECTION("Mapping L2 to H1")
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
               MixedBilinearForm blf(&fespace_h1, &fespace_l2);
               blf.AddDomainIntegrator(
                  new MixedDirectionalDerivativeIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedScalarWeakDivergenceIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(Vf3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_l2,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dVf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Weak Divergence Integrators",
          "[MixedVectorWeakDivergenceIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient  F3_coef(dim, F3);
   FunctionCoefficient        q3_coef(q3);
   VectorFunctionCoefficient  D3_coef(dim, V3);
   MatrixFunctionCoefficient  M3_coef(dim, M3);
   MatrixFunctionCoefficient MT3_coef(dim, MT3);
   VectorFunctionCoefficient qF3_coef(dim, qF3);
   VectorFunctionCoefficient DF3_coef(dim, DF3);
   VectorFunctionCoefficient MF3_coef(dim, MF3);
   FunctionCoefficient       dF3_coef(DivF3);
   FunctionCoefficient      dqF3_coef(Div_qF3);
   FunctionCoefficient      dDF3_coef(Div_DF3);
   FunctionCoefficient      dMF3_coef(Div_MF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
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

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(F3_coef));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(qF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dqF3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(D3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(DF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dDF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_nd);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(MT3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(M3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(MF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dMF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(F3_coef));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dF3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(q3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(qF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dqF3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(D3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(DF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dDF3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_rt);
               blf.AddDomainIntegrator(
                  new MixedVectorGradientIntegrator(MT3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedVectorWeakDivergenceIntegrator(M3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(MF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dMF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Weak Curl Integrators",
          "[MixedVectorWeakCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient   F3_coef(dim, F3);
   FunctionCoefficient         q3_coef(q3);
   VectorFunctionCoefficient   D3_coef(dim, V3);
   MatrixFunctionCoefficient   M3_coef(dim, M3);
   MatrixFunctionCoefficient  MT3_coef(dim, MT3);
   VectorFunctionCoefficient  qF3_coef(dim, qF3);
   VectorFunctionCoefficient  DF3_coef(dim, DF3);
   VectorFunctionCoefficient  MF3_coef(dim, MF3);
   VectorFunctionCoefficient  dF3_coef(dim, CurlF3);
   VectorFunctionCoefficient dqF3_coef(dim, Curl_qF3);
   VectorFunctionCoefficient dDF3_coef(dim, Curl_DF3);
   VectorFunctionCoefficient dMF3_coef(dim, Curl_MF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND to ND")
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
                  new VectorFEBoundaryTangentLFIntegrator(F3_coef));
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
                  new VectorFEBoundaryTangentLFIntegrator(qF3_coef, 1, 2));
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
                  new VectorFEBoundaryTangentLFIntegrator(DF3_coef, 1, 2));
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
                  new VectorFEBoundaryTangentLFIntegrator(MF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dMF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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
                  new VectorFEBoundaryTangentLFIntegrator(F3_coef));
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
                  new VectorFEBoundaryTangentLFIntegrator(qF3_coef, 1, 2));
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
                  new VectorFEBoundaryTangentLFIntegrator(DF3_coef, 1, 2));
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
                  new VectorFEBoundaryTangentLFIntegrator(MF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dMF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Weak Div Cross Integrators",
          "[MixedWeakDivCrossIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient   F3_coef(dim, F3);
   VectorFunctionCoefficient   V3_coef(dim, V3);
   VectorFunctionCoefficient  VF3_coef(dim, VcrossF3);
   FunctionCoefficient       dVF3_coef(Div_VcrossF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
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

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(VF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dVF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(VF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dVF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Weak Curl Cross Integrators",
          "[MixedWeakCurlCrossIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient    F3_coef(dim, F3);
   VectorFunctionCoefficient    V3_coef(dim, V3);
   VectorFunctionCoefficient  VxF3_coef(dim, VcrossF3);
   VectorFunctionCoefficient dVxF3_coef(dim, Curl_VcrossF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

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
                  new VectorFEBoundaryTangentLFIntegrator(VxF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dVxF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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
                  new VectorFEBoundaryTangentLFIntegrator(VxF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dVxF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Weak Grad Dot Product Integrators",
          "[MixedWeakGradDotIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryFluxLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(dim, F3);
   VectorFunctionCoefficient     V3_coef(dim, V3);
   FunctionCoefficient       VdotF3_coef(VdotF3);
   VectorFunctionCoefficient   dVF3_coef(dim, GradVdotF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

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
                  new VectorFEBoundaryFluxLFIntegrator(VdotF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_nd,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dVF3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

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
                  new VectorFEBoundaryFluxLFIntegrator(VdotF3_coef, 1, 2));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dVF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Grad Div Integrators",
          "[MixedGradDivIntegrator]"
          "[MixedDivGradIntegrator]"
          "[MixedScalarVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[VectorFEBoundaryFluxLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient          f3_coef(f3);
   VectorFunctionCoefficient    F3_coef(dim, F3);
   VectorFunctionCoefficient    V3_coef(dim, V3);
   FunctionCoefficient        Vdf3_coef(VdotGrad_f3);
   VectorFunctionCoefficient  VdF3_coef(dim, VDivF3);
   VectorFunctionCoefficient dVdf3_coef(dim, GradVdotGrad_f3);
   FunctionCoefficient       dVdF3_coef(DivVDivF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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

            SECTION("With Vector Coefficient")
            {
               MixedBilinearForm blf(&fespace_rt, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedDivGradIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blfw(&fespace_h1, &fespace_rt);
               blfw.AddDomainIntegrator(
                  new MixedGradDivIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               SparseMatrix * blfT = Transpose(blfw.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_rt);
               lf.AddBoundaryIntegrator(
                  new VectorFEBoundaryFluxLFIntegrator(Vdf3_coef));
               lf.Assemble();

               blfw.Mult(f_h1,tmp_rt); tmp_rt += lf; g_rt = 0.0;
               CG(m_rt, tmp_rt, g_rt, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_rt.ComputeL2Error(dVdf3_coef) < tol );
            }
         }
      }
      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
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
               MixedBilinearForm blfw(&fespace_rt, &fespace_h1);
               blfw.AddDomainIntegrator(
                  new MixedDivGradIntegrator(V3_coef));
               blfw.Assemble();
               blfw.Finalize();

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(VdF3_coef));
               lf.Assemble();

               blfw.Mult(f_rt,tmp_h1); tmp_h1 += lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(dVdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Grad Grad Integrators",
          "[DiffusionIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient         f3_coef(f3);
   FunctionCoefficient         q3_coef(q3);
   MatrixFunctionCoefficient   M3_coef(dim, M3);
   MatrixFunctionCoefficient  MT3_coef(dim, MT3);
   FunctionCoefficient      zero3_coef(zero3);
   VectorFunctionCoefficient  df3_coef(dim, Grad_f3);
   VectorFunctionCoefficient qdf3_coef(dim, qGrad_f3);
   VectorFunctionCoefficient Mdf3_coef(dim, MGrad_f3);
   FunctionCoefficient      dqdf3_coef(Div_qGrad_f3);
   FunctionCoefficient      dMdf3_coef(Div_MGrad_f3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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
               BilinearForm blf(&fespace_h1);
               blf.AddDomainIntegrator(new DiffusionIntegrator());
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(df3_coef));
               lf.Assemble();

               blf.Mult(f_h1,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(zero3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               BilinearForm blf(&fespace_h1);
               blf.AddDomainIntegrator(new DiffusionIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(qdf3_coef));
               lf.Assemble();

               blf.Mult(f_h1,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);
               g_h1 *= -1.0;

               REQUIRE( g_h1.ComputeL2Error(dqdf3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               BilinearForm blf(&fespace_h1);
               blf.AddDomainIntegrator(new DiffusionIntegrator(M3_coef));
               blf.Assemble();
               blf.Finalize();

               BilinearForm blft(&fespace_h1);
               blft.AddDomainIntegrator(new DiffusionIntegrator(MT3_coef));
               blft.Assemble();
               blft.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blft.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(Mdf3_coef));
               lf.Assemble();

               blf.Mult(f_h1,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);
               g_h1 *= -1.0;

               REQUIRE( g_h1.ComputeL2Error(dMdf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Mixed Grad Grad Integrators",
          "[MixedGradGradIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient         f3_coef(f3);
   FunctionCoefficient         q3_coef(q3);
   VectorFunctionCoefficient   D3_coef(dim, V3);
   MatrixFunctionCoefficient   M3_coef(dim, M3);
   MatrixFunctionCoefficient  MT3_coef(dim, MT3);
   FunctionCoefficient      zero3_coef(zero3);
   VectorFunctionCoefficient  df3_coef(dim, Grad_f3);
   VectorFunctionCoefficient qdf3_coef(dim, qGrad_f3);
   VectorFunctionCoefficient Ddf3_coef(dim, DGrad_f3);
   VectorFunctionCoefficient Mdf3_coef(dim, MGrad_f3);
   FunctionCoefficient      dqdf3_coef(Div_qGrad_f3);
   FunctionCoefficient      dDdf3_coef(Div_DGrad_f3);
   FunctionCoefficient      dMdf3_coef(Div_MGrad_f3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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
               blf.AddDomainIntegrator(
                  new MixedGradGradIntegrator());
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(df3_coef));
               lf.Assemble();

               blf.Mult(f_h1,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_h1.ComputeL2Error(zero3_coef) < tol );
            }
            SECTION("With Scalar Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedGradGradIntegrator(q3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(qdf3_coef));
               lf.Assemble();

               blf.Mult(f_h1,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);
               g_h1 *= -1.0;

               REQUIRE( g_h1.ComputeL2Error(dqdf3_coef) < tol );
            }
            SECTION("With Diagonal Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedGradGradIntegrator(D3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(Ddf3_coef));
               lf.Assemble();

               blf.Mult(f_h1,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);
               g_h1 *= -1.0;

               REQUIRE( g_h1.ComputeL2Error(dDdf3_coef) < tol );
            }
            SECTION("With Matrix Coefficient")
            {
               MixedBilinearForm blf(&fespace_h1, &fespace_h1);
               blf.AddDomainIntegrator(
                  new MixedGradGradIntegrator(M3_coef));
               blf.Assemble();
               blf.Finalize();

               MixedBilinearForm blft(&fespace_h1, &fespace_h1);
               blft.AddDomainIntegrator(
                  new MixedGradGradIntegrator(MT3_coef));
               blft.Assemble();
               blft.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blft.SpMat(),-1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(Mdf3_coef));
               lf.Assemble();

               blf.Mult(f_h1,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);
               g_h1 *= -1.0;

               REQUIRE( g_h1.ComputeL2Error(dMdf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Mixed Cross Grad Grad Integrators",
          "[MixedCrossGradGradIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient          f3_coef(f3);
   VectorFunctionCoefficient    V3_coef(dim, V3);
   VectorFunctionCoefficient Vxdf3_coef(dim, VcrossGrad_f3);
   FunctionCoefficient      dVxdf3_coef(Div_VcrossGrad_f3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

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
                  new MixedCrossGradGradIntegrator(V3_coef));
               blf.Assemble();
               blf.Finalize();

               SparseMatrix * blfT = Transpose(blf.SpMat());
               SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);

               REQUIRE( diff->MaxNorm() < tol );

               delete blfT;
               delete diff;

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(Vxdf3_coef));
               lf.Assemble();

               blf.Mult(f_h1,tmp_h1); tmp_h1 -= lf; g_h1 = 0.0;
               CG(m_h1, tmp_h1, g_h1, 0, 200, cg_rtol * cg_rtol, 0.0);
               g_h1 *= -1.0;

               REQUIRE( g_h1.ComputeL2Error(dVxdf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Mixed Cross Curl Grad Integrators",
          "[MixedCrossCurlGradIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[BoundaryNormalLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient    F3_coef(dim, F3);
   VectorFunctionCoefficient    V3_coef(dim, V3);
   VectorFunctionCoefficient VxdF3_coef(dim, VcrossCurlF3);
   FunctionCoefficient      dVxdF3_coef(DivVcrossCurlF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

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

               LinearForm lf(&fespace_h1);
               lf.AddBoundaryIntegrator(
                  new BoundaryNormalLFIntegrator(VxdF3_coef));
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

TEST_CASE("3D Bilinear Curl Curl Integrators",
          "[CurlCurlIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(dim, F3);
   FunctionCoefficient           q3_coef(q3);
   FunctionCoefficient        zero3_coef(zero3);
   VectorFunctionCoefficient  Zero3_coef(dim, Zero3);
   VectorFunctionCoefficient    dF3_coef(dim, CurlF3);
   VectorFunctionCoefficient  qdF3_coef(dim, qCurlF3);
   VectorFunctionCoefficient dqdF3_coef(dim, Curl_qCurlF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      type++;
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND to ND")
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
                  new VectorFEBoundaryTangentLFIntegrator(dF3_coef));
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
                  new VectorFEBoundaryTangentLFIntegrator(qdF3_coef, 1, 2));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dqdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Mixed Curl Curl Integrators",
          "[MixedCurlCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(dim, F3);
   FunctionCoefficient           q3_coef(q3);
   VectorFunctionCoefficient     D3_coef(dim, V3);
   MatrixFunctionCoefficient     M3_coef(dim, M3);
   MatrixFunctionCoefficient    MT3_coef(dim, MT3);
   FunctionCoefficient        zero3_coef(zero3);
   VectorFunctionCoefficient  Zero3_coef(dim, Zero3);
   VectorFunctionCoefficient    dF3_coef(dim, CurlF3);
   VectorFunctionCoefficient  qdF3_coef(dim, qCurlF3);
   VectorFunctionCoefficient  DdF3_coef(dim, DCurlF3);
   VectorFunctionCoefficient  MdF3_coef(dim, MCurlF3);
   VectorFunctionCoefficient dqdF3_coef(dim, Curl_qCurlF3);
   VectorFunctionCoefficient dDdF3_coef(dim, Curl_DCurlF3);
   VectorFunctionCoefficient dMdF3_coef(dim, Curl_MCurlF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

         SECTION("Mapping ND to ND")
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
                  new VectorFEBoundaryTangentLFIntegrator(dF3_coef));
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
                  new VectorFEBoundaryTangentLFIntegrator(qdF3_coef, 1, 2));
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
                  new VectorFEBoundaryTangentLFIntegrator(DdF3_coef));
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
                  new VectorFEBoundaryTangentLFIntegrator(MdF3_coef));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dMdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Mixed Cross Curl Curl Integrators",
          "[MixedCrossCurlCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(dim, F3);
   VectorFunctionCoefficient     V3_coef(dim, V3);
   VectorFunctionCoefficient    dF3_coef(dim, CurlF3);
   VectorFunctionCoefficient  VdF3_coef(dim, VcrossCurlF3);
   VectorFunctionCoefficient dVdF3_coef(dim, Curl_VcrossCurlF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on ND for element type " + std::to_string(type))
      {
         ND_FECollection    fec_nd(order, dim);
         FiniteElementSpace fespace_nd(&mesh, &fec_nd);

         GridFunction f_nd(&fespace_nd); f_nd.ProjectCoefficient(F3_coef);

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
                  new VectorFEBoundaryTangentLFIntegrator(VdF3_coef));
               lf.Assemble();

               blf.Mult(f_nd,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dVdF3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Mixed Cross Grad Curl Integrators",
          "[MixedCrossGradCurlIntegrator]"
          "[MixedVectorIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryTangentLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   FunctionCoefficient          f3_coef(f3);
   VectorFunctionCoefficient    V3_coef(dim, V3);
   VectorFunctionCoefficient  Vdf3_coef(dim, VcrossGrad_f3);
   VectorFunctionCoefficient dVdf3_coef(dim, Curl_VcrossGrad_f3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      if (type == Element::TETRAHEDRON)
      {
         mesh.ReorientTetMesh();
      }

      SECTION("Operators on H1 for element type " + std::to_string(type))
      {
         H1_FECollection    fec_h1(order, dim);
         FiniteElementSpace fespace_h1(&mesh, &fec_h1);

         GridFunction f_h1(&fespace_h1); f_h1.ProjectCoefficient(f3_coef);

         SECTION("Mapping ND to ND")
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
                  new VectorFEBoundaryTangentLFIntegrator(Vdf3_coef));
               lf.Assemble();

               blfw.Mult(f_h1,tmp_nd); tmp_nd += lf; g_nd = 0.0;
               CG(m_nd, tmp_nd, g_nd, 0, 200, cg_rtol * cg_rtol, 0.0);

               REQUIRE( g_nd.ComputeL2Error(dVdf3_coef) < tol );
            }
         }
      }
   }
}

TEST_CASE("3D Bilinear Div Div Integrators",
          "[DivDivIntegrator]"
          "[BilinearFormIntegrator]"
          "[NonlinearFormIntegrator]"
          "[VectorFEBoundaryFluxLFIntegrator]"
          "[LinearFormIntegrator]")
{
   int order = 2, n = 1, dim = 3;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   VectorFunctionCoefficient     F3_coef(dim, F3);
   FunctionCoefficient           q3_coef(q3);
   VectorFunctionCoefficient     D3_coef(dim, V3);
   MatrixFunctionCoefficient     M3_coef(dim, M3);
   MatrixFunctionCoefficient    MT3_coef(dim, MT3);
   FunctionCoefficient        zero3_coef(zero3);
   VectorFunctionCoefficient  Zero3_coef(dim, Zero3);
   FunctionCoefficient          dF3_coef(DivF3);
   FunctionCoefficient         qdF3_coef(qDivF3);
   VectorFunctionCoefficient  dqdF3_coef(dim, Grad_qDivF3);

   for (int type = (int)Element::TETRAHEDRON;
        type <= (int)Element::HEXAHEDRON; type++)
   {
      Mesh mesh(n, n, n, (Element::Type)type, 1, 2.0, 3.0, 5.0);

      SECTION("Operators on RT for element type " + std::to_string(type))
      {
         RT_FECollection    fec_rt(order - 1, dim);
         FiniteElementSpace fespace_rt(&mesh, &fec_rt);

         GridFunction f_rt(&fespace_rt); f_rt.ProjectCoefficient(F3_coef);

         SECTION("Mapping RT to RT")
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
                  new VectorFEBoundaryFluxLFIntegrator(dF3_coef));
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
                  new VectorFEBoundaryFluxLFIntegrator(qdF3_coef));
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

} // namespace bilininteg_3d
