// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#include "fem.hpp"
#include "../mesh/nurbs.hpp"
#include "../general/tic_toc.hpp"

#include "../linalg/dtensor.hpp"  // For Reshape
#include "../general/forall.hpp"

using namespace std;

namespace mfem
{

// Adapted from PADiffusionSetup3D
void SetupPatch3D(const int Q1Dx,
                  const int Q1Dy,
                  const int Q1Dz,
                  const int coeffDim,
                  const bool symmetric,
                  const Array<double> &w,
                  const Vector &j,
                  const Vector &c,
                  Vector &d)
{
   const bool const_c = (c.Size() == 1);
   MFEM_VERIFY(coeffDim < 6 ||
               !const_c, "Constant matrix coefficient not supported");

   const auto W = Reshape(w.Read(), Q1Dx,Q1Dy,Q1Dz);
   const auto J = Reshape(j.Read(), Q1Dx,Q1Dy,Q1Dz,3,3);
   const auto C = const_c ? Reshape(c.Read(), 1,1,1,1) :
                  Reshape(c.Read(), coeffDim,Q1Dx,Q1Dy,Q1Dz);
   d.SetSize(Q1Dx * Q1Dy * Q1Dz * (symmetric ? 6 : 9));
   auto D = Reshape(d.Write(), Q1Dx,Q1Dy,Q1Dz, symmetric ? 6 : 9);
   const int NE = 1;  // TODO: MFEM_FORALL_3D without e?
   MFEM_FORALL_3D(e, NE, Q1Dx, Q1Dy, Q1Dz,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1Dx)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1Dy)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1Dz)
            {
               const double J11 = J(qx,qy,qz,0,0);
               const double J21 = J(qx,qy,qz,1,0);
               const double J31 = J(qx,qy,qz,2,0);
               const double J12 = J(qx,qy,qz,0,1);
               const double J22 = J(qx,qy,qz,1,1);
               const double J32 = J(qx,qy,qz,2,1);
               const double J13 = J(qx,qy,qz,0,2);
               const double J23 = J(qx,qy,qz,1,2);
               const double J33 = J(qx,qy,qz,2,2);
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
               /* */               J21 * (J12 * J33 - J32 * J13) +
               /* */               J31 * (J12 * J23 - J22 * J13);
               const double w_detJ = W(qx,qy,qz) / detJ;
               // adj(J)
               const double A11 = (J22 * J33) - (J23 * J32);
               const double A12 = (J32 * J13) - (J12 * J33);
               const double A13 = (J12 * J23) - (J22 * J13);
               const double A21 = (J31 * J23) - (J21 * J33);
               const double A22 = (J11 * J33) - (J13 * J31);
               const double A23 = (J21 * J13) - (J11 * J23);
               const double A31 = (J21 * J32) - (J31 * J22);
               const double A32 = (J31 * J12) - (J11 * J32);
               const double A33 = (J11 * J22) - (J12 * J21);

               if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
               {
                  // Compute entries of R = MJ^{-T} = M adj(J)^T, without det J.
                  const double M11 = C(0, qx,qy,qz);
                  const double M12 = C(1, qx,qy,qz);
                  const double M13 = C(2, qx,qy,qz);
                  const double M21 = (!symmetric) ? C(3, qx,qy,qz) : M12;
                  const double M22 = (!symmetric) ? C(4, qx,qy,qz) : C(3, qx,qy,qz);
                  const double M23 = (!symmetric) ? C(5, qx,qy,qz) : C(4, qx,qy,qz);
                  const double M31 = (!symmetric) ? C(6, qx,qy,qz) : M13;
                  const double M32 = (!symmetric) ? C(7, qx,qy,qz) : M23;
                  const double M33 = (!symmetric) ? C(8, qx,qy,qz) : C(5, qx,qy,qz);

                  const double R11 = M11*A11 + M12*A12 + M13*A13;
                  const double R12 = M11*A21 + M12*A22 + M13*A23;
                  const double R13 = M11*A31 + M12*A32 + M13*A33;
                  const double R21 = M21*A11 + M22*A12 + M23*A13;
                  const double R22 = M21*A21 + M22*A22 + M23*A23;
                  const double R23 = M21*A31 + M22*A32 + M23*A33;
                  const double R31 = M31*A11 + M32*A12 + M33*A13;
                  const double R32 = M31*A21 + M32*A22 + M33*A23;
                  const double R33 = M31*A31 + M32*A32 + M33*A33;

                  // Now set D to J^{-1} R = adj(J) R
                  D(qx,qy,qz,0) = w_detJ * (A11*R11 + A12*R21 + A13*R31); // 1,1
                  const double D12 = w_detJ * (A11*R12 + A12*R22 + A13*R32);
                  D(qx,qy,qz,1) = D12; // 1,2
                  D(qx,qy,qz,2) = w_detJ * (A11*R13 + A12*R23 + A13*R33); // 1,3

                  const double D21 = w_detJ * (A21*R11 + A22*R21 + A23*R31);
                  const double D22 = w_detJ * (A21*R12 + A22*R22 + A23*R32);
                  const double D23 = w_detJ * (A21*R13 + A22*R23 + A23*R33);

                  const double D33 = w_detJ * (A31*R13 + A32*R23 + A33*R33);

                  D(qx,qy,qz,3) = symmetric ? D22 : D21; // 2,2 or 2,1
                  D(qx,qy,qz,4) = symmetric ? D23 : D22; // 2,3 or 2,2
                  D(qx,qy,qz,5) = symmetric ? D33 : D23; // 3,3 or 2,3

                  if (!symmetric)
                  {
                     D(qx,qy,qz,6) = w_detJ * (A31*R11 + A32*R21 + A33*R31); // 3,1
                     D(qx,qy,qz,7) = w_detJ * (A31*R12 + A32*R22 + A33*R32); // 3,2
                     D(qx,qy,qz,8) = D33; // 3,3
                  }
               }
               else  // Vector or scalar coefficient version
               {
                  const double C1 = const_c ? C(0,0,0,0) : C(0,qx,qy,qz);
                  const double C2 = const_c ? C(0,0,0,0) :
                                    (coeffDim == 3 ? C(1,qx,qy,qz) : C(0,qx,qy,qz));
                  const double C3 = const_c ? C(0,0,0,0) :
                                    (coeffDim == 3 ? C(2,qx,qy,qz) : C(0,qx,qy,qz));

                  // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
                  D(qx,qy,qz,0) = w_detJ * (C1*A11*A11 + C2*A12*A12 + C3*A13*A13); // 1,1
                  D(qx,qy,qz,1) = w_detJ * (C1*A11*A21 + C2*A12*A22 + C3*A13*A23); // 2,1
                  D(qx,qy,qz,2) = w_detJ * (C1*A11*A31 + C2*A12*A32 + C3*A13*A33); // 3,1
                  D(qx,qy,qz,3) = w_detJ * (C1*A21*A21 + C2*A22*A22 + C3*A23*A23); // 2,2
                  D(qx,qy,qz,4) = w_detJ * (C1*A21*A31 + C2*A22*A32 + C3*A23*A33); // 3,2
                  D(qx,qy,qz,5) = w_detJ * (C1*A31*A31 + C2*A32*A32 + C3*A33*A33); // 3,3
               }
            }
         }
      }
   });
}

void SolveNNLS(DenseMatrix & Gt, Vector const& w, Vector & sol)
{
#ifndef MFEM_USE_LAPACK
   MFEM_ABORT("MFEM must be built with LAPACK in order to use NNLS");
#else
   NNLS nnls;
   nnls.SetVerbosity(2);

   Vector rhs_ub(Gt.NumCols());
   Gt.MultTranspose(w, rhs_ub);

   Vector rhs_lb(rhs_ub);
   Vector rhs_Gw(rhs_ub);

   const double delta = 1.0e-11;
   for (int i=0; i<rhs_ub.Size(); ++i)
   {
      rhs_lb(i) -= delta;
      rhs_ub(i) += delta;
   }

   nnls.NormalizeConstraints(Gt, rhs_lb, rhs_ub);
   nnls.Solve(Gt, rhs_lb, rhs_ub, sol);

   int nnz = 0;
   for (int i=0; i<sol.Size(); ++i)
   {
      if (sol(i) != 0.0)
      {
         nnz++;
      }
   }

   mfem::out << "Number of nonzeros in MFEM NNLS solution: " << nnz
             << ", out of " << sol.Size() << endl;

   // Check residual of NNLS solution
   Vector res(Gt.NumCols());
   Gt.MultTranspose(sol, res);

   const double normGsol = res.Norml2();
   const double normRHS = rhs_Gw.Norml2();

   res -= rhs_Gw;
   const double relNorm = res.Norml2() / std::max(normGsol, normRHS);
   mfem::out << "Relative residual norm for MFEM NNLS solution of Gs = Gw: "
             << relNorm << endl;
#endif
}

void GetReducedRule(const int nq, const int nd,
                    Array2D<double> const& B,
                    Array2D<double> const& G,
                    std::vector<int> minQ,
                    std::vector<int> maxQ,
                    std::vector<int> minD,
                    std::vector<int> maxD,
                    std::vector<int> minDD,
                    std::vector<int> maxDD,
                    const IntegrationRule *ir,
                    const bool zeroOrder,
                    std::vector<Vector> & reducedWeights,
                    std::vector<std::vector<int>> & reducedIDs)
{
   MFEM_VERIFY(B.NumRows() == nq, "");
   MFEM_VERIFY(B.NumCols() == nd, "");
   MFEM_VERIFY(G.NumRows() == nq, "");
   MFEM_VERIFY(G.NumCols() == nd, "");
   MFEM_VERIFY(ir->GetNPoints() == nq, "");

   for (int dof=0; dof<nd; ++dof)
   {
      // Integrate diffusion for B(:,dof) against all other B(:,i)

      const int nc_dof = maxDD[dof] - minDD[dof] + 1;
      const int nw_dof = maxD[dof] - minD[dof] + 1;

      mfem::out << "NNLS system size " << nc_dof << " by " << nw_dof << endl;

      // G is of size nc_dof x nw_dof
      MFEM_VERIFY(nc_dof <= nw_dof, "");

      DenseMatrix Gt(nw_dof, nc_dof);
      Gt = 0.0;

      Vector w(nw_dof);
      w = 0.0;

      for (int qx = minD[dof]; qx <= maxD[dof]; ++qx)
      {
         const double Bq = zeroOrder ? B(qx,dof) : G(qx,dof);

         const IntegrationPoint &ip = ir->IntPoint(qx);
         const double w_qx = ip.weight;
         w[qx - minD[dof]] = w_qx;

         for (int dx = minQ[qx]; dx <= maxQ[qx]; ++dx)
         {
            const double Bd = zeroOrder ? B(qx,dx) : G(qx,dx);

            Gt(qx - minD[dof], dx - minDD[dof]) = Bq * Bd;
         }
      }

      Vector sol(Gt.NumRows());

      SolveNNLS(Gt, w, sol);
      int nnz = 0;
      for (int i=0; i<sol.Size(); ++i)
      {
         if (sol(i) != 0.0)
         {
            nnz++;
         }
      }

      MFEM_VERIFY(nnz > 0, "");

      Vector wred(nnz);
      std::vector<int> idnnz(nnz);
      nnz = 0;
      for (int i=0; i<sol.Size(); ++i)
      {
         if (sol(i) != 0.0)
         {
            wred[nnz] = sol[i];
            idnnz[nnz] = i;
            nnz++;
         }
      }

      reducedWeights.push_back(wred);
      reducedIDs.push_back(idnnz);
   }
}

// Adapted from AssemblePA
void DiffusionIntegrator::SetupPatchPA(const int patch, Mesh *mesh,
                                       bool unitWeights)
{
   const Array<int>& Q1D = pQ1D[patch];
   const Array<int>& D1D = pD1D[patch];
   const std::vector<Array2D<double>>& B = pB[patch];
   const std::vector<Array2D<double>>& G = pG[patch];

   const std::vector<std::vector<int>>& minD = pminD[patch];
   const std::vector<std::vector<int>>& maxD = pmaxD[patch];
   const std::vector<std::vector<int>>& minQ = pminQ[patch];
   const std::vector<std::vector<int>>& maxQ = pmaxQ[patch];

   const std::vector<std::vector<int>>& minDD = pminDD[patch];
   const std::vector<std::vector<int>>& maxDD = pmaxDD[patch];

   const Array<const IntegrationRule*>& ir1d = pir1d[patch];

   MFEM_VERIFY(Q1D.Size() == 3, "");

   const int dims = dim;  // TODO: generalize
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

   int nq = Q1D[0];
   for (int i=1; i<dim; ++i)
   {
      nq *= Q1D[i];
   }

   int coeffDim = 1;
   Vector coeff;
   Array<double> weights(nq);
   const int MQfullDim = MQ ? MQ->GetHeight() * MQ->GetWidth() : 0;
   IntegrationPoint ip;

   Vector jac(dim * dim * nq);  // Computed as in GeometricFactors::Compute

   for (int qz=0; qz<Q1D[2]; ++qz)
   {
      for (int qy=0; qy<Q1D[1]; ++qy)
      {
         for (int qx=0; qx<Q1D[0]; ++qx)
         {
            const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
            patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);
            const int e = patchRule->GetPointElement(patch, qx, qy, qz);
            ElementTransformation *tr = mesh->GetElementTransformation(e);

            weights[p] = ip.weight;

            tr->SetIntPoint(&ip);

            const DenseMatrix& Jp = tr->Jacobian();
            for (int i=0; i<dim; ++i)
               for (int j=0; j<dim; ++j)
               {
                  jac[p + ((i + (j * dim)) * nq)] = Jp(i,j);
               }
         }
      }
   }

   if (auto *SMQ = dynamic_cast<SymmetricMatrixCoefficient *>(MQ))
   {
      MFEM_VERIFY(SMQ->GetSize() == dim, "");
      coeffDim = symmDims;
      coeff.SetSize(symmDims * nq);

      DenseSymmetricMatrix sym_mat;
      sym_mat.SetSize(dim);

      auto C = Reshape(coeff.HostWrite(), symmDims, nq);
      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
               const int e = patchRule->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               SMQ->Eval(sym_mat, *tr, ip);
               int cnt = 0;
               for (int i=0; i<dim; ++i)
                  for (int j=i; j<dim; ++j, ++cnt)
                  {
                     C(cnt, p) = sym_mat(i,j);
                  }
            }
         }
      }
   }
   else if (MQ)
   {
      symmetric = false;
      MFEM_VERIFY(MQ->GetHeight() == dim && MQ->GetWidth() == dim, "");

      coeffDim = MQfullDim;

      coeff.SetSize(MQfullDim * nq);

      DenseMatrix mat;
      mat.SetSize(dim);

      auto C = Reshape(coeff.HostWrite(), MQfullDim, nq);
      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
               const int e = patchRule->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               MQ->Eval(mat, *tr, ip);
               for (int i=0; i<dim; ++i)
                  for (int j=0; j<dim; ++j)
                  {
                     C(j+(i*dim), p) = mat(i,j);
                  }
            }
         }
      }
   }
   else if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == dim, "");
      coeffDim = VQ->GetVDim();
      coeff.SetSize(coeffDim * nq);
      auto C = Reshape(coeff.HostWrite(), coeffDim, nq);
      Vector DM(coeffDim);
      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
               const int e = patchRule->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               VQ->Eval(DM, *tr, ip);
               for (int i=0; i<coeffDim; ++i)
               {
                  C(i, p) = DM[i];
               }
            }
         }
      }
   }
   else if (Q == nullptr)
   {
      coeff.SetSize(1);
      coeff(0) = 1.0;
   }
   else if (ConstantCoefficient* cQ = dynamic_cast<ConstantCoefficient*>(Q))
   {
      coeff.SetSize(1);
      coeff(0) = cQ->constant;
   }
   else if (dynamic_cast<QuadratureFunctionCoefficient*>(Q))
   {
      MFEM_ABORT("QuadratureFunction not supported yet\n");
   }
   else
   {
      coeff.SetSize(nq);
      auto C = Reshape(coeff.HostWrite(), nq);
      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
               const int e = patchRule->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRule->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               C(p) = Q->Eval(*tr, ip);
            }
         }
      }
   }

   if (unitWeights)
   {
      weights = 1.0;
   }

   SetupPatch3D(Q1D[0], Q1D[1], Q1D[2], coeffDim, symmetric, weights, jac,
                coeff, pa_data);

   numPatches = mesh->NURBSext->GetNP();

   if (!reducedRule)
   {
      return;
   }

   // Solve for reduced 1D quadrature rules
   reducedWeights.resize(numPatches);
   reducedIDs.resize(numPatches);
   reducedWeights[patch].resize(dim);
   reducedIDs[patch].resize(dim);

   const int numTypes = 2;  // Number of rule types

   for (int d=0; d<dim; ++d)
   {
      // The reduced rules could be cached to avoid repeated computation, but
      // the cost of this setup seems low.
      reducedWeights[patch][d].resize(numTypes);
      reducedIDs[patch][d].resize(numTypes);

      GetReducedRule(Q1D[d], D1D[d], B[d], G[d],
                     minQ[d], maxQ[d],
                     minD[d], maxD[d],
                     minDD[d], maxDD[d], ir1d[d], true,
                     reducedWeights[patch][d][0], reducedIDs[patch][d][0]);
      GetReducedRule(Q1D[d], D1D[d], B[d], G[d],
                     minQ[d], maxQ[d],
                     minD[d], maxD[d],
                     minDD[d], maxDD[d], ir1d[d], false,
                     reducedWeights[patch][d][1], reducedIDs[patch][d][1]);
   }
}

// This version uses full 1D quadrature rules, neglecting to take advantage of
// the limited interaction between basis functions and integration points.
// TODO: remove this, or keep it for reference?
void DiffusionIntegrator::AssemblePatchMatrix_simpleButInefficient(
   const int patch,
   Mesh *mesh,
   DenseMatrix &pmat)
{
   MFEM_VERIFY(patchRule, "patchRule must be defined");
   dim = patchRule->GetDim();
   const int spaceDim = dim;  // TODO: generalize?

   Array<const KnotVector*> pkv;
   mesh->NURBSext->GetPatchKnotVectors(patch, pkv);
   MFEM_VERIFY(pkv.Size() == dim, "");

   if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == spaceDim,
                  "Unexpected dimension for VectorCoefficient");
   }
   if (MQ)
   {
      MFEM_VERIFY(MQ->GetWidth() == spaceDim,
                  "Unexpected width for MatrixCoefficient");
      MFEM_VERIFY(MQ->GetHeight() == spaceDim,
                  "Unexpected height for MatrixCoefficient");
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix M(MQ ? spaceDim : 0);
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   M.SetSize(MQ ? spaceDim : 0);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif

   MFEM_VERIFY(pkv.Size() == dim, "Sanity check");

   Array<int> Q1D(dim);
   Array<int> orders(dim);
   Array<int> D1D(dim);
   std::vector<Array2D<double>> B(dim);
   std::vector<Array2D<double>> G(dim);
   Array<const IntegrationRule*> ir1d(dim);

   for (int d=0; d<dim; ++d)
   {
      ir1d[d] = patchRule->GetPatchRule1D(patch, d);

      Q1D[d] = ir1d[d]->GetNPoints();

      orders[d] = pkv[d]->GetOrder();
      D1D[d] = mesh->NURBSext->NumPatchDofs1D(patch,d);

      Vector shapeKV(orders[d]+1);
      Vector dshapeKV(orders[d]+1);

      B[d].SetSize(Q1D[d], D1D[d]);
      G[d].SetSize(Q1D[d], D1D[d]);

      B[d] = 0.0;
      G[d] = 0.0;

      const Array<int>& knotSpan1D = patchRule->GetPatchRule1D_KnotSpan(patch, d);
      MFEM_VERIFY(knotSpan1D.Size() == Q1D[d], "");

      for (int i = 0; i < Q1D[d]; i++)
      {
         const IntegrationPoint &ip = ir1d[d]->IntPoint(i);
         const int ijk = knotSpan1D[i];
         const double kv0 = (*pkv[d])[orders[d] + ijk];
         double kv1 = (*pkv[d])[0];
         for (int j = orders[d] + ijk + 1; j < pkv[d]->Size(); ++j)
         {
            if ((*pkv[d])[j] > kv0)
            {
               kv1 = (*pkv[d])[j];
               break;
            }
         }

         MFEM_VERIFY(kv1 > kv0, "");

         pkv[d]->CalcShape(shapeKV, ijk, (ip.x - kv0) / (kv1 - kv0));
         pkv[d]->CalcDShape(dshapeKV, ijk, (ip.x - kv0) / (kv1 - kv0));

         // Put shapeKV into array B storing shapes for all points.
         // TODO: This should be based on NURBS3DFiniteElement::CalcShape and CalcDShape.
         // For now, it works under the assumption that all NURBS weights are 1.
         for (int j=0; j<orders[d]+1; ++j)
         {
            B[d](i,ijk + j) = shapeKV[j];
            G[d](i,ijk + j) = dshapeKV[j];
         }
      }
   }

   int ndof = D1D[0];
   for (int d=1; d<dim; ++d)
   {
      ndof *= D1D[d];
   }

   pmat.SetSize(ndof, ndof);
   pmat = 0.0;

   MFEM_VERIFY(3 == dim, "Only 3D so far");

   // For each point in patchRule, get the corresponding element and element
   // reference point, in order to use element transformations. This requires
   // data set up in NURBSPatchRule::SetPointToElement.
   SetupPatchPA(patch, mesh);

   const auto qd = Reshape(pa_data.Read(), Q1D[0]*Q1D[1]*Q1D[2],
                           (symmetric ? 6 : 9));

   // NOTE: the following is adapted from PADiffusionApply3D.
   std::vector<Array3D<double>> grad(dim);

   for (int d=0; d<dim; ++d)
   {
      grad[d].SetSize(Q1D[0], Q1D[1], Q1D[2]);
   }

   // NOTE: this is an inefficient loop over mat-vec mults of standard basis
   // vectors, just to verify the patch matrix assembly using sum factorization.
   // A more efficient version is AssemblePatchMatrix_fullQuadrature.
   Vector ej(ndof);

   for (int dof_j=0; dof_j<ndof; ++dof_j)
   {
      ej = 0.0;
      ej[dof_j] = 1.0;

      for (int d=0; d<dim; ++d)
         for (int qz = 0; qz < Q1D[2]; ++qz)
         {
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  grad[d](qx,qy,qz) = 0.0;
               }
            }
         }
      for (int dz = 0; dz < D1D[2]; ++dz)
      {
         Array3D<double> gradXY(Q1D[0], Q1D[1], dim);

         for (int qy = 0; qy < Q1D[1]; ++qy)
         {
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               for (int d=0; d<dim; ++d)
               {
                  gradXY(qx,qy,d) = 0.0;
               }
            }
         }
         for (int dy = 0; dy < D1D[1]; ++dy)
         {
            Array2D<double> gradX(Q1D[0],2);
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               gradX(qx,0) = 0.0;
               gradX(qx,1) = 0.0;
            }
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               const double s = ej[dx + (D1D[0] * (dy + (D1D[1] * dz)))];

               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  gradX(qx,0) += s * B[0](qx,dx);
                  gradX(qx,1) += s * G[0](qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               const double wy  = B[1](qy,dy);
               const double wDy = G[1](qy,dy);
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  const double wx  = gradX(qx,0);
                  const double wDx = gradX(qx,1);
                  gradXY(qx,qy,0) += wDx * wy;
                  gradXY(qx,qy,1) += wx  * wDy;
                  gradXY(qx,qy,2) += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D[2]; ++qz)
         {
            const double wz  = B[2](qz,dz);
            const double wDz = G[2](qz,dz);
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  grad[0](qx,qy,qz) += gradXY(qx,qy,0) * wz;
                  grad[1](qx,qy,qz) += gradXY(qx,qy,1) * wz;
                  grad[2](qx,qy,qz) += gradXY(qx,qy,2) * wDz;
               }
            }
         }
      }
      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < Q1D[2]; ++qz)
      {
         for (int qy = 0; qy < Q1D[1]; ++qy)
         {
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
               const double O11 = qd(q,0);
               const double O12 = qd(q,1);
               const double O13 = qd(q,2);
               const double O21 = symmetric ? O12 : qd(q,3);
               const double O22 = symmetric ? qd(q,3) : qd(q,4);
               const double O23 = symmetric ? qd(q,4) : qd(q,5);
               const double O31 = symmetric ? O13 : qd(q,6);
               const double O32 = symmetric ? O23 : qd(q,7);
               const double O33 = symmetric ? qd(q,5) : qd(q,8);
               const double gradX = grad[0](qx,qy,qz);
               const double gradY = grad[1](qx,qy,qz);
               const double gradZ = grad[2](qx,qy,qz);
               grad[0](qx,qy,qz) = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[1](qx,qy,qz) = (O21*gradX)+(O22*gradY)+(O23*gradZ);
               grad[2](qx,qy,qz) = (O31*gradX)+(O32*gradY)+(O33*gradZ);
            }
         }
      }

      for (int qz = 0; qz < Q1D[2]; ++qz)
      {
         Array3D<double> gradXY(D1D[0], D1D[1], dim);
         for (int dy = 0; dy < D1D[1]; ++dy)
         {
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int d=0; d<dim; ++d)
               {
                  gradXY(dx,dy,d) = 0.0;
               }
            }
         }
         for (int qy = 0; qy < Q1D[1]; ++qy)
         {
            Array2D<double> gradX(D1D[0], dim);

            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int d=0; d<dim; ++d)
               {
                  gradX(dx,d) = 0.0;
               }
            }
            for (int qx = 0; qx < Q1D[0]; ++qx)
            {
               const double gX = grad[0](qx,qy,qz);
               const double gY = grad[1](qx,qy,qz);
               const double gZ = grad[2](qx,qy,qz);
               for (int dx = 0; dx < D1D[0]; ++dx)
               {
                  const double wx  = B[0](qx,dx);
                  const double wDx = G[0](qx,dx);
                  gradX(dx,0) += gX * wDx;
                  gradX(dx,1) += gY * wx;
                  gradX(dx,2) += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D[1]; ++dy)
            {
               const double wy  = B[1](qy,dy);
               const double wDy = G[1](qy,dy);
               for (int dx = 0; dx < D1D[0]; ++dx)
               {
                  gradXY(dx,dy,0) += gradX(dx,0) * wy;
                  gradXY(dx,dy,1) += gradX(dx,1) * wDy;
                  gradXY(dx,dy,2) += gradX(dx,2) * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D[2]; ++dz)
         {
            const double wz  = B[2](qz,dz);
            const double wDz = G[2](qz,dz);
            for (int dy = 0; dy < D1D[1]; ++dy)
            {
               for (int dx = 0; dx < D1D[0]; ++dx)
               {
                  pmat(dx + (D1D[0] * (dy + (D1D[1] * dz))), dof_j) += ((gradXY(dx,dy,0) * wz) +
                                                                        (gradXY(dx,dy,1) * wz) +
                                                                        (gradXY(dx,dy,2) * wDz));
               }
            }
         }
      }
   }
}

// This version uses full 1D quadrature rules, taking into account the
// minimum interaction between basis functions and integration points.
void DiffusionIntegrator::AssemblePatchMatrix_fullQuadrature(
   const int patch, Mesh *mesh, SparseMatrix*& smat)
{
   MFEM_VERIFY(patchRule, "patchRule must be defined");
   dim = patchRule->GetDim();
   const int spaceDim = dim;  // TODO: generalize?

   if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == spaceDim,
                  "Unexpected dimension for VectorCoefficient");
   }
   if (MQ)
   {
      MFEM_VERIFY(MQ->GetWidth() == spaceDim,
                  "Unexpected width for MatrixCoefficient");
      MFEM_VERIFY(MQ->GetHeight() == spaceDim,
                  "Unexpected height for MatrixCoefficient");
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix M(MQ ? spaceDim : 0);
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   M.SetSize(MQ ? spaceDim : 0);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif

   SetupPatchBasisData(mesh, patch);

   SetupPatchPA(patch, mesh);

   const Array<int>& Q1D = pQ1D[patch];
   const Array<int>& D1D = pD1D[patch];
   const std::vector<Array2D<double>>& B = pB[patch];
   const std::vector<Array2D<double>>& G = pG[patch];

   const std::vector<std::vector<int>>& minD = pminD[patch];
   const std::vector<std::vector<int>>& maxD = pmaxD[patch];
   const std::vector<std::vector<int>>& minQ = pminQ[patch];
   const std::vector<std::vector<int>>& maxQ = pmaxQ[patch];

   const std::vector<std::vector<int>>& minDD = pminDD[patch];
   const std::vector<std::vector<int>>& maxDD = pmaxDD[patch];

   int ndof = D1D[0];
   for (int d=1; d<dim; ++d)
   {
      ndof *= D1D[d];
   }

   MFEM_VERIFY(3 == dim, "Only 3D so far");

   // Setup quadrature point data.
   const auto qd = Reshape(pa_data.Read(), Q1D[0]*Q1D[1]*Q1D[2],
                           (symmetric ? 6 : 9));

   // NOTE: the following is adapted from PADiffusionApply3D.
   std::vector<Array3D<double>> grad(dim);

   for (int d=0; d<dim; ++d)
   {
      grad[d].SetSize(Q1D[0], Q1D[1], Q1D[2]);
   }

   Array3D<double> gradDXY(D1D[0], D1D[1], dim);
   Array2D<double> gradDX(D1D[0], dim);

   int nd[3];
   Array3D<int> cdofs;

   int *smati = nullptr;
   int *smatj = nullptr;
   double *smata = nullptr;
   int nnz = 0;

   Array<int> maxw(dim);
   maxw = 0;

   for (int d=0; d<dim; ++d)
   {
      for (int i=0; i<D1D[d]; ++i)
      {
         maxw[d] = std::max(maxw[d], maxDD[d][i] - minDD[d][i] + 1);
      }
   }

   cdofs.SetSize(maxw[0], maxw[1], maxw[2]);

   // Compute sparsity of the sparse matrix
   smati = new int[ndof+1];
   smati[0] = 0;

   for (int dof_j=0; dof_j<ndof; ++dof_j)
   {
      const int jdz = dof_j / (D1D[0] * D1D[1]);
      const int jdy = (dof_j - (jdz * D1D[0] * D1D[1])) / D1D[0];
      const int jdx = dof_j - (jdz * D1D[0] * D1D[1]) - (jdy * D1D[0]);

      MFEM_VERIFY(jdx + (D1D[0] * (jdy + (D1D[1] * jdz))) == dof_j, "");

      const int jd[3] = {jdx, jdy, jdz};
      int ndd = 1;
      for (int i=0; i<dim; ++i)
      {
         nd[i] = maxDD[i][jd[i]] - minDD[i][jd[i]] + 1;
         ndd *= nd[i];
      }

      smati[dof_j + 1] = smati[dof_j] + ndd;
      nnz += ndd;
   }

   smatj = new int[nnz];
   smata = new double[nnz];

   for (int i=0; i<nnz; ++i)
   {
      smatj[i] = -1;
      smata[i] = 0.0;
   }

   for (int dof_j=0; dof_j<ndof; ++dof_j)
   {
      const int jdz = dof_j / (D1D[0] * D1D[1]);
      const int jdy = (dof_j - (jdz * D1D[0] * D1D[1])) / D1D[0];
      const int jdx = dof_j - (jdz * D1D[0] * D1D[1]) - (jdy * D1D[0]);

      const int jd[3] = {jdx, jdy, jdz};
      for (int i=0; i<dim; ++i)
      {
         nd[i] = maxDD[i][jd[i]] - minDD[i][jd[i]] + 1;
      }

      for (int i=0; i<nd[0]; ++i)
         for (int j=0; j<nd[1]; ++j)
            for (int k=0; k<nd[2]; ++k)
            {
               cdofs(i,j,k) = minDD[0][jdx] + i + (D1D[0] *
                                                   (minDD[1][jdy] + j + (D1D[1] * (minDD[2][jdz] + k))));
            }

      for (int d=0; d<dim; ++d)
         for (int qz = minD[2][jdz]; qz <= maxD[2][jdz]; ++qz)
         {
            for (int qy = minD[1][jdy]; qy <= maxD[1][jdy]; ++qy)
            {
               for (int qx = minD[0][jdx]; qx <= maxD[0][jdx]; ++qx)
               {
                  grad[d](qx,qy,qz) = 0.0;
               }
            }
         }

      for (int qz = minD[2][jdz]; qz <= maxD[2][jdz]; ++qz)
      {
         const double wz  = B[2](qz,jdz);
         const double wDz = G[2](qz,jdz);

         for (int qy = minD[1][jdy]; qy <= maxD[1][jdy]; ++qy)
         {
            const double wy  = B[1](qy,jdy);
            const double wDy = G[1](qy,jdy);

            for (int qx = minD[0][jdx]; qx <= maxD[0][jdx]; ++qx)
            {
               const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
               const double O11 = qd(q,0);
               const double O12 = qd(q,1);
               const double O13 = qd(q,2);
               const double O21 = symmetric ? O12 : qd(q,3);
               const double O22 = symmetric ? qd(q,3) : qd(q,4);
               const double O23 = symmetric ? qd(q,4) : qd(q,5);
               const double O31 = symmetric ? O13 : qd(q,6);
               const double O32 = symmetric ? O23 : qd(q,7);
               const double O33 = symmetric ? qd(q,5) : qd(q,8);

               const double wx  = B[0](qx,jdx);
               const double wDx = G[0](qx,jdx);

               const double gradX = wDx * wy * wz;
               const double gradY = wx  * wDy * wz;
               const double gradZ = wx  * wy * wDz;

               grad[0](qx,qy,qz) = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[1](qx,qy,qz) = (O21*gradX)+(O22*gradY)+(O23*gradZ);
               grad[2](qx,qy,qz) = (O31*gradX)+(O32*gradY)+(O33*gradZ);
            }
         }
      }

      for (int qz = minD[2][jdz]; qz <= maxD[2][jdz]; ++qz)
      {
         for (int dy = minDD[1][jdy]; dy <= maxDD[1][jdy]; ++dy)
         {
            for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
            {
               for (int d=0; d<dim; ++d)
               {
                  gradDXY(dx,dy,d) = 0.0;
               }
            }
         }
         for (int qy = minD[1][jdy]; qy <= maxD[1][jdy]; ++qy)
         {
            for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
            {
               for (int d=0; d<dim; ++d)
               {
                  gradDX(dx,d) = 0.0;
               }
            }
            for (int qx = minD[0][jdx]; qx <= maxD[0][jdx]; ++qx)
            {
               const double gX = grad[0](qx,qy,qz);
               const double gY = grad[1](qx,qy,qz);
               const double gZ = grad[2](qx,qy,qz);
               for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
               {
                  const double wx  = B[0](qx,dx);
                  const double wDx = G[0](qx,dx);
                  gradDX(dx,0) += gX * wDx;
                  gradDX(dx,1) += gY * wx;
                  gradDX(dx,2) += gZ * wx;
               }
            }
            for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
            {
               const double wy  = B[1](qy,dy);
               const double wDy = G[1](qy,dy);
               for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
               {
                  gradDXY(dx,dy,0) += gradDX(dx,0) * wy;
                  gradDXY(dx,dy,1) += gradDX(dx,1) * wDy;
                  gradDXY(dx,dy,2) += gradDX(dx,2) * wy;
               }
            }
         }
         for (int dz = minQ[2][qz]; dz <= maxQ[2][qz]; ++dz)
         {
            const double wz  = B[2](qz,dz);
            const double wDz = G[2](qz,dz);
            for (int dy = minDD[1][jdy]; dy <= maxDD[1][jdy]; ++dy)
            {
               for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
               {
                  const double v = (gradDXY(dx,dy,0) * wz) +
                                   (gradDXY(dx,dy,1) * wz) +
                                   (gradDXY(dx,dy,2) * wDz);

                  const int loc = dx - minDD[0][jd[0]] + (nd[0] * (dy - minDD[1][jd[1]] +
                                                                   (nd[1] * (dz - minDD[2][jd[2]]))));

                  const int odof = cdofs(dx - minDD[0][jd[0]],
                                         dy - minDD[1][jd[1]],
                                         dz - minDD[2][jd[2]]);

                  const int m = smati[dof_j] + loc;
                  MFEM_ASSERT(smatj[m] == odof || smatj[m] == -1, "");
                  smatj[m] = odof;
                  smata[m] += v;
               } // dx
            } // dy
         } // dz
      } // qz
   } // dof_j

   // Note that smat takes ownership of its input data.
   smat = new SparseMatrix(smati, smatj, smata, ndof, ndof);
}

void DiffusionIntegrator::SetupPatchBasisData(Mesh *mesh, unsigned int patch)
{
   MFEM_VERIFY(pB.size() == patch && pG.size() == patch, "");
   MFEM_VERIFY(pQ1D.size() == patch && pD1D.size() == patch, "");
   MFEM_VERIFY(pminQ.size() == patch && pmaxQ.size() == patch, "");
   MFEM_VERIFY(pminD.size() == patch && pmaxD.size() == patch, "");
   MFEM_VERIFY(pminDD.size() == patch && pmaxDD.size() == patch, "");
   MFEM_VERIFY(pir1d.size() == patch, "");

   // Set basis functions and gradients for this patch
   Array<const KnotVector*> pkv;
   mesh->NURBSext->GetPatchKnotVectors(patch, pkv);
   MFEM_VERIFY(pkv.Size() == dim, "");

   Array<int> Q1D(dim);
   Array<int> orders(dim);
   Array<int> D1D(dim);
   std::vector<Array2D<double>> B(dim);
   std::vector<Array2D<double>> G(dim);
   Array<const IntegrationRule*> ir1d(dim);

   std::vector<std::vector<int>> minD(dim);
   std::vector<std::vector<int>> maxD(dim);
   std::vector<std::vector<int>> minQ(dim);
   std::vector<std::vector<int>> maxQ(dim);

   std::vector<std::vector<int>> minDD(dim);
   std::vector<std::vector<int>> maxDD(dim);

   for (int d=0; d<dim; ++d)
   {
      ir1d[d] = patchRule->GetPatchRule1D(patch, d);

      Q1D[d] = ir1d[d]->GetNPoints();

      orders[d] = pkv[d]->GetOrder();
      D1D[d] = mesh->NURBSext->NumPatchDofs1D(patch,d);

      Vector shapeKV(orders[d]+1);
      Vector dshapeKV(orders[d]+1);

      B[d].SetSize(Q1D[d], D1D[d]);
      G[d].SetSize(Q1D[d], D1D[d]);

      minD[d].assign(D1D[d], Q1D[d]);
      maxD[d].assign(D1D[d], 0);

      minQ[d].assign(Q1D[d], D1D[d]);
      maxQ[d].assign(Q1D[d], 0);

      B[d] = 0.0;
      G[d] = 0.0;

      const Array<int>& knotSpan1D = patchRule->GetPatchRule1D_KnotSpan(patch, d);
      MFEM_VERIFY(knotSpan1D.Size() == Q1D[d], "");

      for (int i = 0; i < Q1D[d]; i++)
      {
         const IntegrationPoint &ip = ir1d[d]->IntPoint(i);
         const int ijk = knotSpan1D[i];
         const double kv0 = (*pkv[d])[orders[d] + ijk];
         double kv1 = (*pkv[d])[0];
         for (int j = orders[d] + ijk + 1; j < pkv[d]->Size(); ++j)
         {
            if ((*pkv[d])[j] > kv0)
            {
               kv1 = (*pkv[d])[j];
               break;
            }
         }

         MFEM_VERIFY(kv1 > kv0, "");

         pkv[d]->CalcShape(shapeKV, ijk, (ip.x - kv0) / (kv1 - kv0));
         pkv[d]->CalcDShape(dshapeKV, ijk, (ip.x - kv0) / (kv1 - kv0));

         // Put shapeKV into array B storing shapes for all points.
         // TODO: This should be based on NURBS3DFiniteElement::CalcShape and CalcDShape.
         // For now, it works under the assumption that all NURBS weights are 1.
         for (int j=0; j<orders[d]+1; ++j)
         {
            B[d](i,ijk + j) = shapeKV[j];
            G[d](i,ijk + j) = dshapeKV[j];

            minD[d][ijk + j] = std::min(minD[d][ijk + j], i);
            maxD[d][ijk + j] = std::max(maxD[d][ijk + j], i);
         }

         minQ[d][i] = std::min(minQ[d][i], ijk);
         maxQ[d][i] = std::max(maxQ[d][i], ijk + orders[d]);
      }

      // Determine which DOFs each DOF interacts with, in 1D.
      minDD[d].resize(D1D[d]);
      maxDD[d].resize(D1D[d]);
      for (int i=0; i<D1D[d]; ++i)
      {
         const int qmin = minD[d][i];
         minDD[d][i] = minQ[d][qmin];

         const int qmax = maxD[d][i];
         maxDD[d][i] = maxQ[d][qmax];
      }
   }

   // Push patch data to global data structures
   pB.push_back(B);
   pG.push_back(G);

   pQ1D.push_back(Q1D);
   pD1D.push_back(D1D);

   pminQ.push_back(minQ);
   pmaxQ.push_back(maxQ);

   pminD.push_back(minD);
   pmaxD.push_back(maxD);

   pminDD.push_back(minDD);
   pmaxDD.push_back(maxDD);

   pir1d.push_back(ir1d);
}

// This version uses reduced 1D quadrature rules.
void DiffusionIntegrator::AssemblePatchMatrix_reducedQuadrature(
   const int patch, Mesh *mesh, SparseMatrix*& smat)
{
   MFEM_VERIFY(patchRule, "patchRule must be defined");
   dim = patchRule->GetDim();
   const int spaceDim = dim;  // TODO: generalize?

   if (VQ)
   {
      MFEM_VERIFY(VQ->GetVDim() == spaceDim,
                  "Unexpected dimension for VectorCoefficient");
   }
   if (MQ)
   {
      MFEM_VERIFY(MQ->GetWidth() == spaceDim,
                  "Unexpected width for MatrixCoefficient");
      MFEM_VERIFY(MQ->GetHeight() == spaceDim,
                  "Unexpected height for MatrixCoefficient");
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix M(MQ ? spaceDim : 0);
   Vector D(VQ ? VQ->GetVDim() : 0);
#else
   M.SetSize(MQ ? spaceDim : 0);
   D.SetSize(VQ ? VQ->GetVDim() : 0);
#endif

   SetupPatchBasisData(mesh, patch);

   MFEM_VERIFY(3 == dim, "Only 3D so far");

   // Setup quadrature point data.
   // For each point in patchRule, get the corresponding element and element
   // reference point, in order to use element transformations. This requires
   // data set up in NURBSPatchRule::SetPointToElement.
   SetupPatchPA(patch, mesh, true);

   const Array<int>& Q1D = pQ1D[patch];
   const Array<int>& D1D = pD1D[patch];
   const std::vector<Array2D<double>>& B = pB[patch];
   const std::vector<Array2D<double>>& G = pG[patch];

   const std::vector<std::vector<int>>& minD = pminD[patch];
   const std::vector<std::vector<int>>& maxD = pmaxD[patch];
   const std::vector<std::vector<int>>& minQ = pminQ[patch];
   const std::vector<std::vector<int>>& maxQ = pmaxQ[patch];

   const std::vector<std::vector<int>>& minDD = pminDD[patch];
   const std::vector<std::vector<int>>& maxDD = pmaxDD[patch];

   int ndof = D1D[0];
   for (int d=1; d<dim; ++d)
   {
      ndof *= D1D[d];
   }

   const auto qd = Reshape(pa_data.Read(), Q1D[0]*Q1D[1]*Q1D[2],
                           (symmetric ? 6 : 9));

   // NOTE: the following is adapted from PADiffusionApply3D.
   std::vector<Array3D<double>> grad(dim);
   for (int d=0; d<dim; ++d)
   {
      grad[d].SetSize(Q1D[0], Q1D[1], Q1D[2]);
   }

   // Note that this large 3D array, most of which is unused,
   // seems inefficient, but testing showed that it is faster
   // than using a set<int> to store 1D indices of the used points.
   Array3D<bool> gradUsed;
   gradUsed.SetSize(Q1D[0], Q1D[1], Q1D[2]);

   Array3D<double> gradDXY(D1D[0], D1D[1], dim);
   Array2D<double> gradDX(D1D[0], dim);

   int nd[3];
   Array3D<int> cdofs;

   int *smati = nullptr;
   int *smatj = nullptr;
   double *smata = nullptr;
   bool bugfound = false;
   int nnz = 0;

   Array<int> maxw(dim);
   maxw = 0;

   for (int d=0; d<dim; ++d)
   {
      for (int i=0; i<D1D[d]; ++i)
      {
         maxw[d] = std::max(maxw[d], maxDD[d][i] - minDD[d][i] + 1);
      }
   }

   cdofs.SetSize(maxw[0], maxw[1], maxw[2]);

   // Compute sparsity of the sparse matrix
   smati = new int[ndof+1];
   smati[0] = 0;

   for (int dof_j=0; dof_j<ndof; ++dof_j)
   {
      const int jdz = dof_j / (D1D[0] * D1D[1]);
      const int jdy = (dof_j - (jdz * D1D[0] * D1D[1])) / D1D[0];
      const int jdx = dof_j - (jdz * D1D[0] * D1D[1]) - (jdy * D1D[0]);

      MFEM_VERIFY(jdx + (D1D[0] * (jdy + (D1D[1] * jdz))) == dof_j, "");

      const int jd[3] = {jdx, jdy, jdz};
      int ndd = 1;
      for (int i=0; i<dim; ++i)
      {
         nd[i] = maxDD[i][jd[i]] - minDD[i][jd[i]] + 1;
         ndd *= nd[i];
      }

      smati[dof_j + 1] = smati[dof_j] + ndd;
      nnz += ndd;
   }

   smatj = new int[nnz];
   smata = new double[nnz];

   for (int i=0; i<nnz; ++i)
   {
      smatj[i] = -1;
      smata[i] = 0.0;
   }

   for (int dof_j=0; dof_j<ndof; ++dof_j)
   {
      const int jdz = dof_j / (D1D[0] * D1D[1]);
      const int jdy = (dof_j - (jdz * D1D[0] * D1D[1])) / D1D[0];
      const int jdx = dof_j - (jdz * D1D[0] * D1D[1]) - (jdy * D1D[0]);

      const int jd[3] = {jdx, jdy, jdz};
      for (int i=0; i<dim; ++i)
      {
         nd[i] = maxDD[i][jd[i]] - minDD[i][jd[i]] + 1;
      }

      for (int i=0; i<nd[0]; ++i)
         for (int j=0; j<nd[1]; ++j)
            for (int k=0; k<nd[2]; ++k)
            {
               cdofs(i,j,k) = minDD[0][jdx] + i + (D1D[0] *
                                                   (minDD[1][jdy] + j + (D1D[1] * (minDD[2][jdz] + k))));
            }

      for (int qz = minD[2][jdz]; qz <= maxD[2][jdz]; ++qz)
      {
         for (int qy = minD[1][jdy]; qy <= maxD[1][jdy]; ++qy)
         {
            for (int qx = minD[0][jdx]; qx <= maxD[0][jdx]; ++qx)
            {
               gradUsed(qx,qy,qz) = false;
            }
         }
      }

      for (int zquad = 0; zquad<2; ++zquad)
      {
         // Reduced quadrature in z
         const int nwz = reducedIDs[patch][2][zquad][jdz].size();
         for (int irz=0; irz < nwz; ++irz)
         {
            const int qz = reducedIDs[patch][2][zquad][jdz][irz] + minD[2][jdz];
            const double zw = reducedWeights[patch][2][zquad][jdz][irz];

            const double gwz  = B[2](qz,jdz);
            const double gwDz = G[2](qz,jdz);

            for (int dy = minDD[1][jdy]; dy <= maxDD[1][jdy]; ++dy)
            {
               for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
               {
                  for (int d=0; d<dim; ++d)
                  {
                     gradDXY(dx,dy,d) = 0.0;
                  }
               }
            }

            for (int yquad = 0; yquad<2; ++yquad)
            {
               // Reduced quadrature in y
               const int nwy = reducedIDs[patch][1][yquad][jdy].size();
               for (int iry=0; iry < nwy; ++iry)
               {
                  const int qy = reducedIDs[patch][1][yquad][jdy][iry] + minD[1][jdy];
                  const double yw = reducedWeights[patch][1][yquad][jdy][iry];

                  const double gwy  = B[1](qy,jdy);
                  const double gwDy = G[1](qy,jdy);

                  for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
                  {
                     for (int d=0; d<dim; ++d)
                     {
                        gradDX(dx,d) = 0.0;
                     }
                  }

                  // Reduced quadrature in x
                  for (int xquad=0; xquad<2; ++xquad)
                  {
                     const int nwx = reducedIDs[patch][0][xquad][jdx].size();
                     for (int irx=0; irx < nwx; ++irx)
                     {
                        const int qx = reducedIDs[patch][0][xquad][jdx][irx] + minD[0][jdx];

                        if (!gradUsed(qx,qy,qz))
                        {
                           const double gwx  = B[0](qx,jdx);
                           const double gwDx = G[0](qx,jdx);

                           const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
                           const double O11 = qd(q,0);
                           const double O12 = qd(q,1);
                           const double O13 = qd(q,2);
                           const double O21 = symmetric ? O12 : qd(q,3);
                           const double O22 = symmetric ? qd(q,3) : qd(q,4);
                           const double O23 = symmetric ? qd(q,4) : qd(q,5);
                           const double O31 = symmetric ? O13 : qd(q,6);
                           const double O32 = symmetric ? O23 : qd(q,7);
                           const double O33 = symmetric ? qd(q,5) : qd(q,8);

                           const double gradX = gwDx * gwy * gwz;
                           const double gradY = gwx * gwDy * gwz;
                           const double gradZ = gwx * gwy * gwDz;

                           grad[0](qx,qy,qz) = (O11*gradX)+(O12*gradY)+(O13*gradZ);
                           grad[1](qx,qy,qz) = (O21*gradX)+(O22*gradY)+(O23*gradZ);
                           grad[2](qx,qy,qz) = (O31*gradX)+(O32*gradY)+(O33*gradZ);

                           gradUsed(qx,qy,qz) = true;
                        }
                     }
                  }

                  // 00 terms
                  const int nw = reducedIDs[patch][0][0][jdx].size();
                  for (int irx=0; irx < nw; ++irx)
                  {
                     const int qx = reducedIDs[patch][0][0][jdx][irx] + minD[0][jdx];

                     const double gY = grad[1](qx,qy,qz);
                     const double gZ = grad[2](qx,qy,qz);
                     const double xw = reducedWeights[patch][0][0][jdx][irx];
                     for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
                     {
                        const double wx  = B[0](qx,dx);
                        if (yquad == 1)
                        {
                           gradDX(dx,1) += gY * wx * xw;
                        }
                        if (zquad == 1)
                        {
                           gradDX(dx,2) += gZ * wx * xw;
                        }
                     }
                  }

                  // 11 terms
                  const int nw11 = reducedIDs[patch][0][1][jdx].size();

                  for (int irx=0; irx < nw11; ++irx)
                  {
                     const int qx = reducedIDs[patch][0][1][jdx][irx] + minD[0][jdx];

                     const double gX = grad[0](qx,qy,qz);
                     const double xw = reducedWeights[patch][0][1][jdx][irx];
                     for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
                     {
                        const double wDx = G[0](qx,dx);
                        gradDX(dx,0) += gX * wDx * xw;
                     }
                  }

                  for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
                  {
                     const double wy  = B[1](qy,dy);
                     const double wDy = G[1](qy,dy);
                     for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
                     {
                        if (yquad == 0)
                        {
                           if (zquad == 1)
                           {
                              gradDXY(dx,dy,2) += gradDX(dx,2) * wy * yw;
                           }
                           else
                           {
                              gradDXY(dx,dy,0) += gradDX(dx,0) * wy * yw;
                           }
                        }
                        else if (zquad == 0)
                        {
                           gradDXY(dx,dy,1) += gradDX(dx,1) * wDy * yw;
                        }
                     }
                  }
               } // qy
            } // y quadrature type
            for (int dz = minQ[2][qz]; dz <= maxQ[2][qz]; ++dz)
            {
               const double wz  = B[2](qz,dz);
               const double wDz = G[2](qz,dz);
               for (int dy = minDD[1][jdy]; dy <= maxDD[1][jdy]; ++dy)
               {
                  for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
                  {
                     double v = (zquad == 0) ? (gradDXY(dx,dy,0) * wz) +
                                (gradDXY(dx,dy,1) * wz) : gradDXY(dx,dy,2) * wDz;

                     v *= zw;

                     const int loc = dx - minDD[0][jd[0]] + (nd[0] * (dy - minDD[1][jd[1]] +
                                                                      (nd[1] * (dz - minDD[2][jd[2]]))));

                     const int odof = cdofs(dx - minDD[0][jd[0]],
                                            dy - minDD[1][jd[1]],
                                            dz - minDD[2][jd[2]]);

                     const int m = smati[dof_j] + loc;
                     if (!(smatj[m] == odof || smatj[m] == -1))
                     {
                        bugfound = true;
                     }

                     smatj[m] = odof;
                     smata[m] += v;
                  } // dx
               } // dy
            } // dz
         } // qz
      } // zquad
   } // dof_j

   MFEM_VERIFY(!bugfound, "");

   for (int i=0; i<nnz; ++i)
   {
      if (smata[i] == 0.0)
      {
         // This prevents failure of SparseMatrix EliminateRowCol.
         // TODO: is there a better solution?
         smata[i] = 1.0e-16;
      }
   }

   // Note that smat takes ownership of its input data.
   smat = new SparseMatrix(smati, smatj, smata, ndof, ndof);
}

void DiffusionIntegrator::AssemblePatchMatrix(
   const int patch, Mesh *mesh, SparseMatrix*& smat)
{
   if (reducedRule)
   {
      AssemblePatchMatrix_reducedQuadrature(patch, mesh, smat);
   }
   else
   {
      AssemblePatchMatrix_fullQuadrature(patch, mesh, smat);
   }
}

}
