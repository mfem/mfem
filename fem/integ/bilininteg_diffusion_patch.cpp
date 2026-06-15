// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#include "../fem.hpp"
#include "../../mesh/nurbs.hpp"

#include "../../linalg/dtensor.hpp"  // For Reshape
#include "../../general/forall.hpp"

using namespace std;

namespace mfem
{

// Adapted from PADiffusionSetup3D
void SetupPatch3D(const int Q1Dx,
                  const int Q1Dy,
                  const int Q1Dz,
                  const int coeffDim,
                  const bool symmetric,
                  const Array<real_t> &w,
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
               const real_t J11 = J(qx,qy,qz,0,0);
               const real_t J21 = J(qx,qy,qz,1,0);
               const real_t J31 = J(qx,qy,qz,2,0);
               const real_t J12 = J(qx,qy,qz,0,1);
               const real_t J22 = J(qx,qy,qz,1,1);
               const real_t J32 = J(qx,qy,qz,2,1);
               const real_t J13 = J(qx,qy,qz,0,2);
               const real_t J23 = J(qx,qy,qz,1,2);
               const real_t J33 = J(qx,qy,qz,2,2);
               const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
               /* */               J21 * (J12 * J33 - J32 * J13) +
               /* */               J31 * (J12 * J23 - J22 * J13);
               const real_t w_detJ = W(qx,qy,qz) / detJ;
               // adj(J)
               const real_t A11 = (J22 * J33) - (J23 * J32);
               const real_t A12 = (J32 * J13) - (J12 * J33);
               const real_t A13 = (J12 * J23) - (J22 * J13);
               const real_t A21 = (J31 * J23) - (J21 * J33);
               const real_t A22 = (J11 * J33) - (J13 * J31);
               const real_t A23 = (J21 * J13) - (J11 * J23);
               const real_t A31 = (J21 * J32) - (J31 * J22);
               const real_t A32 = (J31 * J12) - (J11 * J32);
               const real_t A33 = (J11 * J22) - (J12 * J21);

               if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
               {
                  // Compute entries of R = MJ^{-T} = M adj(J)^T, without det J.
                  const real_t M11 = C(0, qx,qy,qz);
                  const real_t M12 = C(1, qx,qy,qz);
                  const real_t M13 = C(2, qx,qy,qz);
                  const real_t M21 = (!symmetric) ? C(3, qx,qy,qz) : M12;
                  const real_t M22 = (!symmetric) ? C(4, qx,qy,qz) : C(3, qx,qy,qz);
                  const real_t M23 = (!symmetric) ? C(5, qx,qy,qz) : C(4, qx,qy,qz);
                  const real_t M31 = (!symmetric) ? C(6, qx,qy,qz) : M13;
                  const real_t M32 = (!symmetric) ? C(7, qx,qy,qz) : M23;
                  const real_t M33 = (!symmetric) ? C(8, qx,qy,qz) : C(5, qx,qy,qz);

                  const real_t R11 = M11*A11 + M12*A12 + M13*A13;
                  const real_t R12 = M11*A21 + M12*A22 + M13*A23;
                  const real_t R13 = M11*A31 + M12*A32 + M13*A33;
                  const real_t R21 = M21*A11 + M22*A12 + M23*A13;
                  const real_t R22 = M21*A21 + M22*A22 + M23*A23;
                  const real_t R23 = M21*A31 + M22*A32 + M23*A33;
                  const real_t R31 = M31*A11 + M32*A12 + M33*A13;
                  const real_t R32 = M31*A21 + M32*A22 + M33*A23;
                  const real_t R33 = M31*A31 + M32*A32 + M33*A33;

                  // Now set D to J^{-1} R = adj(J) R
                  D(qx,qy,qz,0) = w_detJ * (A11*R11 + A12*R21 + A13*R31); // 1,1
                  const real_t D12 = w_detJ * (A11*R12 + A12*R22 + A13*R32);
                  D(qx,qy,qz,1) = D12; // 1,2
                  D(qx,qy,qz,2) = w_detJ * (A11*R13 + A12*R23 + A13*R33); // 1,3

                  const real_t D21 = w_detJ * (A21*R11 + A22*R21 + A23*R31);
                  const real_t D22 = w_detJ * (A21*R12 + A22*R22 + A23*R32);
                  const real_t D23 = w_detJ * (A21*R13 + A22*R23 + A23*R33);

                  const real_t D33 = w_detJ * (A31*R13 + A32*R23 + A33*R33);

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
                  const real_t C1 = const_c ? C(0,0,0,0) : C(0,qx,qy,qz);
                  const real_t C2 = const_c ? C(0,0,0,0) :
                                    (coeffDim == 3 ? C(1,qx,qy,qz) : C(0,qx,qy,qz));
                  const real_t C3 = const_c ? C(0,0,0,0) :
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

// Compute a reduced integration rule, using NNLSSolver, for DiffusionIntegrator
// on a NURBS patch with partial assembly.
void GetReducedRule(const int nq, const int nd,
                    Array2D<real_t> const& B,
                    Array2D<real_t> const& G,
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

      // G is of size nc_dof x nw_dof
      MFEM_VERIFY(nc_dof <= nw_dof, "The NNLS system for the reduced "
                  "integration rule requires more full integration points. Try"
                  " increasing the order of the full integration rule.");
      DenseMatrix Gmat(nc_dof, nw_dof);
      Gmat = 0.0;

      Vector w(nw_dof);
      w = 0.0;

      for (int qx = minD[dof]; qx <= maxD[dof]; ++qx)
      {
         const real_t Bq = zeroOrder ? B(qx,dof) : G(qx,dof);

         const IntegrationPoint &ip = ir->IntPoint(qx);
         const real_t w_qx = ip.weight;
         w[qx - minD[dof]] = w_qx;

         for (int dx = minQ[qx]; dx <= maxQ[qx]; ++dx)
         {
            const real_t Bd = zeroOrder ? B(qx,dx) : G(qx,dx);

            Gmat(dx - minDD[dof], qx - minD[dof]) = Bq * Bd;
         }
      }

      Vector sol(Gmat.NumCols());

#ifdef MFEM_USE_LAPACK
      NNLSSolver nnls;
      nnls.SetOperator(Gmat);

      nnls.Mult(w, sol);
#else
      MFEM_ABORT("NNLSSolver requires building with LAPACK");
#endif

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
   const std::vector<Array2D<real_t>>& B = pB[patch];
   const std::vector<Array2D<real_t>>& G = pG[patch];

   const IntArrayVar2D& minD = pminD[patch];
   const IntArrayVar2D& maxD = pmaxD[patch];
   const IntArrayVar2D& minQ = pminQ[patch];
   const IntArrayVar2D& maxQ = pmaxQ[patch];

   const IntArrayVar2D& minDD = pminDD[patch];
   const IntArrayVar2D& maxDD = pmaxDD[patch];

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
   Array<real_t> weights(nq);
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
            patchRules->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);
            const int e = patchRules->GetPointElement(patch, qx, qy, qz);
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
               const int e = patchRules->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRules->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

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
               const int e = patchRules->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRules->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

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
               const int e = patchRules->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRules->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

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
               const int e = patchRules->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh->GetElementTransformation(e);
               patchRules->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

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

   if (integrationMode != PATCHWISE_REDUCED)
   {
      return;
   }

   // Solve for reduced 1D quadrature rules
   const int totalDim = numPatches * dim * numTypes;
   reducedWeights.resize(totalDim);
   reducedIDs.resize(totalDim);

   auto rw = Reshape(reducedWeights.data(), numTypes, dim, numPatches);
   auto rid = Reshape(reducedIDs.data(), numTypes, dim, numPatches);

   for (int d=0; d<dim; ++d)
   {
      // The reduced rules could be cached to avoid repeated computation, but
      // the cost of this setup seems low.
      GetReducedRule(Q1D[d], D1D[d], B[d], G[d],
                     minQ[d], maxQ[d],
                     minD[d], maxD[d],
                     minDD[d], maxDD[d], ir1d[d], true,
                     rw(0,d,patch), rid(0,d,patch));
      GetReducedRule(Q1D[d], D1D[d], B[d], G[d],
                     minQ[d], maxQ[d],
                     minD[d], maxD[d],
                     minDD[d], maxDD[d], ir1d[d], false,
                     rw(1,d,patch), rid(1,d,patch));
   }
}

// This version uses full 1D quadrature rules, taking into account the
// minimum interaction between basis functions and integration points.
void DiffusionIntegrator::AssemblePatchMatrix_fullQuadrature(
   const int patch, const FiniteElementSpace &fes, SparseMatrix*& smat)
{
   MFEM_VERIFY(patchRules, "patchRules must be defined");
   dim = patchRules->GetDim();
   const int spaceDim = dim;  // TODO: generalize?

   Mesh *mesh = fes.GetMesh();

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
   const std::vector<Array2D<real_t>>& B = pB[patch];
   const std::vector<Array2D<real_t>>& G = pG[patch];

   const IntArrayVar2D& minD = pminD[patch];
   const IntArrayVar2D& maxD = pmaxD[patch];
   const IntArrayVar2D& minQ = pminQ[patch];
   const IntArrayVar2D& maxQ = pmaxQ[patch];

   const IntArrayVar2D& minDD = pminDD[patch];
   const IntArrayVar2D& maxDD = pmaxDD[patch];

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
   std::vector<Array3D<real_t>> grad(dim);

   for (int d=0; d<dim; ++d)
   {
      grad[d].SetSize(Q1D[0], Q1D[1], Q1D[2]);
   }

   Array3D<real_t> gradDXY(D1D[0], D1D[1], dim);
   Array2D<real_t> gradDX(D1D[0], dim);

   int nd[3];
   Array3D<int> cdofs;

   int *smati = nullptr;
   int *smatj = nullptr;
   real_t *smata = nullptr;
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
   smati = Memory<int>(ndof+1);
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

   smatj = Memory<int>(nnz);
   smata = Memory<real_t>(nnz);

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
         const real_t wz  = B[2](qz,jdz);
         const real_t wDz = G[2](qz,jdz);

         for (int qy = minD[1][jdy]; qy <= maxD[1][jdy]; ++qy)
         {
            const real_t wy  = B[1](qy,jdy);
            const real_t wDy = G[1](qy,jdy);

            for (int qx = minD[0][jdx]; qx <= maxD[0][jdx]; ++qx)
            {
               const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
               const real_t O11 = qd(q,0);
               const real_t O12 = qd(q,1);
               const real_t O13 = qd(q,2);
               const real_t O21 = symmetric ? O12 : qd(q,3);
               const real_t O22 = symmetric ? qd(q,3) : qd(q,4);
               const real_t O23 = symmetric ? qd(q,4) : qd(q,5);
               const real_t O31 = symmetric ? O13 : qd(q,6);
               const real_t O32 = symmetric ? O23 : qd(q,7);
               const real_t O33 = symmetric ? qd(q,5) : qd(q,8);

               const real_t wx  = B[0](qx,jdx);
               const real_t wDx = G[0](qx,jdx);

               const real_t gradX = wDx * wy * wz;
               const real_t gradY = wx  * wDy * wz;
               const real_t gradZ = wx  * wy * wDz;

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
               const real_t gX = grad[0](qx,qy,qz);
               const real_t gY = grad[1](qx,qy,qz);
               const real_t gZ = grad[2](qx,qy,qz);
               for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
               {
                  const real_t wx  = B[0](qx,dx);
                  const real_t wDx = G[0](qx,dx);
                  gradDX(dx,0) += gX * wDx;
                  gradDX(dx,1) += gY * wx;
                  gradDX(dx,2) += gZ * wx;
               }
            }
            for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
            {
               const real_t wy  = B[1](qy,dy);
               const real_t wDy = G[1](qy,dy);
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
            const real_t wz  = B[2](qz,dz);
            const real_t wDz = G[2](qz,dz);
            for (int dy = minDD[1][jdy]; dy <= maxDD[1][jdy]; ++dy)
            {
               for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
               {
                  const real_t v = (gradDXY(dx,dy,0) * wz) +
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
   std::vector<Array2D<real_t>> B(dim);
   std::vector<Array2D<real_t>> G(dim);
   Array<const IntegrationRule*> ir1d(dim);

   IntArrayVar2D minD(dim);
   IntArrayVar2D maxD(dim);
   IntArrayVar2D minQ(dim);
   IntArrayVar2D maxQ(dim);

   IntArrayVar2D minDD(dim);
   IntArrayVar2D maxDD(dim);

   for (int d=0; d<dim; ++d)
   {
      ir1d[d] = patchRules->GetPatchRule1D(patch, d);

      Q1D[d] = ir1d[d]->GetNPoints();

      orders[d] = pkv[d]->GetOrder();
      D1D[d] = pkv[d]->GetNCP();

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

      const Array<int>& knotSpan1D = patchRules->GetPatchRule1D_KnotSpan(patch, d);
      MFEM_VERIFY(knotSpan1D.Size() == Q1D[d], "");

      for (int i = 0; i < Q1D[d]; i++)
      {
         const IntegrationPoint &ip = ir1d[d]->IntPoint(i);
         const int ijk = knotSpan1D[i];
         const real_t kv0 = (*pkv[d])[orders[d] + ijk];
         real_t kv1 = (*pkv[d])[0];
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
   const int patch, const FiniteElementSpace &fes, SparseMatrix*& smat)
{
   MFEM_VERIFY(patchRules, "patchRules must be defined");
   dim = patchRules->GetDim();
   const int spaceDim = dim;  // TODO: generalize?

   Mesh *mesh = fes.GetMesh();

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
   // For each point in patchRules, get the corresponding element and element
   // reference point, in order to use element transformations. This requires
   // data set up in NURBSPatchRule::SetPointToElement.
   SetupPatchPA(patch, mesh, true);

   const Array<int>& Q1D = pQ1D[patch];
   const Array<int>& D1D = pD1D[patch];
   const std::vector<Array2D<real_t>>& B = pB[patch];
   const std::vector<Array2D<real_t>>& G = pG[patch];

   const IntArrayVar2D& minD = pminD[patch];
   const IntArrayVar2D& maxD = pmaxD[patch];
   const IntArrayVar2D& minQ = pminQ[patch];
   const IntArrayVar2D& maxQ = pmaxQ[patch];

   const IntArrayVar2D& minDD = pminDD[patch];
   const IntArrayVar2D& maxDD = pmaxDD[patch];

   int ndof = D1D[0];
   for (int d=1; d<dim; ++d)
   {
      ndof *= D1D[d];
   }

   auto rw = Reshape(reducedWeights.data(), numTypes, dim, numPatches);
   auto rid = Reshape(reducedIDs.data(), numTypes, dim, numPatches);

   const auto qd = Reshape(pa_data.Read(), Q1D[0]*Q1D[1]*Q1D[2],
                           (symmetric ? 6 : 9));

   // NOTE: the following is adapted from PADiffusionApply3D.
   std::vector<Array3D<real_t>> grad(dim);
   for (int d=0; d<dim; ++d)
   {
      grad[d].SetSize(Q1D[0], Q1D[1], Q1D[2]);
   }

   // Note that this large 3D array, most of which is unused,
   // seems inefficient, but testing showed that it is faster
   // than using a set<int> to store 1D indices of the used points.
   Array3D<bool> gradUsed;
   gradUsed.SetSize(Q1D[0], Q1D[1], Q1D[2]);

   Array3D<real_t> gradDXY(D1D[0], D1D[1], dim);
   Array2D<real_t> gradDX(D1D[0], dim);

   int nd[3];
   Array3D<int> cdofs;

   int *smati = nullptr;
   int *smatj = nullptr;
   real_t *smata = nullptr;
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
   smati = Memory<int>(ndof+1);
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

   smatj = Memory<int>(nnz);
   smata = Memory<real_t>(nnz);

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
         const int nwz = static_cast<int>(rid(zquad,2,patch)[jdz].size());
         for (int irz=0; irz < nwz; ++irz)
         {
            const int qz = rid(zquad,2,patch)[jdz][irz] + minD[2][jdz];
            const real_t zw = rw(zquad,2,patch)[jdz][irz];

            const real_t gwz  = B[2](qz,jdz);
            const real_t gwDz = G[2](qz,jdz);

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
               const int nwy = static_cast<int>(rid(yquad,1,patch)[jdy].size());
               for (int iry=0; iry < nwy; ++iry)
               {
                  const int qy = rid(yquad,1,patch)[jdy][iry] + minD[1][jdy];
                  const real_t yw = rw(yquad,1,patch)[jdy][iry];

                  const real_t gwy  = B[1](qy,jdy);
                  const real_t gwDy = G[1](qy,jdy);

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
                     const int nwx = static_cast<int>(rid(xquad,0,patch)[jdx].size());
                     for (int irx=0; irx < nwx; ++irx)
                     {
                        const int qx = rid(xquad,0,patch)[jdx][irx] + minD[0][jdx];

                        if (!gradUsed(qx,qy,qz))
                        {
                           const real_t gwx  = B[0](qx,jdx);
                           const real_t gwDx = G[0](qx,jdx);

                           const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
                           const real_t O11 = qd(q,0);
                           const real_t O12 = qd(q,1);
                           const real_t O13 = qd(q,2);
                           const real_t O21 = symmetric ? O12 : qd(q,3);
                           const real_t O22 = symmetric ? qd(q,3) : qd(q,4);
                           const real_t O23 = symmetric ? qd(q,4) : qd(q,5);
                           const real_t O31 = symmetric ? O13 : qd(q,6);
                           const real_t O32 = symmetric ? O23 : qd(q,7);
                           const real_t O33 = symmetric ? qd(q,5) : qd(q,8);

                           const real_t gradX = gwDx * gwy * gwz;
                           const real_t gradY = gwx * gwDy * gwz;
                           const real_t gradZ = gwx * gwy * gwDz;

                           grad[0](qx,qy,qz) = (O11*gradX)+(O12*gradY)+(O13*gradZ);
                           grad[1](qx,qy,qz) = (O21*gradX)+(O22*gradY)+(O23*gradZ);
                           grad[2](qx,qy,qz) = (O31*gradX)+(O32*gradY)+(O33*gradZ);

                           gradUsed(qx,qy,qz) = true;
                        }
                     }
                  }

                  // 00 terms
                  const int nw = static_cast<int>(rid(0,0,patch)[jdx].size());
                  for (int irx=0; irx < nw; ++irx)
                  {
                     const int qx = rid(0,0,patch)[jdx][irx] + minD[0][jdx];

                     const real_t gY = grad[1](qx,qy,qz);
                     const real_t gZ = grad[2](qx,qy,qz);
                     const real_t xw = rw(0,0,patch)[jdx][irx];
                     for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
                     {
                        const real_t wx  = B[0](qx,dx);
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
                  const int nw11 = static_cast<int>(rid(1,0,patch)[jdx].size());

                  for (int irx=0; irx < nw11; ++irx)
                  {
                     const int qx = rid(1,0,patch)[jdx][irx] + minD[0][jdx];

                     const real_t gX = grad[0](qx,qy,qz);
                     const real_t xw = rw(1,0,patch)[jdx][irx];
                     for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
                     {
                        const real_t wDx = G[0](qx,dx);
                        gradDX(dx,0) += gX * wDx * xw;
                     }
                  }

                  for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
                  {
                     const real_t wy  = B[1](qy,dy);
                     const real_t wDy = G[1](qy,dy);
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
               const real_t wz  = B[2](qz,dz);
               const real_t wDz = G[2](qz,dz);
               for (int dy = minDD[1][jdy]; dy <= maxDD[1][jdy]; ++dy)
               {
                  for (int dx = minDD[0][jdx]; dx <= maxDD[0][jdx]; ++dx)
                  {
                     real_t v = (zquad == 0) ? (gradDXY(dx,dy,0) * wz) +
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

void DiffusionIntegrator::AssemblePatchMatrix(const int patch,
                                              const FiniteElementSpace &fes,
                                              SparseMatrix*& smat)
{
   if (integrationMode == PATCHWISE_REDUCED)
   {
      AssemblePatchMatrix_reducedQuadrature(patch, fes, smat);
   }
   else
   {
      AssemblePatchMatrix_fullQuadrature(patch, fes, smat);
   }
}

}
