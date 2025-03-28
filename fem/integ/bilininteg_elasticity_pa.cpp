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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../../mesh/nurbs.hpp"
#include "bilininteg_elasticity_kernels.hpp"
#include "fem/integrator.hpp"
#include "linalg/dtensor.hpp"
#include "linalg/tensor.hpp"
using mfem::internal::tensor;
using mfem::internal::make_tensor;

namespace mfem
{

void ElasticityIntegrator::SetUpQuadratureSpaceAndCoefficients(
   const FiniteElementSpace &fes)
{
   if (IntRule == nullptr)
   {
      // This is where it's assumed that all elements are the same.
      const auto &T = *fes.GetMesh()->GetTypicalElementTransformation();
      int quad_order = 2 * T.OrderGrad(fes.GetTypicalFE());
      IntRule = &IntRules.Get(T.GetGeometryType(), quad_order);
   }

   Mesh &mesh = *fespace->GetMesh();

   q_space.reset(new QuadratureSpace(mesh, *IntRule));
   lambda_quad.reset(new CoefficientVector(lambda, *q_space,
                                           CoefficientStorage::FULL));
   mu_quad.reset(new CoefficientVector(mu, *q_space, CoefficientStorage::FULL));
   q_vec.reset(new QuadratureFunction(*q_space, vdim*vdim));
}

void ElasticityIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "Elasticity PA only implemented for byNODES ordering.");

   fespace = &fes;
   Mesh &mesh = *fespace->GetMesh();
   vdim = fespace->GetVDim();
   MFEM_VERIFY(vdim == mesh.Dimension(),
               "Vector dimension and geometric dimension must match.");
   ndofs = fespace->GetTypicalFE()->GetDof();

   SetUpQuadratureSpaceAndCoefficients(fes);

   auto ordering = GetEVectorOrdering(*fespace);
   auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
               DofToQuad::LEXICOGRAPHIC_FULL;
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
   geom = mesh.GetGeometricFactors(*IntRule, GeometricFactors::JACOBIANS);
}

void ElasticityIntegrator::AssembleDiagonalPA(Vector &diag)
{
   q_vec->SetVDim(vdim*vdim*vdim*vdim);
   internal::ElasticityAssembleDiagonalPA(vdim, ndofs, *lambda_quad, *mu_quad,
                                          *geom, *maps, *q_vec, diag);
}

void ElasticityIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   internal::ElasticityAddMultPA(vdim, ndofs, *fespace, *lambda_quad, *mu_quad,
                                 *geom, *maps, x, *q_vec, y);
}

void ElasticityIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   AddMultPA(x, y); // Operator is symmetric
}

void ElasticityComponentIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   fespace = &fes;

   // Avoid projecting the coefficients more than once. If the coefficients
   // change, the parent ElasticityIntegrator must be reassembled.
   if (!parent.q_space)
   {
      parent.SetUpQuadratureSpaceAndCoefficients(fes);
   }
   else
   {
      IntRule = parent.IntRule;
   }

   auto ordering = GetEVectorOrdering(*fespace);
   auto mode = ordering == ElementDofOrdering::NATIVE ? DofToQuad::FULL :
               DofToQuad::LEXICOGRAPHIC_FULL;
   geom = fes.GetMesh()->GetGeometricFactors(*IntRule,
                                             GeometricFactors::JACOBIANS);
   maps = &fespace->GetTypicalFE()->GetDofToQuad(*IntRule, mode);
}

void ElasticityComponentIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   internal::ElasticityComponentAddMultPA(
      parent.vdim, parent.ndofs, *fespace, *parent.lambda_quad, *parent.mu_quad,
      *geom, *maps, x, *parent.q_vec, y, i_block, j_block);
}

void ElasticityComponentIntegrator::AddMultTransposePA(const Vector &x,
                                                       Vector &y) const
{
   // Each block in the operator is symmetric, so we can just switch the roles
   // of i_block and j_block
   internal::ElasticityComponentAddMultPA(
      parent.vdim, parent.ndofs, *fespace, *parent.lambda_quad, *parent.mu_quad,
      *geom, *maps, x, *parent.q_vec, y, j_block, i_block);
}

void ElasticityIntegrator::AssembleNURBSPA(const FiniteElementSpace &fes)
{

   fespace = &fes;
   Mesh &mesh = *fespace->GetMesh();
   vdim = fespace->GetVDim();
   MFEM_VERIFY(vdim == mesh.Dimension(),
               "Vector dimension and geometric dimension must match.");
   // ndofs = fespace->GetTypicalFE()->GetDof();

   numPatches = mesh.NURBSext->GetNP();
   pa_data.resize(numPatches);
   for (int p=0; p<numPatches; ++p)
   {
      AssemblePatchPA(p, fes);
   }
}

void ElasticityIntegrator::AssemblePatchPA(const int patch,
                                           const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   SetupPatchBasisData(mesh, patch);

   SetupPatchPA(patch, mesh);  // For full quadrature, unitWeights = false
}

/**
 * 1) Compute grad_uhat (reference space) interpolated at quadrature points
 *
 * B_{nq} U_n = \sum_n u_n \tilde{\nabla} \tilde{\phi}_n (\tilde{x}_q)
 *
 * Variables:
 * - c  = component
 * - d  = derivative
 * - dx, dy, dz = degrees of freedom (DOF) indices
 * - qx, qy, qz = quadrature indices
 *
 * Tensor product shape functions:
 * phi[dx,dy,dz](qx,qy,qz) = phix[dx](qx) * phiy[dy](qy) * phiz[dz](qz)
 *
 * Gradient of shape functions:
 * grad_phi[dx,dy,dz](qx,qy,qz) = [
 *    dphix[dx](qx) * phiy[dy](qy)  * phiz[dz](qz),
 *    phix[dx](qx)  * dphiy[dy](qy) * phiz[dz](qz),
 *    phix[dx](qx)  * phiy[dy](qy)  * dphiz[dz](qz)
 * ]
 *
 * Computation of grad_uhat[c,0] (sum factorization):
 * grad_uhat[c,0](qx,qy,qz)
 *    = \sum_{dx,dy,dz} U[c][dx,dy,dz] * grad_phi[dx,dy,dz][0](qx,qy,qz)
 *    = \sum_{dx,dy,dz} U[c][dx,dy,dz] * dphix[dx](qx) * phiy[dy](qy) * phiz[dz](qz)
 *    = \sum_{dz} phiz[dz](qz)
 *       \sum_{dy} phiy[dy](qy)
 *          \sum_{dx} U[c][dx,dy,dz] * dphix[dx](qx)
 *
 * Because a nurbs patch is "sparse" compared to an element in terms of shape
 * function support - we can further optimize the computation by restricting
 * interpolation to only the qudrature points supported in each dimension.
 */
void PatchInterpolateGradient(const PatchBasisInfo &pb,
                              const Vector &Uv,
                              Vector &sumXYv,
                              Vector &sumXv,
                              DeviceTensor<5, real_t> &grad_uhat)
{
   // Unpack
   const int vdim = pb.vdim;
   const Array<int>& Q1D = pb.Q1D;
   const Array<int>& D1D = pb.D1D;
   const std::vector<Array2D<real_t>>& B = pb.B;
   const std::vector<Array2D<real_t>>& G = pb.G;
   // minD/maxD are maps from 1D dof index -> 1D quadrature index
   // For component c and dof index d, [minD[c][d], maxD[c][d]] are
   // the min/max quadrature indices supported by the shape function
   // B_{cd}. Because shape functions on patches don't necessarily
   // support the whole domain, these maps are used to eliminate
   // unnecessary interpolations.
   const std::vector<std::vector<int>> minD = pb.minD;
   const std::vector<std::vector<int>> maxD = pb.maxD;
   const std::vector<int> acc = pb.accsize;
   const int NQ = pb.NQ;

   // Shape as tensors
   const auto U = Reshape(Uv.HostRead(), D1D[0], D1D[1], D1D[2], vdim);
   auto sumXY = Reshape(sumXYv.HostReadWrite(), vdim, vdim, acc[0], acc[1]);
   auto sumX = Reshape(sumXv.HostReadWrite(), vdim, vdim, acc[0]);
   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      sumXYv = 0.0;
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         sumXv = 0.0;
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int c = 0; c < vdim; ++c)
            {
               const real_t u = U(dx,dy,dz,c);
               for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
               {
                  sumX(c,0,qx) += u * B[0](qx,dx);
                  sumX(c,1,qx) += u * G[0](qx,dx);
               }
            }
         } // dx
         for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
         {
            const real_t wy  = B[1](qy,dy);
            const real_t wDy = G[1](qy,dy);
            for (int c = 0; c < vdim; ++c)
            {
               // This full range of qx values is generally necessary.
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  const real_t wx  = sumX(c,0,qx);
                  const real_t wDx = sumX(c,1,qx);
                  sumXY(c,0,qx,qy) += wDx * wy;
                  sumXY(c,1,qx,qy) += wx  * wDy;
                  sumXY(c,2,qx,qy) += wx  * wy;
               } // qx
            } // c
         } // qy
      } // dy

      for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
      {
         const real_t wz  = B[2](qz,dz);
         const real_t wDz = G[2](qz,dz);
         for (int c = 0; c < vdim; ++c)
         {
            for (int qy = 0; qy < Q1D[1]; ++qy)
            {
               for (int qx = 0; qx < Q1D[0]; ++qx)
               {
                  grad_uhat(c,0,qx,qy,qz) += sumXY(c,0,qx,qy) * wz;
                  grad_uhat(c,1,qx,qy,qz) += sumXY(c,1,qx,qy) * wz;
                  grad_uhat(c,2,qx,qy,qz) += sumXY(c,2,qx,qy) * wDz;
               }
            } // qy
         } // c
      } // qz
   } // dz
}


/**
 * Transform grad_uhat (reference) to physical space:
 *    grad_u = grad_uhat * J^{-1}
 *
 * Compute stress:
 *    sigma = lambda*tr(grad_u)*I + mu*(grad_u + grad_u^T)
 *
 * lambda and mu have W*detJ factored in
 *
 * Transform back to reference space:
 *    return sigma * J^{-T}
 */
template <int dim>
tensor<mfem::real_t, dim, dim>
LinearElasticityKernel(const tensor<mfem::real_t, dim, dim> Jinvt,
                       const real_t lambda,
                       const real_t mu,
                       const tensor<mfem::real_t, dim, dim> grad_uhat)
{
   // Convert grad_uhat to physical space
   const auto grad_u = grad_uhat * transpose(Jinvt);
   // Compute stress
   constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
   const tensor<mfem::real_t, dim, dim> strain = sym(grad_u);
   const auto stress = lambda * tr(strain) * I + 2.0 * mu * strain;
   // Transform back to reference space
   return stress * Jinvt;
}

/**
 * Transforms grad_u into physical space, computes stress, then transforms back to reference space
 */
void PatchApplyKernel3D(const PatchBasisInfo &pb,
                        const Vector &pa_data,
                        DeviceTensor<5, real_t> &grad,
                        DeviceTensor<5, real_t> &S)
{
   static constexpr int dim = 3;
   // Unpack patch basis info
   const int vdim = pb.vdim;
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;
   // Quadrature data. 11 values per quadrature point.
   // First 9 entries are J^{-T}; then lambda*W*detJ and mu*W*detJ
   const auto qd = Reshape(pa_data.HostRead(), NQ, 11);

   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
            const real_t lambda  = qd(q,9);
            const real_t mu      = qd(q,10);
            const auto Jinvt = make_tensor<dim, dim>(
            [&](int i, int j) { return qd(q, i*dim + j); });
            const auto grad_uhat = make_tensor<dim, dim>(
            [&](int i, int j) { return grad(i,j,qx,qy,qz); });
            const auto Sq = LinearElasticityKernel(Jinvt, lambda, mu, grad_uhat);

            for (int i = 0; i < dim; ++i)
            {
               for (int j = 0; j < dim; ++j)
               {
                  S(i,j,qx,qy,qz) = Sq(i,j);
               }
            }
         } // qx
      } // qy
   } // qz
}

/**
 * 3) Contraction with grad_v (quads -> dofs)
 *
 * S[ij] = [
 *    s00, s01, s02,
 *    s10, s11, s12,
 *    s20, s21, s22,
 * ]
 * grad_v[ij] = e[i] * grad_phi[j]
 *             = e[i] * [ dX*Y*Z, X*dY*Z, X*Y*dZ ]
 *
 * Y[i] = S[ij] * grad_phi[j] = [
 *    s00*dX*Y*Z + s01*X*dY*Z + s02*X*Y*dZ,
 *    s10*dX*Y*Z + s11*X*dY*Z + s12*X*Y*dZ,
 *    s20*dX*Y*Z + s21*X*dY*Z + s22*X*Y*dZ,
 * ]
 *
 * sX = [
 *    s00*dX, s01*X, s02*X,
 *    s10*dX, s11*X, s12*X,
 *    s20*dX, s21*X, s22*X,
 * ]
 *
 * sXY = [
 *    (s00*dX) * Y + (s01*X) * dY, (s02*X) * Y,
 *    (s10*dX) * Y + (s11*X) * dY, (s12*X) * Y,
 *    (s20*dX) * Y + (s21*X) * dY, (s22*X) * Y,
 * ]
 *
 * Y[i] = [
 *    ((s00*dX) * Y + (s01*X) * dY) * Z + ((s02*X) * Y) * dZ,
 *    ((s10*dX) * Y + (s11*X) * dY) * Z + ((s12*X) * Y) * dZ,
 *    ((s20*dX) * Y + (s21*X) * dY) * Z + ((s22*X) * Y) * dZ,
 * ]
 */
void PatchApplyTestFunction(const PatchBasisInfo &pb,
                            DeviceTensor<5, real_t> &S,
                            Vector &sumXYv,
                            Vector &sumXv,
                            Vector &y)
{
   // Unpack patch basis info
   const int vdim = pb.vdim;
   const Array<int>& Q1D = pb.Q1D;
   const Array<int>& D1D = pb.D1D;
   const std::vector<Array2D<real_t>>& B = pb.B;
   const std::vector<Array2D<real_t>>& G = pb.G;
   const std::vector<std::vector<int>> minQ = pb.minQ;
   const std::vector<std::vector<int>> maxQ = pb.maxQ;
   const std::vector<int> acc = pb.accsize;

   // Shape as tensors
   auto sumXY = Reshape(sumXYv.HostReadWrite(), vdim, vdim, acc[0], acc[1]);
   auto sumX = Reshape(sumXv.HostReadWrite(), vdim, vdim, acc[0]);

   auto Y = Reshape(y.HostReadWrite(), D1D[0], D1D[1], D1D[2], vdim);

   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      sumXYv = 0.0;
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         sumXv = 0.0;
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const real_t s[3][3] =
            {
               { S(0,0,qx,qy,qz), S(0,1,qx,qy,qz), S(0,2,qx,qy,qz) },
               { S(1,0,qx,qy,qz), S(1,1,qx,qy,qz), S(1,2,qx,qy,qz) },
               { S(2,0,qx,qy,qz), S(2,1,qx,qy,qz), S(2,2,qx,qy,qz) }
            };
            for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
            {
               const real_t wx  = B[0](qx,dx);
               const real_t wDx = G[0](qx,dx);

               /*
               sX = [
                  s00*dX, s01*X, s02*X,
                  s10*dX, s11*X, s12*X,
                  s20*dX, s21*X, s22*X,
               ]
               */
               for (int c = 0; c < vdim; ++c)
               {
                  sumX(c,0,dx) += s[c][0] * wDx;
                  sumX(c,1,dx) += s[c][1] * wx;
                  sumX(c,2,dx) += s[c][2] * wx;
               }
            }
         }
         for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
         {
            /*
            sXY = [
               (s00*dX) * Y + (s01*X) * dY, (s02*X) * Y,
               (s10*dX) * Y + (s11*X) * dY, (s12*X) * Y,
               (s20*dX) * Y + (s21*X) * dY, (s22*X) * Y,
            ]
            */
            const real_t wy  = B[1](qy,dy);
            const real_t wDy = G[1](qy,dy);
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int c = 0; c < vdim; ++c)
               {
                  sumXY(c,0,dx,dy) += sumX(c,0,dx) * wy + sumX(c,1,dx) * wDy;
                  sumXY(c,1,dx,dy) += sumX(c,2,dx) * wy;
               }
            }
         }
      }
      for (int dz = minQ[2][qz]; dz <= maxQ[2][qz]; ++dz)
      {
         const real_t wz  = B[2](qz,dz);
         const real_t wDz = G[2](qz,dz);
         for (int dy = 0; dy < D1D[1]; ++dy)
         {
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               for (int c = 0; c < vdim; ++c)
               {
                  Y(dx,dy,dz,c) +=
                     (sumXY(c,0,dx,dy) * wz +
                      sumXY(c,1,dx,dy) * wDz);
               }
            }
         }
      } // dz
   } // qz

}

// This version uses full 1D quadrature rules, taking into account the
// minimum interaction between basis functions and integration points.
void ElasticityIntegrator::AddMultPatchPA3D(const Vector &pa_data,
                                            const PatchBasisInfo &pb,
                                            const Vector &x,
                                            Vector &y) const
{
   // Unpack patch basis info
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;

   // grad(i,j,q): derivative of u_i w.r.t. j evaluated at q=(qx,qy,qz)
   Vector grad_uhatv(vdim*vdim*NQ);
   grad_uhatv = 0.0;
   auto grad_uhat = Reshape(grad_uhatv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1],
                            Q1D[2]);

   // S[i,j,q] = D( grad_u )
   //          = stress[i,j,q] * J^{-T}[q]
   Vector Sv(vdim*vdim*NQ);
   Sv = 0.0;
   auto S = Reshape(Sv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   // Accumulators; these are shared between grad_u interpolation and grad_v_T
   // application, so their size is the max of qpts/dofs
   Vector sumXYv(vdim*vdim*pb.accsize[0]*pb.accsize[1]);
   Vector sumXv(vdim*vdim*pb.accsize[0]);

   // 1) Interpolate U at dofs to grad_u in reference quadrature space
   PatchInterpolateGradient(pb, x, sumXYv, sumXv, grad_uhat);

   // 2) Apply the "D" operator at each quadrature point: D( grad_uhat )
   PatchApplyKernel3D(pb, pa_data, grad_uhat, S);

   // 3) Apply test function grad_v
   PatchApplyTestFunction(pb, S, sumXYv, sumXv, y);

}

void AddMultPatchPA3D_SOL(const Vector &pa_data,
                          const PatchBasisInfo &pb,
                          const Vector &x,
                          Vector &y)
{
   constexpr int vdim = 3;
   // Unpack patch basis info
   const std::vector<Array2D<real_t>>& B = pb.B;
   const std::vector<Array2D<real_t>>& G = pb.G;
   const Array<int>& Q1D = pb.Q1D;
   const Array<int>& D1D = pb.D1D;

   const int NQ = pb.NQ;
   const std::vector<std::vector<int>> minD = pb.minD;
   const std::vector<std::vector<int>> maxD = pb.maxD;
   const std::vector<std::vector<int>> minQ = pb.minQ;
   const std::vector<std::vector<int>> maxQ = pb.maxQ;

   // Accumulators; these are shared between grad_u interpolation and grad_v_T
   // application, so their size is the max of qpts/dofs
   // Vector sumXYv(vdim*vdim*pb.accsize[0]*pb.accsize[1]);
   // Vector sumXv(vdim*vdim*pb.accsize[0]);

   // 1) Read dofs
   const auto U = Reshape(x.HostRead(), D1D[0], D1D[1], D1D[2], vdim);
   real_t sum = 0;

   // #pragma omp parallel for
   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int c = 0; c < vdim; ++c)
            {
               const real_t u = U(dx,dy,dz,c);
               sum += u;
               for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
               {
                  sum += B[0](qx,dx);
                  sum += G[0](qx,dx);
               }
            }
         } // dx
         for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
         {
            sum += B[1](qy,dy);
            sum += G[1](qy,dy);
         } // qy
      } // dy
      for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
      {
         sum += B[2](qz,dz);
         sum += G[2](qz,dz);
      }
   }

   // 2) Read quadrature data
   const auto qd = Reshape(pa_data.HostRead(), NQ, 11);
   Vector Sv(vdim*vdim*NQ);
   Sv = 0.0;
   auto S = Reshape(Sv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);
   // #pragma omp parallel for
   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
            for (int i = 0; i < 11; ++i)
            {
               sum += qd(q,i);
            }

         }
      }
   }

   // 3) Write dofs
   auto Y = Reshape(y.HostReadWrite(), D1D[0], D1D[1], D1D[2], vdim);
   // #pragma omp parallel for
   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int c = 0; c < vdim; ++c)
            {
               Y(dx,dy,dz,c) = sum;
            }
         }
      }
   }

}


void ElasticityIntegrator::AddMultPatchPA(const int patch, const Vector &x,
                                          Vector &y) const
{
   if (vdim == 3)
   {
      AddMultPatchPA3D(pa_data[patch], pbinfo[patch], x, y);
      // AddMultPatchPA3D_SOL(pa_data[patch], pbinfo[patch], x, y);
   }
   else
   {
      MFEM_ABORT("Only 3D is supported.");
   }
}


void ElasticityIntegrator::AddMultNURBSPA(const Vector &x, Vector &y) const
{
   Vector xp, yp;

   // #pragma omp parallel for
   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);

      x.GetSubVector(vdofs, xp);
      yp.SetSize(vdofs.Size());
      yp = 0.0;

      AddMultPatchPA(p, xp, yp);

      y.AddElementVector(vdofs, yp);
   }
}

void ElasticityIntegrator::AssembleDiagonalPatchPA(const Vector &pa_data,
                                                   const PatchBasisInfo &pb,
                                                   Vector &diag) const
{
   // Unpack patch basis info
   const Array<int>& D1D = pb.D1D;
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;
   const std::vector<Array2D<real_t>>& B = pb.B;
   const std::vector<Array2D<real_t>>& G = pb.G;
   const std::vector<std::vector<int>> minD = pb.minD;
   const std::vector<std::vector<int>> maxD = pb.maxD;

   const auto qd = Reshape(pa_data.HostRead(), NQ, 11);
   static constexpr int dim = 3;

   // grad(i,j,q): derivative of u_i w.r.t. j evaluated at q=(qx,qy,qz)
   Vector grad_uhatv(vdim*vdim*NQ);
   grad_uhatv = 0.0;
   auto grad_uhat = Reshape(grad_uhatv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1],
                            Q1D[2]);

   auto Y = Reshape(diag.HostReadWrite(), D1D[0], D1D[1], D1D[2], vdim);

   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
            {
               const real_t Bz = B[2](qz,dz);
               const real_t Gz = G[2](qz,dz);
               for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
               {
                  const real_t By = B[1](qy,dy);
                  const real_t Gy = G[1](qy,dy);
                  for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
                  {
                     const real_t Bx = B[0](qx,dx);
                     const real_t Gx = G[0](qx,dx);
                     const real_t grad[3] =
                     {
                        Gx * By * Bz,
                        Bx * Gy * Bz,
                        Bx * By * Gz
                     };

                     const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
                     const real_t lambda  = qd(q,9);
                     const real_t mu      = qd(q,10);
                     const auto Jinvt = make_tensor<dim, dim>(
                     [&](int i, int j) { return qd(q, i*dim + j); });
                     // As second order tensor: e[i] * grad_phi[j]
                     const auto grad_phi = make_tensor<dim, dim>(
                     [&](int i, int j) { return grad[j]; });

                     const auto Sq = LinearElasticityKernel(Jinvt, lambda, mu, grad_phi);

                     const auto grad_phiv = make_tensor<dim>(
                     [&](int i) { return grad[i]; });
                     const auto Yq = dot(Sq, grad_phiv);
                     for (int c = 0; c < vdim; ++c)
                     {
                        Y(dx,dy,dz,c) += Yq[c];
                     }
                  } // qx
               } // qy
            } // qz
         } // dx
      } // dy
   } // dz
}


void ElasticityIntegrator::AssembleDiagonalNURBSPA(Vector &diag)
{
   Vector diagp;

   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);
      diagp.SetSize(vdofs.Size());
      diagp = 0.0;

      AssembleDiagonalPatchPA(pa_data[p], pbinfo[p], diagp);

      diag.AddElementVector(vdofs, diagp);
   }
}

} // namespace mfem
