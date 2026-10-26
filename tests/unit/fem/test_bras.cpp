// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

namespace bras_test
{

// The tests compare BRASPreconditioner against a dense reference built from
// the definition B_W = A_W^{-1} P^T W M_br^{-1} W P A_W^{-1}, check the
// spectral bounds of the theory, and compare parallel against serial runs.

using Weighting = BRASPreconditioner::Weighting;
using NCMethod = BRASPreconditioner::NCMethod;

enum class MeshType { QUAD, TRI, HEX, TET };

// The spaces cover low and high order, nodal (Gauss-Lobatto) and positive
// (Bernstein) bases, vdim > 1, discontinuous spaces, and the signed DOFs of
// ND and RT spaces.
enum class Space { H1_1, H1_3, H1_POS_3, H1_VDIM, L2_2, ND_1, ND_2, RT_0, RT_1 };

// Meshes of [0,1]^d. Quadrilateral and hexahedral meshes are perturbed to
// have non-affine elements; if curved, the mesh has quadratic nodes. The
// perturbation vanishes on the boundary, so the domain is unchanged.
Mesh MakeMesh(MeshType type, int n, bool curved = false)
{
   Mesh mesh;
   switch (type)
   {
      case MeshType::QUAD:
         mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true);
         break;
      case MeshType::TRI:
         mesh = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE, true);
         break;
      case MeshType::HEX:
         mesh = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);
         break;
      case MeshType::TET:
         mesh = Mesh::MakeCartesian3D(n, n, n, Element::TETRAHEDRON);
         break;
   }
   if (curved) { mesh.SetCurvature(2); }
   const bool perturb = curved || type == MeshType::QUAD ||
                        type == MeshType::HEX;
   if (perturb)
   {
      mesh.Transform([](const Vector &x, Vector &y)
      {
         y = x;
         real_t s = 1.0;
         for (int d = 0; d < x.Size(); d++) { s *= sin(M_PI*x(d)); }
         y(0) += 0.1*s;
      });
   }
   return mesh;
}

// Partition the mesh into slabs in the x direction, one per rank. This avoids
// a dependency on METIS, and the explicit partitioning is also needed to map
// serial grid functions to parallel ones.
Array<int> SlabPartitioning(Mesh &mesh, int nparts)
{
   Array<int> part(mesh.GetNE());
   Vector c;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementCenter(e, c);
      part[e] = std::min(nparts - 1, int(std::floor(c(0)*nparts)));
   }
   return part;
}

// The collection, vdim and ordering defining a Space on a mesh.
struct SpaceInfo
{
   std::unique_ptr<FiniteElementCollection> fec;
   int vdim = 1;
   int ordering = Ordering::byNODES;
};

SpaceInfo MakeSpace(Space s, int dim)
{
   SpaceInfo info;
   switch (s)
   {
      case Space::H1_1: info.fec.reset(new H1_FECollection(1, dim)); break;
      case Space::H1_3: info.fec.reset(new H1_FECollection(3, dim)); break;
      case Space::H1_POS_3:
         info.fec.reset(new H1_FECollection(3, dim, BasisType::Positive));
         break;
      case Space::H1_VDIM:
         info.fec.reset(new H1_FECollection(2, dim));
         info.vdim = 2;
         info.ordering = Ordering::byVDIM;
         break;
      case Space::L2_2: info.fec.reset(new L2_FECollection(2, dim)); break;
      case Space::ND_1: info.fec.reset(new ND_FECollection(1, dim)); break;
      case Space::ND_2: info.fec.reset(new ND_FECollection(2, dim)); break;
      case Space::RT_0: info.fec.reset(new RT_FECollection(0, dim)); break;
      case Space::RT_1: info.fec.reset(new RT_FECollection(1, dim)); break;
   }
   return info;
}

// Positive, non-constant scalar coefficient.
real_t ScalarCoeff(const Vector &x)
{
   return 1.0 + x(0)*x(0) + 0.5*x(1);
}

// Symmetric positive definite matrix coefficient of any size (the diagonal
// dominates the constant off-diagonal entries on [0,1]^d for sizes up to 3).
void MatrixCoeff(const Vector &x, DenseMatrix &K)
{
   const int n = K.Height();
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
      {
         K(i,j) = (i == j) ? 1.0 + (i + 1)*x(0)*x(0) : 0.25*x(1);
      }
   }
}

bool IsVectorSpace(Space s)
{
   return s == Space::ND_1 || s == Space::ND_2 || s == Space::RT_0 ||
          s == Space::RT_1;
}

// The spaces tested with a matrix coefficient; the others use the scalar one.
bool UsesMatrixCoeff(Space s)
{
   return IsVectorSpace(s) || s == Space::H1_VDIM;
}

// Add to a the same mass integrator that the BRASPreconditioner constructors
// taking a space create.
void AddMassIntegrator(BilinearForm &a, Space s, Coefficient &Q,
                       MatrixCoefficient &MQ)
{
   if (IsVectorSpace(s)) { a.AddDomainIntegrator(new VectorFEMassIntegrator(MQ)); }
   else if (s == Space::H1_VDIM)
   {
      a.AddDomainIntegrator(new VectorMassIntegrator(MQ));
   }
   else { a.AddDomainIntegrator(new MassIntegrator(Q)); }
}

// The dense matrix of the square operator op, computed column by column.
DenseMatrix ToDense(const Operator &op)
{
   const int n = op.Height();
   DenseMatrix A(n);
   Vector e(n), col(n);
   for (int j = 0; j < n; j++)
   {
      e = 0.0;
      e(j) = 1.0;
      op.Mult(e, col);
      A.SetCol(j, col);
   }
   return A;
}

// Returns true if the symmetric matrix A is positive definite, by attempting
// a Cholesky factorization (in the lower triangle of the copy of A).
bool IsPositiveDefinite(DenseMatrix A)
{
   const int n = A.Height();
   for (int j = 0; j < n; j++)
   {
      real_t d = A(j,j);
      for (int k = 0; k < j; k++) { d -= A(j,k)*A(j,k); }
      if (d <= 0.0) { return false; }
      d = std::sqrt(d);
      A(j,j) = d;
      for (int i = j + 1; i < n; i++)
      {
         real_t s = A(i,j);
         for (int k = 0; k < j; k++) { s -= A(i,k)*A(j,k); }
         A(i,j) = s/d;
      }
   }
   return true;
}

// max_ij |A_ij - B_ij|.
real_t MaxDiff(const DenseMatrix &A, const DenseMatrix &B)
{
   DenseMatrix D(A);
   D -= B;
   return D.MaxMaxNorm();
}

// Extreme eigenvalues of W^{-1} M for SPD M and positive diagonal W = diag(w),
// by bisection on the definiteness of M - t W. These are the sharpest
// constants in alpha W <= M <= beta W. Only Cholesky factorizations are used,
// so the test does not depend on LAPACK.
void ExtremeEigenvalues(const DenseMatrix &M, const Vector &w,
                        real_t &lmin, real_t &lmax)
{
   // Initial brackets: the Rayleigh quotients of the unit vectors are
   // M_ii/w_i, so lmin <= min_i M_ii/w_i and lmax >= max_i M_ii/w_i. The
   // eigenvalues are positive and sum to the trace of W^{-1} M, so the trace
   // bounds lmax from above.
   const int n = M.Height();
   real_t rmin = infinity(), rmax = 0.0, trace = 0.0;
   for (int i = 0; i < n; i++)
   {
      const real_t r = M(i,i)/w(i);
      rmin = std::min(rmin, r);
      rmax = std::max(rmax, r);
      trace += r;
   }
   // Is M - t W positive definite (sign = 1), or t W - M (sign = -1)?
   auto pd = [&](real_t t, real_t sign)
   {
      DenseMatrix A(M);
      for (int i = 0; i < n; i++) { A(i,i) -= t*w(i); }
      A *= sign;
      return IsPositiveDefinite(A);
   };
   // lmin is the largest t with M - t W positive definite.
   real_t lo = 0.0, hi = rmin;
   for (int it = 0; it < 60; it++)
   {
      const real_t t = 0.5*(lo + hi);
      if (pd(t, 1.0)) { lo = t; } else { hi = t; }
   }
   lmin = lo;
   // lmax is the smallest t with t W - M positive definite.
   lo = rmax, hi = trace;
   for (int it = 0; it < 60; it++)
   {
      const real_t t = 0.5*(lo + hi);
      if (pd(t, -1.0)) { hi = t; } else { lo = t; }
   }
   lmax = hi;
}

// R^T A R.
DenseMatrix RtAR(const DenseMatrix &A, const DenseMatrix &R)
{
   DenseMatrix AR(A.Height(), R.Width()), RtAR_(R.Width());
   Mult(A, R, AR);
   MultAtB(R, AR, RtAR_);
   return RtAR_;
}

// The dense matrices of the class documentation for a serial space, with
// P = P_0 E, where P_0 maps the L-DOFs to the element DOFs and E is the
// conforming prolongation (the identity on conforming meshes).
//
// Unlike the implementation, the products with P_0 use the signs of the
// element vdofs (GetElementVDofs() encodes a negative orientation as -1-dof)
// rather than the element restriction, and the products with E are computed
// with dense matrices, so the test does not rely on the structure of P. The
// assembled matrix of the form must be available.
struct Reference
{
   Array<bool> free;    ///< The free true DOFs.
   DenseMatrix E;       ///< E, with the essential columns set to zero.
   Array<int> row_nnz;  ///< The number of nonzeros in each row of E.
   Vector dl;           ///< diag(D_W) = diag(P_0^T W P_0).
   DenseMatrix Gm;      ///< P_0^T W M_br^{-1} W P_0.
   DenseMatrix M;       ///< E^T P_0^T M_br P_0 E, zero on the essential DOFs.
   real_t alpha, beta;  ///< The constants in alpha W <= M_br <= beta W.
};

Reference BuildReference(BilinearForm &a, Weighting weighting,
                         const Array<int> &ess)
{
   const FiniteElementSpace &fes = *a.FESpace();
   const int nl = fes.GetVSize();
   const int nt = fes.GetTrueVSize();
   const int ne = fes.GetNE();
   const int nd = fes.GetFE(0)->GetDof()*fes.GetVDim();
   Reference r;

   // E and its row structure, then zero the essential columns.
   r.E.SetSize(nl, nt);
   if (const SparseMatrix *cP = fes.GetConformingProlongation())
   {
      cP->ToDenseMatrix(r.E);
   }
   else
   {
      REQUIRE(nl == nt);
      r.E = 0.0;
      for (int i = 0; i < nl; i++) { r.E(i,i) = 1.0; }
   }
   r.row_nnz.SetSize(nl);
   r.row_nnz = 0;
   for (int i = 0; i < nl; i++)
   {
      for (int j = 0; j < nt; j++) { if (r.E(i,j) != 0.0) { r.row_nnz[i]++; } }
   }
   r.free.SetSize(nt);
   r.free = true;
   for (int j : ess)
   {
      r.free[j] = false;
      for (int i = 0; i < nl; i++) { r.E(i,j) = 0.0; }
   }

   // Each element contributes to the products with P_0 through its signed
   // vdofs: P_0 has the entry sgn(i) in row (e, i) and column dof(i), so
   // (P_0^T X P_0)(dof(i), dof(k)) gains sgn(i) X_e(i,k) sgn(k) for each block
   // X_e of a block diagonal X. This gives P_0^T M_br P_0, Gm and
   // diag(D_W) = P_0^T diag(w) P_0 without forming P_0.
   DenseMatrix P0tMP0(nl), elmat, inv;
   P0tMP0 = 0.0;
   r.Gm.SetSize(nl);
   r.Gm = 0.0;
   r.dl.SetSize(nl);
   r.dl = 0.0;
   r.alpha = infinity();
   r.beta = 0.0;
   Array<int> vdofs, dof(nd);
   Vector sgn(nd), w_e(nd);
   DofTransformation doftrans;
   for (int e = 0; e < ne; e++)
   {
      // The element matrix in the basis of the global DOFs, as summed into
      // the assembled matrix by BilinearForm::Assemble(), and W = diag(w_e).
      fes.GetElementVDofs(e, vdofs, doftrans);
      a.ComputeElementMatrix(e, elmat);
      doftrans.TransformDual(elmat);
      for (int i = 0; i < nd; i++)
      {
         dof[i] = vdofs[i] >= 0 ? vdofs[i] : -1 - vdofs[i];
         sgn(i) = vdofs[i] >= 0 ? 1.0 : -1.0;
         w_e(i) = weighting == Weighting::DIAGONAL ? elmat(i,i) : 1.0;
      }
      real_t lmin, lmax;
      ExtremeEigenvalues(elmat, w_e, lmin, lmax);
      r.alpha = std::min(r.alpha, lmin);
      r.beta = std::max(r.beta, lmax);
      DenseMatrixInverse(elmat).GetInverseMatrix(inv);
      for (int i = 0; i < nd; i++)
      {
         r.dl(dof[i]) += w_e(i);
         for (int k = 0; k < nd; k++)
         {
            const real_t s = sgn(i)*sgn(k);
            P0tMP0(dof[i], dof[k]) += s*elmat(i,k);
            r.Gm(dof[i], dof[k]) += s*w_e(i)*inv(i,k)*w_e(k);
         }
      }
   }

   // Check that P_0^T M_br P_0 is the assembled matrix on the L-DOFs.
   DenseMatrix ML;
   a.SpMat().ToDenseMatrix(ML);
   REQUIRE(MaxDiff(ML, P0tMP0) <= 1e-12*ML.MaxMaxNorm());
   r.M = RtAR(ML, r.E);
   return r;
}

// Reference weighted BRAS B_W = A^{-1} P^T W M_br^{-1} W P A^{-1}, with
// A = P^T W P, built from its definition with dense matrices. With the
// essential columns of P removed, this is weighted BRAS for the free DOFs. On
// the essential DOFs, B_W is the identity.
DenseMatrix ReferenceBW(const Reference &r)
{
   const int nt = r.E.Width();
   DenseMatrix Dl(r.dl.Size());
   Dl.Diag(r.dl.GetData(), r.dl.Size());
   // A = E^T D_W E and G = E^T Gm E. The essential rows and columns of A are
   // zero, so set A_ii = 1 there to make it invertible; this does not affect
   // the free block of B, and the essential rows and columns of G (and
   // therefore of B) are zero.
   DenseMatrix A = RtAR(Dl, r.E), G = RtAR(r.Gm, r.E), Ainv, AinvG(nt),
               B(nt);
   for (int j = 0; j < nt; j++) { if (!r.free[j]) { A(j,j) = 1.0; } }
   DenseMatrixInverse(A).GetInverseMatrix(Ainv);
   Mult(Ainv, G, AinvG);
   Mult(AinvG, Ainv, B);
   for (int j = 0; j < nt; j++) { if (!r.free[j]) { B(j,j) = 1.0; } }
   return B;
}

// Reference for the nonconforming methods, from the formulas in the
// documentation of BRASPreconditioner::NCMethod, with dense matrices. The J
// rows of E are the rows with at most one nonzero, and the H rows form C.
// On the free DOFs, S = diag(A)^{-1} and Lambda_u = E_J^T D_W E_J; both are
// set to zero on the essential DOFs. If E has no H rows, A is diagonal and
// all methods reduce to B_W.
DenseMatrix ReferenceNC(const Reference &r, NCMethod method, int degree)
{
   const int nl = r.E.Height(), nt = r.E.Width();
   if (r.row_nnz.Max() <= 1) { return ReferenceBW(r); }

   // s = diag(S), lus = diag(Lambda_u^{-1}).
   Vector s(nt), lus(nt);
   for (int j = 0; j < nt; j++)
   {
      real_t d = 0.0, lu = 0.0;
      for (int i = 0; i < nl; i++)
      {
         const real_t e2 = r.E(i,j)*r.E(i,j)*r.dl(i);
         d += e2;
         if (r.row_nnz[i] <= 1) { lu += e2; }
      }
      s(j) = r.free[j] ? 1.0/d : 0.0;
      lus(j) = r.free[j] ? 1.0/lu : 0.0;
   }
   const DenseMatrix G = RtAR(r.Gm, r.E);
   DenseMatrix B(nt);

   if (method == NCMethod::DIAGONAL)
   {
      // S G S.
      for (int i = 0; i < nt; i++)
      {
         for (int j = 0; j < nt; j++) { B(i,j) = s(i)*G(i,j)*s(j); }
      }
   }
   else if (method == NCMethod::LEFT_INVERSE)
   {
      // X C = S C^T Lambda_h C with X = S C^T Lambda_h.
      DenseMatrix XC(nt);
      XC = 0.0;
      for (int i = 0; i < nl; i++)
      {
         if (r.row_nnz[i] <= 1) { continue; }
         for (int j = 0; j < nt; j++)
         {
            if (r.E(i,j) == 0.0) { continue; }
            for (int k = 0; k < nt; k++)
            {
               XC(j,k) += s(j)*r.E(i,j)*r.dl(i)*r.E(i,k);
            }
         }
      }
      // L_X = [(I - X C) Lambda_u^{-1} E_J^T, X Lambda_h^{-1}], with the
      // columns in the order of the L-DOFs; X Lambda_h^{-1} = S C^T. A J row
      // i of E has at most one nonzero, in column k, so column i of
      // (I - X C) Lambda_u^{-1} E_J^T is column k of I - X C, scaled.
      DenseMatrix Lx(nt, nl);
      Lx = 0.0;
      for (int i = 0; i < nl; i++)
      {
         if (r.row_nnz[i] <= 1)
         {
            for (int k = 0; k < nt; k++)
            {
               if (r.E(i,k) == 0.0) { continue; }
               for (int j = 0; j < nt; j++)
               {
                  Lx(j,i) = ((j == k) - XC(j,k))*lus(k)*r.E(i,k);
               }
            }
         }
         else
         {
            for (int j = 0; j < nt; j++) { Lx(j,i) = s(j)*r.E(i,j); }
         }
      }
      // L_X D_W E = I on the free DOFs, so that R_X = L_X P_0^T W is a left
      // inverse of P. The essential rows and columns are zero.
      DenseMatrix LxD(Lx), LxDE(nt);
      LxD.RightScaling(r.dl);
      Mult(LxD, r.E, LxDE);
      for (int j = 0; j < nt; j++) { if (r.free[j]) { LxDE(j,j) -= 1.0; } }
      REQUIRE(LxDE.MaxMaxNorm() <= 1e-12);
      // B_X = L_X Gm L_X^T.
      DenseMatrix LxGm(nt, nl);
      Mult(Lx, r.Gm, LxGm);
      MultABt(LxGm, Lx, B);
   }
   else
   {
      // A_0 = E^T D_W E, with zero essential rows and columns.
      DenseMatrix Dl(nl);
      Dl.Diag(r.dl.GetData(), nl);
      const DenseMatrix A0 = RtAR(Dl, r.E);
      // The interval: a = min(Lambda_u S) on the free DOFs, and b from 20
      // power iterations on S A_0, from the vector of ones on the free DOFs,
      // as in the implementation.
      real_t a = infinity();
      Vector v(nt), Av(nt);
      for (int j = 0; j < nt; j++)
      {
         if (r.free[j]) { a = std::min(a, s(j)/lus(j)); }
         v(j) = r.free[j] ? 1.0 : 0.0;
      }
      real_t b = 1.0;
      for (int it = 0; it < 20; it++)
      {
         A0.Mult(v, Av);
         real_t vSinvv = 0.0;
         for (int j = 0; j < nt; j++) { if (r.free[j]) { vSinvv += v(j)*v(j)/s(j); } }
         b = (v*Av)/vSinvv;
         for (int j = 0; j < nt; j++) { v(j) = s(j)*Av(j); }
         v /= v.Norml2();
      }
      b = std::max({b, 1.0, (1.0 + 1e-6)*a});

      // The Chebyshev semi-iteration for A_0 Q = I, with preconditioner S,
      // from Q = 0, gives the matrix Q = q_k(A_0).
      const real_t theta = 0.5*(b + a), delta = 0.5*(b - a);
      const real_t sigma = theta/delta;
      real_t rho = 1.0/sigma;
      DenseMatrix Q(nt), R(nt), D(nt), AD(nt);
      Q = 0.0;
      R = 0.0;
      for (int j = 0; j < nt; j++) { R(j,j) = 1.0; }
      D = R;
      D.LeftScaling(s);
      D *= 1.0/theta;
      for (int it = 0; ; it++)
      {
         Q += D;
         if (it == degree) { break; }
         Mult(A0, D, AD);
         R -= AD;
         const real_t rho_new = 1.0/(2.0*sigma - rho);
         DenseMatrix SR(R);
         SR.LeftScaling(s);
         D *= rho_new*rho;
         D.Add(2.0*rho_new/delta, SR);
         rho = rho_new;
      }
      // Q G Q.
      DenseMatrix QG(nt);
      Mult(Q, G, QG);
      Mult(QG, Q, B);
   }
   for (int j = 0; j < nt; j++) { if (!r.free[j]) { B(j,j) = 1.0; } }
   return B;
}

// The submatrix of A with the rows and columns in ess removed.
DenseMatrix FreeSubmatrix(const DenseMatrix &A, const Array<int> &ess)
{
   const int n = A.Height();
   Array<int> free_list;
   Array<bool> free(n);
   free = true;
   for (int i : ess) { free[i] = false; }
   for (int i = 0; i < n; i++) { if (free[i]) { free_list.Append(i); } }
   DenseMatrix S(free_list.Size());
   for (int i = 0; i < free_list.Size(); i++)
   {
      for (int j = 0; j < free_list.Size(); j++)
      {
         S(i,j) = A(free_list[i], free_list[j]);
      }
   }
   return S;
}

// Check that B is symmetric, and that M^{-1} <= B <= K M^{-1} on the free
// DOFs, i.e. lmin <= lambda(B M) <= lmax, with a small relative tolerance. If
// lmin is zero, only the upper bound is checked. M is zero on the essential
// DOFs.
void CheckSpectralBounds(const DenseMatrix &B, const DenseMatrix &M,
                         const Array<int> &ess, real_t lmin, real_t lmax)
{
   DenseMatrix Bt(B);
   Bt.Transpose();
   REQUIRE(MaxDiff(B, Bt) <= 1e-12*B.MaxMaxNorm());

   DenseMatrix Minv;
   DenseMatrixInverse(FreeSubmatrix(M, ess)).GetInverseMatrix(Minv);
   const DenseMatrix B_free = FreeSubmatrix(B, ess);
   DenseMatrix S(B_free);
   if (lmin > 0.0)
   {
      S.Add(-lmin*(1.0 - 1e-8), Minv);
      REQUIRE(IsPositiveDefinite(S));
   }
   S = Minv;
   S *= lmax*(1.0 + 1e-8);
   S -= B_free;
   REQUIRE(IsPositiveDefinite(S));
}

// The Kantorovich factor of the theory, (alpha + beta)^2/(4 alpha beta).
real_t Kantorovich(real_t alpha, real_t beta)
{
   return (alpha + beta)*(alpha + beta)/(4.0*alpha*beta);
}

// Refine nonconformingly the elements with center in x < 0.5, y < 0.5, and
// then those with center in x < 0.25, y < 0.25, which creates hanging DOFs of
// two levels. The result is the same for a serial and a parallel mesh.
void RefineNC(Mesh &mesh)
{
   mesh.EnsureNCMesh();
   for (real_t r : {0.5, 0.25})
   {
      Array<int> refs;
      Vector c;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         mesh.GetElementCenter(e, c);
         if (c(0) < r && c(1) < r) { refs.Append(e); }
      }
      mesh.GeneralRefinement(refs, 1);
   }
}

} // namespace bras_test

using namespace bras_test;

// The preconditioner, built with the space constructor, equals the dense
// reference, is symmetric, and satisfies the bounds of the theory, for all
// mesh types, spaces and weightings, with and without essential DOFs.
TEST_CASE("BRAS dense reference", "[BRAS]")
{
   const auto mesh_type = GENERATE(MeshType::QUAD, MeshType::TRI,
                                   MeshType::HEX, MeshType::TET);
   const auto space = GENERATE(Space::H1_1, Space::H1_3, Space::H1_POS_3,
                               Space::H1_VDIM, Space::L2_2, Space::ND_1,
                               Space::ND_2, Space::RT_0, Space::RT_1);
   const auto weighting = GENERATE(Weighting::IDENTITY, Weighting::DIAGONAL);
   const bool use_ess = GENERATE(false, true);
   CAPTURE(int(mesh_type), int(space), int(weighting), use_ess);

   // Small meshes, since the reference uses dense matrices.
   const bool is_3d = mesh_type == MeshType::HEX || mesh_type == MeshType::TET;
   Mesh mesh = MakeMesh(mesh_type, is_3d ? 2 : 3);
   SpaceInfo info = MakeSpace(space, mesh.Dimension());
   FiniteElementSpace fes(&mesh, info.fec.get(), info.vdim, info.ordering);

   FunctionCoefficient Q(ScalarCoeff);
   const int mdim = space == Space::H1_VDIM ? info.vdim : mesh.Dimension();
   MatrixFunctionCoefficient MQ(mdim, MatrixCoeff);

   Array<int> ess;
   if (use_ess) { fes.GetBoundaryTrueDofs(ess); }

   BilinearForm a(&fes);
   AddMassIntegrator(a, space, Q, MQ);
   a.Assemble();
   a.Finalize();

   std::unique_ptr<BRASPreconditioner> bras;
   if (UsesMatrixCoeff(space))
   {
      bras.reset(new BRASPreconditioner(fes, MQ, ess, weighting));
   }
   else
   {
      bras.reset(new BRASPreconditioner(fes, Q, ess, weighting));
   }
   REQUIRE(bras->Height() == fes.GetTrueVSize());

   const DenseMatrix B = ToDense(*bras);
   const Reference ref = BuildReference(a, weighting, ess);
   const DenseMatrix B_ref = ReferenceBW(ref);
   REQUIRE(MaxDiff(B, B_ref) <= 1e-10*B_ref.MaxMaxNorm());

   // Theorem: 1 <= lambda(B M) <= K(alpha, beta) on the free DOFs.
   const real_t K = Kantorovich(ref.alpha, ref.beta);
   CAPTURE(ref.alpha, ref.beta, K);
   CheckSpectralBounds(B, ref.M, ess, 1.0, K);
}

// On nonconforming meshes, each NCMethod equals its dense reference. The
// left inverse method satisfies lambda_min(B_X M) >= 1, and all methods
// satisfy an upper bound close to that of B_W.
TEST_CASE("BRAS nonconforming dense reference", "[BRAS]")
{
   const auto mesh_type = GENERATE(MeshType::QUAD, MeshType::HEX);
   const auto space = GENERATE(Space::H1_1, Space::H1_3, Space::H1_VDIM,
                               Space::ND_1, Space::ND_2, Space::RT_0,
                               Space::RT_1);
   const auto weighting = GENERATE(Weighting::IDENTITY, Weighting::DIAGONAL);
   const auto method = GENERATE(NCMethod::LEFT_INVERSE, NCMethod::DIAGONAL,
                                NCMethod::CHEBYSHEV);
   const bool use_ess = GENERATE(false, true);
   CAPTURE(int(mesh_type), int(space), int(weighting), int(method), use_ess);

   // Only the lowest order spaces on hexahedra, since the others give large
   // dense matrices. On quadrilaterals, RT_1 replaces RT_0.
   const bool lowest_order = space == Space::H1_1 || space == Space::ND_1 ||
                             space == Space::RT_0;
   if (mesh_type == MeshType::HEX ? !lowest_order : space == Space::RT_0)
   {
      return;
   }

   Mesh mesh = MakeMesh(mesh_type, 2);
   RefineNC(mesh);
   SpaceInfo info = MakeSpace(space, mesh.Dimension());
   FiniteElementSpace fes(&mesh, info.fec.get(), info.vdim, info.ordering);
   REQUIRE(fes.Nonconforming());

   FunctionCoefficient Q(ScalarCoeff);
   const int mdim = space == Space::H1_VDIM ? info.vdim : mesh.Dimension();
   MatrixFunctionCoefficient MQ(mdim, MatrixCoeff);
   Array<int> ess;
   if (use_ess) { fes.GetBoundaryTrueDofs(ess); }

   BilinearForm a(&fes);
   AddMassIntegrator(a, space, Q, MQ);
   a.Assemble();
   a.Finalize();
   BRASPreconditioner bras(a, ess, weighting);
   REQUIRE(bras.GetNCMethod() == NCMethod::LEFT_INVERSE);
   const int degree = 2;
   bras.SetNCMethod(method, degree);

   const DenseMatrix B = ToDense(bras);
   const Reference ref = BuildReference(a, weighting, ess);
   const DenseMatrix B_ref = ReferenceNC(ref, method, degree);
   REQUIRE(MaxDiff(B, B_ref) <= 1e-10*B_ref.MaxMaxNorm());

   // The upper bound of B_W holds for B_X and the Chebyshev method only
   // approximately, so allow a factor 2. Only B_X has the lower bound 1.
   const real_t K = Kantorovich(ref.alpha, ref.beta);
   const real_t lmin = method == NCMethod::LEFT_INVERSE ? 1.0 : 0.0;
   CheckSpectralBounds(B, ref.M, ess, lmin, 2.0*K);
}

TEST_CASE("BRAS L2 space is exact", "[BRAS]")
{
   // For discontinuous spaces P = I, so B_W = M^{-1} for any weighting.
   const auto mesh_type = GENERATE(MeshType::QUAD, MeshType::TET);
   const auto weighting = GENERATE(Weighting::IDENTITY, Weighting::DIAGONAL);
   Mesh mesh = MakeMesh(mesh_type, 2, true);
   L2_FECollection fec(3, mesh.Dimension(), BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);
   FunctionCoefficient Q(ScalarCoeff);

   BilinearForm a(&fes);
   a.AddDomainIntegrator(new MassIntegrator(Q));
   a.Assemble();
   a.Finalize();
   BRASPreconditioner bras(a, Array<int>(), weighting);

   Vector x(fes.GetTrueVSize()), Mx(x.Size()), y(x.Size());
   x.Randomize(1);
   a.Mult(x, Mx);
   bras.Mult(Mx, y);
   y -= x;
   REQUIRE(y.Normlinf() <= 1e-10*x.Normlinf());
}

// The bilinear form constructor sums all domain integrators and respects
// their attribute markers.
TEST_CASE("BRAS bilinear form constructor", "[BRAS]")
{
   Mesh mesh = MakeMesh(MeshType::QUAD, 4);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      Vector c;
      mesh.GetElementCenter(e, c);
      mesh.SetAttribute(e, c(0) < 0.5 ? 1 : 2);
   }
   mesh.SetAttributes();
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   Array<int> ess;
   fes.GetBoundaryTrueDofs(ess);

   // Mass integrators restricted to the two attributes, plus a mass
   // integrator everywhere, against the equivalent piecewise constant
   // coefficient.
   Vector q(2);
   q(0) = 2.0;
   q(1) = 5.0;
   PWConstCoefficient pw(q);
   ConstantCoefficient one(1.0), two(2.0);
   Array<int> attr1(2), attr2(2);
   attr1 = 0;
   attr1[0] = 1;
   attr2 = 0;
   attr2[1] = 1;
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new MassIntegrator(one));
   a.AddDomainIntegrator(new MassIntegrator(one), attr1);
   a.AddDomainIntegrator(new MassIntegrator(two), attr2);
   a.AddDomainIntegrator(new MassIntegrator(two), attr2);

   BRASPreconditioner bras_form(a, ess);
   BRASPreconditioner bras_coeff(fes, pw, ess);
   const DenseMatrix B1 = ToDense(bras_form), B2 = ToDense(bras_coeff);
   REQUIRE(MaxDiff(B1, B2) <= 1e-12*B2.MaxMaxNorm());
}

TEST_CASE("BRAS vector FE coefficient", "[BRAS]")
{
   // The coefficient must be used for vector-valued spaces: scaling the
   // coefficient by 4 scales M_br and W by 4, and B_W by 1/4.
   const auto space = GENERATE(Space::ND_1, Space::RT_0);
   Mesh mesh = MakeMesh(MeshType::HEX, 2);
   SpaceInfo info = MakeSpace(space, 3);
   FiniteElementSpace fes(&mesh, info.fec.get());
   ConstantCoefficient four(4.0);
   BRASPreconditioner bras1(fes, Array<int>());
   BRASPreconditioner bras4(fes, four, Array<int>());
   DenseMatrix B1 = ToDense(bras1), B4 = ToDense(bras4);
   B4 *= 4.0;
   REQUIRE(MaxDiff(B1, B4) <= 1e-12*B1.MaxMaxNorm());
}

// Update() rebuilds the preconditioner after a coefficient change and after
// mesh refinement, for both constructors.
TEST_CASE("BRAS Update", "[BRAS]")
{
   const auto weighting = GENERATE(Weighting::IDENTITY, Weighting::DIAGONAL);
   Mesh mesh = MakeMesh(MeshType::TRI, 2);
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   ConstantCoefficient c(1.0);
   Array<int> ess;
   fes.GetBoundaryTrueDofs(ess);

   BilinearForm a(&fes);
   a.AddDomainIntegrator(new MassIntegrator(c));
   BRASPreconditioner bras_form(a, ess, weighting);
   BRASPreconditioner bras_fes(fes, c, ess, weighting);

   SECTION("Coefficient change")
   {
      // The preconditioner is the identity on the essential DOFs, so compare
      // only the free DOFs.
      const DenseMatrix B1 = FreeSubmatrix(ToDense(bras_fes), ess);
      c.constant = 4.0;
      bras_form.Update();
      bras_fes.Update();
      DenseMatrix B4 = FreeSubmatrix(ToDense(bras_fes), ess);
      B4 *= 4.0;
      REQUIRE(MaxDiff(B1, B4) <= 1e-12*B1.MaxMaxNorm());
      REQUIRE(MaxDiff(ToDense(bras_form), ToDense(bras_fes)) <=
              1e-12*B1.MaxMaxNorm());
   }

   SECTION("Mesh refinement")
   {
      // After refinement the space size changes, so the essential DOFs are
      // recomputed and passed to Update(). The result must match a
      // preconditioner built from scratch on the refined space.
      mesh.UniformRefinement();
      fes.Update();
      a.Update();
      fes.GetBoundaryTrueDofs(ess);
      bras_form.Update(ess);
      bras_fes.Update(ess);
      BRASPreconditioner bras_new(fes, c, ess, weighting);
      REQUIRE(bras_form.Height() == fes.GetTrueVSize());
      const DenseMatrix B = ToDense(bras_new);
      REQUIRE(MaxDiff(ToDense(bras_form), B) <= 1e-12*B.MaxMaxNorm());
      REQUIRE(MaxDiff(ToDense(bras_fes), B) <= 1e-12*B.MaxMaxNorm());
   }
}

// Update() after adaptive nonconforming refinement, starting from a mesh
// without hanging DOFs, keeps the nonconforming method and matches a
// preconditioner built from scratch on the refined space.
TEST_CASE("BRAS nonconforming Update", "[BRAS]")
{
   const auto method = GENERATE(NCMethod::LEFT_INVERSE, NCMethod::DIAGONAL,
                                NCMethod::CHEBYSHEV);
   Mesh mesh = MakeMesh(MeshType::QUAD, 2);
   mesh.EnsureNCMesh();
   H1_FECollection fec(2, 2);
   FiniteElementSpace fes(&mesh, &fec);
   FunctionCoefficient Q(ScalarCoeff);
   Array<int> ess;
   fes.GetBoundaryTrueDofs(ess);
   BRASPreconditioner bras(fes, Q, ess);
   bras.SetNCMethod(method, 3);

   for (int it = 0; it < 2; it++)
   {
      Array<int> refs({0});
      mesh.GeneralRefinement(refs, 1);
      fes.Update();
      fes.GetBoundaryTrueDofs(ess);
      bras.Update(ess);
      REQUIRE(bras.GetNCMethod() == method);
      BRASPreconditioner bras_new(fes, Q, ess);
      bras_new.SetNCMethod(method, 3);
      const DenseMatrix B = ToDense(bras_new);
      REQUIRE(MaxDiff(ToDense(bras), B) <= 1e-12*B.MaxMaxNorm());
   }
}

#ifdef MFEM_USE_MPI

// The parallel preconditioner applied to the true DOFs of a distributed grid
// function equals the serial preconditioner applied to the corresponding
// serial grid function. This checks the parallel prolongation, including the
// orientation signs of shared ND and RT DOFs.
TEST_CASE("BRAS parallel", "[BRAS][Parallel]")
{
   const auto mesh_type = GENERATE(MeshType::QUAD, MeshType::TRI,
                                   MeshType::HEX, MeshType::TET);
   const auto space = GENERATE(Space::H1_1, Space::H1_3, Space::H1_VDIM,
                               Space::L2_2, Space::ND_1, Space::ND_2,
                               Space::RT_1);
   const auto weighting = GENERATE(Weighting::IDENTITY, Weighting::DIAGONAL);
   // For high order Nedelec spaces on tetrahedra, the parallel prolongation
   // applies a change of basis to the DOFs on shared triangular faces, which
   // changes A_W, so the serial and parallel preconditioners differ. This
   // case is tested in "BRAS parallel PCG".
   if (mesh_type == MeshType::TET && space == Space::ND_2) { return; }
   CAPTURE(int(mesh_type), int(space), int(weighting));

   const int nranks = Mpi::WorldSize();
   const bool is_3d = mesh_type == MeshType::HEX || mesh_type == MeshType::TET;
   Mesh smesh = MakeMesh(mesh_type, is_3d ? 4 : 8);
   Array<int> part = SlabPartitioning(smesh, nranks);
   ParMesh pmesh(MPI_COMM_WORLD, smesh, part.GetData());

   SpaceInfo info = MakeSpace(space, smesh.Dimension());
   FiniteElementSpace sfes(&smesh, info.fec.get(), info.vdim, info.ordering);

   FunctionCoefficient Q(ScalarCoeff);
   const int mdim = space == Space::H1_VDIM ? info.vdim : smesh.Dimension();
   MatrixFunctionCoefficient MQ(mdim, MatrixCoeff);

   // Every rank builds the same serial grid function xs and extracts its
   // local part xp. The serial space has no prolongation, so xs is also a
   // true-DOF vector; the parallel true DOFs are obtained by projection.
   GridFunction xs(&sfes), ys(&sfes);
   xs.Randomize(1);
   ParGridFunction xp(&pmesh, &xs, part.GetData());
   ParFiniteElementSpace &pfes = *xp.ParFESpace();
   Vector X(pfes.GetTrueVSize()), Y(X.Size()), Y_ref(X.Size());
   xp.ParallelProject(X);

   Array<int> ess_s, ess_p;
   sfes.GetBoundaryTrueDofs(ess_s);
   pfes.GetBoundaryTrueDofs(ess_p);

   std::unique_ptr<BRASPreconditioner> bras_s, bras_p;
   if (UsesMatrixCoeff(space))
   {
      bras_s.reset(new BRASPreconditioner(sfes, MQ, ess_s, weighting));
      bras_p.reset(new BRASPreconditioner(pfes, MQ, ess_p, weighting));
   }
   else
   {
      bras_s.reset(new BRASPreconditioner(sfes, Q, ess_s, weighting));
      bras_p.reset(new BRASPreconditioner(pfes, Q, ess_p, weighting));
   }

   // Apply both and map the serial result to parallel true DOFs.
   bras_s->Mult(xs, ys);
   ParGridFunction yp(&pmesh, &ys, part.GetData());
   yp.ParallelProject(Y_ref);
   bras_p->Mult(X, Y);

   Y -= Y_ref;
   const real_t diff = ParNormlp(Y, infinity(), MPI_COMM_WORLD);
   const real_t norm = ParNormlp(Y_ref, infinity(), MPI_COMM_WORLD);
   REQUIRE(diff <= 1e-12*norm);
}

// Same as above on nonconforming meshes, for each NCMethod. The hanging DOFs
// are the same in serial and in parallel, and the true DOFs differ only by
// orientation signs, so the methods give the same operator. Here the serial
// space has a prolongation, so the serial true-DOF vector is prolongated to a
// grid function before mapping it to the parallel space.
TEST_CASE("BRAS parallel nonconforming", "[BRAS][Parallel]")
{
   const auto mesh_type = GENERATE(MeshType::QUAD, MeshType::HEX);
   const auto space = GENERATE(Space::H1_3, Space::H1_VDIM, Space::ND_2,
                               Space::RT_1);
   const auto method = GENERATE(NCMethod::LEFT_INVERSE, NCMethod::DIAGONAL,
                                NCMethod::CHEBYSHEV);
   CAPTURE(int(mesh_type), int(space), int(method));

   const int nranks = Mpi::WorldSize();
   Mesh smesh = MakeMesh(mesh_type, mesh_type == MeshType::HEX ? 2 : 4);
   RefineNC(smesh);
   Array<int> part = SlabPartitioning(smesh, nranks);
   ParMesh pmesh(MPI_COMM_WORLD, smesh, part.GetData());

   SpaceInfo info = MakeSpace(space, smesh.Dimension());
   FiniteElementSpace sfes(&smesh, info.fec.get(), info.vdim, info.ordering);
   REQUIRE(sfes.Nonconforming());

   FunctionCoefficient Q(ScalarCoeff);
   const int mdim = space == Space::H1_VDIM ? info.vdim : smesh.Dimension();
   MatrixFunctionCoefficient MQ(mdim, MatrixCoeff);

   // xs = E Xs is a conforming grid function, so its parallel true DOFs are
   // obtained by projection.
   Vector Xs(sfes.GetTrueVSize()), Ys(Xs.Size());
   Xs.Randomize(1);
   GridFunction xs(&sfes), ys(&sfes);
   xs.SetFromTrueDofs(Xs);
   ParGridFunction xp(&pmesh, &xs, part.GetData());
   ParFiniteElementSpace &pfes = *xp.ParFESpace();
   Vector X(pfes.GetTrueVSize()), Y(X.Size()), Y_ref(X.Size());
   xp.ParallelProject(X);

   Array<int> ess_s, ess_p;
   sfes.GetBoundaryTrueDofs(ess_s);
   pfes.GetBoundaryTrueDofs(ess_p);

   std::unique_ptr<BRASPreconditioner> bras_s, bras_p;
   if (UsesMatrixCoeff(space))
   {
      bras_s.reset(new BRASPreconditioner(sfes, MQ, ess_s));
      bras_p.reset(new BRASPreconditioner(pfes, MQ, ess_p));
   }
   else
   {
      bras_s.reset(new BRASPreconditioner(sfes, Q, ess_s));
      bras_p.reset(new BRASPreconditioner(pfes, Q, ess_p));
   }
   bras_s->SetNCMethod(method);
   bras_p->SetNCMethod(method);

   bras_s->Mult(Xs, Ys);
   ys.SetFromTrueDofs(Ys);
   ParGridFunction yp(&pmesh, &ys, part.GetData());
   yp.ParallelProject(Y_ref);
   bras_p->Mult(X, Y);

   Y -= Y_ref;
   const real_t diff = ParNormlp(Y, infinity(), MPI_COMM_WORLD);
   const real_t norm = ParNormlp(Y_ref, infinity(), MPI_COMM_WORLD);
   REQUIRE(diff <= 1e-12*norm);
}

// PCG on the mass matrix with essential DOFs eliminated converges in a small,
// mesh-independent number of iterations on curved meshes, on conforming and
// nonconforming meshes, and for Nedelec spaces with DOFs on shared triangular
// faces. The bound is loose, to allow for different partitionings.
TEST_CASE("BRAS parallel PCG", "[BRAS][Parallel]")
{
   const auto mesh_type = GENERATE(MeshType::QUAD, MeshType::HEX,
                                   MeshType::TET);
   const auto space = GENERATE(Space::H1_3, Space::ND_2, Space::RT_1);
   const bool nonconforming = GENERATE(false, true);
   if (nonconforming && mesh_type == MeshType::TET) { return; }
   CAPTURE(int(mesh_type), int(space), nonconforming);

   const int nranks = Mpi::WorldSize();
   const bool is_3d = mesh_type != MeshType::QUAD;
   Mesh smesh = MakeMesh(mesh_type, is_3d ? 4 : 8, true);
   if (nonconforming) { RefineNC(smesh); }
   Array<int> part = SlabPartitioning(smesh, nranks);
   ParMesh pmesh(MPI_COMM_WORLD, smesh, part.GetData());
   SpaceInfo info = MakeSpace(space, smesh.Dimension());
   ParFiniteElementSpace pfes(&pmesh, info.fec.get());

   FunctionCoefficient Q(ScalarCoeff);
   MatrixFunctionCoefficient MQ(smesh.Dimension(), MatrixCoeff);
   ParBilinearForm a(&pfes);
   AddMassIntegrator(a, space, Q, MQ);
   a.Assemble();
   Array<int> ess;
   pfes.GetBoundaryTrueDofs(ess);
   HypreParMatrix M;
   a.FormSystemMatrix(ess, M);

   BRASPreconditioner bras(a, ess);
   Vector B(pfes.GetTrueVSize()), X(B.Size());
   B.Randomize(1);
   B.SetSubVector(ess, 0.0);
   X = 0.0;

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(100);
   cg.SetOperator(M);
   cg.SetPreconditioner(bras);
   cg.Mult(B, X);
   REQUIRE(cg.GetConverged());
   // The curved tetrahedral meshes need about 20 iterations in serial, and up
   // to 25 with Nedelec DOFs on shared triangular faces.
   REQUIRE(cg.GetNumIterations() <= (mesh_type == MeshType::TET ? 30 : 20));
}

#endif // MFEM_USE_MPI
