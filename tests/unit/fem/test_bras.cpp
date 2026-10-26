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

// Reference weighted BRAS B_W = A^{-1} P^T W M_br^{-1} W P A^{-1}, with
// A = P^T W P, built from its definition with dense matrices. Here P is the
// signed Boolean matrix mapping the free DOFs of the serial conforming space to
// the element DOFs. On the essential DOFs, B_W is the identity. Also returns
// the constants alpha and beta of the equivalence alpha W <= M_br <= beta W.
//
// Unlike the implementation, P is built from the signs of the element vdofs
// (GetElementVDofs() encodes a negative orientation as -1-dof) rather than
// from the element restriction, and A is inverted as a dense matrix, so the
// test does not rely on P being Boolean or A being diagonal. The assembled
// matrix of a must be available.
DenseMatrix ReferenceBRAS(BilinearForm &a, Weighting weighting,
                          const Array<int> &ess, real_t &alpha, real_t &beta)
{
   const FiniteElementSpace &fes = *a.FESpace();
   const int n = fes.GetVSize();
   const int ne = fes.GetNE();
   const int nd = fes.GetFE(0)->GetDof()*fes.GetVDim();
   const int nb = ne*nd;

   Array<bool> free(n);
   free = true;
   for (int i : ess) { free[i] = false; }

   // Pmat is P restricted to the free columns (the essential columns are
   // zero), WMinvWP is (W M_br^{-1} W) P.
   DenseMatrix Pmat(nb, n), WMinvWP(nb, n), elmat, inv;
   Pmat = 0.0;
   Vector w(nb);
   DenseMatrix Mbr_P(nb, n); // M_br P, to check P^T M_br P = M
   Mbr_P = 0.0;
   alpha = infinity();
   beta = 0.0;
   Array<int> vdofs;
   DofTransformation doftrans;
   std::vector<DenseMatrix> Me(ne), Me_inv(ne);
   for (int e = 0; e < ne; e++)
   {
      // The element matrix in the basis of the global DOFs, as summed into
      // the assembled matrix by BilinearForm::Assemble().
      fes.GetElementVDofs(e, vdofs, doftrans);
      a.ComputeElementMatrix(e, elmat);
      doftrans.TransformDual(elmat);
      Me[e] = elmat;
      Vector w_e(nd);
      for (int i = 0; i < nd; i++)
      {
         const int j = vdofs[i] >= 0 ? vdofs[i] : -1 - vdofs[i];
         if (free[j]) { Pmat(e*nd + i, j) = vdofs[i] >= 0 ? 1.0 : -1.0; }
         w_e(i) = weighting == Weighting::DIAGONAL ? elmat(i,i) : 1.0;
         w(e*nd + i) = w_e(i);
      }
      real_t lmin, lmax;
      ExtremeEigenvalues(elmat, w_e, lmin, lmax);
      alpha = std::min(alpha, lmin);
      beta = std::max(beta, lmax);
      DenseMatrixInverse(elmat).GetInverseMatrix(inv);
      Me_inv[e] = inv;
   }

   // Block products with M_br and W M_br^{-1} W.
   for (int e = 0; e < ne; e++)
   {
      for (int i = 0; i < nd; i++)
      {
         for (int j = 0; j < n; j++)
         {
            real_t s = 0.0, t = 0.0;
            for (int k = 0; k < nd; k++)
            {
               const real_t p = Pmat(e*nd + k, j);
               s += w(e*nd + i)*Me_inv[e](i,k)*w(e*nd + k)*p;
               t += Me[e](i,k)*p;
            }
            WMinvWP(e*nd + i, j) = s;
            Mbr_P(e*nd + i, j) = t;
         }
      }
   }

   // Check that P^T M_br P is the assembled matrix on the free DOFs.
   DenseMatrix M(n), PtMP(n);
   a.SpMat().ToDenseMatrix(M);
   MultAtB(Pmat, Mbr_P, PtMP);
   real_t err = 0.0;
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
      {
         if (free[i] && free[j]) { err = std::max(err, std::abs(M(i,j) - PtMP(i,j))); }
      }
   }
   REQUIRE(err <= 1e-12*M.MaxMaxNorm());

   // A = P^T W P and G = P^T W M_br^{-1} W P. The essential rows and columns
   // of A are zero, so set A_ii = 1 there to make it invertible; this does
   // not affect the free block of B, and the essential rows and columns of G
   // (and therefore of B) are zero.
   DenseMatrix WP(Pmat), A(n), G(n), AinvG(n), B(n), Ainv;
   for (int i = 0; i < nb; i++)
   {
      for (int j = 0; j < n; j++) { WP(i,j) *= w(i); }
   }
   MultAtB(Pmat, WP, A);
   for (int i : ess) { A(i,i) = 1.0; }
   MultAtB(Pmat, WMinvWP, G);
   DenseMatrixInverse(A).GetInverseMatrix(Ainv);
   Mult(Ainv, G, AinvG);
   Mult(AinvG, Ainv, B);
   for (int i : ess) { B(i,i) = 1.0; }
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

// max_ij |A_ij - B_ij|.
real_t MaxDiff(const DenseMatrix &A, const DenseMatrix &B)
{
   DenseMatrix D(A);
   D -= B;
   return D.MaxMaxNorm();
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

   real_t alpha, beta;
   const DenseMatrix B = ToDense(*bras);
   const DenseMatrix B_ref = ReferenceBRAS(a, weighting, ess, alpha, beta);
   const real_t B_norm = B_ref.MaxMaxNorm();
   REQUIRE(MaxDiff(B, B_ref) <= 1e-10*B_norm);

   DenseMatrix Bt(B);
   Bt.Transpose();
   REQUIRE(MaxDiff(B, Bt) <= 1e-12*B_norm);

   // Theorem: 1 <= lambda(B M) <= K(alpha, beta) on the free DOFs, that is
   // M^{-1} <= B <= K M^{-1}, with the Kantorovich factor K below. Both
   // inequalities are checked with a small relative tolerance.
   DenseMatrix M(fes.GetVSize()), Minv;
   a.SpMat().ToDenseMatrix(M);
   DenseMatrixInverse(FreeSubmatrix(M, ess)).GetInverseMatrix(Minv);
   const DenseMatrix B_free = FreeSubmatrix(B, ess);
   const real_t K = (alpha + beta)*(alpha + beta)/(4.0*alpha*beta);
   CAPTURE(alpha, beta, K);
   DenseMatrix S(B_free);
   S.Add(-(1.0 - 1e-8), Minv);
   REQUIRE(IsPositiveDefinite(S));
   S = Minv;
   S *= K*(1.0 + 1e-8);
   S -= B_free;
   REQUIRE(IsPositiveDefinite(S));
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
   // DOFs on shared triangular faces of high order Nedelec spaces are not
   // supported.
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

// PCG on the mass matrix with essential DOFs eliminated converges in a small,
// mesh-independent number of iterations on curved meshes. The bound is loose,
// to allow for different partitionings.
TEST_CASE("BRAS parallel PCG", "[BRAS][Parallel]")
{
   const auto mesh_type = GENERATE(MeshType::QUAD, MeshType::HEX);
   const auto space = GENERATE(Space::H1_3, Space::ND_2, Space::RT_1);
   CAPTURE(int(mesh_type), int(space));

   const int nranks = Mpi::WorldSize();
   const bool is_3d = mesh_type == MeshType::HEX;
   Mesh smesh = MakeMesh(mesh_type, is_3d ? 4 : 8, true);
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
   REQUIRE(cg.GetNumIterations() <= 20);
}

#endif // MFEM_USE_MPI
