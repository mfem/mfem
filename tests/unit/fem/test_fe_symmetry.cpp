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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

// Use the Hungarian algorithm to find the column permutation of A which
// produces a matrix with the minimum trace.
Array<int> findMinTracePermutation(const DenseMatrix &A)
{
   const int R = A.NumRows(), C = A.NumCols();
   MFEM_VERIFY(R <= C, "Matrix must have at least as many columns as rows");

   Array<int> perm(C + 1); perm = -1;
   Vector potR(R); potR = 0.0;
   Vector potC(C + 1); potC = 0.0;

   const real_t inf = std::numeric_limits<real_t>::max();

   for (int r_cur = 0; r_cur < R; ++r_cur)
   {
      int c_cur = C;
      perm[c_cur] = r_cur;

      Vector min_to(C + 1); min_to = inf;
      Array<int> prv_col(C + 1); prv_col = -1;
      Array<bool> col_in_path(C + 1); col_in_path = false;

      while (perm[c_cur] != -1)
      {
         col_in_path[c_cur] = true;
         const int r = perm[c_cur];
         real_t delta = inf;
         int c_next = 0;
         for (int c = 0; c < C; ++c)
         {
            if (!col_in_path[c])
            {
               const real_t d = A(r,c) - potR[r] - potC[c];
               if (d < min_to[c])
               {
                  min_to[c] = d;
                  prv_col[c] = c_cur;
               }
               if (min_to[c] < delta)
               {
                  delta = min_to[c];
                  c_next = c;
               }
            }
         }
         for (int c = 0; c <= C; ++c)
         {
            if (col_in_path[c])
            {
               potR[perm[c]] += delta;
               potC[c] -= delta;
            }
            else
            {
               min_to[c] -= delta;
            }
         }
         c_cur = c_next;
      }
      for (int c; c_cur != C; c_cur = c)
      {
         c = prv_col[c_cur];
         perm[c_cur] = perm[c];
      }
   }

   return perm;
}

/// We require that our pyramid basis functions possess four-fold rotational
/// symmetry. This implies that:
///    P s1 t1 = s0
/// Where s0 and s1 are the shape functions evaluated at a random point and its
/// image under rotation respectively. Also, t1 is the Piola transform
/// for the finte element type and P is a signed permutation matrix. The signs
/// of the permutation entries can be determined by the conventions used
/// for the DoFs of the various basis functions. These signs are passed to this
/// function in the ps argument. The remaining structure of the permutation is
/// computed using findMinTracePermutation with the matrix ps * s1 * t1 * s0^T.
///
real_t computeVShapeDifference(const DenseMatrix &s0,
                               const DenseMatrix &ts1,
                               const Vector &ps)
{
   const int dof = s0.Height();
   const int dim = s0.Width();

   DenseMatrix pts1(ts1);
   pts1.LeftScaling(ps);

   DenseMatrix sts(dof);
   MultABt(pts1, s0, sts); sts *= -1.0;
   Array<int> perm = findMinTracePermutation(sts);

   real_t nrm = 0.0;
   for (int i=0; i<perm.Size() - 1; i++)
   {
      int i0 = i;
      int i1 = perm[i];

      for (int d=0; d<dim; d++)
      {
         nrm += pow(s0(i0, d) - pts1(i1, d), 2);
      }
   }

   return nrm;
}

TEST_CASE("FE Symmetry",
          "[H1_FuentesPyramidElemet]"
          "[ND_FuentesPyramidElemet]"
          "[RT_FuentesPyramidElemet]"
          "[L2_FuentesPyramidElemet]")
{
   const int order = 3;
   const int npts = 3;
   const real_t tol = 1e-13;

   CAPTURE(order);

   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::PYRAMID);
   {
      DenseMatrix &ptMat = T.GetPointMat();
      ptMat.SetCol(0, Vector({-0.5, -0.5, 0.0}));
      ptMat.SetCol(1, Vector({ 0.5, -0.5, 0.0}));
      ptMat.SetCol(2, Vector({ 0.5,  0.5, 0.0}));
      ptMat.SetCol(3, Vector({-0.5,  0.5, 0.0}));
      ptMat.SetCol(4, Vector({ 0.0,  0.0, M_SQRT1_2}));
      T.Reset();
   }

   for (int k=0; k<npts; k++)
   {
      double a = rand() / double(RAND_MAX);
      double b = rand() / double(RAND_MAX);
      double c = 0.9 * rand() / double(RAND_MAX);

      // Select a random point inside a pyramid
      IntegrationPoint ip0;
      ip0.x = a * (1.0 - c); ip0.y = b * (1.0 - c); ip0.z = c;

      SECTION("H1_FuentesFiniteElement")
      {
         H1_FuentesPyramidElement fe(order);

         DenseMatrix s0(fe.GetDof(), 1);
         DenseMatrix s1(fe.GetDof(), 1);

         Vector v0(s0.GetData(), fe.GetDof());
         Vector v1(s1.GetData(), fe.GetDof());

         T.SetIntPoint(&ip0);
         fe.CalcPhysShape(T, v0);

         // 90 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = ip0.y;
            ip1.y = 1.0 - ip0.x - ip0.z;
            ip1.z = ip0.z;

            T.SetIntPoint(&ip1);
            fe.CalcPhysShape(T, v1);

            Vector ps(fe.GetDof()); ps = 1.0;

            REQUIRE(computeVShapeDifference(s0, s1, ps) < tol);
         }

         // 180 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = 1.0 - ip0.x - ip0.z;
            ip1.y = 1.0 - ip0.y - ip0.z;
            ip1.z = ip0.z;

            T.SetIntPoint(&ip1);
            fe.CalcPhysShape(T, v1);

            Vector ps(fe.GetDof()); ps = 1.0;

            REQUIRE(computeVShapeDifference(s0, s1, ps) < tol);
         }

         // 270 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = 1.0 - ip0.y - ip0.z;
            ip1.y = ip0.x;
            ip1.z = ip0.z;

            T.SetIntPoint(&ip1);
            fe.CalcPhysShape(T, v1);

            Vector ps(fe.GetDof()); ps = 1.0;

            REQUIRE(computeVShapeDifference(s0, s1, ps) < tol);
         }
      }

      SECTION("ND_FuentesPyramidElement")
      {
         const int ne = order; // Num DoFs per edge
         const int nt = order * (order - 1); // Num DoF per tri face
         const int nq = 2 * nt; // Num DoF per quad face
         // Num DoF per interior dir
         const int ni = order * (static_cast<int>(pow(order-1, 2)));
         const int oq = 8 * ne; // Offset to first quad DoF
         const int ot = oq + nq; // Offset to first tri DoF
         const int oi = ot + 4 * nt; // Offset to first interior DoF

         ND_FuentesPyramidElement fe(order);

         DenseMatrix s0(fe.GetDof(), 3);
         DenseMatrix s1(fe.GetDof(), 3);

         DenseMatrix t1(3);
         DenseMatrix t1Inv(3);
         DenseMatrix ts1(fe.GetDof(), 3);

         T.SetIntPoint(&ip0);
         fe.CalcPhysVShape(T, s0);

         // 90 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = ip0.y;
            ip1.y = 1.0 - ip0.x - ip0.z;
            ip1.z = ip0.z;

            t1Inv = 0.0; t1Inv(0,1) = 1.0; t1Inv(1,0) = -1.0; t1Inv(2,2) = 1.0;

            T.SetIntPoint(&ip1);
            fe.CalcPhysVShape(T, s1); Mult(s1, t1Inv, ts1);

            Vector ps(fe.GetDof()); ps = 1.0;
            for (int i = ne; i < 2 * ne; i++) { ps[i] = -1; }
            for (int i = 3 * ne; i<4*ne; i++) { ps[i] = -1; }
            for (int i = oq; i < oq + nt; i++) { ps[i] = -1; }
            for (int i = oi + ni; i < oi + 2 * ni; i++) { ps[i] = -1; }

            REQUIRE(computeVShapeDifference(s0, ts1, ps) < tol);
         }

         // 180 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = 1.0 - ip0.x - ip0.z;
            ip1.y = 1.0 - ip0.y - ip0.z;
            ip1.z = ip0.z;

            t1Inv = 0.0; t1Inv(0,0) = -1.0; t1Inv(1,1) = -1.0; t1Inv(2,2) = 1.0;

            T.SetIntPoint(&ip1);
            fe.CalcPhysVShape(T, s1); Mult(s1, t1Inv, ts1);

            Vector ps(fe.GetDof()); ps = 1.0;
            for (int i = 0; i < 4 * ne; i++) { ps[i] = -1; }
            for (int i = oq; i < oq + nq; i++) { ps[i] = -1; }
            for (int i = oi; i < oi + 2 * ni; i++) { ps[i] = -1; }

            REQUIRE(computeVShapeDifference(s0, ts1, ps) < tol);
         }

         // 270 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = 1.0 - ip0.y - ip0.z;
            ip1.y = ip0.x;
            ip1.z = ip0.z;

            t1Inv = 0.0; t1Inv(0,1) = -1.0; t1Inv(1,0) = 1.0; t1Inv(2,2) = 1.0;

            T.SetIntPoint(&ip1);
            fe.CalcPhysVShape(T, s1); Mult(s1, t1Inv, ts1);

            Vector ps(fe.GetDof()); ps = 1.0;
            for (int i = 0; i < ne; i++) { ps[i] = -1; }
            for (int i = 2 * ne; i < 3 * ne; i++) { ps[i] = -1; }
            for (int i = oq + nt; i < ot; i++) { ps[i] = -1; }
            for (int i = oi; i < oi + ni; i++) { ps[i] = -1; }

            REQUIRE(computeVShapeDifference(s0, ts1, ps) < tol);
         }
      }

      SECTION("RT_FuentesPyramidElement")
      {
         const int nq = order * order; // Num DoF per quad face
         const int nt = (order * (order + 1)) / 2; // Num DoF per tri face
         const int ni = order * order * (order - 1); // Num interior DoF per dir
         const int ot = nq; // Offset to first tri DoF
         const int oi = ot + 4 * nt; // Offset to first interior DoF

         RT_FuentesPyramidElement fe(order - 1);

         DenseMatrix s0(fe.GetDof(), 3);
         DenseMatrix s1(fe.GetDof(), 3);

         DenseMatrix t1(3);
         DenseMatrix ts1(fe.GetDof(), 3);

         T.SetIntPoint(&ip0);
         fe.CalcPhysVShape(T, s0);

         // 90 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = ip0.y;
            ip1.y = 1.0 - ip0.x - ip0.z;
            ip1.z = ip0.z;

            t1 = 0.0; t1(0,1) = -1.0; t1(1,0) = 1.0; t1(2,2) = 1.0;

            T.SetIntPoint(&ip1);
            fe.CalcPhysVShape(T, s1); MultABt(s1, t1, ts1);

            Vector ps(fe.GetDof()); ps = 1.0;
            for (int i = oi + ni; i < oi + 2 * ni; i++)
            {
               ps[i] = -1.0;
            }

            REQUIRE(computeVShapeDifference(s0, ts1, ps) < tol);
         }

         // 180 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = 1.0 - ip0.x - ip0.z;
            ip1.y = 1.0 - ip0.y - ip0.z;
            ip1.z = ip0.z;

            t1 = 0.0; t1(0,0) = -1.0; t1(1,1) = -1.0; t1(2,2) = 1.0;

            T.SetIntPoint(&ip1);
            fe.CalcPhysVShape(T, s1); MultABt(s1, t1, ts1);

            Vector ps(fe.GetDof()); ps = 1.0;
            for (int i = oi; i < oi + 2 * ni; i++)
            {
               ps[i] = -1.0;
            }

            REQUIRE(computeVShapeDifference(s0, ts1, ps) < tol);
         }

         // 270 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = 1.0 - ip0.y - ip0.z;
            ip1.y = ip0.x;
            ip1.z = ip0.z;

            t1 = 0.0; t1(0,1) = 1.0; t1(1,0) = -1.0; t1(2,2) = 1.0;

            T.SetIntPoint(&ip1);
            fe.CalcPhysVShape(T, s1); MultABt(s1, t1, ts1);

            Vector ps(fe.GetDof()); ps = 1.0;
            for (int i = oi; i < oi + ni; i++)
            {
               ps[i] = -1.0;
            }

            REQUIRE(computeVShapeDifference(s0, ts1, ps) < tol);
         }
      }

      SECTION("L2_FuentesFiniteElement")
      {
         L2_FuentesPyramidElement fe(order);

         DenseMatrix s0(fe.GetDof(), 1);
         DenseMatrix s1(fe.GetDof(), 1);

         Vector v0(s0.GetData(), fe.GetDof());
         Vector v1(s1.GetData(), fe.GetDof());

         T.SetIntPoint(&ip0);
         fe.CalcPhysShape(T, v0);

         // 90 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = ip0.y;
            ip1.y = 1.0 - ip0.x - ip0.z;
            ip1.z = ip0.z;

            T.SetIntPoint(&ip1);
            fe.CalcPhysShape(T, v1);

            Vector ps(fe.GetDof()); ps = 1.0;

            REQUIRE(computeVShapeDifference(s0, s1, ps) < tol);
         }

         // 180 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = 1.0 - ip0.x - ip0.z;
            ip1.y = 1.0 - ip0.y - ip0.z;
            ip1.z = ip0.z;

            T.SetIntPoint(&ip1);
            fe.CalcPhysShape(T, v1);

            Vector ps(fe.GetDof()); ps = 1.0;

            REQUIRE(computeVShapeDifference(s0, s1, ps) < tol);
         }

         // 270 Degree Rotational Symmetry
         {
            IntegrationPoint ip1;
            ip1.x = 1.0 - ip0.y - ip0.z;
            ip1.y = ip0.x;
            ip1.z = ip0.z;

            T.SetIntPoint(&ip1);
            fe.CalcPhysShape(T, v1);

            Vector ps(fe.GetDof()); ps = 1.0;

            REQUIRE(computeVShapeDifference(s0, s1, ps) < tol);
         }
      }
   }
}
