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
#include "make_permuted_mesh.hpp"
#include "../linalg/test_same_matrices.hpp"

using namespace mfem;

class DG_LOR_DiffusionPreconditioner : public BilinearFormIntegrator
{
   Mesh &mesh;
   double kappa;
   int p;
   IntegrationRule gl_p, gl_pp1;
   Vector shape1, shape2, nor;

public:
   DG_LOR_DiffusionPreconditioner(Mesh &mesh_, int p_, double kappa_)
      : mesh(mesh_), kappa(kappa_), p(p_)
   {
      QuadratureFunctions1D::GaussLobatto(p+1, &gl_p);
      QuadratureFunctions1D::GaussLobatto(p+2, &gl_pp1);
   }

   double PenaltyFactor(int idx1, int idx2)
   {
      int pp1 = p + 1;

      int x1 = idx1 % pp1;
      int y1 = (idx1 / pp1) % pp1;
      int z1 = (idx1 / pp1) / pp1;

      int x2 = idx2 % pp1;
      int y2 = (idx2 / pp1) % pp1;
      int z2 = (idx2 / pp1) / pp1;

      int dim = mesh.Dimension();

      auto compute_factor = [&](int i1, int i2)
      {
         int j = std::min(i1, i2);
         if (i1 == i2)
         {
            double w = gl_p[j].weight;
            double k = gl_pp1[i1+1].x - gl_pp1[i1].x;
            return w/k;
         }
         else
         {
            double h = gl_p[j+1].x - gl_p[j].x;
            double k1 = gl_pp1[i1+1].x - gl_pp1[i1].x;
            double k2 = gl_pp1[i2+1].x - gl_pp1[i2].x;
            double avg = 0.5*k1 + 0.5*k2;
            return avg/h;
         }
      };

      double factor = compute_factor(x1, x2);
      if (dim >= 2) { factor *= compute_factor(y1, y2); }
      if (dim == 3) { factor *= compute_factor(z1, z2); }

      return factor;
   }

   double BdrPenaltyFactor(int idx, int f)
   {
      int pp1 = p+1;

      int x = idx % pp1;
      int y = (idx / pp1) % pp1;
      int z = (idx / pp1) / pp1;

      int dim = mesh.Dimension();

      auto subcell_size = [&](int i)
      {
         return gl_pp1[i+1].x - gl_pp1[i].x;
      };

      double factor = (p+1)*(p+1);
      if (dim == 1)
      {
         factor *= subcell_size(x);
      }
      else if (dim == 2)
      {
         int ni, nj;
         ni = (f == 1 || f == 3) ? x : y;
         nj = (f == 1 || f == 3) ? y : x;
         factor *= subcell_size(ni)/subcell_size(nj)*gl_p[nj].weight;
      }
      else if (dim == 3)
      {
         int ni, nj, nk;
         if (f == 2 || f == 4) { ni = x; nj = y; nk = z; }
         else if (f == 1 || f == 3) { ni = y; nj = x; nk = z; }
         else { ni = z; nj = x; nk = y; }
         factor *= subcell_size(ni)/subcell_size(nj)/subcell_size(nk);
         factor *= gl_p[nj].weight*gl_p[nk].weight;
      }

      return factor;
   }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat) override
   {
      int dim, ndof1, ndof2, ndofs;
      double w, wq = 0.0;

      dim = el1.GetDim();
      ndof1 = el1.GetDof();

      nor.SetSize(dim);

      shape1.SetSize(ndof1);
      if (Trans.Elem2No >= 0)
      {
         ndof2 = el2.GetDof();
         shape2.SetSize(ndof2);
      }
      else
      {
         ndof2 = 0;
      }

      int face_no;
      if (ndof2) { face_no = Trans.ElementNo; }
      else { face_no = mesh.GetBdrElementFaceIndex(Trans.ElementNo); }

      int info1, info2;
      mesh.GetFaceInfos(face_no, &info1, &info2);
      int local_face = info1/64;

      const CoarseFineTransformations &cftr = mesh.GetRefinementTransforms();

      double factor;
      bool interior = false;
      if (Trans.Elem2No >= 0 && Trans.Elem2No < mesh.GetNE())
      {
         int parent_el1 = cftr.embeddings[Trans.Elem1No].parent;
         int parent_el2 = cftr.embeddings[Trans.Elem2No].parent;
         if (parent_el1 == parent_el2)
         {
            interior = true;
            factor = PenaltyFactor(cftr.embeddings[Trans.Elem1No].matrix,
                                   cftr.embeddings[Trans.Elem2No].matrix);
         }
      }
      if (!interior)
      {
         factor = kappa*BdrPenaltyFactor(cftr.embeddings[Trans.Elem1No].matrix,
                                         local_face);
      }

      ndofs = ndof1 + ndof2;
      elmat.SetSize(ndofs);
      elmat = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL) { ir = &IntRules.Get(Trans.GetGeometryType(), 1); }

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         Trans.SetAllIntPoints(&ip);
         const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
         const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

         if (dim == 1) { nor(0) = 2*eip1.x - 1.0; }
         else { CalcOrtho(Trans.Jacobian(), nor); }

         el1.CalcShape(eip1, shape1);
         w = ip.weight;

         double h_face = nor.Norml2();
         double h_el = Trans.Elem1->Weight();
         double h = h_el/h_face; // perpendicular element size

         if (ndof2)
         {
            el2.CalcShape(eip2, shape2);
            double h_el_2 = Trans.Elem2->Weight();
            h = 0.5*h + 0.5*h_el_2/h_face; // average both element sizes
         }

         if (interior)
         {
            wq = w*factor*h_face/h;
         }
         else
         {
            wq = w*factor*h_face/h;
         }
         for (int i = 0; i < ndof1; i++)
         {
            const double wsi = wq*shape1(i);
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += wsi * shape1(j);
            }
         }
         if (ndof2)
         {
            for (int i = 0; i < ndof2; i++)
            {
               const double wsi = wq*shape2(i);
               for (int j = 0; j < ndof1; j++)
               {
                  elmat(ndof1 + i, j) -= wsi * shape1(j);
                  elmat(j, ndof1 + i) -= wsi * shape1(j);
               }
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1 + i, ndof1 + j) += wsi * shape2(j);
               }
            }
         }
      }
   }
};

class DG_LOR_MassPreconditioner : public BilinearFormIntegrator
{
   Mesh &mesh_ho, &mesh_lor;
   const int p;
   IntegrationRule gll;
   Coefficient *Q;

public:
   DG_LOR_MassPreconditioner(Mesh &mesh_ho_,
                             Mesh &mesh_lor_,
                             int p_,
                             Coefficient *Q_)
      : mesh_ho(mesh_ho_),
        mesh_lor(mesh_lor_),
        p(p_),
        Q(Q_)
   {
      QuadratureFunctions1D::GaussLobatto(p+1, &gll);
   }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Tr,
                              DenseMatrix &elmat) override
   {
      const CoarseFineTransformations &cftr = mesh_lor.GetRefinementTransforms();
      const int parent_el = cftr.embeddings[Tr.ElementNo].parent;
      // We use the point matrix index to identify the local LOR element index
      // within the high-order coarse element.
      const int lor_index = cftr.embeddings[Tr.ElementNo].matrix;

      // Assuming piecewise constant
      elmat.SetSize(1);

      const int dim = mesh_ho.Dimension();
      IntegrationPoint ip;
      if (dim == 2)
      {
         const int iy = lor_index / (p + 1);
         const int ix = lor_index  % (p + 1);
         ip.x = gll[ix].x;
         ip.y = gll[iy].x;

         elmat(0,0) = gll[ix].weight * gll[iy].weight;
      }
      else if (dim == 3)
      {
         const int iz = lor_index / (p + 1) / (p + 1);
         const int iy = (lor_index / (p + 1)) % (p + 1);
         const int ix = lor_index  % (p + 1);

         ip.x = gll[ix].x;
         ip.y = gll[iy].x;
         ip.z = gll[iz].x;

         elmat(0,0) = gll[ix].weight * gll[iy].weight * gll[iz].weight;
      }

      ElementTransformation &Tr_ho = *mesh_ho.GetElementTransformation(parent_el);
      Tr_ho.SetIntPoint(&ip);
      const real_t detJ = Tr_ho.Weight();
      elmat(0,0) *= detJ;

      if (Q)
      {
         elmat(0,0) *= Q->Eval(Tr_ho, ip);
      }
   }
};

static void TestBatchedLOR_DG(Mesh &mesh, int order)
{
   DG_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);

   // Set up some coefficients using H1 grid functions
   H1_FECollection h1fec(2, mesh.Dimension());
   FiniteElementSpace h1fes(&mesh, &h1fec);
   GridFunction gf1(&h1fes), gf2(&h1fes);
   gf1.Randomize(1);
   gf2.Randomize(2);
   GridFunctionCoefficient mass_coeff(&gf1);
   GridFunctionCoefficient diff_coeff(&gf2);

   ConstantCoefficient one(1.0);
   constexpr real_t sigma = -1.0;
   const int eta = 2;
   const int kappa = eta * (order + 1) * (order + 1);
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));

   Array<int> ess_dofs; // Empty
   LORDiscretization lor(fespace);
   lor.AssembleSystem(a, ess_dofs);
   SparseMatrix &A1 = lor.GetAssembledMatrix();

   FiniteElementSpace &fes_lor = lor.GetFESpace();
   Mesh &mesh_lor = *fes_lor.GetMesh();
   BilinearForm a_lor(&fes_lor);
   a_lor.AddBdrFaceIntegrator(new DG_LOR_DiffusionPreconditioner(
                                 mesh_lor, order, eta));
   a_lor.AddInteriorFaceIntegrator(new DG_LOR_DiffusionPreconditioner(
                                      mesh_lor, order, eta));

   a_lor.Assemble();
   a_lor.Finalize();
   SparseMatrix &A2 = a_lor.SpMat();

   TestSameMatrices(A1, A2);
}

TEST_CASE("LOR Batched DG Orientation", "[LOR][BatchedLOR][CUDA]")
{
   const int order = 3;
   const int dim = launch_all_non_regression_tests ? GENERATE(2, 3) : 2;
   const int orientation1 = GENERATE_COPY(range(0, dim == 2 ? 4 : 24));
   const int orientation2 = GENERATE_COPY(range(0, dim == 2 ? 4 : 24));

   CAPTURE(order, dim, orientation1, orientation2);

   Mesh mesh = MeshOrientation(dim, orientation1, orientation2);
   TestBatchedLOR_DG(mesh, order);
}

TEST_CASE("LOR Batched DG", "[LOR][BatchedLOR][CUDA]")
{
   const int order = 3;
   const auto mesh_fname = GENERATE(
                              "../../data/beam-quad.mesh",
                              "../../data/l-shape.mesh",
                              "../../data/beam-hex.mesh",
                              "../../data/fichera.mesh"
                           );
   CAPTURE(mesh_fname);
   Mesh mesh = Mesh::LoadFromFile(mesh_fname);

   mesh.Transform([](const Vector &xin, Vector &xout)
   {
      for (int d = 0; d < xin.Size(); ++d)
      {
         xout[d] = xin[d] * (1.0 + d / 3.0);
      }
   });

   TestBatchedLOR_DG(mesh, order);
}
