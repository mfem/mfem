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

#include <cstring>
#include <random>

using namespace mfem;

namespace hyperelastic_assembleh
{

bool Same(const DenseMatrix &A, const DenseMatrix &B)
{
   REQUIRE(A.Height() == B.Height());
   REQUIRE(A.Width() == B.Width());
   const int n = A.Height()*A.Width();
   return std::memcmp(A.Data(), B.Data(), n*sizeof(real_t)) == 0;
}

void FillRandom(DenseMatrix &M, std::mt19937 &gen, real_t lo, real_t hi)
{
   std::uniform_real_distribution<real_t> dist(lo, hi);
   for (int i = 0; i < M.Height()*M.Width(); i++) { M.Data()[i] = dist(gen); }
}

// Copy of the loops of NeoHookeanModel::AssembleH before the loop reordering.
void NeoHookeanAssembleHOld(real_t mu, real_t K, real_t g,
                            const DenseMatrix &J, const DenseMatrix &DS,
                            real_t weight, DenseMatrix &A)
{
   int dof = DS.Height(), dim = DS.Width();
   DenseMatrix Z(dim), G(dof, dim), C(dof, dim);

   real_t dJ = J.Det();
   real_t sJ = dJ/g;
   real_t a  = mu*pow(dJ, -2.0/dim);
   real_t bc = a*(J*J)/dim;
   real_t b  = bc - K*sJ*(sJ - 1.0);
   real_t c  = 2.0*bc/dim + K*sJ*(2.0*sJ - 1.0);

   CalcAdjugateTranspose(J, Z);
   Z *= (1.0/dJ);

   MultABt(DS, J, C);
   MultABt(DS, Z, G);

   a *= weight;
   b *= weight;
   c *= weight;

   for (int i = 0; i < dof; i++)
      for (int k = 0; k <= i; k++)
      {
         real_t s = 0.0;
         for (int d = 0; d < dim; d++)
         {
            s += DS(i,d)*DS(k,d);
         }
         s *= a;

         for (int d = 0; d < dim; d++)
         {
            A(i+d*dof,k+d*dof) += s;
         }

         if (k != i)
            for (int d = 0; d < dim; d++)
            {
               A(k+d*dof,i+d*dof) += s;
            }
      }

   a *= (-2.0/dim);

   for (int i = 0; i < dof; i++)
      for (int j = 0; j < dim; j++)
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dim; l++)
            {
               A(i+j*dof,k+l*dof) +=
                  a*(C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                  b*G(i,l)*G(k,j) + c*G(i,j)*G(k,l);
            }
}

} // namespace hyperelastic_assembleh

using namespace hyperelastic_assembleh;

TEST_CASE("NeoHookeanModel AssembleH loop order", "[HyperelasticModel]")
{
   std::mt19937 gen(5);
   const int dims[] = {2, 3, 3, 3, 2, 2};
   const int dofs[] = {4, 8, 27, 10, 9, 1};
   for (int t = 0; t < 6; t++)
   {
      const int dim = dims[t], dof = dofs[t];
      DYNAMIC_SECTION("dim " << dim << " dof " << dof)
      {
         NeoHookeanModel model(0.37, 1.9, 1.3);
         DenseMatrix J(dim), DS(dof, dim);
         for (int rep = 0; rep < 10; rep++)
         {
            do
            {
               FillRandom(J, gen, -0.5, 0.5);
               for (int i = 0; i < dim; i++) { J(i,i) += 1.0; }
            }
            while (J.Det() < 0.2);
            FillRandom(DS, gen, -1.5, 1.5);

            DenseMatrix A(dof*dim), Aref(dof*dim);
            FillRandom(A, gen, -1.0, 1.0);
            Aref = A;
            const real_t w = 0.31 + 0.1*rep;
            model.AssembleH(J, DS, w, A);
            NeoHookeanAssembleHOld(0.37, 1.9, 1.3, J, DS, w, Aref);
            REQUIRE(Same(A, Aref));
         }
      }
   }
}

TEST_CASE("IncompressibleNeoHookeanIntegrator element gradient loop order",
          "[HyperelasticModel]")
{
   std::mt19937 gen(11);
   for (int dim = 2; dim <= 3; dim++)
   {
      DYNAMIC_SECTION("dim " << dim)
      {
         Mesh mesh = dim == 2 ? Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL)
                     : Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
         H1_FECollection fec_u(2, dim), fec_p(1, dim);
         const FiniteElement &fe_u = *fec_u.FiniteElementForGeometry(
                                        mesh.GetElementGeometry(0));
         const FiniteElement &fe_p = *fec_p.FiniteElementForGeometry(
                                        mesh.GetElementGeometry(0));
         const int dof_u = fe_u.GetDof(), dof_p = fe_p.GetDof();

         // Nodal positions of the deformed element.
         Vector xu(dof_u*dim), pr(dof_p);
         std::uniform_real_distribution<real_t> dist(-0.1, 0.1);
         const IntegrationRule &nodes = fe_u.GetNodes();
         for (int i = 0; i < dof_u; i++)
         {
            for (int d = 0; d < dim; d++)
            {
               xu(i + d*dof_u) = (&nodes.IntPoint(i).x)[d] + dist(gen);
            }
         }
         for (int i = 0; i < dof_p; i++) { pr(i) = 0.7 + dist(gen); }

         ConstantCoefficient mu(0.43);
         IncompressibleNeoHookeanIntegrator integ(mu);
         Array<const FiniteElement *> els(2);
         els[0] = &fe_u; els[1] = &fe_p;
         Array<const Vector *> elfun(2);
         elfun[0] = &xu; elfun[1] = &pr;
         DenseMatrix m00, m01, m10, m11;
         Array2D<DenseMatrix *> elmats(2, 2);
         elmats(0,0) = &m00; elmats(0,1) = &m01;
         elmats(1,0) = &m10; elmats(1,1) = &m11;
         ElementTransformation &Tr = *mesh.GetElementTransformation(0);
         integ.AssembleElementGrad(els, Tr, elfun, elmats);

         // Reference: the u,u block with the loop order before the change.
         DenseMatrix ref(dof_u*dim);
         ref = 0.0;
         DenseMatrix DSh_u(dof_u, dim), DS_u(dof_u, dim), J0i(dim), F(dim),
                     FinvT(dim), PMatI_u;
         Vector Sh_p(dof_p);
         PMatI_u.UseExternalData(xu.GetData(), dof_u, dim);
         const IntegrationRule &ir = IntRules.Get(fe_u.GetGeomType(),
                                                  2*fe_u.GetOrder() + 3);
         for (int p = 0; p < ir.GetNPoints(); ++p)
         {
            const IntegrationPoint &ip = ir.IntPoint(p);
            Tr.SetIntPoint(&ip);
            CalcInverse(Tr.Jacobian(), J0i);
            fe_u.CalcDShape(ip, DSh_u);
            Mult(DSh_u, J0i, DS_u);
            MultAtB(PMatI_u, DS_u, F);
            fe_p.CalcShape(ip, Sh_p);
            real_t pres = Sh_p * pr;
            real_t mu_v = mu.Eval(Tr, ip);
            real_t dJ = F.Det();
            CalcInverseTranspose(F, FinvT);
            for (int i_u = 0; i_u < dof_u; ++i_u)
               for (int i_dim = 0; i_dim < dim; ++i_dim)
                  for (int j_u = 0; j_u < dof_u; ++j_u)
                     for (int j_dim = 0; j_dim < dim; ++j_dim)
                        for (int n=0; n<dim; ++n)
                           for (int l=0; l<dim; ++l)
                           {
                              ref(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                                 dJ * (mu_v * F(i_dim, l) - pres * FinvT(i_dim,l)) *
                                 FinvT(j_dim,n) * DS_u(i_u,l) * DS_u(j_u, n) *
                                 ip.weight * Tr.Weight();
                              if (j_dim == i_dim && n==l)
                              {
                                 ref(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                                    dJ * mu_v * DS_u(i_u, l) * DS_u(j_u,n) *
                                    ip.weight * Tr.Weight();
                              }
                              ref(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                                 dJ * pres * FinvT(i_dim, n) *
                                 FinvT(j_dim,l) * DS_u(i_u,l) * DS_u(j_u,n) *
                                 ip.weight * Tr.Weight();
                           }
         }
         REQUIRE(Same(m00, ref));
      }
   }
}
