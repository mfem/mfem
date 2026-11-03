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
#include "catch.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace blocknonlinearform
{
real_t rf0(const Vector &coords)
{
   real_t x = coords(0);
   real_t y = coords(1);
   real_t z = coords(2);

   real_t p = std::sqrt(x * x + y * y + z * z);
   real_t rez = 0.0;
   if (p < 1.0)
   {
      rez = 1.0;
   }
   return rez;
}

real_t uf0(const Vector &coords)
{
   real_t x = coords(0);
   real_t y = coords(1);
   real_t z = coords(2);

   real_t p = std::sqrt(x * x + y * y + z * z);
   real_t rez = 0.0;
   if (p < 1.5)
   {
      rez = 1;
   }
   return rez;
}

class CExample : public BlockNonlinearFormIntegrator
{
private:
public:
   CExample() {}
   ~CExample() override {}

   real_t GetElementEnergy(const Array<const FiniteElement *> &el,
                           ElementTransformation &trans,
                           const Array<const Vector *> &elfun) override
   {
      real_t energy = 0;
      int dof_u = el[0]->GetDof();
      int dof_r = el[1]->GetDof();

      const IntegrationRule *ir = NULL;
      int order = el[0]->GetOrder() + el[1]->GetOrder() + 1;
      ir = &IntRules.Get(el[0]->GetGeomType(), order);
      Vector shaperr(dof_r); // densities
      Vector shaperu(dof_u); // prime field

      real_t w;
      real_t c1, c2;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         w = ip.weight * w;
         el[0]->CalcPhysShape(trans, shaperu);
         el[1]->CalcPhysShape(trans, shaperr);

         c1 = shaperr * (*elfun[1]);
         c2 = shaperu * (*elfun[0]);

         energy = energy + w * c1 * c2;
      }

      return energy;
   }
};

// GradientIntegrator implemented as a BlockNonlinearFormIntegrator: reads only
// space 0 (scalar u) and writes only space 1 (vector v), r = (grad u, v)
class MixedExample : public BlockNonlinearFormIntegrator
{
private:
   GradientIntegrator grad;
   DenseMatrix elmat;
public:
   void AssembleElementVector(const Array<const FiniteElement *> &el,
                              ElementTransformation &trans,
                              const Array<const Vector *> &elfun,
                              const Array<Vector *> &elvec) override
   {
      if (elvec[0]) { elvec[0]->SetSize(0); }
      grad.AssembleElementMatrix2(*el[0], *el[1], trans, elmat);
      elvec[1]->SetSize(elmat.Height());
      elmat.Mult(*elfun[0], *elvec[1]);
   }
};

TEST_CASE("BlockNonlinearForm Array of Vectors Mult",
          "[BlockNonlinearForm], [Mult]")
{
   const bool nonconforming = GENERATE(false, true);
   const bool with_bc = GENERATE(false, true);
   CAPTURE(nonconforming, with_bc);

   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true);
   if (nonconforming)
   {
      mesh.EnsureNCMesh();
      Array<int> refs({0, 5, 10});
      mesh.GeneralRefinement(refs);
   }
   const int dim = mesh.Dimension();

   H1_FECollection ufec(2, dim);
   H1_FECollection vfec(1, dim);
   FiniteElementSpace ufes(&mesh, &ufec);
   FiniteElementSpace vfes(&mesh, &vfec, dim);

   Array<FiniteElementSpace *> fes({&ufes, &vfes});
   BlockNonlinearForm form(fes);
   form.AddDomainIntegrator(new MixedExample);

   // Essential b.c. only on the output space
   Array<int> no_bdr(mesh.bdr_attributes.Max());
   no_bdr = 0;
   Array<int> ess_bdr(no_bdr);
   ess_bdr[0] = 1;
   Array<Array<int> *> bdr({&no_bdr, with_bc ? &ess_bdr : &no_bdr});
   Array<Vector *> rhs(2);
   rhs = nullptr;
   form.SetEssentialBC(bdr, rhs);

   Array<int> ess_tdofs;
   vfes.GetEssentialTrueDofs(*bdr[1], ess_tdofs);

   GridFunction u_gf(&ufes);
   FunctionCoefficient u_coeff([](const Vector &x)
   {
      return 1.0 + x(0) * x(1) + x(0) * x(0);
   });
   u_gf.ProjectCoefficient(u_coeff);
   u_gf.SetTrueVector();

   // Reference: the full block Mult, with a zero dummy input for space 1
   BlockVector x_full(form.GetBlockTrueOffsets());
   BlockVector y_full(form.GetBlockTrueOffsets());
   x_full.GetBlock(0) = u_gf.GetTrueVector();
   x_full.GetBlock(1) = 0.0;
   form.Mult(x_full, y_full);
   const Vector &ref_y = y_full.GetBlock(1);
   REQUIRE(ref_y.Normlinf() > 0.0);

   Vector r(vfes.GetTrueVSize());

   Array<const Vector *> x(2);
   x[0] = &u_gf.GetTrueVector();
   x[1] = nullptr;
   Array<Vector *> y(2);
   y[0] = nullptr;
   y[1] = &r;
   form.Mult(x, y);

   for (int d : ess_tdofs)
   {
      REQUIRE(r(d) == 0.0);
   }
   r -= ref_y;
   REQUIRE(r.Normlinf() == MFEM_Approx(0.0));
}

#ifdef MFEM_USE_MPI
TEST_CASE("ParBlockNonlinearForm",
          "[Parallel], [ParBlockNonlinearForm], [GetEnergy]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   FunctionCoefficient u0_coef(uf0);
   FunctionCoefficient r0_coef(rf0);

   for (int type = (int) Element::TETRAHEDRON;
        type <= (int) Element::WEDGE;
        type++)
   {
      int n = 4;
      Mesh mesh = Mesh::MakeCartesian3D(
                     n, n, n, (Element::Type) type, 2.0, 2.0, 2.0);
      int dim = mesh.Dimension();
      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();
      pmesh->UniformRefinement();
      int uorder = 3;
      int rorder = 2;
      H1_FECollection ufec(uorder, dim);
      H1_FECollection rfec(rorder, dim);

      ParFiniteElementSpace ufes(pmesh, &ufec);
      ParFiniteElementSpace rfes(pmesh, &rfec);

      Array<ParFiniteElementSpace *> fes(2);
      fes[0] = &ufes;
      fes[1] = &rfes;

      Array<int> block_trueOffsets(3);
      block_trueOffsets[0] = 0;
      block_trueOffsets[1] = ufes.TrueVSize();
      block_trueOffsets[2] = rfes.TrueVSize();
      block_trueOffsets.PartialSum();

      ParGridFunction u_gf(&ufes);
      ParGridFunction r_gf(&rfes);

      FunctionCoefficient u0_coeff(uf0);
      FunctionCoefficient r0_coeff(rf0);

      u_gf.ProjectCoefficient(u0_coeff);
      r_gf.ProjectCoefficient(r0_coeff);

      BlockVector x(block_trueOffsets);

      u_gf.GetTrueDofs(x.GetBlock(0));
      r_gf.GetTrueDofs(x.GetBlock(1));

      ParBlockNonlinearForm *nf = new ParBlockNonlinearForm(fes);
      nf->AddDomainIntegrator(new CExample());

      // Compute the energy: integral over 1/8 sphere = Pi*1*1*1/6
      real_t A4 = nf->GetEnergy(x);

      mfem::out << "Rank " << my_rank
                << ": ParBlockNonlinearForm::GetEnergy = " << A4
                << ", expected = " << M_PI / 6.0
                << ", diff = " << (A4 - M_PI / 6.0)
                << std::endl;

      REQUIRE(fabs(A4 - M_PI / 6.0) < 1e-2);

      delete nf;
      delete pmesh;
   }
}
#endif

} // namespace blocknonlinearform
