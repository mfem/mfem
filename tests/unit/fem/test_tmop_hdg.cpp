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

namespace tmop_hdg
{

// HDG_TMOP_Integrator drives mesh optimisation with the HDG face energy, so
// that the nodes move to reduce the trace jump rather than to improve a purely
// geometric quality measure. Its energy is a function of the node positions,
// and AssembleFaceVector is that energy's derivative with respect to them --
// which is the thing worth checking, since a wrong derivative moves the mesh
// in a plausible but wrong direction rather than failing.
//
// The integrator asserts dim == 2, so these are two-dimensional only.

struct Fixture
{
   Mesh mesh;
   L2_FECollection u_coll;
   DG_Interface_FECollection t_coll;
   FiniteElementSpace fes_u, fes_t;
   GridFunction u, uhat;
   FaceElementTransformations *Tr;
   const FiniteElement *nfe1, *nfe2;
   Vector nodes;

   Fixture(int order)
      : mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false,
                                   1.0, 1.0)),
        u_coll(order, 2), t_coll(order, 2),
        fes_u(&mesh, &u_coll), fes_t(&mesh, &t_coll),
        u(&fes_u), uhat(&fes_t)
   {
      mesh.EnsureNodes();

      int f = -1;
      for (int i = 0; i < mesh.GetNumFaces(); i++)
      {
         if (mesh.FaceIsInterior(i)) { f = i; break; }
      }
      Tr = mesh.GetFaceElementTransformations(f);

      const FiniteElementSpace *nfes = mesh.GetNodalFESpace();
      nfe1 = nfes->GetFE(Tr->Elem1No);
      nfe2 = nfes->GetFE(Tr->Elem2No);

      // The integrator takes the node positions of both elements, packed one
      // element after the other.
      Vector n1, n2;
      mesh.GetNodes()->GetElementDofValues(Tr->Elem1No, n1);
      mesh.GetNodes()->GetElementDofValues(Tr->Elem2No, n2);
      nodes.SetSize(n1.Size() + n2.Size());
      for (int i = 0; i < n1.Size(); i++) { nodes(i) = n1(i); }
      for (int i = 0; i < n2.Size(); i++) { nodes(n1.Size() + i) = n2(i); }
   }
};

void FillVarying(Vector &v, real_t shift)
{
   for (int i = 0; i < v.Size(); i++)
   {
      v(i) = std::sin(1.7 * i + shift) + 0.5 * std::cos(0.3 * i);
   }
}

} // namespace tmop_hdg

TEST_CASE("HDG_TMOP_Integrator energy vanishes with the trace jump",
          "[TMOP][HDG]")
{
   using namespace tmop_hdg;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Fixture fx(order);
   fx.u = 1.0;
   fx.uhat = 1.0;

   ConstantCoefficient q(2.5);
   HDG_TMOP_Integrator integ(q, fx.u, fx.uhat, 0.5);

   const real_t e = integ.GetFaceEnergy(*fx.nfe1, *fx.nfe2, *fx.Tr, fx.nodes);
   INFO("energy with no jump: " << e);
   REQUIRE(std::abs(e) < 1e-12);
}

TEST_CASE("HDG_TMOP_Integrator energy is positive and scales with td",
          "[TMOP][HDG]")
{
   using namespace tmop_hdg;

   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Fixture fx(order);
   FillVarying(fx.u, 0.0);
   FillVarying(fx.uhat, 1.1);

   ConstantCoefficient q(2.5);
   HDG_TMOP_Integrator integ(q, fx.u, fx.uhat, 0.5);

   const real_t e1 = integ.GetFaceEnergy(*fx.nfe1, *fx.nfe2, *fx.Tr, fx.nodes);
   REQUIRE(e1 > 0.0);

   // tau enters the energy linearly, so doubling td doubles it.
   integ.SetTd(1.0);
   const real_t e2 = integ.GetFaceEnergy(*fx.nfe1, *fx.nfe2, *fx.Tr, fx.nodes);
   INFO("td 0.5 gives " << e1 << ", td 1.0 gives " << e2);
   REQUIRE(e2 == MFEM_Approx(2.0 * e1, 1e-10, 1e-9));
}

TEST_CASE("HDG_TMOP_Integrator: the face vector is the energy's derivative",
          "[TMOP][HDG]")
{
   using namespace tmop_hdg;

   // The gradient with respect to the node positions, checked against a
   // central difference of the energy the same object reports. A wrong
   // gradient here does not fail; it optimises the mesh in the wrong
   // direction.
   const int order = GENERATE(1, 2);
   CAPTURE(order);

   Fixture fx(order);
   FillVarying(fx.u, 0.0);
   FillVarying(fx.uhat, 1.1);

   ConstantCoefficient q(2.5);
   HDG_TMOP_Integrator integ(q, fx.u, fx.uhat, 0.5);

   Vector grad;
   integ.AssembleFaceVector(*fx.nfe1, *fx.nfe2, *fx.Tr, fx.nodes, grad);
   REQUIRE(grad.Size() == fx.nodes.Size());

   const real_t h = std::cbrt(std::numeric_limits<real_t>::epsilon());

   for (int i = 0; i < fx.nodes.Size(); i++)
   {
      Vector xp(fx.nodes), xm(fx.nodes);
      xp(i) += h;
      xm(i) -= h;

      const real_t ep = integ.GetFaceEnergy(*fx.nfe1, *fx.nfe2, *fx.Tr, xp);
      const real_t em = integ.GetFaceEnergy(*fx.nfe1, *fx.nfe2, *fx.Tr, xm);
      const real_t fd = (ep - em) / (2.0 * h);

      CAPTURE(i, grad(i), fd);
      REQUIRE(grad(i) == MFEM_Approx(fd, 1e-6, 1e-5));
   }
}
