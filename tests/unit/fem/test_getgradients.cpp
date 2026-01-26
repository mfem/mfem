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
#include <random>

using namespace mfem;

Mesh MakePerturbedMesh(Geometry::Type geom, int mesh_p)
{
   MFEM_VERIFY(Geometry::POINT < geom && geom < Geometry::NUM_GEOMETRIES,
               "invalid geom: " << geom);
   const int dim = Geometry::Dimension[geom];
   const Element::Type type = Element::TypeFromGeometry(geom);
   Mesh mesh = (dim == 1) ? Mesh::MakeCartesian1D(7) :
               (dim == 2) ? Mesh::MakeCartesian2D(3, 5, type, true) :
               Mesh::MakeCartesian3D(2, 3, 3, type);
   mesh.SetCurvature(mesh_p); // mesh_p <= 0 means no nodes

   real_t h_min = infinity();
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      h_min = std::fmin(h_min, mesh.GetElementSize(i, 1));
   }
   if (mesh_p > 0)
   {
      h_min /= mesh_p;
   }

   Vector pert(mesh.GetNodes() ? mesh.GetNodes()->Size() : mesh.GetNV()*dim);
   pert.Randomize(473'099'612);
   pert -= 0.5_r;
   pert *= 0.25_r*h_min;

   mesh.MoveNodes(pert);

   return mesh;
}

real_t CompareGradients(const GridFunction &f,
                        const IntegrationRule &ir,
                        const Vector &grad_f,
                        QVectorLayout ql = QVectorLayout::byNODES)
{
   real_t max_rel_error = 0;

   const FiniteElementSpace &fes = *f.FESpace();
   const int dim = fes.GetMesh()->Dimension();
   const int NE  = fes.GetNE();
   const int NQ  = ir.GetNPoints();

   MFEM_VERIFY(fes.GetVDim() == 1, "only vdim == 1 is implemented!");

   Vector grad_error(dim), grad;
   DenseMatrix loc_grad;
   grad_f.HostRead();
   for (int i = 0; i < NE; i++)
   {
      f.GetGradients(i, ir, loc_grad); // loc_grad is (dim x NQ)

      // The layout of grad_f is:
      // ql == QVectorLayout::byNODES :   NQ x VDIM x DIM x NE
      // ql == QVectorLayout::byVDIM  : VDIM x  DIM x  NQ x NE

      real_t max_error = 0, max_grad_norm = 0;
      for (int j = 0; j < NQ; j++)
      {
         loc_grad.GetColumnReference(j, grad);
         for (int d = 0; d < dim; d++)
         {
            // vdim is 1
            real_t g = (ql == QVectorLayout::byNODES) ?
                       grad_f[j + NQ*(d + dim*i)] :
                       grad_f[d + dim*(j + NQ*i)];
            grad_error[d] = g - grad(d);
         }
         max_error = std::fmax(max_error, grad_error.Norml2());
         max_grad_norm = std::fmax(max_grad_norm, grad.Norml2());
      }
      real_t rel_error = // element relative error
         (max_grad_norm > 0_r) ? max_error/max_grad_norm :
         (max_error == 0_r) ? 0_r : infinity();
      max_rel_error = std::fmax(max_rel_error, rel_error);
   }

   return max_rel_error;
}

real_t RandReal()
{
   static std::mt19937_64 gen(8'656'127'438'685'088'196);
   static std::uniform_real_distribution<real_t> dis_real(0_r, 1_r); // [0, 1)
   return dis_real(gen);
}

TEST_CASE("GetGradients All Elements", "[GridFunction][GPU]")
{
   auto geom = GENERATE(range(int(Geometry::SEGMENT),
                              int(Geometry::NUM_GEOMETRIES)));
   auto mesh_p = GENERATE(0, 2);
   auto p = GENERATE(1, 3);

   Mesh mesh = MakePerturbedMesh((Geometry::Type)geom, mesh_p);
   H1_FECollection h1_fec(p, Geometry::Dimension[geom]);
   FiniteElementSpace h1_fes(&mesh, &h1_fec);
   GridFunction h1_gf(&h1_fes);
   for (auto &d : h1_gf) { d = RandReal(); }
   Vector grad_h1_gf;
   const IntegrationRule &ir = IntRules.Get(geom, 2*p+1);
   for (auto ql : {QVectorLayout::byNODES, QVectorLayout::byVDIM})
   {
      h1_gf.GetGradients(ir, grad_h1_gf, ql);
      real_t rel_err = CompareGradients(h1_gf, ir, grad_h1_gf, ql);
      CAPTURE(geom, mesh_p, p, ql);
      CHECK(rel_err == MFEM_Approx(0_r));
   }
}
