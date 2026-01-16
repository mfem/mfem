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

#include "catch.hpp"
#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

real_t compare_pa_id_assembly(int dim, int num_elements, int order,
                              bool transpose)
{
   Mesh mesh;
   if (num_elements == 0)
   {
      if (dim == 2)
      {
         mesh = Mesh::LoadFromFile("../../data/star.mesh", order);
      }
      else
      {
         mesh = Mesh::LoadFromFile("../../data/beam-hex.mesh", order);

         // Transform mesh vertices to test without alignment with coordinate axes.
         for (int i=0; i<mesh.GetNV(); ++i)
         {
            real_t *v = mesh.GetVertex(i);
            const real_t yscale = 1.0 + v[1];
            const real_t zscale = 1.0 + v[2];
            v[0] *= zscale;
            v[1] *= zscale;
            v[2] *= yscale;
         }
      }
   }
   else
   {
      if (dim == 2)
      {
         mesh = Mesh::MakeCartesian2D(num_elements, num_elements, Element::QUADRILATERAL,
                                      true);
      }
      else
      {
         mesh = Mesh::MakeCartesian3D(num_elements, num_elements, num_elements,
                                      Element::HEXAHEDRON);
      }
   }
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);
   FiniteElementCollection *nd_fec = new ND_FECollection(order, dim);
   FiniteElementSpace h1_fespace(&mesh, h1_fec, dim);
   FiniteElementSpace nd_fespace(&mesh, nd_fec);

   DiscreteLinearOperator assembled_id(&h1_fespace, &nd_fespace);
   assembled_id.AddDomainInterpolator(new IdentityInterpolator);
   const int skip_zeros = 1;
   assembled_id.Assemble(skip_zeros);
   assembled_id.Finalize(skip_zeros);
   const SparseMatrix& assembled_id_mat = assembled_id.SpMat();

   DiscreteLinearOperator pa_id(&h1_fespace, &nd_fespace);
   pa_id.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_id.AddDomainInterpolator(new IdentityInterpolator);
   pa_id.Assemble();
   pa_id.Finalize();

   int insize, outsize;
   if (transpose)
   {
      insize = nd_fespace.GetVSize();
      outsize = h1_fespace.GetVSize();
   }
   else
   {
      insize = h1_fespace.GetVSize();
      outsize = nd_fespace.GetVSize();
   }
   Vector x(insize);
   Vector assembled_y(outsize);
   Vector pa_y(outsize);

   x.Randomize();
   if (transpose)
   {
      assembled_id_mat.MultTranspose(x, assembled_y);
      pa_id.MultTranspose(x, pa_y);
   }
   else
   {
      assembled_id.Mult(x, assembled_y);
      pa_id.Mult(x, pa_y);
   }

   pa_y -= assembled_y;
   real_t error = pa_y.Norml2() / assembled_y.Norml2();
   INFO("dim " << dim << " ne " << num_elements << " order " << order
        << (transpose ? " T:" : ":") << " error in PA identity: " << error);

   delete h1_fec;
   delete nd_fec;

   return error;
}

TEST_CASE("PAIdentityInterp", "[GPU]")
{
   auto transpose = GENERATE(true, false);
   auto order = GENERATE(1, 2, 3, 4);
   auto dim = GENERATE(2, 3);
   auto num_elements = GENERATE(0, 1, 2, 3, 4);

   real_t error = compare_pa_id_assembly(dim, num_elements, order, transpose);
   REQUIRE(error == MFEM_Approx(0.0, 1.0e-14));
}
