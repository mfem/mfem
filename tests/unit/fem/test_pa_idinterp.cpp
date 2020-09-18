// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

using namespace mfem;

double compare_pa_id_assembly(int dim, int num_elements, int order,
                              bool transpose)
{
   Mesh * mesh;
   if (num_elements == 0)
   {
      if (dim == 2)
      {
         mesh = new Mesh("../../data/star.mesh", order);
      }
      else
      {
         mesh = new Mesh("../../data/beam-hex.mesh", order);
      }
   }
   else
   {
      if (dim == 2)
      {
         mesh = new Mesh(num_elements, num_elements, Element::QUADRILATERAL, true);
      }
      else
      {
         mesh = new Mesh(num_elements, num_elements, num_elements,
                         Element::HEXAHEDRON, true);
      }
   }
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);
   FiniteElementCollection *nd_fec = new ND_FECollection(order, dim);
   FiniteElementSpace h1_fespace(mesh, h1_fec, dim);
   FiniteElementSpace nd_fespace(mesh, nd_fec);

   DiscreteLinearOperator assembled_id(&h1_fespace, &nd_fespace);
   assembled_id.AddDomainInterpolator(new IdentityInterpolator);
   const int skip_zeros = 1;
   assembled_id.Assemble(skip_zeros);
   assembled_id.Finalize(skip_zeros);

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
      assembled_id.MultTranspose(x, assembled_y);
      pa_id.MultTranspose(x, pa_y);
   }
   else
   {
      assembled_id.Mult(x, assembled_y);
      pa_id.Mult(x, pa_y);
   }

   if (false)
   {
      std::cout << "true   \tpa\n";
      for (int i = 0; i < assembled_y.Size(); ++i)
      {
         std::cout << i << " : " << assembled_y(i) << "\t" << pa_y(i) << std::endl;
      }
   }

   pa_y -= assembled_y;
   double error = pa_y.Norml2() / assembled_y.Norml2();
   std::cout << "dim " << dim << " ne " << num_elements << " order "
             << order;
   if (transpose)
   {
      std::cout << " T";
   }
   std::cout << ": error in PA identity: " << error << std::endl;

   delete h1_fec;
   delete nd_fec;
   delete mesh;

   return error;
}

TEST_CASE("PAIdentityInterp", "[PAIdentityInterp]")
{
   for (bool transpose : {false, true})
   {
      for (int dim = 2; dim < 4; ++dim)
      {
         for (int num_elements = 0; num_elements < 5; ++num_elements)
         {
            for (int order = 1; order < 5; ++order)
            {
               double error = compare_pa_id_assembly(dim, num_elements, order, transpose);
               REQUIRE(error < 1.e-14);
            }
         }
      }
   }
}
