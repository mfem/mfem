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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

namespace pa_kernels
{

///////////////////////////////////////////////////////////////////////////////
inline Mesh MakeCartesianNonaligned(const int dim, const int ne)
{
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   // Remap vertices so that the mesh is not aligned with axes.
   for (int i=0; i<mesh.GetNV(); ++i)
   {
      real_t *vcrd = mesh.GetVertex(i);
      vcrd[1] += 0.2 * vcrd[0];
      if (dim == 3) { vcrd[2] += 0.3 * vcrd[0]; }
   }

   return mesh;
}

///////////////////////////////////////////////////////////////////////////////
real_t test_nl_convection_pa(int dim)
{
   const int ne = 2;
   Mesh mesh = MakeCartesianNonaligned(dim, ne);

   int order = 2;

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(3);

   NonlinearForm nlf_fa(&fes);
   nlf_fa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_fa.Mult(x, y_fa);

   NonlinearForm nlf_pa(&fes);
   nlf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   nlf_pa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_pa.Setup();
   nlf_pa.Mult(x, y_pa);

   y_fa -= y_pa;
   real_t difference = y_fa.Norml2();

   return difference;
}

TEST_CASE("NL Convection PA", "[PartialAssembly][NonlinearPA][GPU][CHOP]")
{
   dbg("NL Convection PA");
   SECTION("2D")
   {
      REQUIRE(test_nl_convection_pa(2) == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_nl_convection_pa(3) == MFEM_Approx(0.0));
   }
}

} // namespace pa_kernels
