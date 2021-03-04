// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

namespace gf_project
{

void vector_fn(const Vector& x, Vector& v)
{
   v = 0.0;
   for (int i = 0; i < std::min(v.Size(), x.Size()); i++)
   {
      v(i) = std::sqrt(x(i)*x(i));
   }
}

/* Test ParGridFunction::ProjectDistCoef on an RT space. The averaging on
 * parallel partition boundaries must be done correctly, particularly the
 * ldof signs must be applied in the NC case to get the correct result.
 */
#ifdef MFEM_USE_MPI
TEST_CASE("ProjectDistCoefficient",
          "[Parallel], [ParGridFunction], [NCMesh]")
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int order = GENERATE(1, 2);

   auto eltype = GENERATE(Element::TRIANGLE,
                          Element::QUADRILATERAL,
                          Element::TETRAHEDRON,
                          Element::HEXAHEDRON
                          /*Element::WEDGE*/ /* no ND/RT support yet */);

   bool nc = GENERATE(false, true);

   if (myid == 0)
   {
      mfem::out << "Testing ProjectDiscCoefficient for "
                << "order " << order
                << ", el type " << eltype
                << ", NC " << std::boolalpha << nc
                << std::endl;
   }

   int dim = (eltype <= Element::QUADRILATERAL) ? 2 : 3;

   Mesh *mesh = (dim <= 2) ? new Mesh(3, 3, eltype) : new Mesh(3, 3, 3, eltype);

   mesh->ReorientTetMesh();
   if (nc) { mesh->EnsureNCMesh(true); }

   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   pmesh.ReorientTetMesh();

   VectorFunctionCoefficient vcoef(dim, vector_fn);

   RT_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec, 1, Ordering::byVDIM);

   ParGridFunction gf(&fes);
   gf.ProjectDiscCoefficient(vcoef, GridFunction::ARITHMETIC);

#ifdef MFEM_UNIT_DEBUG_VISUALIZE
   socketstream sol_sock("localhost", 19916);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << pmesh << gf << std::flush;
#endif

   double error = gf.ComputeL2Error(vcoef);
   REQUIRE(error == MFEM_Approx(0.0));

   delete mesh;
}
#endif // MFEM_USE_MPI

} // namespace gf_project
