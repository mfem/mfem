// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
#include "catch.hpp"

using namespace mfem;

namespace assemblediagonalpa
{

TEST_CASE("massdiag")
{
   for (int ne = 1; ne < 3; ++ne)
   {
      std::cout << "Testing partial assembly mass diagonal: "
                << ne*ne << " elements." << std::endl;
      for (int order = 1; order < 5; ++order)
      {
         Mesh mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
         FiniteElementCollection *h1_fec = new H1_FECollection(order, 2);
         FiniteElementSpace h1_fespace(&mesh, h1_fec);
         BilinearForm paform(&h1_fespace);
         ConstantCoefficient one(1.0);
         paform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         paform.AddDomainIntegrator(new MassIntegrator(one));
         paform.Assemble();
         Vector pa_diag(h1_fespace.GetVSize());
         paform.AssembleDiagonal(pa_diag);

         BilinearForm assemblyform(&h1_fespace);
         assemblyform.AddDomainIntegrator(new MassIntegrator(one));
         assemblyform.Assemble();
         assemblyform.Finalize();
         Vector assembly_diag(h1_fespace.GetVSize());
         assemblyform.SpMat().GetDiag(assembly_diag);

         assembly_diag -= pa_diag;
         double error = assembly_diag.Norml2();
         std::cout << "    order: " << order << ", error norm: " << error << std::endl;
         REQUIRE(assembly_diag.Norml2() < 1.e-10);

         delete h1_fec;
      }
   }
}

} // namespace assemblediagonalpa
