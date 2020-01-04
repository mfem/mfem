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

#include "catch.hpp"
#include "mfem.hpp"

using namespace mfem;

namespace pa_conv
{

int dimension;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{

  if (dimension == 2)
  {
    v(0) = sqrt(2./3.); v(1) = sqrt(1./3.);
  }

  if (dimension == 3)
  {
    v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
  }

}


//Basic unit test for convection
TEST_CASE("conv")
{

 for (dimension = 2; dimension < 4; ++dimension)
 {

     for (int imesh = 0; imesh<2; ++imesh)
     {

        const char *mesh_file;
        if(dimension == 2) {

          switch (imesh)
          {
          case 0: mesh_file = "../../data/periodic-square.mesh"; break;
          case 1: mesh_file = "../../data/amr-quad.mesh"; break;
          }
        }

        if(dimension == 3) {
          switch (imesh)
          {
          case 0: mesh_file = "../../data/periodic-cube.mesh"; break;
          case 1: mesh_file = "../../data/amr-hex.mesh"; break;
          }
        }

        Mesh *mesh = new Mesh(mesh_file, 1, 1);
        for(int order = 1; order < 5; ++order) {

          H1_FECollection *fec = new H1_FECollection(order, dimension);
          FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

          BilinearForm k(fespace);
          BilinearForm pak(fespace); //Partial assembly version of k

          VectorFunctionCoefficient velocity(dimension, velocity_function);

          k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
          pak.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));

          int skip_zeros = 0;
          k.Assemble(skip_zeros);
          k.Finalize(skip_zeros);

          pak.SetAssemblyLevel(AssemblyLevel::PARTIAL);
          pak.Assemble();

          Vector x(k.Size());
          Vector y(k.Size()), y_pa(k.Size());

          for(int i=0; i<x.Size(); ++i) {x(i) = i/10.0;};

          pak.Mult(x,y_pa);
          k.Mult(x,y);

          y_pa -= y;
          double pa_error =- y_pa.Norml2();
          std::cout << "  order: " << order
                    << ", pa error norm: " << pa_error << std::endl;
          REQUIRE(pa_error < 1.e-12);
        }//order loop
     }//mesh loop
 }//dimension loop

}//case

} // namespace pa_conv
