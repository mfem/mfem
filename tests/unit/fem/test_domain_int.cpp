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

#include "mfem.hpp"
#include "unit_tests.hpp"
#include "common_get_mesh.hpp"

using namespace mfem;
using namespace mfem_test_fem;

namespace domain_int
{

static double a_ = 5.0;
static double b_ = 3.0;
static double c_ = 2.0;

double integral(int dim)
{
   if (dim == 1)
   {
      return a_;
   }
   else if (dim == 2)
   {
      return a_ * b_;
   }
   else
   {
      return a_ * b_ * c_;
   }
}

enum FEType
{
   H1_FEC = 0,
   ND_FEC,
   RT_FEC,
   L2V_FEC,
   L2I_FEC,
};

TEST_CASE("Domain Integration (Scalar Field)",
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[GridFunction]"
          "[LinearForm]")
{
   int order = 2;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
      int  dim = mesh->Dimension();
      mesh->UniformRefinement();

      ConstantCoefficient oneCoef(1.0);

      for (int ft = (int)FEType::H1_FEC; ft <= (int)FEType::L2I_FEC; ft++)
      {
         if (ft == (int)FEType::ND_FEC || ft == (int)FEType::RT_FEC)
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {

            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::H1_FEC:
                  fec = new H1_FECollection(order, dim);
                  break;
               case FEType::L2V_FEC:
                  fec = new L2_FECollection(order-1, dim);
                  break;
               case FEType::L2I_FEC:
                  fec = new L2_FECollection(order-1, dim,
                                            BasisType::GaussLegendre,
                                            FiniteElement::INTEGRAL);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            FiniteElementSpace fespace(mesh, fec);

            GridFunction u(&fespace);
            u.ProjectCoefficient(oneCoef);

            LinearForm b(&fespace);
            b.AddDomainIntegrator(new DomainLFIntegrator(oneCoef));
            b.Assemble();

            double id = b(u);

            if (dim == 1)
            {
               REQUIRE(id == MFEM_Approx( 5.0));
            }
            else if (dim == 2)
            {
               REQUIRE(id == MFEM_Approx(15.0));
            }
            else
            {
               REQUIRE(id == MFEM_Approx(30.0));
            }

            delete fec;
         }
      }

      delete mesh;
   }
}

TEST_CASE("Domain Integration (Vector Field)",
          "[ND_FECollection]"
          "[RT_FECollection]"
          "[GridFunction]"
          "[LinearForm]")
{
   int order = 1;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
      int  dim = mesh->Dimension();
      int sdim = mesh->SpaceDimension();
      mesh->UniformRefinement();

      Vector f1(sdim); f1 = 1.0;
      Vector fx(sdim); fx = 0.0; fx[0] = 1.0;
      Vector fy(sdim); fy = 0.0;
      if (sdim > 1) { fy[1] = 1.0; }
      Vector fz(sdim); fz = 0.0;
      if (sdim > 2) { fz[2] = 1.0; }

      VectorConstantCoefficient f1Coef(f1);
      VectorConstantCoefficient fxCoef(fx);
      VectorConstantCoefficient fyCoef(fy);
      VectorConstantCoefficient fzCoef(fz);

      for (int ft = (int)FEType::ND_FEC; ft <= (int)FEType::RT_FEC; ft++)
      {
         if (dim == 1 && ft == (int)FEType::RT_FEC) { continue; }
         if (mt == (int)MeshType::WEDGE2 || mt == (int)MeshType::WEDGE4 ||
             mt == (int)MeshType::MIXED3D6 || mt == (int)MeshType::MIXED3D8)
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {

            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::ND_FEC:
                  fec = new ND_FECollection(order, dim);
                  break;
               case FEType::RT_FEC:
                  fec = new RT_FECollection(order-1, dim);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            FiniteElementSpace fespace(mesh, fec);

            GridFunction u(&fespace);
            u.ProjectCoefficient(f1Coef);

            LinearForm bx(&fespace);
            LinearForm by(&fespace);
            LinearForm bz(&fespace);
            bx.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fxCoef));
            by.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fyCoef));
            bz.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fzCoef));
            bx.Assemble();
            by.Assemble();
            bz.Assemble();

            double ix = bx(u);
            double iy = by(u);
            double iz = bz(u);

            if (dim == 1)
            {
               REQUIRE(ix == MFEM_Approx( 5.0));
            }
            else if (dim == 2)
            {
               REQUIRE(ix == MFEM_Approx(15.0));
               REQUIRE(iy == MFEM_Approx(15.0));
            }
            else
            {
               REQUIRE(ix == MFEM_Approx(30.0));
               REQUIRE(iy == MFEM_Approx(30.0));
               REQUIRE(iz == MFEM_Approx(30.0));
            }

            delete fec;
         }
      }

      delete mesh;
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("Domain Integration in Parallel (Scalar Field)",
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[ParGridFunction]"
          "[ParLinearForm]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int order = 3;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
      int dim = mesh->Dimension();
      while (mesh->GetNE() < num_procs)
      {
         mesh->UniformRefinement();
      }
      ParMesh pmesh(MPI_COMM_WORLD, *mesh);
      delete mesh;

      ConstantCoefficient oneCoef(1.0);

      for (int ft = (int)FEType::H1_FEC; ft <= (int)FEType::L2I_FEC; ft++)
      {
         if (ft == (int)FEType::ND_FEC || ft == (int)FEType::RT_FEC)
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {
            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::H1_FEC:
                  fec = new H1_FECollection(order, dim);
                  break;
               case FEType::L2V_FEC:
                  fec = new L2_FECollection(order-1, dim);
                  break;
               case FEType::L2I_FEC:
                  fec = new L2_FECollection(order-1, dim,
                                            BasisType::GaussLegendre,
                                            FiniteElement::INTEGRAL);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            ParFiniteElementSpace fespace(&pmesh, fec);

            ParGridFunction u(&fespace);
            u.ProjectCoefficient(oneCoef);

            ParLinearForm b(&fespace);
            b.AddDomainIntegrator(new DomainLFIntegrator(oneCoef));
            b.Assemble();

            double id = b(u);

            if (dim == 1)
            {
               REQUIRE(id == MFEM_Approx( 5.0));
            }
            else if (dim == 2)
            {
               REQUIRE(id == MFEM_Approx(15.0));
            }
            else
            {
               REQUIRE(id == MFEM_Approx(30.0));
            }

            delete fec;
         }
      }
   }
}

TEST_CASE("Domain Integration in Parallel (Vector Field)",
          "[ParGridFunction]"
          "[ParLinearForm]"
          "[Parallel]")
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   int order = 3;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt, a_, b_, c_);
      int  dim = mesh->Dimension();
      int sdim = mesh->SpaceDimension();
      while (mesh->GetNE() < num_procs)
      {
         mesh->UniformRefinement();
      }
      ParMesh pmesh(MPI_COMM_WORLD, *mesh);
      delete mesh;

      Vector f1(sdim); f1 = 1.0;
      Vector fx(sdim); fx = 0.0; fx[0] = 1.0;
      Vector fy(sdim); fy = 0.0;
      if (sdim > 1) { fy[1] = 1.0; }
      Vector fz(sdim); fz = 0.0;
      if (sdim > 2) { fz[2] = 1.0; }

      VectorConstantCoefficient f1Coef(f1);
      VectorConstantCoefficient fxCoef(fx);
      VectorConstantCoefficient fyCoef(fy);
      VectorConstantCoefficient fzCoef(fz);

      for (int ft = (int)FEType::ND_FEC; ft <= (int)FEType::RT_FEC; ft++)
      {
         if (dim == 1 && ft == (int)FEType::RT_FEC) { continue; }
         if (mt == (int)MeshType::WEDGE2 || mt == (int)MeshType::WEDGE4 ||
             mt == (int)MeshType::MIXED3D6 || mt == (int)MeshType::MIXED3D8)
         { continue; }

         SECTION("Integral of field " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt) )
         {
            FiniteElementCollection *fec = NULL;
            switch ((FEType)ft)
            {
               case FEType::ND_FEC:
                  fec = new ND_FECollection(order, dim);
                  break;
               case FEType::RT_FEC:
                  fec = new RT_FECollection(order-1, dim);
                  break;
               default:
                  MFEM_ABORT("Invalid vector FE type");
            }
            ParFiniteElementSpace fespace(&pmesh, fec);

            ParGridFunction u(&fespace);
            u.ProjectCoefficient(f1Coef);

            ParLinearForm bx(&fespace);
            ParLinearForm by(&fespace);
            ParLinearForm bz(&fespace);
            bx.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fxCoef));
            by.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fyCoef));
            bz.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fzCoef));
            bx.Assemble();
            by.Assemble();
            bz.Assemble();

            double ix = bx(u);
            double iy = by(u);
            double iz = bz(u);

            if (dim == 1)
            {
               REQUIRE(ix == MFEM_Approx( 5.0));
            }
            else if (dim == 2)
            {
               REQUIRE(ix == MFEM_Approx(15.0));
               REQUIRE(iy == MFEM_Approx(15.0));
            }
            else
            {
               REQUIRE(ix == MFEM_Approx(30.0));
               REQUIRE(iy == MFEM_Approx(30.0));
               REQUIRE(iz == MFEM_Approx(30.0));
            }

            delete fec;
         }
      }
   }
}

#endif // MFEM_USE_MPI

} // namespace domain_int
