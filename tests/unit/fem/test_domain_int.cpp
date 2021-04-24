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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

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

enum MeshType
{
   SEGMENT = 0,
   QUADRILATERAL = 1,
   TRIANGLE2A = 2,
   TRIANGLE2B = 3,
   TRIANGLE2C = 4,
   TRIANGLE4 = 5,
   MIXED2D = 6,
   HEXAHEDRON = 7,
   HEXAHEDRON2A = 8,
   HEXAHEDRON2B = 9,
   HEXAHEDRON2C = 10,
   HEXAHEDRON2D = 11,
   WEDGE2 = 12,
   TETRAHEDRA = 13,
   WEDGE4 = 14,
   MIXED3D6 = 15,
   MIXED3D8 = 16,
   PYRAMID = 17
};

Mesh * GetMesh(MeshType type);

TEST_CASE("Domain Integration (Scalar Field)",
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[GridFunction]"
          "[LinearForm]")
{
   int order = 2;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::PYRAMID; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt);
      int  dim = mesh->Dimension();
      mesh->UniformRefinement();

      ConstantCoefficient oneCoef(1.0);

      for (int ft = (int)FEType::H1_FEC; ft <= (int)FEType::L2I_FEC; ft++)
      {
         if (ft == (int)FEType::ND_FEC || ft == (int)FEType::RT_FEC)
         { continue; }

         SECTION("Integrating field type " + std::to_string(ft) +
                 " on mesh type " + std::to_string(mt))
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
                  fec = new L2_FECollection(order-1, dim, BasisType::GaussLegendre,
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
   int order = 3;

   for (int mt = (int)MeshType::SEGMENT;
        mt <= (int)MeshType::MIXED3D8; mt++)
   {
      Mesh *mesh = GetMesh((MeshType)mt);
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

      delete mesh;
   }
}

#ifdef MFEM_USE_MPI
#
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
      Mesh *mesh = GetMesh((MeshType)mt);
      int  dim = mesh->Dimension();
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
               fec = new L2_FECollection(order-1, dim, BasisType::GaussLegendre,
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
      Mesh *mesh = GetMesh((MeshType)mt);
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

#endif // MFEM_USE_MPI

Mesh * GetMesh(MeshType type)
{
   Mesh * mesh = NULL;
   double c[3];
   int    v[8];

   switch (type)
   {
      case SEGMENT:
         mesh = new Mesh(1, 2, 1);
         c[0] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_;
         mesh->AddVertex(c);
         v[0] = 0; v[1] = 1;
         mesh->AddSegment(v);
         {
            Element * el = mesh->NewElement(Geometry::POINT);
            el->SetAttribute(1);
            el->SetVertices(&v[0]);
            mesh->AddBdrElement(el);
         }
         {
            Element * el = mesh->NewElement(Geometry::POINT);
            el->SetAttribute(2);
            el->SetVertices(&v[1]);
            mesh->AddBdrElement(el);
         }
         break;
      case QUADRILATERAL:
         mesh = new Mesh(2, 4, 1);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         mesh->AddQuad(v);
         break;
      case TRIANGLE2A:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 0;
         mesh->AddTri(v);
         break;
      case TRIANGLE2B:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 1; v[1] = 2; v[2] = 0;
         mesh->AddTri(v);
         v[0] = 3; v[1] = 0; v[2] = 2;
         mesh->AddTri(v);
         break;
      case TRIANGLE2C:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 2; v[1] = 0; v[2] = 1;
         mesh->AddTri(v);
         v[0] = 0; v[1] = 2; v[2] = 3;
         mesh->AddTri(v);
         break;
      case TRIANGLE4:
         mesh = new Mesh(2, 5, 4);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 1; v[1] = 2; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 3; v[1] = 0; v[2] = 4;
         mesh->AddTri(v);
         break;
      case MIXED2D:
         mesh = new Mesh(2, 6, 4);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.5 * b_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * b_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 5; v[3] = 4;
         mesh->AddQuad(v);
         v[0] = 1; v[1] = 2; v[2] = 5;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 4; v[3] = 5;
         mesh->AddQuad(v);
         v[0] = 3; v[1] = 0; v[2] = 4;
         mesh->AddTri(v);
         break;
      case HEXAHEDRON:
         mesh = new Mesh(3, 8, 1);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         v[4] = 4; v[5] = 5; v[6] = 6; v[7] = 7;
         mesh->AddHex(v);
         break;
      case HEXAHEDRON2A:
      case HEXAHEDRON2B:
      case HEXAHEDRON2C:
      case HEXAHEDRON2D:
         mesh = new Mesh(3, 12, 2);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 5; v[2] = 11; v[3] = 6;
         v[4] = 1; v[5] = 4; v[6] = 10; v[7] = 7;
         mesh->AddHex(v);

         switch (type)
         {
            case HEXAHEDRON2A: // Face Orientation 1
               v[0] = 4; v[1] = 10; v[2] = 7; v[3] = 1;
               v[4] = 3; v[5] = 9; v[6] = 8; v[7] = 2;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2B: // Face Orientation 3
               v[0] = 10; v[1] = 7; v[2] = 1; v[3] = 4;
               v[4] = 9; v[5] = 8; v[6] = 2; v[7] = 3;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2C: // Face Orientation 5
               v[0] = 7; v[1] = 1; v[2] = 4; v[3] = 10;
               v[4] = 8; v[5] = 2; v[6] = 3; v[7] = 9;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2D: // Face Orientation 7
               v[0] = 1; v[1] = 4; v[2] = 10; v[3] = 7;
               v[4] = 2; v[5] = 3; v[6] = 9; v[7] = 8;
               mesh->AddHex(v);
               break;
            default:
               // Cannot happen
               break;
         }
         break;
      case WEDGE2:
         mesh = new Mesh(3, 8, 2);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 4; v[4] = 5; v[5] = 6;
         mesh->AddWedge(v);
         v[0] = 0; v[1] = 2; v[2] = 3; v[3] = 4; v[4] = 6; v[5] = 7;
         mesh->AddWedge(v);
         break;
      case TETRAHEDRA:
         mesh = new Mesh(3, 8, 5);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 2; v[2] = 7; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 6; v[1] = 7; v[2] = 2; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 4; v[1] = 7; v[2] = 5; v[3] = 0;
         mesh->AddTet(v);
         v[0] = 1; v[1] = 0; v[2] = 5; v[3] = 2;
         mesh->AddTet(v);
         v[0] = 3; v[1] = 7; v[2] = 0; v[3] = 2;
         mesh->AddTet(v);
         break;
      case WEDGE4:
         mesh = new Mesh(3, 10, 4);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 4; v[3] = 5; v[4] = 6; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 1; v[1] = 2; v[2] = 4; v[3] = 6; v[4] = 7; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 2; v[1] = 3; v[2] = 4; v[3] = 7; v[4] = 8; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 0; v[2] = 4; v[3] = 8; v[4] = 5; v[5] = 9;
         mesh->AddWedge(v);
         break;
      case MIXED3D6:
         mesh = new Mesh(3, 12, 6);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * c_; c[1] = 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * c_; c[1] = 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * c_; c[1] = b_ - 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * c_; c[1] = b_ - 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         v[4] = 4; v[5] = 5; v[6] = 6; v[7] = 7;
         mesh->AddHex(v);
         v[0] = 0; v[1] = 4; v[2] = 8; v[3] = 1; v[4] = 5; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 1; v[1] = 5; v[2] = 9; v[3] = 2; v[4] = 6; v[5] = 10;
         mesh->AddWedge(v);
         v[0] = 2; v[1] = 6; v[2] = 10; v[3] = 3; v[4] = 7; v[5] = 11;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 7; v[2] = 11; v[3] = 0; v[4] = 4; v[5] = 8;
         mesh->AddWedge(v);
         v[0] = 4; v[1] = 5; v[2] = 6; v[3] = 7;
         v[4] = 8; v[5] = 9; v[6] = 10; v[7] = 11;
         mesh->AddHex(v);
         break;
      case MIXED3D8:
         mesh = new Mesh(3, 10, 8);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);

         c[0] = 0.25 * a_; c[1] = 0.5 * b_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.75 * a_; c[1] = 0.5 * b_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);

         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 3; v[2] = 4; v[3] = 1; v[4] = 2; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 9; v[2] = 4; v[3] = 2; v[4] = 8; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 9; v[1] = 6; v[2] = 4; v[3] = 8; v[4] = 7; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 6; v[1] = 0; v[2] = 4; v[3] = 7; v[4] = 1; v[5] = 5;
         mesh->AddWedge(v);
         v[0] = 0; v[1] = 3; v[2] = 9; v[3] = 4;
         mesh->AddTet(v);
         v[0] = 0; v[1] = 9; v[2] = 6; v[3] = 4;
         mesh->AddTet(v);
         v[0] = 1; v[1] = 7; v[2] = 2; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 8; v[1] = 2; v[2] = 7; v[3] = 5;
         mesh->AddTet(v);
         break;
      case PYRAMID:
         mesh = new Mesh(3, 9, 6);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;
         mesh->AddPyramid(v);
         v[0] = 0; v[1] = 5; v[2] = 6; v[3] = 1; v[4] = 4;
         mesh->AddPyramid(v);
         v[0] = 1; v[1] = 6; v[2] = 7; v[3] = 2; v[4] = 4;
         mesh->AddPyramid(v);
         v[0] = 2; v[1] = 7; v[2] = 8; v[3] = 3; v[4] = 4;
         mesh->AddPyramid(v);
         v[0] = 3; v[1] = 8; v[2] = 5; v[3] = 0; v[4] = 4;
         mesh->AddPyramid(v);
         v[0] = 8; v[1] = 7; v[2] = 6; v[3] = 5; v[4] = 4;
         mesh->AddPyramid(v);
         break;
   }
   mesh->FinalizeTopology();

   return mesh;
}

} // namespace domain_int
