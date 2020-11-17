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
#include <fstream>
#include <sstream>

using namespace mfem;

/**
 * \file test_linear_fes
 *
 * Some simple tests to confirm that linear finite element collections
 * and H1_FECollections of order 1 have the correct number of dofs.
 * Also optionally outputs the mesh and grid functions in mfem and vtk formats
 */

namespace
{
static const bool dumpMeshes = true;  // will dump mesh files when true
}

/// Generate a name for the file
/// using the mesh's element type, the fec collection and the file extension
template<typename FEColType>
std::string generateMeshFilename(FEColType& fec, Mesh& mesh,
                                 const std::string& prefix = "",
                                 const std::string& fileExt = "")
{
   std::stringstream sstr;

   REQUIRE ( mesh.GetNE() == 1 );
   int geom = mesh.GetElementBaseGeometry(0);

   sstr << prefix
        << Geometry::Name[geom]<< "_" << fec.Name()
        << fileExt;

   return sstr.str();
}



/// Tests a scalar and vector grid function on the fes defined by fec and mesh
template<typename FEColType>
void testGridFunctions(FEColType& fec, Mesh& mesh, int expScalarDofs)
{
   // Note: we assume the mesh has a single element
   REQUIRE( mesh.GetNE() == 1);
   const int eltId = 0;
   const int geom = mesh.GetElementBaseGeometry(eltId);

   // Create a scalar grid function on this fec and test setting/getting values
   FiniteElementSpace scalarFes(&mesh, &fec);
   GridFunction scalarGF(&scalarFes);
   {
      GridFunction& gf = scalarGF;
      REQUIRE( gf.FESpace()->GetVDim() == 1 );
      REQUIRE( gf.Size() == expScalarDofs );
      gf = 0.;

      // Set the values of the dofs to non-zero values
      mfem::Array<int> dofs;

      gf.FESpace()->GetElementDofs(eltId,dofs);
      REQUIRE( dofs.Size() > 0 );

      for (int i = 0; i < dofs.Size(); ++i)
      {
         gf[ dofs[i] ] = (1. + i)/(1. + dofs.Size());
      }

      // Test access to the dof values by getting the value at the elt midpoint
      double centerValue = gf.GetValue( eltId, Geometries.GetCenter(geom) );
      REQUIRE( centerValue > 0. );
      REQUIRE( centerValue < 1. );
   }

   // Create a vector grid function on this fec and test setting/getting values
   int vdim = mesh.SpaceDimension();
   FiniteElementSpace vectorFes(&mesh, &fec, vdim);
   GridFunction vectorGF(&vectorFes);
   {
      GridFunction& gf = vectorGF;
      REQUIRE( gf.FESpace()->GetVDim() == vdim );
      REQUIRE( gf.Size() == expScalarDofs * vdim );
      gf = 0.;

      // Test setting some vector values
      mfem::Array<int> vdofs;
      gf.FESpace()->GetElementVDofs(eltId,vdofs);
      REQUIRE( vdofs.Size() > 0 );

      // If we only expect dofs at one point, set non-zero as in the scalar case
      if (expScalarDofs == 1)
      {
         for (int i = 0; i < vdofs.Size(); ++i)
         {
            gf[ vdofs[i] ] = (1. + i)/(1. + vdofs.Size());
         }

         // Check that the vector is not zero at the element center
         Vector centerVec;
         gf.GetVectorValue( eltId, Geometries.GetCenter(geom), centerVec);
         REQUIRE( centerVec.Norml2() > 0. );
      }
      // Otherwise create a vector from each vertex to the element center
      else
      {
         // Get position of element center
         Vector ctr;
         IsoparametricTransformation tr;
         mesh.GetElementTransformation(eltId, &tr);
         tr.Transform(Geometries.GetCenter(geom), ctr);

         // Get each vertex position and subtract from ctr
         Array<int> vertIndices;
         mesh.GetElementVertices(eltId, vertIndices);
         const int nv = vertIndices.Size();
         for (int i=0; i < nv; ++i)
         {
            double* vertPos = mesh.GetVertex(vertIndices[i]);

            for (int j=0; j < vdim; ++j)
            {
               gf[ vdofs[j*nv + i] ] = ctr[j] - vertPos[j];
            }
         }

         // Check that none of the vectors are zero at vertices
         Vector vec;
         DenseMatrix dm;
         gf.GetVectorValues(tr, *Geometries.GetVertices(geom), dm);
         for (int i=0; i< nv; ++i)
         {
            dm.GetColumnReference(i, vec);
            REQUIRE( vec.Norml2() > 0. );
         }

         // But, the vectors should cancel each other out at the midpoint
         gf.GetVectorValue( eltId, Geometries.GetCenter(geom), vec);
         REQUIRE( vec.Norml2() == MFEM_Approx(0.0) );
      }

   }

   if (dumpMeshes)
   {
      const std::string prefix_path = "output_meshes/";

      // Save mfem meshes using VisitDataCollection
      VisItDataCollection dc(generateMeshFilename(fec, mesh), & mesh);
      dc.SetPrefixPath(prefix_path);
      dc.RegisterField("scalar_gf", &scalarGF);
      dc.RegisterField("vector_gf", &vectorGF);
      dc.Save();

      // Save meshes and grid functions in VTK format
      std::string fname = generateMeshFilename(fec, mesh, prefix_path, ".vtk");
      std::fstream vtkFs( fname.c_str(), std::ios::out);

      const int ref = 0;
      mesh.PrintVTK( vtkFs, ref);
      scalarGF.SaveVTK( vtkFs, "scalar_gf", ref);
      vectorGF.SaveVTK( vtkFs, "vector_gf", ref);
   }
}


TEST_CASE("Point mesh with linear grid function",
          "[LinearFECollection]"
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[FiniteElementSpace]"
          "[GridFunction]")
{
   // Create a simple one element point mesh
   int dim = 0, nv = 1, ne = 1, nb = 0, sdim = 2, order =1, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);

   mesh.AddVertex(Vertex(0.,0.)());

   int idx[1] = {0};
   mesh.AddElement(new mfem::Point(idx, attrib));

   REQUIRE( mesh.GetNV() == nv);
   REQUIRE( mesh.GetNE() == ne);
   REQUIRE( mesh.Dimension() == dim);
   REQUIRE( mesh.SpaceDimension() == sdim);

   SECTION("GridFunction with LinearFECollection")
   {
      LinearFECollection fec;
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with order 1 H1_FECollection")
   {
      H1_FECollection fec(order,dim);
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with L2_FECollection")
   {
      L2_FECollection fec(0,dim);
      testGridFunctions(fec, mesh, 1);
   }
}


TEST_CASE("Segment mesh with linear grid function",
          "[LinearFECollection]"
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[FiniteElementSpace]"
          "[GridFunction]")
{
   // Create a simple one element segment mesh
   int dim = 1, nv = 2, ne = 1, nb = 0, sdim = 2, order =1, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);

   mesh.AddVertex(Vertex(0.,0.)());
   mesh.AddVertex(Vertex(1.,0.)());

   int idx[2] = {0,1};
   mesh.AddElement(new Segment(idx, attrib));

   REQUIRE( mesh.GetNV() == nv);
   REQUIRE( mesh.GetNE() == ne);
   REQUIRE( mesh.Dimension() == dim);
   REQUIRE( mesh.SpaceDimension() == sdim);

   SECTION("GridFunction with LinearFECollection")
   {
      LinearFECollection fec;
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with order 1 H1_FECollection")
   {
      H1_FECollection fec(order,dim);
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with L2_FECollection")
   {
      L2_FECollection fec(0,dim);
      testGridFunctions(fec, mesh, 1);
   }
}

TEST_CASE("Triangle mesh with linear grid function",
          "[LinearFECollection]"
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[FiniteElementSpace]"
          "[GridFunction]")
{
   // Create a simple one element triangle mesh
   int dim = 2, nv = 3, ne = 1, nb = 0, sdim = 2, order =1, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);

   mesh.AddVertex(Vertex(0.,0.)());
   mesh.AddVertex(Vertex(1.,0.)());
   mesh.AddVertex(Vertex(1.,1.)());

   int idx[3] = {0,1,2};
   mesh.AddElement(new Triangle(idx, attrib));

   REQUIRE( mesh.GetNV() == nv);
   REQUIRE( mesh.GetNE() == ne);
   REQUIRE( mesh.Dimension() == dim);
   REQUIRE( mesh.SpaceDimension() == sdim);

   SECTION("GridFunction with LinearFECollection")
   {
      LinearFECollection fec;
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with order 1 H1_FECollection")
   {
      H1_FECollection fec(order,dim);
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with L2_FECollection")
   {
      L2_FECollection fec(0,dim);
      testGridFunctions(fec, mesh, 1);
   }
}

TEST_CASE("Quad mesh with linear grid function",
          "[LinearFECollection]"
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[FiniteElementSpace]"
          "[GridFunction]")
{
   // Create a simple one element quad mesh
   int dim = 2, nv = 4, ne = 1, nb = 0, sdim = 2, order =1, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);

   mesh.AddVertex(Vertex(0.,0.)());
   mesh.AddVertex(Vertex(1.,0.)());
   mesh.AddVertex(Vertex(1.,1.)());
   mesh.AddVertex(Vertex(0.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Quadrilateral(idx, attrib));

   REQUIRE( mesh.GetNV() == nv);
   REQUIRE( mesh.GetNE() == ne);
   REQUIRE( mesh.Dimension() == dim);
   REQUIRE( mesh.SpaceDimension() == sdim);

   SECTION("GridFunction with LinearFECollection")
   {
      LinearFECollection fec;
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with order 1 H1_FECollection")
   {
      H1_FECollection fec(order,dim);
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with L2_FECollection")
   {
      L2_FECollection fec(0,dim);
      testGridFunctions(fec, mesh, 1);
   }

}


TEST_CASE("Tet mesh with linear grid function",
          "[LinearFECollection]"
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[FiniteElementSpace]"
          "[GridFunction]")
{
   // Create a simple one element tet mesh
   int dim = 3, nv = 4, ne = 1, nb = 0, sdim = 3, order =1, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);

   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(1.,1.,0.)());
   mesh.AddVertex(Vertex(1.,1.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Tetrahedron(idx, attrib));

   REQUIRE( mesh.GetNV() == nv);
   REQUIRE( mesh.GetNE() == ne);
   REQUIRE( mesh.Dimension() == dim);
   REQUIRE( mesh.SpaceDimension() == sdim);

   SECTION("GridFunction with LinearFECollection")
   {
      LinearFECollection fec;
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with order 1 H1_FECollection")
   {
      H1_FECollection fec(order,dim);
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with L2_FECollection")
   {
      L2_FECollection fec(0,dim);
      testGridFunctions(fec, mesh, 1);
   }

}

TEST_CASE("Hex mesh with linear grid function",
          "[LinearFECollection]"
          "[H1_FECollection]"
          "[L2_FECollection]"
          "[FiniteElementSpace]"
          "[GridFunction]")
{
   // Create a simple one element hex mesh
   int dim = 3, nv = 8, ne = 1, nb = 0, sdim = 3, order =1, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);

   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(1.,1.,0.)());
   mesh.AddVertex(Vertex(0.,1.,0.)());
   mesh.AddVertex(Vertex(0.,0.,1.)());
   mesh.AddVertex(Vertex(1.,0.,1.)());
   mesh.AddVertex(Vertex(1.,1.,1.)());
   mesh.AddVertex(Vertex(0.,1.,1.)());

   int idx[8] = {0,1,2,3,4,5,6,7};
   mesh.AddElement(new Hexahedron(idx, attrib));

   REQUIRE( mesh.GetNV() == nv);
   REQUIRE( mesh.GetNE() == ne);
   REQUIRE( mesh.Dimension() == dim);
   REQUIRE( mesh.SpaceDimension() == sdim);

   SECTION("GridFunction with LinearFECollection")
   {
      LinearFECollection fec;
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with order 1 H1_FECollection")
   {
      H1_FECollection fec(order,dim);
      testGridFunctions(fec, mesh, nv);
   }

   SECTION("GridFunction with L2_FECollection")
   {
      L2_FECollection fec(0,dim);
      testGridFunctions(fec, mesh, 1);
   }

}
