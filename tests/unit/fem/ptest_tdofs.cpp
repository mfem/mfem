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

#include <fstream>
#include <sstream>
#include <iostream>

/**
 * \file ptest_tdofs
 *
 * Some tests related to parallel finite element spaces
 *
 * Specifically, computes global IDs for the vertices, edges,
 * faces, elements and boundary elements of an mfem ParMesh.
 *
 * \note This test requires MPI and Hypre
 */

// define helper macros to print to cout from a given rank?
#define PTEST_TDOF_USE_DEBUG_PRINT 0

#ifdef MFEM_USE_MPI

namespace
{

#if PTEST_TDOF_USE_DEBUG_PRINT
#define MPRINT_FLUSH                          \
   do {                                       \
      MPI_Barrier(MPI_COMM_WORLD);            \
      std::cout.flush();                      \
       MPI_Barrier(MPI_COMM_WORLD);           \
   } while(0)

#define MPRINT(_rank,_msg)                    \
    do {                                      \
          std::cout << "[" << _rank << "]"    \
                    << _msg                   \
                    << std::endl;             \
    } while(0)

#define MPRINT_MASTER(_rank,_msg)             \
    do {                                      \
        MPRINT_FLUSH;                         \
        if(_rank == 0) { MPRINT(_rank,_msg);} \
        MPRINT_FLUSH;                         \
    } while(0)

#else  // PTEST_TDOF_USE_DEBUG_PRINT
#define MPRINT_FLUSH                do{}while(0)
#define MPRINT(_rank,_msg)          do{}while(0)
#define MPRINT_MASTER(_rank,_msg)   do{}while(0)
#endif //PTEST_TDOF_USE_DEBUG_PRINT

// Define some parameters for the tests
const int NUM_SER_REF_LEVELS = 1;
const int NUM_PAR_REF_LEVELS = 0;
const bool VERBOSE_OUTPUT = false;

/// Helper function to refine the mesh uniformly
inline void refineMesh(mfem::Mesh* mesh, int num_levels)
{
   for (int l = 0; l < num_levels; ++l)
   {
      mesh->UniformRefinement();
   }
}

/// Helper function to collect some stats about the mesh
inline std::string getMeshStats(mfem::Mesh* mesh)
{
   std::stringstream sstr;

   sstr << "Mesh has: "
        << "\n\t"<< mesh->GetNV() << " vertices."
        << "\n\t"<< mesh->GetNEdges() << " edges."
        << "\n\t"<< mesh->GetNFaces() << " faces."
        << "\n\t"<< mesh->GetNE() << " elements."
        << "\n\t"<< mesh->GetNBE() << " boundary elements.";

   return sstr.str();
}

/// Helper to remove encoded sign from a DOF
/// Adapted from protected static function in mfem::FiniteElementSpace
inline int decodeDof(int dof)
{
   return (dof >= 0) ? dof: (-1 - dof);
}

/// Helper class for testing true dofs on parmesh elements
struct TDOFTester
{
   TDOFTester(mfem::ParMesh* mesh, bool verbose)
      : m_mesh(mesh)
      , m_verbose(verbose)
   {
      rank = m_mesh->GetMyRank();

      MPRINT(rank, "Details for parallel mesh:\n" << getMeshStats(m_mesh));
   }

   /// Given an array of TDofs, apply some checks
   /// typeStr parameter is only used for logging
   void checkDOFs(const mfem::Array<int>& arr, const std::string& typeStr)
   {
      MPRINT_MASTER(rank,
                    "List of "<< typeStr << " tdofs in pmesh\n"
                    << "  -----------------------------");

      //const bool isNC = m_mesh->Nonconforming();
      //const int CHECK_VAL = isNC ? -1 : 0;
      const int CHECK_VAL = 0;

      std::stringstream sstr;
      for (int i=0; i< arr.Size(); ++i)
      {
         // Check that the TDOF is non-negative
         CHECK( arr[i] >= CHECK_VAL);

         sstr << "\n\t" << typeStr << " "
              << i << " -> " << arr[i];
      }
      MPRINT(rank, sstr.str());

      MPRINT_MASTER(rank,"-----------------------------");
   }

   /// Generate and test tdofs for the mesh vertices
   void testVertexTDofs()
   {
      const int dim = m_mesh->Dimension();
      const int order = 1;

      const int NV = m_mesh->GetNV();

      MPRINT_MASTER(rank,"Finding vertex tdofs");

      auto* fec = new mfem::H1_FECollection(order, dim);
      mfem::ParFiniteElementSpace fes(m_mesh, fec);
      mfem::Array<int> vert_tdof(NV);
      vert_tdof = UNITIALIZED_VAL;

      mfem::Array<int> idofs;

      std::stringstream sstr;
      for (int v=0; v< NV; ++v)
      {
         fes.GetVertexDofs(v, idofs);
         CHECK( idofs.Size() == 1);

         auto ldof = idofs[0];
         CHECK( ldof >= 0);

         // WARNING -- The following does not account for NCMesh!
         //            Might need to check against GetLocalTDofNumber()
         auto tdof = fes.GetGlobalTDofNumber(ldof);
         CHECK( tdof >= 0);

         vert_tdof[v] =tdof;

         // generate some debug output
         if (m_verbose)
         {
            double coord[3];
            m_mesh->GetNode(ldof,coord);
            sstr<< std::endl <<"\t"
                <<"{vertex:" << v
                <<",l:" << ldof
                <<",T_l:" << fes.GetLocalTDofNumber(ldof)
                <<",T_g:" << tdof
                <<"} @ (x:"<< coord[0] << ",y:"<<coord[1]
                << ")";
         }
      }
      MPRINT(rank, sstr.str());

      // check that all vertices have valid tdofs
      checkDOFs(vert_tdof, "vertex");

      delete fec;
   }

   /// Generate and test tdofs for the mesh edges (2d or 3D)
   void testEdgeTDofs()
   {
      const int dim = m_mesh->Dimension();
      const int order = 1;

      const int NEdge = m_mesh->GetNEdges();

      MPRINT_MASTER(rank,"Finding edge tdofs");

      mfem::FiniteElementCollection* fec = nullptr;

      switch (dim)
      {
         case 2:
            fec = new mfem::RT_FECollection(order-1,dim);
            break;
         case 3:
            fec = new mfem::ND_FECollection(order,dim);
            break;
         default:
            FAIL("Requires dim==2 or dim==3");
            break;
      }

      mfem::ParFiniteElementSpace fes(m_mesh, fec);
      mfem::Array<int> edge_tdof(NEdge);
      edge_tdof = UNITIALIZED_VAL;

      mfem::Array<int> idofs;

      std::stringstream sstr;
      for (int e=0; e< NEdge; ++e)
      {
         fes.GetEdgeDofs(e, idofs);
         CHECK( idofs.Size() == 1);

         auto ldof = idofs[0];
         CHECK( ldof >= 0);

         // WARNING -- The following does not account for NCMesh!
         //            Need to check against GetLocalTDofNumber()
         auto tdof = fes.GetGlobalTDofNumber(ldof);
         CHECK( tdof >= 0);

         edge_tdof[e] =tdof;

         // generate some debug output
         if (m_verbose)
         {
            sstr<< std::endl <<"\t"
                <<"{edge:" << e
                <<",l:" << ldof
                <<",T_l:" << fes.GetLocalTDofNumber(ldof)
                <<",T_g:" << tdof
                <<"}";
         }
      }
      MPRINT(rank, sstr.str());

      // check that all edges have valid tdofs
      checkDOFs(edge_tdof, "edge");

      delete fec;
   }

   /// Generate and test tdofs for the mesh faces (3D only)
   void testFaceTDofs()
   {
      const int dim = m_mesh->Dimension();
      if (dim < 3)
      {
         SUCCEED("Faces only apply to 3D meshes");
         return;
      }

      const int order = 0;
      const int NFace = m_mesh->GetNFaces();

      auto* fec = new mfem::RT_FECollection(order,dim);
      mfem::ParFiniteElementSpace fes(m_mesh, fec);
      mfem::Array<int> face_tdof(NFace);
      face_tdof = UNITIALIZED_VAL;

      mfem::Array<int> idofs;

      std::stringstream sstr;
      for (int f=0; f< NFace; ++f)
      {
         fes.GetFaceDofs(f, idofs);
         CHECK( idofs.Size() == 1);

         auto ldof = decodeDof(idofs[0]);
         CHECK( ldof >= 0);

         auto tdof = fes.GetGlobalTDofNumber(ldof);
         CHECK( tdof >= 0);

         face_tdof[f] =tdof;

         // generate some debug output
         if (m_verbose)
         {
            sstr<< std::endl <<"\t"
                <<"{face:" << f
                <<",l:" << ldof
                <<",T_l:" << fes.GetLocalTDofNumber(ldof)
                <<",T_g:" << tdof
                <<"}";
         }
      }
      MPRINT(rank, sstr.str());

      // check that all edges have valid tdofs
      checkDOFs(face_tdof, "face");

      delete fec;
   }

   /// Generate and test tdofs for the mesh elements
   void testElementTDofs()
   {
      const int dim = m_mesh->Dimension();
      const int order = 0;

      const int NE = m_mesh->GetNE();

      MPRINT_MASTER(rank,"Finding element tdofs");

      auto* fec = new mfem::L2_FECollection(order, dim);
      mfem::ParFiniteElementSpace fes(m_mesh, fec);
      mfem::Array<int> elem_tdof(NE);
      elem_tdof = UNITIALIZED_VAL;

      mfem::Array<int> idofs;

      std::stringstream sstr;
      for (int el=0; el< NE; ++el)
      {
         fes.GetElementDofs(el, idofs);
         CHECK( idofs.Size() == 1);

         auto ldof = idofs[0];
         CHECK( ldof >= 0);

         auto tdof = fes.GetGlobalTDofNumber(ldof);
         CHECK( tdof >= 0);

         elem_tdof[el] =tdof;

         // generate some debug output
         if (m_verbose)
         {
            sstr<< std::endl <<"\t"
                <<"{element:" << el
                <<",l:" << ldof
                <<",T_l:" << fes.GetLocalTDofNumber(ldof)
                <<",T_g:" << tdof
                <<"}";
         }
      }
      MPRINT(rank, sstr.str());

      // check that all elements have valid tdofs
      checkDOFs(elem_tdof, "element");

      delete fec;
   }


   /// Generate and test tdofs for the mesh boundary elements
   void testBoundaryElementTDofs()
   {
      MPRINT_MASTER(rank,"Finding boundary element tdofs");

      const int dim = m_mesh->Dimension();
      const int order = 0;
      const int NBE = m_mesh->GetNBE();

      int NCodimFaces = 0;
      switch (dim)
      {
         case 1: NCodimFaces = m_mesh->GetNV(); break;
         case 2: NCodimFaces = m_mesh->GetNEdges(); break;
         case 3: NCodimFaces = m_mesh->GetNFaces(); break;
         default: FAIL("Dim must be 1,2 or 3."); break;
      }

      auto* fec = new mfem::RT_FECollection(order, dim);
      mfem::ParFiniteElementSpace fes(m_mesh, fec);
      mfem::Array<int> be_tdof(NBE);
      be_tdof = UNITIALIZED_VAL;

      mfem::Array<int> idofs;

      std::stringstream sstr;
      for (int be=0; be< NBE; ++be)
      {
         // Find tdof of mesh element corresponding to the boundary element
         auto elem_id = m_mesh->GetBdrElementEdgeIndex(be);
         CHECK( elem_id >= 0);
         CHECK( elem_id < NCodimFaces);

         switch (dim)
         {
            case 1: fes.GetVertexDofs(elem_id, idofs); break;
            case 2: fes.GetEdgeDofs(elem_id, idofs); break;
            case 3: fes.GetFaceDofs(elem_id, idofs); break;
         }
         CHECK( idofs.Size() == 1);

         auto ldof = decodeDof(idofs[0]);
         CHECK( ldof >= 0);

         // Check local tdof?
         // auto ltdof = fes.GetLocalTDofNumber(ldof);

         auto tdof = fes.GetGlobalTDofNumber(ldof);
         CHECK( tdof >= 0);

         be_tdof[be] =tdof;

         // generate some debug output
         if (m_verbose)
         {
            sstr<< std::endl <<"\t"
                <<"{be:" << be
                <<",l:" << ldof
                <<",T_l:" << fes.GetLocalTDofNumber(ldof)
                <<",T_g:" << tdof
                <<"}";
         }
      }
      MPRINT(rank, sstr.str());

      // check that all elements have valid tdofs
      checkDOFs(be_tdof, "boundary element");

      delete fec;
   }

   mfem::ParMesh* m_mesh;
   static constexpr int UNITIALIZED_VAL = -100;
   int rank;
   bool m_verbose;
};

} // end anonymous namespace


TEST_CASE("Segment mesh tdofs",
          "[ParFESpace]"
          "[Hypre]"
          "[TDofs]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPRINT_MASTER(rank, "=== Checking TDOFs for Segment mesh ===");

   const int attrib = 1;
   mfem::Mesh* mesh = nullptr;
   mfem::ParMesh* pmesh = nullptr;

   // Build a simple single element segment mesh
   const int dim = 1, nv = 2, ne = 1, nb = 0, sdim = 2;
   {
      mesh = new mfem::Mesh(dim, nv, ne, nb, sdim);

      mesh->AddVertex(mfem::Vertex(0.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,0.)());

      int idx[2] = {0,1};
      mesh->AddElement(new mfem::Segment(idx, attrib));

      mesh->FinalizeMesh();

      CHECK( mesh->GetNV() == nv);
      CHECK( mesh->GetNE() == ne);
      //CHECK( mesh->GetNBE() == 2);
      CHECK( mesh->Dimension() == dim);
      CHECK( mesh->SpaceDimension() == sdim);
   }

   // refine serial mesh
   {
      refineMesh(mesh, NUM_SER_REF_LEVELS);

      // Check results
      int dyad = 1 << NUM_SER_REF_LEVELS;
      const int exp_nv = (dyad+1);
      CHECK( mesh->GetNV() == exp_nv );

      const int exp_ne = (dyad);
      CHECK( mesh->GetNE() == exp_ne );
   }

   MPRINT_MASTER(rank, "Details for serial mesh, after "<< NUM_SER_REF_LEVELS
                 << " levels of uniform refinement:\n" <<  getMeshStats(mesh));

   // Create and refine parallel mesh
   {
      pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);

      delete mesh;
      mesh = nullptr;

      refineMesh(pmesh, NUM_PAR_REF_LEVELS);
   }

   SECTION("Testing true dofs, w/ curvature order 1")
   {
      pmesh->SetCurvature(1);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testElementTDofs();
   }

   delete pmesh;
}


TEST_CASE("Quad mesh tdofs",
          "[ParFESpace]"
          "[Hypre]"
          "[TDofs]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPRINT_MASTER(rank, "=== Checking TDOFs for Quad mesh ===");

   const int attrib = 1;
   mfem::Mesh* mesh = nullptr;
   mfem::ParMesh* pmesh = nullptr;

   // Build a simple single element quad mesh
   const int dim = 2, nv = 4, ne = 1, nb = 0, sdim = 2;
   {
      mesh = new mfem::Mesh(dim, nv, ne, nb, sdim);

      mesh->AddVertex(mfem::Vertex(0.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,1.)());
      mesh->AddVertex(mfem::Vertex(0.,1.)());

      int idx[4] = {0,1,2,3};
      mesh->AddElement(new mfem::Quadrilateral(idx, attrib));

      mesh->FinalizeMesh();

      CHECK( mesh->GetNV() == nv);
      CHECK( mesh->GetNE() == ne);
      CHECK( mesh->GetNBE() == 4);
      CHECK( mesh->Dimension() == dim);
      CHECK( mesh->SpaceDimension() == sdim);
   }

   // refine serial mesh
   {
      refineMesh(mesh, NUM_SER_REF_LEVELS);

      // Check results
      int dyad = 1 << NUM_SER_REF_LEVELS;
      const int exp_nv = (dyad+1)*(dyad+1);
      CHECK( mesh->GetNV() == exp_nv );

      const int exp_ne = (dyad)*(dyad);
      CHECK( mesh->GetNE() == exp_ne );
   }

   MPRINT_MASTER(rank, "Details for serial mesh, after "<< NUM_SER_REF_LEVELS
                 << " levels of uniform refinement:\n" <<  getMeshStats(mesh));

   // Create and refine parallel mesh
   {
      pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);

      delete mesh;
      mesh = nullptr;

      refineMesh(pmesh, NUM_PAR_REF_LEVELS);
   }

   SECTION("Testing true dofs, no curvature")
   {
      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   SECTION("Testing true dofs, w/ curvature order 1")
   {
      pmesh->SetCurvature(1);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   SECTION("Testing true dofs, w/ curvature order 2")
   {
      pmesh->SetCurvature(2);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   delete pmesh;
}


TEST_CASE("Tri mesh tdofs",
          "[ParFESpace]"
          "[Hypre]"
          "[TDofs]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPRINT_MASTER(rank, "=== Checking TDOFs for Triangle mesh ===");

   const int attrib = 1;
   mfem::Mesh* mesh = nullptr;
   mfem::ParMesh* pmesh = nullptr;

   // Build a simple single element triangle mesh
   const int dim = 2, nv = 3, ne = 1, nb = 0, sdim = 2;
   {
      mesh = new mfem::Mesh(dim, nv, ne, nb, sdim);

      mesh->AddVertex(mfem::Vertex(0.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,1.)());

      int idx[3] = {0,1,2};
      mesh->AddElement(new mfem::Triangle(idx, attrib));

      mesh->FinalizeMesh();

      CHECK( mesh->GetNV() == nv);
      CHECK( mesh->GetNE() == ne);
      CHECK( mesh->GetNBE() == 3);
      CHECK( mesh->Dimension() == dim);
      CHECK( mesh->SpaceDimension() == sdim);
   }

   // refine serial mesh
   {
      refineMesh(mesh, NUM_SER_REF_LEVELS);

      // Check results
      const int dyad = 1 << NUM_SER_REF_LEVELS;
      const int exp_nv = (dyad+1)*(dyad+2)/2;
      CHECK( mesh->GetNV() == exp_nv );

      const int exp_ne = dyad*dyad;
      CHECK( mesh->GetNE() == exp_ne );
   }

   MPRINT_MASTER(rank, "Details for serial mesh, after "<< NUM_SER_REF_LEVELS
                 << " levels of uniform refinement:\n" <<  getMeshStats(mesh));

   // Create and refine parallel mesh
   {
      pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);

      delete mesh;
      mesh = nullptr;

      refineMesh(pmesh, NUM_PAR_REF_LEVELS);
   }

   SECTION("Testing true dofs, no curvature")
   {
      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   SECTION("Testing true dofs, w/ curvature order 1")
   {
      pmesh->SetCurvature(1);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   SECTION("Testing true dofs, w/ curvature order 2")
   {
      pmesh->SetCurvature(2);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   delete pmesh;
}


TEST_CASE("Hex mesh tdofs",
          "[ParFESpace]"
          "[Hypre]"
          "[TDofs]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPRINT_MASTER(rank, "=== Checking TDOFs for Hex mesh ===");

   const int attrib = 1;
   mfem::Mesh* mesh = nullptr;
   mfem::ParMesh* pmesh = nullptr;

   // Build a simple single element hex mesh
   const int dim = 3, nv = 8, ne = 1, nb = 0, sdim = 3;
   {
      mesh = new mfem::Mesh(dim, nv, ne, nb, sdim);

      mesh->AddVertex(mfem::Vertex(0.,0.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,0.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,1.,0.)());
      mesh->AddVertex(mfem::Vertex(0.,1.,0.)());
      mesh->AddVertex(mfem::Vertex(0.,0.,1.)());
      mesh->AddVertex(mfem::Vertex(1.,0.,1.)());
      mesh->AddVertex(mfem::Vertex(1.,1.,1.)());
      mesh->AddVertex(mfem::Vertex(0.,1.,1.)());

      int idx[8] = {0,1,2,3,4,5,6,7};
      mesh->AddElement(new mfem::Hexahedron(idx, attrib));

      mesh->FinalizeMesh();

      REQUIRE( mesh->GetNV() == nv);
      REQUIRE( mesh->GetNE() == ne);
      REQUIRE( mesh->GetNBE() == 6);
      REQUIRE( mesh->Dimension() == dim);
      REQUIRE( mesh->SpaceDimension() == sdim);
   }

   // refine and finalize the mesh
   {
      refineMesh(mesh, NUM_SER_REF_LEVELS);

      // Check results
      int dyad = 1 << NUM_SER_REF_LEVELS;
      const int exp_nv = (dyad+1)*(dyad+1)*(dyad+1);
      CHECK( mesh->GetNV() == exp_nv );

      const int exp_ne = (dyad)*(dyad)*(dyad);
      CHECK( mesh->GetNE() == exp_ne );
   }

   MPRINT_MASTER(rank, "Details for serial mesh, after "<< NUM_SER_REF_LEVELS
                 << " levels of uniform refinement:\n" <<  getMeshStats(mesh));

   // Create and refine parallel mesh
   {
      pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);

      delete mesh;
      mesh = nullptr;

      refineMesh(pmesh, NUM_PAR_REF_LEVELS);
   }

   SECTION("Testing true dofs, without curvature")
   {
      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testFaceTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   SECTION("Testing true dofs, w/ curvature order 1")
   {
      pmesh->SetCurvature(1);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testFaceTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   SECTION("Testing true dofs, w/ curvature order 2")
   {
      pmesh->SetCurvature(2);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testFaceTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   delete pmesh;
}


TEST_CASE("Tet mesh tdofs",
          "[ParFESpace]"
          "[Hypre]"
          "[TDofs]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPRINT_MASTER(rank, "=== Checking TDOFs for Tet mesh ===");

   const int attrib = 1;
   mfem::Mesh* mesh = nullptr;
   mfem::ParMesh* pmesh = nullptr;

   // Build a simple single element tet mesh
   const int dim = 3, nv = 4, ne = 1, nb = 0, sdim = 3;
   {
      mesh = new mfem::Mesh(dim, nv, ne, nb, sdim);

      mesh->AddVertex(mfem::Vertex(0.,0.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,0.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,1.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,1.,1.)());

      int idx[4] = {0,1,2,3};
      mesh->AddElement(new mfem::Tetrahedron(idx, attrib));

      mesh->FinalizeMesh();

      REQUIRE( mesh->GetNV() == nv);
      REQUIRE( mesh->GetNE() == ne);
      REQUIRE( mesh->GetNBE() == 4);
      REQUIRE( mesh->Dimension() == dim);
      REQUIRE( mesh->SpaceDimension() == sdim);
   }

   // refine mesh
   {
      refineMesh(mesh, NUM_SER_REF_LEVELS);

      // Check results
      int dyad = 1 << NUM_SER_REF_LEVELS;

      const int exp_nv = (dyad+1)*(dyad+2)*(dyad+3)/6;
      CHECK( mesh->GetNV() == exp_nv );

      const int exp_ne = (dyad)*(dyad)*(dyad);
      CHECK( mesh->GetNE() == exp_ne );
   }

   MPRINT_MASTER(rank, "Details for serial mesh, after "<< NUM_SER_REF_LEVELS
                 << " levels of uniform refinement:\n" <<  getMeshStats(mesh));

   // Create and refine parallel mesh
   {
      pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);

      delete mesh;
      mesh = nullptr;

      refineMesh(pmesh, NUM_PAR_REF_LEVELS);
   }

   SECTION("Testing true dofs, without curvature")
   {
      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testFaceTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   SECTION("Testing true dofs, w/ curvature order 1")
   {
      pmesh->SetCurvature(1);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testFaceTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   SECTION("Testing true dofs, w/ curvature order 2")
   {
      pmesh->SetCurvature(2);

      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);
      tester.testVertexTDofs();
      tester.testEdgeTDofs();
      tester.testFaceTDofs();
      tester.testElementTDofs();

      tester.testBoundaryElementTDofs();
   }

   delete pmesh;
}

TEST_CASE("NC Quad mesh tdofs",
          "[ParFESpace]"
          "[NCMesh]"
          "[Hypre]"
          "[TDofs]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPRINT_MASTER(rank,
                 "=== Checking TDOFs for Non-conforming Quad mesh ===");

   const double refine_prob = .5;

   const int attrib = 1;
   mfem::Mesh* mesh = nullptr;
   mfem::ParMesh* pmesh = nullptr;

   // Build a simple single element quad mesh
   const int dim = 2, nv = 4, ne = 1, nb = 0, sdim = 2;
   {
      mesh = new mfem::Mesh(dim, nv, ne, nb, sdim);

      mesh->AddVertex(mfem::Vertex(0.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,0.)());
      mesh->AddVertex(mfem::Vertex(1.,1.)());
      mesh->AddVertex(mfem::Vertex(0.,1.)());

      int idx[4] = {0,1,2,3};
      mesh->AddElement(new mfem::Quadrilateral(idx, attrib));

      mesh->FinalizeMesh();

      CHECK( mesh->GetNV() == nv);
      CHECK( mesh->GetNE() == ne);
      CHECK( mesh->GetNBE() == 4);
      CHECK( mesh->Dimension() == dim);
      CHECK( mesh->SpaceDimension() == sdim);

      mesh->SetCurvature(2);
      mesh->EnsureNCMesh();
   }

   // refine serial mesh
   {
      // Refine the mesh once before random refinement
      refineMesh(mesh, 1);

      for (int i=0; i < NUM_SER_REF_LEVELS; ++i)
      {
         mesh->RandomRefinement(refine_prob);
      }
   }

   MPRINT_MASTER(rank, "Details for serial mesh, after random NC refinement:\n"
                 << getMeshStats(mesh));

   // Create and refine parallel mesh
   {
      pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);

      delete mesh;
      mesh = nullptr;

      for (int i=0; i < NUM_PAR_REF_LEVELS; ++i)
      {
         pmesh->RandomRefinement(refine_prob);
      }
   }

   SECTION("Testing true dofs for nc mesh")
   {
      TDOFTester  tester(pmesh, VERBOSE_OUTPUT);

      /// WARNING:
      //      vertex and edge tdof tests currently fail for ncmesh
      //
      //tester.testVertexTDofs();
      //tester.testEdgeTDofs();

      tester.testElementTDofs();
      tester.testBoundaryElementTDofs();
   }

   delete pmesh;
}


#undef PTEST_TDOF_USE_DEBUG_PRINT

#endif // MFEM_USE_MPI