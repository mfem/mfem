// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//


#include "SDF_Generator.hpp"
#include <fstream>

using namespace mfem;


int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int refinement = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&refinement,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.Parse();

   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

   mfem::Vector tObjectOffset(3); tObjectOffset= 0.0;

   //sdf::SDF_Generator tSDFGenerator( "./Part_1.obj", tObjectOffset, true );
      sdf::SDF_Generator tSDFGenerator( "./Utah_teapot.obj", tObjectOffset, true );

   // Create mesh
   double Lx = 8.0; double Ly = 4.0; double Lz = 4.0;  //double Lx = 6.4; double Ly = 3.2; double Lz = 3.4;
   int NX = 150;      int NY = 75;      int NZ = 75;
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(NX, NY, NZ, mfem::Element::HEXAHEDRON, Lx, Ly, Lz, true);

   int dim = mesh.Dimension();

   int tNumVertices  = mesh.GetNV();
   for (int i = 0; i < tNumVertices; ++i)
   {
       double * Coords = mesh.GetVertex(i);
      //  Coords[ 0 ] = Coords[ 0 ] - 5.3;  
      //  Coords[ 1 ] = Coords[ 1 ] + 3.05;
      //  Coords[ 2 ] = Coords[ 2 ] - 1.7;

       Coords[ 0 ] = Coords[ 0 ] - 4.0;  
       Coords[ 1 ] = Coords[ 1 ] + 0.0;
       Coords[ 2 ] = Coords[ 2 ] - 2.0;
   }

   for (int i = 0; i < refinement; ++i)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   // build h1 desing space
   int orderDesing = 1;
   ::mfem::H1_FECollection desFECol_H1(orderDesing, dim);
   ::mfem::ParFiniteElementSpace desFESpace_scalar_H1(pmesh, &desFECol_H1 );

   // desing variable vector
   mfem::ParGridFunction SDF_GridFunc(&desFESpace_scalar_H1); SDF_GridFunc=0.0;


   tSDFGenerator.calculate_sdf( SDF_GridFunc );

   //SDF_GridFunc.Print();

   std::cout<<"SDF done"<<std::endl;

   
   mfem::ParaViewDataCollection paraview_dc("sdf", pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetTime(1.0);
   paraview_dc.RegisterField("design",&SDF_GridFunc);
   paraview_dc.Save();

 
   // delete pmesh;

   return 0;
}
