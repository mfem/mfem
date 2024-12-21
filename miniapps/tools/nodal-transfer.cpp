// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details
//
//  -------------------------------------------------------------------------
//  Nodal Transfer Miniapp: Map ParGridFunction to Different MPI Partitioning
//  -------------------------------------------------------------------------
//
// The Nodal Transfer Miniapp maps partitioned parallel grid function to a
// parallel grid function partitioned on a different number of processes. The
// miniapp has two regimes: 1) Generates partitioned parallel grid function
// and saves it to a set of files; 2) Reads the partitioned grid function and
// maps it to the current partition. The map assumes that the position of the
// nodal DOFs does not change between the original grid function and the target
// grid function. The transfer does not perform any interpolation. It just
// copies the nodal values between the two grid functions.
//
// Generate second order mesh on 4 processes
//    mpirun -np 4 ./nodal-transfer -rs 2 -rp 1 -gd 1 -o 2
// Read the generated data and map it to a grid function defined on two processes
//    mpirun -np 2 ./nodal-transfer -rs 2 -rp 0 -gd 0 -snp 4 -o 2
//
// Generate first order grid function on 8 processes
//    mpirun -np 8 ./nodal-transfer -rs 2 -rp 2 -gd 1 -o 1 -m ../../data/star.mesh
// Read the generated data on 4 processes and coarser mesh
//    mpirun -np 4 ./nodal-transfer -rs 2 -rp 0 -gd 0 -snp 8 -o 1 -m ../../data/star.mesh

#include <mfem.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include "../common/mfem-common.hpp"

using namespace mfem;

class TestCoeff : public Coefficient
{
public:
   TestCoeff() {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      if (T.GetSpaceDim()==3)
      {
         real_t x[3];
         Vector transip(x, 3);
         T.Transform(ip, transip);
         return std::sin(x[0])*std::cos(x[1]) +
                std::sin(x[1])*std::cos(x[2]) +
                std::sin(x[2])*std::cos(x[0]);
      }
      else if (T.GetSpaceDim()==2)
      {
         real_t x[2];
         Vector transip(x, 2);
         T.Transform(ip, transip);
         return std::sin(x[0])*std::cos(x[1]) +
                std::sin(x[1])*std::cos(x[0]);
      }
      else
      {
         real_t x;
         Vector transip(&x,1);
         T.Transform(ip, transip);
         return std::sin(x)+std::cos(x);
      }
   }
};

int main(int argc, char* argv[])
{
   // Initialize MPI.
   Mpi::Init(argc, argv);
   int myrank = Mpi::WorldRank();

   // Parse command-line options
   const char *mesh_file = "../../data/beam-tet.mesh";
   int ser_ref_levels = 3;
   int par_ref_levels = 1;
   int order = 1;
   int gen_data = 1;
   int src_num_procs = 4;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&gen_data,
                  "-gd",
                  "--generate-data",
                  "Generate input data for the transfer.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&src_num_procs,
                  "-snp",
                  "--src_num_procs",
                  "Number of processes for the src grid function.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable ParaView visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors. We
   // can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   // with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.SpaceDimension();

   // Refine the mesh in serial to increase the resolution. In this example
   // we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   // a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // Define the finite element spaces for the solution
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec, 1, Ordering::byVDIM);
   HYPRE_Int glob_size = fespace.GlobalTrueVSize();
   if (myrank == 0)
   {
      std::cout << "Number of finite element unknowns: " << glob_size
                << std::endl;
   }

   ParGridFunction x(&fespace); x=0.0;
   TestCoeff prco;
   if (gen_data)
   {
      Coefficient* coef[2]; coef[0]=&prco; coef[1]=&prco;
      x.ProjectCoefficient(coef);

      // Save the grid function
      {
         // Save the mesh and the data
         std::ostringstream oss;
         oss << std::setw(10) << std::setfill('0') << myrank;
         std::string mname="mesh_"+oss.str()+".msh";
         std::string gname="gridfunc_"+oss.str()+".gf";
         std::ofstream sout;

         // Save the mesh
         sout.open(mname.c_str(),std::ios::out);
         sout.precision(20);
         pmesh.ParPrint(sout);
         sout.close();

         // Save the grid function data
         sout.open(gname.c_str(),std::ios::out);
         sout.precision(20);
         x.Save(sout);
         sout.close();
      }
   }
   else
   {
      // Read the grid function written to files and map it to the current
      // partition scheme.
      // x grid function will be the target of the transfer
      // y will be utilized later for comparison
      ParGridFunction y(&fespace);
      Coefficient* coef[2]; coef[0]=&prco; coef[1]=&prco;
      y.ProjectCoefficient(coef);

      // Map the src grid function
      {
         std::ifstream in;
         BaseKDTreeNodalProjection* map;
         if (dim==2)
         {
            map = new KDTreeNodalProjection<2>(x);
         }
         else
         {
            map = new KDTreeNodalProjection<3>(x);
         }
         for (int p=0; p<src_num_procs; p++)
         {
            std::ostringstream oss;
            oss << std::setw(10) << std::setfill('0') << p;
            std::string mname="mesh_"+oss.str()+".msh";
            std::string gname="gridfunc_"+oss.str()+".gf";

            // Read the mesh
            Mesh lmesh;
            in.open(mname.c_str(),std::ios::in);
            lmesh.Load(in);
            in.close();

            in.open(gname.c_str(),std::ios::in);
            GridFunction gf(&lmesh,in);
            in.close();

            // Project the grid function
            map->Project(gf,1e-8);
         }
         delete map;
      }

      // Write the result into a ParaView file
      if (visualization)
      {
         ParaViewDataCollection paraview_dc("GridFunc", &pmesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
         paraview_dc.SetCycle(0);
         paraview_dc.SetTime(0.0);
         paraview_dc.RegisterField("x",&x);
         paraview_dc.RegisterField("y",&y);
         paraview_dc.Save();
      }

      // Compare the results
      Vector tmpv = x;
      tmpv -= y;
      real_t l2err = mfem::InnerProduct(MPI_COMM_WORLD,tmpv,tmpv);
      if (myrank==0)
      {
         std::cout<<"|l2 error|="<<sqrt(l2err)<<std::endl;
      }
   }

   return 0;
}
