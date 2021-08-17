// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// Compile with: make parrsb
//
// Sample runs:
//    mpirun -np 4 parrsb -m ../../data/inline-quad.mesh
//    mpirun -np 4 parrsb -m ../../data/inline-hex.mesh
//    mpirun -np 4 parrsb -m ../../data/armadillo.vtk

#include "../../mfem.hpp"

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Set the method's default parameters.
   const char *mesh_file = "../../data/rt-2d-q3.mesh";
   int order             = 3;
   int mesh_poly_deg     = 3;
   int rs_levels         = 0;
   int rp_levels         = 0;
   bool visualization    = false;
   int fieldtype         = 0;
   int ncomp             = 1;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&mesh_poly_deg, "-mo", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&fieldtype, "-ft", "--field-type",
                  "Field type: 0 - H1, 1 - L2, 2 - H(div), 3 - H(curl).");
   args.AddOption(&ncomp, "-nc", "--ncomp",
                  "Number of components for H1 or L2 GridFunctions");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   if (myid == 0)
   {
      cout << "Mesh curvature of the original mesh: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }

   // Mesh bounding box (for the full serial mesh).
   Vector pos_min, pos_max;
   MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be positive.");
   mesh->GetBoundingBox(pos_min, pos_max, mesh_poly_deg);
   if (myid == 0)
   {
      cout << "--- Generating equidistant point for:\n"
           << "x in [" << pos_min(0) << ", " << pos_max(0) << "]\n"
           << "y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
      if (dim == 3)
      {
         cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
      }
   }

   // Distribute the mesh.

   //mesh->EnsureNCMesh();
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   int nel = pmesh.GetNE();
   int nel_min, nel_max;
   MPI_Allreduce(&nel, &nel_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&nel, &nel_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   if (myid == 0) {
       std::cout << nel_max << " " << nel_min << " METIS-NELMAX-NELMIN\n";
   }

   // Curve the mesh based on the chosen polynomial degree.
   L2_FECollection fecm(0, dim);
   ParFiniteElementSpace pfespace(&pmesh, &fecm);
   ParGridFunction x(&pfespace);
   for (int i = 0; i < pmesh.GetNE(); i++) {
       x(i) = myid;
   }

   if (visualization) {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sout;
       sout.open(vishost, visport);
       sout << "parallel " << num_procs << " " << myid << "\n";
       sout.precision(8);
       sout << "solution\n" << pmesh << x;
       if (myid == 0)
       {
          sout << "window_title 'Default partitioning'\n"
               << "window_geometry "
               << 00 << " " << 0 << " " << 400 << " " << 400 << "\n"
               << "keys jRmclA" << endl;
       }
   }

   MFEMPARRSB parrsb = MFEMPARRSB(pmesh);
   Array<int> partition;
   parrsb.GetPartitioningParallel(partition);
   for (int i = 0; i < pmesh.GetNE(); i++) {
       x(i) = partition[i];
   }

   if (visualization) {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sout;
       sout.open(vishost, visport);
       sout << "parallel " << num_procs << " " << myid << "\n";
       sout.precision(8);
       sout << "solution\n" << pmesh << x;
       if (myid == 0)
       {
          sout << "window_title 'ParRSB'\n"
               << "window_geometry "
               << 400 << " " << 0 << " " << 400 << " " << 400 << "\n"
               << "keys jRmclA" << endl;
       }
   }


   MFEMPARRSB parrsb2 = MFEMPARRSB(*mesh);
   parrsb2.GetPartitioningSerial(MPI_COMM_WORLD, partition);
   ParMesh pmesh2(MPI_COMM_WORLD, *mesh, partition);
   ParFiniteElementSpace pfespace2(&pmesh2, &fecm);
   ParGridFunction x2(&pfespace2);
   for (int i = 0; i < pmesh2.GetNE(); i++) {
       x2(i) = myid;
   }

   if (visualization) {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sout;
       sout.open(vishost, visport);
       sout << "parallel " << num_procs << " " << myid << "\n";
       sout.precision(8);
       sout << "solution\n" << pmesh2 << x2;
       if (myid == 0)
       {
          sout << "window_title 'ParRSB Serial'\n"
               << "window_geometry "
               << 800 << " " << 0 << " " << 400 << " " << 400 << "\n"
               << "keys jRmclA" << endl;
       }
   }


   //mesh->EnsureNCMesh();
   partition.SetSize(mesh->GetNE());
   for (int i = 0; i < mesh->GetNE(); i++) {
       partition[i] = i % num_procs;
   }
   ParMesh pmesh3(MPI_COMM_WORLD, *mesh, partition);
   MFEMPARRSB parrsb3 = MFEMPARRSB(pmesh3);
   parrsb3.GetPartitioningParallel(partition);
   ParFiniteElementSpace pfespace3(&pmesh3, &fecm);
   ParGridFunction x3(&pfespace3);
   for (int i = 0; i < pmesh3.GetNE(); i++) {
       x3(i) = partition[i];
   }

   if (visualization) {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sout;
       sout.open(vishost, visport);
       sout << "parallel " << num_procs << " " << myid << "\n";
       sout.precision(8);
       sout << "solution\n" << pmesh3 << x3;
       if (myid == 0)
       {
          sout << "window_title 'Manual partition then parRSB'\n"
               << "window_geometry "
               << 400 << " " << 0 << " " << 400 << " " << 400 << "\n"
               << "keys jRmclA" << endl;
       }
   }










   MPI_Finalize();





   return 0;
}
