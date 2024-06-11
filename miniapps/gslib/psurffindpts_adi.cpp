// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//    ----------------------------------------------------------------------
//    Parallel Find Points Miniapp: Evaluate grid function in physical space
//    ----------------------------------------------------------------------
//
// This miniapp demonstrates the interpolation of a high-order grid function on
// a set of points in physical-space. The miniapp is based on GSLIB-FindPoints,
// which provides two key functionalities. First, for a given set of points in
// the physical-space, it determines the computational coordinates (element
// number, reference-space coordinates inside the element, and processor number
// [in parallel]) for each point. Second, based on computational coordinates, it
// interpolates a grid function in the given points. Inside GSLIB, computation
// of the coordinates requires use of a Hash Table to identify the candidate
// processor and element for each point, followed by the Newton's method to
// determine the reference-space coordinates inside the candidate element.
//
// Compile with: make pfindpts
//
// Sample runs:
//    make psurffindpts_adi -j && mpirun -np 1 psurffindpts_adi -mo 4 -nx 1 --dim 2 -o 1 -visit
//    make psurffindpts_adi -j && mpirun -np 2 psurffindpts_adi -mo 2 -dim 3 -nx 2 -o 1 -visit

#include "mfem.hpp"
#include "general/forall.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

double func_order;  // order of the user-defined field function

// Scalar function to project
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++)
   {
      res += std::pow(x(d), func_order);
   }
   return res;
}

void F_exact(const Vector &p, Vector &F)
{
   F(0) = field_func(p);
   for (int i = 1; i < F.Size(); i++)
   {
      F(i) = (i+1)*F(0);
   }
}

int main (int argc, char *argv[])
{
   std::remove("out");

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   int num_procs = Mpi::WorldSize();
   int myid      = Mpi::WorldRank();
   
   ofstream ofile("out", std::ios::app);

   // Set the method's default parameters.
   const char *mesh_file = "NULL";
   int order             = 3;
   int mesh_poly_deg     = 3;
   int rs_levels         = 0;
   int rp_levels         = 0;
   bool visualization    = false;
   int ncomp             = 1;
   bool search_on_rank_0 = false;
   bool hrefinement       = false;
   int point_ordering    = 0;
   int gf_ordering       = 0;
   const char *devopt    = "cpu";
   double jitter         = 0.0;
   int exact_sol_order   = 1;
   int smooth            = 0; //kershaw transformation parameter
   int jobid             = 0;
   int npt               = 100; //points per proc
   int nx                = 6; //points per proc
   int dim               = 3;
   int etype             = 0;
   bool visit            = false;
   int gpucode           = 0;

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
   args.AddOption(&ncomp, "-nc", "--ncomp",
                  "Number of components for H1 or L2 GridFunctions");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&search_on_rank_0, "-sr0", "--search-on-r0", "-no-sr0",
                  "--no-search-on-r0",
                  "Enable search only on rank 0 (disable to search points on all tasks).");
   args.AddOption(&hrefinement, "-hr", "--h-refinement", "-no-hr",
                  "--no-h-refinement",
                  "Do random h refinements to mesh (does not work for pyramids).");
   args.AddOption(&point_ordering, "-po", "--point-ordering",
                  "Ordering of points to be found."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&gf_ordering, "-gfo", "--gridfunc-ordering",
                  "Ordering of fespace that will be used for gridfunction to be interpolated."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&exact_sol_order, "-eo", "--exact-sol-order",
                  "Order for analytic solution.");
   args.AddOption(&smooth, "-smooth", "--smooth",
                  "smooth parameter of kershaw");
   args.AddOption(&jobid, "-jid", "--jid",
                  "job id used for visit  save files");
   args.AddOption(&npt, "-npt", "--npt",
                  "# points per proc");
   args.AddOption(&nx, "-nx", "--nx",
                  "# of elements in x(is multipled by rs)");
   args.AddOption(&dim, "-dim", "--dim",
                  "Dimension");
   args.AddOption(&etype, "-et", "--et",
                  "element type: 0 - quad/hex, 1 - triangle/tetrahedron");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable VISIT output");
   args.AddOption(&gpucode, "-gpucode", "--gpucode",
                  "code for custom gpu kernels");
   args.Parse();

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid==0)
   {
      args.PrintOptions(cout);
   }
   bool cpu_mode = strcmp(devopt,"cpu")==0;

   if (hrefinement)
   {
      MFEM_VERIFY(strcmp(devopt,"cpu")==0, "HR-adaptivity is currently only"
                  " supported on cpus.");
   }
   Device device(devopt);
   if (myid == 0)
   {
      device.Print();
   }

   // restricting max user-defined field function order
   func_order = std::min(exact_sol_order, 2);

   // Initialize and refine the starting mesh.
   int nex = nx*std::pow(2, rs_levels); // mesh size after refinement
   Mesh *mesh = NULL;
   if (strcmp(mesh_file,"NULL") != 0)
   {
      mesh = new Mesh(mesh_file, 1, 1, false);
      for (int lev = 0; lev < rs_levels; lev++)
      {
         mesh->UniformRefinement();
      }
      dim = mesh->Dimension();  // Ref. space dim (might be different from SpaceDim)
   }
   else
   {
      // It seems the mesh generated by code below has unit side length
      if (dim == 2)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(nex, nex, etype == 0 ?
                                               Element::QUADRILATERAL :
                                               Element::TRIANGLE));

      }
      else if (dim == 3)
      {
         mesh = new Mesh(Mesh::MakeCartesian3D(nex, nex, nex, etype == 0 ?
                                               Element::HEXAHEDRON :
                                               Element::TETRAHEDRON));
      }
      else
      {
         MFEM_ABORT("Only 2D and 3D supported at the moment.");
      }
   }
   mesh->EnsureNodes();  // Ensure gridfunction object exists in mesh object

   // Display the volume/area mesh.
   if (visualization && myid == 0)
   {
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "mesh\n";
      }
      mesh->Print(sock);
      // mesh->GetNodes()->Save(sock);
      if (myid == 0)
      {
         sock << "window_title 'Volume/area mesh'\n"
              << "window_geometry "
              << 0 << " " << 0 << " " << 400 << " " << 400 << "\n"
              << "keys Rmjpee" << endl;
      }
   }

   if (visit && myid == 0)
   {
      VisItDataCollection dc("volmesh", mesh);
      dc.SetFormat(DataCollection::SERIAL_FORMAT);
      dc.Save();
   }
   MPI_Barrier(MPI_COMM_WORLD);

   if (myid == 0)
   {
      cout << "Mesh curvature of the original mesh: ";
      if (mesh->GetNodes())
      {
         cout << mesh->GetNodes()->OwnFEC()->Name() << endl;
      }
      else
      {
         cout << "(NONE)";
      }
      cout << endl;
   }

   MFEM_VERIFY(mesh_poly_deg>0, "The order of the mesh must be positive.");

   // H1_2D_P<mesh_poly_deg> for false, L2_T1_2D_P<mesh_poly_deg> for true
   mesh->SetCurvature(mesh_poly_deg, false, -1, 0);
   if (myid == 0)
   {
      cout << "Mesh curvature after SetCurvature: "
           << mesh->GetNodes()->OwnFEC()->Name()
           << endl;
   }

   int nattr = mesh->bdr_attributes.Max();
   ofile << "nattr " << nattr << "myid"<<myid << endl;
   mesh->bdr_attributes.Print(ofile);
   Array<int> subdomain_attributes(nattr);
   for (int i=0; i<nattr; i++)
   {
      subdomain_attributes[i] = i+1;
   }
   auto submesh = SubMesh::CreateFromBoundary(*mesh, subdomain_attributes);
   if (visit && myid == 0)
   {
      VisItDataCollection dc("submesh", &submesh);
      dc.SetFormat(DataCollection::SERIAL_FORMAT);
      dc.Save();
   }
   MPI_Barrier(MPI_COMM_WORLD);

   ParMesh psubmesh(MPI_COMM_WORLD, submesh);

   ofile << "After ParMesh " << "myid"<<myid << endl;

   ofile << "Number of boundary elements: " << psubmesh.GetNE()
        << " " << "myid"<<myid << endl;


   MFEM_VERIFY(ncomp > 0, "Invalid number of components.");
   int vec_dim = ncomp;
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   if (myid == 0) cout << "H1-GridFunction\n";
   ParFiniteElementSpace sc_fes(&psubmesh, fec, ncomp, gf_ordering);
   ParGridFunction field_vals(&sc_fes);
   VectorFunctionCoefficient F(vec_dim, F_exact);
   field_vals.ProjectCoefficient(F);

   if (visualization)
   {
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      psubmesh.PrintAsOne(sock);
      field_vals.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Surface mesh'\n"
              << "window_geometry "
              << 400 << " " << 0 << " " << 400 << " " << 400 << "\n"
              << "keys RmjAppe" << endl;
      }
   }

   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.SetupSurf(psubmesh);
   cout << "SetupSurf done" << endl;

   Mesh *mesh_abb, *mesh_lhbb, *mesh_ghbb;
   if (visit)
   {
      mesh_abb  = finder.GetBoundingBoxMeshSurf(0);  // Axis aligned bounding box
      mesh_lhbb = finder.GetBoundingBoxMeshSurf(2);  // Local Hash bounding box
      mesh_ghbb = finder.GetBoundingBoxMeshSurf(3);  // Global Hash bounding box
      if (myid == 0)
      {
         VisItDataCollection dc0("finderabb", mesh_abb);
         dc0.SetFormat(DataCollection::SERIAL_FORMAT);
         dc0.Save();

         VisItDataCollection dc2("finderlhbb", mesh_lhbb);
         dc2.SetFormat(DataCollection::SERIAL_FORMAT);
         dc2.Save();

         VisItDataCollection dc3("finderghbb", mesh_ghbb);
         dc3.SetFormat(DataCollection::SERIAL_FORMAT);
         dc3.Save();
      }
   }

   if (visit && myid == 0)
   {
      // Array<int> attrlist(1);
      // for (int i = 0; i < mesh_abb->GetNE(); i++)
      // {
      //    attrlist[0] = i+1;
      //    auto mesh_abbt = SubMesh::CreateFromDomain(*mesh_abb, attrlist);
      //    VisItDataCollection dct("finderabbt", &mesh_abbt);
      //    dct.SetFormat(DataCollection::SERIAL_FORMAT);
      //    dct.SetCycle(i);
      //    dct.SetTime(i*1.0);
      //    dct.Save();
      // }
   }
   MPI_Barrier(MPI_COMM_WORLD);

   // Generate random points
   npt = 1;
   Vector point_pos(npt*dim);
   // ordering byNodes
   for (int i=0; i<npt; i++) {
      point_pos(i*dim + 0) = 0.98;
      point_pos(i*dim + 1) = 0.50;
   }

   finder.FindPointsSurf(point_pos);

   delete fec;

   cout << "Just before FreeData" << endl;
   // finder.FreeData();

   return 0;
}
