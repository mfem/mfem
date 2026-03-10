// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
//            -------------------------------------------------
//            Overlapping Grids Miniapp: Poisson problem (ex1p)
//            -------------------------------------------------
//
// This example code demonstrates use of MFEM to solve the Poisson problem:
//
//                -Delta u = 1 \in [0, 1]^2, u_b = 0 \in \dO
//
// on two overlapping grids. Using simultaneous Schwarz iterations, the Poisson
// equation is solved iteratively, with boundary data interpolated between the
// overlapping boundaries for each grid. The overlapping Schwarz method was
// introduced by H. A. Schwarz in 1870, see also Section 2.2 of "Stability
// analysis of a singlerate and multirate predictor-corrector scheme for
// overlapping grids" by Mittal, Dutta and Fischer, arXiv:2010.00118.
//
// Compile with: make schwarz_ex1p
//
// Sample runs:  mpirun -np 4 schwarz_ex1p -nm 3 -np1 2 -np2 1 -np3 1
//               mpirun -np 4 schwarz_ex1p -nm 2 -np1 2 -np2 2
//               mpirun -np 4 schwarz_ex1p -nm 2 -np1 2 -np2 2 -m1 ../../data/star.mesh -m2 ../../data/beam-quad.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Method to use FindPointsGSLIB to determine the boundary points of a mesh that
// are interior to another mesh.
void GetInterdomainBoundaryPoints(OversetFindPointsGSLIB &finder,
                                  Vector &vxyz, int color,
                                  Array<int> ess_tdof_list,
                                  Array<int> &ess_tdof_list_int, int dim)
{
   int number_boundary = ess_tdof_list.Size(),
       number_true = vxyz.Size()/dim;

   Vector bnd(number_boundary*dim);
   Array<unsigned int> colorv(number_boundary);
   for (int i = 0; i < number_boundary; i++)
   {
      int idx = ess_tdof_list[i];
      for (int d = 0; d < dim; d++)
      {
         bnd(i+d*number_boundary) = vxyz(idx + d*number_true);
      }
      colorv[i] = (unsigned int)color;
   }

   finder.FindPoints(bnd, colorv);

   const Array<unsigned int> &code_out = finder.GetCode();

   // Setup ess_tdof_list_int
   for (int i = 0; i < number_boundary; i++)
   {
      int idx = ess_tdof_list[i];
      if (code_out[i] != 2) { ess_tdof_list_int.Append(idx); }
   }
}

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   int lim_meshes = 3; // should be greater than nmeshes
   Array <const char *> mesh_file_list(lim_meshes);
   Array <int> np_list(lim_meshes), rs_levels(lim_meshes),
         rp_levels(lim_meshes);
   mesh_file_list[0]         = "../../data/square-disc.mesh";
   mesh_file_list[1]         = "../../data/inline-quad.mesh";
   mesh_file_list[2]         = "../../data/inline-quad.mesh";
   int order                 = 2;
   bool visualization        = true;
   rs_levels                 = 0;
   rp_levels                 = 0;
   np_list                   = 0;
   double rel_tol            = 1.e-8;
   int visport               = 19916;
   int nmeshes               = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file_list[0], "-m1", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_file_list[1], "-m2", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_file_list[2], "-m3", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rs_levels[0], "-r1", "--refine-serial",
                  "Number of times to refine the mesh 1 uniformly in serial.");
   args.AddOption(&rs_levels[1], "-r2", "--refine-serial",
                  "Number of times to refine the mesh 2 uniformly in serial.");
   args.AddOption(&rs_levels[2], "-r3", "--refine-serial",
                  "Number of times to refine the mesh 3 uniformly in serial.");
   args.AddOption(&rp_levels[0], "-rp1", "--refine-parallel",
                  "Number of times to refine the mesh 1 uniformly in parallel.");
   args.AddOption(&rp_levels[1], "-rp2", "--refine-parallel",
                  "Number of times to refine the mesh 2 uniformly in parallel.");
   args.AddOption(&rp_levels[2], "-rp3", "--refine-parallel",
                  "Number of times to refine the mesh 3 uniformly in parallel.");
   args.AddOption(&np_list[0], "-np1", "--np1",
                  "number of MPI ranks for mesh 1");
   args.AddOption(&np_list[1], "-np2", "--np2",
                  "number of MPI ranks for mesh 2");
   args.AddOption(&np_list[2], "-np3", "--np3",
                  "number of MPI ranks for mesh 3");
   args.AddOption(&nmeshes, "-nm", "--nm",
                  "number of meshes");
   args.AddOption(&rel_tol, "-rt", "--relative tolerance",
                  "Tolerance for Schwarz iteration convergence criterion.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Check number of mpi ranks specified for each mesh. If individual mpi ranks
   // are not specified for all the meshes, set some default values.
   MFEM_VERIFY(num_procs >= nmeshes, "Not enough MPI ranks.");
   if (np_list.Sum() == 0)
   {
      int np_per_mesh = num_procs/nmeshes;
      for (int i = 0; i < nmeshes; i++)
      {
         np_list[i] = np_per_mesh;
      }
      np_list[nmeshes-1] += num_procs - nmeshes*np_per_mesh;
   }
   MFEM_VERIFY(np_list.Sum() == num_procs, " The individual mpi ranks for each"
               " of the meshes do not add up to the total ranks specified.");

   // Setup MPI communicator for each mesh by splitting MPI_COMM_WORLD.
   MPI_Comm *comml = new MPI_Comm;
   int color = 0;
   int npsum = 0;
   for (int i = 0; i < nmeshes; i++)
   {
      npsum += np_list[i];
      if (myid < npsum) { color = i; break; }
   }

   MPI_Comm_split(MPI_COMM_WORLD, color, myid, comml);
   int myidlocal, numproclocal;
   MPI_Comm_rank(*comml, &myidlocal);
   MPI_Comm_size(*comml, &numproclocal);

   // Read the mesh from the given mesh file. We can handle triangular,
   // quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   // the same code.
   Mesh *mesh = new Mesh(mesh_file_list[color], 1, 1);
   int dim = mesh->Dimension();

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   // largest number that gives a final mesh with no more than 50,000 elements.
   for (int lev = 0; lev < rs_levels[color]; lev++) { mesh->UniformRefinement(); }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine this
   // mesh further in parallel to increase the resolution. Once the parallel
   // mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh;
   pmesh = new ParMesh(*comml, *mesh);
   for (int l = 0; l < rp_levels[color]; l++)
   {
      pmesh->UniformRefinement();
   }
   delete mesh;

   // Define a finite element space on the mesh. Here we use continuous
   // Lagrange finite elements of the specified order. If order < 1, we
   // instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // Determine the list of true (i.e. conforming) essential boundary dofs.
   // In this example, the boundary conditions are defined by marking all
   // the boundary attributes from the mesh as essential (Dirichlet) and
   // converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // Set up the linear form b(.) which corresponds to the right-hand side of
   // the FEM linear system, which in this case is (1,phi_i) where phi_i are
   // the basis functions in the finite element fespace1.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // Define the solution vector x as a finite element grid function
   // corresponding to fespace1. Initialize x with initial guess of zero,
   // which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;
   x.SetTrueVector();

   // Setup FindPointsGSLIB and determine points on each mesh's boundary that
   // are interior to another mesh.
   pmesh->SetCurvature(order, false, dim, Ordering::byNODES);
   {
      Vector vxyz = *pmesh->GetNodes();

      // For the default mesh inputs, we need to rescale inline-quad.mesh such
      // that it does not cover the entire domain [0, 1]^2 and still has a
      // non-trivial overlap with the other mesh.
      if (strcmp(mesh_file_list[0], "../../data/square-disc.mesh") == 0 &&
          strcmp(mesh_file_list[1], "../../data/inline-quad.mesh") == 0 &&
          strcmp(mesh_file_list[2], "../../data/inline-quad.mesh") == 0 )
      {
         if (nmeshes == 2)
         {
            if (color == 1)   // rescale from [0, 1]^2 to [0.25, 0.75]^2
            {
               for (int i = 0; i < vxyz.Size(); i++)
               {
                  vxyz(i) = 0.5 + 0.5*(vxyz(i)-0.5);
               }
            }
         }
         else if (nmeshes == 3)
         {
            if (color == 1)
            {
               // rescale from [0, 1]^2 to [0.21, 0.61] in x and [0.25, 0.75] in y
               const int pts_cnt = vxyz.Size()/dim;
               for (int i = 0; i < pts_cnt; i++)
               {
                  vxyz(i) = 0.41 + 0.4*(vxyz(i)-0.5);
               }
               for (int i = 0; i < pts_cnt; i++)
               {
                  vxyz(i+pts_cnt) = 0.5 + 0.5*(vxyz(i+pts_cnt)-0.5);
               }
            }
            else if (color == 2)
            {
               // rescale from [0, 1]^2 to [0.4, 0.8] in x and [0.2, 0.8] in y
               const int pts_cnt = vxyz.Size()/dim;
               for (int i = 0; i < pts_cnt; i++)
               {
                  vxyz(i) = 0.6 + 0.4*(vxyz(i)-0.5);
               }
               for (int i = 0; i < pts_cnt; i++)
               {
                  vxyz(i+pts_cnt) = 0.5 + 0.6*(vxyz(i+pts_cnt)-0.5);
               }
            }
         }
      }
      pmesh->SetNodes(vxyz);
   }

   pmesh->GetNodes()->SetTrueVector();
   Vector vxyz = pmesh->GetNodes()->GetTrueVector();

   OversetFindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(*pmesh, color);

   Array<int> ess_tdof_list_int;
   GetInterdomainBoundaryPoints(finder, vxyz, color,
                                ess_tdof_list, ess_tdof_list_int, dim);

   // Use FindPointsGSLIB to interpolate the solution at interdomain boundary
   // points.
   const int number_boundary = ess_tdof_list_int.Size(),
             number_true = vxyz.Size()/dim;

   int number_boundary_g = number_boundary;
   MPI_Allreduce(&number_boundary, &number_boundary_g, 1, MPI_INT, MPI_SUM,
                 *comml);
   MFEM_VERIFY(number_boundary_g != 0, " Please use overlapping grids.");

   Array<unsigned int> colorv;
   colorv.SetSize(number_boundary);

   MPI_Barrier(MPI_COMM_WORLD);
   Vector bnd(number_boundary*dim);
   for (int i = 0; i < number_boundary; i++)
   {
      int idx = ess_tdof_list_int[i];
      for (int d = 0; d < dim; d++)
      {
         bnd(i+d*number_boundary) = vxyz(idx + d*number_true);
      }
      colorv[i] = (unsigned int)color;
   }
   Vector interp_vals1(number_boundary);
   finder.Interpolate(bnd, colorv, x, interp_vals1);

   // Set up the bilinear form a(.,.) on the finite element space corresponding
   // to the Laplacian operator -Delta, by adding a Diffusion integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // Assemble the bilinear form and the corresponding linear system,
   // applying any necessary transformations such as: eliminating boundary
   // conditions, applying conforming constraints for non-conforming AMR,
   // static condensation, etc.
   a->Assemble();

   delete b;

   // Use simultaneous Schwarz iterations to iteratively solve the PDE and
   // interpolate interdomain boundary data to impose Dirichlet boundary
   // conditions.

   int NiterSchwarz = 100;
   for (int schwarz = 0; schwarz < NiterSchwarz; schwarz++)
   {
      b = new ParLinearForm(fespace);
      b->AddDomainIntegrator(new DomainLFIntegrator(one));
      b->Assemble();

      OperatorPtr A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

      // Solve the linear system A X = B.
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      Solver *prec = NULL;
      prec = new HypreBoomerAMG;
      dynamic_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(-1);
      CGSolver cg(*comml);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(*prec);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete prec;

      // Recover the solution as a finite element grid function.
      a->RecoverFEMSolution(X, *b, x);

      // Interpolate boundary condition
      finder.Interpolate(x, interp_vals1);

      double dxmax = std::numeric_limits<float>::min();
      double xinf = x.Normlinf();
      double xinfg = xinf;
      MPI_Allreduce(&xinf, &xinfg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      x.SetTrueVector();
      Vector xt = x.GetTrueVector();
      for (int i = 0; i < number_boundary; i++)
      {
         int idx = ess_tdof_list_int[i];
         double dx = std::abs(xt(idx)-interp_vals1(i))/xinfg;
         if (dx > dxmax) { dxmax = dx; }
         xt(idx) = interp_vals1(i);
      }
      x.SetFromTrueDofs(xt);
      double dxmaxg = dxmax;
      MPI_Allreduce(&dxmax, &dxmaxg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      delete b;

      if (myid == 0)
      {
         std::cout << std::setprecision(8)    <<
                   "Iteration: "           << schwarz <<
                   ", Relative residual: " << dxmaxg   << endl;
      }

      if (dxmaxg < rel_tol) { break; }
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 15. Free the used memory.
   delete a;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;
   delete comml;


   return 0;
}
