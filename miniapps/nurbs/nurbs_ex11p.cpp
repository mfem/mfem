//                    MFEM Example 11 - Parallel NURBS Version
//
// Compile with: make nurbs_ex11p
//
// Sample runs:  mpirun -np 4 nurbs_ex11p -m ../../data/square-disc.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/star.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/escher.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/fichera.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 nurbs_ex11p -m ../../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 nurbs_ex11p -m ../../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 nurbs_ex11p -m ../../data/disc-nurbs.mesh -o -1 -n 20
//               mpirun -np 4 nurbs_ex11p -m ../../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 nurbs_ex11p -m ../../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 nurbs_ex11p -m ../../data/star-surf.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/square-disc-surf.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/inline-segment.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/amr-quad.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/amr-hex.mesh
//               mpirun -np 4 nurbs_ex11p -m ../../data/mobius-strip.mesh -n 8
//               mpirun -np 4 nurbs_ex11p -m ../../data/klein-bottle.mesh -n 10
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               eigenvalue problem -Delta u = lambda u with homogeneous
//               Dirichlet boundary conditions.
//
//               We compute a number of the lowest eigenmodes by discretizing
//               the Laplacian and Mass operators using a FE space of the
//               specified order, or an isoparametric/isogeometric space if
//               order < 1 (quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of the LOBPCG eigenvalue solver
//               together with the BoomerAMG preconditioner in HYPRE, as well as
//               optionally the SuperLU or STRUMPACK parallel direct solvers.
//               Reusing a single GLVis visualization window for multiple
//               eigenfunctions is also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   Array<int> order(1);
   order[0] = 0;
   int nev = 5;
   int seed = 75;
   bool slu_solver  = false;
   bool sp_solver = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&seed, "-s", "--seed",
                  "Random seed used to initialize LOBPCG.");
#ifdef MFEM_USE_SUPERLU
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
#endif
#ifdef MFEM_USE_STRUMPACK
   args.AddOption(&sp_solver, "-sp", "--strumpack", "-no-sp",
                  "--no-strumpack", "Use the STRUMPACK Solver.");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (slu_solver && sp_solver)
   {
      if (myid == 0)
         cout << "WARNING: Both SuperLU and STRUMPACK have been selected,"
              << " please choose either one." << endl
              << "         Defaulting to SuperLU." << endl;
      sp_solver = false;
   }
   // The command line options are also passed to the STRUMPACK
   // solver. So do not exit if some options are not recognized.
   if (!sp_solver)
   {
      if (!args.Good())
      {
         if (myid == 0)
         {
            args.PrintUsage(cout);
         }
         MPI_Finalize();
         return 1;
      }
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement (2 by default, or
   //    specified on the command line with -rs).
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution (1 time by
   //    default, or specified on the command line with -rp). Once the parallel
   //    mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   NURBSExtension *NURBSext = NULL;
   int own_fec = 0;

   if (order[0] == 0) // Isoparametric
   {
      if (pmesh->GetNodes())
      {
         fec = pmesh->GetNodes()->OwnFEC();
         own_fec = 0;
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
      else
      {
         cout <<"Mesh does not have FEs --> Assume order 1.\n";
         fec = new H1_FECollection(1, dim);
         own_fec = 1;
      }
   }
   else if (pmesh->NURBSext && (order[0] > 0) )  // Subparametric NURBS
   {
      fec = new NURBSFECollection(order[0]);
      own_fec = 1;
      int nkv = pmesh->NURBSext->GetNKV();

      if (order.Size() == 1)
      {
         int tmp = order[0];
         order.SetSize(nkv);
         order = tmp;
      }
      if (order.Size() != nkv ) { mfem_error("Wrong number of orders set."); }
      NURBSext = new NURBSExtension(pmesh->NURBSext, order);
   }
   else
   {
      if (order.Size() > 1) { cout <<"Wrong number of orders set, needs one.\n"; }
      fec = new H1_FECollection(abs(order[0]), dim);
      own_fec = 1;
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh,NURBSext,fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 7. Set up the parallel bilinear forms a(.,.) and m(.,.) on the finite
   //    element space. The first corresponds to the Laplacian operator -Delta,
   //    while the second is a simple mass matrix needed on the right hand side
   //    of the generalized eigenvalue problem below. The boundary conditions
   //    are implemented by elimination with special values on the diagonal to
   //    shift the Dirichlet eigenvalues out of the computational range. After
   //    serial and parallel assembly we extract the corresponding parallel
   //    matrices A and M.
   ConstantCoefficient one(1.0);
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
   }

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   if (pmesh->bdr_attributes.Size() == 0)
   {
      // Add a mass term if the mesh has no boundary, e.g. periodic mesh or
      // closed surface.
      a->AddDomainIntegrator(new MassIntegrator(one));
   }
   a->Assemble();
   a->EliminateEssentialBCDiag(ess_bdr, 1.0);
   a->Finalize();

   ParBilinearForm *m = new ParBilinearForm(fespace);
   m->AddDomainIntegrator(new MassIntegrator(one));
   m->Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   m->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
   m->Finalize();

   HypreParMatrix *A = a->ParallelAssemble();
   HypreParMatrix *M = m->ParallelAssemble();

#if defined(MFEM_USE_SUPERLU) || defined(MFEM_USE_STRUMPACK)
   Operator * Arow = NULL;
#ifdef MFEM_USE_SUPERLU
   if (slu_solver)
   {
      Arow = new SuperLURowLocMatrix(*A);
   }
#endif
#ifdef MFEM_USE_STRUMPACK
   if (sp_solver)
   {
      Arow = new STRUMPACKRowLocMatrix(*A);
   }
#endif
#endif

   delete a;
   delete m;

   // 8. Define and configure the LOBPCG eigensolver and the BoomerAMG
   //    preconditioner for A to be used within the solver. Set the matrices
   //    which define the generalized eigenproblem A x = lambda M x.
   Solver * precond = NULL;
   if (!slu_solver && !sp_solver)
   {
      HypreBoomerAMG * amg = new HypreBoomerAMG(*A);
      amg->SetPrintLevel(0);
      precond = amg;
   }
   else
   {
#ifdef MFEM_USE_SUPERLU
      if (slu_solver)
      {
         SuperLUSolver * superlu = new SuperLUSolver(MPI_COMM_WORLD);
         superlu->SetPrintStatistics(false);
         superlu->SetSymmetricPattern(true);
         superlu->SetColumnPermutation(superlu::PARMETIS);
         superlu->SetOperator(*Arow);
         precond = superlu;
      }
#endif
#ifdef MFEM_USE_STRUMPACK
      if (sp_solver)
      {
         STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, MPI_COMM_WORLD);
         strumpack->SetPrintFactorStatistics(true);
         strumpack->SetPrintSolveStatistics(false);
         strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
         strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
         strumpack->DisableMatching();
         strumpack->SetOperator(*Arow);
         strumpack->SetFromCommandLine();
         precond = strumpack;
      }
#endif
   }


   HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
   lobpcg->SetNumModes(nev);
   lobpcg->SetRandomSeed(seed);
   lobpcg->SetPreconditioner(*precond);
   lobpcg->SetMaxIter(200);
   lobpcg->SetTol(1e-8);
   lobpcg->SetPrecondUsageMode(1);
   lobpcg->SetPrintLevel(1);
   lobpcg->SetMassMatrix(*M);
   lobpcg->SetOperator(*A);

   // 9. Compute the eigenmodes and extract the array of eigenvalues. Define a
   //    parallel grid function to represent each of the eigenmodes returned by
   //    the solver.
   Array<double> eigenvalues;
   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues);
   ParGridFunction x(fespace);

   // 10. Save the refined mesh and the modes in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g mode".
   {
      ostringstream mesh_name, mode_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      for (int i=0; i<nev; i++)
      {
         // convert eigenvector from HypreParVector to ParGridFunction
         x = lobpcg->GetEigenvector(i);

         mode_name << "mode_" << setfill('0') << setw(2) << i << "."
                   << setfill('0') << setw(6) << myid;

         ofstream mode_ofs(mode_name.str().c_str());
         mode_ofs.precision(8);
         x.Save(mode_ofs);
         mode_name.str("");
      }
   }

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);

      for (int i=0; i<nev; i++)
      {
         if ( myid == 0 )
         {
            cout << "Eigenmode " << i+1 << '/' << nev
                 << ", Lambda = " << eigenvalues[i] << endl;
         }

         // convert eigenvector from HypreParVector to ParGridFunction
         x = lobpcg->GetEigenvector(i);

         mode_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << *pmesh << x << flush
                   << "window_title 'Eigenmode " << i+1 << '/' << nev
                   << ", Lambda = " << eigenvalues[i] << "'" << endl;

         char c;
         if (myid == 0)
         {
            cout << "press (q)uit or (c)ontinue --> " << flush;
            cin >> c;
         }
         MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

         if (c != 'c')
         {
            break;
         }
      }
      mode_sock.close();
   }

   // 12. Free the used memory.
   delete lobpcg;
   delete precond;
   delete M;
   delete A;
#if defined(MFEM_USE_SUPERLU) || defined(MFEM_USE_STRUMPACK)
   delete Arow;
#endif

   delete fespace;
   if (own_fec)
   {
      delete fec;
   }
   delete pmesh;

   MPI_Finalize();

   return 0;
}
