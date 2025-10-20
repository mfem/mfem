//                       MFEM Example 11-cyl - Parallel Version
//
// Compile with: make ex11p-cyl
//
// Sample runs:  mpirun -np 4 ex11p-cyl
//               mpirun -np 4 ex11p-cyl -o 2
//               mpirun -np 4 ex11p-cyl -o 2 -e 0
//
// Description:  This example code demonstrates the use of MFEM to solve PDEs
//               on an axisymmetric domain.  The eigenvalue problem:
//                  -Delta u = lambda u
//               with homogeneous Dirichlet boundary conditions is solved on
//               a cylindrical domain by meshing only a rectangle in the
//               rho, z plane.  In cylindrical coordinates the weak form of
//               the eigenvalue problem is given by:
//                  (rho Grad(u), Grad(v)) = lambda (rho u, v)
//
//               We compute the five lowest eigenmodes by discretizing
//               the Laplacian and Mass operators using a FE space of the
//               specified order and compare to the known values.  Because the
//               eigenvalue spectrum of a domain is unique this provides a
//               reliable test that the axisymmetric domain is being faithfully
//               characterized.
//
//               The example highlights the use of specialized coefficients
//               with existing operators to mimic axisymmetric domains.  The
//               gradient of each eigenmode is also computed and displayed to
//               ilustrate that no special steps need to be taken to compute
//               gradients in this coordinate system.
//
//               We recommend viewing Example 11 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Zeros of Bessel function J_0
static double J0z[] = {2.40482555769577,
                       5.52007811028631,
                       8.65372791291101,
                       11.7915344390143
                      };

// Modes numbers in the rho and z directions for the first five eigenmodes
static int mode_nums[] = {0, 1,
                          1, 1,
                          0, 2,
                          1, 2,
                          2, 1
                         };

double rhoFunc(const Vector &x)
{
   return x[0];
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int nr = 1;
   int nz = 1;
   int el_type_flag = 1;
   Element::Type el_type;
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   int nev = 5;
   int seed = 75;
   bool slu_solver  = false;
   bool sp_solver = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of elements in z-direction.");
   args.AddOption(&nr, "-nr", "--num-elements-rho",
                  "Number of elements in radial direction.");
   args.AddOption(&el_type_flag, "-e", "--element-type",
                  "Element type: 0 - Triangle, 1 - Quadrilateral.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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

   // The output mesh could be quadrilaterals or triangles
   el_type = (el_type_flag == 0) ? Element::TRIANGLE : Element::QUADRILATERAL;
   if (el_type != Element::TRIANGLE && el_type != Element::QUADRILATERAL)
   {
      cout << "Unsupported element type" << endl;
      exit(1);
   }

   // 3. Prepare a rectangular mesh with the desired dimensions and element
   //    type.  Other 2D meshes could be used but then we couldn't check the
   //    eigenvalues.
   Mesh *mesh = new Mesh(nr, nz, el_type);
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
   //    use continuous Lagrange finite elements (H1) of the specified order.
   //    We also create a Nedelec space to represent the gradients of the modes.
   H1_FECollection fec_h1(order, dim);
   ND_FECollection fec_nd(order, dim);
   ParFiniteElementSpace fespace_h1(pmesh, &fec_h1);
   ParFiniteElementSpace fespace_nd(pmesh, &fec_nd);
   HYPRE_Int size = fespace_h1.GlobalTrueVSize();
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
   FunctionCoefficient rhoCoef(rhoFunc);
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;     // Homogeneous Dirichlet BCs everywhere except for
   ess_bdr[3] = 0;  // attribute 4 which is on the axis of symmetry.

   ParBilinearForm *a = new ParBilinearForm(&fespace_h1);
   a->AddDomainIntegrator(new DiffusionIntegrator(rhoCoef));
   a->Assemble();
   a->EliminateEssentialBCDiag(ess_bdr, 1.0);
   a->Finalize();

   ParBilinearForm *m = new ParBilinearForm(&fespace_h1);
   m->AddDomainIntegrator(new MassIntegrator(rhoCoef));
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
   //    the solver.  Also define a discrete gradient operator.
   Array<double> eigenvalues;
   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues);
   ParGridFunction x(&fespace_h1);
   ParGridFunction dx(&fespace_nd);

   ParDiscreteLinearOperator grad(&fespace_h1, &fespace_nd);
   grad.AddDomainInterpolator(new GradientInterpolator());
   grad.Assemble();

   if ( myid == 0 )
   {
      // Display the eigenvalues and their relative errors
      cout << "\nRelative error in eigenvalues:\n";
      for (int i=0; i<nev; i++)
      {
         double lambda =
            pow(J0z[mode_nums[2*i]], 2) +
            pow(M_PI * mode_nums[2*i+1], 2);
         cout << "Lambda " << i+1 << '/' << nev << " = " << eigenvalues[i]
              << ", rel err = " << fabs(eigenvalues[i] - lambda) / lambda
              << endl;
      }
      cout << endl;
   }

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
      socketstream grad_sock(vishost, visport);
      grad_sock.precision(8);

      for (int i=0; i<nev; i++)
      {
         if ( myid == 0 )
         {
            cout << "Eigenmode " << i+1 << '/' << nev
                 << ", Lambda = " << eigenvalues[i] << endl;
         }

         // convert eigenvector from HypreParVector to ParGridFunction
         x = lobpcg->GetEigenvector(i);

         grad.Mult(x, dx);

         mode_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << *pmesh << x << flush
                   << "window_title 'Eigenmode " << i+1 << '/' << nev
                   << ", Lambda = " << eigenvalues[i] << "'" << endl;

         grad_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << *pmesh << dx << flush
                   << "window_title 'Grad of Eigenmode " << i+1 << '/' << nev
                   << ", Lambda = " << eigenvalues[i] << "'"
                   << "window_geometry 400 0 400 350" << endl;

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
   delete pmesh;

   MPI_Finalize();

   return 0;
}
