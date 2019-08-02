//                       MFEM Example 12 - Parallel Version
//
// Compile with: make ex12p
//
// Sample runs:
//    mpirun -np 4 ex12p -m ../data/beam-tri.mesh
//    mpirun -np 4 ex12p -m ../data/beam-quad.mesh
//    mpirun -np 4 ex12p -m ../data/beam-tet.mesh -s 79 -n 10 -o 2 -elast
//    mpirun -np 4 ex12p -m ../data/beam-hex.mesh -s 3876
//    mpirun -np 4 ex12p -m ../data/beam-wedge.mesh -s 79
//    mpirun -np 4 ex12p -m ../data/beam-tri.mesh -s 3876 -o 2 -sys
//    mpirun -np 4 ex12p -m ../data/beam-quad.mesh -s 4526 -n 6 -o 3 -elast
//    mpirun -np 4 ex12p -m ../data/beam-quad-nurbs.mesh
//    mpirun -np 4 ex12p -m ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves the linear elasticity eigenvalue
//               problem for a multi-material cantilever beam.
//
//               Specifically, we compute a number of the lowest eigenmodes by
//               approximating the weak form of -div(sigma(u)) = lambda u where
//               sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder. The geometry of the domain is
//               assumed to be as follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |
//                    attribute 1  |    1     |    2     |
//                    (fixed)      +----------+----------+
//
//               The example highlights the use of the LOBPCG eigenvalue solver
//               together with the BoomerAMG preconditioner in HYPRE. Reusing a
//               single GLVis visualization window for multiple eigenfunctions
//               is also illustrated.
//
//               We recommend viewing examples 2 and 11 before viewing this
//               example.

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
   const char *mesh_file = "../data/beam-tri.mesh";
   int order = 1;
   int nev = 5;
   int seed = 75;
   bool visualization = 1;
   bool amg_elast = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&seed, "-s", "--seed",
                  "Random seed used to initialize LOBPCG.");
   args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                  "--amg-for-systems",
                  "Use the special AMG elasticity solver (GM/LN approaches), "
                  "or standard AMG for systems (unknown approach).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
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

   if (mesh->attributes.Max() < 2)
   {
      if (myid == 0)
         cerr << "\nInput mesh should have at least two materials!"
              << " (See schematic in ex12p.cpp)\n"
              << endl;
      MPI_Finalize();
      return 3;
   }

   // 4. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. We use the ordering by vector dimension (the last argument of
   //    the FiniteElementSpace constructor) which is expected in the systems
   //    version of BoomerAMG preconditioner. For NURBS meshes, we use the
   //    (degree elevated) NURBS space associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   const bool use_nodal_fespace = pmesh->NURBSext && !amg_elast;
   if (use_nodal_fespace)
   {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
   }
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   // 8. Set up the parallel bilinear forms a(.,.) and m(.,.) on the finite
   //    element space corresponding to the linear elasticity integrator with
   //    piece-wise constants coefficient lambda and mu, a simple mass matrix
   //    needed on the right hand side of the generalized eigenvalue problem
   //    below. The boundary conditions are implemented by marking only boundary
   //    attribute 1 as essential. We use special values on the diagonal to
   //    shift the Dirichlet eigenvalues out of the computational range. After
   //    serial/parallel assembly we extract the corresponding parallel matrices
   //    A and M.
   Vector lambda(pmesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
   if (myid == 0)
   {
      cout << "matrix ... " << flush;
   }
   a->Assemble();
   a->EliminateEssentialBCDiag(ess_bdr, 1.0);
   a->Finalize();

   ParBilinearForm *m = new ParBilinearForm(fespace);
   m->AddDomainIntegrator(new VectorMassIntegrator());
   m->Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   m->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
   m->Finalize();
   if (myid == 0)
   {
      cout << "done." << endl;
   }

   HypreParMatrix *A = a->ParallelAssemble();
   HypreParMatrix *M = m->ParallelAssemble();

   delete a;
   delete m;

   // 9. Define and configure the LOBPCG eigensolver and the BoomerAMG
   //    preconditioner for A to be used within the solver. Set the matrices
   //    which define the generalized eigenproblem A x = lambda M x.
   HypreBoomerAMG * amg = new HypreBoomerAMG(*A);
   amg->SetPrintLevel(0);
   if (amg_elast)
   {
      amg->SetElasticityOptions(fespace);
   }
   else
   {
      amg->SetSystemsOptions(dim);
   }

   HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
   lobpcg->SetNumModes(nev);
   lobpcg->SetRandomSeed(seed);
   lobpcg->SetPreconditioner(*amg);
   lobpcg->SetMaxIter(100);
   lobpcg->SetTol(1e-8);
   lobpcg->SetPrecondUsageMode(1);
   lobpcg->SetPrintLevel(1);
   lobpcg->SetMassMatrix(*M);
   lobpcg->SetOperator(*A);

   // 10. Compute the eigenmodes and extract the array of eigenvalues. Define a
   //     parallel grid function to represent each of the eigenmodes returned by
   //     the solver.
   Array<double> eigenvalues;
   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues);
   ParGridFunction x(fespace);

   // 11. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element. This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!use_nodal_fespace)
   {
      pmesh->SetNodalFESpace(fespace);
   }

   // 12. Save the refined mesh and the modes in parallel. This output can be
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

   // 13. Send the above data by socket to a GLVis server. Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);

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

   // 14. Free the used memory.
   delete lobpcg;
   delete amg;
   delete M;
   delete A;

   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete pmesh;

   MPI_Finalize();

   return 0;
}
