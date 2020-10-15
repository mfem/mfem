//                       MFEM Brick Eigenvalue test
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double a_ = M_PI;
static double b_ = M_PI / sqrt(2.0);
static double c_ = M_PI / 2.0;

enum MeshType
{
   QUADRILATERAL = 1,
   TRIANGLE2A = 2,
   TRIANGLE2B = 3,
   TRIANGLE2C = 4,
   TRIANGLE4 = 5,
   MIXED2D = 6,
   HEXAHEDRON = 7,
   HEXAHEDRON2A = 8,
   HEXAHEDRON2B = 9,
   HEXAHEDRON2C = 10,
   HEXAHEDRON2D = 11,
   WEDGE2 = 12,
   TETRAHEDRA = 13,
   WEDGE4 = 14,
   MIXED3D = 15
};

Mesh * GetMesh(MeshType &type);


int eig(int i, int j, int k)
{
   return i * i + 2 * j * j + 4 * k * k;
}

int eigs[14] =
{
   3,6,9,11,12,17,18,
   7,10,13,15,16,19,21
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   MeshType mt = QUADRILATERAL;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 3;
   int nev = 7;
   int seed = 75;
   bool slu_solver  = false;
   bool sp_solver = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption((int*)&mt, "-m", "--mesh",
                  "Mesh type to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   // args.AddOption(&nev, "-n", "--num-eigs",
   //                "Number of desired eigenmodes.");
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
   Mesh *mesh = GetMesh(mt);
   int dim = mesh->Dimension();
   nev = dim + 1;

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
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
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
   Array<int> ess_bdr_tdofs;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);
   int bsize = ess_bdr_tdofs.Size();
   cout << "Number of BC DoFs " << bsize << endl;

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

   ParGridFunction x(fespace);

   Array<int> exact_eigs(&eigs[7 * (dim - 2)], 7);

   if ((dim == 2 && size < 50) || (dim == 3 && size < 120))
   {
      tic_toc.Clear();
      tic_toc.Start();

      DenseMatrix Ad(size);
      DenseMatrix Md(size);
      DenseMatrix vd(size);

      Ad = 0.0;
      Md = 0.0;
      Vector one(size);
      Vector done(size);
      one = 0.0;
      for (int i=0; i<size; i++)
      {
         one[i] = 1.0;
         A->Mult(one, done);
         for (int j=0; j<size; j++)
         {
            Ad(j, i) = done[j];
         }
         M->Mult(one, done);
         for (int j=0; j<size; j++)
         {
            Md(j, i) = done[j];
         }
         one[i] = 0.0;
      }
      for (int i=0; i<bsize; i++)
      {
         int ei = ess_bdr_tdofs[i];
         Ad(ei,ei) = 0.0;
         Md(ei,ei) = 1.0;
      }

      Array<double> eigenvalues(nev);
      Vector deigs(size);
      Ad.Eigenvalues(Md, deigs, vd);

      for (int i=bsize; i<min(size,bsize+nev); i++)
      {
         eigenvalues[i-bsize] = deigs[i];
         cout << "Eigenvalue lambda   " << deigs[i] << '\n';
      }

      tic_toc.Stop();
      cout << " done, " << tic_toc.RealTime() << "s." << endl;

      Vector err(nev); err = 0.0;
      for (int i=0; i<min(size-bsize,nev); i++)
      {
         err[i] = eigenvalues[i] / double(exact_eigs[i]) - 1.0;
      }
      cout << "Error: " << err.Norml1() << endl;

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
            Vector ev;
            vd.GetColumnReference(bsize+i, ev);
            x = ev;

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
                    << ", Lambda = " << eigenvalues[i]
                    << " (" << exact_eigs[i] << ")" << endl;
            }

            // convert eigenvector from HypreParVector to ParGridFunction
            Vector ev;
            vd.GetColumnReference(bsize+i, ev);
            x = ev;

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
   }
   else
   {
      tic_toc.Clear();
      tic_toc.Start();

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

      tic_toc.Stop();
      cout << " done, " << tic_toc.RealTime() << "s." << endl;

      Vector err(nev);
      for (int i=0; i<nev; i++)
      {
         err[i] = eigenvalues[i] / double(exact_eigs[i]) - 1.0;
         cout << i << " " << err[i] << endl;
      }
      cout << "Error: " << err.Norml1() << endl;

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
                    << ", Lambda = " << eigenvalues[i]
                    << " (" << exact_eigs[i] << ")" << endl;
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
      delete lobpcg;
      delete precond;
#if defined(MFEM_USE_SUPERLU) || defined(MFEM_USE_STRUMPACK)
      delete Arow;
#endif
   }

   tic_toc.Clear();
   tic_toc.Start();


   // 12. Free the used memory.
   delete a;
   delete m;
   delete M;
   delete A;
   delete fespace;
   if (order > 0)
   {
      delete fec;
   }
   delete pmesh;

   MPI_Finalize();

   return 0;
}

Mesh * GetMesh(MeshType &type)
{
   Mesh * mesh = NULL;
   double c[3];
   int    v[8];

   switch (type)
   {
      case QUADRILATERAL:
         mesh = new Mesh(2, 4, 1);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         mesh->AddQuad(v);
         break;
      case TRIANGLE2A:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 0;
         mesh->AddTri(v);
         break;
      case TRIANGLE2B:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 1; v[1] = 2; v[2] = 0;
         mesh->AddTri(v);
         v[0] = 3; v[1] = 0; v[2] = 2;
         mesh->AddTri(v);
         break;
      case TRIANGLE2C:
         mesh = new Mesh(2, 4, 2);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);

         v[0] = 2; v[1] = 0; v[2] = 1;
         mesh->AddTri(v);
         v[0] = 0; v[1] = 2; v[2] = 3;
         mesh->AddTri(v);
         break;
      case TRIANGLE4:
         mesh = new Mesh(2, 5, 4);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 1; v[1] = 2; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 4;
         mesh->AddTri(v);
         v[0] = 3; v[1] = 0; v[2] = 4;
         mesh->AddTri(v);
         break;
      case MIXED2D:
         mesh = new Mesh(2, 6, 4);
         c[0] = 0.0; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_;
         mesh->AddVertex(c);
         c[0] = 0.5 * b_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * b_; c[1] = 0.5 * b_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 5; v[3] = 4;
         mesh->AddQuad(v);
         v[0] = 1; v[1] = 2; v[2] = 5;
         mesh->AddTri(v);
         v[0] = 2; v[1] = 3; v[2] = 4; v[3] = 5;
         mesh->AddQuad(v);
         v[0] = 3; v[1] = 0; v[2] = 4;
         mesh->AddTri(v);
         break;
      case HEXAHEDRON:
         mesh = new Mesh(3, 8, 1);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         v[4] = 4; v[5] = 5; v[6] = 6; v[7] = 7;
         mesh->AddHex(v);
         break;
      case HEXAHEDRON2A:
      case HEXAHEDRON2B:
      case HEXAHEDRON2C:
      case HEXAHEDRON2D:
         mesh = new Mesh(3, 12, 2);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 5; v[2] = 11; v[3] = 6;
         v[4] = 1; v[5] = 4; v[6] = 10; v[7] = 7;
         mesh->AddHex(v);

         switch (type)
         {
            case HEXAHEDRON2A: // Face Orientation 1
               v[0] = 4; v[1] = 10; v[2] = 7; v[3] = 1;
               v[4] = 3; v[5] = 9; v[6] = 8; v[7] = 2;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2B: // Face Orientation 3
               v[0] = 10; v[1] = 7; v[2] = 1; v[3] = 4;
               v[4] = 9; v[5] = 8; v[6] = 2; v[7] = 3;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2C: // Face Orientation 5
               v[0] = 7; v[1] = 1; v[2] = 4; v[3] = 10;
               v[4] = 8; v[5] = 2; v[6] = 3; v[7] = 9;
               mesh->AddHex(v);
               break;
            case HEXAHEDRON2D: // Face Orientation 7
               v[0] = 1; v[1] = 4; v[2] = 10; v[3] = 7;
               v[4] = 2; v[5] = 3; v[6] = 9; v[7] = 8;
               mesh->AddHex(v);
               break;
            default:
               // Cannot happen
               break;
         }
         break;
      case WEDGE2:
         mesh = new Mesh(3, 8, 2);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 4; v[4] = 5; v[5] = 6;
         mesh->AddWedge(v);
         v[0] = 0; v[1] = 2; v[2] = 3; v[3] = 4; v[4] = 6; v[5] = 7;
         mesh->AddWedge(v);
         break;
      case TETRAHEDRA:
         mesh = new Mesh(3, 8, 5);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 2; v[2] = 7; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 6; v[1] = 7; v[2] = 2; v[3] = 5;
         mesh->AddTet(v);
         v[0] = 4; v[1] = 7; v[2] = 5; v[3] = 0;
         mesh->AddTet(v);
         v[0] = 1; v[1] = 0; v[2] = 5; v[3] = 2;
         mesh->AddTet(v);
         v[0] = 3; v[1] = 7; v[2] = 0; v[3] = 2;
         mesh->AddTet(v);
         break;
      case WEDGE4:
         mesh = new Mesh(3, 10, 4);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * a_; c[1] = 0.5 * b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 4; v[3] = 5; v[4] = 6; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 1; v[1] = 2; v[2] = 4; v[3] = 6; v[4] = 7; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 2; v[1] = 3; v[2] = 4; v[3] = 7; v[4] = 8; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 0; v[2] = 4; v[3] = 8; v[4] = 5; v[5] = 9;
         mesh->AddWedge(v);
         break;
      case MIXED3D:
         mesh = new Mesh(3, 12, 6);
         c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = 0.0;
         mesh->AddVertex(c);
         c[0] = 0.5 * c_; c[1] = 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * c_; c[1] = 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = a_ - 0.5 * c_; c[1] = b_ - 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.5 * c_; c[1] = b_ - 0.5 * c_; c[2] = 0.5 * c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = 0.0; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = a_; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);
         c[0] = 0.0; c[1] = b_; c[2] = c_;
         mesh->AddVertex(c);

         v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;
         v[4] = 4; v[5] = 5; v[6] = 6; v[7] = 7;
         mesh->AddHex(v);
         v[0] = 0; v[1] = 4; v[2] = 8; v[3] = 1; v[4] = 5; v[5] = 9;
         mesh->AddWedge(v);
         v[0] = 1; v[1] = 5; v[2] = 9; v[3] = 2; v[4] = 6; v[5] = 10;
         mesh->AddWedge(v);
         v[0] = 2; v[1] = 6; v[2] = 10; v[3] = 3; v[4] = 7; v[5] = 11;
         mesh->AddWedge(v);
         v[0] = 3; v[1] = 7; v[2] = 11; v[3] = 0; v[4] = 4; v[5] = 8;
         mesh->AddWedge(v);
         v[0] = 4; v[1] = 5; v[2] = 6; v[3] = 7;
         v[4] = 8; v[5] = 9; v[6] = 10; v[7] = 11;
         mesh->AddHex(v);
         break;
   }
   mesh->FinalizeTopology();

   if (mesh->Dimension() == 3)
   {
      Array<int> fcs;
      Array<int> cor;
      for (int i=0; i<mesh->GetNE(); i++)
      {
         mesh->GetElementFaces(i, fcs, cor);
         for (int j=0; j<fcs.Size(); j++)
         {
            cout << i << '\t' << j << '\t' << fcs[j] << '\t' << cor[j] << '\n';
         }
      }
   }

   return mesh;
}
