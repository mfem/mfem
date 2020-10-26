//            MFEM test - mesh I/O using the parallel mesh format
//
// Compile with: make ex1p
//
// Sample runs:
//
//    The following sample runs alternate between the following two modes:
//
//    1. (serial mode) Read a serial mesh, refine it (before and after parallel
//       partitioning), solve a diffusion problem, and write the result, mesh
//       and solution, using a VisItDataCollection using the parallel format for
//       the mesh.
//    2. (parallel mode) Read a VisItDataCollection saved using the parallel
//       mesh format, then solve the same diffusion problem as above and compare
//       the result to the saved solution.
//
//    (This sequence is used to support testing with the script sample-runs.sh)
//
//       mpirun -np 4 ex1p -m ../../data/star.mesh
//       mpirun -np 4 ex1p
//       mpirun -np 4 ex1p -m ../../data/square-disc.mesh
//       mpirun -np 4 ex1p
//       mpirun -np 4 ex1p -m ../../data/star-mixed.mesh
//       mpirun -np 4 ex1p
//       mpirun -np 4 ex1p -m ../../data/escher.mesh
//       mpirun -np 4 ex1p
//       mpirun -np 4 ex1p -m ../../data/fichera.mesh
//       mpirun -np 4 ex1p
//       mpirun -np 4 ex1p -m ../../data/fichera-mixed.mesh
//       mpirun -np 4 ex1p

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
   const char *not_set = "(not set)";
   const char *mesh_file = not_set;
   const char *coll_name = "ex1p-dc";
   int order = 1;
   int serial_ref_levels = 1;
   int parallel_ref_levels = 2;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&coll_name, "-n", "--collection-name",
                  "Set the data collection name to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&serial_ref_levels, "-rs", "--refine-serial",
                  "Number of uniform refinements of the mesh before parallel"
                  " decomposition.");
   args.AddOption(&parallel_ref_levels, "-rp", "--refine-parallel",
                  "Number of uniform refinements to perform after parallel"
                  " decomposition.\n\t" "This is the number of levels used for"
                  " the convergence study.");
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

   const bool serial_mode = (mesh_file != not_set);

   ParMesh *pmesh;
   VisItDataCollection visit_dc(MPI_COMM_WORLD, coll_name);

   if (serial_mode)
   {
      // 3. Read the serial mesh on all processors, refine it in serial, then
      //    partition it across all processors and refine it in parallel.
      Mesh *mesh = new Mesh(mesh_file, 1, 1);

      for (int l = 0; l < serial_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }

      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      delete mesh;

      for (int l = 0; l < parallel_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
      visit_dc.SetMesh(pmesh);
   }
   else
   {
      // 4. Read the given data collection.
      visit_dc.Load();
      if (visit_dc.Error())
      {
         if (myid == 0)
         {
            cout << "Error loading data collection: " << coll_name << endl;
         }
         return 1;
      }
      pmesh = dynamic_cast<ParMesh*>(visit_dc.GetMesh());
      if (pmesh == NULL)
      {
         if (myid == 0)
         {
            cout << "The given data collection does not have a parallel mesh."
                 << endl;
         }
         return 2;
      }
   }
   int dim = pmesh->Dimension();

   // 5. Solve a simple diffusion problem on the parallel mesh.
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
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   ParGridFunction x(fespace);
   x = 0.0;

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   HypreBoomerAMG *amg = new HypreBoomerAMG;
   amg->SetPrintLevel(0);
   amg->SetOperator(A);
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(5);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(B, X);

   a->RecoverFEMSolution(X, *b, x);

   if (serial_mode)
   {
      // 6. Save the parallel mesh and the solution using the data collection.
      visit_dc.RegisterField("temperature", &x);
      visit_dc.SetFormat(DataCollection::PARALLEL_FORMAT);
      visit_dc.SetPrecision(16);
      if (myid == 0)
      {
         cout << "\nSaving data collection '" << coll_name << "' ..." << flush;
      }
      visit_dc.Save();
      if (myid == 0)
      {
         cout << " done.\n" << endl;
      }
   }
   else
   {
      ParGridFunction *saved_x = visit_dc.GetParField("temperature");
      if (!saved_x)
      {
         if (myid == 0)
         {
            cout << "The given data collection has no 'temperature' field."
                 << endl;
         }
      }
      else
      {
         ParGridFunction err(fespace);
         subtract(x, *saved_x, err);
         ConstantCoefficient zero(0.0);
         double err_norm = err.ComputeL2Error(zero);
         if (myid == 0)
         {
            cout << "\n|| x - x_saved ||_L2 = " << err_norm << '\n' << endl;
         }
      }
   }

   // 7. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 8. Free the used memory.
   delete pcg;
   delete amg;
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   if (serial_mode) { delete pmesh; }

   MPI_Finalize();

   return 0;
}
