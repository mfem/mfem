//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "mpi.h"
#include "dmumps_c.h"
// #define JOB_INIT -1
// #define JOB_END -2
#define USE_COMM_WORLD -987654
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
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels = 5;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // // 13. Solve the linear system A X = B.
   // //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   // //     * With partial assembly, use Jacobi smoothing, for now.
   // HypreBoomerAMG *prec = new HypreBoomerAMG;
   // CGSolver cg(MPI_COMM_WORLD);
   // cg.SetRelTol(1e-12);
   // cg.SetMaxIter(2000);
   // cg.SetPrintLevel(1);
   // if (prec) { cg.SetPreconditioner(*prec); }
   // cg.SetOperator(A);
   // cg.Mult(B, X);
   // delete prec;

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   // a.RecoverFEMSolution(X, b, x);

   // A.Threshold();


   hypre_ParCSRMatrix * parcsr_op = (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(A);

   hypre_CSRMatrix *csr_op = hypre_MergeDiagAndOffd(parcsr_op);
#if MFEM_HYPRE_VERSION >= 21600
   hypre_CSRMatrixBigJtoJ(csr_op);
#endif

   int * Iptr = csr_op->i;
   int * Jptr = csr_op->j;
   double * data = csr_op->data;
   
   int nnz = csr_op->num_nonzeros;
   int I[nnz];
   int J[nnz];

   int n_loc = csr_op->num_rows;

   int k = 0;
   for (int i = 0; i<n_loc; i++)
   {
      for (int j = Iptr[i]; j<Iptr[i+1]; j++)
      {
         // "tdof offsets" can be determined by parcsr_op -> rowstarts
         I[k] = i+fespace.GetMyTDofOffset()+1;
         J[k] = Jptr[k]+1;
         k++;
      }
   }


   DMUMPS_STRUC_C id;
   MUMPS_INT ierr;
   int error = 0;
   /* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
   id.comm_fortran=USE_COMM_WORLD;
   
   id.job=-1; id.par=1; id.sym=0; 
   // Mumps init
   dmumps_c(&id);

   #define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
   #define INFO(I) info[(I)-1] /* macro s.t. indices match documentation */
  /* No outputs */
//   id.ICNTL(1)=-1; id.ICNTL(2)=-1; id.ICNTL(3)=-1; id.ICNTL(4)=0;
   id.ICNTL(5) = 0;
   id.ICNTL(18) = 3;
   id.ICNTL(20) = 10; // distributed rhs
   id.ICNTL(21) = 1; // distributed solution

   // Global number of rows/colums on the host
   if (myid == 0) {id.n = A.GetGlobalNumRows();} 

   // on all procs
   id.nnz_loc = nnz;
   id.irn_loc = I;
   id.jcn_loc = J;
   id.a_loc = data;

   id.job=1;
   dmumps_c(&id);

   id.job=2;
   dmumps_c(&id);

   // local to global row map
   int *irhs_loc = new int[n_loc];
   for (int i = 0; i < n_loc; i++)
   {
      irhs_loc[i] = parcsr_op->first_row_index + i + 1; // 1-based indexing offset
   }
   id.rhs_loc = B.GetData();
   id.nloc_rhs = n_loc;
   id.lrhs_loc = n_loc;
   id.irhs_loc = irhs_loc;

   int num_pivots = id.INFO(23);
   printf("num_pivots = %d\n", num_pivots);

   id.lsol_loc = num_pivots;
   Vector Xaux(X);
   id.sol_loc = Xaux.GetData();
   int *isol_loc = new int[num_pivots];
   id.isol_loc = isol_loc;

   id.job=3;
   dmumps_c(&id);

   // @TODO
   // On exit from the solve phase, ISOL loc(i) contains the index of the
   // variables for which the solution (in SOL loc) is available on the local
   // processor.

   // printf("\nordering rhs rank %d\n", myid);
   // for (int i = 0; i < n_loc; i++)
   // {
   //    printf("%d ", id.irhs_loc[i]);
   // }
   // printf("\n");

   // MPI_Barrier(MPI_COMM_WORLD);

   // printf("\nordering sol rank %d\n", myid);
   // for (int i = 0; i < num_pivots; i++)
   // {
   //    printf("%d ", id.isol_loc[i]);
   // }
   // printf("\n");

   // MPI_Barrier(MPI_COMM_WORLD);

   for (int i = 0; i < id.lsol_loc; i++)
   {
      X(isol_loc[i] - 1) = Xaux(i);
   }

   id.job=-2; // mumps finalize
   dmumps_c(&id);

   a.RecoverFEMSolution(X, b, x);

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   // 17. Free the used memory.
   delete fec;
   MPI_Finalize();

   return 0;
}
