//                       MFEM Example 26 - Parallel Version
//
// Compile with: make ex26p
//
// Sample runs:  mpirun -np 4 ex26p -m ../data/star.mesh
//               mpirun -np 4 ex26p -m ../data/fichera.mesh
//               mpirun -np 4 ex26p -m ../data/beam-hex.mesh
//
// Device sample runs:
//               mpirun -np 4 ex26p -d cuda
//               mpirun -np 4 ex26p -d occa-cuda
//               mpirun -np 4 ex26p -d raja-omp
//               mpirun -np 4 ex26p -d ceed-cpu
//               mpirun -np 4 ex26p -d ceed-cuda
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions
//               as in Example 1.
//
//               It highlights on the creation of a hierarchy of discretization
//               spaces with partial assembly and the construction of an
//               efficient multigrid preconditioner for the iterative solver.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Class for constructing a multigrid preconditioner for the diffusion operator.
// This example multigrid preconditioner class demonstrates the creation of the
// parallel diffusion bilinear forms and operators using partial assembly for
// all spaces except the coarsest one in the ParFiniteElementSpaceHierarchy.
// The multigrid uses a PCG solver preconditioned with AMG on the coarsest level
// and second order Chebyshev accelerated smoothers on the other levels.
class DiffusionMultigrid : public Multigrid
{
private:
   ConstantCoefficient one;
   HypreBoomerAMG* amg;

public:
   // Constructs a diffusion multigrid for the ParFiniteElementSpaceHierarchy
   // and the array of essential boundaries
   DiffusionMultigrid(ParFiniteElementSpaceHierarchy& fespaces,
                      Array<int>& ess_bdr)
      : Multigrid(fespaces), one(1.0)
   {
      ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0), ess_bdr);

      for (int level = 1; level < fespaces.GetNumLevels(); ++level)
      {
         ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), ess_bdr);
      }
   }

   virtual ~DiffusionMultigrid()
   {
      delete amg;
   }

private:
   void ConstructBilinearForm(ParFiniteElementSpace& fespace, Array<int>& ess_bdr,
                              bool partial_assembly)
   {
      ParBilinearForm* form = new ParBilinearForm(&fespace);
      if (partial_assembly)
      {
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
      form->Assemble();
      bfs.Append(form);

      essentialTrueDofs.Append(new Array<int>());
      fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
   }

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace& coarse_fespace,
                                         Array<int>& ess_bdr)
   {
      ConstructBilinearForm(coarse_fespace, ess_bdr, false);

      HypreParMatrix* hypreCoarseMat = new HypreParMatrix();
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), *hypreCoarseMat);

      amg = new HypreBoomerAMG(*hypreCoarseMat);
      amg->SetPrintLevel(-1);

      CGSolver* pcg = new CGSolver(MPI_COMM_WORLD);
      pcg->SetPrintLevel(-1);
      pcg->SetMaxIter(10);
      pcg->SetRelTol(sqrt(1e-4));
      pcg->SetAbsTol(0.0);
      pcg->SetOperator(*hypreCoarseMat);
      pcg->SetPreconditioner(*amg);

      AddLevel(hypreCoarseMat, pcg, true, true);
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace,
                                     Array<int>& ess_bdr)
   {
      ConstructBilinearForm(fespace, ess_bdr, true);

      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
      opr.SetOperatorOwner(false);

      Vector diag(fespace.GetTrueVSize());
      bfs.Last()->AssembleDiagonal(diag);

      Solver* smoother = new OperatorChebyshevSmoother(opr.Ptr(), diag,
                                                       *essentialTrueDofs.Last(), 2, fespace.GetParMesh()->GetComm());

      AddLevel(opr.Ptr(), smoother, true, true);
   }
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int geometric_refinements = 0;
   int order_refinements = 2;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&geometric_refinements, "-gr", "--geometric-refinements",
                  "Number of geometric refinements done prior to order refinements.");
   args.AddOption(&order_refinements, "-or", "--order-refinements",
                  "Number of order refinements. Finest level in the hierarchy has order 2^{or}.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

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
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space hierarchy on the parallel mesh.
   //    Here we use continuous Lagrange finite elements. We start with order 1
   //    on the coarse level and geometrically refine the spaces by the specified
   //    amount. Afterwards, we increase the order of the finite elements by a
   //    factor of 2 for each additional level.
   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace *coarse_fespace = new ParFiniteElementSpace(pmesh, fec);

   Array<FiniteElementCollection*> collections;
   collections.Append(fec);
   ParFiniteElementSpaceHierarchy* fespaces = new ParFiniteElementSpaceHierarchy(
      pmesh, coarse_fespace, true, true);
   for (int level = 0; level < geometric_refinements; ++level)
   {
      fespaces->AddUniformlyRefinedLevel();
   }
   for (int level = 0; level < order_refinements; ++level)
   {
      collections.Append(new H1_FECollection(std::pow(2, level+1), dim));
      fespaces->AddOrderRefinedLevel(collections.Last());
   }

   HYPRE_Int size = fespaces->GetFinestFESpace().GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(&fespaces->GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(&fespaces->GetFinestFESpace());
   x = 0.0;

   // 10. Create the multigrid operator using the previously created parallel
   //     FiniteElementSpaceHierarchy and additional boundary information. This
   //     operator is then used to create the MultigridSolver as preconditioner
   //     in the iterative solver.
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr = 1;
   }

   DiffusionMultigrid* M = new DiffusionMultigrid(*fespaces, ess_bdr);
   M->SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);

   OperatorPtr A;
   Vector X, B;
   M->FormFineLinearSystem(x, *b, A, X, B);

   // 11. Solve the linear system A X = B.
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(*M);
   cg.Mult(B, X);

   // 12. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   M->RecoverFineFEMSolution(X, *b, x);

   // 13. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      fespaces->GetFinestFESpace().GetParMesh()->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *fespaces->GetFinestFESpace().GetParMesh()
               << x << flush;
   }

   // 15. Free the used memory.
   delete M;
   delete b;
   delete fespaces;
   for (int level = 0; level < collections.Size(); ++level)
   {
      delete collections[level];
   }

   MPI_Finalize();

   return 0;
}
