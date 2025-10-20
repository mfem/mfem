//                       MFEM Example 2 - Parallel Version
//                              PETSc Modification
//
// Compile with: make ex2p
//
// Sample runs:
//    mpirun -np 4 ex2p -m ../../data/beam-quad.mesh --petscopts rc_ex2p
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)
//
//               The example demonstrates the use of high-order and NURBS vector
//               finite element spaces with the linear elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and vector coefficient objects. Static condensation is
//               also illustrated. The example also shows how to form a linear
//               system using a PETSc matrix and solve with a PETSc solver.
//
//               The example also show how to use the non-overlapping feature of
//               the ParBilinearForm class to obtain the linear operator in
//               a format suitable for the BDDC preconditioner in PETSc.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/beam-tri.mesh";
   int ser_ref_levels = -1;
   int par_ref_levels = 1;
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   bool amg_elast = 0;
   bool use_petsc = true;
   const char *petscrc_file = "";
   bool use_nonoverlapping = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                  "--amg-for-systems",
                  "Use the special AMG elasticity solver (GM/LN approaches), "
                  "or standard AMG for systems (unknown approach).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the linear system.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&use_nonoverlapping, "-nonoverlapping", "--nonoverlapping",
                  "-no-nonoverlapping", "--no-nonoverlapping",
                  "Use or not the block diagonal PETSc's matrix format "
                  "for non-overlapping domain decomposition.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 2b. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 2c. We initialize PETSc
   if (use_petsc) { MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL); }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
   {
      if (myid == 0)
         cerr << "\nInput mesh should have at least two materials and "
              << "two boundary attributes! (See schematic in ex2.cpp)\n"
              << endl;
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
      int ref_levels = ser_ref_levels >= 0 ? ser_ref_levels :
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
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined by
   //    marking only boundary attribute 1 from the mesh as essential and
   //    converting it to a list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system. In this case, b_i equals the
   //    boundary integral of f*phi_i where f represents a "pull down" force on
   //    the Neumann part of the boundary and phi_i are the basis functions in
   //    the finite element fespace. The force is defined by the object f, which
   //    is a vector of Coefficient objects. The fact that f is non-zero on
   //    boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(pmesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   if (myid == 0)
   {
      cout << "r.h.s. ... " << flush;
   }
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu.
   Vector lambda(pmesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (myid == 0) { cout << "matrix ... " << flush; }
   if (static_cond) { a->EnableStaticCondensation(); }
   // Here we want to try out block-size aware AMG solver in PETSc.
   // For that to work properly, we need a fully-compliant block-size
   // structure and we do not skip zeros when assembling.
   a->Assemble(use_petsc ? 0 : 1);

   Vector B, X;
   if (!use_petsc)
   {
      HypreParMatrix A;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      if (myid == 0)
      {
         cout << "done." << endl;
         cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
      }

      // 13. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG *amg = new HypreBoomerAMG(A);
      if (amg_elast && !a->StaticCondensationIsEnabled())
      {
         amg->SetElasticityOptions(fespace);
      }
      else
      {
         amg->SetSystemsOptions(dim);
      }
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-8);
      pcg->SetMaxIter(500);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);
      delete pcg;
      delete amg;
   }
   else
   {
      // 13b. Use PETSc to solve the linear system.
      //      Assemble a PETSc matrix, so that PETSc solvers can be used natively.
      PetscParMatrix A;
      a->SetOperatorType(use_nonoverlapping ?
                         Operator::PETSC_MATIS : Operator::PETSC_MATAIJ);
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      if (myid == 0)
      {
         cout << "done." << endl;
         cout << "Size of linear system: " << A.M() << endl;
      }
      // Tell PETSc the matrix has a block structure
      A.SetBlockSize(dim);

      // The preconditioner for the PCG solver can be specified in the
      // PETSc config file
      PetscPCGSolver *pcg = new PetscPCGSolver(A);
      PetscPreconditioner *prec = NULL;
      if (use_nonoverlapping) // Specialized BDDC construction
      {
         // Compute dofs belonging to the natural boundary
         Array<int> nat_tdof_list, nat_bdr(pmesh->bdr_attributes.Max());
         nat_bdr = 1;
         nat_bdr[0] = 0;
         fespace->GetEssentialTrueDofs(nat_bdr, nat_tdof_list);

         // Auxiliary class for BDDC customization
         PetscBDDCSolverParams opts;
         // Inform the solver about the finite element space
         opts.SetSpace(fespace);
         // Inform the solver about essential dofs
         opts.SetEssBdrDofs(&ess_tdof_list);
         // Inform the solver about natural dofs
         opts.SetNatBdrDofs(&nat_tdof_list);
         // Create a BDDC solver with parameters
         prec = new PetscBDDCSolver(A,opts);
         pcg->SetPreconditioner(*prec);
      }

      pcg->SetMaxIter(500);
      pcg->SetTol(1e-8);
      pcg->SetPrintLevel(2);
      pcg->Mult(B, X);
      delete pcg;
      delete prec;
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 15. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element.  This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!use_nodal_fespace)
   {
      pmesh->SetNodalFESpace(fespace);
   }

   // 16. Save in parallel the displaced mesh and the inverted solution (which
   //     gives the backward displacements to the original grid). This output
   //     can be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      GridFunction *nodes = pmesh->GetNodes();
      *nodes += x;
      x *= -1;

      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 17. Send the above data by socket to a GLVis server.  Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 18. Free the used memory.
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete pmesh;

   // We finalize PETSc
   if (use_petsc) { MFEMFinalizePetsc(); }

   return 0;
}
