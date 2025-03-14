//                       MFEM Example 28 - Parallel Version
//
// Compile with: make ex28p
//
// Sample runs:  ex28p
//               ex28p --visit-datafiles
//               ex28p --order 4
//               ex28p --penalty 1e+5
//
//               mpirun -np 4 ex28p
//               mpirun -np 4 ex28p --penalty 1e+5
//
// Description:  Demonstrates a sliding boundary condition in an elasticity
//               problem. A trapezoid, roughly as pictured below, is pushed
//               from the right into a rigid notch. Normal displacement is
//               restricted, but tangential movement is allowed, so the
//               trapezoid compresses into the notch.
//
//                                       /-------+
//               normal constrained --->/        | <--- boundary force (2)
//               boundary (4)          /---------+
//                                          ^
//                                          |
//                                normal constrained boundary (1)
//
//               This example demonstrates the use of the ConstrainedSolver
//               framework.
//
//               We recommend viewing Example 2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Return a mesh with a single element with vertices (0, 0), (1, 0), (1, 1),
// (offset, 1) to demonstrate boundary conditions on a surface that is not
// axis-aligned.
Mesh * build_trapezoid_mesh(real_t offset)
{
   MFEM_VERIFY(offset < 0.9, "offset is too large!");

   const int dimension = 2;
   const int nvt = 4; // vertices
   const int nbe = 4; // num boundary elements
   Mesh * mesh = new Mesh(dimension, nvt, 1, nbe);

   // vertices
   real_t vc[dimension];
   vc[0] = 0.0; vc[1] = 0.0;
   mesh->AddVertex(vc);
   vc[0] = 1.0; vc[1] = 0.0;
   mesh->AddVertex(vc);
   vc[0] = offset; vc[1] = 1.0;
   mesh->AddVertex(vc);
   vc[0] = 1.0; vc[1] = 1.0;
   mesh->AddVertex(vc);

   // element
   Array<int> vert(4);
   vert[0] = 0; vert[1] = 1; vert[2] = 3; vert[3] = 2;
   mesh->AddQuad(vert, 1);

   // boundary
   Array<int> sv(2);
   sv[0] = 0; sv[1] = 1;
   mesh->AddBdrSegment(sv, 1);
   sv[0] = 1; sv[1] = 3;
   mesh->AddBdrSegment(sv, 2);
   sv[0] = 2; sv[1] = 3;
   mesh->AddBdrSegment(sv, 3);
   sv[0] = 0; sv[1] = 2;
   mesh->AddBdrSegment(sv, 4);

   mesh->FinalizeQuadMesh(1, 0, true);

   return mesh;
}

int main(int argc, char *argv[])
{
#ifdef HYPRE_USING_GPU
   cout << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this example\n"
        << "is NOT supported with the GPU version of hypre.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   int order = 1;
   bool visualization = 1;
   bool reorder_space = false;
   real_t offset = 0.3;
   bool visit = false;
   real_t penalty = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&reorder_space, "-nodes", "--by-nodes", "-vdim", "--by-vdim",
                  "Use byNODES ordering of vector space instead of byVDIM");
   args.AddOption(&offset, "--offset", "--offset",
                  "How much to offset the trapezoid.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&penalty, "-p", "--penalty",
                  "Penalty parameter; 0 means use elimination solver.");
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

   // 3. Build a trapezoidal mesh with a single quadrilateral element, where
   //    'offset' determines how far off it is from a rectangle.
   Mesh *mesh = build_trapezoid_mesh(offset);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
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

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
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

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. We use the ordering by vector dimension (the last argument of
   //    the FiniteElementSpace constructor) which is expected in the systems
   //    version of BoomerAMG preconditioner. For NURBS meshes, we use the
   //    (degree elevated) NURBS space associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   const bool use_nodal_fespace = pmesh->NURBSext;
   if (use_nodal_fespace)
   {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      if (reorder_space)
      {
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byNODES);
      }
      else
      {
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
      }
   }
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling matrix and r.h.s... " << flush;
   }

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, there are no essential boundary
   //    conditions in the usual sense, but we leave the machinery here for
   //    users to modify if they wish.
   Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 8. Set up the parallel linear form b(.) which corresponds to the
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

   // 9. Put a leftward force on the right side of the trapezoid
   {
      Vector push_force(pmesh->bdr_attributes.Max());
      push_force = 0.0;
      push_force(1) = -5.0e-2; // index 1 attribute 2
      f.Set(0, new PWConstCoefficient(push_force));
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu. We use constant coefficients,
   //     but see ex2 for how to set up piecewise constant coefficients based
   //     on attribute.
   Vector lambda(pmesh->attributes.Max());
   lambda = 1.0;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   mu = 1.0;
   PWConstCoefficient mu_func(mu);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, etc.
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   if (myid == 0)
   {
      cout << "done." << endl;
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 13. Set up constraint matrix to constrain normal displacement (but
   //     allow tangential displacement) on specified boundaries.
   Array<int> constraint_atts(2);
   constraint_atts[0] = 1;  // attribute 1 bottom
   constraint_atts[1] = 4;  // attribute 4 left side
   Array<int> constraint_rowstarts;
   SparseMatrix* local_constraints =
      ParBuildNormalConstraints(*fespace, constraint_atts,
                                constraint_rowstarts);

   // 14. Define and apply a parallel PCG solver for the constrained system
   //     where the normal boundary constraints have been separately eliminated
   //     from the system.
   ConstrainedSolver * solver;
   if (penalty == 0.0)
   {
      solver = new EliminationCGSolver(A, *local_constraints,
                                       constraint_rowstarts, dim,
                                       reorder_space);
   }
   else
   {
      solver = new PenaltyPCGSolver(A, *local_constraints, penalty,
                                    dim, reorder_space);
   }

   solver->SetRelTol(1e-8);
   solver->SetMaxIter(500);
   solver->SetPrintLevel(1);
   solver->Mult(B, X);

   // 15. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 16. For non-NURBS meshes, make the mesh curved based on the finite element
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

   GridFunction *nodes = pmesh->GetNodes();
   *nodes += x;

   // 17. Save the refined mesh and the solution in VisIt format.
   if (visit)
   {
      VisItDataCollection visit_dc(MPI_COMM_WORLD, "ex28p", pmesh);
      visit_dc.SetLevelsOfDetail(4);
      visit_dc.RegisterField("displacement", &x);
      visit_dc.Save();
   }

   // 18. Save in parallel the displaced mesh and the inverted solution (which
   //     gives the backward displacements to the original grid). This output
   //     can be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      x *= -1; // sign convention for GLVis displacements

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

   // 19. Send the above data by socket to a GLVis server.  Use the "n" and "b"
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

   // 20. Free the used memory.
   delete local_constraints;
   delete solver;
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete pmesh;

   // HYPRE_Finalize();

   return 0;
}
