//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
//               The following are examples of using EntitySets to define
//               homogeneous Dirichlet boundary condition.  These examples
//               require a modified mesh file and a specialized version of
//               example 1 called "ex1p_es".
//               mpirun -np 4 ex1p_es -m ./fichera-set.mesh -bt 0 -bs Origin
//               mpirun -np 4 ex1p_es -m ./fichera-set.mesh -bt 1 -bs Axes
//               mpirun -np 4 ex1p_es -m ./fichera-set.mesh
//                                    -bt 1 -bs "Negative Axes"
//               mpirun -np 4 ex1p_es -m ./fichera-set.mesh
//                                    -bt 2 -bs "Interior Corner"
//               mpirun -np 4 ex1p_es -m ./fichera-set.mesh
//                                    -bt 2 -bs "Exterior Corner"
//               mpirun -np 4 ex1p_es -m ./fichera-set.mesh
//                                    -bt 3 -bs "Interior Corner"
//               mpirun -np 4 ex1p_es -m ./fichera-set.mesh
//                                    -bt 3 -bs "Exterior Corner"
//               mpirun -np 4 ex1p_es -m ./fichera-set.mesh -bt 3 -bs "Steps"
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

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
   const char *mesh_file = "./star-set.mesh";
   int order = 1;
   int rs = -1;
   int rp = 2;
   int ra = 0;
   int bt = EntitySets::INVALID;
   const char *bs = "Origin";
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&rs, "-rs", "--refine-serial",
                  "Number of serial refinement levels");
   args.AddOption(&rp, "-rp", "--refine-parallel",
                  "Number of parallel refinement levels");
   args.AddOption(&ra, "-ra", "--refine-adaptive",
                  "Number of adaptive refinement levels");
   args.AddOption(&bt, "-bt", "--bc-entity-type",
                  "");
   args.AddOption(&bs, "-bs", "--bc-entity-set-name",
                  "");
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

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels = ( rs >= 0 ) ? rs :
                       (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         if ( myid == 0 ) { cout << "Uniform refinement in serial..."; }
         mesh->UniformRefinement();
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if ( myid == 0 && rs > 0 ) { cout << "Done" << endl; }
   }
   if ( mesh->ent_sets )
   {
      cout << "mesh->ent_sets is non NULL" << endl;
      mesh->ent_sets->PrintSetInfo(cout);
   }
   else
   {
      cout << "mesh->ent_sets is NULL" << endl;
   }
   /*
     At this point we have a serial mesh containing an EntitySets
     object which stores the current node/edge/face/element indices
     for each entity in each set.  This data is duplicated on each MPI
     rank.
    */
   if ( ra > 0 )
   {
      cout << "calling EnsureNCMesh" << endl;
      mesh->EnsureNCMesh();
      cout << "back from EnsureNCMesh" << endl;
   }
   if ( mesh->ent_sets )
   {
      cout << "mesh->ent_sets is non NULL" << endl;
      mesh->ent_sets->PrintSetInfo(cout);
   }
   else
   {
      cout << "mesh->ent_sets is NULL" << endl;
   }
   /*
     We now have an NCEntitySets object which stores the node indices
     describing each enity in each node/edge/face set and the element
     indices for the elements in each element set.  This data is
     duplicated on each MPI rank.
    */

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   cout << "creating ParMesh from serial mesh" << endl;
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   cout << "done creating ParMesh from serial mesh" << endl;
   delete mesh;
   if ( pmesh->pent_sets )
   {
      cout << "pmesh->pent_sets is non NULL" << endl;
      pmesh->pent_sets->PrintSetInfo(cout);
   }
   else
   {
      cout << "pmesh->pent_sets is NULL" << endl;
   }
   /*
     We now have a ParEntitySets object which marshals the data stored
     in EntitySets objects.  The data has now been pruned so that each
     rank only contains indices of local entities.

     The NCEntitySets object remains unchanged...

     If we have an NC mesh a different path is taken and the
     EntitySets are ignored.

     1) ParNCMesh is created from NCMesh
       a) Creates a ParNCEntitySets object from ncmesh (every rank contains
          information to find every entity)
     2) ParNCMesh is pruned which involves renumbering elements and vertices
     3) ParMesh is initialized from ParNCMesh
     4) ParNCMesh::OnMeshUpdated is called
     5) Mesh::GenerateNCFaceInfo is called
    */
   {
      int par_ref_levels = rp;
      for (int l = 0; l < par_ref_levels; l++)
      {
         if ( myid == 0 ) { cout << "Uniform refinement in parallel..."; }
         pmesh->UniformRefinement();
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if ( myid == 0 && rs > 0 ) { cout << "Done" << endl; }
   }
   /*
     RandomRefinement will end up calling
     ParMesh::NonconformingRefinement which will create a new ParMesh
     object using the ParNCMesh object and then call
     ParMesh::OnMeshUpdated on this new mesh.
    */

   for (int l = 0; l < ra; l++)
   {
      pmesh->RandomRefinement(0.2);
   }
   if ( ra > 0 )
   {
      if ( pmesh->pent_sets )
      {
         cout << "pmesh->pent_sets is non NULL post random refinement" << endl;
         pmesh->pent_sets->PrintSetInfo(cout);
      }
      else
      {
         cout << "pmesh->pent_sets is NULL post random refinement" << endl;
      }
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

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if ( bt == EntitySets::INVALID )
   {
      if (pmesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 1;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
   }
   else
   {
      fespace->GetEssentialTrueDofs((EntitySets::EntityType)bt, bs,
                                    ess_tdof_list);
   }
   for (int i=0; i<num_procs; i++)
   {
      if (myid == i)
      {
         cout << "Number of Dirichlet dofs on proc " << i << ": "
              << ess_tdof_list.Size() << endl;
      }
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 10. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
   HypreSolver *amg = new HypreBoomerAMG(A);
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(B, X);

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
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

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 16. Free the used memory.
   delete pcg;
   delete amg;
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}
