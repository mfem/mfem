//                       MFEM Example 2 - Parallel Version
//
// Compile with: make ex2p
//
// Sample runs:  mpirun -np 4 ex2p ../data/beam-tri.mesh
//               mpirun -np 4 ex2p ../data/beam-quad.mesh
//               mpirun -np 4 ex2p ../data/beam-tet.mesh
//               mpirun -np 4 ex2p ../data/beam-hex.mesh
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material Cantilever beam.
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
//               The example demonstrates the use of (high-order) vector finite
//               element spaces with the linear elasticity bilinear form, meshes
//               with curved elements, and the definition of piece-wise constant
//               and vector coefficient objects.
//
//               We recommend viewing example 1 before viewing this example.

#include <fstream>
#include "mfem.hpp"

int main (int argc, char *argv[])
{
   int num_procs, myid;

   // 1. Initialize MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   Mesh *mesh;

   if (argc == 1)
   {
      if (myid == 0)
         cout << "\nUsage: mpirun -np <np> ex2p <mesh_file>\n" << endl;
      MPI_Finalize();
      return 1;
   }

   // 2. Read the (serial) mesh from the given mesh file on all processors.
   //    We can handle triangular, quadrilateral, tetrahedral or hexahedral
   //    elements with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      if (myid == 0)
         cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   int dim = mesh->Dimension();

   // 3. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }

   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. We use the ordering by vector dimension (the last argument of
   //    the FiniteElementSpace constructor) which is expected in the systems
   //    version of BoomerAMG preconditioner.
   FiniteElementCollection *fec;
   int fec_type;
   if (myid == 0)
   {
      cout << "Choose the finite element space:\n"
           << " 1) Linear\n"
           << " 2) Quadratic\n"
           << " 3) Cubic\n"
           << " ---> ";
      cin >> fec_type;
   }
   MPI_Bcast(&fec_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
   switch (fec_type)
   {
   default:
   case 1:
      fec = new LinearFECollection; break;
   case 2:
      fec = new QuadraticFECollection; break;
   case 3:
      fec = new CubicFECollection; break;
   }
   if (myid == 0)
      cout << "Assembling: " << flush;
   ParFiniteElementSpace *fespace =
      new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);

   // 6. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system. In this case, b_i equals the
   //    boundary integral of f*phi_i where f represents a "pull down" force on
   //    the Neumann part of the boundary and phi_i are the basis functions in
   //    the finite element fespace. The force is defined by the object f, which
   //    is a vector of Coefficient objects. The fact that f is non-zero on
   //    boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
      f.Set(i, new ConstantCoefficient(0.0));
   {
      Vector pull_force(pmesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorBoundaryLFIntegrator(f));
   if (myid == 0)
      cout << "r.h.s. ... " << flush;
   b->Assemble();

   // 7. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   (Vector &)x = 0.0;

   // 8. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu. The boundary conditions are
   //    implemented by marking only boundary attribute 1 as essential. After
   //    serial/parallel assembly we extract the corresponding parallel matrix.
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
   if (myid == 0)
      cout << "matrix ... " << flush;
   a->Assemble();
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[0] = 1;
      Array<int> ess_dofs;
      fespace->GetEssentialVDofs(ess_bdr, ess_dofs);
      a->EliminateEssentialBCFromDofs(ess_dofs, x, *b);
   }
   a->Finalize();
   if (myid == 0)
      cout << "done." << endl;

   // 9. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //    b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();

   delete a;
   delete b;

   // 10. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
   HypreBoomerAMG *amg = new HypreBoomerAMG(*A);
   amg->SetSystemsOptions(dim);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-8);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(*B, *X);

   // 11. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 12. Make the mesh curved based on the finite element space. This means
   //     that we define the mesh elements through a fespace-based
   //     transformation of the reference element.  This allows us to save the
   //     displaced mesh as a curved mesh when using high-order finite element
   //     displacement field. We assume that the initial mesh (read from the
   //     file) is not higher order curved mesh compared to the FE space chosen
   //     from the menu.
   pmesh->SetNodalFESpace(fespace);

   // 13. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      GridFunction *nodes = pmesh->GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs;
      if (myid == 0)
         mesh_ofs.open("displaced.mesh");
      pmesh->PrintAsOne(mesh_ofs);
      if (myid == 0)
         mesh_ofs.close();

      ofstream sol_ofs;
      if (myid == 0)
         sol_ofs.open("sol.gf");
      x.SaveAsOne(sol_ofs);
      if (myid == 0)
         sol_ofs.close();
   }

   // 14. (Optional) Send the above data by socket to a GLVis server. Note that
   //     we use "vfem" instead of "fem" in the initial string, to indicate
   //     vector grid function. Use the "n" and "b" keys in GLVis to visualize
   //     the displacements.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream *sol_sock;
   if (myid == 0)
   {
      sol_sock = new osockstream(visport, vishost);
      if (dim == 2)
         *sol_sock << "vfem2d_gf_data\n";
      else
         *sol_sock << "vfem3d_gf_data\n";
   }
   pmesh->PrintAsOne(*sol_sock);
   x.SaveAsOne(*sol_sock);
   if (myid == 0)
   {
      sol_sock->send();
      delete sol_sock;
   }

   // 15. Free the used memory.
   delete pcg;
   delete amg;
   delete X;
   delete B;
   delete A;

   delete fespace;
   delete fec;

   MPI_Finalize();

   return 0;
}
