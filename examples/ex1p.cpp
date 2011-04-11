//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p ../data/square-disc.mesh
//               mpirun -np 4 ex1p ../data/star.mesh
//               mpirun -np 4 ex1p ../data/escher.mesh
//               mpirun -np 4 ex1p ../data/fichera.mesh
//               mpirun -np 4 ex1p ../data/square-disc-p2.vtk
//               mpirun -np 4 ex1p ../data/square-disc-p3.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple linear finite element discretization of the Laplace
//               problem -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions.
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of boundary conditions on all boundary edges, and the optional
//               connection to the GLVis tool for visualization.

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
         cout << "\nUsage: mpirun -np <np> ex1p <mesh_file>\n" << endl;
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

   // 3. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }

   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    use linear finite elements.
   FiniteElementCollection *fec = new LinearFECollection;
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   // 6. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 7. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   (Vector &)x = 0.0;

   // 8. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential. After serial and
   //    parallel assembly we extract the corresponding parallel matrix A.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      Array<int> ess_dofs;
      fespace->GetEssentialVDofs(ess_bdr, ess_dofs);
      a->EliminateEssentialBCFromDofs(ess_dofs, x, *b);
   }
   a->Finalize();

   // 9. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //    b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();

   delete a;
   delete b;

   // 10. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
   HypreSolver *amg = new HypreBoomerAMG(*A);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(*B, *X);

   // 11. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 12. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs;
      if (myid == 0)
         mesh_ofs.open("refined.mesh");
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
      if (myid == 0)
         mesh_ofs.close();

      ofstream sol_ofs;
      if (myid == 0)
         sol_ofs.open("sol.gf");
      sol_ofs.precision(8);
      x.SaveAsOne(sol_ofs);
      if (myid == 0)
         sol_ofs.close();
   }

   // 13. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream *sol_sock;
   if (myid == 0)
   {
      sol_sock = new osockstream(visport, vishost);
      if (pmesh->Dimension() == 2)
         *sol_sock << "fem2d_gf_data\n";
      else
         *sol_sock << "fem3d_gf_data\n";
      sol_sock->precision(8);
   }
   pmesh->PrintAsOne(*sol_sock);
   x.SaveAsOne(*sol_sock);
   if (myid == 0)
   {
      sol_sock->send();
      delete sol_sock;
   }

   // 14. Free the used memory.
   delete pcg;
   delete amg;
   delete X;
   delete B;
   delete A;

   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
