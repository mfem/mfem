//                                MFEM Example 4
//
// Compile with: make ex4
//
// Sample runs:  mpirun -np 4 ex4p ../data/square-disc.mesh
//               mpirun -np 4 ex4p ../data/star.mesh
//               mpirun -np 4 ex4p ../data/beam-tet.mesh
//               mpirun -np 4 ex4p ../data/beam-hex.mesh
//               mpirun -np 4 ex4p ../data/escher.mesh
//               mpirun -np 4 ex4p ../data/fichera.mesh
//               mpirun -np 4 ex4p ../data/fichera-q2.vtk
//               mpirun -np 4 ex4p ../data/fichera-q3.mesh
//               mpirun -np 4 ex4p ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex4p ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple 2D/3D H(div) diffusion
//               problem corresponding to the second order definite equation
//               -grad(alpha div F) + beta F = f with boundary condition F dot n
//               = <given normal field>. Here, we use a given exact solution F
//               and compute the corresponding r.h.s. f.  We discretize with
//               Raviart-Thomas finite elements.
//
//               The example demonstrates the use of H(div) finite element
//               spaces with the grad-div and H(div) vector finite element mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known.
//
//               We recommend viewing examples 1-3 before viewing this example.

#include <fstream>
#include "mfem.hpp"

// Exact solution, F, and r.h.s., f. See below for implementation.
void F_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);

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
      cout << "\nUsage: mpirun -np <np> ex4p <mesh_file>\n" << endl;
      MPI_Finalize();
      return 1;
   }

   // 2. Read the (serial) mesh from the given mesh file on all processors.
   //    In this 2D/3D example, we can handle triangular, quadrilateral,
   //    tetrahedral or hexahedral meshes with the same code.
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

   const int dim = mesh->Dimension();

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
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them (this is needed in the ADS solver below).
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }
   pmesh->ReorientTetMesh();

   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    use the lowest order Raviart-Thomas finite elements, but we can easily
   //    swich to higher-order spaces by changing the value of p.
   int p = 1;
   FiniteElementCollection *fec = new RT_FECollection(p-1, pmesh -> Dimension());
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   int size = fespace->GlobalTrueVSize();
   if (myid == 0)
      cout << "Number of unknowns: " << size << endl;

   // 6. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(dim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 7. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary faces will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient F(dim, F_exact);
   x.ProjectCoefficient(F);

   // 8. Set up the parallel bilinear form corresponding to the H(div) diffusion
   //    operator grad alpha div + beta I, by adding the div-div and the
   //    mass domain integrators and finally imposing non-homogeneous Dirichlet
   //    boundary conditions. The boundary conditions are implemented by
   //    marking all the boundary attributes from the mesh as essential
   //    (Dirichlet). After serial and parallel assembly we extract the
   //    parallel matrix A.
   Coefficient *alpha = new ConstantCoefficient(1.0);
   Coefficient *beta  = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DivDivIntegrator(*alpha));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));
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
   *X = 0.0;

   delete a;
   delete alpha;
   delete beta;
   delete b;

   // 10. Define and apply a parallel PCG solver for AX=B with the 2D AMS or the
   //     3D ADS preconditioners from hypre.
   HypreSolver *prec;
   if (dim == 2)
      prec = new HypreAMS(*A, fespace);
   else
      prec = new HypreADS(*A, fespace);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-10);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*prec);
   pcg->Mult(*B, *X);

   // 11. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 12. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(F);
      if (myid == 0)
         cout << "\n|| F_h - F ||_{L^2} = " << err << '\n' << endl;
   }

   // 13. Save the refined mesh and the solution in parallel. This output can
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

   // 14. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock << "solution\n";
   sol_sock.precision(8);
   pmesh->Print(sol_sock);
   x.Save(sol_sock);
   sol_sock.send();

   // 15. Free the used memory.
   delete pcg;
   delete prec;
   delete X;
   delete B;
   delete A;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}


// The exact solution
void F_exact(const Vector &p, Vector &F)
{
   double x,y,z;

   int dim = p.Size();

   x = p(0);
   y = p(1);
   if(dim == 3)
      z = p(2);

   F(0) = cos(M_PI*x)*sin(M_PI*y);
   F(1) = cos(M_PI*y)*sin(M_PI*x);
   if(dim == 3)
      F(2) = 0.0;
}

// The right hand side
void f_exact(const Vector &p, Vector &f)
{
   double x,y,z;

   int dim = p.Size();

   x = p(0);
   y = p(1);
   if(dim == 3)
      z = p(2);

   double temp = 1 + 2*M_PI*M_PI;

   f(0) = temp*cos(M_PI*x)*sin(M_PI*y);
   f(1) = temp*cos(M_PI*y)*sin(M_PI*x);
   if(dim == 3)
      f(2) = 0;
}
