//                       MFEM Example 0 - Parallel Version
//
// Compile with: make ex0p
//
// Sample runs:  mpirun -np 4 ex0p
//               mpirun -np 4 ex0p -m ../data/fichera.mesh
//               mpirun -np 4 ex0p -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic parallel usage of
//              MFEM to define a simple finite element discretization of the
//              Laplace problem -Delta u = 1 with zero Dirichlet boundary
//              conditions. General 2D/3D serial mesh files and finite element
//              polynomial degrees can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "linear_solver.hpp"
#include "funs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command line options.
   int order = 1;
   int ref_levels = 4;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ref_levels, "-rs", "--refine-serial",
                  "Serial refinement level");
   args.ParseCheck();

   // 3. Read the serial mesh from the given mesh file.
   Mesh serial_mesh = Mesh::MakeCartesian2D(4, 4, Element::Type::QUADRILATERAL,
                                            false);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh once in parallel to increase the resolution.
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); // the serial mesh is no longer needed
   for (int i=0; i<ref_levels; i++) {mesh.UniformRefinement();}

   // 5. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fespace(&mesh, &fec);
   HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << total_num_dofs << endl;
   }

   // 6. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> ess_bdr(4); ess_bdr = 0;

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction filter_gf(&fespace);
   ParGridFunction control_gf(&fespace);
   ParGridFunction simp_gf(&fespace);
   const real_t exponent(3.0), rho0(1e-06);
   MappedGFCoefficient simp_cf(filter_gf,
                               new std::function<real_t(const real_t)>([exponent, rho0](const real_t x)
   {
      return simp(x, exponent, rho0);
   }));
   filter_gf = 0.0;

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   real_t r = 1;
   FunctionCoefficient f([&r](const Vector &x) {return x*x < r*r;});
   HelmholtzFilter filter(fespace, ess_bdr, 0.05, false);
   filter.GetLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(f));

   for (int i=0; i<1; i++)
   {
      r=std::pow(2,-i-1);
      filter.Solve(filter_gf);
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream filter_sock(vishost, visport);
      filter_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      filter_sock.precision(8);
      filter_sock << "solution\n" << mesh << filter_gf << flush;
      control_gf.ProjectCoefficient(f);
      socketstream control_sock(vishost, visport);
      control_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      control_sock.precision(8);
      control_sock << "solution\n" << mesh << control_gf << flush;
      simp_gf.ProjectCoefficient(simp_cf);
      socketstream simp_sock(vishost, visport);
      simp_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      simp_sock.precision(8);
      simp_sock << "solution\n" << mesh << simp_gf << flush;

   }
   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"

   return 0;
}
