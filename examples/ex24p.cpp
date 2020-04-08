//                       MFEM Example 24 - Parallel Version
//
// Compile with: make ex24p
//
// Sample runs:  mpirun -np 4 ex24p -m ../data/star.mesh
//               mpirun -np 4 ex24p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex24p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex24p -m ../data/beam-hex.mesh -o 2 -pa
//               mpirun -np 4 ex24p -m ../data/escher.mesh
//               mpirun -np 4 ex24p -m ../data/escher.mesh -o 2
//               mpirun -np 4 ex24p -m ../data/fichera.mesh
//               mpirun -np 4 ex24p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex24p -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex24p -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex24p -m ../data/beam-hex-nurbs.mesh
//               mpirun -np 4 ex24p -m ../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex24p -m ../data/amr-hex.mesh
//
// Device sample runs:
//               mpirun -np 4 ex24p -m ../data/star.mesh -pa -d cuda
//               mpirun -np 4 ex24p -m ../data/star.mesh -pa -d raja-cuda
//               mpirun -np 4 ex24p -m ../data/star.mesh -pa -d raja-omp
//               mpirun -np 4 ex24p -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code illustrates usage of mixed finite element
//               spaces. Using two different approaches, we project a gradient
//               of a function in H^1 to H(curl). Other spaces and example
//               computations are to be added in the future.
//
//               We recommend viewing examples 1 and 3 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double p_exact(const Vector &x);
void gradp_exact(const Vector &, Vector &);

int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
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
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementCollection *H1fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   ParFiniteElementSpace *H1fespace = new ParFiniteElementSpace(pmesh, H1fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   HYPRE_Int H1size = H1fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of Nedelec finite element unknowns: " << size << endl;
      cout << "Number of H1 finite element unknowns: " << H1size << endl;
   }

   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   FunctionCoefficient p_coef(p_exact);
   ParGridFunction p(H1fespace);
   p.ProjectCoefficient(p_coef);
   p.SetTrueVector();
   p.SetFromTrueVector();

   VectorFunctionCoefficient gradp_coef(sdim, gradp_exact);

   // 9. Set up the parallel bilinear forms.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   ParMixedBilinearForm *a_NDH1 = new ParMixedBilinearForm(H1fespace, fespace);
   if (pa)
   {
      a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a_NDH1->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   // First approach: L2 projection
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
   a_NDH1->AddDomainIntegrator(new MixedVectorGradientIntegrator(*muinv));

   // 10. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }

   a->Assemble();
   if (!pa) { a->Finalize(); }

   a_NDH1->Assemble();
   if (!pa) { a_NDH1->Finalize(); }

   Vector B(fespace->GetTrueVSize());
   Vector X(fespace->GetTrueVSize());

   if (pa)
   {
      ParLinearForm *b = new ParLinearForm(fespace); // used as a vector
      a_NDH1->Mult(p, *b); // process-local multiplication
      b->ParallelAssemble(B);
      delete b;
   }
   else
   {
      HypreParMatrix *NDH1 = a_NDH1->ParallelAssemble();

      Vector P(H1fespace->GetTrueVSize());
      p.GetTrueDofs(P);

      NDH1->Mult(P,B);

      delete NDH1;
   }

   // 11. Define and apply a parallel PCG solver for AX=B with Jacobi
   //     preconditioner.
   if (pa)
   {
      Array<int> ess_tdof_list; // empty

      OperatorPtr A;
      a->FormSystemMatrix(ess_tdof_list, A);

      OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(Jacobi);
      X = 0.0;
      cg.Mult(B, X);
   }
   else
   {
      HypreParMatrix *Amat = a->ParallelAssemble();
      HypreDiagScale Jacobi(*Amat);
      HyprePCG pcg(*Amat);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      X = 0.0;
      pcg.Mult(B, X);

      delete Amat;
   }

   x.SetFromTrueDofs(X);

   // 12. Second approach: compute the same solution by applying
   //     GradientInterpolator in H(curl).
   ParDiscreteLinearOperator grad(H1fespace, fespace);
   grad.AddDomainInterpolator(new GradientInterpolator());
   grad.Assemble();

   ParGridFunction gradp(fespace);
   grad.Mult(p, gradp);

   // 13. Compute the projection of the exact grad p.
   ParGridFunction exact_gradp(fespace);
   exact_gradp.ProjectCoefficient(gradp_coef);
   exact_gradp.SetTrueVector();
   exact_gradp.SetFromTrueVector();

   // 14. Compute and print the L^2 norm of the error.
   {
      double errSol = x.ComputeL2Error(gradp_coef);
      double errInterp = gradp.ComputeL2Error(gradp_coef);
      double errProj = exact_gradp.ComputeL2Error(gradp_coef);

      if (myid == 0)
      {
         cout << "\n Solution of (E_h,v) = (grad p_h,v) for E_h and v in "
              "H(curl): || E_h - grad p ||_{L^2} = " << errSol << '\n' << endl;
         cout << " Gradient interpolant E_h = grad p_h in H(curl): || E_h - "
              "grad p ||_{L^2} = " << errInterp << '\n' << endl;
         cout << " Projection E_h of exact grad p in H(curl): || E_h - grad p "
              "||_{L^2} = " << errProj << '\n' << endl;
      }
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
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

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 17. Free the used memory.
   delete a;
   delete a_NDH1;
   delete sigma;
   delete muinv;
   delete fespace;
   delete H1fespace;
   delete fec;
   delete H1fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

double p_exact(const Vector &x)
{
   if (dim == 3)
   {
      return sin(x(0)) * sin(x(1)) * sin(x(2));
   }
   else if (dim == 2)
   {
      return sin(x(0)) * sin(x(1));
   }

   return 0.0;
}

void gradp_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = cos(x(0)) * sin(x(1)) * sin(x(2));
      f(1) = sin(x(0)) * cos(x(1)) * sin(x(2));
      f(2) = sin(x(0)) * sin(x(1)) * cos(x(2));
   }
   else
   {
      f(0) = cos(x(0)) * sin(x(1));
      f(1) = sin(x(0)) * cos(x(1));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
