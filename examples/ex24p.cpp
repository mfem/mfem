//                       MFEM Example 24 - Parallel Version
//
// Compile with: make ex24p
//
// Sample runs:  mpirun -np 4 ex24p -m ../data/star.mesh
//               mpirun -np 4 ex24p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex24p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex24p -m ../data/beam-hex.mesh -o 2 -pa
//               mpirun -np 4 ex24p -m ../data/beam-hex.mesh -o 2 -pa -p 1
//               mpirun -np 4 ex24p -m ../data/beam-hex.mesh -o 2 -pa -p 2
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
//               spaces, with three variants:
//
//               1) (grad p, u) for p in H^1 tested against u in H(curl)
//               2) (curl v, u) for v in H(curl) tested against u in H(div), 3D
//               3) (div v, q) for v in H(div) tested against q in L_2
//
//               Using different approaches, we project the gradient, curl, or
//               divergence to the appropriate space.
//
//               We recommend viewing examples 1, 3, and 5 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double p_exact(const Vector &x);
void gradp_exact(const Vector &, Vector &);
double div_gradp_exact(const Vector &x);
void v_exact(const Vector &x, Vector &v);
void curlv_exact(const Vector &x, Vector &cv);

int dim;
double freq = 1.0, kappa;

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
   int prob = 0;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&prob, "-p", "--problem-type",
                  "Choose between 0: grad, 1: curl, 2: div");
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
   kappa = freq * M_PI;

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
   //    use Nedelec or Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *trial_fec = NULL;
   FiniteElementCollection *test_fec = NULL;

   if (prob == 0)
   {
      trial_fec = new H1_FECollection(order, dim);
      test_fec = new ND_FECollection(order, dim);
   }
   else if (prob == 1)
   {
      trial_fec = new ND_FECollection(order, dim);
      test_fec = new RT_FECollection(order-1, dim);
   }
   else
   {
      trial_fec = new RT_FECollection(order-1, dim);
      test_fec = new L2_FECollection(order-1, dim);
   }

   ParFiniteElementSpace trial_fes(pmesh, trial_fec);
   ParFiniteElementSpace test_fes(pmesh, test_fec);

   HYPRE_Int trial_size = trial_fes.GlobalTrueVSize();
   HYPRE_Int test_size = test_fes.GlobalTrueVSize();

   if (myid == 0)
   {
      if (prob == 0)
      {
         cout << "Number of Nedelec finite element unknowns: " << test_size << endl;
         cout << "Number of H1 finite element unknowns: " << trial_size << endl;
      }
      else if (prob == 1)
      {
         cout << "Number of Nedelec finite element unknowns: " << trial_size << endl;
         cout << "Number of Raviart-Thomas finite element unknowns: " << test_size <<
              endl;
      }
      else
      {
         cout << "Number of Raviart-Thomas finite element unknowns: "
              << trial_size << endl;
         cout << "Number of L2 finite element unknowns: " << test_size << endl;
      }
   }

   // 8. Define the solution vector as a parallel finite element grid function
   //    corresponding to the trial fespace.
   ParGridFunction gftest(&test_fes);
   ParGridFunction gftrial(&trial_fes);
   ParGridFunction x(&test_fes);
   FunctionCoefficient p_coef(p_exact);
   VectorFunctionCoefficient gradp_coef(sdim, gradp_exact);
   VectorFunctionCoefficient v_coef(sdim, v_exact);
   VectorFunctionCoefficient curlv_coef(sdim, curlv_exact);
   FunctionCoefficient divgradp_coef(div_gradp_exact);

   if (prob == 0)
   {
      gftrial.ProjectCoefficient(p_coef);
   }
   else if (prob == 1)
   {
      gftrial.ProjectCoefficient(v_coef);
   }
   else
   {
      gftrial.ProjectCoefficient(gradp_coef);
   }

   gftrial.SetTrueVector();
   gftrial.SetFromTrueVector();

   // 9. Set up the parallel bilinear forms for L2 projection.
   ConstantCoefficient one(1.0);
   ParBilinearForm a(&test_fes);
   ParMixedBilinearForm a_mixed(&trial_fes, &test_fes);
   if (pa)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a_mixed.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   if (prob == 0)
   {
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      a_mixed.AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
   }
   else if (prob == 1)
   {
      a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      a_mixed.AddDomainIntegrator(new MixedVectorCurlIntegrator(one));
   }
   else
   {
      a.AddDomainIntegrator(new MassIntegrator(one));
      a_mixed.AddDomainIntegrator(new VectorFEDivergenceIntegrator(one));
   }

   // 10. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }

   a.Assemble();
   if (!pa) { a.Finalize(); }

   a_mixed.Assemble();
   if (!pa) { a_mixed.Finalize(); }

   Vector B(test_fes.GetTrueVSize());
   Vector X(test_fes.GetTrueVSize());

   if (pa)
   {
      ParLinearForm b(&test_fes); // used as a vector
      a_mixed.Mult(gftrial, b); // process-local multiplication
      b.ParallelAssemble(B);
   }
   else
   {
      HypreParMatrix *mixed = a_mixed.ParallelAssemble();

      Vector P(trial_fes.GetTrueVSize());
      gftrial.GetTrueDofs(P);

      mixed->Mult(P,B);

      delete mixed;
   }

   // 11. Define and apply a parallel PCG solver for AX=B with Jacobi
   //     preconditioner.
   if (pa)
   {
      Array<int> ess_tdof_list; // empty

      OperatorPtr A;
      a.FormSystemMatrix(ess_tdof_list, A);

      OperatorJacobiSmoother Jacobi(a, ess_tdof_list);

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
      HypreParMatrix *Amat = a.ParallelAssemble();
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

   // 12. Compute the same field by applying a DiscreteInterpolator.
   ParGridFunction discreteInterpolant(&test_fes);
   ParDiscreteLinearOperator dlo(&trial_fes, &test_fes);
   if (prob == 0)
   {
      dlo.AddDomainInterpolator(new GradientInterpolator());
   }
   else if (prob == 1)
   {
      dlo.AddDomainInterpolator(new CurlInterpolator());
   }
   else
   {
      dlo.AddDomainInterpolator(new DivergenceInterpolator());
   }

   dlo.Assemble();
   dlo.Mult(gftrial, discreteInterpolant);

   // 13. Compute the projection of the exact field.
   ParGridFunction exact_proj(&test_fes);
   if (prob == 0)
   {
      exact_proj.ProjectCoefficient(gradp_coef);
   }
   else if (prob == 1)
   {
      exact_proj.ProjectCoefficient(curlv_coef);
   }
   else
   {
      exact_proj.ProjectCoefficient(divgradp_coef);
   }

   exact_proj.SetTrueVector();
   exact_proj.SetFromTrueVector();

   // 14. Compute and print the L_2 norm of the error.
   if (prob == 0)
   {
      double errSol = x.ComputeL2Error(gradp_coef);
      double errInterp = discreteInterpolant.ComputeL2Error(gradp_coef);
      double errProj = exact_proj.ComputeL2Error(gradp_coef);

      if (myid == 0)
      {
         cout << "\n Solution of (E_h,v) = (grad p_h,v) for E_h and v in H(curl)"
              ": || E_h - grad p ||_{L_2} = " << errSol << '\n' << endl;
         cout << " Gradient interpolant E_h = grad p_h in H(curl): || E_h - grad"
              " p ||_{L_2} = " << errInterp << '\n' << endl;
         cout << " Projection E_h of exact grad p in H(curl): || E_h - grad p "
              "||_{L_2} = " << errProj << '\n' << endl;
      }
   }
   else if (prob == 1)
   {
      double errSol = x.ComputeL2Error(curlv_coef);
      double errInterp = discreteInterpolant.ComputeL2Error(curlv_coef);
      double errProj = exact_proj.ComputeL2Error(curlv_coef);

      if (myid == 0)
      {
         cout << "\n Solution of (E_h,w) = (curl v_h,w) for E_h and w in "
              "H(div): || E_h - curl v ||_{L_2} = " << errSol << '\n' << endl;
         cout << " Curl interpolant E_h = curl v_h in H(div): || E_h - curl v "
              "||_{L_2} = " << errInterp << '\n' << endl;
         cout << " Projection E_h of exact curl v in H(div): || E_h - curl v "
              "||_{L_2} = " << errProj << '\n' << endl;
      }
   }
   else
   {
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double errSol = x.ComputeL2Error(divgradp_coef, irs);
      double errInterp = discreteInterpolant.ComputeL2Error(divgradp_coef, irs);
      double errProj = exact_proj.ComputeL2Error(divgradp_coef, irs);

      if (myid == 0)
      {
         cout << "\n Solution of (f_h,q) = (div v_h,q) for f_h and q in L_2: "
              "|| f_h - div v ||_{L_2} = " << errSol << '\n' << endl;
         cout << " Divergence interpolant f_h = div v_h in L_2: || f_h - div v"
              " ||_{L_2} = " << errInterp << '\n' << endl;
         cout << " Projection f_h of exact div v in L_2: || f_h - div v "
              "||_{L_2} = " << errProj << '\n' << endl;
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
   delete trial_fec;
   delete test_fec;
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

double div_gradp_exact(const Vector &x)
{
   if (dim == 3)
   {
      return -3.0 * sin(x(0)) * sin(x(1)) * sin(x(2));
   }
   else if (dim == 2)
   {
      return -2.0 * sin(x(0)) * sin(x(1));
   }

   return 0.0;
}

void v_exact(const Vector &x, Vector &v)
{
   if (dim == 3)
   {
      v(0) = sin(kappa * x(1));
      v(1) = sin(kappa * x(2));
      v(2) = sin(kappa * x(0));
   }
   else
   {
      v(0) = sin(kappa * x(1));
      v(1) = sin(kappa * x(0));
      if (x.Size() == 3) { v(2) = 0.0; }
   }
}

void curlv_exact(const Vector &x, Vector &cv)
{
   if (dim == 3)
   {
      cv(0) = -kappa * cos(kappa * x(2));
      cv(1) = -kappa * cos(kappa * x(0));
      cv(2) = -kappa * cos(kappa * x(1));
   }
   else
   {
      cv = 0.0;
   }
}
