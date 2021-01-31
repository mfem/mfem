//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p -m ../data/star.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh -o 2 -pa
//               mpirun -np 4 ex3p -m ../data/escher.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/fichera.mesh
//               mpirun -np 4 ex3p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex3p -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex3p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/mobius-strip.mesh -o 2 -f 0.1
//               mpirun -np 4 ex3p -m ../data/klein-bottle.mesh -o 2 -f 0.1
//
// Device sample runs:
//               mpirun -np 4 ex3p -m ../data/star.mesh -pa -d cuda
//               mpirun -np 4 ex3p -m ../data/star.mesh -pa -d raja-cuda
//               mpirun -np 4 ex3p -m ../data/star.mesh -pa -d raja-omp
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-tet.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   bool ams = true;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&ams, "-ams", "--hypre-ams", "-slu",
                  "--superlu", "Use AMS or SuperLU solver.");
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
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   kappa = freq * M_PI;

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (mpi.Root()) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement (2 by default, or
   //    specified on the command line with -rs).
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution (1 time by
   //    default, or specified on the command line with -rp). Once the parallel
   //    mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }
   pmesh.ReorientTetMesh();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = NULL;
   if (dim == 1)
   {
      fec = new ND_R1D_FECollection(order, dim);
   }
   else if (dim == 2)
   {
      fec = new ND_R2D_FECollection(order, dim);
   }
   else
   {
      fec = new ND_FECollection(order, dim);
   }
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   mfem::out << "Number of finite element unknowns: " << size << endl;

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(3, f_exact);
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x by projecting the exact
   //     solution. Note that only values from the boundary edges will be used
   //     when eliminating the non-homogeneous boundary condition to modify the
   //     r.h.s. vector b.
   ParGridFunction sol(&fespace);
   VectorFunctionCoefficient E(3, E_exact);
   sol.ProjectCoefficient(E);

   {
      double err = sol.ComputeL2Error(E);
      mfem::out << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
   }
   if (dim == 2)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      /*
      DenseMatrix xyMat(2,3); xyMat = 0.0;
      xyMat(0,0) = 1.0; xyMat(1,1) = 1.0;
      MatrixConstantCoefficient xyMatCoef(xyMat);
      Vector zVec(3); zVec = 0.0; zVec(2) = 1;
      VectorConstantCoefficient zVecCoef(zVec);
      VectorGridFunctionCoefficient solCoef(&sol);

      MatrixVectorProductCoefficient xyCoef(xyMatCoef, solCoef);
      InnerProductCoefficient zCoef(zVecCoef, solCoef);

      ND_FECollection fec_nd(order, dim);
      H1_FECollection fec_h1(order, dim);
      ParFiniteElementSpace fes_nd(&pmesh, &fec_nd);
      ParFiniteElementSpace fes_h1(&pmesh, &fec_h1);

      ParGridFunction xyComp(&fes_nd);
      ParGridFunction zComp(&fes_h1);

      xyComp.ProjectCoefficient(xyCoef);
      zComp.ProjectCoefficient(zCoef);
      */
      socketstream xy_sock(vishost, visport);
      xy_sock << "parallel " << mpi.WorldSize() << " "
              << mpi.WorldRank() << "\n";
      xy_sock.precision(8);
      xy_sock << "solution\n" << pmesh << sol
              << "window_title 'ex xy components'\n" << flush;
      /*
      socketstream z_sock(vishost, visport);
      z_sock << "parallel " << mpi.WorldSize() << " "
        << mpi.WorldRank() << "\n";
      z_sock.precision(8);
      z_sock << "solution\n" << pmesh << zComp
        << "window_title 'ex z component'"
        << "window_geometry 402 0 400 350" << flush;
      */
      {
         //MatrixVectorProductCoefficient dxyCoef(xyMatCoef, E);
         ParGridFunction dxyComp(&fespace);
         dxyComp.ProjectCoefficient(E);
         dxyComp.Add(-1.0, sol);
         /*
         InnerProductCoefficient exact_zCoef(zVecCoef, E);
         ParGridFunction dzComp(&fes_h1);
         dzComp.ProjectCoefficient(exact_zCoef);
         dzComp.Add(-1.0, zComp);
         */
         socketstream dxy_sock(vishost, visport);
         dxy_sock << "parallel " << mpi.WorldSize() << " "
                  << mpi.WorldRank() << "\n";
         dxy_sock.precision(8);
         dxy_sock << "solution\n" << pmesh << dxyComp
                  << "window_title 'ex dxy components'"
                  << "window_geometry 0 375 400 350" << flush;
         /*
         socketstream dz_sock(vishost, visport);
         dz_sock << "parallel " << mpi.WorldSize() << " "
            << mpi.WorldRank() << "\n";
         dz_sock.precision(8);
         dz_sock << "solution\n" << pmesh << dzComp
            << "window_title 'ex dz component'"
            << "window_geometry 402 375 400 350" << flush;
         */
      }
   }

   // 11. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   DenseMatrix sigmaMat(3);
   sigmaMat(0,0) = 2.0; sigmaMat(1,1) = 2.0; sigmaMat(2,2) = 2.0;
   sigmaMat(0,2) = 0.0; sigmaMat(2,0) = 0.0;
   sigmaMat(0,1) = M_SQRT1_2; sigmaMat(1,0) = M_SQRT1_2;
   sigmaMat(1,2) = M_SQRT1_2; sigmaMat(2,1) = M_SQRT1_2;

   Coefficient *muinv = new ConstantCoefficient(1.0);
   MatrixConstantCoefficient sigma(sigmaMat);
   ParBilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;

   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);

   // 13. Solve the system AX=B using PCG with the AMS preconditioner from hypre
   //     (in the full assembly case) or CG with Jacobi preconditioner (in the
   //     partial assembly case).

   if (pa) // Jacobi preconditioning in partial assembly mode
   {
      OperatorJacobiSmoother Jacobi(a, ess_tdof_list);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      if (ams)
      {
         mfem::out << "Size of linear system: "
                   << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;

         HypreSolver * prec = NULL;
         /*
         if (dim == 1)
         {
            prec = new HypreBoomerAMG(*A.As<HypreParMatrix>());
         }
         else if (dim == 2)
         {
            ParFiniteElementSpace *prec_fespace =
               (a.StaticCondensationIsEnabled() ? a.SCParFESpace() : &fespace);
            prec = new HypreAMS(*A.As<HypreParMatrix>(), prec_fespace);

            // prec = new HypreBoomerAMG(*A.As<HypreParMatrix>());
         }
         else
         */
         {
            ParFiniteElementSpace *prec_fespace =
               (a.StaticCondensationIsEnabled() ? a.SCParFESpace() : &fespace);
            HypreAMS *ams = new HypreAMS(*A.As<HypreParMatrix>(), prec_fespace);
            prec = ams;
         }
         HyprePCG pcg(*A.As<HypreParMatrix>());
         pcg.SetTol(1e-12);
         pcg.SetMaxIter(1000);
         pcg.SetPrintLevel(2);
         pcg.SetPreconditioner(*prec);
         pcg.Mult(B, X);
      }
      else
#ifdef MFEM_USE_SUPERLU
      {
         mfem::out << "Size of linear system: "
                   << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;

         SuperLURowLocMatrix A_SuperLU(*A.As<HypreParMatrix>());
         SuperLUSolver AInv(MPI_COMM_WORLD);
         AInv.SetOperator(A_SuperLU);
         AInv.Mult(B,X);
      }
#else
      {
         mfme::out << "No solvers available." << endl;
         return 1;
      }
#endif
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, sol);

   // 15. Compute and print the L^2 norm of the error.
   {
      double err = sol.ComputeL2Error(E);
      mfem::out << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
   }

   // 16. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << mpi.WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << mpi.WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      sol.Save(sol_ofs);
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      if (dim == 2)
      {
         /*
          DenseMatrix xyMat(2,3); xyMat = 0.0;
               xyMat(0,0) = 1.0; xyMat(1,1) = 1.0;
               MatrixConstantCoefficient xyMatCoef(xyMat);
               Vector zVec(3); zVec = 0.0; zVec(2) = 1;
               VectorConstantCoefficient zVecCoef(zVec);
               VectorGridFunctionCoefficient solCoef(&sol);

               MatrixVectorProductCoefficient xyCoef(xyMatCoef, solCoef);
               InnerProductCoefficient zCoef(zVecCoef, solCoef);

               ND_FECollection fec_nd(order, dim);
               H1_FECollection fec_h1(order, dim);
               ParFiniteElementSpace fes_nd(&pmesh, &fec_nd);
               ParFiniteElementSpace fes_h1(&pmesh, &fec_h1);

               ParGridFunction xyComp(&fes_nd);
               ParGridFunction zComp(&fes_h1);

               xyComp.ProjectCoefficient(xyCoef);
               zComp.ProjectCoefficient(zCoef);
         */
         socketstream xy_sock(vishost, visport);
         xy_sock << "parallel " << mpi.WorldSize() << " "
                 << mpi.WorldRank() << "\n";
         xy_sock.precision(8);
         xy_sock << "solution\n" << pmesh << sol
                 << "window_title 'xy components'\n" << flush;
         /*
              socketstream z_sock(vishost, visport);
              z_sock << "parallel " << mpi.WorldSize() << " "
                     << mpi.WorldRank() << "\n";
              z_sock.precision(8);
              z_sock << "solution\n" << pmesh << zComp
                     << "window_title 'z component'"
                     << "window_geometry 402 0 400 350" << flush;
         */
         {
            // MatrixVectorProductCoefficient dxyCoef(xyMatCoef, E);
            ParGridFunction dxyComp(&fespace);
            dxyComp.ProjectCoefficient(E);
            dxyComp.Add(-1.0, sol);
            /*
                 InnerProductCoefficient exact_zCoef(zVecCoef, E);
                 ParGridFunction dzComp(&fes_h1);
                 dzComp.ProjectCoefficient(exact_zCoef);
                 dzComp.Add(-1.0, zComp);
            */
            socketstream dxy_sock(vishost, visport);
            dxy_sock << "parallel " << mpi.WorldSize() << " "
                     << mpi.WorldRank() << "\n";
            dxy_sock.precision(8);
            dxy_sock << "solution\n" << pmesh << dxyComp
                     << "window_title 'dxy components'"
                     << "window_geometry 0 375 400 350" << flush;
            /*
                 socketstream dz_sock(vishost, visport);
                 dz_sock << "parallel " << mpi.WorldSize() << " "
                         << mpi.WorldRank() << "\n";
                 dz_sock.precision(8);
                 dz_sock << "solution\n" << pmesh << dzComp
                         << "window_title 'dz component'"
                         << "window_geometry 402 375 400 350" << flush;
            */
         }
      }
      else
      {
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << mpi.WorldSize() << " "
                  << mpi.WorldRank() << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << pmesh << sol << flush;
      }
   }

   // 18. Free the used memory.
   delete muinv;
   delete fec;

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 1)
   {
      E(0) = 1.1 * sin(kappa * x(0) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * x(0) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * x(0) + 0.9 * M_PI);
   }
   else if (dim == 2)
   {
      E(0) = 1.1 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
   }
   else
   {
      E(0) = 1.1 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      E *= cos(kappa * x(2));
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 1)
   {
      double s0 = sin(kappa * x(0) + 0.0 * M_PI);
      double s4 = sin(kappa * x(0) + 0.4 * M_PI);
      double s9 = sin(kappa * x(0) + 0.9 * M_PI);

      f(0) = 2.2 * s0 + 1.2 * M_SQRT1_2 * s4;
      f(1) = 1.2 * (2.0 + kappa * kappa) * s4 +
             M_SQRT1_2 * (1.1 * s0 + 1.3 * s9);
      f(2) = 1.3 * (2.0 + kappa * kappa) * s9 + 1.2 * M_SQRT1_2 * s4;
   }
   else if (dim == 2)
   {
      double s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      f(0) = 0.55 * (4.0 + kappa * kappa) * s0 +
             0.6 * (M_SQRT2 - kappa * kappa) * s4;
      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 +
             0.6 * (4.0 + kappa * kappa) * s4 +
             0.65 * M_SQRT2 * s9;
      f(2) = 0.6 * M_SQRT2 * s4 + 1.3 * (2.0 + kappa * kappa) * s9;
   }
   else
   {
      double s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      double s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      double s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      double c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      double sk = sin(kappa * x(2));
      double ck = cos(kappa * x(2));

      f(0) = 0.55 * (4.0 + 3.0 * kappa * kappa) * s0 * ck +
             0.6 * (M_SQRT2 - kappa * kappa) * s4 * ck -
             0.65 * M_SQRT2 * kappa * kappa * c9 * sk;

      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 * ck +
             0.6 * (4.0 + 3.0 * kappa * kappa) * s4 * ck +
             0.65 * M_SQRT2 * s9 * ck -
             0.65 * M_SQRT2 * kappa * kappa * c9 * sk;

      f(2) = 0.6 * M_SQRT2 * s4 * ck -
             M_SQRT2 * kappa * kappa * (0.55 * c0 + 0.6 * c4) * sk
             + 1.3 * (2.0 + kappa * kappa) * s9 * ck;
   }
}
