//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p_complex
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void curlE_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;


#define COMPLEX_VERSION
#define NEUMANN
#define INDEFINITE

const double omega = 1.4;


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-tet.mesh";
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
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
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
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   ess_bdr.SetSize(pmesh->bdr_attributes.Max());
   ess_bdr = 0;

#ifndef NEUMANN
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
#endif

   //const double imscale = 0.0;
   const double imscale = omega;

   Coefficient *im = new ConstantCoefficient(imscale);  // im part
   //Coefficient *im = new ConstantCoefficient(0.0);  // im part

   VectorFunctionCoefficient E_Re(sdim, E_exact);
   VectorFunctionCoefficient curlE_Re(sdim, curlE_exact);

   ScalarVectorProductCoefficient omegaE(imscale, E_Re);  // im part
   //ScalarVectorProductCoefficient omegaE(0.0, E_Re);  // im part

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(sdim, f_exact);
#ifdef COMPLEX_VERSION
   ParComplexLinearForm *b = new ParComplexLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f), NULL);
   b->AddBoundaryIntegrator(NULL,
                            new VectorFEDomainLFIntegrator(omegaE));  // im part
#else
   // Real version
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
#endif

#ifdef NEUMANN
   b->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(curlE_Re),
                            NULL);
#endif

   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x by projecting the exact
   //     solution. Note that only values from the boundary edges will be used
   //     when eliminating the non-homogeneous boundary condition to modify the
   //     r.h.s. vector b.
   /*
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);
   */

#ifdef COMPLEX_VERSION
   // Complex version
   ParComplexGridFunction x(fespace);
   x = 0.0;
   Vector zero(sdim);
   zero = 0.0;
   VectorConstantCoefficient E_Im(zero);
   //x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);
   x.ProjectCoefficient(E_Re, E_Im);
#else
   ParGridFunction x(fespace);
   x = 0.0;
   x.ProjectCoefficient(E_Re);
#endif

   // 11. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
#ifdef INDEFINITE
   Coefficient *sigma = new ConstantCoefficient(
      -omega*omega); // indefinite -, definite +
#else
   Coefficient *sigma = new ConstantCoefficient(
      omega*omega); // indefinite -, definite +
#endif
   Coefficient *abssigma = new ConstantCoefficient(omega*omega);
   Coefficient *imabs = new ConstantCoefficient(imscale);  // im part
   //Coefficient *imabs = new ConstantCoefficient(0.0);  // im part
   //Coefficient *im = new ConstantCoefficient(0.0);

#ifdef COMPLEX_VERSION
   // Complex version
   ParSesquilinearForm *a = new ParSesquilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv), NULL);
   //a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma), new VectorFEMassIntegrator(*im));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma), NULL);
   a->AddBoundaryIntegrator(NULL, new VectorFEMassIntegrator(*im));  // im part
#else
   // Real version
   ParBilinearForm *a = new ParBilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   //a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma), new VectorFEMassIntegrator(*im));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
#endif

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   //if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   ParBilinearForm a_Re(fespace);
   a_Re.AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a_Re.AddDomainIntegrator(new VectorFEMassIntegrator(*abssigma));

   //if (pa) { a_Re.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a_Re.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a_Re.Assemble();

   OperatorPtr A_Re;
   a_Re.FormSystemMatrix(ess_tdof_list, A_Re);

   ParBilinearForm a_Im(fespace);
   a_Im.AddBoundaryIntegrator(new VectorFEMassIntegrator(*imabs));
   a_Im.Assemble();

   OperatorPtr A_Im;
   a_Im.FormSystemMatrix(ess_tdof_list, A_Im);

   // 13. Solve the system AX=B using PCG with the AMS preconditioner from hypre
   //     (in the full assembly case) or CG with Jacobi preconditioner (in the
   //     partial assembly case).

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = fespace->GetTrueVSize();
   offsets[2] = fespace->GetTrueVSize();
   offsets.PartialSum();

   //OperatorJacobiSmoother massJacobi(a_Im, ess_tdof_list);

   StopWatch sw;
   sw.Clear();
   sw.Start();

   if (pa) // Jacobi preconditioning in partial assembly mode
   {
      MFEM_VERIFY(false, "TODO");
      //OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      //cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      if (myid == 0)
      {
         cout << "Size of linear system: "
              << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
      }

      //HypreAMS ams(*A_Re.As<HypreParMatrix>(), fespace);

      // One option is to use the standard real-valued MatrixFreeAMS to precondition
      // the real part of the complex system in the PMHSS preconditioner (BlockDiagonalPreconditioner).
      // Another option is to use complex MatrixFreeAMS to precondition the
      // complex system without PMHSS and without a BlockDiagonalPreconditioner.
      //#define COMPLEX_AMS

#ifdef MFEM_USE_AMGX
      bool useAmgX = false;
      cout << "Built with AMGX, using AMGX " << useAmgX << endl;
      MatrixFreeAMS ams(a_Re, *A_Re, *fespace, muinv, abssigma, im, imabs, NULL,
                        ess_bdr, useAmgX);
      MatrixFreeAMS ams(a_Re, *A_Re, *fespace, muinv, abssigma, NULL, NULL, ess_bdr,
                        useAmgX);
#ifdef COMPLEX_AMS
      MFEM_VERIFY(false, "TODO");
#endif

#else
      cout << "Not built with AMGX" << endl;
#ifdef COMPLEX_AMS
      MatrixFreeAMS ams(a_Re, *A_Re, A.Ptr(), *fespace, muinv, abssigma, im, imabs,
                        NULL, ess_bdr);
#else
      MatrixFreeAMS ams(a_Re, *A_Re, NULL, *fespace, muinv, abssigma, NULL, NULL,
                        NULL, ess_bdr);
#endif
#endif

#ifdef COMPLEX_VERSION

#ifdef COMPLEX_AMS
      //MFEM_VERIFY(false, "TODO");
#else
      BlockDiagonalPreconditioner BlockDP(offsets);
      BlockDP.SetDiagonalBlock(0, &ams);
      BlockDP.SetDiagonalBlock(1, &ams);

      /*
      BlockDiagonalPreconditioner BlockDP_Im(offsets);
      BlockDP_Im.SetDiagonalBlock(0, &massJacobi); // TODO: this won't work if it has zeros on diagonal
      BlockDP_Im.SetDiagonalBlock(1, &massJacobi);
      */

      //Complex_PMHSS PMHSS(A_Re, A_Im, &BlockDP, &BlockDP_Im);
      //Complex_PMHSS PMHSS(A_Re, A_Im, &BlockDP, NULL, 2.0 * omega);
      //Complex_PMHSS PMHSS(A_Re, A_Im, &BlockDP, NULL, omega);
      Complex_PMHSS PMHSS(A_Re.Ptr(), A_Im.Ptr(), &BlockDP, NULL, 1.0);

      ComplexOperator AspdComplex(A_Re.Ptr(), A_Im.Ptr(), false, false);

      GMRESSolver PMHSSgmres(MPI_COMM_WORLD);
      PMHSSgmres.SetPrintLevel(1);
      PMHSSgmres.SetKDim(100);
      PMHSSgmres.SetMaxIter(100);
      PMHSSgmres.SetRelTol(1e-6);
      PMHSSgmres.SetAbsTol(0.0);
      PMHSSgmres.SetOperator(AspdComplex);
      PMHSSgmres.SetPreconditioner(PMHSS);
#endif

      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetPrintLevel(1);
      gmres.SetKDim(1000);
      gmres.SetMaxIter(100);
      gmres.SetRelTol(1e-8);
      gmres.SetAbsTol(0.0);
      gmres.SetOperator(*A);
      //gmres.SetPreconditioner(BlockDP);
#ifdef COMPLEX_AMS
      //MFEM_VERIFY(false, "TODO");
      gmres.SetPreconditioner(ams);
#else
      gmres.SetPreconditioner(PMHSS);
      //gmres.SetPreconditioner(PMHSSgmres);
#endif

#else
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetPrintLevel(1);
      gmres.SetKDim(1000);
      gmres.SetMaxIter(100);
      gmres.SetRelTol(1e-8);
      gmres.SetAbsTol(0.0);
      gmres.SetOperator(*A);
      gmres.SetPreconditioner(ams);
#endif

      gmres.Mult(B, X);
   }

   sw.Stop();
   mfem::out << "Total solve time " <<sw.RealTime() << endl;

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 15. Compute and print the L^2 norm of the error.
   {
#ifdef COMPLEX_VERSION
      double err = x.real().ComputeL2Error(E_Re);
#else
      double err = x.ComputeL2Error(E_Re);
#endif
      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
      }
   }

   // 16. Save the refined mesh and the solution in parallel. This output can
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
#ifdef COMPLEX_VERSION
      x.real().Save(sol_ofs);
#else
      x.Save(sol_ofs);
#endif
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
#ifdef COMPLEX_VERSION
      sol_sock << "solution\n" << *pmesh << x.real() << flush;
#else
      sol_sock << "solution\n" << *pmesh << x << flush;
#endif
   }

   // 18. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void curlE_exact(const Vector &x, Vector &curl)
{
   if (dim == 3)
   {
      curl(0) = kappa * cos(kappa * x(2));
      curl(1) = kappa * cos(kappa * x(0));
      curl(2) = kappa * cos(kappa * x(1));
   }
   else
   {
      MFEM_VERIFY(false, "");
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      // indefinite -m, definite +m
      const double c = kappa * kappa;
#ifdef INDEFINITE
      const double m = -omega * omega;
#else
      const double m = omega * omega;
#endif
      f(0) = (c + m) * sin(kappa * x(1));
      f(1) = (c + m) * sin(kappa * x(2));
      f(2) = (c + m) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
