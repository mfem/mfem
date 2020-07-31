//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
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

double u_exact(const Vector &x);
double f_exact(const Vector &x);

// #define FORM_DEFINITE
#define USE_GMRES

#define USE_CSL

#define K2 250.0

int dim;
double kappa;

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
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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
   dim = mesh->Dimension();

   kappa = 2.0 * M_PI;

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(100000./mesh->GetNE())/log(2.)/dim);
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
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   {
      double minsize = pmesh->GetElementSize(0);
      double maxsize = minsize;
      for (int i=1; i<pmesh->GetNE(); ++i)
      {
         const double size_i = pmesh->GetElementSize(i);
         minsize = std::min(minsize, size_i);
         maxsize = std::max(maxsize, size_i);
      }

      cout << myid << ": Element size range: (" << minsize << ", " << maxsize << ")"
           << endl;
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
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);

   //ConstantCoefficient bcoef(1.0);
   FunctionCoefficient bcoef(f_exact);

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   ConstantCoefficient neg(-K2);
   ConstantCoefficient pos(K2);
   b->AddDomainIntegrator(new DomainLFIntegrator(bcoef));
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
   a->AddDomainIntegrator(new MassIntegrator(neg));

#ifdef FORM_DEFINITE
   ParBilinearForm *adef = new ParBilinearForm(fespace);
   adef->AddDomainIntegrator(new DiffusionIntegrator(one));
   adef->AddDomainIntegrator(new MassIntegrator(pos));

   if (static_cond) { adef->EnableStaticCondensation(); }
   adef->Assemble();

   ParGridFunction xdef(fespace);
   xdef = 0.0;

   ParLinearForm *bdef = new ParLinearForm(fespace);
   bdef->AddDomainIntegrator(new DomainLFIntegrator(bcoef));
   bdef->Assemble();

   HypreParMatrix Adef;
   Vector Bdef, Xdef;
   adef->FormLinearSystem(ess_tdof_list, xdef, *bdef, Adef, Xdef, Bdef);
#endif

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

   //A.Print("helmholtz");

   // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
#ifdef FORM_DEFINITE
   HypreSolver *amg = new HypreBoomerAMG(Adef);
#else
   HypreSolver *amg = new HypreBoomerAMG(A);
#endif

   const bool fullDirect = true;

   if (fullDirect)
   {
#ifdef USE_CSL
      Vector Bdef, Xdef;

      ParBilinearForm *Mform =  new ParBilinearForm(fespace);
      Mform->AddDomainIntegrator(new MassIntegrator(pos));
      Mform->Assemble();

      HypreParMatrix Mmat, Smat, Mcopy;
      Mform->FormLinearSystem(ess_tdof_list, x, *b, Mmat, Xdef, Bdef);
      Mform->FormLinearSystem(ess_tdof_list, x, *b, Mcopy, Xdef,
                              Bdef);  // There must be a better way than creating two identical matrices.

      ParBilinearForm *Sform =  new ParBilinearForm(fespace);
      Sform->AddDomainIntegrator(new DiffusionIntegrator(one));
      Sform->Assemble();

      Sform->FormLinearSystem(ess_tdof_list, x, *b, Smat, Xdef, Bdef);

      const double beta1 = 1.0;
      const double beta2 = 1.0;

      Mmat *= -beta1;

      HypreParMatrix * cslRe = ParAdd(&Smat, &Mmat);

      Mcopy *= beta2;

      ComplexHypreParMatrix chpm(cslRe, &Mcopy, false, false);

      HypreParMatrix *cSysMat = chpm.GetSystemMatrix();

      Array<int> block_trueOffsets(3); // number of variables + 1
      block_trueOffsets[0] = 0;
      block_trueOffsets[1] = fespace->TrueVSize();
      block_trueOffsets[2] = fespace->TrueVSize();
      block_trueOffsets.PartialSum();

      // Note that B is of true size.
      BlockVector trueY(block_trueOffsets), trueX(block_trueOffsets),
                  trueRhs(block_trueOffsets);

      trueRhs.GetBlock(0) = B;
      trueRhs.GetBlock(1) = 0.0;

      Operator * Arow = new STRUMPACKRowLocMatrix(*cSysMat);

      STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, MPI_COMM_WORLD);
      strumpack->SetPrintFactorStatistics(true);
      strumpack->SetPrintSolveStatistics(false);
      strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
      strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
      // strumpack->SetMC64Job(strumpack::MC64Job::NONE);
      // strumpack->SetSymmetricPattern(true);
      strumpack->SetOperator(*Arow);
      strumpack->SetFromCommandLine();
      //Solver * precond = strumpack;

      // strumpack->Mult(B, X);

      BlockOperator blockDiagA(block_trueOffsets);
      for (int i=0; i<2; ++i)
      {
         blockDiagA.SetDiagonalBlock(i, &A);
      }

      ProductOperator prod(&blockDiagA, strumpack, false, false);
      //GMRESSolver *gmres = new GMRESSolver(fespace->GetComm());
      BiCGSTABSolver *gmres = new BiCGSTABSolver(fespace->GetComm());

      gmres->SetOperator(prod);
      gmres->SetRelTol(1e-8);
      gmres->SetMaxIter(10000);
      gmres->SetPrintLevel(1);

      gmres->Mult(trueRhs, trueY);
      strumpack->Mult(trueY, trueX);

      X = trueX.GetBlock(0);
      double xim2 = trueX.GetBlock(1).Norml2();
      xim2 *= xim2;
      double sumxim2 = 0.0;

      MPI_Allreduce(&xim2, &sumxim2, 1, MPI_DOUBLE, MPI_SUM, fespace->GetComm());

      if (myid == 0)
      {
         cout << myid << ": norm of Xim " << trueX.GetBlock(1).Norml2() << ", global " <<
              sqrt(sumxim2) << endl;
      }

      delete gmres;
      delete strumpack;
      delete Arow;
#else
      Operator * Arow = new STRUMPACKRowLocMatrix(A);

      STRUMPACKSolver * strumpack = new STRUMPACKSolver(argc, argv, MPI_COMM_WORLD);
      strumpack->SetPrintFactorStatistics(true);
      strumpack->SetPrintSolveStatistics(false);
      strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
      strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
      // strumpack->SetMC64Job(strumpack::MC64Job::NONE);
      // strumpack->SetSymmetricPattern(true);
      strumpack->SetOperator(*Arow);
      strumpack->SetFromCommandLine();
      //Solver * precond = strumpack;

      strumpack->Mult(B, X);

      delete strumpack;
      delete Arow;
#endif
   }
   else
   {
#ifdef USE_GMRES
      HypreGMRES *gmres = new HypreGMRES(A);
      gmres->SetTol(1e-12);
      gmres->SetMaxIter(1000);
      gmres->SetPrintLevel(10);
      gmres->SetPreconditioner(*amg);
      gmres->Mult(B, X);
      delete gmres;
#else
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(100);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);
#endif
   }

   /*
   HYPRE_ParCSRMatrix* amgP = amg->Get_Restriction();
   HypreParMatrix P0(amgP[0], false);
   HypreParMatrix P1(amgP[1], false);
   HypreParMatrix P2(amgP[2], false);
   //HypreParMatrix P3(amgP[3], false);

   P0.Print("P0");
   */

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // Compute and print the L^2 norm of the error.
   {
      FunctionCoefficient uex(u_exact);

      double err = x.ComputeL2Error(uex);
      double xnrm = x.ComputeL2Error(zero);
      ParGridFunction zerogf(fespace);
      zerogf = 0.0;
      double normE = zerogf.ComputeL2Error(uex);
      if (myid == 0)
      {
         cout << "|| E_h - E ||_{L^2} = " << err << endl;
         cout << "|| E_h ||_{L^2} = " << xnrm << endl;
         cout << "|| E ||_{L^2} = " << normE << endl;
      }
   }

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
   //delete pcg;
   delete amg;
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}


double u_exact(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(1.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return sin(kappa*xi)*sin(kappa*yi)*sin(kappa*zi);
}

double f_exact(const Vector &x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(1.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   const double s = 1.0;

   return ((3.0*kappa*kappa) - (s*K2)) * sin(kappa*xi)*sin(kappa*yi)*sin(
             kappa*zi) / s;
}
