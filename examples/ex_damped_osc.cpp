//                       MFEM Example 1
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
//               of essential boundary conditions and the optional connection
//               to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double tau_ = 1.0;
static double rho_ = 1.0;
static double sigma_ = 1.0;
static double omega_ = 1.0;

double x_real_exact(const Vector &);
double x_imag_exact(const Vector &);
double f_real_exact(const Vector &);
double f_imag_exact(const Vector &);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;
   bool herm_conv = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;


   // 2. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels = 1;
      //        (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system.
   ComplexLinearForm *b = new ComplexLinearForm(fespace, conv);
   FunctionCoefficient f_r(f_real_exact);
   FunctionCoefficient f_i(f_imag_exact);
   b->AddDomainIntegrator(new DomainLFIntegrator(f_r),
                          new DomainLFIntegrator(f_i));
   b->Assemble();

   // 7. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ComplexGridFunction x(fespace);
   ComplexGridFunction x_exact(fespace);
   FunctionCoefficient x_r(x_real_exact);
   FunctionCoefficient x_i(x_imag_exact);
   x.ProjectCoefficient(x_r, x_i);
   x_exact.ProjectCoefficient(x_r, x_i);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *mesh << x_exact.real()
                 << "window_title 'Exact Real Part'" << flush;
      sol_sock_i << "solution\n" << *mesh << x_exact.imag()
                 << "window_title 'Exact Imaginary Part'" << flush;
   }

   // 8. Set up the parallel sesquilinear form a(.,.) on the finite element
   //    space corresponding to the damped harmonic oscillator operator
   //    -Div(tau Grad) + omega^2 rho - i omega sigma, by adding the
   //    Diffusion domain integrator and appropriate Mass domain integrators.

   ConstantCoefficient negTauCoef(-tau_);
   ConstantCoefficient rhoCoef(omega_ * omega_ * rho_);
   ConstantCoefficient sigmaCoef(omega_ * sigma_);
   ConstantCoefficient negSigmaCoef(-omega_ * sigma_);

   SesquilinearForm *a = new SesquilinearForm(fespace, conv);
   a->AddDomainIntegrator(new DiffusionIntegrator(negTauCoef),
                          NULL);
   a->AddDomainIntegrator(new MassIntegrator(rhoCoef),
                          new MassIntegrator(negSigmaCoef));

   BilinearForm *pcOp = new BilinearForm(fespace);
   pcOp->AddDomainIntegrator(new DiffusionIntegrator(negTauCoef));
   pcOp->AddDomainIntegrator(new MassIntegrator(rhoCoef));
   pcOp->AddDomainIntegrator(new MassIntegrator(sigmaCoef));

   // 9. Assemble the parallel bilinear form and the corresponding linear
   //    system, applying any necessary transformations such as: parallel
   //    assembly, eliminating boundary conditions, applying conforming
   //    constraints for non-conforming AMR, etc.
   a->Assemble();
   pcOp->Assemble();

   OperatorHandle A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   x = 0.0;
   X = 0.0;

   SparseMatrix PCOp;
   pcOp->FormSystemMatrix(ess_tdof_list, PCOp);

   // 10. Define and apply a parallel FGMRES solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
   {

#ifdef MFEM_USE_SUITESPARSE
      // 11. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //     solve the system A X = B with FGMRES.
      cout << "Solving with FGMRES using a block diagonal "
           << "Gauss-Seidel preconditioner" << endl;

      Array<int> blockTrueOffsets;
      blockTrueOffsets.SetSize(3);
      blockTrueOffsets[0] = 0;
      blockTrueOffsets[1] = PCOp.Height();
      blockTrueOffsets[2] = PCOp.Height();
      blockTrueOffsets.PartialSum();

      BlockDiagonalPreconditioner BDP(blockTrueOffsets);

      GSSmoother gssr(PCOp);
      ScaledOperator gssi(&gssr,
                          (conv == ComplexOperator::HERMITIAN)?1.0:-1.0);

      BDP.SetDiagonalBlock(0,&gssr);
      BDP.SetDiagonalBlock(1,&gssi);
      BDP.owns_blocks = 0;

      FGMRESSolver fgmres;
      fgmres.SetPreconditioner(BDP);
      fgmres.SetOperator(*A.Ptr());
      fgmres.SetRelTol(1e-12);
      fgmres.SetMaxIter(1000);
      fgmres.SetPrintLevel(1);
      fgmres.Mult(B, X);

#else
      // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
      //     system.
      cout << "Forming SparseMatrix" << endl;
      SparseMatrix * A_sp =
         dynamic_cast<ComplexSparseMatrix*>(A.Ptr())->GetSystemMatrix();

      cout << "Solving with UMFPack" << endl;
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A_sp);
      umf_solver.Mult(B, X);

      delete A_sp;
#endif
   }

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   cout << "Recover solution" << endl;
   a->RecoverFEMSolution(X, *b, x);

   double err_r = x.real().ComputeL2Error(x_r);
   double err_i = x.imag().ComputeL2Error(x_i);

   cout << endl;
   cout << "|| Re (x_h - x) ||_{L^2} = " << err_r << endl;
   cout << "|| Im (x_h - x) ||_{L^2} = " << err_i << endl;
   cout << endl;

   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream sol_r_ofs("sol_r.gf");
      ofstream sol_i_ofs("sol_i.gf");
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      x.real().Save(sol_r_ofs);
      x.imag().Save(sol_i_ofs);
   }

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *mesh << x.real()
                 << "window_title 'Comp Real Part'" << flush;
      sol_sock_i << "solution\n" << *mesh << x.imag()
                 << "window_title 'Comp Imaginary Part'" << flush;
   }
   if (visualization)
   {
      x_exact -= x;

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *mesh << x_exact.real()
                 << "window_title 'Exact-Comp Real Part'" << flush;
      sol_sock_i << "solution\n" << *mesh << x_exact.imag()
                 << "window_title 'Exact-Comp Imaginary Part'" << flush;
   }
   if (visualization)
   {
      GridFunction x_t(fespace);
      x_t = x.real();
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x_t
               << "window_title 'Harmonic Solution'"
               << "pause\n" << flush;
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
      int num_frames = 32;
      int i = 0;
      while (sol_sock)
      {
         add(cos(2.0 * M_PI * (i % num_frames) / num_frames), x.real(),
             sin(2.0 * M_PI * (i % num_frames) / num_frames), x.imag(), x_t);
         sol_sock << "solution\n" << *mesh << x_t << flush;
         i++;
      }
   }

   // 16. Free the used memory.
   delete a;
   delete b;
   // delete pcOp;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}

double x_real_exact(const Vector &x)
{
   int dim = x.Size();

   switch (dim)
   {
      case 1:
         return sin(M_PI * x(0));
         break;
      case 2:
         return sin(M_PI * x(0)) * sin(M_PI * x(1));
         break;
      case 3:
         return sin(M_PI * x(0)) * sin(M_PI * x(1)) * sin(M_PI * x(2));
         break;
   }
   return 0.0;
}

double x_imag_exact(const Vector &x)
{
   int dim = x.Size();

   switch (dim)
   {
      case 1:
         return cos(M_PI * x(0));
         break;
      case 2:
         return cos(M_PI * x(0)) * cos(M_PI * x(1));
         break;
      case 3:
         return cos(M_PI * x(0)) * cos(M_PI * x(1)) * cos(M_PI * x(2));
         break;
   }
   return 0.0;
}

complex<double> f_exact(const Vector &x)
{
   int dim = x.Size();

   double cx = cos(M_PI * x(0));
   double cy = (dim>1)?cos(M_PI * x(1)):1.0;
   double cz = (dim>2)?cos(M_PI * x(2)):1.0;

   double sx = sin(M_PI * x(0));
   double sy = (dim>1)?sin(M_PI * x(1)):1.0;
   double sz = (dim>2)?sin(M_PI * x(2)):1.0;

   complex<double> a(omega_ * sigma_,
                     rho_ * pow(omega_, 2) - pow(M_PI, 2) * tau_ * dim);
   complex<double> z(cx * cy * cz, - sx * sy * sz);

   return a * z;
}

double f_real_exact(const Vector &x)
{
   complex<double> f = f_exact(x);

   return f.real();
}

double f_imag_exact(const Vector &x)
{
   complex<double> f = f_exact(x);

   return f.imag();
}
