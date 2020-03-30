//                               MFEM Example 22
//
// Compile with: make ex22
//
// Sample runs:  ex22 -m ../data/inline-segment.mesh -o 3
//               ex22 -m ../data/inline-tri.mesh -o 3
//               ex22 -m ../data/inline-quad.mesh -o 3
//               ex22 -m ../data/inline-quad.mesh -o 3 -p 1
//               ex22 -m ../data/inline-quad.mesh -o 3 -p 2
//               ex22 -m ../data/inline-tet.mesh -o 2
//               ex22 -m ../data/inline-hex.mesh -o 2
//               ex22 -m ../data/inline-hex.mesh -o 2 -p 1
//               ex22 -m ../data/inline-hex.mesh -o 2 -p 2
//               ex22 -m ../data/star.mesh -r 1 -o 2 -sigma 10.0
//
// Description:  This example code demonstrates the use of MFEM to define and
//               solve simple complex-valued linear systems. It implements three
//               variants of a damped harmonic oscillator:
//
//               1) A scalar H1 field
//                  -Div(a Grad u) - omega^2 b u + i omega c u = 0
//
//               2) A vector H(Curl) field
//                  Curl(a Curl u) - omega^2 b u + i omega c u = 0
//
//               3) A vector H(Div) field
//                  -Grad(a Div u) - omega^2 b u + i omega c u = 0
//
//               In each case the field is driven by a forced oscillation, with
//               angular frequency omega, imposed at the boundary or a portion
//               of the boundary.
//
//               In electromagnetics, the coefficients are typically named the
//               permeability, mu = 1/a, permittivity, epsilon = b, and
//               conductivity, sigma = c. The user can specify these constants
//               using either set of names.
//
//               The example also demonstrates how to display a time-varying
//               solution as a sequence of fields sent to a single GLVis socket.
//
//               We recommend viewing examples 1, 3 and 4 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double mu_ = 1.0;
static double epsilon_ = 1.0;
static double sigma_ = 20.0;
static double omega_ = 10.0;

double u0_real_exact(const Vector &);
double u0_imag_exact(const Vector &);

void u1_real_exact(const Vector &, Vector &);
void u1_imag_exact(const Vector &, Vector &);

void u2_real_exact(const Vector &, Vector &);
void u2_imag_exact(const Vector &, Vector &);

bool check_for_inline_mesh(const char * mesh_file);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_levels = 0;
   int order = 1;
   int prob = 0;
   double freq = -1.0;
   double a_coef = 0.0;
   bool visualization = 1;
   bool herm_conv = true;
   bool exact_sol = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&prob, "-p", "--problem-type",
                  "Choose between 0: H_1, 1: H(Curl), or 2: H(Div) "
                  "damped harmonic oscillator.");
   args.AddOption(&a_coef, "-a", "--stiffness-coef",
                  "Stiffness coefficient (spring constant or 1/mu).");
   args.AddOption(&epsilon_, "-b", "--mass-coef",
                  "Mass coefficient (or epsilon).");
   args.AddOption(&sigma_, "-c", "--damping-coef",
                  "Damping coefficient (or sigma).");
   args.AddOption(&mu_, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon_, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&sigma_, "-sigma", "--conductivity",
                  "Conductivity (or damping constant).");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency (in Hz).");
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

   MFEM_VERIFY(prob >= 0 && prob <=2,
               "Unrecognized problem type: " << prob);

   if ( a_coef != 0.0 )
   {
      mu_ = 1.0 / a_coef;
   }
   if ( freq > 0.0 )
   {
      omega_ = 2.0 * M_PI * freq;
   }

   exact_sol = check_for_inline_mesh(mesh_file);
   if (exact_sol)
   {
      cout << "Identified a mesh with known exact solution" << endl;
   }

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase resolution. In this example we do
   //    'ref_levels' of uniform refinement where the user specifies
   //    the number of levels with the '-r' option.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange, Nedelec, or Raviart-Thomas finite elements of the specified
   //    order.
   if (dim == 1 && prob != 0 )
   {
      cout << "Switching to problem type 0, H1 basis functions, "
           << "for 1 dimensional mesh." << endl;
      prob = 0;
   }

   FiniteElementCollection *fec = NULL;
   switch (prob)
   {
      case 0:  fec = new H1_FECollection(order, dim);      break;
      case 1:  fec = new ND_FECollection(order, dim);      break;
      case 2:  fec = new RT_FECollection(order - 1, dim);  break;
      default: break; // This should be unreachable
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined based on the type
   //    of mesh and the problem type.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   ComplexLinearForm b(fespace, conv);
   b.Vector::operator=(0.0);

   // 7. Define the solution vector u as a complex finite element grid function
   //    corresponding to fespace. Initialize u with initial guess of 1+0i or
   //    the exact solution if it is known.
   ComplexGridFunction u(fespace);
   ComplexGridFunction * u_exact = NULL;
   if (exact_sol) { u_exact = new ComplexGridFunction(fespace); }

   FunctionCoefficient u0_r(u0_real_exact);
   FunctionCoefficient u0_i(u0_imag_exact);
   VectorFunctionCoefficient u1_r(dim, u1_real_exact);
   VectorFunctionCoefficient u1_i(dim, u1_imag_exact);
   VectorFunctionCoefficient u2_r(dim, u2_real_exact);
   VectorFunctionCoefficient u2_i(dim, u2_imag_exact);

   ConstantCoefficient zeroCoef(0.0);
   ConstantCoefficient oneCoef(1.0);

   Vector zeroVec(dim); zeroVec = 0.0;
   Vector  oneVec(dim);  oneVec = 0.0; oneVec[(prob==2)?(dim-1):0] = 1.0;
   VectorConstantCoefficient zeroVecCoef(zeroVec);
   VectorConstantCoefficient oneVecCoef(oneVec);

   u = 0.0;
   switch (prob)
   {
      case 0:
         if (exact_sol)
         {
            u.ProjectBdrCoefficient(u0_r, u0_i, ess_bdr);
            u_exact->ProjectCoefficient(u0_r, u0_i);
         }
         else
         {
            u.ProjectBdrCoefficient(oneCoef, zeroCoef, ess_bdr);
         }
         break;
      case 1:
         if (exact_sol)
         {
            u.ProjectBdrCoefficientTangent(u1_r, u1_i, ess_bdr);
            u_exact->ProjectCoefficient(u1_r, u1_i);
         }
         else
         {
            u.ProjectBdrCoefficientTangent(oneVecCoef, zeroVecCoef, ess_bdr);
         }
         break;
      case 2:
         if (exact_sol)
         {
            u.ProjectBdrCoefficientNormal(u2_r, u2_i, ess_bdr);
            u_exact->ProjectCoefficient(u2_r, u2_i);
         }
         else
         {
            u.ProjectBdrCoefficientNormal(oneVecCoef, zeroVecCoef, ess_bdr);
         }
         break;
      default: break; // This should be unreachable
   }

   if (visualization && exact_sol)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *mesh << u_exact->real()
                 << "window_title 'Exact: Real Part'" << flush;
      sol_sock_i << "solution\n" << *mesh << u_exact->imag()
                 << "window_title 'Exact: Imaginary Part'" << flush;
   }

   // 8. Set up the sesquilinear form a(.,.) on the finite element space
   //    corresponding to the damped harmonic oscillator operator of the
   //    appropriate type:
   //
   //    0) A scalar H1 field
   //       -Div(a Grad) - omega^2 b + i omega c
   //
   //    1) A vector H(Curl) field
   //       Curl(a Curl) - omega^2 b + i omega c
   //
   //    2) A vector H(Div) field
   //       -Grad(a Div) - omega^2 b + i omega c
   //
   ConstantCoefficient stiffnessCoef(1.0/mu_);
   ConstantCoefficient massCoef(-omega_ * omega_ * epsilon_);
   ConstantCoefficient lossCoef(omega_ * sigma_);
   ConstantCoefficient negMassCoef(omega_ * omega_ * epsilon_);

   SesquilinearForm *a = new SesquilinearForm(fespace, conv);
   switch (prob)
   {
      case 0:
         a->AddDomainIntegrator(new DiffusionIntegrator(stiffnessCoef),
                                NULL);
         a->AddDomainIntegrator(new MassIntegrator(massCoef),
                                new MassIntegrator(lossCoef));
         break;
      case 1:
         a->AddDomainIntegrator(new CurlCurlIntegrator(stiffnessCoef),
                                NULL);
         a->AddDomainIntegrator(new VectorFEMassIntegrator(massCoef),
                                new VectorFEMassIntegrator(lossCoef));
         break;
      case 2:
         a->AddDomainIntegrator(new DivDivIntegrator(stiffnessCoef),
                                NULL);
         a->AddDomainIntegrator(new VectorFEMassIntegrator(massCoef),
                                new VectorFEMassIntegrator(lossCoef));
         break;
      default: break; // This should be unreachable
   }

   // 8a. Set up the bilinear form for the preconditioner corresponding to the
   //     appropriate operator
   //
   //      0) A scalar H1 field
   //         -Div(a Grad) - omega^2 b + omega c
   //
   //      1) A vector H(Curl) field
   //         Curl(a Curl) + omega^2 b + omega c
   //
   //      2) A vector H(Div) field
   //         -Grad(a Div) - omega^2 b + omega c
   //
   BilinearForm *pcOp = new BilinearForm(fespace);
   switch (prob)
   {
      case 0:
         pcOp->AddDomainIntegrator(new DiffusionIntegrator(stiffnessCoef));
         pcOp->AddDomainIntegrator(new MassIntegrator(massCoef));
         pcOp->AddDomainIntegrator(new MassIntegrator(lossCoef));
         break;
      case 1:
         pcOp->AddDomainIntegrator(new CurlCurlIntegrator(stiffnessCoef));
         pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(negMassCoef));
         pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(lossCoef));
         break;
      case 2:
         pcOp->AddDomainIntegrator(new DivDivIntegrator(stiffnessCoef));
         pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(massCoef));
         pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(lossCoef));
         break;
      default: break; // This should be unreachable
   }

   // 9. Assemble the form and the corresponding linear system, applying any
   //    necessary transformations such as: assembly, eliminating boundary
   //    conditions, conforming constraints for non-conforming AMR, etc.
   a->Assemble();
   pcOp->Assemble();

   OperatorHandle A;
   Vector B, U;

   a->FormLinearSystem(ess_tdof_list, u, b, A, U, B);
   u = 0.0;
   U = 0.0;

   OperatorHandle PCOp;
   pcOp->FormSystemMatrix(ess_tdof_list, PCOp);

   {
      ComplexSparseMatrix * Asp =
         dynamic_cast<ComplexSparseMatrix*>(A.Ptr());

      cout << "Size of linear system: "
           << 2 * Asp->real().Width() << endl << endl;
   }

   // 10. Define and apply a GMRES solver for AU=B with a block diagonal
   //     preconditioner based on the appropriate sparse smoother.
   {
      Array<int> blockOffsets;
      blockOffsets.SetSize(3);
      blockOffsets[0] = 0;
      blockOffsets[1] = PCOp.Ptr()->Height();
      blockOffsets[2] = PCOp.Ptr()->Height();
      blockOffsets.PartialSum();

      BlockDiagonalPreconditioner BDP(blockOffsets);

      Operator * pc_r = NULL;
      Operator * pc_i = NULL;

      double s = 1.0;
      switch (prob)
      {
         case 0:
            pc_r = new DSmoother(*PCOp.As<SparseMatrix>());
            break;
         case 1:
            pc_r = new GSSmoother(*PCOp.As<SparseMatrix>());
            s = -1.0;
            break;
         case 2:
            pc_r = new DSmoother(*PCOp.As<SparseMatrix>());
            break;

         default: break; // This should be unreachable
      }
      pc_i = new ScaledOperator(pc_r,
                                (conv == ComplexOperator::HERMITIAN) ?
                                s:-s);

      BDP.SetDiagonalBlock(0, pc_r);
      BDP.SetDiagonalBlock(1, pc_i);
      BDP.owns_blocks = 1;

      GMRESSolver gmres;
      gmres.SetPreconditioner(BDP);
      gmres.SetOperator(*A.Ptr());
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(1000);
      gmres.SetPrintLevel(1);
      gmres.Mult(B, U);
   }

   // 11. Recover the solution as a finite element grid function and compute the
   //     errors if the exact solution is known.
   a->RecoverFEMSolution(U, b, u);

   if (exact_sol)
   {
      double err_r = -1.0;
      double err_i = -1.0;

      switch (prob)
      {
         case 0:
            err_r = u.real().ComputeL2Error(u0_r);
            err_i = u.imag().ComputeL2Error(u0_i);
            break;
         case 1:
            err_r = u.real().ComputeL2Error(u1_r);
            err_i = u.imag().ComputeL2Error(u1_i);
            break;
         case 2:
            err_r = u.real().ComputeL2Error(u2_r);
            err_i = u.imag().ComputeL2Error(u2_i);
            break;
         default: break; // This should be unreachable
      }

      cout << endl;
      cout << "|| Re (u_h - u) ||_{L^2} = " << err_r << endl;
      cout << "|| Im (u_h - u) ||_{L^2} = " << err_i << endl;
      cout << endl;
   }

   // 12. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m mesh -g sol".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream sol_r_ofs("sol_r.gf");
      ofstream sol_i_ofs("sol_i.gf");
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      u.real().Save(sol_r_ofs);
      u.imag().Save(sol_i_ofs);
   }

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *mesh << u.real()
                 << "window_title 'Solution: Real Part'" << flush;
      sol_sock_i << "solution\n" << *mesh << u.imag()
                 << "window_title 'Solution: Imaginary Part'" << flush;
   }
   if (visualization && exact_sol)
   {
      *u_exact -= u;

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *mesh << u_exact->real()
                 << "window_title 'Error: Real Part'" << flush;
      sol_sock_i << "solution\n" << *mesh << u_exact->imag()
                 << "window_title 'Error: Imaginary Part'" << flush;
   }
   if (visualization)
   {
      GridFunction u_t(fespace);
      u_t = u.real();
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << u_t
               << "window_title 'Harmonic Solution (t = 0.0 T)'"
               << "pause\n" << flush;

      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
      int num_frames = 32;
      int i = 0;
      while (sol_sock)
      {
         double t = (double)(i % num_frames) / num_frames;
         ostringstream oss;
         oss << "Harmonic Solution (t = " << t << " T)";

         add(cos( 2.0 * M_PI * t), u.real(),
             sin(-2.0 * M_PI * t), u.imag(), u_t);
         sol_sock << "solution\n" << *mesh << u_t
                  << "window_title '" << oss.str() << "'" << flush;
         i++;
      }
   }

   // 14. Free the used memory.
   delete a;
   delete u_exact;
   delete pcOp;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

bool check_for_inline_mesh(const char * mesh_file)
{
   string file(mesh_file);
   size_t p0 = file.find_last_of("/");
   string s0 = file.substr((p0==string::npos)?0:(p0+1),7);
   return s0 == "inline-";
}

complex<double> u0_exact(const Vector &x)
{
   int dim = x.Size();
   complex<double> i(0.0, 1.0);
   complex<double> alpha = (epsilon_ * omega_ - i * sigma_);
   complex<double> kappa = std::sqrt(mu_ * omega_* alpha);
   return std::exp(-i * kappa * x[dim - 1]);
}

double u0_real_exact(const Vector &x)
{
   return u0_exact(x).real();
}

double u0_imag_exact(const Vector &x)
{
   return u0_exact(x).imag();
}

void u1_real_exact(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim); v = 0.0; v[0] = u0_real_exact(x);
}

void u1_imag_exact(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim); v = 0.0; v[0] = u0_imag_exact(x);
}

void u2_real_exact(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim); v = 0.0; v[dim-1] = u0_real_exact(x);
}

void u2_imag_exact(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim); v = 0.0; v[dim-1] = u0_imag_exact(x);
}
