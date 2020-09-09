//                       MFEM Example 22 - Parallel Version
//
// Compile with: make ex22p
//
// Sample runs:  mpirun -np 4 ex22p -m ../data/inline-segment.mesh -o 3
//               mpirun -np 4 ex22p -m ../data/inline-tri.mesh -o 3
//               mpirun -np 4 ex22p -m ../data/inline-quad.mesh -o 3
//               mpirun -np 4 ex22p -m ../data/inline-quad.mesh -o 3 -p 1
//               mpirun -np 4 ex22p -m ../data/inline-quad.mesh -o 3 -p 2
//               mpirun -np 4 ex22p -m ../data/inline-quad.mesh -o 1 -p 1 -pa
//               mpirun -np 4 ex22p -m ../data/inline-tet.mesh -o 2
//               mpirun -np 4 ex22p -m ../data/inline-hex.mesh -o 2
//               mpirun -np 4 ex22p -m ../data/inline-hex.mesh -o 2 -p 1
//               mpirun -np 4 ex22p -m ../data/inline-hex.mesh -o 2 -p 2
//               mpirun -np 4 ex22p -m ../data/inline-hex.mesh -o 1 -p 2 -pa
//               mpirun -np 4 ex22p -m ../data/star.mesh -o 2 -sigma 10.0
//
// Device sample runs:
//               mpirun -np 4 ex22p -m ../data/inline-quad.mesh -o 1 -p 1 -pa -d cuda
//               mpirun -np 4 ex22p -m ../data/inline-hex.mesh -o 1 -p 2 -pa -d cuda
//               mpirun -np 4 ex22p -m ../data/star.mesh -o 2 -sigma 10.0 -pa -d cuda
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
//               In electromagnetics the coefficients are typically named the
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
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 1;
   int prob = 0;
   double freq = -1.0;
   double a_coef = 0.0;
   bool visualization = 1;
   bool herm_conv = true;
   bool exact_sol = true;
   bool pa = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
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
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
   if (myid == 0 && exact_sol)
   {
      cout << "Identified a mesh with known exact solution" << endl;
   }

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution.
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange, Nedelec, or Raviart-Thomas finite elements of
   //    the specified order.
   if (dim == 1 && prob != 0 )
   {
      if (myid == 0)
      {
         cout << "Switching to problem type 0, H1 basis functions, "
              << "for 1 dimensional mesh." << endl;
      }
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
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    based on the type of mesh and the problem type.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system.
   ParComplexLinearForm b(fespace, conv);
   b.Vector::operator=(0.0);

   // 10. Define the solution vector u as a parallel complex finite element grid
   //     function corresponding to fespace. Initialize u with initial guess of
   //     1+0i or the exact solution if it is known.
   ParComplexGridFunction u(fespace);
   ParComplexGridFunction * u_exact = NULL;
   if (exact_sol) { u_exact = new ParComplexGridFunction(fespace); }

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
      sol_sock_r << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *pmesh << u_exact->real()
                 << "window_title 'Exact: Real Part'" << flush;
      sol_sock_i << "solution\n" << *pmesh << u_exact->imag()
                 << "window_title 'Exact: Imaginary Part'" << flush;
   }

   // 11. Set up the parallel sesquilinear form a(.,.) on the finite element
   //     space corresponding to the damped harmonic oscillator operator of the
   //     appropriate type:
   //
   //     0) A scalar H1 field
   //        -Div(a Grad) - omega^2 b + i omega c
   //
   //     1) A vector H(Curl) field
   //        Curl(a Curl) - omega^2 b + i omega c
   //
   //     2) A vector H(Div) field
   //        -Grad(a Div) - omega^2 b + i omega c
   //
   ConstantCoefficient stiffnessCoef(1.0/mu_);
   ConstantCoefficient massCoef(-omega_ * omega_ * epsilon_);
   ConstantCoefficient lossCoef(omega_ * sigma_);
   ConstantCoefficient negMassCoef(omega_ * omega_ * epsilon_);

   ParSesquilinearForm *a = new ParSesquilinearForm(fespace, conv);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
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

   // 11a. Set up the parallel bilinear form for the preconditioner
   //      corresponding to the appropriate operator
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
   ParBilinearForm *pcOp = new ParBilinearForm(fespace);
   if (pa) { pcOp->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
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

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, etc.
   a->Assemble();
   pcOp->Assemble();

   OperatorHandle A;
   Vector B, U;

   a->FormLinearSystem(ess_tdof_list, u, b, A, U, B);

   if (myid == 0)
   {
      cout << "Size of linear system: "
           << 2 * fespace->GlobalTrueVSize() << endl << endl;
   }

   // 13. Define and apply a parallel FGMRES solver for AU=B with a block
   //     diagonal preconditioner based on the appropriate multigrid
   //     preconditioner from hypre.
   {
      Array<int> blockTrueOffsets;
      blockTrueOffsets.SetSize(3);
      blockTrueOffsets[0] = 0;
      blockTrueOffsets[1] = A->Height() / 2;
      blockTrueOffsets[2] = A->Height() / 2;
      blockTrueOffsets.PartialSum();

      BlockDiagonalPreconditioner BDP(blockTrueOffsets);

      Operator * pc_r = NULL;
      Operator * pc_i = NULL;

      if (pa)
      {
         pc_r = new OperatorJacobiSmoother(*pcOp, ess_tdof_list);
      }
      else
      {
         OperatorHandle PCOp;
         pcOp->FormSystemMatrix(ess_tdof_list, PCOp);
         switch (prob)
         {
            case 0:
               pc_r = new HypreBoomerAMG(*PCOp.As<HypreParMatrix>());
               break;
            case 1:
               pc_r = new HypreAMS(*PCOp.As<HypreParMatrix>(), fespace);
               break;
            case 2:
               if (dim == 2 )
               {
                  pc_r = new HypreAMS(*PCOp.As<HypreParMatrix>(), fespace);
               }
               else
               {
                  pc_r = new HypreADS(*PCOp.As<HypreParMatrix>(), fespace);
               }
               break;
            default: break; // This should be unreachable
         }
      }
      pc_i = new ScaledOperator(pc_r,
                                (conv == ComplexOperator::HERMITIAN) ?
                                -1.0:1.0);

      BDP.SetDiagonalBlock(0, pc_r);
      BDP.SetDiagonalBlock(1, pc_i);
      BDP.owns_blocks = 1;

      FGMRESSolver fgmres(MPI_COMM_WORLD);
      fgmres.SetPreconditioner(BDP);
      fgmres.SetOperator(*A.Ptr());
      fgmres.SetRelTol(1e-12);
      fgmres.SetMaxIter(1000);
      fgmres.SetPrintLevel(1);
      fgmres.Mult(B, U);
   }
   // 14. Recover the parallel grid function corresponding to U. This is the
   //     local finite element solution on each processor.
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

      if ( myid == 0 )
      {
         cout << endl;
         cout << "|| Re (u_h - u) ||_{L^2} = " << err_r << endl;
         cout << "|| Im (u_h - u) ||_{L^2} = " << err_i << endl;
         cout << endl;
      }
   }

   // 15. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_r_name, sol_i_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_r_name << "sol_r." << setfill('0') << setw(6) << myid;
      sol_i_name << "sol_i." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_r_ofs(sol_r_name.str().c_str());
      ofstream sol_i_ofs(sol_i_name.str().c_str());
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      u.real().Save(sol_r_ofs);
      u.imag().Save(sol_i_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *pmesh << u.real()
                 << "window_title 'Solution: Real Part'" << flush;
      sol_sock_i << "solution\n" << *pmesh << u.imag()
                 << "window_title 'Solution: Imaginary Part'" << flush;
   }
   if (visualization && exact_sol)
   {
      *u_exact -= u;

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *pmesh << u_exact->real()
                 << "window_title 'Error: Real Part'" << flush;
      sol_sock_i << "solution\n" << *pmesh << u_exact->imag()
                 << "window_title 'Error: Imaginary Part'" << flush;
   }
   if (visualization)
   {
      ParGridFunction u_t(fespace);
      u_t = u.real();
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << u_t
               << "window_title 'Harmonic Solution (t = 0.0 T)'"
               << "pause\n" << flush;
      if (myid == 0)
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
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << *pmesh << u_t
                  << "window_title '" << oss.str() << "'" << flush;
         i++;
      }
   }

   // 17. Free the used memory.
   delete a;
   delete u_exact;
   delete pcOp;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

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
