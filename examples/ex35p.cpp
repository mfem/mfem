//                       MFEM Example 35 - Parallel Version
//
// Compile with: make ex35p
//
// Sample runs:  mpirun -np 4 ex35p -p 0 -o 2
//               mpirun -np 4 ex35p -p 0 -o 2 -pbc '22 23 24' -em 0
//               mpirun -np 4 ex35p -p 1 -o 1 -rp 2
//               mpirun -np 4 ex35p -p 1 -o 2
//               mpirun -np 4 ex35p -p 2 -o 1 -rp 2 -c 15
//
// Device sample runs:
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
//               of the boundary. The spatial variation of the boundary
//               condition is computed as an eigenmode of an appropriate
//               operator defined on a portion of the boundary i.e. a port
//               boundary condition.
//
//               In electromagnetics the coefficients are typically named the
//               permeability, mu = 1/a, permittivity, epsilon = b, and
//               conductivity, sigma = c. The user can specify these constants
//               using either set of names.
//
//               This example demonstrates how to transfer fields computed on a
//               boundary generated SubMesh to the full mesh and apply them as
//               boundary conditions. The default mesh and corresponding
//               boundary attributes were chosen to verify proper behavior on
//               both triangular and quadrilateral faces of tetrahedral,
//               wedge-shaped, and hexahedral elements.
//
//               The example also demonstrates how to display a time-varying
//               solution as a sequence of fields sent to a single GLVis socket.
//
//               We recommend viewing examples 11, 13, and 22 before viewing
//               this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static real_t mu_ = 1.0;
static real_t epsilon_ = 1.0;
static real_t sigma_ = 2.0;

void SetPortBC(int prob, int dim, int mode, ParGridFunction &port_bc);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/fichera-mixed.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 1;
   Array<int> port_bc_attr;
   int prob = 0;
   int mode = 1;
   real_t freq = -1.0;
   real_t omega = 2.0 * M_PI;
   real_t a_coef = 0.0;
   bool herm_conv = true;
   bool slu_solver  = false;
   bool visualization = 1;
   bool mixed = true;
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
   args.AddOption(&mode, "-em", "--eigenmode",
                  "Choose the index of the port eigenmode.");
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
   args.AddOption(&port_bc_attr, "-pbc", "--port-bc-attr",
                  "Attributes of port boundary condition");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
#ifdef MFEM_USE_SUPERLU
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&mixed, "-mixed", "--mixed-mesh", "-hex",
                  "--hex-mesh", "Mixed mesh of hexahedral mesh.");
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
      return 1;
   }

   if (!mixed || pa)
   {
      mesh_file = "../data/fichera.mesh";
   }

   if ( a_coef != 0.0 )
   {
      mu_ = 1.0 / a_coef;
   }
   if ( freq > 0.0 )
   {
      omega = 2.0 * M_PI * freq;
   }
   if (port_bc_attr.Size() == 0 &&
       (strcmp(mesh_file, "../data/fichera-mixed.mesh") == 0 ||
        strcmp(mesh_file, "../data/fichera.mesh") == 0))
   {
      port_bc_attr.SetSize(4);
      port_bc_attr[0] =  7;
      port_bc_attr[1] =  8;
      port_bc_attr[2] = 11;
      port_bc_attr[3] = 12;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   MFEM_VERIFY(prob >= 0 && prob <=2,
               "Unrecognized problem type: " << prob);

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

   // 6a. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // 6b. Extract a submesh covering a portion of the boundary
   ParSubMesh pmesh_port(ParSubMesh::CreateFromBoundary(pmesh, port_bc_attr));

   // 7a. Define a parallel finite element space on the parallel mesh. Here we
   //     use continuous Lagrange, Nedelec, or Raviart-Thomas finite elements
   //     of the specified order.
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
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 7b. Define a parallel finite element space on the sub-mesh. Here we
   //    use continuous Lagrange, Nedelec, or L2 finite elements of
   //    the specified order.
   FiniteElementCollection *fec_port = NULL;
   switch (prob)
   {
      case 0:  fec_port = new H1_FECollection(order, dim-1);      break;
      case 1:
         if (dim == 3)
         {
            fec_port = new ND_FECollection(order, dim-1);
         }
         else
         {
            fec_port = new L2_FECollection(order - 1, dim-1,
                                           BasisType::GaussLegendre,
                                           FiniteElement::INTEGRAL);
         }
         break;
      case 2:  fec_port = new L2_FECollection(order - 1, dim-1,
                                                 BasisType::GaussLegendre,
                                                 FiniteElement::INTEGRAL); break;
      default: break; // This should be unreachable
   }
   ParFiniteElementSpace fespace_port(&pmesh_port, fec_port);
   HYPRE_BigInt size_port = fespace_port.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element port BC unknowns: " << size_port
           << endl;
   }

   // 8a. Define a parallel grid function on the SubMesh which will contain
   //     the field to be applied as a port boundary condition.
   ParGridFunction port_bc(&fespace_port);
   port_bc = 0.0;

   SetPortBC(prob, dim, mode, port_bc);

   // 8b. Save the SubMesh and associated port boundary condition in parallel.
   //     This output can be viewed later using GLVis:
   //        "glvis -np <np> -m port_mesh -g port_mode"
   {
      ostringstream mesh_name, port_name;
      mesh_name << "port_mesh." << setfill('0') << setw(6) << myid;
      port_name << "port_mode." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh_port.Print(mesh_ofs);

      ofstream port_ofs(port_name.str().c_str());
      port_ofs.precision(8);
      port_bc.Save(port_ofs);
   }
   // 8c. Send the port bc, computed on the SubMesh, to a GLVis server.
   if (visualization && dim == 3)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream port_sock(vishost, visport);
      port_sock << "parallel " << num_procs << " " << myid << "\n";
      port_sock.precision(8);
      port_sock << "solution\n" << pmesh_port << port_bc
                << "window_title 'Port BC'"
                << "window_geometry 0 0 400 350" << flush;
   }

   // 9. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    using an eigenmode of the appropriate type computed on the SubMesh.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 10. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system.
   ParComplexLinearForm b(&fespace, conv);
   b.Vector::operator=(0.0);

   // 11a. Define the solution vector u as a parallel complex finite element
   //      grid function corresponding to fespace. Initialize u to equal zero.
   ParComplexGridFunction u(&fespace);
   u = 0.0;
   pmesh_port.Transfer(port_bc, u.real());

   // 11b. Send the transferred port bc field to a GLVis server.
   {
      ParGridFunction full_bc(&fespace);
      ParTransferMap port_to_full(port_bc, full_bc);

      full_bc = 0.0;
      port_to_full.Transfer(port_bc, full_bc);

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream full_sock(vishost, visport);
         full_sock << "parallel " << num_procs << " " << myid << "\n";
         full_sock.precision(8);
         full_sock << "solution\n" << pmesh << full_bc
                   << "window_title 'Transferred BC'"
                   << "window_geometry 400 0 400 350"<< flush;
      }
   }

   // 12. Set up the parallel sesquilinear form a(.,.) on the finite element
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
   ConstantCoefficient massCoef(-omega * omega * epsilon_);
   ConstantCoefficient lossCoef(omega * sigma_);
   ConstantCoefficient negMassCoef(omega * omega * epsilon_);

   ParSesquilinearForm a(&fespace, conv);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   switch (prob)
   {
      case 0:
         a.AddDomainIntegrator(new DiffusionIntegrator(stiffnessCoef),
                               NULL);
         a.AddDomainIntegrator(new MassIntegrator(massCoef),
                               new MassIntegrator(lossCoef));
         break;
      case 1:
         a.AddDomainIntegrator(new CurlCurlIntegrator(stiffnessCoef),
                               NULL);
         a.AddDomainIntegrator(new VectorFEMassIntegrator(massCoef),
                               new VectorFEMassIntegrator(lossCoef));
         break;
      case 2:
         a.AddDomainIntegrator(new DivDivIntegrator(stiffnessCoef),
                               NULL);
         a.AddDomainIntegrator(new VectorFEMassIntegrator(massCoef),
                               new VectorFEMassIntegrator(lossCoef));
         break;
      default: break; // This should be unreachable
   }

   // 13. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, etc.
   a.Assemble();

   OperatorHandle A;
   Vector B, U;

   a.FormLinearSystem(ess_tdof_list, u, b, A, U, B);

   if (myid == 0)
   {
      cout << "Size of linear system: "
           << 2 * size << endl << endl;
   }

   if (!slu_solver)
   {
      // 14a. Set up the parallel bilinear form for the preconditioner
      //      corresponding to the appropriate operator
      //
      //      0) A scalar H1 field
      //         -Div(a Grad) - omega^2 b + i omega c
      //
      //      1) A vector H(Curl) field
      //         Curl(a Curl) + omega^2 b + i omega c
      //
      //      2) A vector H(Div) field
      //         -Grad(a Div) - omega^2 b + i omega c
      ParBilinearForm pcOp(&fespace);
      if (pa) { pcOp.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      switch (prob)
      {
         case 0:
            pcOp.AddDomainIntegrator(new DiffusionIntegrator(stiffnessCoef));
            pcOp.AddDomainIntegrator(new MassIntegrator(massCoef));
            pcOp.AddDomainIntegrator(new MassIntegrator(lossCoef));
            break;
         case 1:
            pcOp.AddDomainIntegrator(new CurlCurlIntegrator(stiffnessCoef));
            pcOp.AddDomainIntegrator(new VectorFEMassIntegrator(negMassCoef));
            pcOp.AddDomainIntegrator(new VectorFEMassIntegrator(lossCoef));
            break;
         case 2:
            pcOp.AddDomainIntegrator(new DivDivIntegrator(stiffnessCoef));
            pcOp.AddDomainIntegrator(new VectorFEMassIntegrator(massCoef));
            pcOp.AddDomainIntegrator(new VectorFEMassIntegrator(lossCoef));
            break;
         default: break; // This should be unreachable
      }
      pcOp.Assemble();

      // 14b. Define and apply a parallel FGMRES solver for AU=B with a block
      //      diagonal preconditioner based on the appropriate multigrid
      //      preconditioner from hypre.
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
         pc_r = new OperatorJacobiSmoother(pcOp, ess_tdof_list);
      }
      else
      {
         OperatorHandle PCOp;
         pcOp.FormSystemMatrix(ess_tdof_list, PCOp);

         switch (prob)
         {
            case 0:
               pc_r = new HypreBoomerAMG(*PCOp.As<HypreParMatrix>());
               break;
            case 1:
               pc_r = new HypreAMS(*PCOp.As<HypreParMatrix>(), &fespace);
               break;
            case 2:
               if (dim == 2 )
               {
                  pc_r = new HypreAMS(*PCOp.As<HypreParMatrix>(), &fespace);
               }
               else
               {
                  pc_r = new HypreADS(*PCOp.As<HypreParMatrix>(), &fespace);
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
      fgmres.SetRelTol(1e-6);
      fgmres.SetMaxIter(1000);
      fgmres.SetPrintLevel(1);
      fgmres.Mult(B, U);
   }
#ifdef MFEM_USE_SUPERLU
   else
   {
      // 14. Solve using a direct solver
      // Transform to monolithic HypreParMatrix
      HypreParMatrix *A_hyp = A.As<ComplexHypreParMatrix>()->GetSystemMatrix();
      SuperLURowLocMatrix SA(*A_hyp);
      SuperLUSolver superlu(MPI_COMM_WORLD);
      superlu.SetPrintStatistics(true);
      superlu.SetSymmetricPattern(false);
      superlu.SetColumnPermutation(superlu::PARMETIS);
      superlu.SetOperator(SA);
      superlu.Mult(B, U);
      delete A_hyp;
   }
#endif

   // 15. Recover the parallel grid function corresponding to U. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(U, b, u);

   // 16. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol_r" or
   //     "glvis -np <np> -m mesh -g sol_i".
   {
      ostringstream mesh_name, sol_r_name, sol_i_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_r_name << "sol_r." << setfill('0') << setw(6) << myid;
      sol_i_name << "sol_i." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_r_ofs(sol_r_name.str().c_str());
      ofstream sol_i_ofs(sol_i_name.str().c_str());
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      u.real().Save(sol_r_ofs);
      u.imag().Save(sol_i_ofs);
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      sol_sock_r << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_r.precision(8);
      sol_sock_r << "solution\n" << pmesh << u.real()
                 << "window_title 'Solution: Real Part'"
                 << "window_geometry 800 0 400 350" << flush;

      MPI_Barrier(MPI_COMM_WORLD);

      socketstream sol_sock_i(vishost, visport);
      sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_i.precision(8);
      sol_sock_i << "solution\n" << pmesh << u.imag()
                 << "window_title 'Solution: Imaginary Part'"
                 << "window_geometry 1200 0 400 350" << flush;
   }
   if (visualization)
   {
      ParGridFunction u_t(&fespace);
      u_t = u.real();
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << u_t
               << "window_title 'Harmonic Solution (t = 0.0 T)'"
               << "window_geometry 0 432 600 450"
               << "pause\n" << flush;
      if (myid == 0)
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      int num_frames = 32;
      int i = 0;
      while (sol_sock)
      {
         real_t t = (real_t)(i % num_frames) / num_frames;
         ostringstream oss;
         oss << "Harmonic Solution (t = " << t << " T)";

         add(cos( 2.0 * M_PI * t), u.real(),
             sin(-2.0 * M_PI * t), u.imag(), u_t);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << u_t
                  << "window_title '" << oss.str() << "'" << flush;
         i++;
      }
   }

   // 18. Free the used memory.
   delete fec_port;
   delete fec;

   return 0;
}

/**
   Solves the eigenvalue problem -Div(Grad x) = lambda x with homogeneous
   Dirichlet boundary conditions on the boundary of the domain. Returns mode
   number "mode" (counting from zero) in the ParGridFunction "x".
*/
void ScalarWaveGuide(int mode, ParGridFunction &x)
{
   int nev = std::max(mode + 2, 5);
   int seed = 75;

   ParFiniteElementSpace &fespace = *x.ParFESpace();
   ParMesh &pmesh = *fespace.GetParMesh();

   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();
   a.EliminateEssentialBCDiag(ess_bdr, 1.0);
   a.Finalize();

   ParBilinearForm m(&fespace);
   m.AddDomainIntegrator(new MassIntegrator);
   m.Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<real_t>::min());
   m.Finalize();

   HypreParMatrix *A = a.ParallelAssemble();
   HypreParMatrix *M = m.ParallelAssemble();

   HypreBoomerAMG amg(*A);
   amg.SetPrintLevel(0);

   HypreLOBPCG lobpcg(MPI_COMM_WORLD);
   lobpcg.SetNumModes(nev);
   lobpcg.SetRandomSeed(seed);
   lobpcg.SetPreconditioner(amg);
   lobpcg.SetMaxIter(200);
   lobpcg.SetTol(1e-8);
   lobpcg.SetPrecondUsageMode(1);
   lobpcg.SetPrintLevel(1);
   lobpcg.SetMassMatrix(*M);
   lobpcg.SetOperator(*A);
   lobpcg.Solve();

   x = lobpcg.GetEigenvector(mode);

   delete A;
   delete M;
}

/**
   Solves the eigenvalue problem -Curl(Curl x) = lambda x with homogeneous
   Dirichlet boundary conditions, on the tangential component of x, on the
   boundary of the domain. Returns mode number "mode" (counting from zero) in
   the ParGridFunction "x".
*/
void VectorWaveGuide(int mode, ParGridFunction &x)
{
   int nev = std::max(mode + 2, 5);

   ParFiniteElementSpace &fespace = *x.ParFESpace();
   ParMesh &pmesh = *fespace.GetParMesh();

   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator);
   a.Assemble();
   a.EliminateEssentialBCDiag(ess_bdr, 1.0);
   a.Finalize();

   ParBilinearForm m(&fespace);
   m.AddDomainIntegrator(new VectorFEMassIntegrator);
   m.Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<real_t>::min());
   m.Finalize();

   HypreParMatrix *A = a.ParallelAssemble();
   HypreParMatrix *M = m.ParallelAssemble();

   HypreAMS ams(*A,&fespace);
   ams.SetPrintLevel(0);
   ams.SetSingularProblem();

   HypreAME ame(MPI_COMM_WORLD);
   ame.SetNumModes(nev);
   ame.SetPreconditioner(ams);
   ame.SetMaxIter(100);
   ame.SetTol(1e-8);
   ame.SetPrintLevel(1);
   ame.SetMassMatrix(*M);
   ame.SetOperator(*A);
   ame.Solve();

   x = ame.GetEigenvector(mode);

   delete A;
   delete M;
}

/**
   Solves the eigenvalue problem -Div(Grad x) = lambda x with homogeneous
   Neumann boundary conditions on the boundary of the domain. Returns mode
   number "mode" (counting from zero) in the ParGridFunction "x_l2". Note that
   mode 0 is a constant field so higher mode numbers are often more
   interesting. The eigenmode is solved using continuous H1 basis of the
   appropriate order and then projected onto the L2 basis and returned.
*/
void PseudoScalarWaveGuide(int mode, ParGridFunction &x_l2)
{
   int nev = std::max(mode + 2, 5);
   int seed = 75;

   ParFiniteElementSpace &fespace_l2 = *x_l2.ParFESpace();
   ParMesh &pmesh = *fespace_l2.GetParMesh();
   int order_l2 = fespace_l2.FEColl()->GetOrder();

   H1_FECollection fec(order_l2+1, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, &fec);
   ParGridFunction x(&fespace);
   x = 0.0;

   GridFunctionCoefficient xCoef(&x);

   if (mode == 0)
   {
      x = 1.0;
      x_l2.ProjectCoefficient(xCoef);
      return;
   }

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator); // Shift eigenvalues by 1
   a.Assemble();
   a.Finalize();

   ParBilinearForm m(&fespace);
   m.AddDomainIntegrator(new MassIntegrator);
   m.Assemble();
   m.Finalize();

   HypreParMatrix *A = a.ParallelAssemble();
   HypreParMatrix *M = m.ParallelAssemble();

   HypreBoomerAMG amg(*A);
   amg.SetPrintLevel(0);

   HypreLOBPCG lobpcg(MPI_COMM_WORLD);
   lobpcg.SetNumModes(nev);
   lobpcg.SetRandomSeed(seed);
   lobpcg.SetPreconditioner(amg);
   lobpcg.SetMaxIter(200);
   lobpcg.SetTol(1e-8);
   lobpcg.SetPrecondUsageMode(1);
   lobpcg.SetPrintLevel(1);
   lobpcg.SetMassMatrix(*M);
   lobpcg.SetOperator(*A);
   lobpcg.Solve();

   x = lobpcg.GetEigenvector(mode);

   x_l2.ProjectCoefficient(xCoef);

   delete A;
   delete M;
}

// Compute eigenmode "mode" of either a Dirichlet or Neumann Laplacian or of a
// Dirichlet curl curl operator based on the problem type and dimension of the
// domain.
void SetPortBC(int prob, int dim, int mode, ParGridFunction &port_bc)
{
   switch (prob)
   {
      case 0:
         ScalarWaveGuide(mode, port_bc);
         break;
      case 1:
         if (dim == 3)
         {
            VectorWaveGuide(mode, port_bc);
         }
         else
         {
            PseudoScalarWaveGuide(mode, port_bc);
         }
         break;
      case 2:
         PseudoScalarWaveGuide(mode, port_bc);
         break;
   }
}
