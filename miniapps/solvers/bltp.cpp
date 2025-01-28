#include "mfem.hpp"
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

#undef DBG_COLOR
#define DBG_COLOR ::debug::kCyan
#include "general/debug.hpp"

// Prescribed exact velocity
void vel_ex_steady(const Vector &x, Vector &u) {
   int dim = x.Size();
   double xi = x(0);
   double yi = x(1);

   u(0) = -cos(M_PI * xi) * sin(M_PI * yi);
   u(1) = sin(M_PI * xi) * cos(M_PI * yi);

   if (dim > 2) {
      // double zi = x(2);
      u(2) = 0;
   }
}

// Prescribed exact pressure. Underdetermined system yields pressure that is off
// by a scalar from exact answer
double p_ex_steady(const Vector &x) {
   int dim = x.Size();
   double xi = x(0);
   double yi = x(1);
   double zi;
   if (dim > 2) {
      zi = x(2);
   } else {
      zi = 0;
   }

   return xi + yi + 0 * zi;
}

// Momentum forcing function
void ffun_steady(const Vector &x, Vector &f) {
   int dim = x.Size();
   double xi = x(0);
   double yi = x(1);

   f(0) = 1.0 - 2.0 * M_PI * M_PI * cos(M_PI * xi) * sin(M_PI * yi);
   f(1) = 1.0 + 2.0 * M_PI * M_PI * cos(M_PI * yi) * sin(M_PI * xi);

   if (dim > 2) {
      // double zi = x(2);
      f(2) = 0;
   }
}

void ffun_steady2(const Vector &x, Vector &f) {
   int dim = x.Size();
   double xi = x(0);
   double yi = x(1);

   f(0) = 1.0 + (1.0 - 2.0 * M_PI * M_PI) * cos(M_PI * xi) * sin(M_PI * yi);
   f(1) = 1.0 + (1.0 + 2.0 * M_PI * M_PI) * cos(M_PI * yi) * sin(M_PI * xi);

   if (dim > 2) {
      // double zi = x(2);
      f(2) = 0;
   }
}

void MakeMeanZero(ParGridFunction &p) {
   ConstantCoefficient one(1.0);

   ParLinearForm mass_lf(p.ParFESpace());
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   mass_lf.Assemble();

   ParGridFunction one_gf(p.ParFESpace());
   one_gf.ProjectCoefficient(one);
   double volume = mass_lf(one_gf);
   double integ = mass_lf(p);
   p -= integ / volume;
}

int main(int argc, char *argv[]) {
   Mpi::Init(argc, argv);
   const int num_procs = Mpi::WorldSize(), myid = Mpi::WorldRank();
   Hypre::Init();
   dbg();

   // Read in arguments
   const char *mesh_file = "../../data/inline-quad.mesh";
   const char *device_config = "cpu"; // can't use debug yet with Hypre !!
   int order = 2;
   int ref_levels = 2;
   double tol = 1e-8;
   int npatches = 1;
   int print_level = 2;
   bool triangular = false;
   bool pa = false;
   bool vis = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of refinement levels.");
   args.AddOption(&tol, "-tol", "--tolerance", "Solver relative tolerance");
   args.AddOption(&npatches, "-n", "--npatches",
                  "Number of patches to use in additive Schwarz method");
   args.AddOption(&print_level, "-pl", "--print-level", "Solver print level");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&triangular, "-tri", "--triangular-preconditioner", "-diag",
                  "--diagonal-preconditioner",
                  "Use triangular or diagonal preconditioner");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good()) {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device(device_config);
   device.Print();

   int vel_order = order;
   int pres_order = order - 1;

   dbg("Setup parallel mesh from serial mesh");
   // with optional uniform refinement
   std::unique_ptr<Mesh> mesh(new Mesh(mesh_file));
   mesh->SetCurvature(order, false, -1, Ordering::byNODES);
   int dim = mesh->Dimension();
   for (int l = 0; l < ref_levels; l++) { mesh->UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   mesh.reset();
   pmesh.PrintInfo();

   dbg("Define vector FE space for velocity and scalar FE space for pressure");
   H1_FECollection vel_fec(vel_order, dim);
   H1_FECollection pres_fec(pres_order);

   ParFiniteElementSpace vel_fes(&pmesh, &vel_fec, dim);
   dbg("vel_fes.GetVSize();:{}", vel_fes.GetVSize());

   ParFiniteElementSpace pres_fes(&pmesh, &pres_fec);
   dbg("pres_fes.GetVSize();:{}", pres_fes.GetVSize());

   dbg("Determine the list of conforming essential boundary dofs");
   Array<int> ess_tdof_list, ess_tdof_list_pres;
   MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   vel_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   dbg("Setup block structure of dofs and true dofs for Saddle Point (SP)");
   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = vel_fes.GetVSize();
   block_offsets[2] = pres_fes.GetVSize();
   block_offsets.PartialSum();

   Array<int> block_true_offsets(3);
   block_true_offsets[0] = 0;
   block_true_offsets[1] = vel_fes.GetTrueVSize();
   block_true_offsets[2] = pres_fes.GetTrueVSize();
   block_true_offsets.PartialSum();

   int vel_global_vsize = vel_fes.GlobalVSize();
   int pres_global_vsize = pres_fes.GlobalVSize();
   if (myid == 0) {
      cout << "Velocity dofs: " << vel_global_vsize << endl;
      cout << "Pressure dofs: " << pres_global_vsize << endl;
   }

   dbg("Cannot directly use the subblock vectors on the GPU...");
   Vector trueXU(vel_fes.GetTrueVSize()), trueXP(pres_fes.GetTrueVSize());
   Vector trueRhsU(vel_fes.GetTrueVSize()), trueRhsP(pres_fes.GetTrueVSize());
   trueXU = 0.0;
   trueXP = 0.0;
   trueRhsU = 0.0;
   trueRhsP = 0.0;

   // Define spaces and operators

   dbg("fcoeff is RHS of first block row in SP system");
   VectorFunctionCoefficient fcoeff(dim, ffun_steady);
   // Exact solutions from known functions above - method of manufactured
   // solution
   VectorFunctionCoefficient uexcoeff(dim, vel_ex_steady);
   FunctionCoefficient pexcoeff(p_ex_steady);

   dbg("Define solution vector u as FE grid function for velocity space.");
   // Initial guess x[0] satisfies Dirichlet B.C.
   ParGridFunction u_gf(&vel_fes);
   u_gf.ProjectBdrCoefficient(uexcoeff, ess_bdr);

   dbg("Define solution scalar p as FE grid function for pressure space.");
   ParGridFunction p_gf(&pres_fes);
   p_gf = 0.0;

   ParGridFunction rhsU_gf(&vel_fes);
   rhsU_gf = 0.0;
   ParGridFunction rhsP_gf(&pres_fes);
   rhsP_gf = 0.0;

   dbg("Linear form b(.) is 1st (vector) block of RHS, (f, phi_i)");
   ParLinearForm fform;
   rhsU_gf.HostReadWrite();
   fform.Update(&vel_fes, rhsU_gf, 0); // only works on host
   const IntegrationRule &ir =
      IntRules.Get(pres_fes.GetFE(0)->GetGeomType(), 2 * vel_order);
   fform.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   (*fform.GetDLFI())[0]->SetIntRule(&ir);
   fform.Assemble();

   dbg("Bilinear form a(.,.) on velocity FE space corresponding to vel "
       "Laplacian operator");
   ParBilinearForm lapform(&vel_fes);
   lapform.AddDomainIntegrator(new VectorDiffusionIntegrator);
   if (pa) { lapform.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   lapform.Assemble();
   // Eliminate Dirichlet BCs so that known velocity dofs are of the form 1 u_i
   // = rhsval
   OperatorHandle Ahandle;
   lapform.FormLinearSystem(ess_tdof_list, u_gf, rhsU_gf, Ahandle, trueXU,
                            trueRhsU);

   dbg("Bilinear form d(.,.) corresponding to divergence from vel FE space to "
       "pressure FE space");
   ParMixedBilinearForm dform(&vel_fes, &pres_fes);
   dform.AddDomainIntegrator(new VectorDivergenceIntegrator);
   if (pa) { dform.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   dform.Assemble();
   dbg("Eliminate dofs of mixed system using BCs of div(u) = rhs subsystem");
   OperatorHandle Dhandle;
   dform.FormRectangularLinearSystem(ess_tdof_list, ess_tdof_list_pres, u_gf,
                                     rhsP_gf, Dhandle, trueXU, trueRhsP);

   dbg("Bilinear form g(.,.) corresponding to gradient from pressure FE space "
       "to vel FE space");
   ParMixedBilinearForm gform(&pres_fes, &vel_fes);
   gform.AddDomainIntegrator(new GradientIntegrator);
   if (pa) { gform.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   gform.Assemble();
   OperatorHandle Ghandle;
   // Eliminate dofs of mixed system using transpose of BCs of div(u) = rhs
   // subsystem
   gform.FormRectangularSystemMatrix(ess_tdof_list_pres, ess_tdof_list,
                                     Ghandle);

   dbg("Orthogonalize RHS");
   trueRhsP -= trueRhsP.Sum() / trueRhsP.Size();

   dbg("Copy subvectors into large block vectors");
   BlockVector trueX(block_true_offsets, Device::GetMemoryType());
   trueX.UseDevice(true);
   trueXU.Read();
   trueXP.Read();
   trueX.GetBlock(0) = trueXU;
   trueX.GetBlock(1) = trueXP;
   BlockVector trueRhs(block_true_offsets, Device::GetMemoryType());
   trueRhs.UseDevice(true);
   trueRhsU.Read();
   trueRhsP.Read();
   trueRhs.GetBlock(0) = trueRhsU;
   trueRhs.GetBlock(1) = trueRhsP;

   dbg("Finalize SP system");
   BlockOperator stokesop(block_true_offsets);
   stokesop.SetBlock(0, 0, Ahandle.Ptr());
   stokesop.SetBlock(0, 1, Ghandle.Ptr());
   stokesop.SetBlock(1, 0, Dhandle.Ptr());

   dbg("Setup preconditioner");
   ParBilinearForm mpform(&pres_fes);
   HypreParMatrix *A;

   std::unique_ptr<Solver> invMp;
   std::shared_ptr<Solver> invLap;
   std::unique_ptr<Solver> stokesprec;

   if (pa) {
      assert(false);
      /*
        // Vector implementation of low-order refined preconditioner
        MassNI Mp_ni(pres_fes, ess_tdof_list_pres);
        invMp.reset(new DiagonalInverse(Mp_ni.D));
        ConstantCoefficient one(1.0);
        VectorHOASPreconditioner * hoasp =
           new VectorHOASPreconditioner(vel_fes, one, ess_bdr, true, npatches);
        hoasp->SetParameters(1.0);
        invLap.reset(hoasp);*/
   } else {
      A = Ahandle.As<HypreParMatrix>(); // owned by lapform

      mpform.AddDomainIntegrator(new MassIntegrator);
      mpform.Assemble();
      mpform.Finalize();
      HypreParMatrix *Mp = mpform.ParallelAssemble();
      // One V-cycle of AMG approximates action of Laplacian^{-1}
      invMp.reset(new HypreDiagScale(*Mp));
      HypreBoomerAMG *amg = new HypreBoomerAMG(*A);
      amg->SetPrintLevel(0);
      amg->iterative_mode = false;
      invLap.reset(amg);
   }

   if (triangular) {
      dbg("BlockLowerTriangularPreconditioner");
      assert(false);
      auto *triprec =
         new BlockLowerTriangularPreconditioner(block_true_offsets);
      triprec->SetBlock(0, 0, invLap.get());
      // Lower triangular of block is Divergence operator
      triprec->SetBlock(1, 0, Dhandle.Ptr());
      triprec->SetBlock(1, 1, invMp.get());
      stokesprec.reset(triprec);
   } else {
      dbg("BlockDiagonalPreconditioner");
      auto *diagprec = new BlockDiagonalPreconditioner(block_true_offsets);
      diagprec->SetDiagonalBlock(0, invLap.get());
      diagprec->SetDiagonalBlock(1, invMp.get());
      stokesprec.reset(diagprec);
   }

   dbg("Setup actual linear solver: block-preconditioned GMRES");
   GMRESSolver solver(MPI_COMM_WORLD);
   solver.iterative_mode = false;
   solver.SetAbsTol(0.0);
   solver.SetRelTol(tol);
   solver.SetMaxIter(500);
   solver.SetPreconditioner(*stokesprec);
   solver.SetOperator(stokesop);
   solver.SetPrintLevel(print_level);

   dbg("Solve the linear SP system");
   /*
   trueX.HostReadWrite();
   trueX.Randomize(3);
   trueX.ReadWrite();
   */
   solver.Mult(trueRhs, trueX);

   dbg("Extract the results from trueX");
   trueX.GetBlock(0).SyncAliasMemory(trueX);
   trueX.GetBlock(1).SyncAliasMemory(trueX);
   u_gf.Distribute(&(trueX.GetBlock(0)));
   p_gf.Distribute(&(trueX.GetBlock(1)));

   MakeMeanZero(p_gf);
   u_gf.HostReadWrite();
   p_gf.HostReadWrite();

   dbg("Setup all integration rules for any element type");
   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i) {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   dbg("Compute the velocity and pressure error from manufactured solution");
   double err_u = u_gf.ComputeL2Error(uexcoeff, irs);
   double norm_u = ComputeGlobalLpNorm(2, uexcoeff, pmesh, irs);

   ParGridFunction p_ex_gf(&pres_fes);
   p_ex_gf.ProjectCoefficient(pexcoeff);
   MakeMeanZero(p_ex_gf);
   GridFunctionCoefficient p_ex_gf_coeff(&p_ex_gf);

   double err_p = p_gf.ComputeL2Error(p_ex_gf_coeff, irs);
   double norm_p = ComputeGlobalLpNorm(2, pexcoeff, pmesh, irs);

   if (myid == 0) {
      cout << "|| u_h - u_ex || = " << err_u << "\n";
      cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
      cout << "|| p_h - p_ex || = " << err_p << "\n";
      cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
   }

   if (vis) {
      dbg("Visualization");
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n"
             << pmesh << u_gf << "window_title 'velocity'" << "keys Rjlc\n"
             << endl;

      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << num_procs << " " << myid << "\n";
      p_sock.precision(8);
      p_sock << "solution\n"
             << pmesh << p_gf << "window_title 'pressure'" << "keys Rjlc\n"
             << endl;

      /* Visualize the exact pressure
      // First project function onto grid function
      ParGridFunction pex_gf;
      pex_gf.SetSpace(&pres_fes);
      pex_gf.ProjectCoefficient(pexcoeff);

      // Then visualize
      socketstream pex_sock(vishost, visport);
      pex_sock << "parallel " << num_procs << " " << myid << "\n";
      pex_sock.precision(8);
      pex_sock << "solution\n" << pmesh << pex_gf << "window_title
      'pressure_exact'" << "keys Rjlc\n"<< endl; delete pex_gf;
      */
   }

   dbg("Exit");
   return 0;
}
