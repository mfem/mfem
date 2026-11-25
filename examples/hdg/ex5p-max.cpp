//                 MFEM Example 5 (Maxwell) - Parallel HDG Version
//
// Compile with: make ex5p-max
//
// Sample runs:  mpirun -np 4 ex5p-max -m ../data/square-disc.mesh
//               mpirun -np 4 ex5p-max -m ../data/square-disc.mesh -hb
//               mpirun -np 4 ex5p-max -m ../data/star.mesh
//               mpirun -np 4 ex5p-max -m ../data/star.mesh -r 2 -pa
//               mpirun -np 4 ex5p-max -m ../data/star.mesh -hb
//               mpirun -np 4 ex5p-max -m ../data/beam-tet.mesh
//               mpirun -np 4 ex5p-max -m ../data/beam-hex.mesh
//               mpirun -np 4 ex5p-max -m ../data/beam-hex.mesh -pa
//               mpirun -np 4 ex5p-max -m ../data/beam-hex.mesh -hb
//               mpirun -np 4 ex5p-max -m ../data/escher.mesh
//               mpirun -np 4 ex5p-max -m ../data/fichera.mesh
//               mpirun -np 4 ex5p-max -m ../data/fichera.mesh -hb
//
// Device sample runs:
//               mpirun -np 4 ex5p-max -m ../data/star.mesh -r 2 -pa -d cuda
//               mpirun -np 4 ex5p-max -m ../data/star.mesh -r 2 -pa -d raja-cuda
//               mpirun -np 4 ex5p-max -m ../data/star.mesh -r 2 -pa -d raja-omp
//               mpirun -np 4 ex5p-max -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                                 sigma*E - curl B = f
//                                 curl E  + mu*B   = g
//
//               with natural boundary condition -B = <given magnetic field>.
//               Here, we use a given exact solution (E,B) and compute the
//               corresponding r.h.s. (f,g).  We piecewise discontinuous
//               polynomials finite elements (electric field E and magnetic
//               field B).
//
//               The example demonstrates the use of the ParMixedSystemForm
//               class, as well as algebraic reduction and hybridization of
//               mixed systems and the collective saving of several grid
//               functions in VisIt (visit.llnl.gov) and ParaView (paraview.org)
//               formats. Optional saving with ADIOS2 (adios2.readthedocs.io)
//               streams is also illustrated.
//
//               We recommend viewing the standard version of this example
//               to understand the extension for the Local / Hybridizable
//               Discontinuous Galerkin methods. The benefits of lower
//               iterations count, number of DOFs, and essential pressure trace
//               boundary condition can be assessed.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void EFun_ex(const Vector & x, Vector & E);
void BFun_ex(const Vector & x, Vector & B);
void fFun(const Vector & x, Vector & f);
real_t freq = 1.0, kappa;

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   bool verbose = (myid == 0);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/beam-tet.mesh";
   int ref_levels = -1;
   int order = 1;
   real_t td = 0.5;
   Array<int> bbcs;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = false;
   bool trace_ess_bc = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   bool par_format = false;
   bool adios2 = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default).");
   args.AddOption(&bbcs, "-bbc", "--magnetic-bcs",
                  "List of magnetic boundary conditions.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction of DG electric field.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&trace_ess_bc, "-trbc", "--trace-ess-bc", "-no-trbc",
                  "--no-trace-ess-bc", "Switch between essential and weak trace BC.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&par_format, "-pf", "--parallel-format", "-sf",
                  "--serial-format",
                  "Format to use when saving the results for VisIt.");
   args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                  "--no-adios2-streams",
                  "Save data using adios2 streams.");
   args.Parse();
   if (!args.Good())
   {
      if (verbose)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (verbose)
   {
      args.PrintOptions(cout);
   }
   kappa = freq * M_PI;


   if (pa && (hybridization || reduction))
   {
      if (verbose)
      {
         cerr << "Partial assembly is not supported for hybridization or reduction."
              << endl;
      }
      return 1;
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   const int dim_curl = dim == 3 ? 3 : 1;

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements, unless the user specifies it as input.
   {
      if (ref_levels == -1)
      {
         ref_levels = (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use discontinuous finite elements of the specified order.
   auto E_coll = make_unique<L2_FECollection>(order, dim,
                                              BasisType::GaussLobatto);
   auto B_coll = make_unique<L2_FECollection>(order, dim);

   auto E_space = make_unique<ParFiniteElementSpace>(&pmesh, E_coll.get(), dim);
   auto B_space = make_unique<ParFiniteElementSpace>(&pmesh, B_coll.get(),
                                                     dim_curl);

   HYPRE_BigInt dimE = E_space->GlobalTrueVSize();
   HYPRE_BigInt dimB = B_space->GlobalTrueVSize();

   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(E) = " << dimE << "\n";
      std::cout << "dim(B) = " << dimB << "\n";
      std::cout << "dim(E+B) = " << dimE + dimB << "\n";
      std::cout << "***********************************************************\n";
   }

   // 8. Define the block structure of the problem. block_offsets is used
   //    for Vector based on dof (like ParGridFunction or ParLinearForm),
   //    block_trueOffsets is used for Vector based on true dof (HypreParVector
   //    for the rhs and solution of the linear system).  The offsets computed
   //    here are local to the processor.
   auto darcy = make_unique<ParMixedSystemForm>(E_space.get(), B_space.get());
   const Array<int> &block_offsets = darcy->GetOffsets();
   const Array<int> &block_trueOffsets = darcy->GetTrueOffsets();

   // 9. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient sigmacoeff(1.0); // Diffusivity
   ConstantCoefficient invsigmacoeff(1.0); // Inverse diffusivity
   ConstantCoefficient mucoeff(1.0); // Permeability

   VectorFunctionCoefficient fcoeff(dim, fFun); // Electric rhs

   VectorFunctionCoefficient Ecoeff(dim, EFun_ex); // Electric field
   VectorFunctionCoefficient Bcoeff(dim_curl, BFun_ex); // Magnetic field

   // 10. Define the parallel grid function and parallel linear forms, solution
   //     vector and rhs.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt);

   // Mark attributes of boundary conditions
   const int nbdr_attrs = pmesh.bdr_attributes.Size() > 0 ?
                          pmesh.bdr_attributes.Max() : 1;
   Array<int> bdr_E(nbdr_attrs), bdr_B(nbdr_attrs);
   bdr_E = -1;
   bdr_B = 0;
   for (int bbc : bbcs)
   {
      if (bbc <= 0 || bbc > nbdr_attrs) { continue; }
      bdr_E[bbc-1] = 0;
      bdr_B[bbc-1] = -1;
   }

   // Electric RHS
   ParLinearForm *fform = darcy->GetParFluxRHS();
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   if (!hybridization || !trace_ess_bc)
   {
      fform->AddBdrFaceIntegrator(new VectorBoundaryTangentialLFIntegrator(
                                     Bcoeff), bdr_B);
   }

   // Magnetic RHS
   ParLinearForm *gform = darcy->GetParPotentialRHS();
   if (!hybridization)
   {
      if (dim_curl > 1)
      {
         gform->AddBdrFaceIntegrator(new VectorBoundaryTangentialLFIntegrator(Ecoeff),
                                     bdr_E);
      }
      else
      {
         gform->AddBdrFaceIntegrator(new BoundaryTangentialLFIntegrator(Ecoeff), bdr_E);
      }
   }

   // 11. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   ParBilinearForm *mVarf = darcy->GetParFluxMassForm();
   ParMixedBilinearForm *cVarf = darcy->GetParFluxDivForm();
   ParBilinearForm *mbVarf = darcy->GetParPotentialMassForm();

   mVarf->AddDomainIntegrator(new VectorMassIntegrator(sigmacoeff));

   cVarf->AddDomainIntegrator(new MixedCurlIntegrator(
                                 MixedCurlIntegrator::Mode::OutOfPlane));
   cVarf->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                       new DGTangentialTraceIntegrator(+1.)));
   mbVarf->AddInteriorFaceIntegrator(new HDGCurlCurlIntegrator(invsigmacoeff, td));
   cVarf->AddBdrFaceIntegrator(new TransposeIntegrator(
                                  new DGTangentialTraceIntegrator(+2.)), bdr_E);
   if (hybridization)
   {
      mbVarf->AddBdrFaceIntegrator(new HDGCurlCurlIntegrator(invsigmacoeff, td),
                                   bdr_E);
   }
   if (hybridization && trace_ess_bc)
   {
      cVarf->AddBdrFaceIntegrator(new TransposeIntegrator(
                                     new DGTangentialTraceIntegrator(+2.)), bdr_B);
      mbVarf->AddBdrFaceIntegrator(new HDGCurlCurlIntegrator(invsigmacoeff, td),
                                   bdr_B);
   }

   if (dim_curl > 1)
   {
      mbVarf->AddDomainIntegrator(new VectorMassIntegrator(mucoeff));
   }
   else
   {
      mbVarf->AddDomainIntegrator(new MassIntegrator(mucoeff));
   }

   // Set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;

   unique_ptr<FiniteElementCollection> trace_coll;
   unique_ptr<ParFiniteElementSpace> trace_space;
   Vector X, RHS;

   chrono.Clear();
   chrono.Start();

   if (hybridization)
   {
      if (trace_h1)
      {
         trace_coll = make_unique<H1_Trace_FECollection>(max(order, 1), dim);
      }
      else
      {
         trace_coll = make_unique<DG_Interface_FECollection>(order, dim);
      }
      trace_space = make_unique<ParFiniteElementSpace>(&pmesh, trace_coll.get(),
                                                       dim - 1);
      darcy->EnableHybridization(trace_space.get(),
                                 new TangentTraceJumpIntegrator(-1.),
                                 ess_flux_tdofs_list);
      // Set essential BC
      if (trace_ess_bc && pmesh.bdr_attributes.Size() > 0)
      {
         X.SetSize(trace_space->GetTrueVSize());
         // Project essential BC
         ParGridFunction Bhat;
         Bhat.MakeTRef(trace_space.get(), X, 0);
         Bhat = 0.;
         TangentTraceCoefficient Bbc_coeff(Bcoeff);
         Bhat.ProjectBdrCoefficient(Bbc_coeff, bdr_B);
         Bhat.SetTrueVector();

         darcy->GetHybridization()->SetEssentialBC(bdr_B);
      }
      // Set natural BC
      {
         RHS.SetSize(trace_space->GetTrueVSize());
         auto hform = make_unique<ParLinearForm>(trace_space.get());
         hform->AddBoundaryIntegrator(new VectorBoundaryTangentTraceLFIntegrator(Ecoeff),
                                      bdr_E);
         hform->Assemble();
         hform->ParallelAssemble(RHS);
      }
   }
   else if (reduction)
   {
      darcy->EnableFluxReduction();
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   darcy->Assemble();

   OperatorPtr A;
   x = 0.;
   darcy->FormLinearSystem(ess_flux_tdofs_list, x, A, X, RHS, true);

   chrono.Stop();
   if (verbose)
   {
      std::cout << "Assembly took " << chrono.RealTime() << "s.\n";
   }


   const int maxIter(pa ? 1000 : 500);
   constexpr real_t rtol(1.e-6);
   constexpr real_t atol(1.e-10);

   if (hybridization || reduction)
   {
      // 12. Construct the preconditioner
      HypreBoomerAMG prec;

      // 13. Solve the linear system with GMRES.
      //     Check the norm of the unpreconditioned residual.
      chrono.Clear();
      chrono.Start();
      GMRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetPreconditioner(prec);
      solver.SetOperator(*A);
      solver.SetPrintLevel(1);

      solver.Mult(RHS, X);
      darcy->RecoverFEMSolution(X, x);

      chrono.Stop();

      if (verbose)
      {
         if (solver.GetConverged())
         {
            std::cout << "GMRES converged in " << solver.GetNumIterations()
                      << " iterations with a residual norm of "
                      << solver.GetFinalNorm() << ".\n";
         }
         else
         {
            std::cout << "GMRES did not converge in " << solver.GetNumIterations()
                      << " iterations. Residual norm is " << solver.GetFinalNorm()
                      << ".\n";
         }
         std::cout << "GMRES solver took " << chrono.RealTime() << "s.\n";
      }
   }
   else
   {
      // 12. Construct the operators for preconditioner
      //
      //                 P = [ diag(M)         0         ]
      //                     [  0       B diag(M)^-1 B^T ]
      //
      //     Here we use BoomerAMG to approximate the inverse of the
      //     magnetic field Schur Complement
      unique_ptr<HypreParMatrix> MinvBt, S;
      unique_ptr<Solver> invM, invS;

      if (pa)
      {
         Vector Md(E_space->GetTrueVSize());
         mVarf->AssembleDiagonal(Md);
         auto Md_host = Md.HostRead();
         Vector invMd(Md.Size());
         for (int i=0; i<Md.Size(); ++i)
         {
            invMd(i) = 1.0 / Md_host[i];
         }

         Vector BMBt_diag(B_space->GetTrueVSize());
         cVarf->AssembleDiagonal_ADAt(invMd, BMBt_diag);

         if (mbVarf)
         {
            Vector MBd(mbVarf->Height());
            mbVarf->AssembleDiagonal(MBd);
            BMBt_diag += MBd;
         }

         Array<int> ess_tdof_list;  // empty

         invM = make_unique<OperatorJacobiSmoother>(Md, ess_tdof_list);
         invS = make_unique<OperatorJacobiSmoother>(BMBt_diag,
                                                    ess_tdof_list);
      }
      else
      {
         HypreParMatrix &MEm = *mVarf->ParallelAssembleInternalMatrix();
         Vector Md;
         MEm.GetDiag(Md);

         HypreParMatrix &Cm = *cVarf->ParallelAssembleInternalMatrix();
         MinvBt.reset(Cm.Transpose());
         MinvBt->InvScaleRows(Md);
         S.reset(ParMult(&Cm, MinvBt.get()));

         if (mbVarf)
         {
            HypreParMatrix &MBm = *mbVarf->ParallelAssembleInternalMatrix();
            S.reset(ParAdd(&MBm, S.get()));
         }

         invM = make_unique<HypreDiagScale>(MEm);
         invS = make_unique<HypreBoomerAMG>(*S);
      }

      invM->iterative_mode = false;
      invS->iterative_mode = false;

      BlockDiagonalPreconditioner darcyPrec(block_trueOffsets);
      darcyPrec.SetDiagonalBlock(0, invM.get());
      darcyPrec.SetDiagonalBlock(1, invS.get());

      // 13. Solve the linear system with MINRES.
      //     Check the norm of the unpreconditioned residual.

      chrono.Clear();
      chrono.Start();
      MINRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*A);
      solver.SetPreconditioner(darcyPrec);
      solver.SetPrintLevel(1);

      solver.Mult(RHS, X);
      darcy->RecoverFEMSolution(X, x);

      if (device.IsEnabled()) { x.HostRead(); }
      chrono.Stop();

      if (verbose)
      {
         if (solver.GetConverged())
         {
            std::cout << "MINRES converged in " << solver.GetNumIterations()
                      << " iterations with a residual norm of "
                      << solver.GetFinalNorm() << ".\n";
         }
         else
         {
            std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                      << " iterations. Residual norm is " << solver.GetFinalNorm()
                      << ".\n";
         }
         std::cout << "MINRES solver took " << chrono.RealTime() << "s.\n";
      }
   }

   // 14. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.
   ParGridFunction E, B;
   E.MakeRef(E_space.get(), x.GetBlock(0), 0);
   B.MakeRef(B_space.get(), x.GetBlock(1), 0);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   real_t err_E  = E.ComputeL2Error(Ecoeff, irs);
   real_t norm_E = ComputeGlobalLpNorm(2., Ecoeff, pmesh, irs);
   real_t err_B  = B.ComputeL2Error(Bcoeff, irs);
   real_t norm_B = ComputeGlobalLpNorm(2., Bcoeff, pmesh, irs);

   if (verbose)
   {
      std::cout << "|| E_h - E_ex || / || E_ex || = " << err_E / norm_E << "\n";
      std::cout << "|| B_h - B_ex || / || B_ex || = " << err_B / norm_B << "\n";
   }

   // 15. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol_*".
   {
      ostringstream mesh_name, E_name, B_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      E_name << "sol_E." << setfill('0') << setw(6) << myid;
      B_name << "sol_B." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream E_ofs(E_name.str().c_str());
      E_ofs.precision(8);
      E.Save(E_ofs);

      ofstream B_ofs(B_name.str().c_str());
      B_ofs.precision(8);
      B.Save(B_ofs);
   }

   // 16. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5-Parallel", &pmesh);
   visit_dc.RegisterField("electric field", &E);
   visit_dc.RegisterField("magnetic field", &B);
   visit_dc.SetFormat(!par_format ?
                      DataCollection::SERIAL_FORMAT :
                      DataCollection::PARALLEL_FORMAT);
   visit_dc.Save();

   // 17. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example5P", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("electric field",&E);
   paraview_dc.RegisterField("magnetic field",&B);
   paraview_dc.Save();

   // 18. Optionally output a BP (binary pack) file using ADIOS2. This can be
   //     visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
   if (adios2)
   {
      std::string postfix(mesh_file);
      postfix.erase(0, std::string("../data/").size() );
      postfix += "_o" + std::to_string(order);
      const std::string collection_name = "ex5-p-" + postfix + ".bp";

      ADIOS2DataCollection adios2_dc(MPI_COMM_WORLD, collection_name, &pmesh);
      adios2_dc.SetLevelsOfDetail(1);
      adios2_dc.SetCycle(1);
      adios2_dc.SetTime(0.0);
      adios2_dc.RegisterField("electric field",&E);
      adios2_dc.RegisterField("magnetic field",&B);
      adios2_dc.Save();
   }
#endif

   // 19. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream E_sock(vishost, visport);
      E_sock << "parallel " << num_procs << " " << myid << "\n";
      E_sock.precision(8);
      E_sock << "solution\n" << pmesh << E << "window_title 'Electric field'"
             << endl;
      // Make sure all ranks have sent their 'E' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh.GetComm());
      socketstream B_sock(vishost, visport);
      B_sock << "parallel " << num_procs << " " << myid << "\n";
      B_sock.precision(8);
      B_sock << "solution\n" << pmesh << B << "window_title 'Magnetic field'"
             << endl;
   }

   return 0;
}


void EFun_ex(const Vector &x, Vector &E)
{
   const int dim = x.Size();
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
      //if (dim == 3) { E(2) = 0.0; }
   }
}

// Change if needed
void BFun_ex(const Vector & x, Vector &B)
{
   const int dim = x.Size();
   if (dim < 3)
   {
      B(0) = kappa * (-cos(kappa * x(0)) + cos(kappa * x(1)));
   }
   else
   {
      B(0) = kappa * (cos(kappa * x(2)));
      B(1) = kappa * (cos(kappa * x(0)));
      B(2) = kappa * (cos(kappa * x(1)));
   }
}

void fFun(const Vector & x, Vector & f)
{
   const int dim = x.Size();
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
