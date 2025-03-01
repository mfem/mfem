//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 -m ../data/square-disc.mesh
//               ex5 -m ../data/star.mesh
//               ex5 -m ../data/star.mesh -pa
//               ex5 -m ../data/beam-tet.mesh
//               ex5 -m ../data/beam-hex.mesh
//               ex5 -m ../data/beam-hex.mesh -pa
//               ex5 -m ../data/escher.mesh
//               ex5 -m ../data/fichera.mesh
//
// Device sample runs:
//               ex5 -m ../data/star.mesh -pa -d cuda
//               ex5 -m ../data/star.mesh -pa -d raja-cuda
//               ex5 -m ../data/star.mesh -pa -d raja-omp
//               ex5 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                                 k*u + grad p = f
//                                 - div u      = g
//
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p). Alternatively, the piecewise discontinuous
//               polynomials are used for both quantities.
//
//               The example demonstrates the use of the DarcyForm class, as
//               well as hybridization of mixed systems and the collective saving
//               of several grid functions in VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
real_t pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
real_t gFun(const Vector & x);
real_t f_natural(const Vector & x);

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
   const char *mesh_file = "";
   int nx = 0;
   int ny = 0;
   int ref_levels = -1;
   int order = 1;
   bool dg = false;
   real_t td = 0.5;
   bool hybridization = false;
   bool reduction = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   bool par_format = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction of DG flux.");
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

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   if (ny <= 0)
   {
      ny = nx;
   }

   Mesh *mesh = NULL;
   if (strlen(mesh_file) > 0)
   {
      mesh = new Mesh(mesh_file, 1, 1);
   }
   else
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL));
   }

   int dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements, unless the user specifies it as input.
   if (strlen(mesh_file) > 0)
   {
      if (ref_levels == -1)
      {
         ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   /*{
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }*/

   // 7. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *R_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      R_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else
   {
      R_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim);

   ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh, R_coll,
                                                              (dg)?(dim):(1));
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, W_coll);

   ParDarcyForm *darcy = new ParDarcyForm(R_space, W_space);

   HYPRE_BigInt dimR = R_space->GlobalTrueVSize();
   HYPRE_BigInt dimW = W_space->GlobalTrueVSize();

   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(R) = " << dimR << "\n";
      std::cout << "dim(W) = " << dimW << "\n";
      std::cout << "dim(R+W) = " << dimR + dimR << "\n";
      std::cout << "***********************************************************\n";
   }

   // 8. Define the two BlockStructure of the problem.  block_offsets is used
   //    for Vector based on dof (like ParGridFunction or ParLinearForm),
   //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
   //    for the rhs and solution of the linear system).  The offsets computed
   //    here are local to the processor.

   const Array<int> &block_offsets = darcy->GetOffsets();
   const Array<int> &block_trueOffsets = darcy->GetTrueOffsets();

   // 9. Define the coefficients, analytical solution, and rhs of the PDE.
   const double k = 1.0;
   ConstantCoefficient kcoeff(k);
   RatioCoefficient ikcoeff(1., kcoeff);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient fnatcoeff(f_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   // 10. Define the parallel grid function and parallel linear forms, solution
   //     vector and rhs.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   ParLinearForm *fform(new ParLinearForm);
   fform->Update(R_space, rhs.GetBlock(0), 0);
   if (dg)
   {
      fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
      fform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(fnatcoeff));
   }
   else
   {
      fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
      fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
   }
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   ParLinearForm *gform(new ParLinearForm);
   gform->Update(W_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // 11. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   ParBilinearForm *mVarf = darcy->GetParFluxMassForm();
   ParMixedBilinearForm *bVarf = darcy->GetParFluxDivForm();
   ParBilinearForm *mtVarf = (dg)?(darcy->GetParPotentialMassForm()):(NULL);

   if (dg)
   {
      mVarf->AddDomainIntegrator(new VectorMassIntegrator(kcoeff));
      bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator());
      bVarf->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                          new DGNormalTraceIntegrator(-1.)));
      mtVarf->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ikcoeff, td));
   }
   else
   {
      mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(kcoeff));
      bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   }

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;

   FiniteElementCollection *trace_coll = NULL;
   ParFiniteElementSpace *trace_space = NULL;

   chrono.Clear();
   chrono.Start();

   if (hybridization)
   {
      trace_coll = new RT_Trace_FECollection(order, dim, 0);
      trace_space = new ParFiniteElementSpace(pmesh, trace_coll);
      darcy->EnableHybridization(trace_space,
                                 new NormalTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);
   }
   else if (reduction && dg)
   {
      darcy->EnableFluxReduction();
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   darcy->Assemble();

   OperatorHandle pDarcyOp;
   Vector X, B;
   x = 0.;
   darcy->FormLinearSystem(ess_flux_tdofs_list, x, rhs,
                           pDarcyOp, X, B);

   chrono.Stop();
   if (verbose)
   {
      std::cout << "Assembly took " << chrono.RealTime() << "s.\n";
   }


   int maxIter(1000);
   real_t rtol(1.e-6);
   real_t atol(1.e-10);

   if (hybridization || (reduction && dg))
   {
      // 12. Construct the preconditioner
      Solver *prec;
      if (hybridization && dim > 1)
         if (dim == 2)
         {
            prec = new HypreAMS(trace_space);
         }
         else
         {
            prec = new HypreADS(trace_space);
         }
      else
      {
         prec = new HypreBoomerAMG();
      }

      // 13. Solve the linear system with GMRES.
      //     Check the norm of the unpreconditioned residual.
      chrono.Clear();
      chrono.Start();
      GMRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetPreconditioner(*prec);
      solver.SetOperator(*pDarcyOp);
      solver.SetPrintLevel(verbose);

      solver.Mult(B, X);
      darcy->RecoverFEMSolution(X, rhs, x);

      delete prec;

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
      //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
      //     pressure Schur Complement
      HypreParMatrix *MinvBt = NULL;
      HypreParVector *Md = NULL;
      HypreParMatrix *S = NULL;
      Solver *invM, *invS;

      if (pa)
      {
         Vector Md(R_space->GetTrueVSize());
         mVarf->AssembleDiagonal(Md);
         auto Md_host = Md.HostRead();
         Vector invMd(Md.Size());
         for (int i=0; i<Md.Size(); ++i)
         {
            invMd(i) = 1.0 / Md_host[i];
         }

         Vector BMBt_diag(W_space->GetTrueVSize());
         bVarf->AssembleDiagonal_ADAt(invMd, BMBt_diag);

         Array<int> ess_tdof_list;  // empty

         invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
         invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
      }
      else
      {
         HypreParMatrix &M = static_cast<HypreParMatrix&>(
                                pDarcyOp.As<BlockOperator>()->GetBlock(0,0));
         Md = new HypreParVector(MPI_COMM_WORLD, M.GetGlobalNumRows(),
                                 M.GetRowStarts());
         M.GetDiag(*Md);

         HypreParMatrix &B = static_cast<HypreParMatrix&>
                             (pDarcyOp.As<BlockOperator>()->GetBlock(1,0));
         MinvBt = B.Transpose();
         MinvBt->InvScaleRows(*Md);
         S = ParMult(&B, MinvBt);

         if (mtVarf)
         {
            HypreParMatrix &Mt = static_cast<HypreParMatrix&>(
                                    pDarcyOp.As<BlockOperator>()->GetBlock(1,1));
            HypreParMatrix *Snew = ParAdd(&Mt, S);
            delete S;
            S = Snew;
         }

         invM = new HypreDiagScale(M);
         invS = new HypreBoomerAMG(*S);
      }

      invM->iterative_mode = false;
      invS->iterative_mode = false;

      BlockDiagonalPreconditioner darcyPrec(block_trueOffsets);
      darcyPrec.SetDiagonalBlock(0, invM);
      darcyPrec.SetDiagonalBlock(1, invS);

      // 13. Solve the linear system with MINRES.
      //     Check the norm of the unpreconditioned residual.

      chrono.Clear();
      chrono.Start();
      MINRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*pDarcyOp);
      solver.SetPreconditioner(darcyPrec);
      solver.SetPrintLevel(verbose);

      solver.Mult(B, X);
      darcy->RecoverFEMSolution(X, rhs, x);

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

      delete invM;
      delete invS;
      delete S;
      delete Md;
      delete MinvBt;
   }

   // 14. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.
   ParGridFunction u, p;
   u.MakeRef(R_space, x.GetBlock(0), 0);
   p.MakeRef(W_space, x.GetBlock(1), 0);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   real_t err_u  = u.ComputeL2Error(ucoeff, irs);
   real_t norm_u = ComputeGlobalLpNorm(2., ucoeff, *pmesh, irs);
   real_t err_p  = p.ComputeL2Error(pcoeff, irs);
   real_t norm_p = ComputeGlobalLpNorm(2., pcoeff, *pmesh, irs);

   if (verbose)
   {
      std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
      std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
   }

   // 15. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol_*".
   {
      ostringstream mesh_name, u_name, p_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      u_name << "sol_u." << setfill('0') << setw(6) << myid;
      p_name << "sol_p." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream u_ofs(u_name.str().c_str());
      u_ofs.precision(8);
      u.Save(u_ofs);

      ofstream p_ofs(p_name.str().c_str());
      p_ofs.precision(8);
      p.Save(p_ofs);
   }

   // 16. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5-Parallel", pmesh);
   visit_dc.RegisterField("velocity", &u);
   visit_dc.RegisterField("pressure", &p);
   visit_dc.SetFormat(!par_format ?
                      DataCollection::SERIAL_FORMAT :
                      DataCollection::PARALLEL_FORMAT);
   visit_dc.Save();

   // 17. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example5P", pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.Save();

   // 18. Optionally output a BP (binary pack) file using ADIOS2. This can be
   //     visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
   if (adios2)
   {
      std::string postfix(mesh_file);
      postfix.erase(0, std::string("../data/").size() );
      postfix += "_o" + std::to_string(order);
      const std::string collection_name = "ex5-p_" + postfix + ".bp";

      ADIOS2DataCollection adios2_dc(MPI_COMM_WORLD, collection_name, pmesh);
      adios2_dc.SetLevelsOfDetail(1);
      adios2_dc.SetCycle(1);
      adios2_dc.SetTime(0.0);
      adios2_dc.RegisterField("velocity",&u);
      adios2_dc.RegisterField("pressure",&p);
      adios2_dc.Save();
   }
#endif

   // 19. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << u << "window_title 'Velocity'"
             << endl;
      u_sock << "keys Rljvvvvvmmc" << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << num_procs << " " << myid << "\n";
      p_sock.precision(8);
      p_sock << "solution\n" << *pmesh << p << "window_title 'Pressure'"
             << endl;
      p_sock << "keys Rljmmc" << endl;
   }

   // 20. Free the used memory.
   delete fform;
   delete gform;
   delete darcy;
   delete W_space;
   delete R_space;
   delete trace_space;
   delete W_coll;
   delete R_coll;
   delete trace_coll;
   delete pmesh;

   return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(0.0);
   if (x.Size() == 3)
   {
      zi = x(2);
   }

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);

   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi);
   }
}

// Change if needed
real_t pFun_ex(const Vector & x)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
   f = 0.0;
}

real_t gFun(const Vector & x)
{
   if (x.Size() == 3)
   {
      return -pFun_ex(x);
   }
   else
   {
      return 0;
   }
}

real_t f_natural(const Vector & x)
{
   return (-pFun_ex(x));
}
