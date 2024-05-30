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
//                                 k*q + grad t = f
//                                 - div q      = g
//
//               with natural boundary condition -t = <given temperature>.
//               Here, we use a given exact solution (q,t) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (heat flux q) and piecewise discontinuous
//               polynomials (temperature t).
//
//               The example demonstrates the use of the BlockOperator class, as
//               well as the collective saving of several grid functions in
//               VisIt (visit.llnl.gov) and ParaView (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include "darcyform.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<real_t(const Vector &, real_t)> TFunc;
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, real_t, Vector &)> VecTFunc;

TFunc GetTFun(int prob, real_t t_0);
VecTFunc GetQFun(int prob, real_t t_0, real_t k, real_t c);
VecFunc GetCFun(int prob, real_t c);
TFunc GetFFun(int prob, real_t t_0, real_t k, const VecFunc &cFun);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "";
   int nx = 0;
   int ny = 0;
   int order = 1;
   bool dg = false;
   bool upwinded = false;
   int problem = 1;
   real_t k = 1.;
   real_t c = 1.;
   real_t td = 0.5;
   bool hybridization = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool mfem = false;
   bool visit = false;
   bool paraview = false;
   bool visualization = true;
   bool analytic = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&upwinded, "-up", "--upwinded", "-ce", "--centered",
                  "Switches between upwinded (1) and centered (0=default) stabilization.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem to solve from the Nguyen paper.");
   args.AddOption(&k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&c, "-c", "--velocity",
                  "Convection velocity");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mfem, "-mfem", "--mfem", "-no-mfem",
                  "--no-mfem",
                  "Enable or disable MFEM output.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable Visit output.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView output.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&analytic, "-anal", "--analytic", "-no-anal",
                  "--no-analytic",
                  "Enable or disable analytic solution.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Set the problem options
   bool bconv;
   switch (problem)
   {
      case 1:
         bconv = false;
         break;
      case 3:
         bconv = true;
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }

   if (!bconv && upwinded)
   {
      cerr << "Upwinded scheme cannot work without advection" << endl;
      return 1;
   }

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
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

   // Mark boundary conditions
   Array<int> bdr_is_dirichlet(mesh->bdr_attributes.Max());
   Array<int> bdr_is_neumann(mesh->bdr_attributes.Max());
   bdr_is_dirichlet = 0;
   bdr_is_neumann = 0;

   switch (problem)
   {
      case 1://free
         break;
      case 3:
         bdr_is_dirichlet[3] = -1;//inflow
         bdr_is_neumann[0] = -1;//outflow
         break;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   if (strlen(mesh_file) > 0)
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *V_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      V_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else
   {
      V_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim,
                                                         BasisType::GaussLobatto);

   FiniteElementSpace *V_space = new FiniteElementSpace(mesh, V_coll,
                                                        (dg)?(dim):(1));
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, W_coll);

   DarcyForm *darcy = new DarcyForm(V_space, W_space);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   const Array<int> &block_offsets = darcy->GetOffsets();

   std::cout << "***********************************************************\n";
   std::cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(V+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   const real_t t_0 = 1.; //base temperature

   ConstantCoefficient kcoeff(k);
   ConstantCoefficient ikcoeff(1./k);

   auto cFun = GetCFun(problem, c);
   VectorFunctionCoefficient ccoeff(dim, cFun);

   auto tFun = GetTFun(problem, t_0);
   FunctionCoefficient tcoeff(tFun);
   SumCoefficient gcoeff(0., tcoeff, 1., -1.);

   auto fFun = GetFFun(problem, t_0, k, cFun);
   FunctionCoefficient fcoeff(fFun);

   auto qFun = GetQFun(problem, t_0, k, c);
   VectorFunctionCoefficient qcoeff(dim, qFun);
   ConstantCoefficient one;
   VectorSumCoefficient qtcoeff(ccoeff, qcoeff, tcoeff, one);//total flux

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction q,t for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (q,t) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   x = 0.;
   GridFunction q_h, t_h;
   q_h.MakeRef(V_space, x.GetBlock(0), 0);
   t_h.MakeRef(W_space, x.GetBlock(1), 0);
   if (!dg)
   {
      q_h.ProjectBdrCoefficientNormal(qcoeff,
                                      bdr_is_neumann);   //essential Neumann BC
   }

   LinearForm *gform(new LinearForm);
   gform->Update(V_space, rhs.GetBlock(0), 0);
   if (dg)
   {
      gform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(gcoeff),
                                  bdr_is_dirichlet);
   }
   else
   {
      gform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                   bdr_is_dirichlet);
   }

   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   LinearForm *fform(new LinearForm);
   fform->Update(W_space, rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   if (!hybridization)
   {
      if (upwinded)
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qtcoeff, +1.),
                                     bdr_is_neumann);
      else
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qtcoeff, +1., 0.),
                                     bdr_is_neumann);
   }
   if (bconv)
   {
      fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +1.),
                                  bdr_is_dirichlet);
   }
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   q_h, v_h \in V_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   q_h \in V_h, w_h \in W_h
   BilinearForm *Mq = darcy->GetFluxMassForm();
   MixedBilinearForm *B = darcy->GetFluxDivForm();
   BilinearForm *Mt = ((dg && td > 0.) ||
                       bconv)?(darcy->GetPotentialMassForm()):(NULL);

   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
      B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      if (upwinded)
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(ccoeff, -1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            ccoeff, -1.)), bdr_is_neumann);
         if (td > 0. && hybridization)
         {
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td));
         }
      }
      else
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(-1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            -1.)), bdr_is_neumann);
         if (td > 0.)
         {
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
         }
      }
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
      B->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   }
   if (bconv)
   {
      Mt->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
      if (upwinded)
      {
         Mt->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
         Mt->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff),
                                  bdr_is_neumann);
      }
      else
      {
         Mt->AddInteriorFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                  bdr_is_neumann);
      }
   }

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg)
   {
      V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;

   chrono.Clear();
   chrono.Start();

   if (hybridization)
   {
      trace_coll = new RT_Trace_FECollection(order, dim, 0);
      //trace_coll = new DG_Interface_FECollection(order, dim, 0);
      trace_space = new FiniteElementSpace(mesh, trace_coll);
      darcy->EnableHybridization(trace_space,
                                 new NormalTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   darcy->Assemble();

   OperatorHandle pDarcyOp;
   Vector X, RHS;

   //Neumann BC for the hybridized system

   LinearForm *hform = NULL;

   if (hybridization)
   {
      RHS.SetSize(trace_space->GetVSize());
      hform = new LinearForm();
      hform->Update(trace_space, RHS, 0);
      //note that Neumann BC must be applied only for the heat flux
      //and not the total flux for stability reasons
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(qcoeff, 2),
                                   bdr_is_neumann);
      hform->Assemble();
   }

   //form the linear system

   darcy->FormLinearSystem(ess_flux_tdofs_list, x, rhs,
                           pDarcyOp, X, RHS);

   chrono.Stop();
   std::cout << "Assembly took " << chrono.RealTime() << "s.\n";


   int maxIter(1000);
   real_t rtol(1.e-6);
   real_t atol(1.e-10);

   if (hybridization)
   {
      // 10. Construct the preconditioner
      GSSmoother prec(*pDarcyOp.As<SparseMatrix>());

      // 11. Solve the linear system with GMRES.
      //     Check the norm of the unpreconditioned residual.
      chrono.Clear();
      chrono.Start();
      GMRESSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*pDarcyOp);
      solver.SetPreconditioner(prec);
      solver.SetPrintLevel(1);

      solver.Mult(RHS, X);
      darcy->RecoverFEMSolution(X, rhs, x);

      chrono.Stop();

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
   else
   {
      // 10. Construct the operators for preconditioner
      //
      //                 P = [ diag(M)         0         ]
      //                     [  0       B diag(M)^-1 B^T ]
      //
      //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
      //     temperature Schur Complement
      SparseMatrix *MinvBt = NULL;
      Vector Md(Mq->Height());

      BlockDiagonalPreconditioner darcyPrec(block_offsets);
      Solver *invM, *invS;
      SparseMatrix *S = NULL;

      if (pa)
      {
         Mq->AssembleDiagonal(Md);
         auto Md_host = Md.HostRead();
         Vector invMd(Mq->Height());
         for (int i=0; i<Mq->Height(); ++i)
         {
            invMd(i) = 1.0 / Md_host[i];
         }

         Vector BMBt_diag(B->Height());
         B->AssembleDiagonal_ADAt(invMd, BMBt_diag);

         Array<int> ess_tdof_list;  // empty

         invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
         invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
      }
      else
      {
         SparseMatrix &Mqm(Mq->SpMat());
         Mqm.GetDiag(Md);
         Md.HostReadWrite();

         SparseMatrix &Bm(B->SpMat());
         MinvBt = Transpose(Bm);

         for (int i = 0; i < Md.Size(); i++)
         {
            MinvBt->ScaleRow(i, 1./Md(i));
         }

         S = Mult(Bm, *MinvBt);
         if (Mt)
         {
            SparseMatrix &Mtm(Mt->SpMat());
            SparseMatrix *Snew = Add(Mtm, *S);
            delete S;
            S = Snew;
         }

         invM = new DSmoother(Mqm);

#ifndef MFEM_USE_SUITESPARSE
         invS = new GSSmoother(*S);
#else
         invS = new UMFPackSolver(*S);
#endif
      }

      invM->iterative_mode = false;
      invS->iterative_mode = false;

      darcyPrec.SetDiagonalBlock(0, invM);
      darcyPrec.SetDiagonalBlock(1, invS);

      // 11. Solve the linear system with MINRES.
      //     Check the norm of the unpreconditioned residual.

      chrono.Clear();
      chrono.Start();
      MINRESSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*pDarcyOp);
      solver.SetPreconditioner(darcyPrec);
      solver.SetPrintLevel(1);

      solver.Mult(RHS, X);
      darcy->RecoverFEMSolution(X, rhs, x);

      if (device.IsEnabled()) { x.HostRead(); }
      chrono.Stop();

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

      delete invM;
      delete invS;
      delete S;
      //delete Bt;
      delete MinvBt;
   }

   // 12. Compute the L2 error norms.

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   qcoeff.SetTime(1.);
   tcoeff.SetTime(1.);

   real_t err_q  = q_h.ComputeL2Error(qcoeff, irs);
   real_t norm_q = ComputeLpNorm(2., qcoeff, *mesh, irs);
   real_t err_t  = t_h.ComputeL2Error(tcoeff, irs);
   real_t norm_t = ComputeLpNorm(2., tcoeff, *mesh, irs);

   std::cout << "|| q_h - q_ex || / || q_ex || = " << err_q / norm_q << "\n";
   std::cout << "|| t_h - t_ex || / || t_ex || = " << err_t / norm_t << "\n";

   // Project the analytic solution

   GridFunction q_a, qt_a, t_a, c_gf;

   q_a.SetSpace(V_space);
   q_a.ProjectCoefficient(qcoeff);

   qt_a.SetSpace(V_space);
   qt_a.ProjectCoefficient(qtcoeff);

   t_a.SetSpace(W_space);
   t_a.ProjectCoefficient(tcoeff);

   if (bconv)
   {
      c_gf.SetSpace(V_space);
      c_gf.ProjectCoefficient(ccoeff);
   }

   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_q.gf" or "glvis -m ex5.mesh -g
   //     sol_t.gf".
   if (mfem)
   {
      ofstream mesh_ofs("ex5.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream q_ofs("sol_q.gf");
      q_ofs.precision(8);
      q_h.Save(q_ofs);

      ofstream t_ofs("sol_t.gf");
      t_ofs.precision(8);
      t_h.Save(t_ofs);
   }

   // 14. Save data in the VisIt format
   if (visit)
   {
      VisItDataCollection visit_dc("Example5", mesh);
      visit_dc.RegisterField("heat flux", &q_h);
      visit_dc.RegisterField("temperature", &t_h);
      if (analytic)
      {
         visit_dc.RegisterField("heat flux analytic", &q_a);
         visit_dc.RegisterField("temperature analytic", &t_a);
      }
      visit_dc.Save();
   }

   // 15. Save data in the ParaView format
   if (paraview)
   {
      ParaViewDataCollection paraview_dc("Example5", mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("heat flux",&q_h);
      paraview_dc.RegisterField("temperature",&t_h);
      if (analytic)
      {
         paraview_dc.RegisterField("heat flux analytic", &q_a);
         paraview_dc.RegisterField("temperature analytic", &t_a);
      }
      paraview_dc.Save();
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      const char vishost[] = "localhost";
      const int  visport   = 19916;
      socketstream q_sock(vishost, visport);
      q_sock.precision(8);
      q_sock << "solution\n" << *mesh << q_h << "window_title 'Heat flux'" << endl;
      q_sock << "keys Rljvvvvvmmc" << endl;
      q_sock.close();
      socketstream t_sock(vishost, visport);
      t_sock.precision(8);
      t_sock << "solution\n" << *mesh << t_h << "window_title 'Temperature'" << endl;
      t_sock << "keys Rljmmc" << endl;
      t_sock.close();
      if (analytic)
      {
         socketstream qa_sock(vishost, visport);
         qa_sock.precision(8);
         qa_sock << "solution\n" << *mesh << q_a << "window_title 'Heat flux analytic'"
                 << endl;
         qa_sock << "keys Rljvvvvvmmc" << endl;
         qa_sock.close();
         socketstream qta_sock(vishost, visport);
         qta_sock.precision(8);
         qta_sock << "solution\n" << *mesh << qt_a <<
                  "window_title 'Total flux analytic'"
                  << endl;
         qta_sock << "keys Rljvvvvvmmc" << endl;
         qta_sock.close();
         socketstream ta_sock(vishost, visport);
         ta_sock.precision(8);
         ta_sock << "solution\n" << *mesh << t_a << "window_title 'Temperature analytic'"
                 << endl;
         ta_sock << "keys Rljmmc" << endl;
         ta_sock.close();
         if (bconv)
         {
            socketstream c_sock(vishost, visport);
            c_sock.precision(8);
            c_sock << "solution\n" << *mesh << c_gf << "window_title 'Velocity'" << endl;
            c_sock << "keys Rljvvvvvmmc" << endl;
            c_sock.close();
         }
      }
   }

   // 17. Free the used memory.
   delete fform;
   delete gform;
   delete hform;
   //delete B;
   delete darcy;
   delete W_space;
   delete V_space;
   delete trace_space;
   delete W_coll;
   delete V_coll;
   delete trace_coll;
   delete mesh;

   return 0;
}

TFunc GetTFun(int prob, real_t t_0)
{
   switch (prob)
   {
      case 1:
         return [=](const Vector &x, real_t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0 * exp(x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));
            }

            return t0;
         };
      case 2:
         // null
         break;
      case 3:
         return [=](const Vector &x, real_t) -> real_t
         {
            Vector xc(x);
            //xc -= .5;
            real_t t0 = 1. - tanh(10. * (-1. + 4.*xc.Norml2()));
            return t0;
         };

   }
   return TFunc();
}

VecTFunc GetQFun(int prob, real_t t_0, real_t k, real_t c)
{
   switch (prob)
   {
      case 1:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            v = 0.;
            v(0) = t_0 * (sin(M_PI*x(0)) + M_PI * cos(M_PI*x(0))) * exp(
                      x.Sum()) * sin(M_PI*x(1));
            v(1) = t_0 * (sin(M_PI*x(1)) + M_PI * cos(M_PI*x(1))) * exp(
                      x.Sum()) * sin(M_PI*x(0));
            if (vdim > 2)
            {
               v(0) *= sin(M_PI*x(2));
               v(1) *= sin(M_PI*x(2));
               v(2) = t_0 * (sin(M_PI*x(2)) + M_PI * cos(M_PI*x(2))) * exp(
                         x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            }

            v *= -k;
         };
      case 2:
         // null
         break;
      case 3:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
            Vector xc(x);
            //xc -= .5;

            real_t r = xc.Norml2();
            if (r <= 0.) { return; }
            real_t csh = cosh(10. * (-1. + 4. * r));
            real_t q0 = k * 10. * 4. / (csh*csh * r);
            v.Set(q0, xc);
         };
         break;
   }
   return VecTFunc();
}

VecFunc GetCFun(int prob, real_t c)
{
   switch (prob)
   {
      case 1:
         // null
         break;
      case 2:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;

            v(0) = c;
            v(1) = c;
            if (ndim > 2)
            {
               v(2) = c;
            }
         };
      case 3:
         return [=](const Vector &x, Vector &v)
         {
            const int ndim = x.Size();
            v.SetSize(ndim);
            v = 0.;
            Vector xc(x);
            //xc -= .5;

            v(0) = +xc(1) * c;
            v(1) = -xc(0) * c;
         };
   }
   return VecFunc();
}

TFunc GetFFun(int prob, real_t t_0, real_t k, const VecFunc &cFun)
{
   switch (prob)
   {
      case 1:
         return [=](const Vector &x, real_t) -> real_t
         {
            const int ndim = x.Size();

            real_t t0   = t_0 * exp(x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            real_t diff = -k * exp(x.Sum()) * (sin(M_PI*x(1)) * ((1-M_PI*M_PI) * sin(M_PI*x(0))
                                                                 + 2 * M_PI * cos(M_PI*x(0)))
                                               + sin(M_PI*x(0)) * ((1-M_PI*M_PI)
                                                                   * sin(M_PI*x(1)) + 2 * M_PI
                                                                   * cos(M_PI*x(1))));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));

               diff *= sin(M_PI*x(2));
               diff -= k*M_PI*M_PI*t0;
            }

            return -diff;
         };
      case 2:
         return [=](const Vector &x, real_t) -> real_t
         {
            // PLACEHOLDER
            const int ndim = x.Size();
            Vector c;
            c.SetSize(ndim);
            cFun(x, c);

            real_t t0   = t_0 * exp(x.Sum()) * sin(M_PI*x(0)) * sin(M_PI*x(1));
            real_t conv = t_0 * c(0) * (t0 + M_PI * exp(x.Sum()) * cos(M_PI*x(0))
                                        * sin(M_PI*x(1))) + t_0 * c(1) * (t0 + M_PI
                                                                          * exp(x.Sum() * cos(M_PI*x(1))
                                                                                * sin(M_PI*x(0))));
            if (ndim > 2)
            {
               conv *= sin(M_PI*x(2));
               conv += c(2) * (t0 + M_PI * exp(x.Sum()) * cos(M_PI*x(2)) * sin(M_PI*x(
                                                                                  0))) * sin(M_PI*x(1));
            }
            return -conv;
         };
      case 3:
      {
         return [](const Vector &x, real_t) -> real_t { return 0.; };
      }
   }
   return TFunc();
}
