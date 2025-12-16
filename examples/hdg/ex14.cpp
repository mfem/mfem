//                                MFEM Example 14
//
// Compile with: make ex14
//
// Sample runs:  ex14 -m ../data/inline-quad.mesh -o 0
//               ex14 -m ../data/star.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 2 -o 2 -k 0 -e 1
//               ex14 -m ../data/escher.mesh -s 1
//               ex14 -m ../data/fichera.mesh -s 1 -k 1
//               ex14 -m ../data/fichera-mixed.mesh -s 1 -k 1
//               ex14 -m ../data/square-disc-p2.vtk -r 3 -o 2
//               ex14 -m ../data/square-disc-p3.mesh -r 2 -o 3
//               ex14 -m ../data/square-disc-nurbs.mesh -o 1
//               ex14 -m ../data/disc-nurbs.mesh -r 3 -o 2 -s 1 -k 0
//               ex14 -m ../data/pipe-nurbs.mesh -o 1
//               ex14 -m ../data/inline-segment.mesh -r 5
//               ex14 -m ../data/amr-quad.mesh -r 3
//               ex14 -m ../data/amr-hex.mesh
//               ex14 -m ../data/fichera-amr.mesh
//               ex14 -pa -r 1 -o 3
//               ex14 -pa -r 1 -o 3 -m ../data/fichera.mesh
//
// Device sample runs:
//               ex14 -pa -r 2 -d cuda -o 3
//               ex14 -pa -r 2 -d cuda -o 3 -m ../data/fichera.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Poisson problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Finite element spaces of any order,
//               including zero on regular grids, are supported. The example
//               highlights the use of discontinuous spaces and DG-specific face
//               integrators.
//
//               We recommend viewing examples 1 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = -1;
   int order = 1;
   bool dg = false;
   bool brt = false;
   real_t td = 0.5;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = false;
   bool pa = false;
   bool visualization = 1;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction of DG flux.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default DG).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. By default, or if ref_levels < 0,
   //    we choose it to be the largest number that gives a final mesh with no
   //    more than 50,000 elements.
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }

   // 5. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *R_coll;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      R_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else if (brt)
   {
      R_coll = new BrokenRT_FECollection(order, dim);
   }
   else
   {
      R_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim);

   FiniteElementSpace *R_space = new FiniteElementSpace(&mesh, R_coll,
                                                        (dg)?(dim):(1));
   FiniteElementSpace *W_space = new FiniteElementSpace(&mesh, W_coll);

   cout << "Number of flux unknowns: " << R_space->GetVSize() << endl;
   cout << "Number of potential unknowns: " << W_space->GetVSize() << endl;

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.

   DarcyForm darcy(R_space, W_space);
   const Array<int> &block_offsets = darcy.GetOffsets();
   const Array<int> &block_trueOffsets = darcy.GetTrueOffsets();

   ConstantCoefficient one(1.0), negone(-1.0);

   LinearForm *f = darcy.GetPotentialRHS();
   f->AddDomainIntegrator(new DomainLFIntegrator(negone));

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After assembly and finalizing we
   //    extract the corresponding sparse matrix A.
   BilinearForm *Mq = darcy.GetFluxMassForm();
   MixedBilinearForm *Bq = darcy.GetFluxDivForm();
   BilinearForm *Mu = (dg)?(darcy.GetPotentialMassForm()):(NULL);

   if (dg)
   {
      Mq->AddDomainIntegrator(new VectorMassIntegrator());
      Bq->AddDomainIntegrator(new VectorDivergenceIntegrator());
      Bq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                       new DGNormalTraceIntegrator(-1.)));
      Mu->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, td));
   }
   else
   {
      Mq->AddDomainIntegrator(new VectorFEMassIntegrator());
      Bq->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      if (brt)
      {
         Bq->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                          new DGNormalTraceIntegrator(-1.)));
      }
   }

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;

   if (hybridization)
   {
      if (trace_h1)
      {
         trace_coll = new H1_Trace_FECollection(order+1, dim);
      }
      else
      {
         trace_coll = new DG_Interface_FECollection(order, dim);
      }
      trace_space = new FiniteElementSpace(&mesh, trace_coll);
      darcy.EnableHybridization(trace_space,
                                new NormalTraceJumpIntegrator(),
                                ess_flux_tdofs_list);
   }
   else if (reduction && (dg || brt))
   {
      darcy.EnableFluxReduction();
   }

   if (pa) { darcy.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   darcy.Assemble();

   OperatorHandle A;
   Vector X, B;
   darcy.FormLinearSystem(ess_flux_tdofs_list, x, A, X, B);

   // 9. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one. (Note that tolerances are squared: 1e-12 corresponds
   //    to a relative tolerance of 1e-6).
   //
   //    If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.

   constexpr int maxIter(500);
   constexpr real_t rtol(1.0e-6);
   constexpr real_t atol(0.0);

   if (hybridization || (reduction && (dg || brt)))
   {
      // 10. Construct the preconditioner
      GSSmoother prec;

      // 11. Solve the linear system with GMRES.
      //     Check the norm of the unpreconditioned residual.
      GMRESSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetPreconditioner(prec);
      solver.SetOperator(*A);
      solver.SetPrintLevel(1);

      solver.Mult(B, X);
   }
   else
   {
      // 10. Construct the operators for preconditioner
      //
      //                 P = [ diag(M)         0         ]
      //                     [  0       B diag(M)^-1 B^T ]
      //
      //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
      //     pressure Schur Complement
      SparseMatrix *MinvBt = NULL;
      Vector Md(Mq->Height());

      BlockDiagonalPreconditioner darcyPrec(block_trueOffsets);
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

         Vector BMBt_diag(Bq->Height());
         Bq->AssembleDiagonal_ADAt(invMd, BMBt_diag);

         Array<int> ess_tdof_list;  // empty

         invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
         invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
      }
      else
      {
         SparseMatrix &M(Mq->SpMat());
         M.GetDiag(Md);
         Md.HostReadWrite();

         SparseMatrix &B(Bq->SpMat());
         MinvBt = Transpose(B);

         for (int i = 0; i < Md.Size(); i++)
         {
            MinvBt->ScaleRow(i, 1./Md(i));
         }

         S = Mult(B, *MinvBt);
         if (Mu)
         {
            SparseMatrix &Mum(Mu->SpMat());
            SparseMatrix *Snew = Add(Mum, *S);
            delete S;
            S = Snew;
         }

         invM = new DSmoother(M);

#ifndef MFEM_USE_SUITESPARSE
         invS = new GSSmoother(*S);
#else
         auto umf_solver = new UMFPackSolver(*S);
         umf_solver->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         invS = umf_solver;
#endif
      }

      invM->iterative_mode = false;
      invS->iterative_mode = false;

      darcyPrec.SetDiagonalBlock(0, invM);
      darcyPrec.SetDiagonalBlock(1, invS);

      // 11. Solve the linear system with MINRES.
      //     Check the norm of the unpreconditioned residual.

      MINRESSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*A);
      solver.SetPreconditioner(darcyPrec);
      solver.SetPrintLevel(1);

      solver.Mult(B, X);

      delete invM;
      delete invS;
      delete S;
      delete MinvBt;
   }

   darcy.RecoverFEMSolution(X, x);
   if (device.IsEnabled()) { x.HostRead(); }

   // 10. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   GridFunction u(W_space, x.GetBlock(1), 0);

   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   u.Save(sol_ofs);

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << u << flush;
   }

   delete W_space;
   delete R_space;
   delete trace_space;
   delete W_coll;
   delete R_coll;
   delete trace_coll;

   return 0;
}
