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
// Description:  This example code solves a simple 2D mixed electromagnetic
//               diffusion problem corresponding to the mixed system
//
//                                 sigma E - curl B = f
//                                  curl E +      B = g
//
//               with essential boundary condition E x n = <given tangential field>.
//               Here, we use a given exact solution (E,B) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Nedelec
//               finite elements (electric field E) and piecewise discontinuous
//               integral polynomials (magnetic field B).
//
//               The example demonstrates the use of the DarcyForm class, as
//               well as the collective saving of several grid functions in
//               VisIt (visit.llnl.gov) and ParaView (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void EFun_ex(const Vector & x, Vector & E);
real_t BFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
real_t gFun(const Vector & x);
real_t freq = 1.0, kappa;

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "";
   int nx = 0;
   int ny = 0;
   int order = 1;
   bool hybridization = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
   kappa = freq * M_PI;

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
   FiniteElementCollection *R_coll(new ND_FECollection(order+1, dim));
   FiniteElementCollection *W_coll(new L2_FECollection(order, dim, 0,
                                                       FiniteElement::INTEGRAL));

   FiniteElementSpace *R_space = new FiniteElementSpace(mesh, R_coll);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, W_coll);

   DarcyForm *darcy = new DarcyForm(R_space, W_space);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   const Array<int> &block_offsets = darcy->GetOffsets();

   std::cout << "***********************************************************\n";
   std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient muinvsqrt(1.0);
   ConstantCoefficient sigma(1.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient Ecoeff(dim, EFun_ex);
   FunctionCoefficient Bcoeff(BFun_ex);

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction E,B for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (E,B) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   LinearForm *fform(new LinearForm);
   fform->Update(R_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   LinearForm *gform(new LinearForm);
   gform->Update(W_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  C^T ]
   //                                [ C   0  ]
   //     where:
   //
   //     M = \int_\Omega \sigma E_h \cdot v_h d\Omega   E_h, v_h \in R_h
   //     C   = \int_\Omega \curl E_h q_h d\Omega   E_h \in R_h, q_h \in W_h
   //BilinearForm *mEVarf(new BilinearForm(R_space));
   //MixedBilinearForm *cVarf(new MixedBilinearForm(R_space, W_space));
   BilinearForm *mEVarf = darcy->GetFluxMassForm();
   MixedBilinearForm *cVarf = darcy->GetFluxDivForm();
   BilinearForm *mBVarf = darcy->GetPotentialMassForm();

   //if (pa) { mEVarf->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   mEVarf->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   //mEVarf->Assemble();
   //if (!pa) { mEVarf->Finalize(); }

   //if (pa) { cVarf->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   cVarf->AddDomainIntegrator(new MixedScalarCurlIntegrator(muinvsqrt));
   //cVarf->Assemble();
   //if (!pa) { cVarf->Finalize(); }

   mBVarf->AddDomainIntegrator(new MassIntegrator());

   //set essential boundary condition

   GridFunction E, B;
   E.MakeRef(R_space, x.GetBlock(0), 0);
   B.MakeRef(W_space, x.GetBlock(1), 0);

   Array<int> ess_flux_tdofs_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      R_space->GetEssentialTrueDofs(ess_bdr, ess_flux_tdofs_list);
      E.ProjectBdrCoefficientTangent(Ecoeff, ess_bdr);
   }

   //set hybridization / assembly level

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;

   chrono.Clear();
   chrono.Start();

   if (hybridization)
   {
      trace_coll = new ND_Trace_FECollection(order+1, dim);
      trace_space = new FiniteElementSpace(mesh, trace_coll);
      darcy->EnableHybridization(trace_space,
                                 new TangentTraceJumpIntegrator(),
                                 ess_flux_tdofs_list);
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   darcy->Assemble();
   //if (!pa) { darcy->Finalize(); }

   OperatorHandle pDarcyOp;
   Vector X, RHS;

   //darcy->FormSystemMatrix(ess_flux_tdofs_list, pDarcyOp);
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
      //                     [  0       C diag(M)^-1 C^T ]
      //
      //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
      //     magnetic field Schur Complement
      SparseMatrix *MinvBt = NULL;
      Vector Md(mEVarf->Height());

      BlockDiagonalPreconditioner darcyPrec(block_offsets);
      Solver *invM, *invS;
      SparseMatrix *S = NULL;

      if (pa)
      {
         mEVarf->AssembleDiagonal(Md);
         auto Md_host = Md.HostRead();
         Vector invMd(mEVarf->Height());
         for (int i=0; i<mEVarf->Height(); ++i)
         {
            invMd(i) = 1.0 / Md_host[i];
         }

         Vector BMBt_diag(cVarf->Height());
         cVarf->AssembleDiagonal_ADAt(invMd, BMBt_diag);

         Array<int> ess_tdof_list;  // empty

         invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
         invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
      }
      else
      {
         SparseMatrix &M(mEVarf->SpMat());
         M.GetDiag(Md);
         Md.HostReadWrite();

         SparseMatrix &C(cVarf->SpMat());
         MinvBt = Transpose(C);

         for (int i = 0; i < Md.Size(); i++)
         {
            MinvBt->ScaleRow(i, 1./Md(i));
         }

         S = Mult(C, *MinvBt);
         if (mBVarf)
         {
            SparseMatrix &Mtm(mBVarf->SpMat());
            SparseMatrix *Snew = Add(Mtm, *S);
            delete S;
            S = Snew;
         }

         invM = new DSmoother(M);

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
      FGMRESSolver solver;
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
      std::cout << "MINRES solver took " << chrono.RealTime() << "s.\n";

      delete invM;
      delete invS;
      delete S;
      //delete Bt;
      delete MinvBt;
   }

   // 12. Create the grid functions E and B. Compute the L2 error norms.

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   real_t err_E  = E.ComputeL2Error(Ecoeff, irs);
   real_t norm_E = ComputeLpNorm(2., Ecoeff, *mesh, irs);
   real_t err_B  = B.ComputeL2Error(Bcoeff, irs);
   real_t norm_B = ComputeLpNorm(2., Bcoeff, *mesh, irs);

   std::cout << "|| E_h - E_ex || / || E_ex || = " << err_E / norm_E << "\n";
   std::cout << "|| B_h - B_ex || / || B_ex || = " << err_B / norm_B << "\n";

   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("ex5.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream E_ofs("sol_E.gf");
      E_ofs.precision(8);
      E.Save(E_ofs);

      ofstream B_ofs("sol_B.gf");
      B_ofs.precision(8);
      B.Save(B_ofs);
   }

   // 14. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5", mesh);
   visit_dc.RegisterField("electric field", &E);
   visit_dc.RegisterField("magnetic field", &B);
   visit_dc.Save();

   // 15. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example5", mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("electric field",&E);
   paraview_dc.RegisterField("magnetic field",&B);
   paraview_dc.Save();


   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream E_sock(vishost, visport);
      E_sock.precision(8);
      E_sock << "solution\n" << *mesh << E << "window_title 'Electric field'" << endl;
      E_sock << "keys Rljvvvvvmmc" << endl;
      socketstream B_sock(vishost, visport);
      B_sock.precision(8);
      B_sock << "solution\n" << *mesh << B << "window_title 'Magnetic field'" << endl;
      B_sock << "keys Rljmmc" << endl;
   }

   // 17. Free the used memory.
   delete fform;
   delete gform;
   //delete mEVarf;
   //delete cVarf;
   delete darcy;
   delete W_space;
   delete R_space;
   delete trace_space;
   delete W_coll;
   delete R_coll;
   delete trace_coll;
   delete mesh;

   return 0;
}


void EFun_ex(const Vector & x, Vector & E)
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
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

real_t BFun_ex(const Vector & x)
{
   return kappa * (-cos(kappa * x(0)) + cos(kappa * x(1)));
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

real_t gFun(const Vector & x)
{
   return 0.;
}
