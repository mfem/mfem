//                                MFEM Example 40
//
// Compile with: make ex40
//
// Sample runs:  ex40 -m ../data/square-disc.mesh
//               ex40 -m ../data/star.mesh
//               ex40 -m ../data/star.mesh -pa
//               ex40 -m ../data/beam-tet.mesh
//               ex40 -m ../data/beam-hex.mesh
//               ex40 -m ../data/beam-hex.mesh -pa
//               ex40 -m ../data/escher.mesh
//               ex40 -m ../data/fichera.mesh
//
// Device sample runs:
//               ex40 -m ../data/star.mesh -pa -d cuda
//               ex40 -m ../data/star.mesh -pa -d raja-cuda
//               ex40 -m ../data/star.mesh -pa -d raja-omp
//               ex40 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                                 k*u + grad p = f
//                                 - div u      = g + dp/dt
//
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockOperator class, as
//               well as the collective saving of several grid functions in
//               VisIt (visit.llnl.gov) and ParaView (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.
//
//               We use backward Euler with dt = t_n+1 - t_n to discretize dp/dt
//
//                                 k*u(t_n+1) + grad p(t_n+1)   = f(t_n+1)
//                                 - div u(t_n+1) - p(t_n+1)/dt = g(t_n+1) - p(t_n)/dt

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double f_natural(const Vector & x);

void Getrhs(FiniteElementSpace *R, FiniteElementSpace *W,
            LinearForm *f, LinearForm *g, VectorCoefficient &fcoeff,
            Coefficient &fnatcoeff, Coefficient &gcoeff, int i, double dt,
            Vector &p, SparseMatrix *C, BlockVector &rhs);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   double t_final = 1.0;
   double dt = 0.5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
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
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   //Mesh *mesh = new Mesh(mesh_file, 1, 1);
   //int dim = mesh->Dimension();

   int n = 10, dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, 1, 1.0, 1.0);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   //{
   //   int ref_levels =
   //      (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
   //   for (int l = 0; l < ref_levels; l++)
   //   {
   //      mesh->UniformRefinement();
   //   }
   //}

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

   FiniteElementSpace *R_space = new FiniteElementSpace(&mesh, hdiv_coll);
   FiniteElementSpace *W_space = new FiniteElementSpace(&mesh, l2_coll);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = R_space->GetVSize();
   block_offsets[2] = W_space->GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0);
   ConstantCoefficient c(1.0/dt);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient fnatcoeff(f_natural);
   DeltaCoefficient gcoeff(0.5,0.5,1.0);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   // Initial condition
   GridFunction pp(W_space);
   pp.ProjectCoefficient(pcoeff);
   pp.SetTrueVector();
   Vector p0;

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   LinearForm *fform(new LinearForm);
   LinearForm *gform(new LinearForm);

   // 9. Assemble the finite element matrices for the Darcy operator with backward Euler
   //
   //                                [ M  B^T ] [u(t_n+1)] = [f(t_n+1)] + [0     0][u(t_n)]
   //                                [ B  C/dt] [p(t_n+1)] = [g(t_n+1)] + [0  C/dt][p(t_n)]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   //     C   = -\int_\Omega p_h q_h d\Omega        p_h \in W_h, q_h \in W_h
   BilinearForm *mVarf(new BilinearForm(R_space));
   MixedBilinearForm *bVarf(new MixedBilinearForm(R_space, W_space));
   BilinearForm *cVarf(new BilinearForm(W_space));

   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   mVarf->Assemble();
   mVarf->Finalize();

   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->Assemble();
   bVarf->Finalize();

   cVarf->AddDomainIntegrator(new MassIntegrator(c));
   cVarf->Assemble();
   cVarf->Finalize();

   BlockOperator darcyOp(block_offsets);

   TransposeOperator *Bt = NULL;

   SparseMatrix &M(mVarf->SpMat());
   SparseMatrix &B(bVarf->SpMat());
   SparseMatrix &C(cVarf->SpMat());
   C *= -1;
   B *= -1.;
   if (Device::IsEnabled()) { B.BuildTranspose(); }
   Bt = new TransposeOperator(&B);

   darcyOp.SetBlock(0,0, &M);
   darcyOp.SetBlock(0,1, Bt);
   darcyOp.SetBlock(1,0, &B);
   darcyOp.SetBlock(1,1, &C);

   // 10. Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement
   SparseMatrix *MinvBt = NULL;
   Vector Md(mVarf->Height());

   BlockDiagonalPreconditioner darcyPrec(block_offsets);
   Solver *invM, *invS;
   SparseMatrix *S = NULL;

   M.GetDiag(Md);
   Md.HostReadWrite();

   MinvBt = Transpose(B);

   for (int i = 0; i < Md.Size(); i++)
   {
      MinvBt->ScaleRow(i, 1./Md(i));
   }

   S = Mult(B, *MinvBt);
   invM = new DSmoother(M);

#ifndef MFEM_USE_SUITESPARSE
   invS = new GSSmoother(*S);
#else
   invS = new UMFPackSolver(*S);
#endif

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   darcyPrec.SetDiagonalBlock(0, invM);
   darcyPrec.SetDiagonalBlock(1, invS);

   // 11a. Solve the linear system with MINRES at each time step
   // Check the norm of the unpreconditioned residual.
   int maxIter(1000);
   double rtol(1.e-6);
   double atol(1.e-10);
   MINRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(darcyOp);
   solver.SetPreconditioner(darcyPrec);
   solver.SetPrintLevel(1);

   // 11b. Perform time-integration
   int nt = (int) t_final / dt;
   for (int i = 0; i <= nt; i++)
   {
      // Initialize solution vector x=[u,p]
      x = 0.0;
      if (i==0)
      {
         p0 = pp.GetTrueVector();
      }
      else
      {
         p0 =  x.GetBlock(1);
      }
      Getrhs(R_space, W_space, fform, gform, fcoeff,
             fnatcoeff, gcoeff, i, dt, p0, &C, rhs);

      solver.Mult(rhs, x);

      if (device.IsEnabled()) { x.HostRead(); }

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

   }

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p;
   u.MakeRef(R_space, x.GetBlock(0), 0);
   p.MakeRef(W_space, x.GetBlock(1), 0);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u  = u.ComputeL2Error(ucoeff, irs);
   double norm_u = ComputeLpNorm(2., ucoeff, mesh, irs);
   double err_p  = p.ComputeL2Error(pcoeff, irs);
   double norm_p = ComputeLpNorm(2., pcoeff, mesh, irs);

   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   //{
   //   ofstream mesh_ofs("ex5.mesh");
   //   mesh_ofs.precision(8);
   //   mesh->Print(mesh_ofs);

   //   ofstream u_ofs("sol_u.gf");
   //   u_ofs.precision(8);
   //   u.Save(u_ofs);

   //   ofstream p_ofs("sol_p.gf");
   //   p_ofs.precision(8);
   //   p.Save(p_ofs);
   //}

   // 14. Save data in the VisIt format
   VisItDataCollection visit_dc("Example40", &mesh);
   visit_dc.RegisterField("velocity", &u);
   visit_dc.RegisterField("pressure", &p);
   visit_dc.Save();

   // 15. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example40", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.Save();

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << mesh << u << "window_title 'Velocity'" << endl;
      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << mesh << p << "window_title 'Pressure'" << endl;
   }

   // 17. Free the used memory.
   delete fform;
   delete gform;
   delete invM;
   delete invS;
   delete S;
   delete Bt;
   delete MinvBt;
   delete mVarf;
   delete bVarf;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;
   //delete mesh;

   return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);
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
double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
   f = 9.8;
}


double f_natural(const Vector & x)
{
   return (-pFun_ex(x));
}

void Getrhs(FiniteElementSpace *R, FiniteElementSpace *W,
            LinearForm *f, LinearForm *g, VectorCoefficient &fcoeff,
            Coefficient &fnatcoeff, Coefficient &gcoeff, int i, double dt,
            Vector &p, SparseMatrix *C, BlockVector &rhs)
{
   // First block of rhs, rhs[0] = f
   f->Update(R, rhs.GetBlock(0), 0);
   f->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   f->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
   f->Assemble();
   f->SyncAliasMemory(rhs);
   // Second block of rhs, rhs[1] = g
   g->Update(W, rhs.GetBlock(1), 0);
   g->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   g->Assemble();
   g->SyncAliasMemory(rhs);


   // Allocate vector
   Vector v;
   int vsize = W->GetVSize();
   v.UseDevice(true);
   v.SetSize(vsize);
   v = 0.0;
   // v = C*p
   C->Mult(p, v);
   // v = v/dt
   v /= dt;

   rhs *=exp(-(i+1)*dt);
   // Update rhs[1]= rhs[1] + v
   add(v, rhs.GetBlock(1), rhs.GetBlock(1));

}