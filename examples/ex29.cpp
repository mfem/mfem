//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//             > mpirun -np 4 ex1p -pa -d cuda
//             > mpirun -np 4 ex1p -pa -d occa-cuda
//             > mpirun -np 4 ex1p -pa -d raja-omp
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double a_ = 0.2;

// Normal to hole with boundary attribute 4
void n4Vec(const Vector &x, Vector &n) { n = x; n[0] -= 0.5; n /= -n.Norml2(); }

Mesh * GenerateSerialMesh(int ref);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ser_ref_levels = 2;
   int order = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   bool h1 = true;
   bool visualization = true;

   double mat_val = 1.0;
   double dbc_val = 0.0;
   double nbc_val = 1.0;
   double rbc_a_val = 1.0; // du/dn + a * u = b
   double rbc_b_val = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&h1, "-h1", "--continuous", "-dg", "--discontinuous",
                  "Select continuous \"H1\" or discontinuous \"DG\" basis.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&mat_val, "-mat", "--material-value",
                  "Constant value for material coefficient "
                  "in the Laplace operator.");
   args.AddOption(&dbc_val, "-dbc", "--dirichlet-value",
                  "Constant value for Dirichlet Boundary Condition.");
   args.AddOption(&nbc_val, "-nbc", "--neumann-value",
                  "Constant value for Neumann Boundary Condition.");
   args.AddOption(&rbc_a_val, "-rbc-a", "--robin-a-value",
                  "Constant 'a' value for Robin Boundary Condition: "
                  "du/dn + a * u = b.");
   args.AddOption(&rbc_b_val, "-rbc-b", "--robin-b-value",
                  "Constant 'b' value for Robin Boundary Condition: "
                  "du/dn + a * u = b.");
   args.AddOption(&a_, "-a", "--radius",
                  "Radius of holes in the mesh.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   if (kappa < 0 && !h1)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(mfem::out);

   if (a_ < 0.01)
   {
      mfem::out << "Hole radius too small, resetting to 0.01.\n";
      a_ = 0.01;
   }
   if (a_ > 0.49)
   {
      mfem::out << "Hole radius too large, resetting to 0.49.\n";
      a_ = 0.49;
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = GenerateSerialMesh(ser_ref_levels);
   int dim = mesh->Dimension();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec = h1 ?
                                  (FiniteElementCollection*)new H1_FECollection(order, dim) :
                                  (FiniteElementCollection*)new DG_FECollection(order, dim);
   FiniteElementSpace fespace(mesh, fec);
   int size = fespace.GetTrueVSize();
   mfem::out << "Number of finite element unknowns: " << size << endl;

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> nbc_bdr(mesh->bdr_attributes.Max());
   Array<int> rbc_bdr(mesh->bdr_attributes.Max());
   Array<int> dbc_bdr(mesh->bdr_attributes.Max());

   nbc_bdr = 0; nbc_bdr[0] = 1;
   rbc_bdr = 0; rbc_bdr[1] = 1;
   dbc_bdr = 0; dbc_bdr[2] = 1;

   Array<int> ess_tdof_list(0);
   if (h1 && mesh->bdr_attributes.Size())
   {
      fespace.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ConstantCoefficient matCoef(mat_val);
   ConstantCoefficient dbcCoef(dbc_val);
   ConstantCoefficient nbcCoef(nbc_val);
   ConstantCoefficient rbcACoef(rbc_a_val);
   ConstantCoefficient rbcBCoef(rbc_b_val);

   ProductCoefficient m_nbcCoef(matCoef, nbcCoef);
   ProductCoefficient m_rbcACoef(matCoef, rbcACoef);
   ProductCoefficient m_rbcBCoef(matCoef, rbcBCoef);

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
   if (h1)
   {
      a.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
   }
   else
   {
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(matCoef,
                                                            sigma, kappa));
      a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(matCoef, sigma, kappa),
                             dbc_bdr);
      a.AddBdrFaceIntegrator(new BoundaryMassIntegrator(m_rbcACoef),
                             rbc_bdr);
   }

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   a.Assemble();

   LinearForm b(&fespace);

   if (h1)
   {
      x.ProjectCoefficient(dbcCoef);

      b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);
      b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_rbcBCoef), rbc_bdr);
   }
   else
   {
      b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(dbcCoef, matCoef,
                                                         sigma, kappa),
                             dbc_bdr);
      b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(m_nbcCoef),
                             nbc_bdr);
      b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(m_rbcBCoef),
                             rbc_bdr);
   }
   b.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   GSSmoother M((SparseMatrix&)(*A));
   if (sigma == -1.0)
   {
      PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
   }
   else
   {
      GMRES(*A, M, B, X, 1, 500, 10, 1e-12, 0.0);
   }
#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
#endif

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   BilinearForm *m = new BilinearForm(&fespace);
   m->AddDomainIntegrator(new MassIntegrator);
   m->Assemble();

   BilinearForm *n = new BilinearForm(&fespace);
   {
      Vector nVec(2); nVec[0] = 0.0; nVec[1] = -1.0;
      VectorConstantCoefficient nCoef(nVec);
      n->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(nCoef));
      n->Assemble();
   }

   BilinearForm *n0 = new BilinearForm(&fespace);
   {
      VectorFunctionCoefficient n0Coef(2, n4Vec);
      n0->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(n0Coef));
      n0->Assemble();
   }

   BilinearForm *r = new BilinearForm(&fespace);
   {
      Vector rVec(2); rVec[0] = 0.0; rVec[1] = 1.0;
      VectorConstantCoefficient rCoef(rVec);
      r->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(rCoef));
      r->Assemble();
   }

   GridFunction dx(&fespace);
   GridFunction nx(&fespace);
   GridFunction n0x(&fespace);
   GridFunction rx(&fespace);
   {
      LinearForm db(&fespace);

      ess_tdof_list.SetSize(0);

      OperatorPtr M;

      dx = x;

      ConstantCoefficient one(1.0);
      BilinearForm m_dbc(&fespace);
      if (h1)
      {
         m_dbc.AddBoundaryIntegrator(new MassIntegrator, dbc_bdr);
      }
      else
      {
         m_dbc.AddBdrFaceIntegrator(new BoundaryMassIntegrator(one), dbc_bdr);
      }
      m_dbc.Assemble();

      LinearForm m1x(&fespace);
      m_dbc.Mult(dx, m1x);

      double dbc_int = copysign(sqrt(m1x(dx) / (2.0 * M_PI * a_)), dbc_val);
      double dbc_err = fabs(dbc_int - dbc_val);

      bool hom_dbc = (dbc_val == 0.0);
      dbc_err /=  hom_dbc ? 1.0 : fabs(dbc_val);
      mfem::out << "dbc " << dbc_int << ", "
                << (hom_dbc ? "absolute" : "relative")
                << " error " << dbc_err << endl;
   }
   {
      LinearForm nb(&fespace);

      ess_tdof_list.SetSize(0);

      OperatorPtr M;

      n->Mult(x, nb);
      m->FormLinearSystem(ess_tdof_list, nx, nb, M, X, B);

      CGSolver mcg;
      mcg.SetRelTol(1e-12);
      mcg.SetMaxIter(2000);
      mcg.SetPrintLevel(0);
      mcg.SetOperator(*M);
      mcg.Mult(B, X);

      m->RecoverFEMSolution(X, nb, nx);

      // nx -= nbc_val;

      ConstantCoefficient one(1.0);
      BilinearForm m_nbc(&fespace);
      if (h1)
      {
         m_nbc.AddBoundaryIntegrator(new MassIntegrator, nbc_bdr);
      }
      else
      {
         m_nbc.AddBdrFaceIntegrator(new BoundaryMassIntegrator(one), nbc_bdr);
      }
      m_nbc.Assemble();

      LinearForm m3x(&fespace);
      m_nbc.Mult(nx, m3x);

      double nbc_int = copysign(sqrt(m3x(nx) * 0.5), nbc_val);
      double nbc_err = fabs(nbc_int - nbc_val);

      bool hom_nbc = (nbc_val == 0.0);
      nbc_err /=  hom_nbc ? 1.0 : fabs(nbc_val);
      mfem::out << "nbc " << nbc_int << ", "
                << (hom_nbc ? "absolute" : "relative")
                << " error " << nbc_err << endl;
   }
   {
      LinearForm nb(&fespace);

      ess_tdof_list.SetSize(0);

      OperatorPtr M;

      n0->Mult(x, nb);
      m->FormLinearSystem(ess_tdof_list, n0x, nb, M, X, B);

      CGSolver mcg;
      mcg.SetRelTol(1e-12);
      mcg.SetMaxIter(2000);
      mcg.SetPrintLevel(0);
      mcg.SetOperator(*M);
      mcg.Mult(B, X);

      m->RecoverFEMSolution(X, nb, n0x);

      // nx -= nbc_val;
      Array<int> nbc0_bdr(mesh->bdr_attributes.Max());
      nbc0_bdr = 0;
      nbc0_bdr[3] = 1;

      ConstantCoefficient one(1.0);
      BilinearForm m_nbc(&fespace);
      if (h1)
      {
         m_nbc.AddBoundaryIntegrator(new MassIntegrator, nbc0_bdr);
      }
      else
      {
         m_nbc.AddBdrFaceIntegrator(new BoundaryMassIntegrator(one), nbc0_bdr);
      }
      m_nbc.Assemble();

      LinearForm m4x(&fespace);
      m_nbc.Mult(n0x, m4x);

      double nbc_int = sqrt(m4x(n0x));
      double nbc_err = fabs(nbc_int);

      bool hom_nbc = true;
      mfem::out << "nbc0 " << nbc_int << ", "
                << (hom_nbc ? "absolute" : "relative")
                << " error " << nbc_err << endl;
   }
   {
      OperatorPtr M;

      LinearForm rb(&fespace);
      r->Mult(x, rb);
      m->FormLinearSystem(ess_tdof_list, rx, rb, M, X, B);

      CGSolver mcg;
      mcg.SetRelTol(1e-12);
      mcg.SetMaxIter(2000);
      mcg.SetPrintLevel(0);
      mcg.SetOperator(*M);
      mcg.Mult(B, X);

      m->RecoverFEMSolution(X, rb, rx);
      rx.Add(rbc_a_val, x);

      ConstantCoefficient one(1.0);
      BilinearForm m_rbc(&fespace);
      if (h1)
      {
         m_rbc.AddBoundaryIntegrator(new MassIntegrator, rbc_bdr);
      }
      else
      {
         m_rbc.AddBdrFaceIntegrator(new BoundaryMassIntegrator(one), rbc_bdr);
      }
      m_rbc.Assemble();

      LinearForm m2x(&fespace);
      m_rbc.Mult(rx, m2x);

      double rbc_int = copysign(sqrt(m2x(rx) * 0.5), rbc_b_val);
      double rbc_err = fabs(rbc_int - rbc_b_val);

      bool hom_rbc = (rbc_b_val == 0.0);
      rbc_err /=  hom_rbc ? 1.0 : fabs(rbc_b_val);
      mfem::out << "rbc " << rbc_int << ", "
                << (hom_rbc ? "absolute" : "relative")
                << " error " << rbc_err << endl;
   }

   // 16. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      string h1_str = h1 ? "H1" : "DG";
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x
               << "window_title '" << h1_str << " Solution'"
               << " keys 'mmc'" << flush;

      socketstream n_sol_sock(vishost, visport);
      n_sol_sock.precision(8);
      n_sol_sock << "solution\n" << *mesh << nx
                 << "window_title 'Neumann'" << flush;

      socketstream n0_sol_sock(vishost, visport);
      n0_sol_sock.precision(8);
      n0_sol_sock << "solution\n" << *mesh << n0x
                  << "window_title 'Homogeneous Neumann'" << flush;

      socketstream r_sol_sock(vishost, visport);
      r_sol_sock.precision(8);
      r_sol_sock << "solution\n" << *mesh << rx
                 << "window_title 'Robin'" << flush;
   }

   // 18. Free the used memory.
   delete fec;

   return 0;
}

void quad_trans(double u, double v, double &x, double &y, bool log = false)
{
   double a = a_; // Radius of disc

   double d = 4.0 * a * (M_SQRT2 - 2.0 * a) * (1.0 - 2.0 * v);

   double v0 = (1.0 + M_SQRT2) * (M_SQRT2 * a - 2.0 * v) *
               ((4.0 - 3 * M_SQRT2) * a +
                (8.0 * (M_SQRT2 - 1.0) * a - 2.0) * v) / d;

   double r = 2.0 * ((M_SQRT2 - 1.0) * a * a * (1.0 - 4.0 *v) +
                     2.0 * (1.0 + M_SQRT2 *
                            (1.0 + 2.0 * (2.0 * a - M_SQRT2 - 1.0) * a)) * v * v
                    ) / d;

   double t = asin(v / r) * u / v;
   if (log)
   {
      mfem::out << "u, v, r, v0, t "
                << u << " " << v << " " << r << " " << v0 << " " << t
                << endl;
   }
   x = r * sin(t);
   y = r * cos(t) - v0;
}

void trans(const Vector &u, Vector &x)
{
   double tol = 1e-4;

   if (u[1] > 0.5 - tol || u[1] < -0.5 + tol)
   {
      x = u;
      return;
   }
   if (u[0] > 1.0 - tol || u[0] < -1.0 + tol || fabs(u[0]) < tol)
   {
      x = u;
      return;
   }

   if (u[0] > 0.0)
   {
      if (u[1] > fabs(u[0] - 0.5))
      {
         quad_trans(u[0] - 0.5, u[1], x[0], x[1]);
         x[0] += 0.5;
         return;
      }
      if (u[1] < -fabs(u[0] - 0.5))
      {
         quad_trans(u[0] - 0.5, -u[1], x[0], x[1]);
         x[0] += 0.5;
         x[1] *= -1.0;
         return;
      }
      if (u[0] - 0.5 > fabs(u[1]))
      {
         quad_trans(u[1], u[0] - 0.5, x[1], x[0]);
         x[0] += 0.5;
         return;
      }
      if (u[0] - 0.5 < -fabs(u[1]))
      {
         quad_trans(u[1], 0.5 - u[0], x[1], x[0]);
         x[0] *= -1.0;
         x[0] += 0.5;
         return;
      }
   }
   else
   {
      if (u[1] > fabs(u[0] + 0.5))
      {
         quad_trans(u[0] + 0.5, u[1], x[0], x[1]);
         x[0] -= 0.5;
         return;
      }
      if (u[1] < -fabs(u[0] + 0.5))
      {
         quad_trans(u[0] + 0.5, -u[1], x[0], x[1]);
         x[0] -= 0.5;
         x[1] *= -1.0;
         return;
      }
      if (u[0] + 0.5 > fabs(u[1]))
      {
         quad_trans(u[1], u[0] + 0.5, x[1], x[0]);
         x[0] -= 0.5;
         return;
      }
      if (u[0] + 0.5 < -fabs(u[1]))
      {
         quad_trans(u[1], -0.5 - u[0], x[1], x[0]);
         x[0] *= -1.0;
         x[0] -= 0.5;
         return;
      }
   }
   x = u;
}

Mesh * GenerateSerialMesh(int ref)
{
   Mesh * mesh = new Mesh(2, 29, 16, 24, 2);

   int vi[4];

   for (int i=0; i<2; i++)
   {
      int o = 13 * i;
      vi[0] = o + 0; vi[1] = o + 3; vi[2] = o + 4; vi[3] = o + 1;
      mesh->AddQuad(vi);

      vi[0] = o + 1; vi[1] = o + 4; vi[2] = o + 5; vi[3] = o + 2;
      mesh->AddQuad(vi);

      vi[0] = o + 5; vi[1] = o + 8; vi[2] = o + 9; vi[3] = o + 2;
      mesh->AddQuad(vi);

      vi[0] = o + 8; vi[1] = o + 12; vi[2] = o + 15; vi[3] = o + 9;
      mesh->AddQuad(vi);

      vi[0] = o + 11; vi[1] = o + 14; vi[2] = o + 15; vi[3] = o + 12;
      mesh->AddQuad(vi);

      vi[0] = o + 10; vi[1] = o + 13; vi[2] = o + 14; vi[3] = o + 11;
      mesh->AddQuad(vi);

      vi[0] = o + 6; vi[1] = o + 13; vi[2] = o + 10; vi[3] = o + 7;
      mesh->AddQuad(vi);

      vi[0] = o + 0; vi[1] = o + 6; vi[2] = o + 7; vi[3] = o + 3;
      mesh->AddQuad(vi);
   }

   vi[0] =  0; vi[1] =  6; mesh->AddBdrSegment(vi, 1);
   vi[0] =  6; vi[1] = 13; mesh->AddBdrSegment(vi, 1);
   vi[0] = 13; vi[1] = 19; mesh->AddBdrSegment(vi, 1);
   vi[0] = 19; vi[1] = 26; mesh->AddBdrSegment(vi, 1);

   vi[0] = 28; vi[1] = 22; mesh->AddBdrSegment(vi, 2);
   vi[0] = 22; vi[1] = 15; mesh->AddBdrSegment(vi, 2);
   vi[0] = 15; vi[1] =  9; mesh->AddBdrSegment(vi, 2);
   vi[0] =  9; vi[1] =  2; mesh->AddBdrSegment(vi, 2);

   for (int i=0; i<2; i++)
   {
      int o = 13 * i;
      vi[0] = o +  3; vi[1] = o +  7; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  7; vi[1] = o + 10; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 10; vi[1] = o + 11; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 11; vi[1] = o + 12; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o + 12; vi[1] = o +  8; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  8; vi[1] = o +  5; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  5; vi[1] = o +  4; mesh->AddBdrSegment(vi, 3 + i);
      vi[0] = o +  4; vi[1] = o +  3; mesh->AddBdrSegment(vi, 3 + i);
   }

   double d[2];
   double a = a_ / M_SQRT2;

   d[0] = -1.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] = -1.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -1.0; d[1] =  0.5; mesh->AddVertex(d);

   d[0] = -0.5 - a; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -0.5 - a; d[1] =    a; mesh->AddVertex(d);

   d[0] = -0.5; d[1] = -0.5; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =    a; mesh->AddVertex(d);
   d[0] = -0.5; d[1] =  0.5; mesh->AddVertex(d);

   d[0] = -0.5 + a; d[1] =   -a; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] = -0.5 + a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  0.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  0.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.0; d[1] =  0.5; mesh->AddVertex(d);

   d[0] =  0.5 - a; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.5 - a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  0.5; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =    a; mesh->AddVertex(d);
   d[0] =  0.5; d[1] =  0.5; mesh->AddVertex(d);

   d[0] =  0.5 + a; d[1] =   -a; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  0.5 + a; d[1] =    a; mesh->AddVertex(d);

   d[0] =  1.0; d[1] = -0.5; mesh->AddVertex(d);
   d[0] =  1.0; d[1] =  0.0; mesh->AddVertex(d);
   d[0] =  1.0; d[1] =  0.5; mesh->AddVertex(d);

   mesh->FinalizeTopology();

   mesh->SetCurvature(1, true);

   // Stitch the ends of the stack together
   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size() - 3; i++)
      {
         v2v[i] = i;
      }
      // identify vertices on the narrow ends of the rectangle
      v2v[v2v.Size() - 3] = 0;
      v2v[v2v.Size() - 2] = 1;
      v2v[v2v.Size() - 1] = 2;

      // renumber elements
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         Element *el = mesh->GetElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      // renumber boundary elements
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         Element *el = mesh->GetBdrElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      mesh->RemoveUnusedVertices();
      mesh->RemoveInternalBoundaries();
   }
   mesh->SetCurvature(3, true);

   for (int l = 0; l < ref; l++)
   {
      mesh->UniformRefinement();
   }

   mesh->Transform(trans);

   return mesh;
}
