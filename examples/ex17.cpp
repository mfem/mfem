//                                MFEM Example 17
//
// Compile with: make ex17
//
// Sample runs:
//
//       ex17 -m ../data/beam-tri.mesh
//       ex17 -m ../data/beam-quad.mesh
//       ex17 -m ../data/beam-tet.mesh
//       ex17 -m ../data/beam-hex.mesh
//       ex17 -m ../data/beam-wedge.mesh
//       ex17 -m ../data/beam-quad.mesh -r 2 -o 3
//       ex17 -m ../data/beam-quad.mesh -r 2 -o 2 -a 1 -k 1
//       ex17 -m ../data/beam-hex.mesh -r 2 -o 2
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam using symmetric or
//               non-symmetric discontinuous Galerkin (DG) formulation.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               Dirichlet, u=u_D on the fixed part of the boundary, namely
//               boundary attributes 1 and 2; on the rest of the boundary we use
//               sigma(u).n=0 b.c. The geometry of the domain is assumed to be
//               as follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (fixed, nonzero)
//
//               The example demonstrates the use of high-order DG vector finite
//               element spaces with the linear DG elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and function vector-coefficient objects. The use of
//               non-homogeneous Dirichlet b.c. imposed weakly, is also
//               illustrated.
//
//               We recommend viewing examples 2 and 14 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Initial displacement, used for Dirichlet boundary conditions on boundary
// attributes 1 and 2.
void InitDisplacement(const Vector &x, Vector &u);

// A Coefficient for computing the components of the stress.
class StressCoefficient : public Coefficient
{
protected:
   Coefficient &lambda, &mu;
   GridFunction *u; // displacement
   int si, sj; // component of the stress to evaluate, 0 <= si,sj < dim

   DenseMatrix grad; // auxiliary matrix, used in Eval

public:
   StressCoefficient(Coefficient &lambda_, Coefficient &mu_)
      : lambda(lambda_), mu(mu_), u(NULL), si(0), sj(0) { }

   void SetDisplacement(GridFunction &u_) { u = &u_; }
   void SetComponent(int i, int j) { si = i; sj = j; }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

// Simple GLVis visualization manager.
class VisMan : public iostream
{
protected:
   const char *host;
   int port;
   Array<socketstream *> sock;
   int sid; // active socket, index inside 'sock'.

   int win_x, win_y, win_w, win_h;
   int win_stride_x, win_stride_y, win_nx;

public:
   VisMan(const char *vishost, const int visport);
   void NewWindow();
   void CloseConnection();
   void PositionWindow();
   virtual ~VisMan();
};

// Manipulators for the GLVis visualization manager.
void new_window      (VisMan &v) { v.NewWindow(); }
void position_window (VisMan &v) { v.PositionWindow(); }
void close_connection(VisMan &v) { v.CloseConnection(); }
ostream &operator<<(ostream &v, void (*f)(VisMan&));

int main(int argc, char *argv[])
{
   // 1. Define and parse command-line options.
   const char *mesh_file = "../data/beam-tri.mesh";
   int ref_levels = -1;
   int order = 1;
   double alpha = -1.0;
   double kappa = -1.0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&alpha, "-a", "--alpha",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGElasticityIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex17.cpp)\n"
           << endl;
      return 3;
   }

   // 3. Refine the mesh to increase the resolution.
   if (ref_levels < 0)
   {
      ref_levels = (int)floor(log(5000./mesh.GetNE())/log(2.)/dim);
   }
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }
   // Since NURBS meshes do not support DG integrators, we convert them to
   // regular polynomial mesh of the specified (solution) order.
   if (mesh.NURBSext) { mesh.SetCurvature(order); }

   // 4. Define a DG vector finite element space on the mesh. Here, we use
   //    Gauss-Lobatto nodal basis because it gives rise to a sparser matrix
   //    compared to the default Gauss-Legendre nodal basis.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec, dim);

   cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
        << "\nAssembling: " << flush;

   // 5. In this example, the Dirichlet boundary conditions are defined by
   //    marking boundary attributes 1 and 2 in the marker Array 'dir_bdr'.
   //    These b.c. are imposed weakly, by adding the appropriate boundary
   //    integrators over the marked 'dir_bdr' to the bilinear and linear forms.
   //    With this DG formulation, there are no essential boundary conditions.
   Array<int> ess_tdof_list; // no essential b.c. (empty list)
   Array<int> dir_bdr(mesh.bdr_attributes.Max());
   dir_bdr = 0;
   dir_bdr[0] = 1; // boundary attribute 1 is Dirichlet
   dir_bdr[1] = 1; // boundary attribute 2 is Dirichlet

   // 6. Define the DG solution vector 'x' as a finite element grid function
   //    corresponding to fespace. Initialize 'x' using the 'InitDisplacement'
   //    function.
   GridFunction x(&fespace);
   VectorFunctionCoefficient init_x(dim, InitDisplacement);
   x.ProjectCoefficient(init_x);

   // 7. Set up the Lame constants for the two materials. They are defined as
   //    piece-wise (with respect to the element attributes) constant
   //    coefficients, i.e. type PWConstCoefficient.
   Vector lambda(mesh.attributes.Max());
   lambda = 1.0;      // Set lambda = 1 for all element attributes.
   lambda(0) = 50.0;  // Set lambda = 50 for element attribute 1.
   PWConstCoefficient lambda_c(lambda);
   Vector mu(mesh.attributes.Max());
   mu = 1.0;      // Set mu = 1 for all element attributes.
   mu(0) = 50.0;  // Set mu = 50 for element attribute 1.
   PWConstCoefficient mu_c(mu);

   // 8. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this example, the linear form b(.) consists
   //    only of the terms responsible for imposing weakly the Dirichlet
   //    boundary conditions, over the attributes marked in 'dir_bdr'. The
   //    values for the Dirichlet boundary condition are taken from the
   //    VectorFunctionCoefficient 'x_init' which in turn is based on the
   //    function 'InitDisplacement'.
   LinearForm b(&fespace);
   cout << "r.h.s. ... " << flush;
   b.AddBdrFaceIntegrator(
      new DGElasticityDirichletLFIntegrator(
         init_x, lambda_c, mu_c, alpha, kappa), dir_bdr);
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) on the DG finite element space
   //    corresponding to the linear elasticity integrator with coefficients
   //    lambda and mu as defined above. The additional interior face integrator
   //    ensures the weak continuity of the displacement field. The additional
   //    boundary face integrator works together with the boundary integrator
   //    added to the linear form b(.) to impose weakly the Dirichlet boundary
   //    conditions.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new ElasticityIntegrator(lambda_c, mu_c));
   a.AddInteriorFaceIntegrator(
      new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa));
   a.AddBdrFaceIntegrator(
      new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa), dir_bdr);

   // 10. Assemble the bilinear form and the corresponding linear system.
   cout << "matrix ... " << flush;
   a.Assemble();

   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "done." << endl;

   // Print some information about the matrix of the linear system.
   A.PrintInfo(cout);

#ifndef MFEM_USE_SUITESPARSE
   // 11. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //     solve the system Ax=b with PCG for the symmetric formulation, or GMRES
   //     for the non-symmetric.
   GSSmoother M(A);
   const double rtol = 1e-6;
   if (alpha == -1.0)
   {
      PCG(A, M, B, X, 3, 5000, rtol*rtol, 0.0);
   }
   else
   {
      GMRES(A, M, B, X, 3, 5000, 50, rtol*rtol, 0.0);
   }
#else
   // 11. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 12. Recover the solution as a finite element grid function 'x'.
   a.RecoverFEMSolution(X, b, x);

   // 13. Use the DG solution space as the mesh nodal space. This allows us to
   //     save the displaced mesh as a curved DG mesh.
   mesh.SetNodalFESpace(&fespace);

   Vector reference_nodes;
   if (visualization) { reference_nodes = *mesh.GetNodes(); }

   // 14. Save the displaced mesh and minus the solution (which gives the
   //     backward displacements to the reference mesh). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      *mesh.GetNodes() += x;
      x.Neg(); // x = -x
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Visualization: send data by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      VisMan vis(vishost, visport);
      const char *glvis_keys = (dim < 3) ? "Rjlc" : "c";

      // Visualize the deformed configuration.
      vis << new_window << setprecision(8)
          << "solution\n" << mesh << x << flush
          << "keys " << glvis_keys << endl
          << "window_title 'Deformed configuration'" << endl
          << "plot_caption 'Backward displacement'" << endl
          << position_window << close_connection;

      // Visualize the stress components.
      const char *c = "xyz";
      FiniteElementSpace scalar_dg_space(&mesh, &fec);
      GridFunction stress(&scalar_dg_space);
      StressCoefficient stress_c(lambda_c, mu_c);
      *mesh.GetNodes() = reference_nodes;
      x.Neg(); // x = -x
      stress_c.SetDisplacement(x);
      for (int si = 0; si < dim; si++)
      {
         for (int sj = si; sj < dim; sj++)
         {
            stress_c.SetComponent(si, sj);
            stress.ProjectCoefficient(stress_c);

            vis << new_window << setprecision(8)
                << "solution\n" << mesh << stress << flush
                << "keys " << glvis_keys << endl
                << "window_title |Stress " << c[si] << c[sj] << '|' << endl
                << position_window << close_connection;
         }
      }
   }

   return 0;
}


void InitDisplacement(const Vector &x, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*x(0);
}


double StressCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "displacement field is not set");

   double L = lambda.Eval(T, ip);
   double M = mu.Eval(T, ip);
   u->GetVectorGradient(T, grad);
   if (si == sj)
   {
      double div_u = grad.Trace();
      return L*div_u + 2*M*grad(si,si);
   }
   else
   {
      return M*(grad(si,sj) + grad(sj,si));
   }
}


VisMan::VisMan(const char *vishost, const int visport)
   : iostream(0),
     host(vishost), port(visport), sid(0)
{
   win_x = 0;
   win_y = 0;
   win_w = 400; // window width
   win_h = 350; // window height
   win_stride_x = win_w;
   win_stride_y = win_h + 20;
   win_nx = 4; // number of windows in a row
}

void VisMan::NewWindow()
{
   sock.Append(new socketstream(host, port));
   sid = sock.Size()-1;
   iostream::rdbuf(sock[sid]->rdbuf());
}

void VisMan::CloseConnection()
{
   if (sid < sock.Size())
   {
      delete sock[sid];
      sock[sid] = NULL;
      iostream::rdbuf(0);
   }
}

void VisMan::PositionWindow()
{
   *this << "window_geometry "
         << win_x + win_stride_x*(sid%win_nx) << ' '
         << win_y + win_stride_y*(sid/win_nx) << ' '
         << win_w << ' ' << win_h << endl;
}

VisMan::~VisMan()
{
   for (int i = sock.Size()-1; i >= 0; i--)
   {
      delete sock[i];
   }
}


ostream &operator<<(ostream &v, void (*f)(VisMan&))
{
   VisMan *vp = dynamic_cast<VisMan*>(&v);
   if (vp) { (*f)(*vp); }
   return v;
}
