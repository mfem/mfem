//                                Solution of distributed control problem
//
// Compile with: make distributed_control
//
// Sample runs:
//    distributed_control -m ../data/inline-quad.mesh
//
// Description:  This examples solves the following PDE-constrained
//               optimization problem:
//
//         min J(f) = 1/2 \| u - w \|_{L^2} + \alpha/2 \| f \|_{L^2} 
//
//         subject to   - \Delta u = f    in \Omega
//                               u = 0    on \partial\Omega
//         and            a <= f(x) <= b
//
//                where w = / 1   if | x | <= 1
//                          \ 0   otherwise
//
//

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

class ReducedSystemOperator;

/** The Lagrangian for this problem is
 *    
 *    L(u,f,p) = 1/2 (u - w,u-w) + \alpha/2 (f,f)
 *             - (\nabla u, \nabla p) + (f,p)
 * 
 *      u, p \in H^1_0(\Omega)
 *      f \in L^2(\Omega)
 * 
 *  Note that
 * 
 *    \partial_p L = 0        (1)
 *  
 *  delivers the state equation
 *    
 *    (\nabla u, \nabla v) = (f,v)  for all v in H^1_0(\Omega)
 * 
 *  and
 *  
 *    \partial_u L = 0        (2)
 * 
 *  delivers the adjoint equation
 * 
 *    (\nabla p, \nabla v) = (u-w,v)  for all v in H^1_0(\Omega)
 *    
 *  and at the solutions u and p(u) of (1) and (2), respectively,
 * 
 *  D_f J = D_f L = \partial_u L \partial_f u + \partial_p L \partial_f p
 *                + \partial_f L
 *                = \partial_f L
 *                = (\alpha f + p, \cdot)
 * 
 * We update the control f_k with projected gradient descent via
 * 
 *  f_{k+1} = P ( f_k - \gamma R_{L^2}^{-1} D_f J )
 * 
 * where P is the projection operator enforcing a <= u(x) <= b, \gamma is
 * a specified step length and R_{L^2} is the L2-Riesz operator. In other
 * words, we have that
 * 
 * f_{k+1} = max { a, min { b, f_k - \gamma (\alpha f_k + p) } }
 * 
 */

double unit_ball(const Vector & x)
{
   double x1 = x(0);
   double x2 = x(1);
   double x3 = 0.0;
   if (x.Size() == 3)
   {
      x3 = x(2);
   }
   double r = sqrt(x1*x1 + x2*x2 + x3*x3);
   if (r <= 0.5)
   {
      return -1.0;
   }
   else
   {
      return 0.0;
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   bool visualization = true;
   double alpha = 1e-4;
   double gamma = 1e1;
   int max_it = 1e4;
   double tol = 1e-5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Define the target function w.
   FunctionCoefficient w_coeff(unit_ball);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim);
   L2_FECollection control_fec(order, dim);
   FiniteElementSpace state_fes(&mesh, &state_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   cout << "Number of state unknowns: " << state_size << endl;
   cout << "Number of control unknowns: " << control_size << endl;

   // 6. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   state_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the bilinear form a(.,.) for the state and adjoint equation.
   BilinearForm a(&state_fes);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   OperatorPtr A;
   Vector B, C, X;

   // 8. Set the initial guess for f and the boundary conditions for u and p.
   GridFunction u(&state_fes);
   GridFunction p(&state_fes);
   GridFunction f(&control_fes);
   u = 0.0;
   p = 0.0;
   f = 0.0;

   // 9. Define the gradient function
   GridFunction grad(&control_fes);
   grad = 1e3;

   // 10. Perform projected gradient descent
   for (int k = 1; k < max_it; k++)
   {
      // A. Form state equation
      LinearForm b(&state_fes);
      GridFunctionCoefficient f_coeff(&f);
      b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
      b.Assemble();
      a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

      // B. Solve state equation
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 0, 200, 1e-12, 0.0);

      // C. Recover state variable
      a.RecoverFEMSolution(X, b, u);

      // D. Send the solution by socket to a GLVis server.
      // if (visualization)
      // {
      //    char vishost[] = "localhost";
      //    int  visport   = 19916;
      //    socketstream sol_sock(vishost, visport);
      //    sol_sock.precision(8);
      //    sol_sock << "solution\n" << mesh << u << flush;
      // }

      // E. Form adjoint equation
      LinearForm c(&state_fes);
      GridFunctionCoefficient u_coeff(&u);
      c.AddDomainIntegrator(new DomainLFIntegrator(u_coeff));
      c.AddDomainIntegrator(new DomainLFIntegrator(w_coeff));
      c.Assemble();
      a.FormLinearSystem(ess_tdof_list, p, c, A, X, C);

      // F. Solve adjoint equation
      PCG(*A, M, C, X, 0, 200, 1e-12, 0.0);

      // G. Recover adjoint variable
      a.RecoverFEMSolution(X, c, p);

      // H. Constuct gradient function (i.e., \alpha f + p)
      grad.ProjectGridFunction(p);
      grad /= alpha;
      grad += f;
      grad *= alpha;

      // I. Compute norm of gradient.
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }
      double norm = grad.ComputeL2Error(zero, irs);

      // J. Update control.
      grad *= gamma;
      f -= grad;

      // K. Exit if norm of grad is small enough.
      mfem::out << "norm of gradient = " << norm << endl;
      if (norm < tol)
      {
         break;
      }

   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << u << flush;
   }

   return 0;
}