//                                MFEM Example 0 with some small changes 
//
// 
//// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with Dirichlet boundary conditions.


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "my_integrators.hpp"


using namespace std;
using namespace mfem;


real_t f_rhs(const Vector &x);
real_t u_ex(const Vector &x);
void u_grad_exact(const Vector &x, Vector &u);
real_t koeff(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../../data/star.mesh";
   int order = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();
	mesh.UniformRefinement();


   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to the function defining boundary conditions.
   GridFunction x(&fespace);
   FunctionCoefficient bc (u_ex);
   x.ProjectCoefficient(bc);

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   FunctionCoefficient f (f_rhs);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   FunctionCoefficient coeff(koeff);

   a.AddDomainIntegrator(new MyDiffusionIntegrator(coeff));
   a.Assemble();

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // 10. Recover the solution x as a grid function and save to file
   a.RecoverFEMSolution(X, b, x);


   // compute errors:     
   VectorFunctionCoefficient u_grad(mesh.Dimension(), u_grad_exact);

   cout << "\n|| u_h - u ||_{L^2} = " << x.ComputeL2Error(bc) << '\n' << endl;
   cout << "\n|| grad u_h - grad u ||_{L^2} = " << x.ComputeH1Error(&bc,&u_grad) << '\n' << endl;


   // save solution with paraview 
   ParaViewDataCollection paraview_dc("diffusion", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&x);
   paraview_dc.Save();

   return 0;
}

real_t f_rhs(const Vector &x)
{
   return sin(x(0)) * sin( x(1)) + (2*x(0)+4)*cos(x(0))*sin(x(1));
}

real_t u_ex(const Vector &x)
{
    return cos(x(0))*sin(x(1));
}


void u_grad_exact(const Vector &x, Vector &u)
{
    u(0) =  - sin(x(0)) * sin( x(1));
    u(1) =  cos(x(0)) * cos( x(1));
}

real_t koeff(const Vector &x)
{
   return x(0) + 2; 
}

// Q = 1
// real_t f_rhs(const Vector &x)
// {
//    return 2*cos(x(0))*sin(x(1));
// }

// real_t u_ex(const Vector &x)
// {
//     return cos(x(0))*sin(x(1));
// }
