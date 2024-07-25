//       Solving the vectorized version of the Laplace  problem -Delta u = 1 with 
//       Dirichlet boundary conditions


#include "mfem.hpp"
#include <fstream>
#include <iostream>



using namespace std;
using namespace mfem;


void f_rhs(const Vector &x,Vector &y);
void u_ex(const Vector &x,Vector &y);



int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../../data/star.mesh";
   int order = 2;
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
   FiniteElementSpace fespace(&mesh, &fec,mesh.Dimension(),Ordering::byVDIM);
   cout << "Dimension: " << mesh.Dimension() << endl;

 
   int dim = mesh.Dimension();

   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   VectorFunctionCoefficient bc(dim, u_ex);
   x.ProjectCoefficient(bc);

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   VectorFunctionCoefficient f (dim,f_rhs);
   LinearForm b(&fespace);

   ConstantCoefficient one(1.0);

   b.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);

   ConstantCoefficient lambda(0.0);
   ConstantCoefficient mu(1.0);

   a.AddDomainIntegrator(new ElasticityIntegrator(lambda,mu));
   // a.AddDomainIntegrator(new VectorDiffusionIntegrator);

   a.Assemble();

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a.RecoverFEMSolution(X, b, x);
   // x.Save("sol.gf");
   // mesh.Save("mesh.mesh");


   cout << "\n|| u_h - u ||_{L^2} = " << x.ComputeL2Error(bc) << '\n' << endl;
   // cout << "\n|| grad u_h - grad u ||_{L^2} = " << x.ComputeH1Error(&bc,&u_grad) << '\n' << endl;
   // todo: find out how to compute H1 for vector case

   ParaViewDataCollection paraview_dc("vecdiffusion", &mesh);
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
// for gradient case
// void f_rhs(const Vector &x,Vector &y)
// {
//    y(0) = sin(x(1)), y(1) = sin(x(0));
// }

// void u_ex(const Vector &x,Vector &y)
// {
//    y(0)= sin(x(1)),y(1)=sin(x(0));
// }


// for symmetric gradient
void f_rhs(const Vector &x,Vector &y)
{
   y(0) = 3*cos(x(0))*sin(x(1)) - cos(x(0))*cos(x(1)), y(1)=3*sin(x(0))*sin(x(1))+sin(x(0))*cos(x(1));
}

void u_ex(const Vector &x,Vector &y)
{
   y(0)= cos(x(0))*sin(x(1)),y(1)=sin(x(0))*sin(x(1));
}



