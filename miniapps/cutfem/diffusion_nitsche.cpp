//       Solving the  Laplace   problem -Delta u = 1 with 
//       Dirichlet boundary conditions weakly inforced. 
//


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "my_integrators.hpp"


using namespace std;
using namespace mfem;


real_t f_rhs(const Vector &x);
real_t u_ex(const Vector &x);
void u_grad_exact(const Vector &x, Vector &u);

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

   // 3. Define a finite element space on the mesh. 
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Array of boundary dofs, empty but needed in FormLinearSystem
   Array<int> boundary_dofs;

   // 5. Define the solution x as a finite element grid function in fespace. 
   GridFunction x(&fespace);

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   LinearForm b(&fespace);

   FunctionCoefficient f (f_rhs);
   ConstantCoefficient one(1.0);
   FunctionCoefficient bc (u_ex);
   real_t sigma = -1.0; // IP 
   real_t lambda = 10.0; // Nitsche penalty parameter

   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   // rhs Nitsche terms 
   b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(bc, one, sigma, lambda));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);

   a.AddDomainIntegrator(new MyDiffusionIntegrator);
   // Nitsche terms 
    a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, lambda)); 
    // a.AddBdrFaceIntegrator(new MyNitscheBilinIntegrator(one, lambda)); //not working
   a.Assemble();

   // 8. Form the linear system A X = B. 
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // 10. Recover the solution x as a grid function 
   a.RecoverFEMSolution(X, b, x);


   VectorFunctionCoefficient u_grad(mesh.Dimension(), u_grad_exact);

   cout << "\n|| u_h - u ||_{L^2} = " << x.ComputeL2Error(bc) << '\n' << endl;
   cout << "\n|| grad u_h - grad u ||_{L^2} = " << x.ComputeH1Error(&bc,&u_grad) << '\n' << endl;


   ParaViewDataCollection paraview_dc("diffusionnit", &mesh);
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
   return 2*cos(x(0))*sin(x(1));
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


