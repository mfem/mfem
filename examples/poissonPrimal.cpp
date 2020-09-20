/*-------------------- Primal pde ----------------------------------*/
/* This code solves the Primal pde for Adjoint Assignment 1 */
/*-------------------------------------------------------------------*/
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;
double u_exact(const Vector &);
double f_exact(const Vector &);
double gamma_x(const Vector &);
int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 1;
   double sigma = -1.0;
   double kappa = 100.0;
   // default mesh size (N x N)
   int N = 24;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&N, "-n", "--#elements",
                  "number of mesh elements.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. generate the mesh
   Mesh *mesh = new Mesh(N, N, Element::QUADRILATERAL, true,
                         1, 1, true);
   int dim = mesh->Dimension();
   cout << "boundary attributes " << mesh->bdr_attributes.Size() << endl;
   cout << "#boundary elements " << mesh->GetNBE() << endl;
   cout << "bdr attr " << mesh->GetBdrAttribute(3) << endl;
   // 3. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   // 4. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   FunctionCoefficient f(f_exact);
   FunctionCoefficient u(u_exact);
   FunctionCoefficient diff(gamma_x);
   b->AddDomainIntegrator(new DomainLFIntegrator(f));
   b->AddBdrFaceIntegrator(
       new DGDirichletLFIntegrator(u, diff, sigma, kappa));
   b->Assemble();

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   GridFunction x(fespace);
   x.ProjectCoefficient(u);

   // 6. Set up the bilinear form a(.,.) on the finite element space.
   //Boundary conditions are imposed weakly.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(diff));
   a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff, sigma, kappa));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff, sigma, kappa));
   a->Assemble();
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

#ifdef MFEM_USE_SUITESPARSE
   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   GSSmoother M(A);
   if (sigma == -1.0)
   {
      PCG(A, M, *b, x, 1, 2000, 1e-12, 1e-24);
   }
   else
   {
      GMRES(A, M, *b, x, 1, 500, 10, 1e-12, 0.0);
   }
#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

   // 9. Save the mesh and the solution.
   ofstream adj_ofs("primal.vtk");
   adj_ofs.precision(14);
   mesh->PrintVTK(adj_ofs, 1);
   x.SaveVTK(adj_ofs, "PrimalSolution", 1);
   adj_ofs.close();

   // 10. Calculate the solution norm
   double norm = x.ComputeL2Error(u);
   cout << "----------------------------- " << endl;
   cout << "mesh size, h = " << 1.0 / N << endl;
   cout << "solution norm: " << norm << endl;
   // 11. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

/*---------- exact solution function ---------------*/
// Input:
//        Vector x
// Output:
//        returns exact solution
/*--------------------------------------------------*/
double u_exact(const Vector &x)
{
   double theta = M_PI * (exp(x(0)) - 1) / (exp(1) - 1);
   return exp(x(1)) * sin(theta);
}
/*---------- source term function ---------------*/
// Input:
//        Vector x
// Output:
//        returns source term, f
/*--------------------------------------------------*/
double f_exact(const Vector &x)
{
   double gamma = (M_PI * exp(x(0))) / (exp(1) - 1);
   double theta = M_PI * (exp(x(0)) - 1) / (exp(1) - 1);
   return gamma * ((-2.0 * gamma * exp(x(1)) * cos(theta)) - (exp(x(1)) * sin(theta))
               + (gamma * gamma * exp(x(1)) * sin(theta)));
 }
/*---------- diffusion coefficient function ---------------*/
// Input:
//        Vector x
// Output:
//        returns diffusion coefficient, gamma(x)
/*--------------------------------------------------*/
double gamma_x(const Vector &x)
{
   return (M_PI * exp(x(0))) / (exp(1.0) - 1.0);
}
